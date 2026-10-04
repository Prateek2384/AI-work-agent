"""Thin provider-agnostic LLM adapter (OpenAI-compatible; configured for GroqCloud)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    """The LLM could not be reached or refused service."""


class LLMQuotaError(LLMError):
    """Daily quota exhausted: retrying now is pointless."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                for c in self.tool_calls
            ]
        return msg


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, max_retries: int = 6):
        from dotenv import find_dotenv, load_dotenv
        from openai import OpenAI  # lazy: tests use fakes and don't need the SDK

        load_dotenv(find_dotenv(usecwd=True))
        self.model = model or os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
        self.max_retries = max_retries
        # Reasoning effort per call type. Agent steps use the provider default (qwen3 acts reliably only with
        # thinking on). Planner/auditor calls pass fast=True: short JSON answers, where "none" (qwen3's
        # hidden-thinking off) is 10-50x faster. Unset = provider default for that type.
        self.reasoning_effort = os.environ.get("LLM_REASONING_EFFORT") or None
        self.fast_effort = os.environ.get("LLM_FAST_EFFORT") or None
        # Output caps stop a runaway generation (a small model can loop) from stalling a whole run:
        # short for planner/auditor JSON, generous for agent steps (which may think before acting).
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "4000"))
        self.fast_max_tokens = int(os.environ.get("LLM_FAST_MAX_TOKENS", "700"))
        self._client = OpenAI(
            api_key=api_key or os.environ["LLM_API_KEY"],
            base_url=base_url or os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout=60, max_retries=0,  # we own retry/backoff so 429 retry-after is honoured
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None, fast: bool = False) -> LLMReply:
        """One model turn (fast=True for planner/auditor JSON calls). Local models occasionally return a completely empty turn; re-sample (with a
        little more randomness) rather than letting an empty turn derail the agent."""
        reply = LLMReply(None, [])
        for attempt, temp in enumerate((0.1, 0.5, 0.8)):
            reply = self._chat_once(messages, tools, temp, self.fast_effort if fast else self.reasoning_effort,
                                    self.fast_max_tokens if fast else self.max_tokens)
            if reply.tool_calls or (reply.content or "").strip() or tools is None and reply.content is not None:
                return reply
        return reply

    def _chat_once(self, messages: list[dict], tools: list[dict] | None, temperature: float,
                   effort: str | None = None, max_tokens: int | None = None) -> LLMReply:
        import openai

        delay = 2.0
        glitch_only, timeouts = False, 0
        self.last_error = None  # surfaced in the trace when a turn comes back empty
        for attempt in range(self.max_retries + 1):
            try:
                extra = {"reasoning_effort": effort} if effort else {}
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools or None,
                    tool_choice="auto" if tools else None, temperature=temperature,
                    max_tokens=max_tokens, **extra,
                )
                return self._normalize(resp.choices[0].message)
            except openai.RateLimitError as e:
                if any(k in str(e) for k in ("per-day", "per day", "(TPD)", "daily")):
                    raise LLMQuotaError(f"daily quota exhausted for {self.model}: {str(e)[:200]}") from e
                wait, glitch_only = _retry_after(e) or delay, False
            except openai.APITimeoutError:
                timeouts += 1  # the model is stuck generating; waiting longer won't help
                if timeouts >= 2:
                    raise LLMError("the model timed out twice (it may be stuck generating)")
                wait, glitch_only = 1.0, False
            except openai.APIConnectionError:
                wait, glitch_only = delay, False
            except openai.APIStatusError as e:
                # 5xx, or Groq's 400 "tool_use_failed" when a model emits a malformed tool call
                glitch = _is_model_glitch(e)
                glitch_only = glitch
                self.last_error = str(e)[:400]
                if e.status_code >= 500 or glitch:
                    wait = 1.0 if glitch else delay
                else:
                    raise
            if attempt == self.max_retries or (glitch_only and attempt >= 2):
                if glitch_only:  # model keeps producing unparseable output: hand back an empty turn
                    return LLMReply(None, [])
                raise LLMError("LLM request failed after retries")
            time.sleep(wait)
            delay = min(delay * 2, 30)
        raise AssertionError("unreachable")

    @staticmethod
    def _normalize(m) -> LLMReply:
        calls = []
        for c in m.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_parse_error": c.function.arguments}
            calls.append(ToolCall(c.id, c.function.name, args if isinstance(args, dict) else {}))
        return LLMReply(m.content, calls)


def _retry_after(e) -> float | None:
    try:
        return float(e.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def _is_model_glitch(e) -> bool:
    """Groq 400s caused by the model emitting malformed output (retrying/resampling usually fixes it)."""
    msg = str(e)
    return e.status_code == 400 and any(k in msg for k in ("tool_use_failed", "Parsing failed", "could not be parsed"))
