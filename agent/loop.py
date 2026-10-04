"""Task-agnostic act/observe/verify loop."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Protocol

from agent.memory import Memory
from llm.client import LLMError
from agent.trace import Trace
import re

from tools.registry import IRREVERSIBLE, WRITE, Registry, ToolResult

ASK_ALLOWED = "; ask (ask_user) only when information is missing or ambiguous."
ASK_FORBIDDEN = (". You CANNOT ask the user questions: if something is unclear, use the task text, your memory and the \
results of earlier parts, pick the most reasonable interpretation, do the work, and state your assumption in the \
final answer.")

SYSTEM_PROMPT = """You are an autonomous AI worker completing a user's task using tools.
Work step by step: act, read the result, decide the next action. Do not ask the user for \
steps you can figure out yourself[[ASK]]
- Use `remember` for any fact you will need later (amounts, dates, ids).
- If an action fails, read the error, then retry or try an alternative; never repeat the same \
failing action more than twice.
- If the task depends on comparing items ("latest", "largest", "all", "total"), first list EVERY candidate from \
the source (use `read_all_pages`), `remember` their key values, compare them yourself, and only then act on the \
winner. Never take the first result. Match entity names exactly (similar names are different entities).
- Prefer the site's own search/filter to narrow results, then `read_all_pages` to see every result and \
`collect_documents(row_contains='<entity>')` to read every matching document. Use \
type_text(submit=true) for search boxes. Do not click through pages one by one.
- The `finish` summary is the FINAL ANSWER the user reads: include every requested fact (list or table for \
multiple items: ids, amounts, dates), what you did, and anything you could not do. Not just "done".
- Never state a value you did not see in an observation. Downloading a file is not reading it: read its contents (read_file / collect_documents) before answering.
- Call `finish` only when the task is done. Your claim is independently verified against the \
real system, so never claim success you have not observed."""


@dataclass
class Criterion:
    name: str
    passed: bool
    evidence: str


class Verifier(Protocol):
    def check(self, goal: str, memory: Memory, transcript: list[dict],
              criteria: list[str] | None = None, claim: str = "") -> list[Criterion]: ...


@dataclass
class RunResult:
    status: str  # done | verification_failed | budget_exhausted | aborted
    summary: str
    steps: int
    criteria: list[Criterion] = field(default_factory=list)
    trace_dir: str = ""


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NON_VALUE_ARGS = {"element_id", "url", "method", "option", "link_text", "row_contains", "path"}


def ungrounded_numbers(args: dict, msgs: list[dict]) -> list[str]:
    """Numbers (3+ digits) the agent is about to write that appear nowhere in the task, the system notes
    or any observation so far. Catches invented/misremembered amounts before they reach a system."""
    corpus = " ".join(str(m.get("content") or "") for m in msgs if m["role"] in ("system", "user", "tool"))
    corpus = re.sub(r"(?<=\d),(?=\d)", "", corpus)
    bad = []
    for key, val in args.items():
        if key in _NON_VALUE_ARGS or not isinstance(val, (str, int, float)) or isinstance(val, bool):
            continue
        for tok in _NUMBER.findall(str(val)):
            plain = tok.replace(",", "")
            core = plain[:-3] if plain.endswith(".00") else plain
            if len(core.replace(".", "")) >= 3 and core not in corpus:
                bad.append(plain)
    return bad


def _trim_old_observations(msgs: list[dict], keep_full: int = 2, stub: int = 200) -> list[dict]:
    """Keep only the latest tool outputs in full; older page dumps are cut to a stub.
    Durable facts live in memory, so this keeps prompts small without losing state."""
    tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    old = set(tool_idx[:-keep_full]) if keep_full else set(tool_idx)
    out = []
    for i, m in enumerate(msgs):
        # only single-page browser dumps are trimmed; lists (read_all_pages), documents and results stay intact
        if i in old and len(m["content"]) > stub and "--- INTERACTIVE ELEMENTS ---" in m["content"]:
            m = {**m, "content": m["content"][:stub] + " ...[older output trimmed]"}
        out.append(m)
    return out


class Agent:
    def __init__(self, llm, registry: Registry, trace: Trace | None = None,
                 verifier: Verifier | None = None, max_steps: int = 40,
                 approve: Callable[[str, dict], bool] | None = None,
                 ask_user: Callable[[str], str] | None = None, context: str = "",
                 use_planner: bool = False,
                 pending_check: Callable[[], str | None] | None = None,
                 commit_preview: Callable[[str, dict], str | None] | None = None,
                 allow_questions: bool = True):
        self.llm, self.registry, self.context = llm, registry, context
        self.allow_questions = allow_questions
        self.use_planner, self.criteria = use_planner, None
        self.pending_check, self._pending_refusals = pending_check, 0
        self.commit_preview, self._commit_refusals = commit_preview, 0
        self.commits: list[str] = []  # every data-entry submission this run performed
        self.trace = trace or Trace()
        self.verifier, self.max_steps = verifier, max_steps
        self.approve = approve or (lambda name, args: False)  # deny by default
        self.ask_user = ask_user
        self.memory = Memory()
        self.cancel = threading.Event()  # set from another thread to stop the run
        self._register_builtins()

    def _register_builtins(self):
        r = self.registry

        @r.tool("remember", "Store a fact for later steps.",
                {"properties": {"key": {"type": "string"}, "value": {"type": "string"},
                                "source": {"type": "string"}}, "required": ["key", "value"]})
        def _remember(key, value, source=""):
            self.memory.remember(key, value, source)
            return f"remembered {key}"

        if self.allow_questions:
            @r.tool("ask_user", "Ask the user a clarifying question when you cannot safely proceed.",
                    {"properties": {"question": {"type": "string"}}, "required": ["question"]})
            def _ask(question):
                if self.ask_user is None:
                    return "User is unavailable; proceed with the safest assumption or finish and report the blocker."
                return self.ask_user(question)

        @r.tool("finish", "Declare the task complete (or blocked) with a concise summary.",
                {"properties": {"summary": {"type": "string"}}, "required": ["summary"]})
        def _finish(summary):
            return summary

    def run(self, task: str) -> RunResult:
        """Plan, then run the task - or, for a request with several independent parts, each part as its own
        short task in a fresh conversation (small models lose track when one conversation does it all)."""
        plan = None
        if self.use_planner:
            from agent.planner import make_plan
            self.trace.log("phase", name="planning")
            plan = make_plan(self.llm, task, self.context)
            self.trace.log("plan", plan=plan)
        subtasks = (plan or {}).get("subtasks") or []
        if len(subtasks) > 1:
            return self._run_subtasks(subtasks)
        return self._run_task(task, plan)

    def _run_subtasks(self, subtasks: list[str]) -> RunResult:
        from agent.planner import make_plan
        parts, steps, criteria = [], 0, []
        for i, sub in enumerate(subtasks, 1):
            self.trace.log("phase", name=f"subtask {i}/{len(subtasks)}: {sub[:90]}")
            res = self._run_task(sub, make_plan(self.llm, sub, self.context, allow_split=False), prior=list(parts))
            steps += res.steps
            criteria += res.criteria
            parts.append(f"{i}. {res.summary}")
            if res.status != "done":
                skipped = f"\n\n(Stopped here; {len(subtasks) - i} later part(s) were not attempted.)" if i < len(subtasks) else ""
                return RunResult(res.status, "\n\n".join(parts) + skipped, steps, criteria, str(self.trace.dir))
        return RunResult("done", "\n\n".join(parts), steps, criteria, str(self.trace.dir))

    def _run_task(self, task: str, plan: dict | None, prior: list[str] | None = None) -> RunResult:
        from agent.planner import render_plan
        sys_prompt = SYSTEM_PROMPT.replace("[[ASK]]", ASK_ALLOWED if self.allow_questions else ASK_FORBIDDEN)
        sys_prompt += f"\n\nEnvironment notes:\n{self.context}" if self.context else ""
        if prior:  # a later part of a multi-part request needs what the earlier parts found
            sys_prompt += "\n\nResults of the earlier parts of this request (use them, do not ask for them):\n" + "\n".join(prior)
        self.criteria = plan["success_criteria"] if plan else None
        if plan:
            sys_prompt += "\n\n" + render_plan(plan)
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Task: {task}"}]
        self.trace.log("start", task=task)
        recent: list[tuple] = []
        schemas = self.registry.schemas()
        verify_failures = 0

        for step in range(1, self.max_steps + 1):
            if self.cancel.is_set():
                self.trace.log("cancelled", step=step)
                return RunResult("aborted", "Stopped by the user.", step - 1, trace_dir=str(self.trace.dir))
            msgs_for_llm = (msgs[:2] + [{"role": "system", "content": f"Memory:\n{self.memory.render()}"}]
                            + _trim_old_observations(msgs[2:]))
            try:
                reply = self.llm.chat(msgs_for_llm, schemas)
            except LLMError as e:
                self.trace.log("llm_error", step=step, error=str(e))
                return RunResult("aborted", f"Stopped: the LLM is unavailable ({e}). Progress so far is in the "
                                 f"trace; re-run later or switch LLM_MODEL.", step - 1, trace_dir=str(self.trace.dir))
            if reply.tool_calls or (reply.content or "").strip():
                msgs.append(reply.to_message())
            self.trace.log("llm", step=step, llm_error=getattr(self.llm, "last_error", None) if not reply.tool_calls else None, prompt_chars=sum(len(str(m.get("content", ""))) for m in msgs_for_llm), content=reply.content,
                           calls=[(c.name, c.args) for c in reply.tool_calls])

            if not reply.tool_calls:
                nudge = {"role": "user", "content": "Continue using tools, or call `finish`."}
                if msgs[-1] != nudge:  # never stack identical nudges
                    msgs.append(nudge)
                continue

            for call in reply.tool_calls:
                tool = self.registry.tools.get(call.name)
                if tool and tool.risk == IRREVERSIBLE and not self.approve(call.name, call.args):
                    res = ToolResult(False, error="user denied approval for this action", error_type="denied")
                elif (call.name == "finish" and self.pending_check and self._pending_refusals < 3
                      and (note := self.pending_check())):
                    self._pending_refusals += 1
                    res = ToolResult(False, error=f"refused: {note}. Submit it, confirm the result on the page, "
                                     "then finish.", error_type="unsubmitted")
                elif (tool and tool.risk == WRITE and self.commit_preview and hasattr(self.verifier, "review_commit")
                      and (preview := self.commit_preview(call.name, call.args))
                      and (refusal := self._review_before_saving(task, preview, msgs, step))):
                    res = ToolResult(False, error=refusal, error_type="precommit")
                elif tool and tool.risk == WRITE and (bad := ungrounded_numbers(call.args, msgs)):
                    res = ToolResult(False, error=f"refused: the value(s) {bad} do not appear in anything you have "
                                     "read. Do not guess or estimate. Re-read the source document and use the "
                                     "exact figures you observed.", error_type="ungrounded")
                elif call.name == "finish":
                    outcome = self._handle_finish(task, call.args.get("summary", ""), msgs, step)
                    if isinstance(outcome, RunResult):
                        return outcome
                    verify_failures += 1
                    if verify_failures >= 3:
                        why = outcome.error.replace("Verification FAILED — task is not complete. ", "")
                        return RunResult("verification_failed",
                                         f"I could not fully verify this. My answer was: {call.args.get('summary', '')}"
                                         f"\n\nUnresolved checks: {why}", step, trace_dir=str(self.trace.dir))
                    res = outcome
                else:
                    preview = self.commit_preview(call.name, call.args) if (
                        tool and tool.risk == WRITE and self.commit_preview) else None
                    res = self.registry.dispatch(call.name, call.args)
                    if preview and res.ok and not re.search(r"\(HTTP [45]\d\d\)", res.render()):
                        self.commits.append(preview)

                key = (call.name, repr(sorted(call.args.items())), res.ok)
                recent.append(key)
                hint = ""
                if not res.ok and recent[-3:].count(key) == 3:
                    hint = "\nSTUCK: this exact action has failed 3 times. Change strategy or ask the user."
                self.trace.log("tool", step=step, name=call.name, ok=res.ok,
                               error_type=res.error_type, result=res.render(1500))
                msgs.append({"role": "tool", "tool_call_id": call.id, "content": res.render() + hint})

        self.trace.log("budget_exhausted")
        return RunResult("budget_exhausted", "Step budget exhausted before the task was verified complete.",
                         self.max_steps, trace_dir=str(self.trace.dir))

    def _review_before_saving(self, task: str, preview: str, msgs: list[dict], step: int) -> str | None:
        """Pre-save audit. Returns a refusal message, or None to let the save proceed."""
        self.trace.log("phase", name="reviewing the values before saving")
        ok, reason = self.verifier.review_commit(task, self.criteria, msgs, preview)
        self.trace.log("precommit", ok=ok, preview=preview, reason=reason)
        if ok:
            return None
        self._commit_refusals += 1
        if self._commit_refusals >= 3 and self.approve("save_form", {"preview": preview, "concern": reason}):
            return None  # the user looked at the concern and chose to save anyway
        return (f"NOT saved - pre-save review found a problem: {reason} Fix the values (re-read the correct "
                "source document / re-check which item the task asks for), then submit again.")

    def _notes(self, summary: str, unverified: bool = False) -> str:
        """Honest disclosure appended to the final answer."""
        if len(self.commits) > 1:
            summary += (f"\n\nRecords saved by this run ({len(self.commits)}), please check each is intended:\n"
                        + "\n".join(f"- {c}" for c in self.commits))
        if unverified:
            summary += "\n\n⚠ Not independently verified: the auditor model gave no usable verdict."
        return summary

    def _handle_finish(self, task, summary, msgs, step):
        if self.verifier is None:
            self.trace.log("finish", summary=summary, verified=None)
            return RunResult("done", self._notes(summary), step, trace_dir=str(self.trace.dir))
        self.trace.log("phase", name="verifying the answer against the evidence")
        criteria = self.verifier.check(task, self.memory, msgs, self.criteria, summary)
        self.trace.log("verify", criteria=[c.__dict__ for c in criteria])
        if all(c.passed for c in criteria):
            unverified = any(c.name == "UNVERIFIED" for c in criteria)
            return RunResult("done", self._notes(summary, unverified), step, criteria, str(self.trace.dir))
        failed = "; ".join(f"{c.name}: {c.evidence}" for c in criteria if not c.passed)
        return ToolResult(False, error=f"Verification FAILED — task is not complete. {failed}. "
                          "Fix: gather the missing evidence from the real source (for several documents use "
                          "collect_documents(row_contains=...)), then call finish again using ONLY values you "
                          "actually observed. Never guess or estimate values.",
                          error_type="verification")
