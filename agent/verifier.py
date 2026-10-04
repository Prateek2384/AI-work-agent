"""Independent check of the agent's claim against what it actually observed.

Two layers:
1. Deterministic grounding: every invoice-style id / money amount in the final answer must appear in
   tool observations (catches invented or misread values, no LLM needed).
2. LLM judge: a separate call, given only the task, criteria, claim and trimmed evidence, decides whether
   each criterion is *demonstrated* (completeness, post-write read-back, correctness).
"""
from __future__ import annotations

import json
import re

from agent.loop import Criterion
from llm.client import LLMError
from agent.planner import extract_json

JUDGE_PROMPT = """You are a strict auditor of an AI worker's output. You see the task, success criteria, the
worker's final answer, and the evidence (tool observations). Decide for each criterion whether the EVIDENCE
demonstrates it. Rules:
- Pass only if the evidence directly shows it; the worker's own claim is not evidence.
- Completeness ("all", "latest", "every"): evidence must show the full result set was covered (e.g. all pages
  read, counts match). Picking the first or an arbitrary item does NOT satisfy "latest" - dates must be compared.
- Data entered into a system: evidence must show a read-back of the saved record AFTER saving, with the right values.
- Every fact in the final answer must be supported by the evidence.
- The final answer itself must contain the substantive details asked for (amounts, dates, statuses...). Saying documents were "retrieved" without stating their contents FAILS.
For each criterion write the EVIDENCE first (quote what the evidence shows), THEN the verdict. If your evidence
shows the claim is wrong or unproven, the verdict MUST be FAIL.
Reply with ONLY JSON: {"criteria": [{"name": "...", "evidence": "one short sentence quoting the evidence", "verdict": "PASS" or "FAIL"}]}"""

REVIEW_PROMPT = """You review a data-entry action BEFORE it is saved. You see the task, the success criteria, the
evidence the worker collected, and the exact values about to be saved. Decide whether saving is correct:
1. The right item/record was chosen for the task. If the task depends on comparing items (latest, oldest, largest,
   all...), the evidence must show the whole candidate set was compared and this is the one that qualifies.
2. Every value being saved comes from THAT item's own document or row in the evidence, not from another item.
3. Values are what the task asked for (e.g. the final total, not a subtotal or a different date field).
Write the EVIDENCE first (quote it), then the verdict. If your evidence shows a problem, the verdict MUST be FAIL
and "fix" must say what to do. Reply with ONLY JSON: {"evidence": "ONE short sentence", "verdict": "PASS" or "FAIL", "fix": "..."}"""

_ID = re.compile(r"\b[A-Z]{2,}-\d+\b")
_MONEY = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\b\d+\.\d{2}\b")


def _tool_calls(msgs):
    return {c["id"]: (c["function"]["name"], c["function"]["arguments"])
            for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls", [])}


def evidence_items(msgs) -> list[tuple[str, str, str]]:
    """(tool name, args, full result) for every observation, excluding finish/remember bookkeeping."""
    calls, out = _tool_calls(msgs), []
    for m in msgs:
        if m["role"] == "tool":
            name, args = calls.get(m["tool_call_id"], ("?", ""))
            content = m["content"]
            advisory = content.startswith(("NOT saved", "ERROR[precommit]", "ERROR[unsubmitted]", "ERROR[verification]",
                                           "ERROR[ungrounded]", "refused:", "ERROR[denied]"))
            if name not in ("finish", "remember", "ask_user") and not advisory:
                out.append((name, args, content))
    return out


_SOURCE_TOOLS = ("read_all_pages", "collect_documents", "read_file", "http_request")
_TRIVIAL = ("typed into", "selected '", "remembered ")


def digest(items, cap: int = 9000, source_cap: int = 6000) -> str:
    """Evidence summary for the auditor. Source material (lists, documents, API reads) is what claims must be
    checked against, so it is always kept; routine page dumps only fill the remaining space, newest first.
    (Keeping just the last N characters used to push the documents out and made the auditor reject correct work.)"""
    rows = []
    for i, (name, args, res) in enumerate(items):
        if res.startswith(_TRIVIAL):
            continue
        is_source = name in _SOURCE_TOOLS
        limit = 3000 if is_source else 700
        res = res.split("--- INTERACTIVE ELEMENTS ---")[0]  # page text only; element lists are noise here
        rows.append((i, is_source, f"[{name} {args[:90]}] {res[:limit]}"))
    keep, used = set(), 0
    for i, is_source, text in reversed(rows):            # newest source material first, within its own budget
        if is_source and used + len(text) <= source_cap:
            keep.add(i); used += len(text)
    for i, is_source, text in reversed(rows):            # then newest routine pages, in the remaining budget
        if not is_source and used + len(text) <= cap:
            keep.add(i); used += len(text)
    return "\n".join(text for i, _, text in rows if i in keep)


def _num(t: str) -> str:
    return t.replace(",", "")


def _window(text: str, start: int, own_id: str, span: int = 700) -> str:
    """Evidence belonging to one entity: from its id up to the next *different* id (or `span` chars)."""
    end = start + span
    for m in _ID.finditer(text, start + len(own_id)):
        if m.group(0) != own_id:
            end = min(end, m.start())
            break
    return text[start:end]


def grounding(claim: str, items) -> Criterion:
    evidence = " ".join(r for _, _, r in items)
    haystack = re.sub(r"(?<=\d),(?=\d)", "", evidence)
    # each id must be paired with the right amounts: amounts in the same clause of the claim must appear
    # in the evidence shortly after that id's first appearance (its own document), not just anywhere.
    mismatched = []
    for clause in re.split(r"[\n;]|(?<=\.)\s", claim):
        ids, money = _ID.findall(clause), [_num(m) for m in _MONEY.findall(clause)]
        if len(ids) != 1 or not money:
            continue
        windows = [_window(haystack, m.start(), ids[0]) for m in re.finditer(re.escape(ids[0]), haystack)]
        if windows and not any(all(a in w for a in money) for w in windows):
            mismatched.append(f"{ids[0]} with {', '.join(money)}")
    missing = []
    for tok in set(_ID.findall(claim)) | {t.replace(",", "") for t in _MONEY.findall(claim)}:
        if tok not in haystack:
            missing.append(tok)
    ok = not missing and not mismatched
    problems = ([f"not found in any observation: {sorted(missing)}"] if missing else []) + \
               ([f"amounts do not match their invoice in the evidence: {mismatched}"] if mismatched else [])
    return Criterion("answer_grounded_in_observations", ok,
                     "all ids/amounts in the answer appear in observations, paired correctly" if ok
                     else "; ".join(problems))


class LLMVerifier:
    def __init__(self, llm, attempts: int = 3):
        self.llm = llm
        self.attempts = attempts

    def _ask_json(self, system: str, user: str, must_have: str) -> dict | None:
        """Ask the auditor model for a JSON object, retrying (a small model often wraps or garbles it)."""
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(self.attempts):
            try:
                data = extract_json(self.llm.chat(msgs, None, fast=True).content)
            except LLMError:
                return None  # auditor unavailable: callers report this visibly as UNVERIFIED

            if data and must_have in data:
                return data
            msgs = msgs[:2] + [{"role": "user", "content": "Reply with ONLY the JSON object, nothing else."}]
        return None

    def review_commit(self, goal: str, criteria: list[str] | None, transcript: list[dict],
                      preview: str) -> tuple[bool, str]:
        """Pre-save review. Returns (ok, reason). If the auditor cannot produce a verdict, the save is allowed
        but the reason says so (the final check still runs)."""
        items = evidence_items(transcript)
        prompt = (f"TASK: {goal}\nSUCCESS CRITERIA:\n" + "\n".join(f"- {c}" for c in (criteria or [])) +
                  f"\nABOUT TO SAVE: {preview}\nEVIDENCE:\n{digest(items)}")
        data = self._ask_json(REVIEW_PROMPT, prompt, "verdict")
        if data is None:
            return True, "UNVERIFIED: pre-save auditor gave no usable verdict"
        ok = str(data.get("verdict", "")).strip().upper().startswith("PASS")
        return ok, f"{data.get('evidence', '')} {data.get('fix', '')}".strip()[:500]

    def check(self, goal, memory, transcript, criteria=None, claim: str = "") -> list[Criterion]:
        items = evidence_items(transcript)
        results = [grounding(claim, items)] if claim else []
        crit = criteria or ["The task is fully completed and the answer is complete and correct."]
        prompt = (f"TASK: {goal}\nSUCCESS CRITERIA:\n" + "\n".join(f"- {c}" for c in crit) +
                  f"\nWORKER'S FINAL ANSWER: {claim}\nEVIDENCE:\n{digest(items)}")
        data = self._ask_json(JUDGE_PROMPT, prompt, "criteria")
        if data and isinstance(data["criteria"], list):
            for c in data["criteria"]:
                verdict = c.get("verdict")
                passed = str(verdict).strip().upper().startswith("PASS") if verdict is not None \
                    else bool(c.get("passed"))
                results.append(Criterion(str(c.get("name", "criterion"))[:80], passed, str(c.get("evidence", ""))[:300]))
            return results
        results.append(Criterion("UNVERIFIED", True, "the auditor model gave no usable verdict after "
                                 f"{self.attempts} tries; only the code-based grounding check was applied"))
        return results
