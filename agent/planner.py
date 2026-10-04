"""Turn a task into checkable success criteria + an approach, before acting."""
from __future__ import annotations

import json
import re

from llm.client import LLMError

PLAN_PROMPT = """You plan work for an AI worker that uses a browser, files and APIs.
Given the task, reply with ONLY a JSON object:
{"kind": "question" | "action" | "both",
 "success_criteria": [3-5 short, checkable statements. Include COMPLETENESS (what full set must be covered: every item, every page); for questions, require the answer to state the substantive details per item (e.g. ids, amounts, dates, status) read from the documents, not just a list of names; for actions, what the saved result must contain],
 "approach": [3-6 short steps. Prefer the site's own search/filter, then read every result page],
 "subtasks": [ONLY if the request names several separate items (e.g. "the latest and the oldest", "for each
   vendor", "A and B"): one END-TO-END instruction per item, doing the whole job for that item (find it AND act on it,
   e.g. enter it), repeating every detail it needs. NEVER split one item's workflow into steps such as "log in",
   "search" or "enter the data". For a single item or a single workflow, return just the one task.
   Split by ITEM, never by PHASE. Example: "Find the latest and oldest invoice from Acme Supplies, extract the
   amount and due date, enter it into our internal system" ->
   ["Find the LATEST invoice from Acme Supplies (compare issue dates of all of them), extract its amount and due date, and enter it into the internal system",
    "Find the OLDEST invoice from Acme Supplies (compare issue dates of all of them), extract its amount and due date, and enter it into the internal system"].
   WRONG: ["identify and extract the invoices", "enter the invoices"] - the second part would have no data.]}
No prose, no markdown."""


def extract_json(text: str | None) -> dict | None:
    """First valid JSON object in the text; tolerates ```json fences, prose, and <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _subtask_text(t) -> str:
    """Models sometimes return {"task": "..."} objects instead of plain strings."""
    if isinstance(t, dict):
        t = t.get("task") or t.get("instruction") or next((v for v in t.values() if isinstance(v, str)), "")
    return str(t).strip()


def make_plan(llm, task: str, context: str = "", allow_split: bool = True) -> dict | None:
    msgs = [{"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": f"Systems:\n{context}\n\nTask: {task}"}]
    for _ in range(2):
        try:
            plan = extract_json(llm.chat(msgs, None, fast=True).content)
        except LLMError:
            return None  # run without a plan rather than failing

        if plan and isinstance(plan.get("success_criteria"), list):
            subtasks = [_subtask_text(t) for t in plan.get("subtasks", [])] if allow_split else []
            subtasks = [t for t in subtasks if t][:4]
            return {"kind": plan.get("kind", "both"), "subtasks": subtasks,
                    "success_criteria": [str(c) for c in plan["success_criteria"]][:6],
                    "approach": [str(a) for a in plan.get("approach", [])][:8]}
    return None


def render_plan(plan: dict) -> str:
    crit = "\n".join(f"- {c}" for c in plan["success_criteria"])
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(plan["approach"], 1))
    return f"Plan (kind: {plan['kind']})\nSuccess criteria:\n{crit}\nApproach:\n{steps}"
