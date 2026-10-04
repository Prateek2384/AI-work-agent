"""Usage: python cli.py "<task>" [--visible] [--flaky] [--variant b] [--max-steps 40]"""
from __future__ import annotations

import argparse
import os
import sys

from agent.loop import Agent
from agent.trace import Trace
from agent.verifier import LLMVerifier
from env import erp
from env.run import ERP_URL, PORTAL_URL, start_all, stop_all
from llm.client import LLMClient
from tools.setup import build_tools

HEADLINE = ("Find the latest invoice from Acme Supplies, extract the amount and due date, "
            "enter it into our internal system, and tell me once it is done.")


def cli_approve(name: str, args: dict) -> bool:
    try:
        return input(f"\n⚠️  Approve irreversible action {name}({args})? [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def cli_ask(question: str) -> str:
    try:
        return input(f"\n❓ {question}\n> ")
    except EOFError:  # not an interactive terminal
        return "(no user is available to answer; use the safest assumption or stop and report the blocker)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default=HEADLINE)
    ap.add_argument("--visible", action="store_true", help="show the browser window")
    ap.add_argument("--flaky", action="store_true", help="ERP fails the first save with a 500")
    ap.add_argument("--variant", default="a", help="UI label variant (a|b)")
    ap.add_argument("--allow-questions", action="store_true", help="let the agent ask you questions (default: it decides itself)")
    ap.add_argument("--max-steps", type=int, default=60)
    a = ap.parse_args(argv)

    os.environ["LAYOUT_VARIANT"] = a.variant
    os.environ["FLAKY"] = "1" if a.flaky else "0"
    servers = start_all()
    erp.reset()
    trace = Trace(echo=True)
    registry, browser, notes = build_tools(trace.dir, PORTAL_URL, ERP_URL, headless=not a.visible)
    try:
        print(f"Task: {a.task}\nRun dir: {trace.dir}\n")
        llm = LLMClient()
        agent = Agent(llm, registry, trace, max_steps=a.max_steps, context=notes, use_planner=True,
                      verifier=LLMVerifier(llm) if os.environ.get("VERIFY", "1") != "0" else None, approve=cli_approve, ask_user=cli_ask, pending_check=browser.unsubmitted, allow_questions=a.allow_questions,
                      commit_preview=browser.commit_preview)
        res = agent.run(a.task)
        print(f"\n=== {res.status.upper()} in {res.steps} steps ===\n{res.summary}")
        for c in res.criteria:
            print(f"  {'✓' if c.passed else '✗'} {c.name}: {c.evidence}")
        print("\nERP bills now:")
        for b in erp.BILLS:
            print(f"  {b['invoice_no']}  {b['vendor']}  {b['amount']:,.2f}  due {b['due_date']}  [{b['status']}]")
        if not erp.BILLS:
            print("  (none)")
        print(f"\nEvidence: {trace.dir}")
        return 0 if res.status == "done" else 1
    finally:
        browser.close()
        stop_all(servers)


if __name__ == "__main__":
    sys.exit(main())
