"""JSONL event log; the run directory doubles as the evidence bundle."""
from __future__ import annotations

import json
import time
from pathlib import Path


class Trace:
    def __init__(self, root: str | Path = "runs", run_id: str | None = None, echo: bool = False, sink=None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trace.jsonl"
        self.echo = echo
        self.sink = sink  # optional callable(event, data) for live UIs

    def log(self, event: str, **data) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "event": event, **data}, default=str) + "\n")
        if self.echo:
            self._print(event, data)
        if self.sink:
            self.sink(event, data)

    @staticmethod
    def _print(event: str, d: dict) -> None:
        if event == "llm":
            if d.get("content"):
                print(f"  💭 {d['content'][:200]}")
            for name, args in d.get("calls", []):
                print(f"  → {name}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in args.items())})")
        elif event == "tool":
            mark = "✓" if d["ok"] else f"✗ [{d['error_type']}]"
            first = d["result"].splitlines()[0][:110] if d["result"] else ""
            print(f"    {mark} {first}")
        elif event == "phase":
            print(f"  ⏳ {d['name']}…")
        elif event in ("verify", "budget_exhausted", "finish"):
            print(f"  == {event}")
