"""Scratchpad of facts discovered during a run; re-injected into every LLM turn."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fact:
    value: str
    source: str


class Memory:
    def __init__(self):
        self.facts: dict[str, Fact] = {}

    def remember(self, key: str, value: str, source: str = "") -> None:
        self.facts[key] = Fact(str(value), source)

    def get(self, key: str) -> str | None:
        f = self.facts.get(key)
        return f.value if f else None

    def render(self) -> str:
        if not self.facts:
            return "(nothing remembered yet)"
        return "\n".join(f"- {k}: {f.value}" + (f"  [source: {f.source}]" if f.source else "")
                         for k, f in self.facts.items())
