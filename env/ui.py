"""Tiny HTML helpers + layout variants (labels/ids change so nothing can be hard-coded)."""
from __future__ import annotations

import html
import os

LABELS = {
    "a": {"login": "Sign in", "search": "Search", "download": "Download PDF", "next": "Next page",
          "save": "Save bill", "new": "New bill"},
    "b": {"login": "Log in", "search": "Filter", "download": "Get invoice (PDF)", "next": "Older invoices",
          "save": "Submit bill", "new": "Add bill"},
}


def labels() -> dict:
    return LABELS.get(os.environ.get("LAYOUT_VARIANT", "a"), LABELS["a"])


def esc(x) -> str:
    return html.escape(str(x))


def page(title: str, body: str, nav: str = "") -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title></head>"
            f"<body><h1>{esc(title)}</h1>{nav}<hr>{body}</body></html>")
