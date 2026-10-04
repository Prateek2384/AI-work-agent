"""File tools, sandboxed to the run's working directory."""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from tools.registry import NOT_FOUND, VALIDATION, Registry, ToolError


def pdf_text(path: str | Path) -> str:
    return "\n".join(pg.extract_text() or "" for pg in PdfReader(path).pages).strip()


def _safe(root: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():  # accept both run-dir-relative and cwd-relative spellings
        p = root / p if (root / p).exists() else p
    p = p.resolve()
    if root.resolve() not in p.parents and p != root.resolve():
        raise ToolError(f"path outside the allowed directory {root}", VALIDATION)
    if not p.exists():
        raise ToolError(f"no such file: {path}", NOT_FOUND)
    return p


def register_file_tools(reg: Registry, root: str | Path) -> None:
    root = Path(root)

    @reg.tool("list_files", "List downloaded/working files.", {"properties": {}})
    def _list():
        return "\n".join(str(p) for p in sorted(root.rglob("*")) if p.is_file()) or "(no files)"

    @reg.tool("read_file", "Read a text or PDF file (PDF text is extracted). Returns its text.",
              {"properties": {"path": {"type": "string"}}, "required": ["path"]})
    def _read(path):
        p = _safe(root, path)
        if p.suffix.lower() == ".pdf":
            text = pdf_text(p)
            return text or "(PDF has no extractable text; it may be scanned)"
        return p.read_text(errors="replace")[:8000]
