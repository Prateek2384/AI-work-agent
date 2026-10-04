"""Tool registry: a new tool is one decorated function."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

READ, WRITE, IRREVERSIBLE = "read", "write", "irreversible"

# error_type values the recovery logic keys off
TRANSIENT, NOT_FOUND, VALIDATION, AUTH, BAD_ARGS, UNKNOWN = (
    "transient", "not_found", "validation", "auth", "bad_args", "unknown")


class ToolError(Exception):
    def __init__(self, message: str, error_type: str = UNKNOWN):
        super().__init__(message)
        self.error_type = error_type


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    error_type: str | None = None

    def render(self, limit: int = 6000) -> str:
        body = str(self.data) if self.ok else f"ERROR[{self.error_type}]: {self.error}"
        return body if len(body) <= limit else body[:limit] + f"\n...[truncated {len(body) - limit} chars]"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    risk: str = READ


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def tool(self, name: str, description: str, parameters: dict | None = None, risk: str = READ):
        schema = {"type": "object", "properties": {}, "required": []}
        schema.update(parameters or {})

        def deco(fn):
            self.tools[name] = Tool(name, description, schema, fn, risk)
            return fn
        return deco

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}}
                for t in self.tools.values()]

    def dispatch(self, name: str, args: dict) -> ToolResult:
        t = self.tools.get(name)
        if t is None:
            return ToolResult(False, error=f"unknown tool '{name}'", error_type=BAD_ARGS)
        if "_parse_error" in args:
            return ToolResult(False, error="tool arguments were not valid JSON; resend", error_type=BAD_ARGS)
        missing = [k for k in t.parameters.get("required", []) if k not in args]
        if missing:
            return ToolResult(False, error=f"missing required args: {missing}", error_type=BAD_ARGS)
        try:
            return ToolResult(True, data=t.fn(**args))
        except ToolError as e:
            return ToolResult(False, error=str(e), error_type=e.error_type)
        except TypeError as e:
            return ToolResult(False, error=str(e), error_type=BAD_ARGS)
        except Exception as e:  # tool bugs must not crash the agent
            return ToolResult(False, error=f"{type(e).__name__}: {e}", error_type=UNKNOWN)
