"""HTTP/API tool, restricted to an allowlist of hosts."""
from __future__ import annotations

from urllib.parse import urlparse

import httpx

from tools.registry import TRANSIENT, VALIDATION, Registry, ToolError, WRITE


def register_http_tools(reg: Registry, allowed_hosts: set[str]) -> None:
    @reg.tool("http_request", "Call an internal HTTP/JSON API. Returns status and body.",
              {"properties": {"method": {"type": "string", "enum": ["GET", "POST"]},
                              "url": {"type": "string"},
                              "json_body": {"type": "object"}}, "required": ["method", "url"]}, risk=WRITE)
    def _req(method, url, json_body=None):
        if urlparse(url).netloc not in allowed_hosts:
            raise ToolError(f"host not allowed; allowed: {sorted(allowed_hosts)}", VALIDATION)
        try:
            r = httpx.request(method, url, json=json_body, timeout=10)
        except httpx.HTTPError as e:
            raise ToolError(str(e), TRANSIENT)
        return f"HTTP {r.status_code}\n{r.text[:4000]}"
