"""Assemble the default toolset for a run."""
from __future__ import annotations

from pathlib import Path

from tools.browser import Browser, register_browser_tools
from tools.files import register_file_tools
from tools.http_api import register_http_tools
from tools.registry import Registry

ENV_NOTES = """You work at Hawkvision. Systems available (all in the browser unless noted):
- Vendor portal: {portal}/login  (username ap-team, password portal123). Suppliers' invoices are PDFs there.
- Internal ERP: {erp}/bills  (NO login needed; enter supplier bills via the "Bills" form). ERP JSON API: {erp}/api/bills
Vendor names must match exactly; similarly-named vendors are different companies."""


def build_tools(run_dir: str | Path, portal_url: str, erp_url: str, **browser_kw):
    """Returns (registry, browser, context_notes)."""
    run_dir = Path(run_dir)
    browser = Browser(run_dir / "downloads", **browser_kw)
    reg = Registry()
    register_browser_tools(reg, browser)
    register_file_tools(reg, run_dir)
    hosts = {u.split("://", 1)[1] for u in (portal_url, erp_url)}
    register_http_tools(reg, hosts)
    return reg, browser, ENV_NOTES.format(portal=portal_url, erp=erp_url)
