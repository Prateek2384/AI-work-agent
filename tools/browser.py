"""Browser tools (Playwright). Observations are text-first: page text + a numbered list of
interactive elements, so the model acts by element id rather than guessing selectors or reading pixels."""
from __future__ import annotations

import os
import re
from pathlib import Path

from playwright.sync_api import Error as PWError
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

from tools.registry import NOT_FOUND, TRANSIENT, VALIDATION, Registry, ToolError, WRITE

# Tags every visible interactive element with data-agent-id and returns a description of each.
_SNAPSHOT_JS = """
() => {
  const sel = 'a[href], button, input:not([type=hidden]), select, textarea, [role=button]';
  const out = []; let n = 0;
  document.querySelectorAll('[data-agent-id]').forEach(e => e.removeAttribute('data-agent-id'));
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || (r.width === 0 && r.height === 0)) continue;
    const id = 'e' + (++n); el.setAttribute('data-agent-id', id);
    const label = (el.labels && el.labels[0] ? el.labels[0].innerText.split('\\n')[0] : '') ||
                  el.getAttribute('aria-label') || el.placeholder || '';
    const d = {id, tag: el.tagName.toLowerCase(), type: el.type || '', name: el.name || '',
               label: label.trim(), text: (el.innerText || '').trim().slice(0, 80),
               value: ('value' in el && el.tagName !== 'BUTTON') ? el.value : '', href: el.getAttribute('href') || ''};
    const row = el.closest('tr'); d.row = row ? row.innerText.replace(/\\s+/g, ' ').trim().slice(0, 120) : '';
    if (el.tagName === 'SELECT') d.options = Array.from(el.options).map(o => o.text);
    out.push(d);
  }
  return {title: document.title, text: document.body ? document.body.innerText : '', elements: out};
}
"""


class Browser:
    def __init__(self, download_dir: str | Path, headless: bool | None = None,
                 channel: str | None = None, timeout_ms: int = 8000):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_ms = timeout_ms
        if headless is None:
            headless = os.environ.get("HEADLESS", "1") != "0"
        self._pw = sync_playwright().start()
        channel = channel or os.environ.get("BROWSER_CHANNEL", "chrome")  # system Chrome; "" = bundled chromium
        self._browser = self._pw.chromium.launch(headless=headless, channel=channel or None)
        self._ctx = self._browser.new_context(accept_downloads=True)
        self.page = self._ctx.new_page()
        self.page.set_default_timeout(timeout_ms)
        self.last_status: int | None = None
        self._chain_start: str | None = None  # first page of a paginated list the agent is paging through
        self.typed: list[str] = []  # fields filled since the last page load (i.e. not yet submitted)
        self.page.on("response", self._on_response)

    def _on_response(self, resp):
        if resp.request.resource_type == "document" and resp.frame == self.page.main_frame:
            self.last_status = resp.status
            self.typed.clear()  # a page load (form submit / navigation) ends the pending-input state

    _FORM_JS = """
    (el) => {
      const f = el.form; if (!f) return null;
      const submit = el.matches('button:not([type]), button[type=submit], input[type=submit]');
      const fields = [...f.elements].filter(e => e.name && e.type !== 'submit' && e.tagName !== 'BUTTON')
        .map(e => [e.name, e.tagName === 'SELECT' ? (e.value !== '' && e.selectedOptions[0] ? e.selectedOptions[0].text : '') : e.value]);
      const auth = [...f.elements].some(e => e.type === 'password');
      return {submit, method: (f.method || 'get').toLowerCase(), fields, auth};
    }
    """

    def commit_preview(self, tool: str, args: dict) -> str | None:
        """If this action would submit a data-entry form (POST, not a login/search), describe exactly what
        would be saved; otherwise None. Lets the agent review values BEFORE they are committed."""
        if tool not in ("click", "type_text") or (tool == "type_text" and not args.get("submit")):
            return None
        try:
            el = self._el(args["element_id"])
            info = self.page.evaluate(self._FORM_JS, el.element_handle())
        except Exception:
            return None
        if not info or info["method"] != "post" or info["auth"]:
            return None
        if tool == "click" and not info["submit"]:
            return None
        fields = dict(info["fields"])
        if tool == "type_text":  # Enter on a field: the typed text is part of what gets submitted
            name = self._field_name(el)
            fields[name] = args.get("text", "")
        filled = {k: v for k, v in fields.items() if str(v).strip()}
        if not filled:
            return None
        return f"submitting a form on {self.page.url} with " + "; ".join(f"{k}={v}" for k, v in filled.items())

    def unsubmitted(self) -> str | None:
        """Describe form input typed but never submitted, or None."""
        if not self.typed:
            return None
        return (f"you typed into {len(self.typed)} form field(s) on {self.page.url} ({', '.join(self.typed[:6])}) "
                "but never submitted the form (click its Save/Submit button), so nothing was saved")

    def close(self):
        for closer in (self._ctx.close, self._browser.close, self._pw.stop):
            try:
                closer()
            except Exception:
                pass

    # ---- observation -------------------------------------------------------------------------
    def read_page(self, max_text: int = 2200) -> str:
        snap = self._snapshot()
        text = "\n".join(ln.strip() for ln in snap["text"].splitlines() if ln.strip())
        if len(text) > max_text:
            text = text[:max_text] + f"\n...[page text truncated, {len(text) - max_text} chars]"
        lines = []
        for e in snap["elements"][:80]:
            desc = e["label"] or e["text"] or e["name"] or e["href"]
            extra = []
            if e["tag"] == "input" or e["tag"] == "textarea":
                extra.append(f'type={e["type"] or "text"} value="{e["value"]}"')
            if e["tag"] == "select":
                extra.append(f'options={e["options"]} value="{e["value"]}"')
            if e["tag"] == "a":
                extra.append(f'href={e["href"]}')
            if e["row"]:
                extra.append(f'(in row: {e["row"]})')
            lines.append(f'[{e["id"]}] {e["tag"]} "{desc}" ' + " ".join(extra))
        status = f" (HTTP {self.last_status})" if self.last_status and self.last_status >= 400 else ""
        return (f"URL: {self.page.url}{status}\nTITLE: {snap['title']}\n--- PAGE TEXT ---\n{text}\n"
                f"--- INTERACTIVE ELEMENTS ---\n" + "\n".join(lines))

    def _snapshot(self):
        """Snapshot the page; if a navigation is in flight (Enter on a form, a click), wait and retry."""
        last = None
        for _ in range(4):
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=4000)
                return self.page.evaluate(_SNAPSHOT_JS)
            except (PWError, PWTimeout) as e:
                last = e
                if "Execution context was destroyed" not in str(e) and not isinstance(e, PWTimeout):
                    break
        raise ToolError(f"page not readable: {str(last).splitlines()[0]}", TRANSIENT)

    def screenshot(self, name: str = "shot.png") -> str:
        path = self.download_dir.parent / name
        self._guard(lambda: self.page.screenshot(path=str(path), full_page=True))
        return str(path)

    # ---- actions -----------------------------------------------------------------------------
    def goto(self, url: str) -> str:
        self._chain_start = None
        self._guard(lambda: self.page.goto(url, wait_until="domcontentloaded"))
        return self.read_page()

    def _el(self, element_id: str):
        loc = self.page.locator(f'[data-agent-id="{element_id}"]')
        if loc.count() == 0:
            raise ToolError(f"element {element_id} not found; the page changed — call read_page "
                            f"and use the new ids", NOT_FOUND)
        return loc.first

    def click(self, element_id: str) -> str:
        el = self._el(element_id)
        before = self.page.url
        try:
            is_next = bool(self._NEXT.search((el.inner_text(timeout=1000) or "").strip()))
        except Exception:
            is_next = False
        if is_next and self._chain_start is None:
            self._chain_start = before  # remember page 1 of this list
        elif not is_next:
            self._chain_start = None
        self._guard(lambda: el.click())
        self._settle()
        return self.read_page()

    def type_text(self, element_id: str, text: str, clear: bool = True, submit: bool = False) -> str:
        el = self._el(element_id)
        self._guard(lambda: el.fill(text) if clear else el.type(text))
        if not submit:
            self.typed.append(self._field_name(el))
        if submit:  # press Enter, e.g. to run a search box
            self._chain_start = None
            self._guard(lambda: el.press("Enter"))
            self._settle()
            try:
                self.page.wait_for_load_state("load", timeout=4000)
            except PWTimeout:
                pass
            return self.read_page()
        return f"typed into {element_id}"

    def select_option(self, element_id: str, option: str) -> str:
        el = self._el(element_id)
        self._guard(lambda: el.select_option(label=option))
        self.typed.append(self._field_name(el))
        return f"selected '{option}' in {element_id}"

    def download(self, element_id: str) -> str:
        el = self._el(element_id)
        with self.page.expect_download(timeout=self.timeout_ms) as dl_info:
            self._guard(lambda: el.click())
        dl = dl_info.value
        path = self.download_dir / dl.suggested_filename
        dl.save_as(path)
        return str(path.resolve())

    # ---- bulk helpers: systematic enumeration of paginated lists ------------------------------
    _NEXT = re.compile(r"\bnext\b|\bolder\b|\bmore\b|[›»>]\s*$", re.I)

    def _clean(self, text: str) -> str:
        return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())

    def _next_el(self, snap) -> str | None:
        for e in snap["elements"]:
            if e["tag"] in ("a", "button") and self._NEXT.search((e["text"] or e["label"] or "").strip()):
                return e["id"]
        return None

    def read_all_pages(self, max_pages: int = 10, per_page: int = 1500) -> str:
        """Read the text of the current paginated list and every following page."""
        start = self._rewind_to_first_page()
        parts, seen = [], set()
        for n in range(1, max_pages + 1):
            snap = self._snapshot()
            if self.page.url in seen:
                break
            seen.add(self.page.url)
            parts.append(f"--- page {n} ({self.page.url}) ---\n{self._clean(snap['text'])[:per_page]}")
            nxt = self._next_el(snap)
            if not nxt:
                break
            self.click(nxt)
        reached_end = self._next_el(self._snapshot()) is None
        self._return_to(start)  # leave the browser where it was found, so later tools start at page 1 again
        return "\n".join(parts) + f"\n[read {len(parts)} page(s); last page reached={reached_end}]"

    def collect_documents(self, link_text: str = "pdf", row_contains: str = "", max_pages: int = 10,
                          max_chars: int = 600) -> str:
        """Download EVERY matching link (default: PDF/download links) on this page and all following
        pages, and return each document's row context + extracted text. `row_contains` keeps only rows whose
        text contains it (e.g. a vendor name), so unrelated documents are not downloaded."""
        from tools.files import pdf_text

        start = self._rewind_to_first_page()
        want, out, seen = link_text.lower(), [], set()
        for _ in range(max_pages):
            snap = self._snapshot()
            if self.page.url in seen:
                break
            seen.add(self.page.url)
            for e in snap["elements"]:
                if (e["tag"] == "a" and want in (e["text"] + e["label"] + e["href"]).lower()
                        and row_contains.lower() in e["row"].lower()):
                    try:
                        path = self.download(e["id"])
                        body = pdf_text(path)[:max_chars] if path.lower().endswith(".pdf") else "(not a PDF)"
                    except Exception as ex:  # keep going; report the failure for this document
                        path, body = "", f"(failed: {ex})"
                    body = re.sub(r"\s+", " ", body)
                    out.append(f"## {e['row'] or e['href']}\n{Path(path).name if path else ''}: {body}")
            nxt = self._next_el(snap)
            if not nxt:
                break
            self.click(nxt)
        self._return_to(start)
        return "\n\n".join(out) if out else (f"(no matching documents found; looked at {len(seen)} page(s) "
                                              f"starting from {start}; if the list has earlier pages, open "
                                              "the first page of the list and retry)")

    @staticmethod
    def _field_name(el) -> str:
        try:
            return el.get_attribute("name") or el.get_attribute("id") or "field"
        except Exception:
            return "field"

    def _rewind_to_first_page(self) -> str:
        """Bulk tools must cover the WHOLE list: if the agent already paged forward, go back to page 1."""
        if self._chain_start and self.page.url != self._chain_start:
            self._return_to(self._chain_start)
        return self.page.url

    def _return_to(self, url: str) -> None:
        if self.page.url != url:
            try:
                self.page.goto(url, wait_until="domcontentloaded")
            except (PWError, PWTimeout):
                pass

    # ---- internals ---------------------------------------------------------------------------
    def _settle(self):
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=3000)
        except PWTimeout:
            pass

    def _guard(self, fn):
        try:
            return fn()
        except PWTimeout as e:
            raise ToolError(f"timed out: {str(e).splitlines()[0]}", TRANSIENT)
        except PWError as e:
            msg = str(e).splitlines()[0]
            if "net::ERR" in msg or "Execution context was destroyed" in msg:
                raise ToolError(msg, TRANSIENT)
            raise ToolError(msg, VALIDATION)


def register_browser_tools(reg: Registry, b: Browser) -> None:
    ID = {"element_id": {"type": "string", "description": "id like e3 from the latest read_page"}}

    @reg.tool("goto", "Open a URL in the browser; returns the page text and numbered interactive elements.",
              {"properties": {"url": {"type": "string"}}, "required": ["url"]})
    def _goto(url): return b.goto(url)

    @reg.tool("read_page", "Re-read the current page (text + numbered interactive elements).")
    def _read(): return b.read_page()

    @reg.tool("click", "Click an element by id. Returns the resulting page.",
              {"properties": ID, "required": ["element_id"]}, risk=WRITE)
    def _click(element_id): return b.click(element_id)

    @reg.tool("type_text", "Type text into an input by id (replaces value). submit=true also presses Enter "
              "(use for search/filter boxes) and returns the resulting page.",
              {"properties": {**ID, "text": {"type": "string"}, "submit": {"type": "boolean"}},
               "required": ["element_id", "text"]}, risk=WRITE)
    def _type(element_id, text, submit=False): return b.type_text(element_id, text, submit=submit)

    @reg.tool("select_option", "Choose an option (by visible text) in a <select> by id.",
              {"properties": {**ID, "option": {"type": "string"}}, "required": ["element_id", "option"]}, risk=WRITE)
    def _select(element_id, option): return b.select_option(element_id, option)

    @reg.tool("read_all_pages", "Read the current list AND all following pages (follows 'next'). Use after "
              "filtering/searching to see every result at once.")
    def _all(): return b.read_all_pages()

    @reg.tool("collect_documents", "Download every document link (default PDFs) on this list and all following "
              "pages; returns each row plus the document's text. Use row_contains (e.g. a vendor name) to take only "
              "matching rows.",
              {"properties": {"link_text": {"type": "string", "description": "substring of the link, default 'pdf'"},
                              "row_contains": {"type": "string", "description": "only rows containing this text"}}})
    def _collect(link_text="pdf", row_contains=""): return b.collect_documents(link_text, row_contains)

    @reg.tool("screenshot", "Save a full-page screenshot as evidence; returns the path.")
    def _shot(): return b.screenshot()
