"""Integration: real Chrome + the sim apps, driven by hand (no LLM) through the headline flow."""
import re

import pytest

from env import erp
from env.data import EXPECTED
from env.run import ERP_URL, PORTAL_URL, start_all, stop_all
from tools.files import _safe, register_file_tools
from tools.registry import Registry, ToolError

pw = pytest.importorskip("playwright")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from tools.browser import Browser
    root = tmp_path_factory.mktemp("run")
    servers = start_all()
    try:
        b = Browser(root / "downloads")
    except Exception as e:  # no usable browser on this machine
        stop_all(servers)
        pytest.skip(f"browser unavailable: {e}")
    reg = Registry()
    register_file_tools(reg, root)
    yield b, reg, root
    b.close()
    stop_all(servers)


def eid(page_text: str, pattern: str) -> str:
    """Find the element id on the line matching pattern."""
    for line in page_text.splitlines():
        if line.startswith("[e") and re.search(pattern, line):
            return re.match(r"\[(e\d+)\]", line).group(1)
    raise AssertionError(f"no element matching {pattern!r} in:\n{page_text}")


def test_full_manual_flow(world):
    b, reg, root = world
    erp.reset()

    # login
    page = b.goto(PORTAL_URL + "/login")
    assert "Username" in page
    b.type_text(eid(page, "Username"), "ap-team")
    b.type_text(eid(page, "Password"), "portal123")
    page = b.click(eid(page, r"button.*Sign in"))
    assert "Invoices" in page and "Page 1 of 3" in page

    # search + paginate to find the target (the decoy is NOT the answer)
    b.type_text(eid(page, "Vendor"), "Acme Supplies")
    page = b.click(eid(page, r"button.*Search"))
    assert "Acme Supply Co" not in page and "Page 1 of 2" in page
    assert EXPECTED.invoice_no not in page
    page = b.click(eid(page, "Next page"))
    assert EXPECTED.invoice_no in page

    # download + extract
    path = b.download(eid(page, f"Download PDF.*{EXPECTED.invoice_no}"))  # row context disambiguates links
    res = reg.dispatch("read_file", {"path": path})
    assert res.ok and f"INR {EXPECTED.amount:,.2f}" in res.data and EXPECTED.invoice_no in res.data


def test_erp_validation_error_visible_and_500_reported(world):
    b, _, _ = world
    erp.reset()
    page = b.goto(ERP_URL + "/bills/new")
    b.select_option(eid(page, "Vendor"), "Acme Supplies")
    b.type_text(eid(page, "Invoice number"), "INV-2047")
    b.type_text(eid(page, "Amount"), "48250.00")
    b.type_text(eid(page, "Due date"), "18/10/2026")  # wrong format on purpose
    page = b.click(eid(page, r"button.*Save bill"))
    assert "YYYY-MM-DD" in page and "HTTP 422" in page
    # fields keep their values so the agent can fix just the date
    b.type_text(eid(page, "Due date"), "2026-10-18")
    page = b.click(eid(page, r"button.*Save bill"))
    assert "INV-2047" in page and "48,250.00" in page
    assert erp.BILLS[0]["due_date"] == "2026-10-18"


def test_stale_element_id_is_reported(world):
    b, _, _ = world
    b.goto(ERP_URL + "/bills")
    with pytest.raises(ToolError) as e:
        b.click("e999")
    assert e.value.error_type == "not_found"


def test_file_sandbox(tmp_path):
    with pytest.raises(ToolError):
        _safe(tmp_path, "/etc/passwd")


def test_read_all_pages_and_collect_documents(world):
    b, _, _ = world
    page = b.goto(PORTAL_URL + "/login")
    b.type_text(eid(page, "Username"), "ap-team")
    b.type_text(eid(page, "Password"), "portal123")
    b.click(eid(page, r"button.*Sign in"))
    page = b.goto(PORTAL_URL + "/invoices?q=Acme+Supplies")
    allp = b.read_all_pages()
    assert allp.count("Acme Supplies") >= 9 and "INV-2047" in allp and "last page reached=True" in allp
    page = b.goto(PORTAL_URL + "/invoices")  # unfiltered list; row_contains must do the narrowing
    docs = b.collect_documents(row_contains="Globex")
    assert docs.count("## ") == 3 and "Acme" not in docs
    for no, amt in (("INV-5498", "4,120.00"), ("INV-5510", "78,200.50"), ("INV-5521", "91,000.00")):
        assert no in docs and f"INR {amt}" in docs


def test_enter_key_login_returns_the_new_page_not_a_navigation_error(world):
    b, _, _ = world
    b.goto(PORTAL_URL + "/login")  # fresh login page (session cookie may exist; login still works)
    page = b.read_page()
    b.type_text(eid(page, "Username"), "ap-team")
    out = b.type_text(eid(page, "Password"), "portal123", submit=True)
    assert "Invoices" in out and "URL: " + PORTAL_URL + "/invoices" in out


def test_unsubmitted_form_is_detected_until_saved(world):
    b, _, _ = world
    erp.reset()
    page = b.goto(ERP_URL + "/bills/new")
    assert b.unsubmitted() is None
    b.select_option(eid(page, "Vendor"), "Acme Supplies")
    b.type_text(eid(page, "Invoice number"), "INV-9")
    assert "never submitted" in b.unsubmitted()
    b.type_text(eid(page, "Amount"), "100.00")
    b.type_text(eid(page, "Due date"), "2026-10-18")
    b.click(eid(page, r"button.*Save bill"))
    assert b.unsubmitted() is None and erp.BILLS[0]["invoice_no"] == "INV-9"


def test_commit_preview_describes_data_entry_but_not_login_or_search(world):
    b, _, _ = world
    erp.reset()
    page = b.goto(ERP_URL + "/bills/new")
    save = eid(page, r"button.*Save bill")
    assert b.commit_preview("click", {"element_id": save}) is None      # empty form: nothing to save
    b.select_option(eid(page, "Vendor"), "Acme Supplies")
    b.type_text(eid(page, "Invoice number"), "INV-9")
    b.type_text(eid(page, "Amount"), "100.00")
    preview = b.commit_preview("click", {"element_id": save})
    assert "vendor=Acme Supplies" in preview and "invoice_no=INV-9" in preview and "amount=100.00" in preview
    due = eid(page, "Due date")
    assert "due_date=2026-10-18" in b.commit_preview("type_text", {"element_id": due, "text": "2026-10-18", "submit": True})
    assert b.commit_preview("type_text", {"element_id": due, "text": "x", "submit": False}) is None

    login = b.goto(PORTAL_URL + "/login")          # POST form with a password: never reviewed
    b.type_text(eid(login, "Username"), "ap-team")
    assert b.commit_preview("click", {"element_id": eid(login, r"button")}) is None
    inv = b.goto(PORTAL_URL + "/invoices?q=x")      # GET search form: never reviewed
    assert b.commit_preview("click", {"element_id": eid(inv, r"button.*Search|button")}) is None


def test_bulk_tools_leave_the_browser_where_they_found_it(world):
    b, _, _ = world
    b.goto(PORTAL_URL + "/login")
    page = b.read_page()
    b.type_text(eid(page, "Username"), "ap-team")
    b.type_text(eid(page, "Password"), "portal123", submit=True)
    b.goto(PORTAL_URL + "/invoices?q=Acme+Supplies")
    start = b.page.url
    b.read_all_pages()
    assert b.page.url == start                            # back on page 1, not stranded on the last page
    docs = b.collect_documents(row_contains="INV-2029")   # an invoice that is on page 1 of the list
    assert "INV-2029" in docs and "INR 8,800.00" in docs and b.page.url == start
    b.goto(PORTAL_URL + "/invoices?q=Nobody")
    assert "no matching documents" in b.collect_documents(row_contains="zzz")


def test_bulk_tools_cover_the_whole_list_even_after_paging_forward_by_hand(world):
    b, _, _ = world
    b.goto(PORTAL_URL + "/login")
    page = b.read_page()
    b.type_text(eid(page, "Username"), "ap-team")
    b.type_text(eid(page, "Password"), "portal123", submit=True)
    page = b.goto(PORTAL_URL + "/invoices?q=Acme+Supplies")
    first = b.page.url
    b.click(eid(page, "Next page"))                       # the agent pages forward by hand...
    assert b.page.url != first
    docs = b.collect_documents(row_contains="Acme Supplies")   # ...then asks for everything
    assert docs.count("## ") == 9 and "INV-2029" in docs        # page 1 AND page 2 documents
    assert b.page.url == first                            # and the browser is back on page 1
