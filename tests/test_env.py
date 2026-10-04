import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from env import erp
from env.data import EXPECTED, INVOICES
from env.vendor_portal import app as portal_app


@pytest.fixture()
def portal():
    c = TestClient(portal_app, follow_redirects=False)
    r = c.post("/login", data={"username": "ap-team", "password": "portal123"})
    assert r.status_code == 303
    return c


@pytest.fixture()
def erp_client():
    erp.reset()
    return TestClient(erp.app, follow_redirects=False)


def test_portal_requires_login():
    c = TestClient(portal_app, follow_redirects=False)
    assert c.get("/invoices").status_code in (302, 307)
    assert c.get("/invoices/1/pdf").status_code == 401


def test_bad_credentials_rejected():
    c = TestClient(portal_app, follow_redirects=False)
    r = c.post("/login", data={"username": "x", "password": "y"})
    assert "error=" in r.headers["location"]


def test_search_hits_decoy_vendor_and_paginates(portal):
    p1 = portal.get("/invoices", params={"q": "acme"}).text
    assert "Acme Supply Co" in p1 and "Page 1 of 3" in p1
    assert EXPECTED.invoice_no not in p1  # latest Acme Supplies invoice is NOT on page 1
    p2 = portal.get("/invoices", params={"q": "acme", "page_no": 2}).text
    p3 = portal.get("/invoices", params={"q": "acme", "page_no": 3}).text
    assert EXPECTED.invoice_no in p1 + p2 + p3


def test_pdf_contains_amount_and_due_date(portal):
    r = portal.get(f"/invoices/{EXPECTED.id}/pdf")
    text = PdfReader(io.BytesIO(r.content)).pages[0].extract_text()
    assert f"INR {EXPECTED.amount:,.2f}" in text
    assert "18 October 2026" in text and EXPECTED.invoice_no in text


def test_decoy_is_newer_than_target():
    decoy = next(i for i in INVOICES if i.vendor == "Acme Supply Co" and i.invoice_no == "INV-3001")
    assert decoy.issued > EXPECTED.issued and EXPECTED.vendor == "Acme Supplies"


def good(**kw):
    return {"vendor": "Acme Supplies", "invoice_no": "INV-2047", "amount": "48250.00",
            "due_date": "2026-10-18", **kw}


def test_erp_validation(erp_client):
    assert erp_client.post("/bills", data=good(due_date="18/10/2026")).status_code == 422
    assert erp_client.post("/bills", data=good(amount="abc")).status_code == 422
    assert erp_client.post("/bills", data=good(vendor="")).status_code == 422
    assert erp_client.get("/api/bills").json() == []


def test_erp_create_and_duplicate(erp_client):
    assert erp_client.post("/bills", data=good()).status_code == 303
    b = erp_client.get("/api/bills/INV-2047").json()
    assert b["amount"] == 48250.0 and b["due_date"] == "2026-10-18"
    assert erp_client.post("/bills", data=good()).status_code == 409


def test_erp_flaky_first_post(erp_client, monkeypatch):
    monkeypatch.setenv("FLAKY", "1")
    assert erp_client.post("/bills", data=good()).status_code == 500
    assert erp_client.get("/api/bills").json() == []
    assert erp_client.post("/bills", data=good()).status_code == 303


def test_layout_variant_changes_labels(portal, monkeypatch):
    assert "Download PDF" in portal.get("/invoices").text
    monkeypatch.setenv("LAYOUT_VARIANT", "b")
    assert "Get invoice (PDF)" in portal.get("/invoices").text
