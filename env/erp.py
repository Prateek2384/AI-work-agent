"""Internal ERP: a "Bills" form with validation, duplicate detection and injectable flakiness.

FLAKY=1 makes the first POST /bills after (re)start fail with a 500 to exercise retry logic.
The JSON API (/api/bills) is the verifier's ground truth.
"""
from __future__ import annotations

import os
import threading
from datetime import date

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from env.data import VENDORS
from env.ui import esc, labels, page

app = FastAPI(title="Internal ERP")
_lock = threading.Lock()
BILLS: list[dict] = []
_state = {"posts": 0}


def reset():
    with _lock:
        BILLS.clear()
        _state["posts"] = 0


def _form(error: str = "", values: dict | None = None) -> str:
    v = values or {}
    L = labels()
    opts = "".join(f"<option{' selected' if v.get('vendor') == x else ''}>{esc(x)}</option>" for x in VENDORS)
    err = f"<p role='alert' style='color:red'>{esc(error)}</p>" if error else ""
    return page("Enter Bill", f"""{err}
<form method='post' action='/bills'>
<label>Vendor <select name='vendor'><option value=''>-- select --</option>{opts}</select></label><br>
<label>Invoice number <input name='invoice_no' value='{esc(v.get('invoice_no', ''))}'></label><br>
<label>Amount (INR) <input name='amount' value='{esc(v.get('amount', ''))}'></label><br>
<label>Due date (YYYY-MM-DD) <input name='due_date' value='{esc(v.get('due_date', ''))}'></label><br>
<label>Notes <input name='notes' value='{esc(v.get('notes', ''))}'></label><br>
<button type='submit'>{L['save']}</button></form>""", "<a href='/bills'>All bills</a>")


@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse("/bills")


@app.get("/bills", response_class=HTMLResponse)
def bills_list():
    L = labels()
    trs = "".join(f"<tr><td>{esc(b['invoice_no'])}</td><td>{esc(b['vendor'])}</td><td>{b['amount']:,.2f}</td>"
                  f"<td>{b['due_date']}</td><td>{esc(b['status'])}</td></tr>" for b in BILLS)
    return page("Bills", f"<a href='/bills/new'>{L['new']}</a>"
                f"<table border='1'><tr><th>Invoice</th><th>Vendor</th><th>Amount</th><th>Due</th><th>Status</th></tr>{trs}</table>")


@app.get("/bills/new", response_class=HTMLResponse)
def new_bill():
    return _form()


@app.post("/bills", response_class=HTMLResponse)
def create_bill(vendor: str = Form(""), invoice_no: str = Form(""), amount: str = Form(""),
                due_date: str = Form(""), notes: str = Form("")):
    values = dict(vendor=vendor, invoice_no=invoice_no, amount=amount, due_date=due_date, notes=notes)
    with _lock:
        _state["posts"] += 1
        if os.environ.get("FLAKY") == "1" and _state["posts"] == 1:
            return HTMLResponse(page("Server Error", "<p>Internal server error, please try again.</p>"),
                                status_code=500)
        missing = [k for k in ("vendor", "invoice_no", "amount", "due_date") if not values[k].strip()]
        if missing:
            return HTMLResponse(_form(f"Required fields missing: {', '.join(missing)}", values), status_code=422)
        if vendor not in VENDORS:
            return HTMLResponse(_form("Unknown vendor.", values), status_code=422)
        try:
            amt = float(amount.replace(",", ""))
            assert amt > 0
        except (ValueError, AssertionError):
            return HTMLResponse(_form("Amount must be a positive number.", values), status_code=422)
        try:
            due = date.fromisoformat(due_date.strip())
        except ValueError:
            return HTMLResponse(_form("Due date must be in YYYY-MM-DD format.", values), status_code=422)
        if any(b["invoice_no"] == invoice_no.strip() for b in BILLS):
            return HTMLResponse(_form(f"Duplicate: invoice {invoice_no} already exists.", values), status_code=409)
        BILLS.append({"id": len(BILLS) + 1, "vendor": vendor, "invoice_no": invoice_no.strip(),
                      "amount": amt, "due_date": due.isoformat(), "notes": notes, "status": "Pending approval"})
    return RedirectResponse("/bills", status_code=303)


@app.get("/api/bills")
def api_bills():
    return BILLS


@app.get("/api/bills/{invoice_no}")
def api_bill(invoice_no: str):
    b = next((b for b in BILLS if b["invoice_no"] == invoice_no), None)
    return b if b else JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/reset")
def api_reset():
    reset()
    return {"ok": True}
