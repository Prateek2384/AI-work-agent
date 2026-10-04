"""Vendor portal: login, searchable/paginated invoice list, PDF downloads."""
from __future__ import annotations

import os

from fastapi import Cookie, FastAPI, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from env.data import INVOICES
from env.pdf import invoice_pdf
from env.ui import esc, labels, page

app = FastAPI(title="Vendor Portal")
PAGE_SIZE = 5
SESSION = "portal-session-ok"


def _creds():
    return os.environ.get("PORTAL_USER", "ap-team"), os.environ.get("PORTAL_PASS", "portal123")


def _authed(session) -> bool:
    return session == SESSION


@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse("/invoices")


@app.get("/login", response_class=HTMLResponse)
def login_form(error: str = ""):
    L = labels()
    err = f"<p role='alert'>{esc(error)}</p>" if error else ""
    return page("Vendor Portal Login", f"""{err}
<form method='post' action='/login'>
<label>Username <input name='username' type='text'></label><br>
<label>Password <input name='password' type='password'></label><br>
<button type='submit'>{L['login']}</button></form>""")


@app.post("/login")
def login(username: str = Form(""), password: str = Form("")):
    if (username, password) != _creds():
        return RedirectResponse("/login?error=Invalid+credentials", status_code=303)
    resp = RedirectResponse("/invoices", status_code=303)
    resp.set_cookie("session", SESSION)
    return resp


@app.get("/invoices", response_class=HTMLResponse)
def invoices(q: str = "", page_no: int = 1, session: str | None = Cookie(None)):
    if not _authed(session):
        return RedirectResponse("/login")
    L = labels()
    rows = [i for i in INVOICES if q.lower() in i.vendor.lower()]  # substring match: "acme" hits 2 vendors
    pages = max(1, -(-len(rows) // PAGE_SIZE))
    page_no = min(max(page_no, 1), pages)
    chunk = rows[(page_no - 1) * PAGE_SIZE: page_no * PAGE_SIZE]
    trs = "".join(
        f"<tr><td>{esc(i.invoice_no)}</td><td>{esc(i.vendor)}</td>"
        f"<td>{i.issued.isoformat()}</td>"
        f"<td><a href='/invoices/{i.id}/pdf'>{L['download']}</a></td></tr>" for i in chunk)
    nxt = (f"<a href='/invoices?q={esc(q)}&page_no={page_no + 1}'>{L['next']}</a>"
           if page_no < pages else "")
    body = f"""<form method='get' action='/invoices'>
<label>Vendor <input name='q' value='{esc(q)}'></label><button type='submit'>{L['search']}</button></form>
<table border='1'><tr><th>Invoice</th><th>Vendor</th><th>Issued</th><th>Document</th></tr>{trs}</table>
<p>Page {page_no} of {pages} ({len(rows)} invoices)</p>{nxt}"""
    return page("Invoices", body)


@app.get("/invoices/{inv_id}/pdf")
def pdf(inv_id: int, session: str | None = Cookie(None)):
    if not _authed(session):
        return Response("unauthorized", status_code=401)
    inv = next((i for i in INVOICES if i.id == inv_id), None)
    if inv is None:
        return Response("not found", status_code=404)
    return Response(invoice_pdf(inv), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={inv.invoice_no}.pdf"})
