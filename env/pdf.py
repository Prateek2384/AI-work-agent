"""Render an invoice PDF in-memory (text-based, so pypdf can extract it)."""
from __future__ import annotations

import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from env.data import Invoice, fmt_date


def invoice_pdf(inv: Invoice) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line, size in [
        (inv.vendor, 18),
        ("TAX INVOICE", 14),
        (f"Invoice No: {inv.invoice_no}", 11),
        (f"Date of Issue: {fmt_date(inv.issued, inv.date_style)}", 11),
        ("Bill To: Hawkvision Pvt Ltd", 11),
        ("", 11),
        ("Description                          Qty        Amount", 10),
        (f"Supplies and services                  1        {inv.amount * 0.9:,.2f}", 10),
        (f"GST (11.11%)                                       {inv.amount * 0.1:,.2f}", 10),
        ("", 11),
        (f"Payment due by: {fmt_date(inv.due, inv.date_style)}", 11),
        (f"Total Amount Payable: INR {inv.amount:,.2f}", 13),
    ]:
        c.setFont("Helvetica-Bold" if size > 12 else "Helvetica", size)
        c.drawString(60, y, line)
        y -= size + 10
    c.save()
    return buf.getvalue()
