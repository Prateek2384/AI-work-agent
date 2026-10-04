"""Deterministic seed data, including the deliberate traps.

Traps for the headline task ("latest invoice from Acme Supplies"):
- a near-duplicate vendor "Acme Supply Co" whose invoice is *newer* than Acme Supplies' latest
- invoices are listed unsorted and paginated; the latest one is not on page 1
- amount/due date exist only inside the PDF, with a different date format per invoice
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Invoice:
    id: int
    invoice_no: str
    vendor: str
    issued: date
    due: date
    amount: float
    date_style: str  # how dates are printed inside the PDF


def _inv(i, no, vendor, issued, due, amount, style):
    return Invoice(i, no, vendor, date.fromisoformat(issued), date.fromisoformat(due), amount, style)


# Deliberately NOT in date order (this is the order the portal lists them).
INVOICES: list[Invoice] = [
    _inv(1, "INV-2031", "Acme Supplies", "2026-03-04", "2026-04-03", 12480.00, "dmy_slash"),
    _inv(2, "INV-5510", "Globex Corp", "2026-08-11", "2026-09-10", 78200.50, "long"),
    _inv(3, "INV-2044", "Acme Supplies", "2026-08-02", "2026-09-01", 31900.75, "long"),
    _inv(4, "INV-2036", "Acme Supplies", "2026-05-19", "2026-06-18", 22150.00, "dmy_slash"),
    _inv(5, "INV-3001", "Acme Supply Co", "2026-09-28", "2026-10-28", 9999.00, "long"),  # decoy, newest overall
    _inv(6, "INV-2029", "Acme Supplies", "2026-01-22", "2026-02-21", 8800.00, "long"),
    _inv(7, "INV-5498", "Globex Corp", "2026-06-30", "2026-07-30", 4120.00, "dmy_slash"),
    _inv(8, "INV-2040", "Acme Supplies", "2026-07-07", "2026-08-06", 17640.40, "dmy_slash"),
    _inv(9, "INV-2047", "Acme Supplies", "2026-09-18", "2026-10-18", 48250.00, "long"),  # TARGET (latest)
    _inv(10, "INV-2033", "Acme Supplies", "2026-04-11", "2026-05-11", 15300.00, "long"),
    _inv(11, "INV-3000", "Acme Supply Co", "2026-02-14", "2026-03-16", 2500.00, "dmy_slash"),
    _inv(12, "INV-2038", "Acme Supplies", "2026-06-15", "2026-07-15", 26775.25, "long"),
    _inv(13, "INV-5521", "Globex Corp", "2026-09-05", "2026-10-05", 91000.00, "long"),
    _inv(14, "INV-2042", "Acme Supplies", "2026-07-29", "2026-08-28", 19990.00, "dmy_slash"),
]

VENDORS = ["Acme Supplies", "Acme Supply Co", "Globex Corp"]


def latest_for(vendor: str) -> Invoice:
    return max((i for i in INVOICES if i.vendor == vendor), key=lambda i: i.issued)


# Ground truth for evals/verification of the headline task.
EXPECTED = latest_for("Acme Supplies")


def fmt_date(d: date, style: str) -> str:
    return d.strftime("%d/%m/%Y") if style == "dmy_slash" else d.strftime("%d %B %Y")
