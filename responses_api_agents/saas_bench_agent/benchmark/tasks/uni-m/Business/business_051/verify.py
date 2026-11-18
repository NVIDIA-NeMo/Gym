#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/tasks/uni-m/Business/business_051/verify.py
# Original license: CC-BY-4.0; Copyright: SaaS-Bench Authors; see bundled licenses.
# NVIDIA modifications: Use deployed journal schema; require postings in one published journal; report database faults as verifier errors.
"""
Verifier for Business-051-I2: Fundraising Gala Setup with Sponsorship, Accounting, and CRM

Checks: 18 weighted checks across pretix, bigcapital, twenty.
Strategy: docker exec DB queries for all three sites.

Required env vars:
  SERVER_HOSTNAME, PRETIX_PORT, PRETIX_CONTAINER, PRETIX_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER
"""

from contextlib import redirect_stderr
import io
import json
import os
import sys
import subprocess

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

PRETIX_PORT = os.getenv("PRETIX_PORT")
PRETIX_CONTAINER = os.getenv("PRETIX_CONTAINER")
PRETIX_DB = os.getenv("PRETIX_DB_CONTAINER")

BC_PORT = os.getenv("BIGCAPITAL_PORT")
BC_CONTAINER = os.getenv("BIGCAPITAL_CONTAINER")
BC_DB = os.getenv("BIGCAPITAL_DB_CONTAINER")

TWENTY_PORT = os.getenv("TWENTY_PORT")
TWENTY_CONTAINER = os.getenv("TWENTY_CONTAINER")
TWENTY_DB = os.getenv("TWENTY_DB_CONTAINER")

for _var in [
    "PRETIX_PORT", "PRETIX_CONTAINER", "PRETIX_DB_CONTAINER",
    "BIGCAPITAL_PORT", "BIGCAPITAL_CONTAINER", "BIGCAPITAL_DB_CONTAINER",
    "TWENTY_PORT", "TWENTY_CONTAINER", "TWENTY_DB_CONTAINER",
]:
    if not os.getenv(_var):
        print(f"FATAL: {_var} not set", file=sys.stderr)
        sys.exit(1)


# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []
_verifier_errors: list[str] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Helpers ───────────────────────────────────────────────────────────────────
def database_error(message: str) -> RuntimeError:
    """Keep database/execution errors distinct from an unmet task requirement."""
    _verifier_errors.append(message)
    return RuntimeError(message)


def docker_exec(container: str, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            ["docker", "exec", container, *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise database_error(f"Database execution in {container}: {exc}") from exc
    if r.returncode != 0:
        raise database_error(
            f"Database query in {container} exited {r.returncode}: {r.stderr.strip()}"
        )
    return r.returncode, r.stdout, r.stderr


def pretix_q(sql: str) -> str:
    """Run SQL on Pretix Postgres DB."""
    _, out, _ = docker_exec(
        PRETIX_DB, "psql", "-U", "pretix", "-d", "pretix", "-t", "-A", "-c", sql,
    )
    return out.strip()


_bc_tenant_db: str = ""


def _discover_bc_tenant_db() -> None:
    global _bc_tenant_db
    rc, out, err = docker_exec(
        BC_DB, "mysql",
        "--default-character-set=utf8mb4",
        "-u", "bigcapital", "-pbigcapital123",
        "-N", "-B", "-e",
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
        "WHERE SCHEMA_NAME LIKE 'bigcapital_tenant_%' ORDER BY SCHEMA_NAME LIMIT 1;",
    )
    if rc == 0 and out.strip():
        _bc_tenant_db = out.strip().split("\n")[0]
    else:
        raise database_error(f"BigCapital tenant discovery failed: {err.strip() or 'no tenant database'}")


def bc_q(sql: str) -> str:
    """Run SQL on BigCapital MariaDB (tenant DB, auto-discovered)."""
    if not _bc_tenant_db:
        _discover_bc_tenant_db()
    rc, out, err = docker_exec(
        BC_DB, "mysql",
        "--default-character-set=utf8mb4",
        "-u", "bigcapital", "-pbigcapital123",
        "-D", _bc_tenant_db,
        "-N", "-B", "-e", sql,
    )
    if rc != 0:
        raise database_error(f"bigcapital mysql error: {err.strip()}")
    return out.strip()


def twenty_q(sql: str) -> str:
    """Run SQL on Twenty Postgres DB."""
    _, out, _ = docker_exec(
        TWENTY_DB, "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c", sql,
    )
    return out.strip()


_ws_schema: str | None = None


def ws() -> str:
    """Get (cached) Twenty workspace schema name."""
    global _ws_schema
    if _ws_schema is None:
        r = twenty_q(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'workspace_%' ORDER BY schema_name LIMIT 1;"
        )
        if not r:
            raise database_error("No workspace schema found in Twenty DB")
        _ws_schema = r.split("\n")[0].strip()
    return _ws_schema


# ── Pretix checks (1-7) ──────────────────────────────────────────────────────

def check_1_pretix_event() -> None:
    """Event exists with correct name, slug, date, currency, and is live."""
    try:
        r = pretix_q(
            "SELECT slug, date_from, currency, live, name::text "
            "FROM pretixbase_event WHERE slug = 'stars-stripes-gala-2025';"
        )
        if not r:
            check("1. Pretix event + live", 2, False, "event not found")
            return
        p = r.split("|")
        slug_ok = p[0] == "stars-stripes-gala-2025"
        date_ok = p[1].startswith("2025-12-06")
        curr_ok = p[2] == "USD"
        live_ok = p[3].lower() in ("t", "true", "1")
        name_text = "|".join(p[4:]) if len(p) > 4 else r
        name_ok = "Stars & Stripes Charity Gala 2025" in name_text
        ok = slug_ok and date_ok and curr_ok and live_ok and name_ok
        check("1. Pretix event + live", 2, ok,
              f"slug={p[0]}, date={p[1]}, curr={p[2]}, live={p[3]}, name_ok={name_ok}")
    except Exception as e:
        check("1. Pretix event + live", 2, False, f"exception: {e}")


def check_2_pretix_categories() -> None:
    """Three categories: Platinum Benefactors, Gold Benefactors, Silver Supporters Circle."""
    try:
        r = pretix_q(
            "SELECT c.name::text FROM pretixbase_itemcategory c "
            "JOIN pretixbase_event e ON c.event_id = e.id "
            "WHERE e.slug = 'stars-stripes-gala-2025';"
        )
        expected = ["Platinum Benefactors", "Gold Benefactors", "Silver Supporters Circle"]
        found = [c for c in expected if c in r]
        check("2. Pretix categories", 1, len(found) == 3,
              f"found {len(found)}/3: {found}")
    except Exception as e:
        check("2. Pretix categories", 1, False, f"exception: {e}")


def check_3_pretix_products() -> None:
    """Three products with correct prices and category assignments."""
    try:
        r = pretix_q(
            "SELECT i.name::text, i.default_price, c.name::text "
            "FROM pretixbase_item i "
            "JOIN pretixbase_event e ON i.event_id = e.id "
            "LEFT JOIN pretixbase_itemcategory c ON i.category_id = c.id "
            "WHERE e.slug = 'stars-stripes-gala-2025';"
        )
        lines = [line for line in r.split("\n") if line.strip()]
        expected = [
            ("Platinum Gala Table", 15000.0, "Platinum Benefactors"),
            ("Gold Gala Table", 7500.0, "Gold Benefactors"),
            ("Silver Gala Seat", 750.0, "Silver Supporters Circle"),
        ]
        issues = []
        for name, price, cat in expected:
            matched = [line for line in lines if name in line]
            if not matched:
                issues.append(f"{name}: not found")
                continue
            parts = matched[0].split("|")
            if len(parts) >= 2:
                try:
                    if abs(float(parts[1]) - price) > 0.01:
                        issues.append(f"{name}: price={parts[1]}, expected={price}")
                except ValueError:
                    issues.append(f"{name}: price parse error ({parts[1]})")
            if len(parts) >= 3 and cat not in parts[2]:
                issues.append(f"{name}: wrong category ({parts[2]})")
        ok = not issues
        check("3. Pretix products", 2, ok,
              "all 3 correct" if ok else str(issues))
    except Exception as e:
        check("3. Pretix products", 2, False, f"exception: {e}")


def check_4_pretix_quotas() -> None:
    """Three quotas with correct sizes."""
    try:
        r = pretix_q(
            "SELECT q.name, q.size FROM pretixbase_quota q "
            "JOIN pretixbase_event e ON q.event_id = e.id "
            "WHERE e.slug = 'stars-stripes-gala-2025';"
        )
        expected = {
            "Platinum Benefactors Quota": 4,
            "Gold Benefactors Quota": 8,
            "Silver Supporters Quota": 60,
        }
        found = {}
        for line in r.split("\n"):
            if "|" in line:
                parts = line.split("|")
                try:
                    found[parts[0].strip()] = int(parts[1].strip())
                except (ValueError, IndexError):
                    pass
        issues = []
        for name, size in expected.items():
            if name not in found:
                issues.append(f"{name}: not found")
            elif found[name] != size:
                issues.append(f"{name}: got {found[name]}, expected {size}")
        check("4. Pretix quotas", 2, not issues,
              "all 3 correct" if not issues else str(issues))
    except Exception as e:
        check("4. Pretix quotas", 2, False, f"exception: {e}")


def check_5_pretix_question() -> None:
    """Required 'Company Name' text question exists."""
    try:
        r = pretix_q(
            "SELECT q.question::text, q.type, q.required "
            "FROM pretixbase_question q "
            "JOIN pretixbase_event e ON q.event_id = e.id "
            "WHERE e.slug = 'stars-stripes-gala-2025';"
        )
        found = False
        for line in r.split("\n"):
            if "Company Name" in line and "|" in line:
                parts = line.split("|")
                qtype = parts[1].strip() if len(parts) > 1 else ""
                req = parts[2].strip() if len(parts) > 2 else ""
                # Pretix type 'S' = String (one line), 'T' = Text (multi-line)
                if qtype in ("S", "T") and req.lower() in ("t", "true", "1"):
                    found = True
        check("5. Pretix Company Name question", 1, found,
              "found required text question" if found else "not found or misconfigured")
    except Exception as e:
        check("5. Pretix Company Name question", 1, False, f"exception: {e}")


def check_6_pretix_voucher() -> None:
    """Voucher GALASPONSOR2025: 20% discount, max 15 usages, valid until 2025-12-06."""
    try:
        r = pretix_q(
            "SELECT v.code, v.price_mode, v.value, v.max_usages, v.valid_until "
            "FROM pretixbase_voucher v "
            "JOIN pretixbase_event e ON v.event_id = e.id "
            "WHERE e.slug = 'stars-stripes-gala-2025' AND v.code = 'GALASPONSOR2025';"
        )
        if not r:
            check("6. Pretix voucher GALASPONSOR2025", 2, False, "not found")
            return
        p = r.split("|")
        mode_ok = p[1].strip() == "percent"
        val_ok = abs(float(p[2].strip()) - 20.0) < 0.01
        max_ok = int(p[3].strip()) == 15
        valid_ok = "2025-12-06" in (p[4].strip() if len(p) > 4 else "")
        ok = mode_ok and val_ok and max_ok and valid_ok
        check("6. Pretix voucher GALASPONSOR2025", 2, ok,
              f"mode={p[1].strip()}, value={p[2].strip()}, max={p[3].strip()}, "
              f"valid_until={p[4].strip() if len(p) > 4 else 'N/A'}")
    except Exception as e:
        check("6. Pretix voucher GALASPONSOR2025", 2, False, f"exception: {e}")


def check_7_pretix_checkin() -> None:
    """Check-in list 'Stars & Stripes Gala Check-In List' exists."""
    try:
        r = pretix_q(
            "SELECT cl.name FROM pretixbase_checkinlist cl "
            "JOIN pretixbase_event e ON cl.event_id = e.id "
            "WHERE e.slug = 'stars-stripes-gala-2025';"
        )
        found = "Stars & Stripes Gala Check-In List" in r
        check("7. Pretix check-in list", 1, found,
              "found" if found else f"not found, got: {r[:200]}")
    except Exception as e:
        check("7. Pretix check-in list", 1, False, f"exception: {e}")


# ── BigCapital checks (8-13) ─────────────────────────────────────────────────

def check_8_bc_accounts() -> None:
    """Income account 'Stars & Stripes Gala Revenue' and liability account 'Restricted Gala Sponsorship Fund'."""
    try:
        r = bc_q(
            "SELECT NAME, ACCOUNT_TYPE FROM ACCOUNTS "
            "WHERE NAME IN ('Stars & Stripes Gala Revenue', "
            "'Restricted Gala Sponsorship Fund');"
        )
        rev = "Stars & Stripes Gala Revenue" in r
        fund = "Restricted Gala Sponsorship Fund" in r
        check("8. BigCapital accounts", 1, rev and fund,
              f"revenue={rev}, fund={fund}")
    except Exception as e:
        check("8. BigCapital accounts", 1, False, f"exception: {e}")


def check_9_bc_customers() -> None:
    """Customers 'Pinnacle Ventures Corp' and 'Horizon Media Group' with emails."""
    try:
        r = bc_q(
            "SELECT DISPLAY_NAME, EMAIL FROM CONTACTS "
            "WHERE DISPLAY_NAME IN ('Pinnacle Ventures Corp', 'Horizon Media Group') "
            "AND CONTACT_SERVICE = 'customer';"
        )
        pin = "Pinnacle Ventures Corp" in r
        hor = "Horizon Media Group" in r
        check("9. BigCapital customers", 1, pin and hor,
              f"pinnacle={pin}, horizon={hor}")
    except Exception as e:
        check("9. BigCapital customers", 1, False, f"exception: {e}")


def check_10_bc_items() -> None:
    """Service items with correct sell prices."""
    try:
        r = bc_q(
            "SELECT NAME, SELL_PRICE, TYPE FROM ITEMS "
            "WHERE NAME IN ('Platinum Gala Sponsorship Service', "
            "'Gold Gala Sponsorship Service');"
        )
        plat = "Platinum Gala Sponsorship Service" in r
        gold = "Gold Gala Sponsorship Service" in r
        # Rough price check
        price_ok = "15000" in r and "7500" in r
        check("10. BigCapital service items", 1, plat and gold and price_ok,
              f"platinum={plat}, gold={gold}, prices_ok={price_ok}")
    except Exception as e:
        check("10. BigCapital service items", 1, False, f"exception: {e}")


def check_11_bc_invoices() -> None:
    """Two delivered invoices for Pinnacle and Horizon with correct amounts."""
    try:
        r = bc_q(
            "SELECT c.DISPLAY_NAME, si.INVOICE_DATE, si.DUE_DATE, si.BALANCE, "
            "si.DELIVERED_AT "
            "FROM SALES_INVOICES si "
            "JOIN CONTACTS c ON si.CUSTOMER_ID = c.ID "
            "WHERE c.DISPLAY_NAME IN ('Pinnacle Ventures Corp', 'Horizon Media Group') "
            "ORDER BY c.DISPLAY_NAME;"
        )
        if not r:
            check("11. BigCapital invoices", 2, False, "no invoices found")
            return
        pin_found = False
        hor_found = False
        for line in r.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            name = parts[0].strip() if parts else ""
            delivered = parts[4].strip() if len(parts) > 4 else ""
            is_delivered = delivered not in ("", "\\N", "NULL", "None")
            if "Pinnacle" in name:
                pin_found = is_delivered
            if "Horizon" in name:
                hor_found = is_delivered
        ok = pin_found and hor_found
        check("11. BigCapital invoices", 2, ok,
              f"pinnacle_delivered={pin_found}, horizon_delivered={hor_found}")
    except Exception as e:
        check("11. BigCapital invoices", 2, False, f"exception: {e}")


def check_12_bc_payment() -> None:
    """Payment of 15000.00 recorded for Pinnacle Ventures Corp."""
    try:
        r = bc_q(
            "SELECT pr.AMOUNT, pr.PAYMENT_DATE, c.DISPLAY_NAME "
            "FROM PAYMENT_RECEIVES pr "
            "JOIN CONTACTS c ON pr.CUSTOMER_ID = c.ID "
            "WHERE c.DISPLAY_NAME = 'Pinnacle Ventures Corp' "
            "ORDER BY pr.ID DESC LIMIT 1;"
        )
        if not r:
            check("12. BigCapital payment", 2, False, "no payment found")
            return
        p = r.split("\t")
        try:
            amt_ok = abs(float(p[0].strip()) - 15000.0) < 0.01
        except ValueError:
            amt_ok = False
        date_ok = "2025-10-22" in r
        check("12. BigCapital payment", 2, amt_ok and date_ok,
              f"amount={p[0].strip()}, date_ok={date_ok}")
    except Exception as e:
        check("12. BigCapital payment", 2, False, f"exception: {e}")


def check_13_bc_journal() -> None:
    """Published journal entry: debit Revenue 15000, credit Restricted Fund 15000."""
    try:
        r = bc_q(
            "SELECT mj.ID, mj.DESCRIPTION, (mj.PUBLISHED_AT IS NOT NULL), "
            "mje.DEBIT, mje.CREDIT, a.NAME "
            "FROM MANUAL_JOURNALS mj "
            "JOIN MANUAL_JOURNALS_ENTRIES mje ON mje.MANUAL_JOURNAL_ID = mj.ID "
            "JOIN ACCOUNTS a ON mje.ACCOUNT_ID = a.ID "
            "WHERE mj.DESCRIPTION LIKE '%Reclassify%' "
            "OR mj.DESCRIPTION LIKE '%Stars & Stripes%' "
            "ORDER BY mj.ID;"
        )
        if not r:
            check("13. BigCapital journal entry", 3, False, "journal not found")
            return
        journals = {}
        for line in r.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            # All required postings must belong to the same published journal.
            journal = journals.setdefault(parts[0], dict(debit_revenue=False,
                                                        credit_fund=False,
                                                        published=parts[2].strip() == "1"))
            try:
                debit = float(parts[3].strip() or "0")
                credit = float(parts[4].strip() or "0")
                acct = parts[5].strip()
                if "Gala Revenue" in acct and debit >= 14999.99:
                    journal["debit_revenue"] = True
                if "Restricted" in acct and "Fund" in acct and credit >= 14999.99:
                    journal["credit_fund"] = True
            except ValueError:
                pass
        ok = any(all(journal.values()) for journal in journals.values())
        check("13. BigCapital journal entry", 3, ok,
              f"matching_journals={len(journals)}, published_with_required_postings={ok}")
    except Exception as e:
        check("13. BigCapital journal entry", 3, False, f"exception: {e}")


# ── Twenty CRM checks (14-18) ────────────────────────────────────────────────

def check_14_twenty_companies() -> None:
    """Companies 'Pinnacle Ventures Corp' and 'Horizon Media Group' with correct domains."""
    try:
        s = ws()
        r = twenty_q(
            f"SELECT * FROM {s}.company "
            f"WHERE name IN ('Pinnacle Ventures Corp', 'Horizon Media Group');"
        )
        pin = "Pinnacle Ventures Corp" in r and "pinnacleventures.com" in r
        hor = "Horizon Media Group" in r and "horizonmediagroup.com" in r
        check("14. Twenty companies", 1, pin and hor,
              f"pinnacle(+domain)={pin}, horizon(+domain)={hor}")
    except Exception as e:
        check("14. Twenty companies", 1, False, f"exception: {e}")


def check_15_twenty_people() -> None:
    """Margaret Holloway and Thomas Beaumont with correct titles, emails, company links."""
    try:
        s = ws()
        # Get company IDs
        pin_id = twenty_q(
            f"SELECT id FROM {s}.company WHERE name = 'Pinnacle Ventures Corp';"
        ).split("\n")[0].strip()
        hor_id = twenty_q(
            f"SELECT id FROM {s}.company WHERE name = 'Horizon Media Group';"
        ).split("\n")[0].strip()

        # Get people — use SELECT * to handle column name uncertainty
        r = twenty_q(f"SELECT * FROM {s}.person;")

        margaret_ok = (
            "Margaret" in r and "Holloway" in r
            and "contact@pinnacleventures.com" in r
            and "Chief Executive Officer" in r
            and (pin_id in r if pin_id else False)
        )
        thomas_ok = (
            "Thomas" in r and "Beaumont" in r
            and "info@horizonmediagroup.com" in r
            and "Director of Corporate Partnerships" in r
            and (hor_id in r if hor_id else False)
        )
        check("15. Twenty people", 2, margaret_ok and thomas_ok,
              f"margaret={margaret_ok}, thomas={thomas_ok}")
    except Exception as e:
        check("15. Twenty people", 2, False, f"exception: {e}")


def check_16_twenty_opportunities() -> None:
    """Won opportunity for Pinnacle and Qualification opportunity for Horizon."""
    try:
        s = ws()
        r1 = twenty_q(
            f"SELECT * FROM {s}.opportunity "
            f"WHERE name LIKE '%Platinum Sponsorship%';"
        )
        r2 = twenty_q(
            f"SELECT * FROM {s}.opportunity "
            f"WHERE name LIKE '%Gold Sponsorship%';"
        )
        plat_ok = bool(r1) and ("WON" in r1.upper())
        gold_ok = bool(r2) and ("QUALIFICATION" in r2.upper())
        check("16. Twenty opportunities", 2, plat_ok and gold_ok,
              f"platinum_won={plat_ok}, gold_qualification={gold_ok}")
    except Exception as e:
        check("16. Twenty opportunities", 2, False, f"exception: {e}")


def check_17_twenty_task() -> None:
    """Task 'Collect sponsorship payment — Horizon Media Group' with correct body."""
    try:
        s = ws()
        r = twenty_q(
            f"SELECT * FROM {s}.task "
            f"WHERE title LIKE '%Collect sponsorship payment%';"
        )
        if not r:
            check("17. Twenty collection task", 2, False, "task not found")
            return
        has_amt = "15000" in r
        has_contact = "Thomas Beaumont" in r or "info@horizonmediagroup.com" in r
        has_voucher = "GALASPONSOR2025" in r
        ok = has_amt and has_contact and has_voucher
        check("17. Twenty collection task", 2, ok,
              f"amount={has_amt}, contact={has_contact}, voucher={has_voucher}")
    except Exception as e:
        check("17. Twenty collection task", 2, False, f"exception: {e}")


def check_18_twenty_note() -> None:
    """Note 'Stars & Stripes Charity Gala 2025 — Sponsorship Tracker' with full content."""
    try:
        s = ws()
        r = twenty_q(
            f"SELECT * FROM {s}.note "
            f"WHERE title LIKE '%Sponsorship Tracker%';"
        )
        if not r:
            check("18. Twenty sponsorship note", 2, False, "note not found")
            return
        has_pin = "Pinnacle Ventures Corp" in r
        has_hor = "Horizon Media Group" in r
        has_paid = "PAID" in r
        has_pend = "PENDING" in r
        has_total = "30000" in r
        has_outstanding = "15000" in r
        ok = has_pin and has_hor and has_paid and has_pend and has_total and has_outstanding
        check("18. Twenty sponsorship note", 2, ok,
              f"sponsors={has_pin and has_hor}, paid={has_paid}, pending={has_pend}, "
              f"totals={has_total and has_outstanding}")
    except Exception as e:
        check("18. Twenty sponsorship note", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Buffer ordinary check records until database execution is known to be valid.
    # Older, already-running harness processes also recognize an error with no
    # scored check lines, so they cannot mistake a database failure for a FAIL.
    diagnostics = io.StringIO()
    with redirect_stderr(diagnostics):
        for verify_check in [
            check_1_pretix_event, check_2_pretix_categories, check_3_pretix_products,
            check_4_pretix_quotas, check_5_pretix_question, check_6_pretix_voucher,
            check_7_pretix_checkin, check_8_bc_accounts, check_9_bc_customers,
            check_10_bc_items, check_11_bc_invoices, check_12_bc_payment,
            check_13_bc_journal, check_14_twenty_companies, check_15_twenty_people,
            check_16_twenty_opportunities, check_17_twenty_task, check_18_twenty_note,
        ]:
            verify_check()

    if _verifier_errors:
        print(diagnostics.getvalue(), end="", file=sys.stdout)
        print("VERIFIER_ERROR: " + json.dumps({"kind": "database", "messages": _verifier_errors}),
              file=sys.stderr)
        sys.exit(2)
    print(diagnostics.getvalue(), end="", file=sys.stderr)

    total = sum(w for _, w, _, _ in _checks)
    earned = sum(w for _, w, p, _ in _checks if p)
    all_pass = all(p for _, _, p, _ in _checks) and bool(_checks)
    score = (earned / total) if total else 0.0

    print(
        f"SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})",
        file=sys.stderr,
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
