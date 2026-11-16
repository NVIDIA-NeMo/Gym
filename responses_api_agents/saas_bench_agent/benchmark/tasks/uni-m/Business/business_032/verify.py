#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/tasks/uni-m/Business/business_032/verify.py
# Original license: CC-BY-4.0; Copyright: SaaS-Bench Authors; see bundled licenses.
# NVIDIA modifications: Normalize NULL debit/credit database values before numeric comparisons.
"""
Verifier for Business-032-I4: Milestone-Based Client Engagement — Arcturus Digital

Checks: 18 weighted checks across twenty, bigcapital, pretix.
Strategy: docker exec DB queries — Twenty (psql), BigCapital (mysql), Pretix (psql).

Required env vars:
  SERVER_HOSTNAME,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  PRETIX_PORT, PRETIX_CONTAINER, PRETIX_DB_CONTAINER.
"""

import os
import subprocess
import sys

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")


def _require(var: str) -> str:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    return val


TWENTY_PORT = _require("TWENTY_PORT")
TWENTY_CONTAINER = _require("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = _require("TWENTY_DB_CONTAINER")
BIGCAPITAL_PORT = _require("BIGCAPITAL_PORT")
BIGCAPITAL_CONTAINER = _require("BIGCAPITAL_CONTAINER")
BIGCAPITAL_DB_CONTAINER = _require("BIGCAPITAL_DB_CONTAINER")
PRETIX_PORT = _require("PRETIX_PORT")
PRETIX_CONTAINER = _require("PRETIX_CONTAINER")
PRETIX_DB_CONTAINER = _require("PRETIX_DB_CONTAINER")


# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Helpers ───────────────────────────────────────────────────────────────────
def docker_exec(container: str, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    r = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


def twenty_sql(query: str) -> str:
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default",
        "-t", "-A", "-c", query,
    )
    if rc != 0:
        raise RuntimeError(f"Twenty SQL error: {err.strip()}")
    return out.strip()


# BigCapital: MySQL embedded in app container. Multi-tenant DB.
_bc_tenant_db: str = ""


def _init_bc_tenant_db() -> None:
    global _bc_tenant_db
    if _bc_tenant_db:
        return
    rc, out, err = docker_exec(
        BIGCAPITAL_DB_CONTAINER,
        "mysql", "-u", "root", "--password=",
        "--default-character-set=utf8mb4",
        "--batch", "--skip-column-names",
        "bigcapital", "-e", "SELECT ORGANIZATION_ID FROM TENANTS LIMIT 1;",
    )
    if rc != 0 or not out.strip():
        raise RuntimeError(f"Cannot find BC tenant DB: {err.strip()}")
    _bc_tenant_db = f"bigcapital_tenant_{out.strip()}"


def bigcapital_sql(query: str) -> str:
    _init_bc_tenant_db()
    rc, out, err = docker_exec(
        BIGCAPITAL_DB_CONTAINER,
        "mysql", "-u", "root", "--password=",
        "--default-character-set=utf8mb4",
        "--batch", "--skip-column-names",
        _bc_tenant_db, "-e", query,
    )
    if rc != 0:
        raise RuntimeError(f"BigCapital SQL error: {err.strip()}")
    return out.strip()


def pretix_sql(query: str) -> str:
    rc, out, err = docker_exec(
        PRETIX_DB_CONTAINER,
        "psql", "-U", "pretix", "-d", "pretix",
        "-t", "-A", "-c", query,
    )
    if rc != 0:
        raise RuntimeError(f"Pretix SQL error: {err.strip()}")
    return out.strip()


def get_twenty_workspace_schema() -> str:
    result = twenty_sql(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'workspace_%' LIMIT 1;"
    )
    if not result:
        raise RuntimeError("No workspace schema found in Twenty DB")
    return result.split("\n")[0].strip()


# Cross-check IDs populated by earlier checks
_twenty_company_id: str = ""
_twenty_person_id: str = ""


# ── Twenty checks ────────────────────────────────────────────────────────────

def check_1_twenty_company() -> None:
    """Company 'Arcturus Digital' with domain arcturusdigital.com and 275 employees."""
    global _twenty_company_id
    try:
        ws = get_twenty_workspace_schema()
        row = twenty_sql(
            f'SELECT id, name, "domainNamePrimaryLinkUrl", employees '
            f'FROM "{ws}".company '
            f"WHERE name = 'Arcturus Digital' AND \"deletedAt\" IS NULL "
            f"LIMIT 1;"
        )
        if not row:
            check("1. Twenty company Arcturus Digital", 2, False, "not found")
            return
        parts = row.split("|")
        cid = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        domain = parts[2].strip() if len(parts) > 2 else ""
        employees = parts[3].strip() if len(parts) > 3 else ""
        _twenty_company_id = cid
        ok = (
            name == "Arcturus Digital"
            and "arcturusdigital.com" in domain
            and str(employees) == "275"
        )
        check("1. Twenty company Arcturus Digital", 2, ok,
              f"domain={domain}, employees={employees}")
    except Exception as e:
        check("1. Twenty company Arcturus Digital", 2, False, f"exception: {e}")


def check_2_twenty_person() -> None:
    """Person 'Elena Vasquez' with email, title, phone, linked to Arcturus Digital."""
    global _twenty_person_id
    try:
        ws = get_twenty_workspace_schema()
        row = twenty_sql(
            f'SELECT id, "nameFirstName", "nameLastName", "emailsPrimaryEmail", '
            f'"jobTitle", "phonesPrimaryPhoneNumber", "companyId" '
            f'FROM "{ws}".person '
            f"WHERE \"emailsPrimaryEmail\" = 'elena.vasquez@arcturusdigital.com' "
            f"AND \"deletedAt\" IS NULL "
            f"LIMIT 1;"
        )
        if not row:
            check("2. Twenty person Elena Vasquez", 2, False, "not found")
            return
        parts = row.split("|")
        pid = parts[0].strip()
        first = parts[1].strip() if len(parts) > 1 else ""
        last = parts[2].strip() if len(parts) > 2 else ""
        email = parts[3].strip() if len(parts) > 3 else ""
        title = parts[4].strip() if len(parts) > 4 else ""
        phone = parts[5].strip() if len(parts) > 5 else ""
        company_id = parts[6].strip() if len(parts) > 6 else ""
        _twenty_person_id = pid
        ok = (
            first == "Elena" and last == "Vasquez"
            and email == "elena.vasquez@arcturusdigital.com"
            and "Director of Engineering" in title
            and "503" in phone and "4480" in phone
            and (not _twenty_company_id or company_id == _twenty_company_id)
        )
        check("2. Twenty person Elena Vasquez", 2, ok,
              f"name={first} {last}, title={title}, phone={phone}, "
              f"company_linked={'yes' if company_id == _twenty_company_id else 'no'}")
    except Exception as e:
        check("2. Twenty person Elena Vasquez", 2, False, f"exception: {e}")


def check_3_twenty_opportunity() -> None:
    """Opportunity with amount 100000, stage Won, close date 2026-01-31, correct links."""
    try:
        ws = get_twenty_workspace_schema()
        row = twenty_sql(
            f'SELECT name, "amountAmountMicros", stage, "closeDate", '
            f'"companyId", "pointOfContactId" '
            f'FROM "{ws}".opportunity '
            f"WHERE name LIKE '%Arcturus Digital%Enterprise Software Modernization%' "
            f"AND \"deletedAt\" IS NULL "
            f"LIMIT 1;"
        )
        if not row:
            check("3. Twenty opportunity", 2, False, "not found")
            return
        parts = row.split("|")
        name = parts[0].strip()
        amount_raw = parts[1].strip() if len(parts) > 1 else "0"
        stage = parts[2].strip() if len(parts) > 2 else ""
        close_date = parts[3].strip() if len(parts) > 3 else ""
        company_id = parts[4].strip() if len(parts) > 4 else ""
        poc_id = parts[5].strip() if len(parts) > 5 else ""

        try:
            amount = int(amount_raw) / 1_000_000
        except (ValueError, TypeError):
            amount = float(amount_raw or 0)
        amount_ok = abs(amount - 100000.0) < 1.0
        stage_ok = stage.upper() == "WON"
        date_ok = "2026-01-31" in close_date
        company_ok = not _twenty_company_id or company_id == _twenty_company_id
        poc_ok = not _twenty_person_id or poc_id == _twenty_person_id

        ok = amount_ok and stage_ok and date_ok and company_ok and poc_ok
        check("3. Twenty opportunity", 2, ok,
              f"amount={amount}, stage={stage}, closeDate={close_date}, "
              f"company={'ok' if company_ok else 'wrong'}, poc={'ok' if poc_ok else 'wrong'}")
    except Exception as e:
        check("3. Twenty opportunity", 2, False, f"exception: {e}")


def check_4_twenty_favorite() -> None:
    """Company 'Arcturus Digital' is marked as a favorite."""
    try:
        if not _twenty_company_id:
            check("4. Twenty company is favorite", 1, False, "company ID unknown from check 1")
            return
        ws = get_twenty_workspace_schema()
        row = twenty_sql(
            f'SELECT id FROM "{ws}".favorite '
            f"WHERE \"companyId\" = '{_twenty_company_id}' "
            f"AND \"deletedAt\" IS NULL "
            f"LIMIT 1;"
        )
        ok = bool(row)
        check("4. Twenty company is favorite", 1, ok,
              "found" if ok else "no favorite record")
    except Exception as e:
        check("4. Twenty company is favorite", 1, False, f"exception: {e}")


def check_5_twenty_task_milestone2() -> None:
    """Task 'Milestone 2 collection — Arcturus Digital' with due 2026-02-04 and body keywords."""
    try:
        ws = get_twenty_workspace_schema()
        row = twenty_sql(
            f'SELECT title, "dueAt"::text, "bodyV2Markdown" '
            f'FROM "{ws}".task '
            f"WHERE title LIKE '%Milestone 2 collection%Arcturus Digital%' "
            f"AND \"deletedAt\" IS NULL "
            f"LIMIT 1;"
        )
        if not row:
            check("5. Twenty task Milestone 2 collection", 2, False, "not found")
            return
        parts = row.split("|", 2)
        title = parts[0].strip()
        due_at = parts[1].strip() if len(parts) > 1 else ""
        body = parts[2] if len(parts) > 2 else ""

        date_ok = "2026-02-04" in due_at
        body_ok = (
            "55000" in body
            and "ARCTURUS100VIP" in body
            and "elena" in body.lower()
        )
        ok = date_ok and body_ok
        check("5. Twenty task Milestone 2 collection", 2, ok,
              f"dueAt={'ok' if date_ok else due_at}, "
              f"body_amount={'yes' if '55000' in body else 'no'}, "
              f"body_voucher={'yes' if 'ARCTURUS100VIP' in body else 'no'}")
    except Exception as e:
        check("5. Twenty task Milestone 2 collection", 2, False, f"exception: {e}")


def check_6_twenty_task_celebration() -> None:
    """Task 'Send celebration invite — Arcturus Digital' with due 2026-02-25 and body keywords."""
    try:
        ws = get_twenty_workspace_schema()
        row = twenty_sql(
            f'SELECT title, "dueAt"::text, "bodyV2Markdown" '
            f'FROM "{ws}".task '
            f"WHERE title LIKE '%Send celebration invite%Arcturus Digital%' "
            f"AND \"deletedAt\" IS NULL "
            f"LIMIT 1;"
        )
        if not row:
            check("6. Twenty task Send celebration invite", 2, False, "not found")
            return
        parts = row.split("|", 2)
        title = parts[0].strip()
        due_at = parts[1].strip() if len(parts) > 1 else ""
        body = parts[2] if len(parts) > 2 else ""

        date_ok = "2026-02-25" in due_at
        body_ok = (
            "ARCTURUS100VIP" in body
            and "elena" in body.lower()
            and ("100%" in body or "100 %" in body)
        )
        ok = date_ok and body_ok
        check("6. Twenty task Send celebration invite", 2, ok,
              f"dueAt={'ok' if date_ok else due_at}, "
              f"body_voucher={'yes' if 'ARCTURUS100VIP' in body else 'no'}")
    except Exception as e:
        check("6. Twenty task Send celebration invite", 2, False, f"exception: {e}")


# ── BigCapital checks (MySQL, UPPERCASE tables) ──────────────────────────────

def check_7_bc_customer() -> None:
    """Customer 'Arcturus Digital' with email elena.vasquez@arcturusdigital.com."""
    try:
        row = bigcapital_sql(
            "SELECT DISPLAY_NAME, IFNULL(EMAIL, '') "
            "FROM CONTACTS "
            "WHERE DISPLAY_NAME = 'Arcturus Digital' "
            "AND CONTACT_SERVICE = 'customer' "
            "LIMIT 1;"
        )
        if not row:
            check("7. BigCapital customer Arcturus Digital", 1, False, "not found")
            return
        parts = row.split("\t")
        name = parts[0].strip()
        email = parts[1].strip() if len(parts) > 1 else ""
        ok = name == "Arcturus Digital" and "elena.vasquez@arcturusdigital.com" in email
        check("7. BigCapital customer Arcturus Digital", 1, ok,
              f"name={name}, email={email}")
    except Exception as e:
        check("7. BigCapital customer Arcturus Digital", 1, False, f"exception: {e}")


def check_8_bc_items() -> None:
    """Items 'Phase 1' (45000) and 'Phase 2' (55000) exist as service type."""
    try:
        rows = bigcapital_sql(
            "SELECT NAME, TYPE, SELL_PRICE "
            "FROM ITEMS "
            "WHERE NAME LIKE 'Phase 1%' OR NAME LIKE 'Phase 2%' "
            "ORDER BY NAME;"
        )
        if not rows:
            check("8. BigCapital items Phase 1 & 2", 2, False, "no items found")
            return
        p1_ok = False
        p2_ok = False
        for line in rows.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            price = float(parts[2].strip() or 0)
            if "Requirements" in name and "Solution Design" in name:
                p1_ok = abs(price - 45000.0) < 1.0
            elif "Development" in name and "Go-Live" in name:
                p2_ok = abs(price - 55000.0) < 1.0

        ok = p1_ok and p2_ok
        check("8. BigCapital items Phase 1 & 2", 2, ok,
              f"Phase1={'ok' if p1_ok else 'FAIL'}, Phase2={'ok' if p2_ok else 'FAIL'}")
    except Exception as e:
        check("8. BigCapital items Phase 1 & 2", 2, False, f"exception: {e}")


def check_9_bc_account() -> None:
    """Account 'Deferred Revenue — Enterprise Engagements' as Other Current Liability."""
    try:
        row = bigcapital_sql(
            "SELECT NAME, ACCOUNT_TYPE "
            "FROM ACCOUNTS "
            "WHERE NAME LIKE '%Deferred Revenue%Enterprise Engagements%' "
            "LIMIT 1;"
        )
        if not row:
            check("9. BigCapital deferred revenue account", 1, False, "not found")
            return
        parts = row.split("\t")
        name = parts[0].strip()
        acct_type = parts[1].strip() if len(parts) > 1 else ""
        # BigCapital stores account types with hyphens (e.g. 'other-current-liability')
        normalised = acct_type.lower().replace(" ", "-").replace("_", "-")
        ok = "Deferred Revenue" in name and "other-current-liability" in normalised
        check("9. BigCapital deferred revenue account", 1, ok,
              f"name={name}, type={acct_type}")
    except Exception as e:
        check("9. BigCapital deferred revenue account", 1, False, f"exception: {e}")


def check_10_bc_invoice1() -> None:
    """Milestone 1 invoice dated 2025-10-15 for Arcturus Digital, amount 45000, delivered."""
    try:
        row = bigcapital_sql(
            "SELECT si.ID, si.INVOICE_DATE, si.DELIVERED_AT, si.BALANCE + si.PAYMENT_AMOUNT AS TOTAL "
            "FROM SALES_INVOICES si "
            "JOIN CONTACTS c ON si.CUSTOMER_ID = c.ID "
            "WHERE c.DISPLAY_NAME = 'Arcturus Digital' "
            "AND si.INVOICE_DATE = '2025-10-15' "
            "LIMIT 1;"
        )
        if not row:
            check("10. BigCapital Milestone 1 invoice", 2, False, "not found")
            return
        parts = row.split("\t")
        inv_date = parts[1].strip() if len(parts) > 1 else ""
        delivered = parts[2].strip() if len(parts) > 2 else ""
        amount = float(parts[3].strip() or 0) if len(parts) > 3 else 0.0

        date_ok = "2025-10-15" in inv_date
        delivered_ok = delivered not in ("", "NULL", "None", "null", "0000-00-00")
        amount_ok = abs(amount - 45000.0) < 1.0

        ok = date_ok and delivered_ok and amount_ok
        check("10. BigCapital Milestone 1 invoice", 2, ok,
              f"date={inv_date}, delivered={'yes' if delivered_ok else 'no'}, amount={amount}")
    except Exception as e:
        check("10. BigCapital Milestone 1 invoice", 2, False, f"exception: {e}")


def check_11_bc_invoice2() -> None:
    """Milestone 2 invoice dated 2026-01-05 for Arcturus Digital, amount 55000, delivered."""
    try:
        row = bigcapital_sql(
            "SELECT si.ID, si.INVOICE_DATE, si.DELIVERED_AT, si.BALANCE + si.PAYMENT_AMOUNT AS TOTAL "
            "FROM SALES_INVOICES si "
            "JOIN CONTACTS c ON si.CUSTOMER_ID = c.ID "
            "WHERE c.DISPLAY_NAME = 'Arcturus Digital' "
            "AND si.INVOICE_DATE = '2026-01-05' "
            "LIMIT 1;"
        )
        if not row:
            check("11. BigCapital Milestone 2 invoice", 2, False, "not found")
            return
        parts = row.split("\t")
        inv_date = parts[1].strip() if len(parts) > 1 else ""
        delivered = parts[2].strip() if len(parts) > 2 else ""
        amount = float(parts[3].strip() or 0) if len(parts) > 3 else 0.0

        date_ok = "2026-01-05" in inv_date
        delivered_ok = delivered not in ("", "NULL", "None", "null", "0000-00-00")
        amount_ok = abs(amount - 55000.0) < 1.0

        ok = date_ok and delivered_ok and amount_ok
        check("11. BigCapital Milestone 2 invoice", 2, ok,
              f"date={inv_date}, delivered={'yes' if delivered_ok else 'no'}, amount={amount}")
    except Exception as e:
        check("11. BigCapital Milestone 2 invoice", 2, False, f"exception: {e}")


def check_12_bc_payment() -> None:
    """Payment of 45000 dated 2025-11-20 deposited to Petty Cash."""
    try:
        row = bigcapital_sql(
            "SELECT pr.PAYMENT_DATE, pr.AMOUNT, a.NAME "
            "FROM PAYMENT_RECEIVES pr "
            "JOIN CONTACTS c ON pr.CUSTOMER_ID = c.ID "
            "JOIN ACCOUNTS a ON pr.DEPOSIT_ACCOUNT_ID = a.ID "
            "WHERE c.DISPLAY_NAME = 'Arcturus Digital' "
            "LIMIT 1;"
        )
        if not row:
            check("12. BigCapital payment received", 2, False, "not found")
            return
        parts = row.split("\t")
        pay_date = parts[0].strip()
        amount = float(parts[1].strip() or 0) if len(parts) > 1 else 0.0
        account = parts[2].strip() if len(parts) > 2 else ""

        date_ok = "2025-11-20" in pay_date
        amount_ok = abs(amount - 45000.0) < 1.0
        account_ok = "Petty Cash" in account

        ok = date_ok and amount_ok and account_ok
        check("12. BigCapital payment received", 2, ok,
              f"date={pay_date}, amount={amount}, account={account}")
    except Exception as e:
        check("12. BigCapital payment received", 2, False, f"exception: {e}")


def check_13_bc_journal() -> None:
    """Journal entry dated 2025-11-20: debit Deferred Revenue 45000, credit Uncategorized Income 45000."""
    try:
        rows = bigcapital_sql(
            "SELECT mj.DATE, mj.DESCRIPTION, "
            "IFNULL(mje.DEBIT, 0), IFNULL(mje.CREDIT, 0), a.NAME "
            "FROM MANUAL_JOURNALS mj "
            "JOIN MANUAL_JOURNALS_ENTRIES mje ON mje.MANUAL_JOURNAL_ID = mj.ID "
            "JOIN ACCOUNTS a ON mje.ACCOUNT_ID = a.ID "
            "WHERE mj.DESCRIPTION LIKE '%Revenue recognition%Milestone 1%Arcturus Digital%' "
            "ORDER BY mje.DEBIT DESC;"
        )
        if not rows:
            check("13. BigCapital journal entry", 3, False, "not found")
            return

        date_ok = False
        memo_ok = False
        debit_ok = False
        credit_ok = False

        for line in rows.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            jdate = parts[0].strip()
            desc = parts[1].strip()
            debit_raw = parts[2].strip()
            credit_raw = parts[3].strip()
            debit = float(0 if debit_raw.upper() in {"", "NULL", "NONE"} else debit_raw)
            credit = float(0 if credit_raw.upper() in {"", "NULL", "NONE"} else credit_raw)
            acct = parts[4].strip()

            if "2025-11-20" in jdate:
                date_ok = True
            if "Revenue recognition" in desc and "Milestone 1" in desc:
                memo_ok = True
            if abs(debit - 45000.0) < 1.0 and "Deferred Revenue" in acct:
                debit_ok = True
            if abs(credit - 45000.0) < 1.0 and "Uncategorized Income" in acct:
                credit_ok = True

        ok = date_ok and memo_ok and debit_ok and credit_ok
        check("13. BigCapital journal entry", 3, ok,
              f"date={'ok' if date_ok else 'wrong'}, memo={'ok' if memo_ok else 'wrong'}, "
              f"debit={'ok' if debit_ok else 'wrong'}, credit={'ok' if credit_ok else 'wrong'}")
    except Exception as e:
        check("13. BigCapital journal entry", 3, False, f"exception: {e}")


def check_14_bc_balance() -> None:
    """Customer outstanding balance for Arcturus Digital = 55000."""
    try:
        row = bigcapital_sql(
            "SELECT IFNULL(SUM(si.BALANCE), 0) "
            "FROM SALES_INVOICES si "
            "JOIN CONTACTS c ON si.CUSTOMER_ID = c.ID "
            "WHERE c.DISPLAY_NAME = 'Arcturus Digital';"
        )
        balance = float(row.strip() or 0) if row else 0.0
        ok = abs(balance - 55000.0) < 1.0
        check("14. BigCapital customer balance 55000", 3, ok,
              f"balance={balance}, expected=55000")
    except Exception as e:
        check("14. BigCapital customer balance 55000", 3, False, f"exception: {e}")


# ── Pretix checks ────────────────────────────────────────────────────────────

def check_15_pretix_event() -> None:
    """Event 'Arcturus Digital Milestone Celebration' live, slug, dates, currency."""
    try:
        row = pretix_sql(
            "SELECT e.slug, e.name::text, e.date_from::text, e.currency, e.live "
            "FROM pretixbase_event e "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'culinary-arts' "
            "AND e.slug = 'arcturus-milestone-celebration' "
            "LIMIT 1;"
        )
        if not row:
            check("15. Pretix event", 2, False, "not found")
            return
        parts = row.split("|")
        slug = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        date_from = parts[2].strip() if len(parts) > 2 else ""
        currency = parts[3].strip() if len(parts) > 3 else ""
        live = parts[4].strip() if len(parts) > 4 else ""

        name_ok = "Arcturus Digital Milestone Celebration" in name
        date_ok = "2026-03-10" in date_from
        currency_ok = currency == "USD"
        live_ok = live in ("t", "true", "True", "1")

        ok = name_ok and date_ok and currency_ok and live_ok
        check("15. Pretix event", 2, ok,
              f"name_ok={name_ok}, date={date_from}, currency={currency}, live={live}")
    except Exception as e:
        check("15. Pretix event", 2, False, f"exception: {e}")


def check_16_pretix_product() -> None:
    """Product 'Celebration Gala Ticket' at price 175."""
    try:
        row = pretix_sql(
            "SELECT i.name::text, i.default_price "
            "FROM pretixbase_item i "
            "JOIN pretixbase_event e ON i.event_id = e.id "
            "WHERE e.slug = 'arcturus-milestone-celebration' "
            "AND i.name::text LIKE '%Celebration Gala Ticket%' "
            "LIMIT 1;"
        )
        if not row:
            check("16. Pretix product Celebration Gala Ticket", 1, False, "not found")
            return
        parts = row.split("|")
        name = parts[0].strip()
        price = float(parts[1].strip() or 0) if len(parts) > 1 else 0.0
        ok = "Celebration Gala Ticket" in name and abs(price - 175.0) < 0.01
        check("16. Pretix product Celebration Gala Ticket", 1, ok,
              f"price={price}")
    except Exception as e:
        check("16. Pretix product Celebration Gala Ticket", 1, False, f"exception: {e}")


def check_17_pretix_quota() -> None:
    """Quota 'Gala Ticket Quota' with size 40."""
    try:
        row = pretix_sql(
            "SELECT q.name, q.size "
            "FROM pretixbase_quota q "
            "JOIN pretixbase_event e ON q.event_id = e.id "
            "WHERE e.slug = 'arcturus-milestone-celebration' "
            "AND q.name LIKE '%Gala Ticket Quota%' "
            "LIMIT 1;"
        )
        if not row:
            check("17. Pretix quota Gala Ticket Quota", 1, False, "not found")
            return
        parts = row.split("|")
        name = parts[0].strip()
        size = int(parts[1].strip() or 0) if len(parts) > 1 else 0
        ok = "Gala Ticket Quota" in name and size == 40
        check("17. Pretix quota Gala Ticket Quota", 1, ok,
              f"name={name}, size={size}")
    except Exception as e:
        check("17. Pretix quota Gala Ticket Quota", 1, False, f"exception: {e}")


def check_18_pretix_voucher() -> None:
    """Voucher ARCTURUS100VIP with 100% discount, max 3, valid until 2026-03-10."""
    try:
        row = pretix_sql(
            "SELECT v.code, v.price_mode, v.value, v.max_usages, v.valid_until::text "
            "FROM pretixbase_voucher v "
            "JOIN pretixbase_event e ON v.event_id = e.id "
            "WHERE e.slug = 'arcturus-milestone-celebration' "
            "AND v.code = 'ARCTURUS100VIP' "
            "LIMIT 1;"
        )
        if not row:
            check("18. Pretix voucher ARCTURUS100VIP", 2, False, "not found")
            return
        parts = row.split("|")
        code = parts[0].strip()
        price_mode = parts[1].strip() if len(parts) > 1 else ""
        value = float(parts[2].strip() or 0) if len(parts) > 2 else 0.0
        max_usages = int(parts[3].strip() or 0) if len(parts) > 3 else 0
        valid_until = parts[4].strip() if len(parts) > 4 else ""

        code_ok = code == "ARCTURUS100VIP"
        discount_ok = price_mode == "percent" and abs(value - 100.0) < 0.01
        max_ok = max_usages == 3
        valid_ok = "2026-03-10" in valid_until

        ok = code_ok and discount_ok and max_ok and valid_ok
        check("18. Pretix voucher ARCTURUS100VIP", 2, ok,
              f"mode={price_mode}, value={value}, max={max_usages}, valid_until={valid_until}")
    except Exception as e:
        check("18. Pretix voucher ARCTURUS100VIP", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_twenty_company()
    check_2_twenty_person()
    check_3_twenty_opportunity()
    check_4_twenty_favorite()
    check_5_twenty_task_milestone2()
    check_6_twenty_task_celebration()
    check_7_bc_customer()
    check_8_bc_items()
    check_9_bc_account()
    check_10_bc_invoice1()
    check_11_bc_invoice2()
    check_12_bc_payment()
    check_13_bc_journal()
    check_14_bc_balance()
    check_15_pretix_event()
    check_16_pretix_product()
    check_17_pretix_quota()
    check_18_pretix_voucher()

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
