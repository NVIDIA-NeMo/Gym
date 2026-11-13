# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/tasks/uni-m/Business/business_052/verify.py
# Original license: CC-BY-4.0; Copyright: SaaS-Bench Authors; see bundled licenses.
# NVIDIA modifications: Require exactly 24 percent voucher discount (instead of upstream approximate 10.04 or 24); use SALES_ESTIMATES and search matching estimates.
"""
Verifier for Business-052-I5: Leadership Development Training Series
across Pretix, Twenty CRM, and BigCapital.

Checks: 15 weighted checks (22 total points).
Strategy: docker exec (Pretix Postgres, Twenty Postgres, BigCapital Postgres).

Required env vars:
  SERVER_HOSTNAME,
  PRETIX_PORT, PRETIX_CONTAINER, PRETIX_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER.
"""

import json
import os
import subprocess
import sys

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

PRETIX_PORT = os.environ.get("PRETIX_PORT")
PRETIX_CONTAINER = os.environ.get("PRETIX_CONTAINER")
PRETIX_DB_CONTAINER = os.environ.get("PRETIX_DB_CONTAINER")
TWENTY_PORT = os.environ.get("TWENTY_PORT")
TWENTY_CONTAINER = os.environ.get("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = os.environ.get("TWENTY_DB_CONTAINER")
BIGCAPITAL_PORT = os.environ.get("BIGCAPITAL_PORT")
BIGCAPITAL_CONTAINER = os.environ.get("BIGCAPITAL_CONTAINER")
BIGCAPITAL_DB_CONTAINER = os.environ.get("BIGCAPITAL_DB_CONTAINER")

_required = {
    "PRETIX_PORT": PRETIX_PORT,
    "PRETIX_CONTAINER": PRETIX_CONTAINER,
    "PRETIX_DB_CONTAINER": PRETIX_DB_CONTAINER,
    "TWENTY_PORT": TWENTY_PORT,
    "TWENTY_CONTAINER": TWENTY_CONTAINER,
    "TWENTY_DB_CONTAINER": TWENTY_DB_CONTAINER,
    "BIGCAPITAL_PORT": BIGCAPITAL_PORT,
    "BIGCAPITAL_CONTAINER": BIGCAPITAL_CONTAINER,
    "BIGCAPITAL_DB_CONTAINER": BIGCAPITAL_DB_CONTAINER,
}
for _var, _val in _required.items():
    if not _val:
        print(f"FATAL: {_var} not set", file=sys.stderr)
        sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
ORGANIZER = "nyc-cultural"
EVENT1_SLUG = "leadership-comm-skills"
EVENT2_SLUG = "leadership-negotiation-influence"
EVENT3_SLUG = "leadership-strategy-exec"

EVENT1_TICKET = "Leadership Communication Skills Ticket"
EVENT1_PRICE = 239.00
EVENT1_QUOTA_NAME = "Leadership Communication Skills Quota"
EVENT1_QUOTA_SIZE = 48
EVENT1_DISCOUNT_NAME = "Early Enrollment Discount"
EVENT1_DISCOUNT_PERCENT = 24.0

EVENT2_TICKET = "Leadership Negotiation Workshop Ticket"
EVENT2_PRICE = 289.00
EVENT2_QUOTA_NAME = "Leadership Negotiation Workshop Quota"
EVENT2_QUOTA_SIZE = 38
EVENT2_VOUCHER_CODE = "LDRNEGO17"
EVENT2_VOUCHER_PCT = 17
EVENT2_VOUCHER_MAX = 14

EVENT3_TICKET = "Leadership Strategy Masterclass Ticket"
EVENT3_PRICE = 369.00
EVENT3_QUOTA_NAME = "Leadership Strategy Masterclass Quota"
EVENT3_QUOTA_SIZE = 28
EVENT3_QUESTION = "Please describe your current leadership role and the size of the team you manage."

SERIES_TOTAL = (11 * 239.00) + (9 * 289.00) + (5 * 369.00)  # 7075.00

COMPANY_NAME = "Mediaocean"
OPP_TITLE = "Mediaocean Leadership Training Series 2026"
CONTACT_FIRST = "Victoria"
CONTACT_LAST = "Lam"
CONTACT_EMAIL = "victoria.lam@mediaocean-training.com"
CONTACT_TITLE = "Head of People Development"

TASK_TITLES = [
    "Send Leadership Communication Skills registration link",
    "Send Leadership Negotiation & Influence Workshop registration link",
    "Send Leadership Strategy & Executive Presence Masterclass registration link",
]

BC_ITEM_NAME = "Leadership Development Training Series Package"
BC_CUSTOMER = "Mediaocean Training Account"

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


def pretix_sql(query: str) -> str:
    """Run a SQL query against the Pretix Postgres DB and return stdout."""
    rc, out, err = docker_exec(
        PRETIX_DB_CONTAINER,
        "psql", "-U", "pretix", "-d", "pretix", "-t", "-A", "-c", query,
    )
    if rc != 0:
        raise RuntimeError(f"pretix psql error: {err.strip()}")
    return out.strip()


def twenty_sql(query: str) -> str:
    """Run a SQL query against the Twenty Postgres DB and return stdout."""
    # Try common user/db combos
    for user, db in [("twenty", "default"), ("twenty", "twenty"), ("postgres", "default"), ("postgres", "twenty")]:
        rc, out, err = docker_exec(
            TWENTY_DB_CONTAINER,
            "psql", "-U", user, "-d", db, "-t", "-A", "-c", query,
        )
        if rc == 0:
            return out.strip()
    raise RuntimeError(f"twenty psql error: {err.strip()}")


_bc_tenant_db: str = ""


def _discover_bc_tenant_db() -> None:
    global _bc_tenant_db
    rc, out, err = docker_exec(
        BIGCAPITAL_DB_CONTAINER, "mysql",
        "--default-character-set=utf8mb4",
        "-u", "bigcapital", "-pbigcapital123",
        "-N", "-B", "-e",
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
        "WHERE SCHEMA_NAME LIKE 'bigcapital_tenant_%' ORDER BY SCHEMA_NAME LIMIT 1;",
    )
    if rc == 0 and out.strip():
        _bc_tenant_db = out.strip().split("\n")[0]
    else:
        _bc_tenant_db = "bigcapital"


def bigcapital_sql(query: str) -> str:
    """Run a SQL query against the BigCapital MariaDB (auto-detects tenant DB)."""
    if not _bc_tenant_db:
        _discover_bc_tenant_db()
    rc, out, err = docker_exec(
        BIGCAPITAL_DB_CONTAINER, "mysql",
        "--default-character-set=utf8mb4",
        "-u", "bigcapital", "-pbigcapital123",
        "-D", _bc_tenant_db,
        "-N", "-B", "-e", query,
    )
    if rc != 0:
        raise RuntimeError(f"bigcapital mysql error: {err.strip()}")
    return out.strip()


def get_twenty_workspace_schema() -> str:
    """Find the workspace schema in Twenty's Postgres."""
    result = twenty_sql(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'workspace_%' LIMIT 1;"
    )
    if not result:
        raise RuntimeError("No workspace schema found in Twenty DB")
    return result.split("\n")[0].strip()


def name_str(val: str) -> str:
    """Extract text from a Pretix i18n JSON name field."""
    try:
        d = json.loads(val)
        if isinstance(d, dict):
            return d.get("en") or next(iter(d.values()), "")
    except (json.JSONDecodeError, TypeError):
        pass
    return val


# ── Pretix: get event ID by slug ──────────────────────────────────────────────
def get_pretix_event_id(slug: str) -> tuple[int, bool]:
    """Return (event_id, is_live) for a given event slug under ORGANIZER."""
    row = pretix_sql(
        f"SELECT e.id, e.live FROM pretixbase_event e "
        f"JOIN pretixbase_organizer o ON e.organizer_id = o.id "
        f"WHERE e.slug = '{slug}' AND o.slug = '{ORGANIZER}';"
    )
    if not row:
        return -1, False
    parts = row.split("|")
    return int(parts[0]), parts[1].strip().lower() in ("t", "true")


# ── Check 1: Pretix Event 1 exists and is live ───────────────────────────────
def check_1_event1_exists_live() -> None:
    """Event 'leadership-comm-skills' exists under nyc-cultural and is live."""
    try:
        eid, live = get_pretix_event_id(EVENT1_SLUG)
        ok = eid > 0 and live
        check("1. Pretix Event 1 exists & live", 1, ok,
              f"event_id={eid}, live={live}")
    except Exception as e:
        check("1. Pretix Event 1 exists & live", 1, False, f"exception: {e}")


# ── Check 2: Event 1 product + quota ─────────────────────────────────────────
def check_2_event1_product_quota() -> None:
    """Product at 239.00 and quota with size 48."""
    try:
        eid, _ = get_pretix_event_id(EVENT1_SLUG)
        if eid < 0:
            check("2. Event 1 product & quota", 2, False, "event not found")
            return

        # Check product
        item_row = pretix_sql(
            f"SELECT id, name, default_price FROM pretixbase_item "
            f"WHERE event_id = {eid};"
        )
        item_ok = False
        item_id = -1
        for line in item_row.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                iname = name_str(parts[1].strip())
                iprice = float(parts[2].strip())
                if EVENT1_TICKET.lower() in iname.lower() and abs(iprice - EVENT1_PRICE) < 0.01:
                    item_ok = True
                    item_id = int(parts[0].strip())

        # Check quota
        quota_row = pretix_sql(
            f"SELECT name, size FROM pretixbase_quota WHERE event_id = {eid};"
        )
        quota_ok = False
        for line in quota_row.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                qname = parts[0].strip()
                qsize = int(parts[1].strip())
                if EVENT1_QUOTA_NAME.lower() in qname.lower() and qsize == EVENT1_QUOTA_SIZE:
                    quota_ok = True

        ok = item_ok and quota_ok
        check("2. Event 1 product & quota", 2, ok,
              f"item_ok={item_ok}(id={item_id}), quota_ok={quota_ok}")
    except Exception as e:
        check("2. Event 1 product & quota", 2, False, f"exception: {e}")


# ── Check 3: Event 1 discount rule ───────────────────────────────────────────
def check_3_event1_discount() -> None:
    """Discount rule 'Early Enrollment Discount' exists for Event 1."""
    try:
        eid, _ = get_pretix_event_id(EVENT1_SLUG)
        if eid < 0:
            check("3. Event 1 discount rule", 2, False, "event not found")
            return

        row = pretix_sql(
            f"SELECT id, internal_name, active, benefit_discount_matching_percent "
            f"FROM pretixbase_discount WHERE event_id = {eid};"
        )
        found = False
        detail_parts = []
        for line in row.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                dname = parts[1].strip()
                active = parts[2].strip().lower() in ("t", "true")
                pct = float(parts[3].strip()) if parts[3].strip() else 0.0
                if EVENT1_DISCOUNT_NAME.lower() in dname.lower():
                    value_ok = abs(pct - EVENT1_DISCOUNT_PERCENT) < 0.01
                    found = active and value_ok
                    detail_parts.append(f"name={dname}, active={active}, pct={pct}")

        if not detail_parts:
            detail_parts.append("discount not found")

        check("3. Event 1 discount rule", 2, found, "; ".join(detail_parts))
    except Exception as e:
        check("3. Event 1 discount rule", 2, False, f"exception: {e}")


# ── Check 4: Pretix Event 2 exists and is live ───────────────────────────────
def check_4_event2_exists_live() -> None:
    """Event 'leadership-negotiation-influence' exists and is live."""
    try:
        eid, live = get_pretix_event_id(EVENT2_SLUG)
        ok = eid > 0 and live
        check("4. Pretix Event 2 exists & live", 1, ok,
              f"event_id={eid}, live={live}")
    except Exception as e:
        check("4. Pretix Event 2 exists & live", 1, False, f"exception: {e}")


# ── Check 5: Event 2 product + quota ─────────────────────────────────────────
def check_5_event2_product_quota() -> None:
    """Product at 289.00 and quota with size 38."""
    try:
        eid, _ = get_pretix_event_id(EVENT2_SLUG)
        if eid < 0:
            check("5. Event 2 product & quota", 2, False, "event not found")
            return

        item_row = pretix_sql(
            f"SELECT id, name, default_price FROM pretixbase_item WHERE event_id = {eid};"
        )
        item_ok = False
        for line in item_row.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                iname = name_str(parts[1].strip())
                iprice = float(parts[2].strip())
                if EVENT2_TICKET.lower() in iname.lower() and abs(iprice - EVENT2_PRICE) < 0.01:
                    item_ok = True

        quota_row = pretix_sql(
            f"SELECT name, size FROM pretixbase_quota WHERE event_id = {eid};"
        )
        quota_ok = False
        for line in quota_row.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                qname = parts[0].strip()
                qsize = int(parts[1].strip())
                if EVENT2_QUOTA_NAME.lower() in qname.lower() and qsize == EVENT2_QUOTA_SIZE:
                    quota_ok = True

        ok = item_ok and quota_ok
        check("5. Event 2 product & quota", 2, ok,
              f"item_ok={item_ok}, quota_ok={quota_ok}")
    except Exception as e:
        check("5. Event 2 product & quota", 2, False, f"exception: {e}")


# ── Check 6: Event 2 voucher ─────────────────────────────────────────────────
def check_6_event2_voucher() -> None:
    """Voucher LDRNEGO17 with 17% discount, max 14 usages."""
    try:
        eid, _ = get_pretix_event_id(EVENT2_SLUG)
        if eid < 0:
            check("6. Event 2 voucher LDRNEGO17", 2, False, "event not found")
            return

        row = pretix_sql(
            f"SELECT code, price_mode, value, max_usages FROM pretixbase_voucher "
            f"WHERE event_id = {eid} AND code = '{EVENT2_VOUCHER_CODE}';"
        )
        if not row:
            check("6. Event 2 voucher LDRNEGO17", 2, False, "voucher not found")
            return

        parts = row.split("|")
        price_mode = parts[1].strip() if len(parts) > 1 else ""
        value = float(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else 0.0
        max_usages = int(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else 0

        mode_ok = price_mode == "percent"
        val_ok = abs(value - EVENT2_VOUCHER_PCT) < 0.01
        max_ok = max_usages == EVENT2_VOUCHER_MAX
        ok = mode_ok and val_ok and max_ok

        check("6. Event 2 voucher LDRNEGO17", 2, ok,
              f"mode={price_mode}, value={value}, max={max_usages}")
    except Exception as e:
        check("6. Event 2 voucher LDRNEGO17", 2, False, f"exception: {e}")


# ── Check 7: Pretix Event 3 exists and is live ───────────────────────────────
def check_7_event3_exists_live() -> None:
    """Event 'leadership-strategy-exec' exists and is live."""
    try:
        eid, live = get_pretix_event_id(EVENT3_SLUG)
        ok = eid > 0 and live
        check("7. Pretix Event 3 exists & live", 1, ok,
              f"event_id={eid}, live={live}")
    except Exception as e:
        check("7. Pretix Event 3 exists & live", 1, False, f"exception: {e}")


# ── Check 8: Event 3 product + quota ─────────────────────────────────────────
def check_8_event3_product_quota() -> None:
    """Product at 369.00 and quota with size 28."""
    try:
        eid, _ = get_pretix_event_id(EVENT3_SLUG)
        if eid < 0:
            check("8. Event 3 product & quota", 2, False, "event not found")
            return

        item_row = pretix_sql(
            f"SELECT id, name, default_price FROM pretixbase_item WHERE event_id = {eid};"
        )
        item_ok = False
        for line in item_row.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                iname = name_str(parts[1].strip())
                iprice = float(parts[2].strip())
                if EVENT3_TICKET.lower() in iname.lower() and abs(iprice - EVENT3_PRICE) < 0.01:
                    item_ok = True

        quota_row = pretix_sql(
            f"SELECT name, size FROM pretixbase_quota WHERE event_id = {eid};"
        )
        quota_ok = False
        for line in quota_row.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                qname = parts[0].strip()
                qsize = int(parts[1].strip())
                if EVENT3_QUOTA_NAME.lower() in qname.lower() and qsize == EVENT3_QUOTA_SIZE:
                    quota_ok = True

        ok = item_ok and quota_ok
        check("8. Event 3 product & quota", 2, ok,
              f"item_ok={item_ok}, quota_ok={quota_ok}")
    except Exception as e:
        check("8. Event 3 product & quota", 2, False, f"exception: {e}")


# ── Check 9: Event 3 custom question ─────────────────────────────────────────
def check_9_event3_question() -> None:
    """Required text question exists on Event 3."""
    try:
        eid, _ = get_pretix_event_id(EVENT3_SLUG)
        if eid < 0:
            check("9. Event 3 custom question", 1, False, "event not found")
            return

        row = pretix_sql(
            f"SELECT question, type, required FROM pretixbase_question "
            f"WHERE event_id = {eid};"
        )
        found = False
        for line in row.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                qtext = name_str(parts[0].strip())
                qtype = parts[1].strip()
                qreq = parts[2].strip().lower() in ("t", "true")
                if EVENT3_QUESTION.lower() in qtext.lower() and qtype == "S" and qreq:
                    found = True

        check("9. Event 3 custom question", 1, found,
              "question found" if found else "question not found or wrong type/required")
    except Exception as e:
        check("9. Event 3 custom question", 1, False, f"exception: {e}")


# ── Check 10: Twenty opportunity ──────────────────────────────────────────────
def check_10_twenty_opportunity() -> None:
    """Opportunity with correct title, amount (~7075), and stage PROPOSAL."""
    try:
        ws = get_twenty_workspace_schema()
        row = twenty_sql(
            f"SELECT name, \"amountAmountMicros\", stage "
            f"FROM \"{ws}\".opportunity "
            f"WHERE name = '{OPP_TITLE}';"
        )
        if not row:
            check("10. Twenty opportunity", 2, False, "opportunity not found")
            return

        parts = row.split("|")
        micros = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
        stage = parts[2].strip().upper() if len(parts) > 2 else ""
        amount = micros / 1_000_000.0

        amount_ok = abs(amount - SERIES_TOTAL) < 1.0
        stage_ok = stage == "PROPOSAL"
        ok = amount_ok and stage_ok

        check("10. Twenty opportunity", 2, ok,
              f"amount={amount}, stage={stage}")
    except Exception as e:
        check("10. Twenty opportunity", 2, False, f"exception: {e}")


# ── Check 11: Twenty person Victoria Lam ──────────────────────────────────────
def check_11_twenty_person() -> None:
    """Person Victoria Lam with correct email, job title, linked to Mediaocean."""
    try:
        ws = get_twenty_workspace_schema()
        row = twenty_sql(
            f"SELECT p.\"nameFirstName\", p.\"nameLastName\", "
            f"p.\"emailsPrimaryEmail\", p.\"jobTitle\", c.name "
            f"FROM \"{ws}\".person p "
            f"LEFT JOIN \"{ws}\".company c ON p.\"companyId\" = c.id "
            f"WHERE p.\"nameFirstName\" = '{CONTACT_FIRST}' "
            f"AND p.\"nameLastName\" = '{CONTACT_LAST}';"
        )
        if not row:
            check("11. Twenty person Victoria Lam", 2, False, "person not found")
            return

        parts = row.split("|")
        email = parts[2].strip() if len(parts) > 2 else ""
        title = parts[3].strip() if len(parts) > 3 else ""
        company = parts[4].strip() if len(parts) > 4 else ""

        email_ok = email.lower() == CONTACT_EMAIL.lower()
        title_ok = CONTACT_TITLE.lower() in title.lower()
        company_ok = COMPANY_NAME.lower() in company.lower()
        ok = email_ok and title_ok and company_ok

        check("11. Twenty person Victoria Lam", 2, ok,
              f"email={email}, title={title}, company={company}")
    except Exception as e:
        check("11. Twenty person Victoria Lam", 2, False, f"exception: {e}")


# ── Check 12: Twenty tasks ───────────────────────────────────────────────────
def check_12_twenty_tasks() -> None:
    """Three registration tasks linked to Mediaocean with correct titles."""
    try:
        ws = get_twenty_workspace_schema()

        # Get Mediaocean company ID
        cid = twenty_sql(
            f"SELECT id FROM \"{ws}\".company WHERE name = '{COMPANY_NAME}' LIMIT 1;"
        )
        if not cid:
            check("12. Twenty 3 registration tasks", 2, False, "company Mediaocean not found")
            return
        cid = cid.strip()

        # Get tasks linked to Mediaocean via taskTarget
        task_rows = twenty_sql(
            f"SELECT t.title FROM \"{ws}\".task t "
            f"JOIN \"{ws}\".\"taskTarget\" tt ON t.id = tt.\"taskId\" "
            f"WHERE tt.\"targetCompanyId\" = '{cid}';"
        )
        found_titles = set()
        for line in task_rows.split("\n"):
            line = line.strip()
            if line:
                found_titles.add(line)

        missing = []
        for expected_title in TASK_TITLES:
            if not any(expected_title.lower() in ft.lower() for ft in found_titles):
                missing.append(expected_title[:50])

        ok = len(missing) == 0
        check("12. Twenty 3 registration tasks", 2, ok,
              f"found={len(found_titles)}, missing={missing}" if missing else "all 3 found")
    except Exception as e:
        check("12. Twenty 3 registration tasks", 2, False, f"exception: {e}")


# ── Check 13: BigCapital item ─────────────────────────────────────────────────
def check_13_bc_item() -> None:
    """Item 'Leadership Development Training Series Package' with sell price 7075."""
    try:
        row = bigcapital_sql(
            f"SELECT NAME, TYPE, SELL_PRICE FROM ITEMS "
            f"WHERE NAME = '{BC_ITEM_NAME}';"
        )
        if not row:
            check("13. BigCapital service item", 1, False, "item not found")
            return

        parts = row.split("\t")
        itype = parts[1].strip() if len(parts) > 1 else ""
        sprice = float(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else 0.0

        type_ok = itype.lower() == "service"
        price_ok = abs(sprice - SERIES_TOTAL) < 1.0
        ok = type_ok and price_ok

        check("13. BigCapital service item", 1, ok,
              f"type={itype}, sell_price={sprice}")
    except Exception as e:
        check("13. BigCapital service item", 1, False, f"exception: {e}")


# ── Check 14: BigCapital customer ─────────────────────────────────────────────
def check_14_bc_customer() -> None:
    """Customer 'Mediaocean Training Account' exists."""
    try:
        row = bigcapital_sql(
            f"SELECT DISPLAY_NAME, EMAIL FROM CONTACTS "
            f"WHERE CONTACT_SERVICE = 'customer' AND DISPLAY_NAME = '{BC_CUSTOMER}';"
        )
        if not row:
            check("14. BigCapital customer", 1, False, "customer not found")
            return

        parts = row.split("\t")
        email = parts[1].strip() if len(parts) > 1 else ""
        ok = True
        check("14. BigCapital customer", 1, ok,
              f"email={email}")
    except Exception as e:
        check("14. BigCapital customer", 1, False, f"exception: {e}")


# ── Check 15: BigCapital estimate delivered with correct total ────────────────
def check_15_bc_estimate() -> None:
    """Sales estimate for Mediaocean Training Account, delivered, total = 7075."""
    try:
        # Find customer ID
        cid_row = bigcapital_sql(
            f"SELECT ID FROM CONTACTS "
            f"WHERE CONTACT_SERVICE = 'customer' AND DISPLAY_NAME = '{BC_CUSTOMER}';"
        )
        if not cid_row:
            check("15. BigCapital estimate delivered", 3, False, "customer not found")
            return
        cid = cid_row.strip().split("\n")[0].strip()

        # Find the estimate
        est_row = bigcapital_sql(
            f"SELECT ID, AMOUNT, DELIVERED_AT FROM SALES_ESTIMATES "
            f"WHERE CUSTOMER_ID = {cid};"
        )
        if not est_row:
            check("15. BigCapital estimate delivered", 3, False, "estimate not found")
            return

        expected_lines = [
            (11, 239.00),
            (9, 289.00),
            (5, 369.00),
        ]
        details = []
        for row in est_row.splitlines():
            parts = row.split("\t")
            est_id, amount_text, delivered_at = parts
            amount = float(amount_text)
            delivered_ok = bool(delivered_at.strip()) and delivered_at.strip().upper() != "NULL"
            lines_row = bigcapital_sql(
                f"SELECT DESCRIPTION, QUANTITY, RATE FROM ITEMS_ENTRIES "
                f"WHERE REFERENCE_TYPE = 'SaleEstimate' AND REFERENCE_ID = '{int(est_id)}' "
                f"ORDER BY `INDEX`;"
            )
            actual_lines = []
            for line in lines_row.splitlines():
                _, quantity, rate = line.split("\t")
                actual_lines.append((float(quantity), float(rate)))
            lines_ok = sorted(actual_lines) == sorted(expected_lines)
            ok = delivered_ok and abs(amount - SERIES_TOTAL) < 0.01 and lines_ok
            details.append(f"id={est_id}, amount={amount}, delivered={delivered_ok}, lines_match={lines_ok}")
            if ok:
                check("15. BigCapital estimate delivered", 3, True, details[-1])
                return
        check("15. BigCapital estimate delivered", 3, False, "; ".join(details))
    except Exception as e:
        check("15. BigCapital estimate delivered", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_event1_exists_live()
    check_2_event1_product_quota()
    check_3_event1_discount()
    check_4_event2_exists_live()
    check_5_event2_product_quota()
    check_6_event2_voucher()
    check_7_event3_exists_live()
    check_8_event3_product_quota()
    check_9_event3_question()
    check_10_twenty_opportunity()
    check_11_twenty_person()
    check_12_twenty_tasks()
    check_13_bc_item()
    check_14_bc_customer()
    check_15_bc_estimate()

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
