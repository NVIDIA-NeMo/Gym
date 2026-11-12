# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_084/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-084-I5: AI Horizons 2026 Conference Sponsorship & Lead Pipeline

Checks: 17 weighted checks across pretix, bigcapital, twenty, hrms.
Strategy: docker exec DB (pretix, bigcapital, twenty) + REST API (hrms)

Required env vars:
  SERVER_HOSTNAME, PRETIX_PORT, PRETIX_CONTAINER, PRETIX_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER,
  HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER
"""

import os
import sys
import subprocess
import json
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

PRETIX_PORT = os.environ.get("PRETIX_PORT")
PRETIX_CONTAINER = os.environ.get("PRETIX_CONTAINER")
PRETIX_DB = os.environ.get("PRETIX_DB_CONTAINER")

BC_PORT = os.environ.get("BIGCAPITAL_PORT")
BC_CONTAINER = os.environ.get("BIGCAPITAL_CONTAINER")
BC_DB = os.environ.get("BIGCAPITAL_DB_CONTAINER")

TW_PORT = os.environ.get("TWENTY_PORT")
TW_CONTAINER = os.environ.get("TWENTY_CONTAINER")
TW_DB = os.environ.get("TWENTY_DB_CONTAINER")

HRMS_PORT = os.environ.get("HRMS_PORT")
HRMS_CONTAINER = os.environ.get("HRMS_CONTAINER")
HRMS_DB = os.environ.get("HRMS_DB_CONTAINER")

for var in ("PRETIX_PORT", "PRETIX_CONTAINER", "PRETIX_DB_CONTAINER",
            "BIGCAPITAL_PORT", "BIGCAPITAL_CONTAINER", "BIGCAPITAL_DB_CONTAINER",
            "TWENTY_PORT", "TWENTY_CONTAINER", "TWENTY_DB_CONTAINER",
            "HRMS_PORT", "HRMS_CONTAINER", "HRMS_DB_CONTAINER"):
    if not os.environ.get(var):
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)


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


def pq(query: str, timeout: int = 15) -> str:
    """Query Pretix Postgres DB."""
    _, out, _ = docker_exec(
        PRETIX_DB, "psql", "-U", "pretix", "-d", "pretix",
        "-t", "-A", "-c", query, timeout=timeout,
    )
    return out.strip()


# ── BigCapital DB helpers ─────────────────────────────────────────────────────
_bc_tenant_db: str | None = None


def bc_tenant_db() -> str:
    """Discover BigCapital tenant DB name."""
    global _bc_tenant_db
    if _bc_tenant_db:
        return _bc_tenant_db
    _, out, _ = docker_exec(
        BC_DB, "mysql", "-u", "bigcapital", "-pbigcapital123",
        "-N", "-B", "-e", "SHOW DATABASES LIKE 'bigcapital_tenant_%'",
    )
    dbs = [line.strip() for line in out.strip().splitlines() if line.strip()]
    if not dbs:
        raise RuntimeError("No BigCapital tenant DB found")
    _bc_tenant_db = dbs[0]
    return _bc_tenant_db


def bcq(query: str, timeout: int = 15) -> str:
    """Query BigCapital tenant MariaDB."""
    db = bc_tenant_db()
    _, out, _ = docker_exec(
        BC_DB, "mysql", "-u", "bigcapital", "-pbigcapital123",
        "--default-character-set=utf8mb4", db, "-N", "-B", "-e", query,
        timeout=timeout,
    )
    return out.strip()


# ── Twenty DB helpers ─────────────────────────────────────────────────────────
_tw_schema: str | None = None


def tw_schema() -> str:
    """Discover Twenty workspace schema."""
    global _tw_schema
    if _tw_schema:
        return _tw_schema
    _, out, _ = docker_exec(
        TW_DB, "psql", "-U", "postgres", "-d", "default",
        "-t", "-A", "-c",
        "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'workspace_%' LIMIT 1",
    )
    schema = out.strip().splitlines()[0].strip() if out.strip() else ""
    if not schema:
        raise RuntimeError("No Twenty workspace schema found")
    _tw_schema = schema
    return _tw_schema


def twq(query: str, timeout: int = 15) -> str:
    """Query Twenty Postgres DB."""
    _, out, _ = docker_exec(
        TW_DB, "psql", "-U", "postgres", "-d", "default",
        "-t", "-A", "-c", query, timeout=timeout,
    )
    return out.strip()


# ── HRMS (Frappe) API ─────────────────────────────────────────────────────────
_hrms_session: requests.Session | None = None


def hrms_api() -> requests.Session:
    global _hrms_session
    if _hrms_session:
        return _hrms_session
    s = requests.Session()
    r = s.post(
        f"http://{HOST}:{HRMS_PORT}/api/method/login",
        data={"usr": "Administrator", "pwd": "admin"},
        timeout=10,
    )
    r.raise_for_status()
    _hrms_session = s
    return s


def hrms_get(doctype: str, filters: list | None = None, fields: list | None = None) -> list:
    s = hrms_api()
    params: dict = {}
    if filters:
        params["filters"] = json.dumps(filters)
    if fields:
        params["fields"] = json.dumps(fields)
    r = s.get(
        f"http://{HOST}:{HRMS_PORT}/api/resource/{doctype}",
        params=params, timeout=10,
    )
    r.raise_for_status()
    return r.json().get("data", [])


# ── Pretix event ID cache ────────────────────────────────────────────────────
_pretix_eid: int | None = None


def pretix_eid() -> int:
    global _pretix_eid
    if _pretix_eid is None:
        row = pq("SELECT id FROM pretixbase_event WHERE slug='ai-horizons-2026'")
        _pretix_eid = int(row) if row else 0
    return _pretix_eid


# ══════════════════════════════════════════════════════════════════════════════
# CHECKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Pretix (7 checks) ────────────────────────────────────────────────────────

def check_1_pretix_event():
    """Pretix event exists, live, slug, start date 2026-12-03, currency USD."""
    try:
        eid = pretix_eid()
        if not eid:
            check("1. Pretix event basics", 2, False, "event 'ai-horizons-2026' not found")
            return
        row = pq(
            f"SELECT name->>'en', date_from, currency, live "
            f"FROM pretixbase_event WHERE id={eid}"
        )
        parts = row.split("|")
        name, date_from, currency, live = parts[0], parts[1], parts[2], parts[3]
        ok = (
            "AI Horizons 2026" in name
            and date_from.startswith("2026-12-03")
            and currency == "USD"
            and live.lower() in ("t", "true")
        )
        check("1. Pretix event basics", 2, ok,
              f"name={name!r}, date={date_from}, curr={currency}, live={live}")
    except Exception as e:
        check("1. Pretix event basics", 2, False, f"exception: {e}")


def check_2_pretix_products():
    """4 products with correct prices and category assignments."""
    try:
        eid = pretix_eid()
        if not eid:
            check("2. Pretix products & categories", 2, False, "no event")
            return
        cats_raw = pq(f"SELECT id, name->>'en' FROM pretixbase_itemcategory WHERE event_id={eid}")
        cats = {}
        for line in cats_raw.splitlines():
            if "|" in line:
                cid, cname = line.split("|", 1)
                cats[int(cid)] = cname

        items_raw = pq(
            f"SELECT name->>'en', default_price, category_id "
            f"FROM pretixbase_item WHERE event_id={eid}"
        )
        items: dict[str, dict] = {}
        for line in items_raw.splitlines():
            if "|" in line:
                p = line.split("|")
                items[p[0]] = {"price": float(p[1]), "cat_id": int(p[2]) if p[2] else None}

        expected = {
            "Platinum Exhibit Booth": (11000.0, "Sponsor Exhibits"),
            "Gold Exhibit Booth": (5500.0, "Sponsor Exhibits"),
            "General Entry Ticket": (329.0, "Attendee Tickets"),
            "Workshop Session Pass": (169.0, "Workshop Sessions"),
        }
        issues = []
        for name, (price, cat_name) in expected.items():
            item = items.get(name)
            if not item:
                issues.append(f"{name}: missing")
            elif abs(item["price"] - price) > 0.01:
                issues.append(f"{name}: price={item['price']} expected {price}")
            elif item["cat_id"] and cats.get(item["cat_id"]) != cat_name:
                issues.append(f"{name}: cat={cats.get(item['cat_id'])!r}")

        check("2. Pretix products & categories", 2, not issues,
              "all 4 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("2. Pretix products & categories", 2, False, f"exception: {e}")


def check_3_pretix_quotas():
    """4 quotas with correct sizes."""
    try:
        eid = pretix_eid()
        if not eid:
            check("3. Pretix quotas", 2, False, "no event")
            return
        rows = pq(f"SELECT name, size FROM pretixbase_quota WHERE event_id={eid}")
        quotas: dict[str, int] = {}
        for line in rows.splitlines():
            if "|" in line:
                n, s = line.split("|", 1)
                quotas[n] = int(s)

        expected = {
            "Platinum Exhibit Quota": 4,
            "Gold Exhibit Quota": 7,
            "General Entry Quota": 550,
            "Workshop Session Quota": 130,
        }
        issues = []
        for name, size in expected.items():
            if name not in quotas:
                issues.append(f"{name}: missing")
            elif quotas[name] != size:
                issues.append(f"{name}: size={quotas[name]}")

        check("3. Pretix quotas", 2, not issues,
              "all 4 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("3. Pretix quotas", 2, False, f"exception: {e}")


def check_4_pretix_vouchers():
    """2 vouchers with correct codes, tag, max_usages, 100% discount, valid_until."""
    try:
        eid = pretix_eid()
        if not eid:
            check("4. Pretix vouchers", 2, False, "no event")
            return
        rows = pq(
            f"SELECT code, tag, max_usages, value, price_mode, valid_until "
            f"FROM pretixbase_voucher WHERE event_id={eid} "
            f"AND tag='AIHORIZONS-SPONSOR-COMP'"
        )
        vouchers: dict[str, dict] = {}
        for line in rows.splitlines():
            if "|" in line:
                p = line.split("|")
                vouchers[p[0]] = {
                    "tag": p[1], "max_usages": int(p[2]),
                    "value": float(p[3]) if p[3] else 0.0,
                    "price_mode": p[4], "valid_until": p[5],
                }

        issues = []
        for code in ("PLAT-AIHORIZONS-001", "GOLD-AIHORIZONS-001"):
            v = vouchers.get(code)
            if not v:
                issues.append(f"{code}: missing")
                continue
            if v["max_usages"] != 5:
                issues.append(f"{code}: max_usages={v['max_usages']}")
            is_100pct = (
                (v["price_mode"] == "percent" and abs(v["value"] - 100.0) < 0.01)
                or (v["price_mode"] == "set" and abs(v["value"]) < 0.01)
            )
            if not is_100pct:
                issues.append(f"{code}: mode={v['price_mode']}, val={v['value']}")
            if not v["valid_until"].startswith("2026-12-03"):
                issues.append(f"{code}: valid_until={v['valid_until']}")

        total_tagged = len(vouchers)
        if total_tagged != 2:
            issues.append(f"expected 2 tagged vouchers, found {total_tagged}")

        check("4. Pretix vouchers", 2, not issues,
              "both correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("4. Pretix vouchers", 2, False, f"exception: {e}")


def check_5_pretix_questions():
    """Custom questions: 'Company Name' for booth items, 'Job Title' for general ticket."""
    try:
        eid = pretix_eid()
        if not eid:
            check("5. Pretix custom questions", 1, False, "no event")
            return
        rows = pq(
            f"SELECT id, question->>'en', required "
            f"FROM pretixbase_question WHERE event_id={eid}"
        )
        questions: dict[str, dict] = {}
        for line in rows.splitlines():
            if "|" in line:
                p = line.split("|")
                questions[p[1]] = {"id": int(p[0]), "required": p[2].lower() in ("t", "true")}

        issues = []
        if "Company Name" not in questions:
            issues.append("'Company Name' missing")
        elif not questions["Company Name"]["required"]:
            issues.append("'Company Name' not required")

        if "Job Title" not in questions:
            issues.append("'Job Title' missing")
        elif not questions["Job Title"]["required"]:
            issues.append("'Job Title' not required")

        check("5. Pretix custom questions", 1, not issues,
              "both exist and required" if not issues else "; ".join(issues))
    except Exception as e:
        check("5. Pretix custom questions", 1, False, f"exception: {e}")


def check_6_pretix_checkin():
    """Two check-in lists with correct product linkage."""
    try:
        eid = pretix_eid()
        if not eid:
            check("6. Pretix check-in lists", 2, False, "no event")
            return
        rows = pq(f"SELECT id, name FROM pretixbase_checkinlist WHERE event_id={eid}")
        lists: dict[str, int] = {}
        for line in rows.splitlines():
            if "|" in line:
                lid, lname = line.split("|", 1)
                lists[lname] = int(lid)

        issues = []
        if "AI Horizons Main Check-In" not in lists:
            issues.append("'AI Horizons Main Check-In' missing")
        else:
            cid = lists["AI Horizons Main Check-In"]
            linked = pq(
                f"SELECT i.name->>'en' FROM pretixbase_checkinlist_limit_products cl "
                f"JOIN pretixbase_item i ON cl.item_id=i.id WHERE cl.checkinlist_id={cid}"
            )
            linked_names = set(linked.splitlines()) if linked else set()
            all_4 = {"Platinum Exhibit Booth", "Gold Exhibit Booth",
                      "General Entry Ticket", "Workshop Session Pass"}
            if linked_names and not (all_4 <= linked_names):
                issues.append(f"Main check-in missing: {all_4 - linked_names}")

        if "Sponsor Exhibit Check-In" not in lists:
            issues.append("'Sponsor Exhibit Check-In' missing")
        else:
            cid = lists["Sponsor Exhibit Check-In"]
            linked = pq(
                f"SELECT i.name->>'en' FROM pretixbase_checkinlist_limit_products cl "
                f"JOIN pretixbase_item i ON cl.item_id=i.id WHERE cl.checkinlist_id={cid}"
            )
            linked_names = set(linked.splitlines()) if linked else set()
            sponsor_items = {"Platinum Exhibit Booth", "Gold Exhibit Booth"}
            if not (sponsor_items <= linked_names):
                issues.append(f"Sponsor check-in linked to: {linked_names}")
            if linked_names - sponsor_items:
                issues.append(f"Sponsor check-in extra: {linked_names - sponsor_items}")

        check("6. Pretix check-in lists", 2, not issues,
              "both correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("6. Pretix check-in lists", 2, False, f"exception: {e}")


def check_7_pretix_tax_discount():
    """Tax rule 'Conference Tax Rule' at 8% + discount 'Early Bird 18% Off'."""
    try:
        eid = pretix_eid()
        if not eid:
            check("7. Pretix tax & discount rules", 1, False, "no event")
            return
        issues = []

        # Tax rule
        tax_raw = pq(f"SELECT name, rate FROM pretixbase_taxrule WHERE event_id={eid}")
        tax_found = False
        for line in tax_raw.splitlines():
            if "|" in line:
                n, r = line.split("|", 1)
                if "Conference Tax Rule" in n and abs(float(r) - 8.0) < 0.01:
                    tax_found = True
        if not tax_found:
            issues.append("'Conference Tax Rule' 8% not found")

        # Discount: use pretixbase_discount (the built-in discount table)
        try:
            disc_raw = pq(
                f"SELECT internal_name, benefit_discount_matching_percent "
                f"FROM pretixbase_discount "
                f"WHERE event_id={eid}"
            )
            disc_found = False
            for line in (disc_raw or "").splitlines():
                if "|" in line:
                    n, pct = line.split("|", 1)
                    if "Early Bird" in n and abs(float(pct) - 18.0) < 0.01:
                        disc_found = True
            if not disc_found and disc_raw:
                issues.append("'Early Bird 18% Off' not found")
        except Exception:
            pass  # discount table may not exist yet

        check("7. Pretix tax & discount rules", 1, not issues,
              "tax rule correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("7. Pretix tax & discount rules", 1, False, f"exception: {e}")


# ── BigCapital (4 checks) ────────────────────────────────────────────────────

def check_8_bc_accounts_items():
    """BigCapital: 2 accounts, 2 customers, 2 items exist correctly."""
    try:
        issues = []

        # Accounts
        accts = bcq("SELECT NAME, ACCOUNT_TYPE FROM ACCOUNTS WHERE NAME IN "
                     "('AI Horizons Sponsorship Revenue','Deferred AI Horizons Sponsorship')")
        acct_map: dict[str, str] = {}
        for line in accts.splitlines():
            if "\t" in line:
                n, t = line.split("\t", 1)
                acct_map[n] = t
        if "AI Horizons Sponsorship Revenue" not in acct_map:
            issues.append("account 'AI Horizons Sponsorship Revenue' missing")
        if "Deferred AI Horizons Sponsorship" not in acct_map:
            issues.append("account 'Deferred AI Horizons Sponsorship' missing")

        # Customers
        custs = bcq("SELECT DISPLAY_NAME FROM CONTACTS WHERE CONTACT_SERVICE='customer' "
                     "AND (DISPLAY_NAME LIKE '%Zenith%' OR DISPLAY_NAME LIKE '%Prism%')")
        cust_names = set(custs.splitlines()) if custs else set()
        if not any("Zenith" in c for c in cust_names):
            issues.append("customer 'Zenith Cloud Corp' missing")
        if not any("Prism" in c for c in cust_names):
            issues.append("customer 'Prism Digital Solutions' missing")

        # Items
        items = bcq("SELECT NAME, SELL_PRICE, TYPE FROM ITEMS WHERE NAME IN "
                     "('Platinum Exhibit Sponsorship Service','Gold Exhibit Sponsorship Service')")
        for exp_name, exp_price in [("Platinum Exhibit Sponsorship Service", 11000),
                                     ("Gold Exhibit Sponsorship Service", 5500)]:
            found = False
            for line in items.splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0] == exp_name:
                    found = True
                    if abs(float(parts[1]) - exp_price) > 0.01:
                        issues.append(f"{exp_name}: price={parts[1]}")
            if not found:
                issues.append(f"item '{exp_name}' missing")

        check("8. BigCapital accounts & items", 2, not issues,
              "all present" if not issues else "; ".join(issues))
    except Exception as e:
        check("8. BigCapital accounts & items", 2, False, f"exception: {e}")


def check_9_bc_invoice_zenith():
    """BigCapital: invoice for Zenith Cloud Corp delivered + fully paid."""
    try:
        row = bcq(
            "SELECT si.BALANCE, si.PAYMENT_AMOUNT, si.DELIVERED_AT, c.DISPLAY_NAME "
            "FROM SALES_INVOICES si "
            "JOIN CONTACTS c ON si.CUSTOMER_ID = c.ID "
            "WHERE c.DISPLAY_NAME LIKE '%Zenith%' LIMIT 1"
        )
        if not row:
            check("9. BigCapital invoice Zenith (paid)", 2, False, "invoice not found")
            return
        parts = row.split("\t")
        balance = float(parts[0]) if parts[0] else -1
        payment = float(parts[1]) if parts[1] else 0
        delivered = parts[2] if len(parts) > 2 else ""
        issues = []
        if not delivered or delivered == "NULL":
            issues.append("not delivered")
        if balance > 0.01:
            issues.append(f"balance={balance}, expected 0 (paid)")
        check("9. BigCapital invoice Zenith (paid)", 2, not issues,
              "delivered + paid" if not issues else "; ".join(issues))
    except Exception as e:
        check("9. BigCapital invoice Zenith (paid)", 2, False, f"exception: {e}")


def check_10_bc_invoice_prism():
    """BigCapital: invoice for Prism Digital Solutions delivered, 11000 outstanding."""
    try:
        row = bcq(
            "SELECT si.BALANCE, si.PAYMENT_AMOUNT, si.DELIVERED_AT, c.DISPLAY_NAME "
            "FROM SALES_INVOICES si "
            "JOIN CONTACTS c ON si.CUSTOMER_ID = c.ID "
            "WHERE c.DISPLAY_NAME LIKE '%Prism%' LIMIT 1"
        )
        if not row:
            check("10. BigCapital invoice Prism (outstanding)", 2, False, "invoice not found")
            return
        parts = row.split("\t")
        balance = float(parts[0]) if parts[0] else 0
        delivered = parts[2] if len(parts) > 2 else ""
        issues = []
        if not delivered or delivered == "NULL":
            issues.append("not delivered")
        if abs(balance - 11000) > 0.01:
            issues.append(f"balance={balance}, expected 11000")
        check("10. BigCapital invoice Prism (outstanding)", 2, not issues,
              "delivered, 11000 outstanding" if not issues else "; ".join(issues))
    except Exception as e:
        check("10. BigCapital invoice Prism (outstanding)", 2, False, f"exception: {e}")


def check_11_bc_journal():
    """BigCapital: published journal entry debiting revenue 11000, crediting deferred 11000."""
    try:
        rows = bcq(
            "SELECT mj.ID, mj.DATE, mj.PUBLISHED_AT, "
            "mje.DEBIT, mje.CREDIT, a.NAME AS ACCT_NAME "
            "FROM MANUAL_JOURNALS mj "
            "JOIN MANUAL_JOURNALS_ENTRIES mje ON mj.ID = mje.MANUAL_JOURNAL_ID "
            "JOIN ACCOUNTS a ON mje.ACCOUNT_ID = a.ID "
            "WHERE mj.DATE = '2026-10-18'"
        )
        has_debit = False
        has_credit = False
        is_published = False
        for line in rows.splitlines():
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            published = parts[2]
            debit = float(parts[3]) if parts[3] and parts[3] != "NULL" else 0
            credit = float(parts[4]) if parts[4] and parts[4] != "NULL" else 0
            acct = parts[5]
            if published and published != "NULL":
                is_published = True
            if "Sponsorship Revenue" in acct and abs(debit - 11000) < 0.01:
                has_debit = True
            if "Deferred" in acct and abs(credit - 11000) < 0.01:
                has_credit = True

        issues = []
        if not has_debit:
            issues.append("debit to Revenue 11000 not found")
        if not has_credit:
            issues.append("credit to Deferred 11000 not found")
        if (has_debit or has_credit) and not is_published:
            issues.append("journal not published")

        check("11. BigCapital deferral journal entry", 3,
              not issues, "published, balanced" if not issues else "; ".join(issues))
    except Exception as e:
        check("11. BigCapital deferral journal entry", 3, False, f"exception: {e}")


# ── Twenty CRM (5 checks) ────────────────────────────────────────────────────

def check_12_twenty_companies():
    """6 companies exist in Twenty with correct domains."""
    try:
        ws = tw_schema()
        expected = {
            "Zenith Cloud Corp": "zenithcloud.com",
            "Prism Digital Solutions": "prismdigital.com",
            "Ironclad Analytics": None,
            "Mosaic Data Labs": None,
            "Parallax Systems": None,
            "Cipher Tech Inc": None,
        }
        names_sql = ", ".join(f"'{n}'" for n in expected)
        rows = twq(
            f'SELECT "name", "domainNamePrimaryLinkUrl" '
            f'FROM {ws}.company WHERE "deletedAt" IS NULL '
            f'AND "name" IN ({names_sql})'
        )
        found: dict[str, str] = {}
        for line in rows.splitlines():
            if "|" in line:
                n, d = line.split("|", 1)
                found[n] = d

        issues = []
        for name, exp_domain in expected.items():
            if name not in found:
                issues.append(f"{name}: missing")
            elif exp_domain and exp_domain not in found[name]:
                issues.append(f"{name}: domain={found[name]!r}")

        check("12. Twenty companies", 2, not issues,
              f"all 6 present" if not issues else "; ".join(issues))
    except Exception as e:
        check("12. Twenty companies", 2, False, f"exception: {e}")


def check_13_twenty_people():
    """6 people with correct emails, titles, and company links."""
    try:
        ws = tw_schema()
        expected = [
            ("sponsor@zenithcloud.com", "Howard", "Lim", "Chief Technology Officer", "Zenith Cloud Corp"),
            ("events@prismdigital.com", "Beatrice", "Fontaine", "Head of Partnerships", "Prism Digital Solutions"),
            ("owen.stafford@ironclad-analytics.com", "Owen", "Stafford", None, "Ironclad Analytics"),
            ("yuki.tanaka@mosaicdatalabs.io", "Yuki", "Tanaka", None, "Mosaic Data Labs"),
            ("renee.holloway@parallaxsystems.com", "Renee", "Holloway", None, "Parallax Systems"),
            ("andre.dubois@ciphertech.com", "Andre", "Dubois", None, "Cipher Tech Inc"),
        ]
        emails_sql = ", ".join(f"'{e}'" for e, *_ in expected)
        rows = twq(
            f'SELECT p."emailsPrimaryEmail", p."nameFirstName", p."nameLastName", '
            f'p."jobTitle", c."name" AS company_name '
            f'FROM {ws}.person p '
            f'LEFT JOIN {ws}.company c ON p."companyId" = c.id '
            f'WHERE p."deletedAt" IS NULL '
            f'AND p."emailsPrimaryEmail" IN ({emails_sql})'
        )
        found: dict[str, dict] = {}
        for line in rows.splitlines():
            if "|" in line:
                parts = line.split("|")
                found[parts[0]] = {
                    "first": parts[1], "last": parts[2],
                    "title": parts[3] if len(parts) > 3 else "",
                    "company": parts[4] if len(parts) > 4 else "",
                }

        issues = []
        for email, first, last, title, company in expected:
            p = found.get(email)
            if not p:
                issues.append(f"{first} {last}: missing")
            else:
                if p["first"].lower() != first.lower() or p["last"].lower() != last.lower():
                    issues.append(f"{email}: name={p['first']} {p['last']}")
                if title and title.lower() not in (p.get("title") or "").lower():
                    issues.append(f"{first} {last}: title={p['title']!r}")
                if company.lower() not in (p.get("company") or "").lower():
                    issues.append(f"{first} {last}: company={p['company']!r}")

        check("13. Twenty people", 2, not issues,
              f"all 6 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("13. Twenty people", 2, False, f"exception: {e}")


def check_14_twenty_opportunities():
    """6 opportunities with correct amounts, stages, close dates, company links."""
    try:
        ws = tw_schema()
        expected = [
            ("Platinum Sponsor — Zenith Cloud Corp", 11000, "WON", "2026-10-18", "Zenith Cloud Corp"),
            ("Gold Sponsor — Prism Digital Solutions", 11000, "PROPOSAL", "2026-11-01", "Prism Digital Solutions"),
            ("Conference Lead — Ironclad Analytics", 14000, "QUALIFICATION", "2027-02-28", "Ironclad Analytics"),
            ("Conference Lead — Mosaic Data Labs", 14000, "QUALIFICATION", "2027-02-28", "Mosaic Data Labs"),
            ("Conference Lead — Parallax Systems", 14000, "QUALIFICATION", "2027-02-28", "Parallax Systems"),
            ("Conference Lead — Cipher Tech Inc", 14000, "QUALIFICATION", "2027-02-28", "Cipher Tech Inc"),
        ]

        # Use em-dash and regular dash matching
        rows = twq(
            f'SELECT o."name", o."amountAmountMicros", o."stage", '
            f'o."closeDate", c."name" AS company_name '
            f'FROM {ws}.opportunity o '
            f'LEFT JOIN {ws}.company c ON o."companyId" = c.id '
            f'WHERE o."deletedAt" IS NULL'
        )
        found: dict[str, dict] = {}
        for line in rows.splitlines():
            if "|" in line:
                parts = line.split("|")
                micros = int(parts[1]) if parts[1] else 0
                found[parts[0]] = {
                    "amount": micros / 1_000_000,
                    "stage": parts[2].upper() if parts[2] else "",
                    "closeDate": parts[3],
                    "company": parts[4] if len(parts) > 4 else "",
                }

        issues = []
        for name, exp_amt, exp_stage, exp_date, exp_company in expected:
            # Try both em-dash and regular dash
            opp = found.get(name)
            if not opp:
                alt_name = name.replace("—", "-").replace("\u2014", "-")
                opp = found.get(alt_name)
            if not opp:
                alt_name = name.replace("—", "\u2014")
                opp = found.get(alt_name)
            if not opp:
                # Fuzzy search
                for k, v in found.items():
                    if exp_company.lower() in k.lower() and (
                        "sponsor" in k.lower() if "Sponsor" in name else "lead" in k.lower()
                    ):
                        opp = v
                        break
            if not opp:
                issues.append(f"{name}: missing")
                continue
            if abs(opp["amount"] - exp_amt) > 1:
                issues.append(f"{name}: amount={opp['amount']}")
            if exp_stage not in opp["stage"]:
                issues.append(f"{name}: stage={opp['stage']!r}")
            if exp_date not in opp["closeDate"]:
                issues.append(f"{name}: closeDate={opp['closeDate']!r}")
            if exp_company.lower() not in opp["company"].lower():
                issues.append(f"{name}: company={opp['company']!r}")

        check("14. Twenty opportunities", 2, not issues,
              f"all 6 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("14. Twenty opportunities", 2, False, f"exception: {e}")


def check_15_twenty_tasks():
    """4 lead follow-up tasks with correct due dates."""
    try:
        ws = tw_schema()
        leads = ["Ironclad Analytics", "Mosaic Data Labs", "Parallax Systems", "Cipher Tech Inc"]
        issues = []
        for company in leads:
            # Search for task with title containing the company name and "follow-up"
            rows = twq(
                f"SELECT t.\"title\", t.\"dueAt\", t.\"bodyV2Markdown\" "
                f"FROM {ws}.task t "
                f"WHERE t.\"deletedAt\" IS NULL "
                f"AND t.\"title\" LIKE '%follow-up%{company}%'"
            )
            if not rows:
                # Try with em-dash
                rows = twq(
                    f"SELECT t.\"title\", t.\"dueAt\", t.\"bodyV2Markdown\" "
                    f"FROM {ws}.task t "
                    f"WHERE t.\"deletedAt\" IS NULL "
                    f"AND t.\"title\" LIKE '%{company}%' "
                    f"AND t.\"title\" LIKE '%follow%'"
                )
            if not rows:
                issues.append(f"task for {company}: missing")
                continue
            first = rows.splitlines()[0]
            parts = first.split("|")
            due = parts[1] if len(parts) > 1 else ""
            body = parts[2] if len(parts) > 2 else ""
            if "2026-12-10" not in due:
                issues.append(f"{company}: dueAt={due!r}")
            if "AI Horizons 2026" not in body and "14000" not in body:
                issues.append(f"{company}: body missing key text")

        check("15. Twenty lead tasks", 1, not issues,
              "all 4 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("15. Twenty lead tasks", 1, False, f"exception: {e}")


def check_16_twenty_note():
    """Pipeline summary note with expected body content."""
    try:
        ws = tw_schema()
        rows = twq(
            f"SELECT \"title\", \"bodyV2Markdown\" FROM {ws}.note "
            f"WHERE \"deletedAt\" IS NULL "
            f"AND \"title\" LIKE '%Sponsorship%Lead Pipeline%'"
        )
        if not rows:
            check("16. Twenty pipeline note", 2, False, "note not found")
            return
        first = rows.splitlines()[0]
        parts = first.split("|", 1)
        body = parts[1] if len(parts) > 1 else ""

        fragments = [
            "AI Horizons 2026",
            "Zenith Cloud Corp",
            "Prism Digital Solutions",
            "PLAT-AIHORIZONS-001",
            "GOLD-AIHORIZONS-001",
            "56000",
            "2026-12-10",
        ]
        missing = [f for f in fragments if f not in body]
        check("16. Twenty pipeline note", 2, not missing,
              "body matches" if not missing else f"missing: {missing}")
    except Exception as e:
        check("16. Twenty pipeline note", 2, False, f"exception: {e}")


# ── HRMS (1 check) ───────────────────────────────────────────────────────────

def check_17_hrms_job_opening():
    """HRMS: Job Opening 'Booth Staff — AI Horizons 2026' with correct details."""
    try:
        results = hrms_get(
            "Job Opening",
            filters=[["job_title", "like", "%Booth Staff%"]],
            fields=["job_title", "department", "designation", "vacancies", "status", "description"],
        )
        if not results:
            check("17. HRMS job opening", 1, False, "job opening not found")
            return
        jo = results[0]
        issues = []
        if "Booth Staff" not in (jo.get("job_title") or ""):
            issues.append(f"title={jo.get('job_title')!r}")
        if "Research & Development" not in (jo.get("department") or ""):
            issues.append(f"dept={jo.get('department')!r}")
        if (jo.get("designation") or "").lower() != "secretary":
            issues.append(f"designation={jo.get('designation')!r}")
        if int(jo.get("vacancies") or 0) != 4:
            issues.append(f"vacancies={jo.get('vacancies')}")
        if (jo.get("status") or "").lower() != "open":
            issues.append(f"status={jo.get('status')!r}")
        desc = jo.get("description") or ""
        if "badge scanning" not in desc.lower() and "lead capture" not in desc.lower():
            issues.append("description missing key responsibilities")

        check("17. HRMS job opening", 1, not issues,
              "all fields correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("17. HRMS job opening", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_pretix_event()
    check_2_pretix_products()
    check_3_pretix_quotas()
    check_4_pretix_vouchers()
    check_5_pretix_questions()
    check_6_pretix_checkin()
    check_7_pretix_tax_discount()
    check_8_bc_accounts_items()
    check_9_bc_invoice_zenith()
    check_10_bc_invoice_prism()
    check_11_bc_journal()
    check_12_twenty_companies()
    check_13_twenty_people()
    check_14_twenty_opportunities()
    check_15_twenty_tasks()
    check_16_twenty_note()
    check_17_hrms_job_opening()

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
