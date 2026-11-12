# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_062/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-062-I4: Vendor Payment Reconciliation across BigCapital, Twenty CRM, and Frappe HRMS

Checks: 16 weighted checks across bigcapital, twenty, hrms.
Strategy: docker exec (DB queries) for all three sites.
  - BigCapital: MariaDB (mysql), tenant DB discovered at runtime
  - Twenty: Postgres (psql), workspace schema discovered at runtime
  - HRMS: MariaDB (mysql), site DB discovered at runtime

Required env vars:
  SERVER_HOSTNAME, BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER,
  HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER
"""

import os
import sys
import subprocess

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

BIGCAPITAL_PORT = os.environ.get("BIGCAPITAL_PORT")
BIGCAPITAL_CONTAINER = os.environ.get("BIGCAPITAL_CONTAINER")
BIGCAPITAL_DB_CONTAINER = os.environ.get("BIGCAPITAL_DB_CONTAINER")

TWENTY_PORT = os.environ.get("TWENTY_PORT")
TWENTY_CONTAINER = os.environ.get("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = os.environ.get("TWENTY_DB_CONTAINER")

HRMS_PORT = os.environ.get("HRMS_PORT")
HRMS_CONTAINER = os.environ.get("HRMS_CONTAINER")
HRMS_DB_CONTAINER = os.environ.get("HRMS_DB_CONTAINER")

_required = [
    "BIGCAPITAL_PORT", "BIGCAPITAL_CONTAINER", "BIGCAPITAL_DB_CONTAINER",
    "TWENTY_PORT", "TWENTY_CONTAINER", "TWENTY_DB_CONTAINER",
    "HRMS_PORT", "HRMS_CONTAINER", "HRMS_DB_CONTAINER",
]
for _var in _required:
    if not os.environ.get(_var):
        print(f"FATAL: {_var} not set", file=sys.stderr)
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


# ── BigCapital: MariaDB ───────────────────────────────────────────────────────
_bc_tenant_db = ""
_bc_db_user = ""
_bc_db_pass = ""


def discover_bc_db() -> None:
    """Discover BigCapital DB credentials and tenant DB name."""
    global _bc_tenant_db, _bc_db_user, _bc_db_pass

    # Get DB credentials from the app server container
    rc, out, _ = docker_exec(BIGCAPITAL_CONTAINER, "env")
    env_vars: dict[str, str] = {}
    for line in out.split("\n"):
        if "=" in line:
            k, _, v = line.partition("=")
            env_vars[k] = v

    _bc_db_user = env_vars.get("DB_USER", "bigcapital")
    _bc_db_pass = env_vars.get("DB_PASSWORD", "bigcapital123")
    prefix = env_vars.get("TENANT_DB_NAME_PERFIX", "bigcapital_tenant_")

    # List databases and find tenant DB
    rc, out, _ = docker_exec(
        BIGCAPITAL_DB_CONTAINER,
        "mysql", f"-u{_bc_db_user}", f"-p{_bc_db_pass}",
        "-N", "-B", "-e", "SHOW DATABASES",
    )
    dbs = [d.strip() for d in out.strip().split("\n") if d.strip()]

    # Look for tenant DB by prefix
    for db in dbs:
        if db.startswith(prefix):
            _bc_tenant_db = db
            return

    # Fallback: try system DB to read tenants table
    sys_db = env_vars.get("SYSTEM_DB_NAME", "bigcapital")
    if sys_db in dbs:
        rc2, out2, _ = docker_exec(
            BIGCAPITAL_DB_CONTAINER,
            "mysql", f"-u{_bc_db_user}", f"-p{_bc_db_pass}",
            "-N", "-B", "-e", "SELECT db_name FROM tenants LIMIT 1", sys_db,
        )
        if rc2 == 0 and out2.strip():
            _bc_tenant_db = out2.strip()
            return

    # Fallback: find DB with contacts table
    for db in dbs:
        if db in ("information_schema", "mysql", "performance_schema", "sys"):
            continue
        rc3, out3, _ = docker_exec(
            BIGCAPITAL_DB_CONTAINER,
            "mysql", f"-u{_bc_db_user}", f"-p{_bc_db_pass}",
            "-N", "-B", "-e",
            "SELECT 1 FROM information_schema.TABLES "
            f"WHERE TABLE_SCHEMA='{db}' AND TABLE_NAME='CONTACTS' LIMIT 1",
        )
        if rc3 == 0 and "1" in out3:
            _bc_tenant_db = db
            return

    _bc_tenant_db = dbs[0] if dbs else "bigcapital"


def bc_sql(query: str) -> str:
    """Run SQL against BigCapital tenant DB (MariaDB). Returns tab-separated output."""
    rc, out, err = docker_exec(
        BIGCAPITAL_DB_CONTAINER,
        "mysql", f"-u{_bc_db_user}", f"-p{_bc_db_pass}",
        "--default-character-set=utf8mb4",
        "-N", "-B", "-e", query, _bc_tenant_db,
    )
    return out.strip()


def bc_parse_rows(output: str) -> list[list[str]]:
    """Parse tab-separated mysql output into rows of fields."""
    rows = []
    for line in output.split("\n"):
        line = line.strip()
        if line:
            rows.append(line.split("\t"))
    return rows


# ── Twenty: Postgres ──────────────────────────────────────────────────────────
_twenty_db = "default"
_twenty_user = "twenty"
_twenty_schema = ""


def discover_twenty_schema() -> None:
    global _twenty_db, _twenty_user, _twenty_schema
    for db, user in [("default", "twenty"), ("default", "postgres"),
                     ("twenty", "twenty"), ("twenty", "postgres")]:
        rc, out, _ = docker_exec(
            TWENTY_DB_CONTAINER,
            "psql", "-U", user, "-d", db, "-t", "-A", "-c",
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'workspace_%' LIMIT 1",
        )
        if rc == 0 and out.strip().startswith("workspace_"):
            _twenty_db = db
            _twenty_user = user
            _twenty_schema = out.strip()
            return
    _twenty_schema = "public"


def twenty_sql(query: str) -> str:
    """Run SQL against Twenty workspace schema (Postgres). Returns pipe-separated output."""
    full = f'SET search_path TO "{_twenty_schema}"; {query}'
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", _twenty_user, "-d", _twenty_db, "-t", "-A", "-c", full,
    )
    # Filter out the "SET" line that psql echoes for the SET command
    lines = [l for l in out.strip().split("\n") if l.strip() and l.strip() != "SET"]
    return "\n".join(lines)


# ── HRMS: MariaDB ────────────────────────────────────────────────────────────
_hrms_db = ""
_hrms_db_user = "root"
_hrms_db_pass = ""


def discover_hrms_db() -> None:
    """Discover HRMS MariaDB credentials and site DB name."""
    global _hrms_db, _hrms_db_user, _hrms_db_pass

    # Get root password from DB container env
    rc, out, _ = docker_exec(HRMS_DB_CONTAINER, "env")
    env_vars: dict[str, str] = {}
    for line in out.split("\n"):
        if "=" in line:
            k, _, v = line.partition("=")
            env_vars[k] = v
    _hrms_db_pass = env_vars.get("MYSQL_ROOT_PASSWORD", "")
    _hrms_db_user = "root"

    # Find DB with tabExpense Claim Type (Frappe site DB)
    rc, out, _ = docker_exec(
        HRMS_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        f"-u{_hrms_db_user}", f"-p{_hrms_db_pass}" if _hrms_db_pass else "",
        "-N", "-B", "-e",
        "SELECT TABLE_SCHEMA FROM information_schema.TABLES "
        "WHERE TABLE_NAME='tabExpense Claim Type' LIMIT 1",
    )
    if rc == 0 and out.strip():
        _hrms_db = out.strip()
        return

    # Fallback: find Frappe DB by looking for tabDocType
    rc, out, _ = docker_exec(
        HRMS_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        f"-u{_hrms_db_user}", f"-p{_hrms_db_pass}" if _hrms_db_pass else "",
        "-N", "-B", "-e",
        "SELECT TABLE_SCHEMA FROM information_schema.TABLES "
        "WHERE TABLE_NAME='tabDocType' LIMIT 1",
    )
    if rc == 0 and out.strip():
        _hrms_db = out.strip()
        return

    _hrms_db = "_frappe_bench"


def hrms_sql(query: str) -> str:
    """Run SQL against HRMS Frappe DB (MariaDB). Returns tab-separated output."""
    args = [
        "mysql", "--default-character-set=utf8mb4",
        f"-u{_hrms_db_user}",
    ]
    if _hrms_db_pass:
        args.append(f"-p{_hrms_db_pass}")
    args.extend(["-N", "-B", "-e", query, _hrms_db])
    rc, out, err = docker_exec(HRMS_DB_CONTAINER, *args)
    return out.strip()


# ── BigCapital checks ─────────────────────────────────────────────────────────

def check_1_vendors() -> None:
    """Both vendors exist with correct emails and opening balances."""
    try:
        out = bc_sql(
            "SELECT DISPLAY_NAME, EMAIL, OPENING_BALANCE "
            "FROM CONTACTS "
            "WHERE DISPLAY_NAME IN ('Vantage Systems LLC','Luminary Consulting Group') "
            "ORDER BY DISPLAY_NAME"
        )
        vendors: dict[str, dict] = {}
        for row in bc_parse_rows(out):
            if len(row) >= 3:
                vendors[row[0]] = {"email": row[1], "ob": row[2]}

        v1 = vendors.get("Vantage Systems LLC", {})
        v2 = vendors.get("Luminary Consulting Group", {})
        v1_ok = (v1.get("email") == "ap@vantagesystems.com"
                 and abs(float(v1.get("ob", 0)) - 800) < 0.01)
        v2_ok = (v2.get("email") == "billing@luminarycg.com"
                 and abs(float(v2.get("ob", 0)) - 350) < 0.01)
        check("1. Vendors exist", 2, v1_ok and v2_ok,
              f"V1={'ok' if v1_ok else v1}, V2={'ok' if v2_ok else v2}")
    except Exception as e:
        check("1. Vendors exist", 2, False, f"exception: {e}")


def check_2_items() -> None:
    """Both service items exist with correct cost prices."""
    try:
        out = bc_sql(
            "SELECT NAME, TYPE, COST_PRICE "
            "FROM ITEMS "
            "WHERE NAME IN ('ERP Implementation Services','Business Process Optimization') "
            "ORDER BY NAME"
        )
        items: dict[str, dict] = {}
        for row in bc_parse_rows(out):
            if len(row) >= 3:
                items[row[0]] = {"type": row[1], "cost": row[2]}

        i1 = items.get("ERP Implementation Services", {})
        i2 = items.get("Business Process Optimization", {})
        i1_ok = (i1.get("type", "").lower() in ("service", "services")
                 and abs(float(i1.get("cost", 0)) - 280) < 0.01)
        i2_ok = (i2.get("type", "").lower() in ("service", "services")
                 and abs(float(i2.get("cost", 0)) - 160) < 0.01)
        check("2. Service items", 2, i1_ok and i2_ok,
              f"ERP={'ok' if i1_ok else i1}, BPO={'ok' if i2_ok else i2}")
    except Exception as e:
        check("2. Service items", 2, False, f"exception: {e}")


def check_3_bill_vantage() -> None:
    """Bill for Vantage Systems: total 1160, not draft, dated 2025-10-06."""
    try:
        out = bc_sql(
            "SELECT b.AMOUNT, b.STATUS, b.BILL_DATE "
            "FROM BILLS b JOIN CONTACTS c ON b.VENDOR_ID = c.ID "
            "WHERE c.DISPLAY_NAME = 'Vantage Systems LLC' "
            "ORDER BY b.BILL_DATE LIMIT 1"
        )
        rows = bc_parse_rows(out)
        if not rows:
            check("3. Bill 1 (Vantage)", 2, False, "bill not found")
            return
        p = rows[0]
        amount = float(p[0])
        status = p[1].strip().lower()
        bdate = p[2].strip()
        ok = (abs(amount - 1160) < 0.01
              and status not in ("draft", "void", "deleted")
              and bdate.startswith("2025-10-06"))
        check("3. Bill 1 (Vantage)", 2, ok,
              f"amount={amount}, status={p[1].strip()}, date={bdate}")
    except Exception as e:
        check("3. Bill 1 (Vantage)", 2, False, f"exception: {e}")


def check_4_bill_luminary() -> None:
    """Bill for Luminary Consulting: total 800, not draft, dated 2025-10-15."""
    try:
        out = bc_sql(
            "SELECT b.AMOUNT, b.STATUS, b.BILL_DATE "
            "FROM BILLS b JOIN CONTACTS c ON b.VENDOR_ID = c.ID "
            "WHERE c.DISPLAY_NAME = 'Luminary Consulting Group' "
            "ORDER BY b.BILL_DATE LIMIT 1"
        )
        rows = bc_parse_rows(out)
        if not rows:
            check("4. Bill 2 (Luminary)", 2, False, "bill not found")
            return
        p = rows[0]
        amount = float(p[0])
        status = p[1].strip().lower()
        bdate = p[2].strip()
        ok = (abs(amount - 800) < 0.01
              and status not in ("draft", "void", "deleted")
              and bdate.startswith("2025-10-15"))
        check("4. Bill 2 (Luminary)", 2, ok,
              f"amount={amount}, status={p[1].strip()}, date={bdate}")
    except Exception as e:
        check("4. Bill 2 (Luminary)", 2, False, f"exception: {e}")


def check_5_payment_vantage() -> None:
    """Payment of 950 to Vantage Systems from Bank Account."""
    try:
        out = bc_sql(
            "SELECT bp.AMOUNT, bp.PAYMENT_DATE, a.NAME "
            "FROM BILLS_PAYMENTS bp "
            "JOIN CONTACTS c ON bp.VENDOR_ID = c.ID "
            "JOIN ACCOUNTS a ON bp.PAYMENT_ACCOUNT_ID = a.ID "
            "WHERE c.DISPLAY_NAME = 'Vantage Systems LLC' "
            "ORDER BY bp.PAYMENT_DATE LIMIT 1"
        )
        rows = bc_parse_rows(out)
        if not rows:
            check("5. Payment to Vantage", 2, False, "payment not found")
            return
        p = rows[0]
        amount = float(p[0])
        pdate = p[1].strip()
        acct = p[2].strip()
        ok = (abs(amount - 950) < 0.01
              and pdate.startswith("2025-10-21")
              and "bank account" in acct.lower())
        check("5. Payment to Vantage", 2, ok,
              f"amount={amount}, date={pdate}, account={acct}")
    except Exception as e:
        check("5. Payment to Vantage", 2, False, f"exception: {e}")


def check_6_payment_luminary() -> None:
    """Payment of 650 to Luminary Consulting from Bank Account."""
    try:
        out = bc_sql(
            "SELECT bp.AMOUNT, bp.PAYMENT_DATE, a.NAME "
            "FROM BILLS_PAYMENTS bp "
            "JOIN CONTACTS c ON bp.VENDOR_ID = c.ID "
            "JOIN ACCOUNTS a ON bp.PAYMENT_ACCOUNT_ID = a.ID "
            "WHERE c.DISPLAY_NAME = 'Luminary Consulting Group' "
            "ORDER BY bp.PAYMENT_DATE LIMIT 1"
        )
        rows = bc_parse_rows(out)
        if not rows:
            check("6. Payment to Luminary", 2, False, "payment not found")
            return
        p = rows[0]
        amount = float(p[0])
        pdate = p[1].strip()
        acct = p[2].strip()
        ok = (abs(amount - 650) < 0.01
              and pdate.startswith("2025-10-27")
              and "bank account" in acct.lower())
        check("6. Payment to Luminary", 2, ok,
              f"amount={amount}, date={pdate}, account={acct}")
    except Exception as e:
        check("6. Payment to Luminary", 2, False, f"exception: {e}")


def check_7_purchase_totals() -> None:
    """Purchase totals: ERP=840, BPO=1120 from items_entries."""
    try:
        out = bc_sql(
            "SELECT i.NAME, SUM(ie.QUANTITY * ie.RATE) AS total "
            "FROM ITEMS_ENTRIES ie "
            "JOIN ITEMS i ON ie.ITEM_ID = i.ID "
            "WHERE LOWER(ie.REFERENCE_TYPE) = 'bill' "
            "AND i.NAME IN ('ERP Implementation Services','Business Process Optimization') "
            "GROUP BY i.NAME"
        )
        totals: dict[str, float] = {}
        for row in bc_parse_rows(out):
            if len(row) >= 2:
                totals[row[0]] = float(row[1])
        erp = totals.get("ERP Implementation Services", 0)
        bpo = totals.get("Business Process Optimization", 0)
        ok = abs(erp - 840) < 0.01 and abs(bpo - 1120) < 0.01
        check("7. Purchase totals by item", 2, ok, f"ERP={erp}, BPO={bpo}")
    except Exception as e:
        check("7. Purchase totals by item", 2, False, f"exception: {e}")


def check_8_expense() -> None:
    """Expense 210 published with reference MISC-EXP-2025-004."""
    try:
        out = bc_sql(
            "SELECT TOTAL_AMOUNT, REFERENCE_NO, PUBLISHED_AT "
            "FROM EXPENSES_TRANSACTIONS "
            "WHERE REFERENCE_NO = 'MISC-EXP-2025-004'"
        )
        rows = bc_parse_rows(out)
        if not rows:
            check("8. Expense published", 1, False, "expense not found")
            return
        p = rows[0]
        amount = float(p[0])
        ref = p[1].strip()
        published = p[2].strip() if len(p) > 2 else ""
        ok = abs(amount - 210) < 0.01 and ref == "MISC-EXP-2025-004" and published not in ("", "NULL")
        check("8. Expense published", 1, ok,
              f"amount={amount}, ref={ref}, published={'yes' if published not in ('', 'NULL') else 'no'}")
    except Exception as e:
        check("8. Expense published", 1, False, f"exception: {e}")


def check_9_bank_rule() -> None:
    """Bank rule 'Cost of Goods Auto-Categorization' with 'cogs' condition."""
    try:
        out = bc_sql(
            "SELECT br.NAME, a.NAME AS acct "
            "FROM BANK_RULES br "
            "LEFT JOIN ACCOUNTS a ON br.ASSIGN_ACCOUNT_ID = a.ID "
            "WHERE br.NAME = 'Cost of Goods Auto-Categorization'"
        )
        rows = bc_parse_rows(out)
        if not rows:
            check("9. Bank rule", 2, False, "bank rule not found")
            return
        acct = rows[0][1].strip() if len(rows[0]) > 1 else ""
        acct_ok = "cost of goods sold" in acct.lower()

        cond = bc_sql(
            "SELECT brc.VALUE "
            "FROM BANK_RULE_CONDITIONS brc "
            "JOIN BANK_RULES br ON brc.RULE_ID = br.ID "
            "WHERE br.NAME = 'Cost of Goods Auto-Categorization'"
        )
        cond_ok = "cogs" in cond.lower()
        check("9. Bank rule", 2, acct_ok and cond_ok,
              f"account={acct}, cond_has_cogs={cond_ok}")
    except Exception as e:
        check("9. Bank rule", 2, False, f"exception: {e}")


def check_10_gl_credits() -> None:
    """General ledger credits of 950 and 650 for Bank Account."""
    try:
        out = bc_sql(
            "SELECT at2.CREDIT "
            "FROM ACCOUNTS_TRANSACTIONS at2 "
            "JOIN ACCOUNTS a ON at2.ACCOUNT_ID = a.ID "
            "WHERE a.NAME = 'Bank Account' AND at2.CREDIT > 0 "
            "ORDER BY at2.CREDIT"
        )
        credits: list[float] = []
        for row in bc_parse_rows(out):
            try:
                credits.append(float(row[0]))
            except (ValueError, IndexError):
                pass
        has_950 = any(abs(c - 950) < 0.01 for c in credits)
        has_650 = any(abs(c - 650) < 0.01 for c in credits)
        check("10. GL credits Bank Account", 3, has_950 and has_650,
              f"credits={credits}, has_950={has_950}, has_650={has_650}")
    except Exception as e:
        check("10. GL credits Bank Account", 3, False, f"exception: {e}")


# ── Twenty CRM checks ────────────────────────────────────────────────────────

def check_11_twenty_companies() -> None:
    """Both companies exist in Twenty with correct domains."""
    try:
        rows = twenty_sql(
            "SELECT name, \"domainNamePrimaryLinkUrl\" "
            "FROM company "
            "WHERE name IN ('Vantage Systems LLC','Luminary Consulting Group')"
        )
        comps: dict[str, str] = {}
        for line in rows.split("\n"):
            if "|" in line:
                p = line.split("|")
                comps[p[0]] = p[1].strip()
        v1_ok = "vantagesystems.com" in comps.get("Vantage Systems LLC", "")
        v2_ok = "luminarycg.com" in comps.get("Luminary Consulting Group", "")
        check("11. Twenty companies", 2, v1_ok and v2_ok,
              f"V1_domain={comps.get('Vantage Systems LLC', 'missing')}, "
              f"V2_domain={comps.get('Luminary Consulting Group', 'missing')}")
    except Exception as e:
        check("11. Twenty companies", 2, False, f"exception: {e}")


def check_12_twenty_persons() -> None:
    """Both persons exist with correct emails, titles, and company links."""
    try:
        rows = twenty_sql(
            "SELECT p.\"nameFirstName\", p.\"nameLastName\", "
            "p.\"emailsPrimaryEmail\", p.\"jobTitle\", c.name AS company "
            "FROM person p "
            "LEFT JOIN company c ON p.\"companyId\" = c.id "
            "WHERE p.\"emailsPrimaryEmail\" IN "
            "('ap@vantagesystems.com','billing@luminarycg.com')"
        )
        persons: dict[str, dict] = {}
        for line in rows.split("\n"):
            if "|" in line:
                p = line.split("|")
                email = p[2].strip()
                persons[email] = {
                    "name": f"{p[0].strip()} {p[1].strip()}",
                    "title": p[3].strip(),
                    "company": p[4].strip() if len(p) > 4 else "",
                }

        p1 = persons.get("ap@vantagesystems.com", {})
        p2 = persons.get("billing@luminarycg.com", {})
        p1_ok = ("harrison" in p1.get("name", "").lower()
                 and "blake" in p1.get("name", "").lower()
                 and "vendor relations" in p1.get("title", "").lower()
                 and "vantage" in p1.get("company", "").lower())
        p2_ok = ("celeste" in p2.get("name", "").lower()
                 and "moreau" in p2.get("name", "").lower()
                 and "senior consultant" in p2.get("title", "").lower()
                 and "luminary" in p2.get("company", "").lower())
        check("12. Twenty persons", 2, p1_ok and p2_ok,
              f"Harrison={'ok' if p1_ok else p1}, Celeste={'ok' if p2_ok else p2}")
    except Exception as e:
        check("12. Twenty persons", 2, False, f"exception: {e}")


def check_13_twenty_note() -> None:
    """Reconciliation note exists with correct title and body content."""
    try:
        rows = twenty_sql(
            "SELECT title, body FROM note "
            "WHERE title LIKE '%Vendor Payment Reconciliation%2025-10-27%'"
        )
        if not rows:
            check("13. Twenty note", 2, False, "note not found")
            return
        first = rows.split("\n")[0]
        if "|" not in first:
            check("13. Twenty note", 2, False, f"unexpected format: {first[:80]}")
            return
        parts = first.split("|", 1)
        title = parts[0].strip()
        body = parts[1] if len(parts) > 1 else ""

        title_ok = "Vendor Payment Reconciliation" in title and "2025-10-27" in title
        body_hits = sum([
            "1160" in body,
            "950" in body,
            "1010" in body,
            "800" in body,
            "650" in body,
            "500" in body,
            "210" in body,
            "840" in body,
            "1120" in body,
            "Cost of Goods Auto-Categorization" in body,
        ])
        body_ok = body_hits >= 8
        check("13. Twenty note", 2, title_ok and body_ok,
              f"title_ok={title_ok}, body_hits={body_hits}/10")
    except Exception as e:
        check("13. Twenty note", 2, False, f"exception: {e}")


def check_14_twenty_favorites() -> None:
    """Vantage Systems LLC is in favorites."""
    try:
        row = twenty_sql(
            "SELECT f.id FROM favorite f "
            "JOIN company c ON f.\"companyId\" = c.id "
            "WHERE c.name = 'Vantage Systems LLC' LIMIT 1"
        )
        ok = bool(row.strip())
        check("14. Vantage in favorites", 1, ok,
              "found" if ok else "not in favorites")
    except Exception as e:
        check("14. Vantage in favorites", 1, False, f"exception: {e}")


# ── HRMS checks ───────────────────────────────────────────────────────────────

def check_15_expense_types() -> None:
    """Expense claim types 'Calls' and 'Food' with correct descriptions."""
    try:
        out = hrms_sql(
            "SELECT name, description FROM `tabExpense Claim Type` "
            "WHERE name IN ('Calls','Food')"
        )
        types: dict[str, str] = {}
        for line in out.split("\n"):
            if "\t" in line:
                p = line.split("\t", 1)
                types[p[0]] = p[1] if len(p) > 1 else ""
        calls_ok = "Linked to BigCapital account: Cost of Goods Sold" in types.get("Calls", "")
        food_ok = "Linked to BigCapital account: Advertising Expense" in types.get("Food", "")
        check("15. HRMS expense types", 2, calls_ok and food_ok,
              f"Calls={'ok' if calls_ok else types.get('Calls', 'missing')}, "
              f"Food={'ok' if food_ok else types.get('Food', 'missing')}")
    except Exception as e:
        check("15. HRMS expense types", 2, False, f"exception: {e}")


def check_16_expense_claim() -> None:
    """Submitted expense claim for Deepika Joshi, amount 210, type Calls."""
    try:
        out = hrms_sql(
            "SELECT ec.employee_name, ec.posting_date, ec.total_claimed_amount, "
            "ec.docstatus, ecd.expense_type, ecd.amount, ecd.description "
            "FROM `tabExpense Claim` ec "
            "JOIN `tabExpense Claim Detail` ecd ON ecd.parent = ec.name "
            "WHERE ec.employee_name = 'Deepika Joshi' "
            "AND ec.posting_date = '2025-10-29'"
        )
        if not out:
            check("16. HRMS expense claim", 2, False, "claim not found")
            return
        first = out.split("\n")[0]
        p = first.split("\t")
        emp = p[0] if len(p) > 0 else ""
        pdate = p[1] if len(p) > 1 else ""
        docstatus = int(p[3]) if len(p) > 3 and p[3] else 0
        etype = p[4] if len(p) > 4 else ""
        amount = float(p[5]) if len(p) > 5 and p[5] else 0
        desc = p[6] if len(p) > 6 else ""

        ok = (emp == "Deepika Joshi"
              and "2025-10-29" in pdate
              and abs(amount - 210) < 0.01
              and docstatus == 1
              and etype == "Calls"
              and "MISC-EXP-2025-004" in desc)
        check("16. HRMS expense claim", 2, ok,
              f"emp={emp}, date={pdate}, amt={amount}, "
              f"docstatus={docstatus}, type={etype}, has_ref={'MISC-EXP-2025-004' in desc}")
    except Exception as e:
        check("16. HRMS expense claim", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Discovering BigCapital DB...", file=sys.stderr)
    discover_bc_db()
    print(f"  -> tenant DB: {_bc_tenant_db}, user: {_bc_db_user}", file=sys.stderr)

    print("Discovering Twenty workspace schema...", file=sys.stderr)
    discover_twenty_schema()
    print(f"  -> DB={_twenty_db}, user={_twenty_user}, schema={_twenty_schema}",
          file=sys.stderr)

    print("Discovering HRMS DB...", file=sys.stderr)
    discover_hrms_db()
    print(f"  -> DB: {_hrms_db}, user: {_hrms_db_user}", file=sys.stderr)

    check_1_vendors()
    check_2_items()
    check_3_bill_vantage()
    check_4_bill_luminary()
    check_5_payment_vantage()
    check_6_payment_luminary()
    check_7_purchase_totals()
    check_8_expense()
    check_9_bank_rule()
    check_10_gl_credits()
    check_11_twenty_companies()
    check_12_twenty_persons()
    check_13_twenty_note()
    check_14_twenty_favorites()
    check_15_expense_types()
    check_16_expense_claim()

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
