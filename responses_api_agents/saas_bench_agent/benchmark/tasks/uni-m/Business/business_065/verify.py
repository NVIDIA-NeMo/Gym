# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_065/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-065-I3: FY2026 Financial Audit Preparation Across BigCapital, HRMS, and Twenty CRM

Checks: 15 weighted checks across bigcapital, hrms, twenty.
Strategy: API for BigCapital, docker exec DB for HRMS (MariaDB) and Twenty (Postgres).

Required env vars:
  SERVER_HOSTNAME, BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER
"""

import os
import sys
import subprocess
import json
import re

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"FATAL: {name} not set", file=sys.stderr)
        sys.exit(1)
    return val


BC_PORT = _require("BIGCAPITAL_PORT")
BC_CONTAINER = _require("BIGCAPITAL_CONTAINER")
BC_DB_CONTAINER = _require("BIGCAPITAL_DB_CONTAINER")
HRMS_PORT = _require("HRMS_PORT")
HRMS_CONTAINER = _require("HRMS_CONTAINER")
HRMS_DB_CONTAINER = _require("HRMS_DB_CONTAINER")
TWENTY_PORT = _require("TWENTY_PORT")
TWENTY_CONTAINER = _require("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = _require("TWENTY_DB_CONTAINER")

BC_BASE = f"http://{HOST}:{BC_PORT}"

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


# -- BigCapital API helpers --
_bc_token: str | None = None


def bc_login() -> str:
    global _bc_token
    if _bc_token:
        return _bc_token
    if not HAS_REQUESTS:
        raise RuntimeError("requests module not available")
    r = requests.post(
        f"{BC_BASE}/api/auth/login",
        json={"email": "admin@bigcapital.local", "password": "admin123"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    # BigCapital may return token at top-level or nested under "data"
    _bc_token = data.get("token") or data.get("data", {}).get("token", "")
    if not _bc_token:
        raise RuntimeError(f"no token in login response: {json.dumps(data)[:200]}")
    return _bc_token


def bc_get(path: str, params: dict | None = None) -> requests.Response:
    token = bc_login()
    return requests.get(
        f"{BC_BASE}/api/{path}",
        params=params,
        headers={"Authorization": f"Bearer {token}", "x-access-token": token},
        timeout=15,
    )


# -- BigCapital DB helper (tries mysql then psql) --
def bc_db(sql: str) -> str:
    """Query BigCapital DB — tries mysql first, then psql."""
    # Try MySQL (MariaDB) — actual BigCapital engine per source code
    # Credentials: bigcapital / bigcapital123, tenant DBs prefixed 'bigcapital_tenant_'
    rc, out, err = docker_exec(
        BC_DB_CONTAINER, "mysql", "-u", "bigcapital", "-pbigcapital123", "-N", "-B", "-e",
        "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA "
        "WHERE SCHEMA_NAME LIKE 'bigcapital%' OR SCHEMA_NAME NOT IN "
        "('information_schema','performance_schema','mysql','sys','RECOVER_YOUR_DATA') LIMIT 10",
    )
    if rc == 0:
        dbs = [d.strip() for d in out.strip().split("\n") if d.strip()]
        # Prefer tenant DBs
        tenant_dbs = [d for d in dbs if "tenant" in d]
        search_order = tenant_dbs + [d for d in dbs if d not in tenant_dbs]
        for dbname in search_order:
            rc2, out2, _ = docker_exec(
                BC_DB_CONTAINER, "mysql", "-u", "bigcapital", "-pbigcapital123",
                "-D", dbname, "-N", "-B", "-e", sql,
            )
            if rc2 == 0 and out2.strip():
                return out2.strip()
        return ""
    # Fallback: try Postgres
    rc, out, err = docker_exec(
        BC_DB_CONTAINER, "psql", "-U", "postgres", "-t", "-A", "-c",
        "SELECT datname FROM pg_database WHERE datname NOT IN ('postgres','template0','template1') LIMIT 5",
    )
    if rc == 0:
        dbs = [d.strip() for d in out.strip().split("\n") if d.strip()]
        for dbname in dbs:
            rc2, out2, _ = docker_exec(
                BC_DB_CONTAINER, "psql", "-U", "postgres", "-d", dbname, "-t", "-A", "-c", sql
            )
            if rc2 == 0 and out2.strip():
                return out2.strip()
    return ""


# -- Twenty DB helpers --
_twenty_schema: str | None = None


def twenty_db(sql: str) -> str:
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER, "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c", sql
    )
    if rc != 0:
        rc, out, err = docker_exec(
            TWENTY_DB_CONTAINER, "psql", "-U", "postgres", "-d", "twenty", "-t", "-A", "-c", sql
        )
    return out.strip()


def twenty_schema() -> str:
    global _twenty_schema
    if _twenty_schema is not None:
        return _twenty_schema
    result = twenty_db(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'workspace_%' ORDER BY schema_name LIMIT 1"
    )
    _twenty_schema = result.split("\n")[0].strip() if result.strip() else ""
    return _twenty_schema


# -- HRMS DB helper --
_hrms_db_creds: dict | None = None


def _hrms_creds() -> tuple[str, str, str]:
    """Read HRMS DB credentials from site_config.json inside the app container."""
    global _hrms_db_creds
    if _hrms_db_creds is not None:
        return _hrms_db_creds["user"], _hrms_db_creds["pass"], _hrms_db_creds["db"]
    # Try to read from the app container's site config
    rc, out, err = docker_exec(
        HRMS_CONTAINER, "bash", "-c",
        "cat /home/frappe/frappe-bench/sites/*/site_config.json 2>/dev/null | head -20",
    )
    if rc == 0 and out.strip():
        try:
            cfg = json.loads(out.strip().split("\n{")[0] if "\n{" in out else out.strip())
            _hrms_db_creds = {
                "user": cfg.get("db_user", "root"),
                "pass": cfg.get("db_password", ""),
                "db": cfg.get("db_name", "_frappe_bench"),
            }
            return _hrms_db_creds["user"], _hrms_db_creds["pass"], _hrms_db_creds["db"]
        except json.JSONDecodeError:
            pass
    # Fallback defaults
    _hrms_db_creds = {"user": "root", "pass": "", "db": "_frappe_bench"}
    return "root", "", "_frappe_bench"


def hrms_db(sql: str) -> str:
    user, password, db = _hrms_creds()
    args = ["mysql", "-u", user, "--default-character-set=utf8mb4", "-D", db, "-N", "-B"]
    if password:
        args.insert(3, f"-p{password}")
    args.extend(["-e", sql])
    rc, out, err = docker_exec(HRMS_DB_CONTAINER, *args)
    return out.strip()


# ── Note body cache (fetched once, used by many checks) ──────────────────────
_note_body: str | None = None
_note_found: bool = False


def _fetch_note() -> None:
    global _note_body, _note_found
    if _note_body is not None:
        return
    schema = twenty_schema()
    if not schema:
        _note_body = ""
        return
    # Try exact title (em dash)
    result = twenty_db(
        f"""SELECT body FROM "{schema}".note """
        f"""WHERE title LIKE '%Audit Preparation Package%FY 2026%' LIMIT 1"""
    )
    if result:
        _note_body = result
        _note_found = True
        return
    # Broader search
    result = twenty_db(
        f"""SELECT body FROM "{schema}".note """
        f"""WHERE title ILIKE '%audit%preparation%package%' LIMIT 1"""
    )
    if result:
        _note_body = result
        _note_found = True
    else:
        _note_body = ""


def _body_has(pattern: str) -> bool:
    _fetch_note()
    return bool(re.search(pattern, _note_body or "", re.IGNORECASE))


def _body_has_amount_near(section_pattern: str) -> bool:
    """Check if the note body has a numeric value near a section header."""
    _fetch_note()
    if not _note_body:
        return False
    m = re.search(section_pattern, _note_body, re.IGNORECASE)
    if not m:
        return False
    snippet = _note_body[m.start() : m.start() + 500]
    return bool(re.search(r"\$?\d[\d,]*\.?\d*", snippet))


# ── Individual checks ─────────────────────────────────────────────────────────


def check_1_transaction_lock() -> None:
    """BigCapital: Transactions locked before 2027-01-01."""
    try:
        found = False
        # Strategy A: API
        if HAS_REQUESTS:
            try:
                resp = bc_get("transactions-locking")
                if resp.status_code == 200:
                    body = resp.text
                    if "2027-01-01" in body or "2026-12-31" in body:
                        found = True
            except Exception:
                pass
            if not found:
                try:
                    resp = bc_get("settings")
                    if resp.status_code == 200:
                        body = resp.text
                        if "2027-01-01" in body or "2026-12-31" in body:
                            found = True
                except Exception:
                    pass
        # Strategy B: DB — query settings table for locking group
        if not found:
            result = bc_db(
                "SELECT `value` FROM `settings` "
                "WHERE `group`='transactions-locking' "
                "AND `key` LIKE '%lock_to_date%' "
                "LIMIT 5"
            )
            if "2027-01-01" in result or "2026-12-31" in result:
                found = True
        # Strategy C: DB — broader search in settings
        if not found:
            result = bc_db(
                "SELECT `key`, `value` FROM `settings` "
                "WHERE `value` LIKE '%2027%' OR `value` LIKE '%2026-12-31%' "
                "LIMIT 10"
            )
            if result:
                found = True
        check("1. Transaction lock", 2, found,
              "lock date found" if found else "no lock date found in API or DB")
    except Exception as e:
        check("1. Transaction lock", 2, False, f"exception: {e}")


def check_2_bank_balance() -> None:
    """BigCapital: Bank Account GL closing balance = -$215,382.44."""
    try:
        found_balance = None
        # Strategy A: DB — compute from accounts_transactions
        # First find the account id for Bank Account
        acct_row = bc_db(
            "SELECT `id`, `account_normal` FROM `accounts` "
            "WHERE `name`='Bank Account' LIMIT 1"
        )
        if acct_row:
            parts = acct_row.split("\t")
            acct_id = parts[0].strip()
            acct_normal = parts[1].strip() if len(parts) > 1 else "debit"
            # Sum transactions up to 2026-12-31
            sums = bc_db(
                f"SELECT COALESCE(SUM(`debit`),0), COALESCE(SUM(`credit`),0) "
                f"FROM `accounts_transactions` "
                f"WHERE `account_id`={acct_id} AND `date` <= '2026-12-31'"
            )
            if sums:
                vals = sums.split("\t")
                total_debit = float(vals[0]) if vals[0] else 0.0
                total_credit = float(vals[1]) if len(vals) > 1 and vals[1] else 0.0
                if "debit" in acct_normal.lower():
                    found_balance = total_debit - total_credit
                else:
                    found_balance = total_credit - total_debit
        # Strategy B: API — try general ledger
        if found_balance is None and HAS_REQUESTS:
            try:
                resp = bc_get(
                    "financial-statements/general-ledger",
                    params={"fromDate": "2026-01-01", "toDate": "2026-12-31"},
                )
                if resp.status_code == 200 and "215382" in resp.text:
                    found_balance = -215382.44
            except Exception:
                pass

        if found_balance is not None:
            expected = -215382.44
            ok = abs(found_balance - expected) < 0.50
            check("2. Bank Account GL balance", 2, ok,
                  f"expected={expected}, got={found_balance:.2f}")
        else:
            check("2. Bank Account GL balance", 2, False, "could not retrieve balance")
    except Exception as e:
        check("2. Bank Account GL balance", 2, False, f"exception: {e}")


def check_3_hrms_employees() -> None:
    """HRMS: Active employees exist for TechVista Solutions Pvt. Ltd."""
    try:
        result = hrms_db(
            "SELECT COUNT(*) FROM `tabEmployee` "
            "WHERE `company`='TechVista Solutions Pvt. Ltd.' AND `status`='Active'"
        )
        count = int(result.strip()) if result.strip().isdigit() else 0
        check("3. HRMS active employees", 1, count > 0, f"count={count}")
    except Exception as e:
        check("3. HRMS active employees", 1, False, f"exception: {e}")


def check_4_note_exists() -> None:
    """Twenty: Note 'Audit Preparation Package — FY 2026' exists."""
    try:
        _fetch_note()
        check("4. Audit note exists", 2, _note_found,
              "found" if _note_found else "note not found")
    except Exception as e:
        check("4. Audit note exists", 2, False, f"exception: {e}")


def check_5_trial_balance() -> None:
    """Twenty: Note body contains Trial Balance data with debits/credits."""
    try:
        has_section = _body_has(r"trial\s*balance")
        has_debits = _body_has(r"debit")
        has_credits = _body_has(r"credit")
        has_amounts = _body_has_amount_near(r"trial\s*balance")
        ok = has_section and has_debits and has_credits and has_amounts
        check("5. Note: Trial Balance data", 2, ok,
              f"section={has_section}, debits={has_debits}, credits={has_credits}, amounts={has_amounts}")
    except Exception as e:
        check("5. Note: Trial Balance data", 2, False, f"exception: {e}")


def check_6_balance_sheet() -> None:
    """Twenty: Note body contains Balance Sheet with assets/liabilities/equity."""
    try:
        has_section = _body_has(r"balance\s*sheet")
        has_assets = _body_has(r"assets")
        has_liab = _body_has(r"liabilit")
        has_equity = _body_has(r"equity")
        has_amounts = _body_has_amount_near(r"balance\s*sheet")
        ok = has_section and has_assets and has_liab and has_equity and has_amounts
        check("6. Note: Balance Sheet data", 2, ok,
              f"section={has_section}, assets={has_assets}, liab={has_liab}, equity={has_equity}")
    except Exception as e:
        check("6. Note: Balance Sheet data", 2, False, f"exception: {e}")


def check_7_pnl() -> None:
    """Twenty: Note body contains P&L with revenue/expenses/net income."""
    try:
        has_section = _body_has(r"(p\s*[&/]\s*l|profit.{0,10}loss|income\s*statement)")
        has_revenue = _body_has(r"revenue")
        has_expenses = _body_has(r"expense")
        has_net = _body_has(r"net\s*(income|profit|loss)")
        has_amounts = _body_has_amount_near(r"(p\s*[&/]\s*l|profit|revenue)")
        ok = has_section and has_revenue and has_expenses and has_net and has_amounts
        check("7. Note: P&L data", 2, ok,
              f"section={has_section}, revenue={has_revenue}, expenses={has_expenses}, net={has_net}")
    except Exception as e:
        check("7. Note: P&L data", 2, False, f"exception: {e}")


def check_8_cash_flow() -> None:
    """Twenty: Note body contains Cash Flow operating activities data."""
    try:
        has_section = _body_has(r"cash\s*flow")
        has_operating = _body_has(r"operat")
        has_amounts = _body_has_amount_near(r"cash\s*flow")
        ok = has_section and has_operating and has_amounts
        check("8. Note: Cash Flow data", 1, ok,
              f"section={has_section}, operating={has_operating}, amounts={has_amounts}")
    except Exception as e:
        check("8. Note: Cash Flow data", 1, False, f"exception: {e}")


def check_9_gl_closing_balance() -> None:
    """Twenty: Note body contains GL closing balance -$215,382.44."""
    try:
        has_gl = _body_has(r"(general\s*ledger|bank\s*account)")
        has_balance = _body_has(r"215[,.]?382[.]?44")
        ok = has_gl and has_balance
        check("9. Note: GL closing balance -$215,382.44", 2, ok,
              f"GL_ref={has_gl}, balance_value={has_balance}")
    except Exception as e:
        check("9. Note: GL closing balance -$215,382.44", 2, False, f"exception: {e}")


def check_10_ar_ap() -> None:
    """Twenty: Note body contains A/R and A/P totals."""
    try:
        has_ar = _body_has(r"(a\s*/\s*r|accounts?\s*receivable|receivable)")
        has_ap = _body_has(r"(a\s*/\s*p|accounts?\s*payable|payable)")
        has_amounts = (
            _body_has_amount_near(r"(a\s*/\s*r|receivable)")
            and _body_has_amount_near(r"(a\s*/\s*p|payable)")
        )
        ok = has_ar and has_ap and has_amounts
        check("10. Note: A/R and A/P data", 1, ok,
              f"AR={has_ar}, AP={has_ap}, amounts={has_amounts}")
    except Exception as e:
        check("10. Note: A/R and A/P data", 1, False, f"exception: {e}")


def check_11_hr_payroll() -> None:
    """Twenty: Note body contains employee count and payroll data."""
    try:
        has_employees = _body_has(r"(active\s*employee|employee.*\d)")
        has_gross = _body_has(r"gross")
        has_deductions = _body_has(r"deduction")
        has_net_pay = _body_has(r"net\s*pay")
        ok = has_employees and has_gross and has_deductions and has_net_pay
        check("11. Note: HR payroll data", 2, ok,
              f"employees={has_employees}, gross={has_gross}, deduct={has_deductions}, net={has_net_pay}")
    except Exception as e:
        check("11. Note: HR payroll data", 2, False, f"exception: {e}")


def check_12_tax_pf() -> None:
    """Twenty: Note body contains income tax and PF deduction amounts."""
    try:
        has_tax = _body_has(r"income\s*tax")
        has_pf = _body_has(r"(p\.?f\.?\s*deduction|provident\s*fund)")
        ok = has_tax and has_pf
        check("12. Note: Tax and PF data", 1, ok,
              f"tax={has_tax}, pf={has_pf}")
    except Exception as e:
        check("12. Note: Tax and PF data", 1, False, f"exception: {e}")


def check_13_leave_advance() -> None:
    """Twenty: Note body contains leave balance and advance info."""
    try:
        has_leave = _body_has(r"(leave|sick\s*leave)")
        has_advance = _body_has(r"advance")
        ok = has_leave and has_advance
        check("13. Note: Leave and advance info", 1, ok,
              f"leave={has_leave}, advance={has_advance}")
    except Exception as e:
        check("13. Note: Leave and advance info", 1, False, f"exception: {e}")


def check_14_won_deals() -> None:
    """Twenty: Note body contains Won deal count and revenue for FY 2026."""
    try:
        has_won = _body_has(r"won")
        has_deals = _body_has(r"deal")
        has_revenue_section = _body_has(r"(crm\s*revenue|won\s*deal)")
        has_amounts = _body_has_amount_near(r"(won|crm\s*revenue)")
        ok = has_won and has_deals and has_revenue_section and has_amounts
        check("14. Note: Won deals data", 2, ok,
              f"won={has_won}, deals={has_deals}, section={has_revenue_section}, amounts={has_amounts}")
    except Exception as e:
        check("14. Note: Won deals data", 2, False, f"exception: {e}")


def check_15_task() -> None:
    """Twenty: Task 'Submit audit package to external auditors — FY 2026' with due 2027-03-15."""
    try:
        schema = twenty_schema()
        if not schema:
            check("15. Audit submission task", 2, False, "workspace schema not found")
            return
        result = twenty_db(
            f"""SELECT "dueAt" FROM "{schema}".task """
            f"""WHERE title LIKE '%Submit audit package%external auditor%FY 2026%' LIMIT 1"""
        )
        if not result:
            result = twenty_db(
                f"""SELECT "dueAt" FROM "{schema}".task """
                f"""WHERE title ILIKE '%audit package%auditor%' LIMIT 1"""
            )
        if result:
            has_date = "2027-03-15" in result
            check("15. Audit submission task", 2, has_date,
                  f"found, due_date_match={has_date}, raw={result[:100]}")
        else:
            check("15. Audit submission task", 2, False, "task not found")
    except Exception as e:
        check("15. Audit submission task", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_transaction_lock()
    check_2_bank_balance()
    check_3_hrms_employees()
    check_4_note_exists()
    check_5_trial_balance()
    check_6_balance_sheet()
    check_7_pnl()
    check_8_cash_flow()
    check_9_gl_closing_balance()
    check_10_ar_ap()
    check_11_hr_payroll()
    check_12_tax_pf()
    check_13_leave_advance()
    check_14_won_deals()
    check_15_task()

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
