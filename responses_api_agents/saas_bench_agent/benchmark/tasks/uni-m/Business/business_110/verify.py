#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_110/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-110-I5: New Employee Onboarding (HRMS + BigCapital + Twenty)

Checks: 19 weighted checks across hrms, bigcapital, twenty.
Strategy: docker exec MariaDB (HRMS), REST API (BigCapital), docker exec Postgres (Twenty)

Required env vars:
  SERVER_HOSTNAME, HRMS_DB_CONTAINER, BIGCAPITAL_PORT, TWENTY_DB_CONTAINER
"""

import os
import sys
import subprocess
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")
HRMS_DB = os.getenv("HRMS_DB_CONTAINER")
BC_PORT = os.getenv("BIGCAPITAL_PORT")
TWENTY_DB = os.getenv("TWENTY_DB_CONTAINER")

for var in ("HRMS_DB_CONTAINER", "BIGCAPITAL_PORT", "TWENTY_DB_CONTAINER"):
    if not os.getenv(var):
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


_hrms_db_name: str | None = None


def _hrms_db() -> str:
    """Auto-detect the Frappe bench DB name (contains tabEmployee)."""
    global _hrms_db_name
    if _hrms_db_name is None:
        rc, out, err = docker_exec(
            HRMS_DB, "mysql", "-u", "root", "-phrms123456",
            "--default-character-set=utf8mb4", "-N", "-e",
            "SELECT TABLE_SCHEMA FROM information_schema.TABLES "
            "WHERE TABLE_NAME = 'tabEmployee' LIMIT 1;",
        )
        if rc != 0:
            raise RuntimeError(f"mysql db detection: {err.strip()}")
        _hrms_db_name = out.strip().split("\n")[0].strip()
        if not _hrms_db_name:
            raise RuntimeError("no Frappe bench DB found (no tabEmployee table)")
    return _hrms_db_name


def hrms_sql(query: str) -> str:
    """Query Frappe HRMS MariaDB."""
    rc, out, err = docker_exec(
        HRMS_DB, "mysql", "-u", "root", "-phrms123456",
        "--default-character-set=utf8mb4",
        _hrms_db(), "-N", "-e", query,
    )
    if rc != 0:
        raise RuntimeError(f"mysql: {err.strip()}")
    return out.strip()


def twenty_sql(query: str) -> str:
    """Query Twenty CRM Postgres."""
    rc, out, err = docker_exec(
        TWENTY_DB, "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c", query,
    )
    if rc != 0:
        raise RuntimeError(f"psql: {err.strip()}")
    return out.strip()


_ws: str | None = None


def ws_schema() -> str:
    """Get Twenty workspace schema name."""
    global _ws
    if _ws is None:
        r = twenty_sql(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'workspace_%' LIMIT 1;"
        )
        if not r:
            raise RuntimeError("no workspace schema found")
        _ws = r.split("\n")[0].strip()
    return _ws


_bc_session: tuple[str, requests.Session] | None = None


def bc() -> tuple[str, requests.Session]:
    """Get authenticated BigCapital API session."""
    global _bc_session
    if _bc_session is not None:
        return _bc_session
    base = f"http://{HOST}:{BC_PORT}"
    s = requests.Session()
    s.headers["Content-Type"] = "application/json"
    resp = s.post(
        f"{base}/api/auth/signin",
        json={"email": "admin@bigcapital.local", "password": "admin123"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token", "")
    org_id = data.get("organization_id", "")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    if org_id:
        s.headers["organization-id"] = str(org_id)
    _bc_session = (base, s)
    return _bc_session


_bc_accounts_cache: list[dict] | None = None


def bc_accounts() -> list[dict]:
    """Fetch BigCapital accounts (cached)."""
    global _bc_accounts_cache
    if _bc_accounts_cache is not None:
        return _bc_accounts_cache
    base, s = bc()
    r = s.get(f"{base}/api/accounts", timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        _bc_accounts_cache = data
    elif "accounts" in data:
        _bc_accounts_cache = data["accounts"]
    elif "data" in data:
        d = data["data"]
        _bc_accounts_cache = d if isinstance(d, list) else d.get("accounts", [])
    else:
        _bc_accounts_cache = []
    return _bc_accounts_cache


def bc_find_account(name: str) -> dict | None:
    for a in bc_accounts():
        if a.get("name") == name:
            return a
    return None


def bc_account_name(account_id) -> str:
    for a in bc_accounts():
        if a.get("id") == account_id:
            return a.get("name", "")
    return ""


# ── HRMS Checks ───────────────────────────────────────────────────────────────

def check_1_department() -> None:
    """Department 'Machine Learning - TVS' with parent and approvers."""
    try:
        dept = hrms_sql(
            "SELECT parent_department FROM `tabDepartment` "
            "WHERE name='Machine Learning - TVS' LIMIT 1;"
        )
        if not dept:
            check("1. Department", 2, False, "not found")
            return
        parent_ok = "All Departments" in dept
        la = hrms_sql(
            "SELECT approver FROM `tabDepartment Approver` "
            "WHERE parent='Machine Learning - TVS' AND parentfield='leave_approvers';"
        )
        ea = hrms_sql(
            "SELECT approver FROM `tabDepartment Approver` "
            "WHERE parent='Machine Learning - TVS' AND parentfield='expense_approvers';"
        )
        la_ok = "pooja.malhotra@techvista.com" in la
        ea_ok = "rajesh.kumar@techvista.com" in ea
        ok = parent_ok and la_ok and ea_ok
        check("1. Department", 2, ok,
              f"parent={parent_ok}, leave_appr={la_ok}, expense_appr={ea_ok}")
    except Exception as e:
        check("1. Department", 2, False, f"exception: {e}")


def check_2_designation() -> None:
    """Designation 'Machine Learning Engineer' exists."""
    try:
        r = hrms_sql(
            "SELECT name FROM `tabDesignation` WHERE name='Machine Learning Engineer';"
        )
        check("2. Designation", 1, bool(r), f"found={bool(r)}")
    except Exception as e:
        check("2. Designation", 1, False, f"exception: {e}")


def check_3_employee_core() -> None:
    """Employee HR-EMP-00020 core fields."""
    try:
        r = hrms_sql(
            "SELECT employee_name, date_of_birth, gender, date_of_joining, "
            "department, designation, employment_type, reports_to "
            "FROM `tabEmployee` WHERE name='HR-EMP-00020';"
        )
        if not r:
            check("3. Employee core", 2, False, "not found")
            return
        f = r.split("\t")
        ok = (
            len(f) >= 8
            and "Divya Krishnamurthy" in f[0]
            and "1995-11-14" in f[1]
            and f[2] == "Female"
            and "2026-09-01" in f[3]
            and f[4] == "Machine Learning - TVS"
            and f[5] == "Machine Learning Engineer"
            and "Full-time" in f[6]
            and "HR-EMP-00001" in f[7]
        )
        check("3. Employee core", 2, ok, f"fields={f[:8]}")
    except Exception as e:
        check("3. Employee core", 2, False, f"exception: {e}")


def check_4_emergency_contact() -> None:
    """Employee emergency contact."""
    try:
        r = hrms_sql(
            "SELECT person_to_be_contacted, emergency_phone_number "
            "FROM `tabEmployee` WHERE name='HR-EMP-00020';"
        )
        if not r:
            check("4. Emergency contact", 1, False, "employee not found")
            return
        f = r.split("\t")
        name_ok = len(f) >= 1 and "Ramesh Krishnamurthy" in f[0]
        phone_ok = len(f) >= 2 and "9922334455" in f[1]
        check("4. Emergency contact", 1, name_ok and phone_ok, f"raw={f}")
    except Exception as e:
        check("4. Emergency contact", 1, False, f"exception: {e}")


def check_5_holiday_list() -> None:
    """Holiday List 'ML Engineering Team 2026' with 4 holidays."""
    try:
        hl = hrms_sql(
            "SELECT from_date, to_date FROM `tabHoliday List` "
            "WHERE name='ML Engineering Team 2026';"
        )
        if not hl:
            check("5. Holiday List", 2, False, "not found")
            return
        dates_ok = "2026-01-01" in hl and "2026-12-31" in hl
        holidays = hrms_sql(
            "SELECT description, holiday_date FROM `tabHoliday` "
            "WHERE parent='ML Engineering Team 2026' ORDER BY holiday_date;"
        )
        expected = [
            ("Republic Day", "2026-01-26"),
            ("Eid al-Fitr", "2026-03-31"),
            ("Independence Day", "2026-08-15"),
            ("Diwali", "2026-11-08"),
        ]
        rows = [line.split("\t") for line in holidays.split("\n") if line.strip()]
        count_ok = len(rows) == 4
        content_ok = all(
            any(name in r[0] and date in r[1] for r in rows if len(r) >= 2)
            for name, date in expected
        )
        ok = dates_ok and count_ok and content_ok
        check("5. Holiday List", 2, ok,
              f"dates={dates_ok}, count={len(rows)}, content={content_ok}")
    except Exception as e:
        check("5. Holiday List", 2, False, f"exception: {e}")


def check_6_leave_type() -> None:
    """Leave Type 'Innovation Leave' settings."""
    try:
        r = hrms_sql(
            "SELECT is_lwp, is_carry_forward, max_leaves_allowed "
            "FROM `tabLeave Type` WHERE name='Innovation Leave';"
        )
        if not r:
            check("6. Leave Type", 1, False, "not found")
            return
        f = r.split("\t")
        # is_lwp=0 means paid leave (not Leave Without Pay)
        paid = len(f) >= 1 and f[0].strip() == "0"
        carry = len(f) >= 2 and f[1].strip() == "1"
        max_ok = len(f) >= 3 and f[2].strip().split(".")[0] == "7"
        ok = paid and carry and max_ok
        check("6. Leave Type", 1, ok,
              f"is_lwp={f[0].strip() if f else 'N/A'}(want 0), carry={carry}, max={f[2].strip() if len(f) >= 3 else 'N/A'}")
    except Exception as e:
        check("6. Leave Type", 1, False, f"exception: {e}")


def check_7_leave_policy() -> None:
    """Leave Policy with Innovation Leave (7) and Sick Leave (7)."""
    try:
        lp = hrms_sql(
            "SELECT name FROM `tabLeave Policy` "
            "WHERE name='ML Engineering Leave Policy 2026';"
        )
        if not lp:
            check("7. Leave Policy", 2, False, "not found")
            return
        details = hrms_sql(
            "SELECT leave_type, annual_allocation FROM `tabLeave Policy Detail` "
            "WHERE parent='ML Engineering Leave Policy 2026';"
        )
        rows = [line.split("\t") for line in details.split("\n") if line.strip()]
        alloc = {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2}
        innov = alloc.get("Innovation Leave", "")
        sick = alloc.get("Sick Leave", "")
        innov_ok = innov.split(".")[0] == "7" if innov else False
        sick_ok = sick.split(".")[0] == "7" if sick else False
        check("7. Leave Policy", 2, innov_ok and sick_ok,
              f"Innovation={innov}, Sick={sick}")
    except Exception as e:
        check("7. Leave Policy", 2, False, f"exception: {e}")


def check_8_leave_period() -> None:
    """Leave Period with correct dates and company."""
    try:
        r = hrms_sql(
            "SELECT from_date, to_date, company FROM `tabLeave Period` "
            "WHERE name='ML Engineering Leave Period 2026';"
        )
        if not r:
            check("8. Leave Period", 1, False, "not found")
            return
        ok = "2026-01-01" in r and "2026-12-31" in r and "TechVista" in r
        check("8. Leave Period", 1, ok, f"raw={r}")
    except Exception as e:
        check("8. Leave Period", 1, False, f"exception: {e}")


def check_9_leave_policy_assignment() -> None:
    """Leave Policy Assignment submitted for HR-EMP-00020."""
    try:
        r = hrms_sql(
            "SELECT leave_policy, effective_from, leave_period, docstatus "
            "FROM `tabLeave Policy Assignment` "
            "WHERE employee='HR-EMP-00020' "
            "AND leave_policy='ML Engineering Leave Policy 2026';"
        )
        if not r:
            check("9. Leave Policy Assignment", 2, False, "not found")
            return
        f = r.split("\t")
        policy_ok = len(f) >= 1 and "ML Engineering Leave Policy 2026" in f[0]
        date_ok = len(f) >= 2 and "2026-09-01" in f[1]
        period_ok = len(f) >= 3 and "ML Engineering Leave Period 2026" in f[2]
        submitted = len(f) >= 4 and f[3].strip() == "1"
        ok = policy_ok and date_ok and period_ok and submitted
        check("9. Leave Policy Assignment", 2, ok,
              f"policy={policy_ok}, date={date_ok}, period={period_ok}, submitted={submitted}")
    except Exception as e:
        check("9. Leave Policy Assignment", 2, False, f"exception: {e}")


def check_10_salary_component() -> None:
    """Salary Component 'ML Research Allowance' of type Earning."""
    try:
        r = hrms_sql(
            "SELECT type, description FROM `tabSalary Component` "
            "WHERE name='ML Research Allowance';"
        )
        if not r:
            check("10. Salary Component", 1, False, "not found")
            return
        f = r.split("\t")
        type_ok = len(f) >= 1 and "Earning" in f[0]
        desc_ok = len(f) >= 2 and "ML research tools" in f[1]
        check("10. Salary Component", 1, type_ok and desc_ok,
              f"type={f[0] if f else ''}, desc_match={desc_ok}")
    except Exception as e:
        check("10. Salary Component", 1, False, f"exception: {e}")


def check_11_salary_structure() -> None:
    """Salary Structure with correct frequency and earnings."""
    try:
        ss = hrms_sql(
            "SELECT payroll_frequency, company FROM `tabSalary Structure` "
            "WHERE name='ML Engineer Monthly Structure';"
        )
        if not ss:
            check("11. Salary Structure", 2, False, "not found")
            return
        freq_ok = "Monthly" in ss
        company_ok = "TechVista" in ss
        details = hrms_sql(
            "SELECT salary_component, formula, amount FROM `tabSalary Detail` "
            "WHERE parent='ML Engineer Monthly Structure' AND parentfield='earnings';"
        )
        rows = [line.split("\t") for line in details.split("\n") if line.strip()]
        comp: dict[str, tuple[str, str]] = {}
        for row in rows:
            if len(row) >= 3:
                comp[row[0].strip()] = (row[1].strip(), row[2].strip())
        basic = comp.get("Basic", ("", ""))
        basic_ok = "base * 0.57" in basic[0] or "base*0.57" in basic[0]
        mra = comp.get("ML Research Allowance", ("", ""))
        mra_ok = mra[1].startswith("9000")
        ok = freq_ok and company_ok and basic_ok and mra_ok
        check("11. Salary Structure", 2, ok,
              f"freq={freq_ok}, co={company_ok}, basic_formula={basic[0]!r}, mra_amt={mra[1]!r}")
    except Exception as e:
        check("11. Salary Structure", 2, False, f"exception: {e}")


def check_12_salary_structure_assignment() -> None:
    """Salary Structure Assignment submitted for HR-EMP-00020."""
    try:
        r = hrms_sql(
            "SELECT salary_structure, base, from_date, docstatus "
            "FROM `tabSalary Structure Assignment` "
            "WHERE employee='HR-EMP-00020' "
            "AND salary_structure='ML Engineer Monthly Structure';"
        )
        if not r:
            check("12. Salary Structure Assignment", 2, False, "not found")
            return
        f = r.split("\t")
        struct_ok = len(f) >= 1 and "ML Engineer Monthly Structure" in f[0]
        base_ok = len(f) >= 2 and "95000" in f[1]
        date_ok = len(f) >= 3 and "2026-09-01" in f[2]
        submitted = len(f) >= 4 and f[3].strip() == "1"
        ok = struct_ok and base_ok and date_ok and submitted
        check("12. Salary Structure Assignment", 2, ok,
              f"struct={struct_ok}, base={f[1].strip() if len(f) >= 2 else 'N/A'}, "
              f"date={date_ok}, submitted={submitted}")
    except Exception as e:
        check("12. Salary Structure Assignment", 2, False, f"exception: {e}")


def check_13_onboarding() -> None:
    """Employee Onboarding with 4 activities."""
    try:
        ob = hrms_sql(
            "SELECT name FROM `tabEmployee Onboarding` "
            "WHERE employee='HR-EMP-00020';"
        )
        if not ob:
            check("13. Onboarding", 2, False, "not found")
            return
        ob_name = ob.split("\n")[0].split("\t")[0].strip()
        acts = hrms_sql(
            "SELECT activity_name, user FROM `tabEmployee Boarding Activity` "
            f"WHERE parent='{ob_name}' AND parenttype='Employee Onboarding';"
        )
        rows = [line.split("\t") for line in acts.split("\n") if line.strip()]
        expected = {
            ("Setup ML platform and GPU cluster access",
             "rajesh.kumar@techvista.com"),
            ("HR documentation and benefits enrollment",
             "pooja.malhotra@techvista.com"),
            ("ML tools installation and environment configuration",
             "priya.sharma@techvista.com"),
            ("Team introduction and research project onboarding",
             "rajesh.kumar@techvista.com"),
        }
        found = {(r[0].strip(), r[1].strip()) for r in rows if len(r) >= 2}
        count_ok = len(rows) == 4
        content_ok = expected.issubset(found)
        check("13. Onboarding", 2, count_ok and content_ok,
              f"count={len(rows)}, content_match={content_ok}")
    except Exception as e:
        check("13. Onboarding", 2, False, f"exception: {e}")


# ── BigCapital Checks (API) ──────────────────────────────────────────────────

def check_14_bc_expense_account() -> None:
    """BigCapital account 'ML Engineer Salary Expense' under 'Utilities Expense'."""
    try:
        acct = bc_find_account("ML Engineer Salary Expense")
        if not acct:
            check("14. BC Expense Account", 1, False, "not found")
            return
        atype = str(acct.get("account_type") or acct.get("accountType") or "").lower()
        type_ok = "expense" in atype
        parent_id = acct.get("parent_account_id") or acct.get("parentAccountId")
        parent_name = bc_account_name(parent_id) if parent_id else ""
        parent_ok = "Utilities Expense" in parent_name
        check("14. BC Expense Account", 1, type_ok and parent_ok,
              f"type={atype}, parent={parent_name}")
    except Exception as e:
        check("14. BC Expense Account", 1, False, f"exception: {e}")


def check_15_bc_payable_account() -> None:
    """BigCapital account 'ML Engineer Salary Payable' under 'Accrued Expenses'."""
    try:
        acct = bc_find_account("ML Engineer Salary Payable")
        if not acct:
            check("15. BC Payable Account", 1, False, "not found")
            return
        atype = str(acct.get("account_type") or acct.get("accountType") or "").lower()
        type_ok = "liability" in atype or "current" in atype
        parent_id = acct.get("parent_account_id") or acct.get("parentAccountId")
        parent_name = bc_account_name(parent_id) if parent_id else ""
        parent_ok = "Accrued Expenses" in parent_name
        check("15. BC Payable Account", 1, type_ok and parent_ok,
              f"type={atype}, parent={parent_name}")
    except Exception as e:
        check("15. BC Payable Account", 1, False, f"exception: {e}")


def _verify_bc_journal(date_str: str, memo_substr: str) -> tuple[bool, str]:
    """Verify a BigCapital journal entry by date. Returns (passed, detail)."""
    base, s = bc()
    r = s.get(f"{base}/api/manual-journals", timeout=15)
    r.raise_for_status()
    data = r.json()
    # Handle various response shapes
    journals: list = []
    if isinstance(data, list):
        journals = data
    elif isinstance(data, dict):
        for key in ("manual_journals", "manualJournals", "data"):
            v = data.get(key)
            if isinstance(v, list):
                journals = v
                break
            if isinstance(v, dict):
                for k2 in ("manual_journals", "manualJournals"):
                    if isinstance(v.get(k2), list):
                        journals = v[k2]
                        break
                if journals:
                    break

    target = None
    for j in journals:
        jdate = str(j.get("date") or j.get("journal_date") or "")
        jdesc = str(j.get("description") or j.get("memo") or j.get("reference") or "")
        if date_str in jdate:
            if memo_substr in jdesc:
                target = j
                break
            if target is None:
                target = j  # fallback: match by date alone

    if not target:
        return False, f"journal for {date_str} not found among {len(journals)} journals"

    # Check published
    published = target.get("published_at") or target.get("publishedAt")
    if not published:
        return False, f"journal for {date_str} not published"

    # Get entries
    jid = target.get("id")
    entries = (target.get("entries")
               or target.get("manual_journal_entries")
               or target.get("manualJournalEntries")
               or [])
    if not entries and jid:
        try:
            r2 = s.get(f"{base}/api/manual-journals/{jid}", timeout=15)
            r2.raise_for_status()
            jdata = r2.json()
            j_detail = (jdata.get("manual_journal")
                        or jdata.get("manualJournal")
                        or jdata.get("data")
                        or jdata)
            entries = (j_detail.get("entries")
                       or j_detail.get("manual_journal_entries")
                       or j_detail.get("manualJournalEntries")
                       or [])
        except Exception:
            pass

    if not entries:
        return False, f"no entries found for journal {jid}"

    expense_acct = bc_find_account("ML Engineer Salary Expense")
    payable_acct = bc_find_account("ML Engineer Salary Payable")
    expense_id = expense_acct.get("id") if expense_acct else None
    payable_id = payable_acct.get("id") if payable_acct else None

    debit_ok = False
    credit_ok = False
    for e in entries:
        acct_id = e.get("account_id") or e.get("accountId")
        debit = float(e.get("debit") or 0)
        credit = float(e.get("credit") or 0)
        if acct_id == expense_id and abs(debit - 95000) < 1:
            debit_ok = True
        if acct_id == payable_id and abs(credit - 95000) < 1:
            credit_ok = True

    ok = debit_ok and credit_ok
    return ok, f"debit_expense={debit_ok}, credit_payable={credit_ok}"


def check_16_bc_journal_sept() -> None:
    """Published journal entry dated 2026-09-30 with correct entries."""
    try:
        ok, detail = _verify_bc_journal("2026-09-30", "month 1")
        check("16. BC Journal Sept", 2, ok, detail)
    except Exception as e:
        check("16. BC Journal Sept", 2, False, f"exception: {e}")


def check_17_bc_journal_oct() -> None:
    """Published journal entry dated 2026-10-31 with correct entries."""
    try:
        ok, detail = _verify_bc_journal("2026-10-31", "month 2")
        check("17. BC Journal Oct", 2, ok, detail)
    except Exception as e:
        check("17. BC Journal Oct", 2, False, f"exception: {e}")


# ── Twenty Checks (docker exec Postgres) ─────────────────────────────────────

def check_18_twenty_tasks() -> None:
    """Three onboarding tasks in Twenty CRM."""
    try:
        schema = ws_schema()
        tasks_expected = [
            ("IT equipment provisioning - Divya Krishnamurthy", "2026-08-28"),
            ("Schedule orientation meeting - Divya Krishnamurthy", "2026-09-05"),
            ("Verify payroll setup - Divya Krishnamurthy", "2026-09-15"),
        ]
        issues = []
        for title, due in tasks_expected:
            safe_title = title.replace("'", "''")
            r = twenty_sql(
                f"SELECT title, \"dueAt\"::text FROM \"{schema}\".task "
                f"WHERE \"deletedAt\" IS NULL AND title = '{safe_title}';"
            )
            if not r:
                issues.append(f"missing: {title[:40]}")
            elif due not in r:
                issues.append(f"wrong due date: {title[:40]}")
        ok = not issues
        check("18. Twenty Tasks", 2, ok,
              "all 3 found" if ok else f"issues={issues}")
    except Exception as e:
        check("18. Twenty Tasks", 2, False, f"exception: {e}")


def check_19_twenty_note() -> None:
    """Onboarding summary note in Twenty CRM."""
    try:
        schema = ws_schema()
        r = twenty_sql(
            f"SELECT title, \"bodyV2Markdown\" FROM \"{schema}\".note "
            f"WHERE \"deletedAt\" IS NULL "
            f"AND title = 'Onboarding Summary - Divya Krishnamurthy - 2026-09-01';"
        )
        if not r:
            check("19. Twenty Note", 2, False, "not found")
            return
        keywords = [
            "HR-EMP-00020", "Machine Learning - TVS",
            "Machine Learning Engineer", "Full-time",
            "ML Engineering Leave Policy 2026",
            "ML Engineer Monthly Structure", "95000",
        ]
        missing = [kw for kw in keywords if kw not in r]
        ok = not missing
        check("19. Twenty Note", 2, ok,
              "all keywords found" if ok else f"missing={missing}")
    except Exception as e:
        check("19. Twenty Note", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_department()
    check_2_designation()
    check_3_employee_core()
    check_4_emergency_contact()
    check_5_holiday_list()
    check_6_leave_type()
    check_7_leave_policy()
    check_8_leave_period()
    check_9_leave_policy_assignment()
    check_10_salary_component()
    check_11_salary_structure()
    check_12_salary_structure_assignment()
    check_13_onboarding()
    check_14_bc_expense_account()
    check_15_bc_payable_account()
    check_16_bc_journal_sept()
    check_17_bc_journal_oct()
    check_18_twenty_tasks()
    check_19_twenty_note()

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
