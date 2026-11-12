# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_144/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-144-I5: Shift Scheduling, Overtime Accounting, and CRM Task Management

Checks: 16 weighted checks across hrms, bigcapital, twenty.
Strategy: docker exec MariaDB for hrms; REST API for bigcapital; docker exec Postgres for twenty.

Required env vars:
  SERVER_HOSTNAME, HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER.
"""

import json
import os
import subprocess
import sys

import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

HRMS_PORT = os.getenv("HRMS_PORT")
HRMS_CONTAINER = os.getenv("HRMS_CONTAINER")
HRMS_DB_CONTAINER = os.getenv("HRMS_DB_CONTAINER")
BIGCAPITAL_PORT = os.getenv("BIGCAPITAL_PORT")
BIGCAPITAL_CONTAINER = os.getenv("BIGCAPITAL_CONTAINER")
BIGCAPITAL_DB_CONTAINER = os.getenv("BIGCAPITAL_DB_CONTAINER")
TWENTY_PORT = os.getenv("TWENTY_PORT")
TWENTY_CONTAINER = os.getenv("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = os.getenv("TWENTY_DB_CONTAINER")

_required = {
    "HRMS_PORT": HRMS_PORT, "HRMS_CONTAINER": HRMS_CONTAINER,
    "HRMS_DB_CONTAINER": HRMS_DB_CONTAINER,
    "BIGCAPITAL_PORT": BIGCAPITAL_PORT, "BIGCAPITAL_CONTAINER": BIGCAPITAL_CONTAINER,
    "BIGCAPITAL_DB_CONTAINER": BIGCAPITAL_DB_CONTAINER,
    "TWENTY_PORT": TWENTY_PORT, "TWENTY_CONTAINER": TWENTY_CONTAINER,
    "TWENTY_DB_CONTAINER": TWENTY_DB_CONTAINER,
}
for _var, _val in _required.items():
    if not _val:
        print(f"FATAL: {_var} not set", file=sys.stderr)
        sys.exit(1)

BIGCAPITAL_BASE = f"http://{HOST}:{BIGCAPITAL_PORT}"

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


def _discover_hrms_db() -> str:
    """Discover the Frappe bench database name in the HRMS MariaDB container."""
    rc, out, err = docker_exec(
        HRMS_DB_CONTAINER,
        "mariadb", "-u", "root", "-phrms123456", "--default-character-set=utf8mb4",
        "-N", "-B", "-e",
        "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA "
        "WHERE SCHEMA_NAME LIKE '\\_%' AND SCHEMA_NAME NOT IN "
        "('information_schema','mysql','performance_schema','sys')",
    )
    if rc != 0:
        raise RuntimeError(f"Cannot discover HRMS DB: {err.strip()}")
    # Pick the schema that has tabShift Type
    for schema in out.strip().split("\n"):
        schema = schema.strip()
        if not schema:
            continue
        rc2, out2, _ = docker_exec(
            HRMS_DB_CONTAINER,
            "mariadb", "-u", "root", "-phrms123456", "--default-character-set=utf8mb4",
            "-D", schema, "-N", "-B", "-e",
            "SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_schema='{schema}' AND table_name='tabShift Type'",
        )
        if rc2 == 0 and out2.strip() == "1":
            return schema
    raise RuntimeError("No Frappe bench DB found with tabShift Type")


_hrms_db_name: str | None = None


def hrms_sql(query: str) -> str:
    """Run a MariaDB query against the HRMS Frappe database."""
    global _hrms_db_name
    if _hrms_db_name is None:
        _hrms_db_name = _discover_hrms_db()
    rc, out, err = docker_exec(
        HRMS_DB_CONTAINER,
        "mariadb", "-u", "root", "-phrms123456", "--default-character-set=utf8mb4",
        "-D", _hrms_db_name, "-N", "-B", "-e", query,
    )
    if rc != 0:
        raise RuntimeError(f"mariadb error: {err.strip()}")
    return out.strip()


def _bigcapital_token_and_org() -> tuple[str, str]:
    """Authenticate to BigCapital and return (access_token, organization_id)."""
    r = requests.post(
        f"{BIGCAPITAL_BASE}/api/auth/signin",
        json={"email": "admin@bigcapital.local", "password": "admin123"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return data["access_token"], data["organization_id"]


_bc_auth: tuple[str, str] | None = None


def bc_api_get(path: str, params: dict | None = None) -> dict:
    """GET a BigCapital API endpoint (authenticated)."""
    global _bc_auth
    if _bc_auth is None:
        _bc_auth = _bigcapital_token_and_org()
    token, org_id = _bc_auth
    r = requests.get(
        f"{BIGCAPITAL_BASE}{path}",
        params=params,
        headers={"Authorization": f"Bearer {token}", "organization-id": org_id},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _discover_twenty_workspace() -> str:
    """Discover the Twenty workspace schema name."""
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c",
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'workspace_%' LIMIT 1",
    )
    if rc != 0:
        raise RuntimeError(f"Cannot discover Twenty workspace: {err.strip()}")
    ws = out.strip()
    if not ws:
        raise RuntimeError("No workspace schema found in Twenty DB")
    return ws


_twenty_ws: str | None = None


def twenty_sql(query: str) -> str:
    """Run a Postgres query against the Twenty workspace schema."""
    global _twenty_ws
    if _twenty_ws is None:
        _twenty_ws = _discover_twenty_workspace()
    full_query = f"SET search_path TO {_twenty_ws}; {query}"
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c", full_query,
    )
    if rc != 0:
        raise RuntimeError(f"psql error: {err.strip()}")
    # Remove the 'SET' line from SET search_path output
    lines = out.strip().split("\n")
    result_lines = [l for l in lines if l.strip() != "SET"]
    return "\n".join(result_lines).strip()


# ── HRMS checks ──────────────────────────────────────────────────────────────

def check_1_gamma_shift() -> None:
    """Gamma Shift exists with start 06:30, end 14:30, grace 8min, auto-attendance."""
    try:
        row = hrms_sql(
            "SELECT start_time, end_time, enable_auto_attendance, "
            "early_exit_grace_period, late_entry_grace_period "
            "FROM `tabShift Type` WHERE name='Gamma Shift'"
        )
        if not row:
            check("1. Gamma Shift type", 1, False, "not found")
            return
        parts = row.split("\t")
        ok = (
            parts[0].startswith("06:30") and parts[1].startswith("14:30")
            and parts[2] == "1"
            and float(parts[3]) == 8 and float(parts[4]) == 8
        )
        check("1. Gamma Shift type", 1, ok,
              f"start={parts[0]}, end={parts[1]}, auto={parts[2]}, grace={parts[3]}/{parts[4]}")
    except Exception as e:
        check("1. Gamma Shift type", 1, False, f"exception: {e}")


def check_2_sigma_shift() -> None:
    """Sigma Shift exists with start 14:30, end 22:30, grace 8min, auto-attendance."""
    try:
        row = hrms_sql(
            "SELECT start_time, end_time, enable_auto_attendance, "
            "early_exit_grace_period, late_entry_grace_period "
            "FROM `tabShift Type` WHERE name='Sigma Shift'"
        )
        if not row:
            check("2. Sigma Shift type", 1, False, "not found")
            return
        parts = row.split("\t")
        ok = (
            parts[0].startswith("14:30") and parts[1].startswith("22:30")
            and parts[2] == "1"
            and float(parts[3]) == 8 and float(parts[4]) == 8
        )
        check("2. Sigma Shift type", 1, ok,
              f"start={parts[0]}, end={parts[1]}, auto={parts[2]}, grace={parts[3]}/{parts[4]}")
    except Exception as e:
        check("2. Sigma Shift type", 1, False, f"exception: {e}")


def check_3_theta_shift() -> None:
    """Theta Shift exists with start 22:30, end 06:30, grace 8min, auto-attendance."""
    try:
        row = hrms_sql(
            "SELECT start_time, end_time, enable_auto_attendance, "
            "early_exit_grace_period, late_entry_grace_period "
            "FROM `tabShift Type` WHERE name='Theta Shift'"
        )
        if not row:
            check("3. Theta Shift type", 1, False, "not found")
            return
        parts = row.split("\t")
        ok = (
            parts[0].startswith("22:30") and parts[1].startswith("06:30")
            and parts[2] == "1"
            and float(parts[3]) == 8 and float(parts[4]) == 8
        )
        check("3. Theta Shift type", 1, ok,
              f"start={parts[0]}, end={parts[1]}, auto={parts[2]}, grace={parts[3]}/{parts[4]}")
    except Exception as e:
        check("3. Theta Shift type", 1, False, f"exception: {e}")


def check_4_gamma_bulk_assignment() -> None:
    """Gamma Shift bulk assignment for dept Finance & Accounting - TVS, 2026-09-01 to 2026-09-30."""
    try:
        count = hrms_sql(
            "SELECT COUNT(*) FROM `tabShift Assignment` "
            "WHERE shift_type='Gamma Shift' "
            "AND department='Finance & Accounting - TVS' "
            "AND start_date<='2026-09-01' AND end_date>='2026-09-30' "
            "AND docstatus=1"
        )
        n = int(count) if count else 0
        check("4. Gamma Shift bulk assignment (Finance & Accounting - TVS)", 2, n >= 1,
              f"found {n} active assignments")
    except Exception as e:
        check("4. Gamma Shift bulk assignment (Finance & Accounting - TVS)", 2, False, f"exception: {e}")


def check_5_sigma_individual_assignments() -> None:
    """Sigma Shift assignments for Kavitha Iyer, Arjun Nair, Ananya Reddy."""
    try:
        employees = ["Kavitha Iyer", "Arjun Nair", "Ananya Reddy"]
        found = []
        missing = []
        for emp in employees:
            row = hrms_sql(
                f"SELECT COUNT(*) FROM `tabShift Assignment` "
                f"WHERE shift_type='Sigma Shift' "
                f"AND employee_name='{emp}' "
                f"AND start_date<='2026-09-01' AND end_date>='2026-09-30' "
                f"AND docstatus=1"
            )
            if int(row or 0) > 0:
                found.append(emp)
            else:
                missing.append(emp)
        ok = len(missing) == 0
        check("5. Sigma Shift individual assignments (3 employees)", 2, ok,
              f"found={found}, missing={missing}" if missing else "all 3 found")
    except Exception as e:
        check("5. Sigma Shift individual assignments (3 employees)", 2, False, f"exception: {e}")


def check_6_theta_individual_assignments() -> None:
    """Theta Shift assignments for Mohammed Farooq, Sanjay Krishnan."""
    try:
        employees = ["Mohammed Farooq", "Sanjay Krishnan"]
        found = []
        missing = []
        for emp in employees:
            row = hrms_sql(
                f"SELECT COUNT(*) FROM `tabShift Assignment` "
                f"WHERE shift_type='Theta Shift' "
                f"AND employee_name='{emp}' "
                f"AND start_date<='2026-09-01' AND end_date>='2026-09-30' "
                f"AND docstatus=1"
            )
            if int(row or 0) > 0:
                found.append(emp)
            else:
                missing.append(emp)
        ok = len(missing) == 0
        check("6. Theta Shift individual assignments (2 employees)", 2, ok,
              f"found={found}, missing={missing}" if missing else "all 2 found")
    except Exception as e:
        check("6. Theta Shift individual assignments (2 employees)", 2, False, f"exception: {e}")


def check_7_shift_request_approved() -> None:
    """Shift Request for Deepika Joshi (HR-EMP-00010) from Gamma to Sigma on 2026-09-12, Approved."""
    try:
        row = hrms_sql(
            "SELECT status, shift_type "
            "FROM `tabShift Request` "
            "WHERE employee='HR-EMP-00010' AND from_date='2026-09-12' "
            "AND docstatus=1 LIMIT 1"
        )
        if not row:
            check("7. Shift Request Deepika Joshi approved", 2, False, "not found")
            return
        parts = row.split("\t")
        status = parts[0] if len(parts) > 0 else ""
        to_shift = parts[1] if len(parts) > 1 else ""
        ok = status == "Approved" and "Sigma" in to_shift
        check("7. Shift Request Deepika Joshi approved", 2, ok,
              f"status={status}, shift_type={to_shift}")
    except Exception as e:
        check("7. Shift Request Deepika Joshi approved", 2, False, f"exception: {e}")


def check_8_overtime_type() -> None:
    """Overtime Type 'Night Differential Overtime' with multiplier 1.25."""
    try:
        # First check if the table exists
        exists = hrms_sql(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name='tabOvertime Type'"
        )
        if int(exists or 0) == 0:
            check("8. Overtime Type Night Differential Overtime", 1, False,
                  "tabOvertime Type table does not exist")
            return
        row = hrms_sql(
            "SELECT name FROM `tabOvertime Type` "
            "WHERE name='Night Differential Overtime' LIMIT 1"
        )
        if not row:
            check("8. Overtime Type Night Differential Overtime", 1, False, "not found")
            return
        val = hrms_sql(
            "SELECT standard_multiplier FROM `tabOvertime Type` "
            "WHERE name='Night Differential Overtime'"
        )
        ok = abs(float(val or 0) - 1.25) < 0.01
        check("8. Overtime Type Night Differential Overtime", 1, ok,
              f"standard_multiplier={val}")
    except Exception as e:
        check("8. Overtime Type Night Differential Overtime", 1, False, f"exception: {e}")


def check_9_ot_slip_suresh() -> None:
    """Overtime Slip for Suresh Menon (HR-EMP-00009), 6h, 2026-09-06, submitted."""
    try:
        row = hrms_sql(
            "SELECT employee_name, total_overtime_duration, posting_date "
            "FROM `tabOvertime Slip` "
            "WHERE employee='HR-EMP-00009' AND docstatus=1 LIMIT 1"
        )
        if not row:
            row = hrms_sql(
                "SELECT employee_name, total_overtime_duration, posting_date "
                "FROM `tabOvertime Slip` "
                "WHERE employee_name LIKE '%Suresh%' AND docstatus=1 LIMIT 1"
            )
        if not row:
            check("9. OT Slip Suresh Menon (6h)", 2, False, "not found")
            return
        parts = row.split("\t")
        hours = float(parts[1]) if len(parts) > 1 else 0
        date_val = parts[2] if len(parts) > 2 else ""
        ok = abs(hours - 6.0) < 0.01 and "2026-09-06" in date_val
        check("9. OT Slip Suresh Menon (6h)", 2, ok,
              f"total_overtime_duration={hours}, posting_date={date_val}")
    except Exception as e:
        check("9. OT Slip Suresh Menon (6h)", 2, False, f"exception: {e}")


def check_10_ot_slip_rahul() -> None:
    """Overtime Slip for Rahul Verma (HR-EMP-00013), 4h, 2026-09-13, submitted."""
    try:
        row = hrms_sql(
            "SELECT employee_name, total_overtime_duration, posting_date "
            "FROM `tabOvertime Slip` "
            "WHERE employee='HR-EMP-00013' AND docstatus=1 LIMIT 1"
        )
        if not row:
            row = hrms_sql(
                "SELECT employee_name, total_overtime_duration, posting_date "
                "FROM `tabOvertime Slip` "
                "WHERE employee_name LIKE '%Rahul%' AND docstatus=1 LIMIT 1"
            )
        if not row:
            check("10. OT Slip Rahul Verma (4h)", 2, False, "not found")
            return
        parts = row.split("\t")
        hours = float(parts[1]) if len(parts) > 1 else 0
        date_val = parts[2] if len(parts) > 2 else ""
        ok = abs(hours - 4.0) < 0.01 and "2026-09-13" in date_val
        check("10. OT Slip Rahul Verma (4h)", 2, ok,
              f"total_overtime_duration={hours}, posting_date={date_val}")
    except Exception as e:
        check("10. OT Slip Rahul Verma (4h)", 2, False, f"exception: {e}")


# ── BigCapital checks (REST API) ─────────────────────────────────────────────

def check_11_expense_account() -> None:
    """Overtime Shift Differential Expense account exists as expense type."""
    try:
        data = bc_api_get("/api/accounts")
        accounts = data.get("accounts", [])
        match = [a for a in accounts if a.get("name") == "Overtime Shift Differential Expense"]
        if not match:
            check("11. Expense account (Overtime Shift Differential Expense)", 1, False, "not found")
            return
        atype = match[0].get("account_type", "")
        ok = "expense" in atype.lower()
        check("11. Expense account (Overtime Shift Differential Expense)", 1, ok,
              f"account_type={atype}")
    except Exception as e:
        check("11. Expense account (Overtime Shift Differential Expense)", 1, False, f"exception: {e}")


def check_12_ap_account() -> None:
    """Accounts Payable (A/P) exists as liability type."""
    try:
        data = bc_api_get("/api/accounts")
        accounts = data.get("accounts", [])
        match = [a for a in accounts if a.get("name") == "Accounts Payable (A/P)"]
        if not match:
            check("12. Accounts Payable (A/P) account", 1, False, "not found")
            return
        atype = match[0].get("account_type", "")
        ok = "liabilit" in atype.lower() or "payable" in atype.lower() or "current_liability" in atype.lower()
        check("12. Accounts Payable (A/P) account", 1, ok, f"account_type={atype}")
    except Exception as e:
        check("12. Accounts Payable (A/P) account", 1, False, f"exception: {e}")


def check_13_journal_entry() -> None:
    """Manual journal entry dated 2026-09-30: debit Overtime Shift Differential Expense 625.00, credit AP 625.00."""
    try:
        data = bc_api_get("/api/manual-journals", params={"page_size": 200})
        journals = data.get("manual_journals", [])
        debit_ok = False
        credit_ok = False
        for j in journals:
            j_date = (j.get("date") or "")[:10]
            if j_date != "2026-09-30":
                continue
            status = j.get("status", "")
            if status != "published":
                continue
            for entry in j.get("entries", []):
                acct_name = entry.get("account", {}).get("name", "")
                debit = float(entry.get("debit") or 0)
                credit = float(entry.get("credit") or 0)
                if "Overtime Shift Differential Expense" in acct_name and abs(debit - 625.0) < 0.01:
                    debit_ok = True
                if "Accounts Payable" in acct_name and abs(credit - 625.0) < 0.01:
                    credit_ok = True
        ok = debit_ok and credit_ok
        check("13. Journal entry (625.00 debit/credit on 2026-09-30)", 3, ok,
              f"debit_ok={debit_ok}, credit_ok={credit_ok}")
    except Exception as e:
        check("13. Journal entry (625.00 debit/credit on 2026-09-30)", 3, False, f"exception: {e}")


# ── Twenty checks (DB) ──────────────────────────────────────────────────────

def check_14_review_task() -> None:
    """Task 'Review shift schedule compliance -- 2026-09-01 to 2026-09-30' with due 2026-10-07."""
    try:
        row = twenty_sql(
            "SELECT title, \"dueAt\" FROM task "
            "WHERE \"deletedAt\" IS NULL "
            "AND title = 'Review shift schedule compliance -- 2026-09-01 to 2026-09-30' "
            "LIMIT 1"
        )
        if not row:
            check("14. Twenty task: Review shift schedule compliance", 2, False, "not found")
            return
        parts = row.split("|")
        due = parts[1].strip() if len(parts) > 1 else ""
        ok = "2026-10-07" in due
        check("14. Twenty task: Review shift schedule compliance", 2, ok, f"dueAt={due}")
    except Exception as e:
        check("14. Twenty task: Review shift schedule compliance", 2, False, f"exception: {e}")


def check_15_payments_task() -> None:
    """Task 'Process overtime payments -- 2026-09-30' with due 2026-10-14."""
    try:
        row = twenty_sql(
            "SELECT title, \"dueAt\" FROM task "
            "WHERE \"deletedAt\" IS NULL "
            "AND title = 'Process overtime payments -- 2026-09-30' "
            "LIMIT 1"
        )
        if not row:
            check("15. Twenty task: Process overtime payments", 2, False, "not found")
            return
        parts = row.split("|")
        due = parts[1].strip() if len(parts) > 1 else ""
        ok = "2026-10-14" in due
        check("15. Twenty task: Process overtime payments", 2, ok, f"dueAt={due}")
    except Exception as e:
        check("15. Twenty task: Process overtime payments", 2, False, f"exception: {e}")


def check_16_summary_note() -> None:
    """Note 'Shift & Overtime Summary -- 2026-09-01 to 2026-09-30' exists."""
    try:
        row = twenty_sql(
            "SELECT title FROM note "
            "WHERE \"deletedAt\" IS NULL "
            "AND title = 'Shift & Overtime Summary -- 2026-09-01 to 2026-09-30' "
            "LIMIT 1"
        )
        ok = bool(row)
        check("16. Twenty note: Shift & Overtime Summary", 1, ok,
              "found" if ok else "not found")
    except Exception as e:
        check("16. Twenty note: Shift & Overtime Summary", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_gamma_shift()
    check_2_sigma_shift()
    check_3_theta_shift()
    check_4_gamma_bulk_assignment()
    check_5_sigma_individual_assignments()
    check_6_theta_individual_assignments()
    check_7_shift_request_approved()
    check_8_overtime_type()
    check_9_ot_slip_suresh()
    check_10_ot_slip_rahul()
    check_11_expense_account()
    check_12_ap_account()
    check_13_journal_entry()
    check_14_review_task()
    check_15_payments_task()
    check_16_summary_note()

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
