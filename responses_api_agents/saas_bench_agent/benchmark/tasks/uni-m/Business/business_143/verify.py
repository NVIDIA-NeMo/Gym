# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/tasks/uni-m/Business/business_143/verify.py
# Original license: CC-BY-4.0; Copyright: SaaS-Bench Authors; see bundled licenses.
# NVIDIA modifications: Use MANUAL_JOURNALS_ENTRIES and PUBLISHED_AT schema instead of upstream singular table and STATUS.
"""
Verifier for Business-143-I1: End-to-end performance appraisal cycle

Checks: 12 weighted checks across hrms, bigcapital, twenty.
Strategy: docker exec (MariaDB for HRMS & BigCapital, Postgres for Twenty)

Required env vars:
  SERVER_HOSTNAME,
  HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER
"""

import os
import sys
import subprocess

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

HRMS_PORT = os.environ.get("HRMS_PORT")
HRMS_CONTAINER = os.environ.get("HRMS_CONTAINER")
HRMS_DB_CONTAINER = os.environ.get("HRMS_DB_CONTAINER")

BIGCAPITAL_PORT = os.environ.get("BIGCAPITAL_PORT")
BIGCAPITAL_CONTAINER = os.environ.get("BIGCAPITAL_CONTAINER")
BIGCAPITAL_DB_CONTAINER = os.environ.get("BIGCAPITAL_DB_CONTAINER")

TWENTY_PORT = os.environ.get("TWENTY_PORT")
TWENTY_CONTAINER = os.environ.get("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = os.environ.get("TWENTY_DB_CONTAINER")

_required = {
    "HRMS_PORT": HRMS_PORT, "HRMS_CONTAINER": HRMS_CONTAINER, "HRMS_DB_CONTAINER": HRMS_DB_CONTAINER,
    "BIGCAPITAL_PORT": BIGCAPITAL_PORT, "BIGCAPITAL_CONTAINER": BIGCAPITAL_CONTAINER,
    "BIGCAPITAL_DB_CONTAINER": BIGCAPITAL_DB_CONTAINER,
    "TWENTY_PORT": TWENTY_PORT, "TWENTY_CONTAINER": TWENTY_CONTAINER,
    "TWENTY_DB_CONTAINER": TWENTY_DB_CONTAINER,
}
for _var, _val in _required.items():
    if not _val:
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


# ── DB credential & name discovery ────────────────────────────────────────────
def _get_container_env(container: str, var: str) -> str:
    """Read an environment variable from inside a container."""
    rc, out, _ = docker_exec(container, "printenv", var, timeout=10)
    return out.strip() if rc == 0 else ""


# Cache values discovered at runtime
_hrms_db_name: str | None = None
_hrms_db_pass: str | None = None
_bc_db_name: str | None = None
_bc_db_pass: str | None = None


def _mysql_cmd(container: str, password: str, *extra: str) -> tuple[int, str, str]:
    """Run mysql command with password."""
    cmd = ["mysql", "-u", "root"]
    if password:
        cmd.append(f"-p{password}")
    cmd.extend(extra)
    return docker_exec(container, *cmd, timeout=15)


def _find_hrms_db() -> tuple[str, str]:
    """Dynamically find the Frappe bench database name and root password."""
    global _hrms_db_name, _hrms_db_pass
    if _hrms_db_name and _hrms_db_pass is not None:
        return _hrms_db_name, _hrms_db_pass

    # Get root password from container env
    _hrms_db_pass = _get_container_env(HRMS_DB_CONTAINER, "MYSQL_ROOT_PASSWORD") or ""

    rc, out, err = _mysql_cmd(
        HRMS_DB_CONTAINER, _hrms_db_pass, "-e",
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
        "WHERE SCHEMA_NAME LIKE '\\_%' "
        "AND SCHEMA_NAME NOT IN ('information_schema','mysql','performance_schema','sys');",
    )
    if rc != 0:
        raise RuntimeError(f"Cannot list HRMS databases: {err.strip()}")
    candidates = [l.strip() for l in out.strip().splitlines() if l.strip() and l.strip() != "SCHEMA_NAME"]
    if not candidates:
        raise RuntimeError("No Frappe database found in HRMS MariaDB")
    for db in candidates:
        rc2, out2, _ = _mysql_cmd(
            HRMS_DB_CONTAINER, _hrms_db_pass, "-D", db, "-N", "-B", "-e",
            "SHOW TABLES LIKE 'tabEmployee';",
        )
        if rc2 == 0 and "tabEmployee" in out2:
            _hrms_db_name = db
            return db, _hrms_db_pass
    _hrms_db_name = candidates[0]
    return _hrms_db_name, _hrms_db_pass


def hrms_sql(query: str) -> str:
    """Execute a MariaDB query against the HRMS Frappe database."""
    db, pw = _find_hrms_db()
    cmd = ["mysql", "-u", "root"]
    if pw:
        cmd.append(f"-p{pw}")
    cmd.extend(["--default-character-set=utf8mb4", "-D", db, "-N", "-B", "-e", query])
    rc, out, err = docker_exec(HRMS_DB_CONTAINER, *cmd, timeout=15)
    if rc != 0:
        raise RuntimeError(f"HRMS SQL error: {err.strip()}")
    return out.strip()


def _find_bigcapital_db() -> tuple[str, str]:
    """Find the BigCapital tenant database in MariaDB."""
    global _bc_db_name, _bc_db_pass
    if _bc_db_name and _bc_db_pass is not None:
        return _bc_db_name, _bc_db_pass

    # BigCapital uses bigcapital / bigcapital123 credentials
    _bc_db_pass = "bigcapital123"

    rc, out, err = docker_exec(
        BIGCAPITAL_DB_CONTAINER,
        "mysql", "-u", "bigcapital", f"-p{_bc_db_pass}", "-e",
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
        "WHERE SCHEMA_NAME LIKE 'bigcapital_tenant_%' OR SCHEMA_NAME = 'bigcapital';",
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"Cannot list BigCapital databases: {err.strip()}")
    candidates = [l.strip() for l in out.strip().splitlines() if l.strip() and l.strip() != "SCHEMA_NAME"]
    for c in candidates:
        if c.startswith("bigcapital_tenant_"):
            _bc_db_name = c
            return c, _bc_db_pass
    if candidates:
        _bc_db_name = candidates[0]
        return _bc_db_name, _bc_db_pass
    raise RuntimeError("No BigCapital database found")


def bigcapital_sql(query: str) -> str:
    """Execute a MariaDB query against the BigCapital tenant database."""
    db, pw = _find_bigcapital_db()
    cmd = ["mysql", "-u", "bigcapital", f"-p{pw}",
           "--default-character-set=utf8mb4", "-D", db, "-N", "-B", "-e", query]
    rc, out, err = docker_exec(BIGCAPITAL_DB_CONTAINER, *cmd, timeout=15)
    if rc != 0:
        raise RuntimeError(f"BigCapital SQL error: {err.strip()}")
    return out.strip()


# Cache for Twenty workspace schema
_twenty_schema: str | None = None


def twenty_sql(query: str) -> str:
    """Execute a Postgres query against the Twenty database (workspace schema)."""
    global _twenty_schema
    if not _twenty_schema:
        rc, out, err = docker_exec(
            TWENTY_DB_CONTAINER,
            "psql", "-U", "postgres", "-d", "default",
            "-t", "-A", "-c",
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'workspace_%' LIMIT 1;",
            timeout=15,
        )
        if rc != 0:
            raise RuntimeError(f"Twenty schema lookup error: {err.strip()}")
        _twenty_schema = out.strip()
        if not _twenty_schema:
            raise RuntimeError("No workspace schema found in Twenty DB")

    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default",
        "-t", "-A", "-c",
        f'SET search_path TO "{_twenty_schema}"; {query}',
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"Twenty SQL error: {err.strip()}")
    # Filter out 'SET' lines from SET search_path output
    lines = [l for l in out.strip().splitlines() if l.strip() != "SET"]
    return "\n".join(lines).strip()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_kras_exist() -> None:
    """Verify KRAs 'Technical Delivery Excellence' and 'Customer Satisfaction & Retention' exist."""
    try:
        result = hrms_sql(
            "SELECT name FROM `tabKRA` WHERE name IN "
            "('Technical Delivery Excellence', 'Customer Satisfaction & Retention') "
            "ORDER BY name;"
        )
        found = set(result.splitlines()) if result else set()
        expected = {"Technical Delivery Excellence", "Customer Satisfaction & Retention"}
        missing = expected - found
        check("1. KRAs exist", 1, not missing,
              f"missing: {missing}" if missing else "")
    except Exception as e:
        check("1. KRAs exist", 1, False, f"exception: {e}")


def check_2_appraisal_template() -> None:
    """Verify Appraisal Template with correct KRA weightages 60/40."""
    try:
        result = hrms_sql(
            "SELECT g.key_result_area, g.per_weightage FROM `tabAppraisal Template` t "
            "JOIN `tabAppraisal Template Goal` g ON g.parent = t.name "
            "WHERE t.name = 'Engineering Performance Template 2025' "
            "ORDER BY g.per_weightage DESC;"
        )
        rows = result.splitlines() if result else []
        parsed = {}
        for row in rows:
            parts = row.split("\t")
            if len(parts) == 2:
                parsed[parts[0].strip()] = float(parts[1].strip())

        ok = (
            parsed.get("Technical Delivery Excellence") == 60.0
            and parsed.get("Customer Satisfaction & Retention") == 40.0
        )
        check("2. Appraisal template with weightages", 2, ok,
              f"found: {parsed}" if not ok else "")
    except Exception as e:
        check("2. Appraisal template with weightages", 2, False, f"exception: {e}")


def check_3_goals_vikram() -> None:
    """Verify goals exist for Vikram Singh (HR-EMP-00005)."""
    try:
        result = hrms_sql(
            "SELECT goal_name FROM `tabGoal` "
            "WHERE employee = 'HR-EMP-00005' ORDER BY goal_name;"
        )
        found = set(result.splitlines()) if result else set()
        expected = {"Reduce sprint defect rate by 30%", "Improve client NPS score"}
        missing = expected - found
        check("3. Goals for Vikram Singh", 2, not missing,
              f"missing: {missing}" if missing else "")
    except Exception as e:
        check("3. Goals for Vikram Singh", 2, False, f"exception: {e}")


def check_4_goals_ananya() -> None:
    """Verify goals exist for Ananya Reddy (HR-EMP-00007)."""
    try:
        result = hrms_sql(
            "SELECT goal_name FROM `tabGoal` "
            "WHERE employee = 'HR-EMP-00007' ORDER BY goal_name;"
        )
        found = set(result.splitlines()) if result else set()
        expected = {"Deliver all project milestones on time", "Achieve zero escalations"}
        missing = expected - found
        check("4. Goals for Ananya Reddy", 2, not missing,
              f"missing: {missing}" if missing else "")
    except Exception as e:
        check("4. Goals for Ananya Reddy", 2, False, f"exception: {e}")


def check_5_appraisals_submitted() -> None:
    """Verify appraisals submitted with correct overall scores (4.6 and 3.4)."""
    try:
        result = hrms_sql(
            "SELECT employee, employee_name, docstatus, total_score "
            "FROM `tabAppraisal` "
            "WHERE appraisal_cycle = 'H2 2025 Engineering Performance Review' "
            "ORDER BY employee;"
        )
        rows = result.splitlines() if result else []
        emp_scores = {}
        emp_status = {}
        for row in rows:
            parts = row.split("\t")
            if len(parts) >= 4:
                emp_id = parts[0].strip()
                emp_scores[emp_id] = float(parts[3].strip())
                emp_status[emp_id] = int(parts[2].strip())

        vikram_ok = (
            abs(emp_scores.get("HR-EMP-00005", 0) - 4.6) < 0.05
            and emp_status.get("HR-EMP-00005") == 1
        )
        ananya_ok = (
            abs(emp_scores.get("HR-EMP-00007", 0) - 3.4) < 0.05
            and emp_status.get("HR-EMP-00007") == 1
        )
        details = []
        if not vikram_ok:
            details.append(f"Vikram: score={emp_scores.get('HR-EMP-00005', 'N/A')}, status={emp_status.get('HR-EMP-00005', 'N/A')}")
        if not ananya_ok:
            details.append(f"Ananya: score={emp_scores.get('HR-EMP-00007', 'N/A')}, status={emp_status.get('HR-EMP-00007', 'N/A')}")
        check("5. Appraisals submitted with correct scores", 3,
              vikram_ok and ananya_ok, "; ".join(details) if details else "")
    except Exception as e:
        check("5. Appraisals submitted with correct scores", 3, False, f"exception: {e}")


def check_6_salary_component() -> None:
    """Verify salary component 'Performance Bonus' exists."""
    try:
        result = hrms_sql(
            "SELECT name, type FROM `tabSalary Component` "
            "WHERE name = 'Performance Bonus';"
        )
        check("6. Salary component Performance Bonus", 1, bool(result),
              "not found" if not result else "")
    except Exception as e:
        check("6. Salary component Performance Bonus", 1, False, f"exception: {e}")


def check_7_employee_incentives() -> None:
    """Verify Employee Incentives for both employees with correct amounts and submitted status."""
    try:
        result = hrms_sql(
            "SELECT employee, incentive_amount, docstatus "
            "FROM `tabEmployee Incentive` "
            "WHERE salary_component = 'Performance Bonus' "
            "AND payroll_date = '2026-01-31' "
            "ORDER BY employee;"
        )
        rows = result.splitlines() if result else []
        incentives = {}
        for row in rows:
            parts = row.split("\t")
            if len(parts) >= 3:
                incentives[parts[0].strip()] = (float(parts[1].strip()), int(parts[2].strip()))

        vikram_ok = incentives.get("HR-EMP-00005") == (15000.0, 1)
        ananya_ok = incentives.get("HR-EMP-00007") == (7500.0, 1)
        details = []
        if not vikram_ok:
            details.append(f"Vikram: {incentives.get('HR-EMP-00005', 'not found')}")
        if not ananya_ok:
            details.append(f"Ananya: {incentives.get('HR-EMP-00007', 'not found')}")
        check("7. Employee Incentives submitted", 3,
              vikram_ok and ananya_ok, "; ".join(details) if details else "")
    except Exception as e:
        check("7. Employee Incentives submitted", 3, False, f"exception: {e}")


def check_8_bigcapital_accounts() -> None:
    """Verify accounts 'Performance Bonus Expense' and 'Accrued Performance Bonus Payable' exist."""
    try:
        result = bigcapital_sql(
            "SELECT NAME FROM ACCOUNTS "
            "WHERE NAME IN ('Performance Bonus Expense', 'Accrued Performance Bonus Payable');"
        )
        found = set(line.strip() for line in result.splitlines() if line.strip())
        expected = {"Performance Bonus Expense", "Accrued Performance Bonus Payable"}
        missing = expected - found
        check("8. BigCapital accounts exist", 1, not missing,
              f"missing: {missing}" if missing else "")
    except Exception as e:
        check("8. BigCapital accounts exist", 1, False, f"exception: {e}")


def check_9_journal_entry() -> None:
    """Verify published journal entry dated 2025-12-31 with debit/credit of 22500."""
    try:
        result = bigcapital_sql(
            "SELECT e.CREDIT, e.DEBIT, a.NAME "
            "FROM MANUAL_JOURNALS_ENTRIES e "
            "JOIN MANUAL_JOURNALS j ON j.ID = e.MANUAL_JOURNAL_ID "
            "JOIN ACCOUNTS a ON a.ID = e.ACCOUNT_ID "
            "WHERE j.DATE = '2025-12-31' "
            "AND j.PUBLISHED_AT IS NOT NULL "
            "AND a.NAME IN ('Performance Bonus Expense', 'Accrued Performance Bonus Payable') "
            "ORDER BY a.NAME;"
        )
        rows = result.splitlines() if result else []
        entries = {}
        for row in rows:
            parts = row.split("\t")
            if len(parts) >= 3:
                credit = float(parts[0].strip() or "0")
                debit = float(parts[1].strip() or "0")
                acct = parts[2].strip()
                entries[acct] = {"credit": credit, "debit": debit}

        payable = entries.get("Accrued Performance Bonus Payable", {})
        expense = entries.get("Performance Bonus Expense", {})
        payable_ok = abs(payable.get("credit", 0) - 22500) < 0.01
        expense_ok = abs(expense.get("debit", 0) - 22500) < 0.01
        ok = payable_ok and expense_ok
        details = []
        if not expense_ok:
            details.append(f"expense debit={expense.get('debit', 'N/A')}, expected 22500")
        if not payable_ok:
            details.append(f"payable credit={payable.get('credit', 'N/A')}, expected 22500")
        check("9. Journal entry debit/credit 22500", 3, ok,
              "; ".join(details) if details else "")
    except Exception as e:
        check("9. Journal entry debit/credit 22500", 3, False, f"exception: {e}")


def check_10_twenty_note() -> None:
    """Verify note with appraisal cycle results containing all scores and bonus amounts."""
    try:
        result = twenty_sql(
            "SELECT title, \"bodyV2Markdown\" FROM note "
            "WHERE title LIKE '%Appraisal Cycle Results%H2 2025%';"
        )
        if not result.strip():
            check("10. Twenty note with appraisal results", 2, False, "note not found")
            return

        body_lower = result.lower()
        has_vikram = "vikram" in body_lower
        has_ananya = "ananya" in body_lower
        has_15000 = "15000" in result
        has_7500 = "7500" in result
        ok = has_vikram and has_ananya and has_15000 and has_7500
        missing = []
        if not has_vikram: missing.append("Vikram")
        if not has_ananya: missing.append("Ananya")
        if not has_15000: missing.append("15000")
        if not has_7500: missing.append("7500")
        check("10. Twenty note with appraisal results", 2, ok,
              f"missing in body: {missing}" if missing else "")
    except Exception as e:
        check("10. Twenty note with appraisal results", 2, False, f"exception: {e}")


def check_11_twenty_task_payroll() -> None:
    """Verify task 'Process bonus payroll' with due date 2026-01-31."""
    try:
        result = twenty_sql(
            'SELECT title, "dueAt" FROM task '
            "WHERE title LIKE '%Process bonus payroll%H2 2025%';"
        )
        if not result.strip():
            check("11. Twenty task: process bonus payroll", 1, False, "task not found")
            return
        has_due = "2026-01-31" in result
        check("11. Twenty task: process bonus payroll", 1, has_due,
              f"due date mismatch: {result}" if not has_due else "")
    except Exception as e:
        check("11. Twenty task: process bonus payroll", 1, False, f"exception: {e}")


def check_12_twenty_task_communicate() -> None:
    """Verify task 'Communicate appraisal results to employees' with due date 2026-01-15."""
    try:
        result = twenty_sql(
            'SELECT title, "dueAt" FROM task '
            "WHERE title LIKE '%Communicate appraisal results%';"
        )
        if not result.strip():
            check("12. Twenty task: communicate results", 1, False, "task not found")
            return
        has_due = "2026-01-15" in result
        check("12. Twenty task: communicate results", 1, has_due,
              f"due date mismatch: {result}" if not has_due else "")
    except Exception as e:
        check("12. Twenty task: communicate results", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_kras_exist()
    check_2_appraisal_template()
    check_3_goals_vikram()
    check_4_goals_ananya()
    check_5_appraisals_submitted()
    check_6_salary_component()
    check_7_employee_incentives()
    check_8_bigcapital_accounts()
    check_9_journal_entry()
    check_10_twenty_note()
    check_11_twenty_task_payroll()
    check_12_twenty_task_communicate()

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
