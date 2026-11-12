# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_031/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-031-I3: Offboard Ananya Reddy with HR Separation, Payroll Settlement, and CRM Task Reassignment

Checks: 10 weighted checks across hrms, bigcapital, twenty.
Strategy: docker exec (DB queries) for all three sites.

Required env vars:
  SERVER_HOSTNAME, HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER
"""

import os
import sys
import subprocess
import json

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


def _find_hrms_frappe_db() -> str:
    """Discover the Frappe bench database name in HRMS MariaDB."""
    rc, out, err = docker_exec(
        HRMS_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "root", "-phrms123456",
        "-N", "-B", "-e",
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
        "WHERE SCHEMA_NAME LIKE '\\_%' AND SCHEMA_NAME != 'information_schema' "
        "ORDER BY SCHEMA_NAME;",
    )
    if rc != 0 or not out.strip():
        return "_frappe_bench"
    # Pick the DB that has tabEmployee
    for db in out.strip().split("\n"):
        db = db.strip()
        rc2, out2, _ = docker_exec(
            HRMS_DB_CONTAINER,
            "mysql", "--default-character-set=utf8mb4",
            "-u", "root", "-phrms123456",
            "-D", db, "-N", "-B", "-e",
            "SHOW TABLES LIKE 'tabEmployee';",
        )
        if rc2 == 0 and out2.strip():
            return db
    return "_frappe_bench"


_hrms_db_cache: str | None = None


def hrms_sql(query: str) -> str:
    """Run a MariaDB query against the Frappe HRMS database."""
    global _hrms_db_cache
    if _hrms_db_cache is None:
        _hrms_db_cache = _find_hrms_frappe_db()
    rc, out, err = docker_exec(
        HRMS_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "root", "-phrms123456",
        "-D", _hrms_db_cache,
        "-N", "-B", "-e", query,
    )
    if rc != 0:
        raise RuntimeError(f"hrms mysql error: {err.strip()}")
    return out.strip()


def _find_bigcapital_tenant_db() -> str:
    """Discover the BigCapital tenant database name."""
    rc, out, err = docker_exec(
        BIGCAPITAL_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "root", "-proot123",
        "-N", "-B", "-e",
        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
        "WHERE SCHEMA_NAME LIKE 'bigcapital_tenant_%' ORDER BY SCHEMA_NAME LIMIT 1;",
    )
    if rc != 0 or not out.strip():
        return "bigcapital"
    return out.strip().split("\n")[0]


_bc_db_cache: str | None = None


def bigcapital_sql(query: str) -> str:
    """Run a MariaDB query against the BigCapital tenant database."""
    global _bc_db_cache
    if _bc_db_cache is None:
        _bc_db_cache = _find_bigcapital_tenant_db()
    rc, out, err = docker_exec(
        BIGCAPITAL_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "root", "-proot123",
        "-D", _bc_db_cache,
        "-N", "-B", "-e", query,
    )
    if rc != 0:
        raise RuntimeError(f"bigcapital mysql error: {err.strip()}")
    return out.strip()


def twenty_sql(query: str) -> str:
    """Run a Postgres query against Twenty database (default schema)."""
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default",
        "-t", "-A", "-c", query,
    )
    if rc != 0:
        raise RuntimeError(f"twenty psql error: {err.strip()}")
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


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_employee_separation() -> None:
    """Employee Separation exists for HR-EMP-00007 with date 2026-06-30, submitted."""
    try:
        row = hrms_sql(
            "SELECT name, boarding_status, docstatus FROM `tabEmployee Separation` "
            "WHERE employee = 'HR-EMP-00007' AND boarding_begins_on = '2026-06-30' LIMIT 1;"
        )
        if not row:
            check("1. Employee Separation exists", 1, False, "no record found for HR-EMP-00007 with date 2026-06-30")
            return
        parts = row.split("\t")
        docstatus = int(parts[2]) if len(parts) > 2 else -1
        check("1. Employee Separation exists", 1, docstatus == 1,
              f"name={parts[0]}, docstatus={docstatus}")
    except Exception as e:
        check("1. Employee Separation exists", 1, False, f"exception: {e}")


def check_2_exit_activities() -> None:
    """Three exit activities with correct names and assignees."""
    try:
        rows = hrms_sql(
            "SELECT a.activity_name, a.user "
            "FROM `tabEmployee Boarding Activity` a "
            "JOIN `tabEmployee Separation` s ON a.parent = s.name "
            "WHERE s.employee = 'HR-EMP-00007' AND s.boarding_begins_on = '2026-06-30' "
            "ORDER BY a.activity_name;"
        )
        if not rows:
            check("2. Exit activities (3 with correct assignees)", 2, False, "no activities found")
            return

        activities = {}
        for line in rows.split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                activities[parts[0]] = parts[1]

        # The 'user' field stores email addresses (e.g. rajesh.kumar@...) not full names.
        # Match by checking if a lowercase version of the name (dot-separated) appears in the email.
        expected = {
            "Conduct exit interview": "pooja.malhotra",
            "Return company laptop": "rajesh.kumar",
            "Revoke system access": "rajesh.kumar",
        }

        issues = []
        for act_name, assignee_fragment in expected.items():
            found_user = activities.get(act_name, "")
            if not found_user:
                issues.append(f"'{act_name}' missing")
            elif assignee_fragment not in found_user.lower():
                issues.append(f"'{act_name}' assigned to '{found_user}' not matching '{assignee_fragment}'")

        check("2. Exit activities (3 with correct assignees)", 2, not issues,
              "all 3 correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("2. Exit activities (3 with correct assignees)", 2, False, f"exception: {e}")


def check_5_bigcapital_vendor() -> None:
    """Vendor 'Ananya Reddy - Ex Employee' exists with correct email."""
    try:
        row = bigcapital_sql(
            "SELECT DISPLAY_NAME, EMAIL "
            "FROM CONTACTS "
            "WHERE DISPLAY_NAME LIKE '%Ananya Reddy%Ex Employee%' "
            "OR LOWER(DISPLAY_NAME) = 'ananya reddy - ex employee' "
            "LIMIT 1;"
        )
        if not row:
            row = bigcapital_sql(
                "SELECT DISPLAY_NAME, EMAIL "
                "FROM CONTACTS "
                "WHERE CONTACT_SERVICE = 'vendor' "
                "AND (DISPLAY_NAME LIKE '%Ananya Reddy%' OR FIRST_NAME LIKE '%Ananya%') "
                "LIMIT 1;"
            )
        if not row:
            check("5. Vendor 'Ananya Reddy - Ex Employee'", 1, False, "vendor not found")
            return
        parts = row.split("\t")
        display = parts[0].strip() if len(parts) > 0 else ""
        email = parts[1].strip() if len(parts) > 1 else ""
        email_ok = "ananya.reddy@gmail.com" in email.lower()
        check("5. Vendor 'Ananya Reddy - Ex Employee'", 1, email_ok,
              f"display_name={display}, email={email}")
    except Exception as e:
        check("5. Vendor 'Ananya Reddy - Ex Employee'", 1, False, f"exception: {e}")


def check_6_journal_entry() -> None:
    """Journal entry dated 2026-06-30 with correct memo and 3 line items."""
    try:
        journal_row = bigcapital_sql(
            "SELECT ID, DATE, DESCRIPTION, (PUBLISHED_AT IS NOT NULL) FROM MANUAL_JOURNALS "
            "WHERE DESCRIPTION LIKE '%Final settlement%Ananya Reddy%2026-06-30%' "
            "AND DATE = '2026-06-30' LIMIT 1;"
        )
        if not journal_row:
            check("6. Journal entry (settlement, 3 lines)", 3, False, "journal not found with matching memo/date")
            return
        parts = journal_row.split("\t")
        journal_id = parts[0].strip()
        published = parts[3].strip() if len(parts) > 3 else ""

        entries = bigcapital_sql(
            f"SELECT a.NAME, e.CREDIT, e.DEBIT "
            f"FROM MANUAL_JOURNALS_ENTRIES e "
            f"JOIN ACCOUNTS a ON e.ACCOUNT_ID = a.ID "
            f"WHERE e.MANUAL_JOURNAL_ID = {journal_id} "
            f"ORDER BY e.DEBIT DESC;"
        )
        if not entries:
            check("6. Journal entry (settlement, 3 lines)", 3, False, "no journal entries found")
            return

        debits = {}
        credits = {}
        for line in entries.split("\n"):
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            acct = cols[0].strip()
            cr = float(cols[1]) if cols[1].strip() else 0.0
            dr = float(cols[2]) if cols[2].strip() else 0.0
            if dr > 0:
                debits[acct] = dr
            if cr > 0:
                credits[acct] = cr

        issues = []
        rent_dr = debits.get("Rent", 0)
        if abs(rent_dr - 57950.0) > 0.01:
            issues.append(f"Rent debit expected 57950, got {rent_dr}")
        adv_dr = debits.get("Advertising Expense", 0)
        if abs(adv_dr - 29000.0) > 0.01:
            issues.append(f"Advertising Expense debit expected 29000, got {adv_dr}")
        obl_cr = credits.get("Opening Balance Liabilities", 0)
        if abs(obl_cr - 86950.0) > 0.01:
            issues.append(f"Opening Balance Liabilities credit expected 86950, got {obl_cr}")
        if published not in ("1", "t", "true", "True"):
            issues.append(f"journal not published (published={published})")

        check("6. Journal entry (settlement, 3 lines)", 3, not issues,
              "correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("6. Journal entry (settlement, 3 lines)", 3, False, f"exception: {e}")


def check_7_payment_made() -> None:
    """Payment Made of 86950 to vendor from Sales of Product Income."""
    try:
        row = bigcapital_sql(
            "SELECT bp.AMOUNT, bp.PAYMENT_DATE, a.NAME, c.DISPLAY_NAME "
            "FROM BILLS_PAYMENTS bp "
            "LEFT JOIN ACCOUNTS a ON bp.PAYMENT_ACCOUNT_ID = a.ID "
            "LEFT JOIN CONTACTS c ON bp.VENDOR_ID = c.ID "
            "WHERE bp.AMOUNT = 86950 "
            "AND bp.PAYMENT_DATE = '2026-07-05' "
            "LIMIT 1;"
        )
        if not row:
            check("7. Payment Made (86950 to vendor)", 2, False, "payment not found")
            return
        parts = row.split("\t")
        amount = float(parts[0]) if parts[0] else 0
        account = parts[2].strip() if len(parts) > 2 else ""
        vendor = parts[3].strip() if len(parts) > 3 else ""
        issues = []
        if abs(amount - 86950.0) > 0.01:
            issues.append(f"amount expected 86950, got {amount}")
        if "Sales of Product Income" not in account:
            issues.append(f"account expected 'Sales of Product Income', got '{account}'")
        if "Ananya Reddy" not in vendor:
            issues.append(f"vendor expected 'Ananya Reddy - Ex Employee', got '{vendor}'")
        check("7. Payment Made (86950 to vendor)", 2, not issues,
              "correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("7. Payment Made (86950 to vendor)", 2, False, f"exception: {e}")


def check_8_twenty_tasks_titles() -> None:
    """3 tasks linked to MetricStream with correct titles."""
    try:
        ws = get_twenty_workspace_schema()
        expected_titles = [
            "Schedule MetricStream compliance review meeting",
            "Update MetricStream primary contact details",
            "Follow up on MetricStream contract renewal",
        ]

        found_titles = []
        for title in expected_titles:
            safe_title = title.replace("'", "''")
            row = twenty_sql(
                f"SELECT t.title FROM \"{ws}\".task t "
                f"WHERE t.title = '{safe_title}' LIMIT 1;"
            )
            if row:
                found_titles.append(title)

        missing = [t for t in expected_titles if t not in found_titles]
        check("8. Twenty tasks (3 with correct titles)", 2, not missing,
              f"all 3 found" if not missing else f"missing: {missing}")
    except Exception as e:
        check("8. Twenty tasks (3 with correct titles)", 2, False, f"exception: {e}")


def check_9_twenty_tasks_details() -> None:
    """Tasks have correct due date 2026-07-20 and body text."""
    try:
        ws = get_twenty_workspace_schema()
        expected_body = (
            "Reassigned from Ananya Reddy (separated 2026-06-30). "
            "Original responsibility transferred — review and update client contacts."
        )
        expected_titles = [
            "Schedule MetricStream compliance review meeting",
            "Update MetricStream primary contact details",
            "Follow up on MetricStream contract renewal",
        ]
        issues = []
        for title in expected_titles:
            safe_title = title.replace("'", "''")
            row = twenty_sql(
                f"SELECT t.\"dueAt\"::text, t.\"bodyV2Markdown\" FROM \"{ws}\".task t "
                f"WHERE t.title = '{safe_title}' LIMIT 1;"
            )
            if not row:
                issues.append(f"'{title}' not found")
                continue
            parts = row.split("|", 1)
            due = parts[0].strip() if parts else ""
            body = parts[1].strip() if len(parts) > 1 else ""
            if "2026-07-20" not in due:
                issues.append(f"'{title}' due date={due}, expected 2026-07-20")
            if expected_body not in body:
                issues.append(f"'{title}' body mismatch")

        check("9. Twenty tasks (due date & body)", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("9. Twenty tasks (due date & body)", 2, False, f"exception: {e}")


def check_10_twenty_note() -> None:
    """Note with correct title and body."""
    try:
        ws = get_twenty_workspace_schema()
        expected_title = "Employee Separation Complete — Ananya Reddy"
        expected_body = (
            "Separation date: 2026-06-30. Final settlement: 86,950.00 "
            "(salary: 57,950.00, leave encashment: 29,000.00). "
            "Payment processed 2026-07-05 from Sales of Product Income. "
            "3 client tasks reassigned to company MetricStream."
        )
        safe_title = expected_title.replace("'", "''")

        # Note: Twenty stores notes with title and body fields
        row = twenty_sql(
            f"SELECT n.title, n.\"bodyV2Markdown\" FROM \"{ws}\".note n "
            f"WHERE n.title = '{safe_title}' LIMIT 1;"
        )
        if not row:
            # Try partial match
            row = twenty_sql(
                f"SELECT n.title, n.\"bodyV2Markdown\" FROM \"{ws}\".note n "
                f"WHERE n.title LIKE '%Employee Separation Complete%Ananya Reddy%' LIMIT 1;"
            )
        if not row:
            check("10. Twenty note (separation summary)", 2, False, "note not found")
            return
        parts = row.split("|", 1)
        title = parts[0].strip() if parts else ""
        body = parts[1].strip() if len(parts) > 1 else ""

        issues = []
        if expected_title not in title:
            issues.append(f"title mismatch: got '{title}'")
        if expected_body not in body:
            issues.append(f"body mismatch")

        check("10. Twenty note (separation summary)", 2, not issues,
              "correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("10. Twenty note (separation summary)", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_employee_separation()
    check_2_exit_activities()
    check_5_bigcapital_vendor()
    check_6_journal_entry()
    check_7_payment_made()
    check_8_twenty_tasks_titles()
    check_9_twenty_tasks_details()
    check_10_twenty_note()

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
