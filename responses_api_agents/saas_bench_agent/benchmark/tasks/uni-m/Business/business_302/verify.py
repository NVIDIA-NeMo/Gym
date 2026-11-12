# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_302/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-302-I4: End-to-End Recruitment Pipeline for Business Analyst

Checks: 14 weighted checks across hrms, bigcapital, twenty.
Strategy: docker exec MariaDB (hrms), REST API (bigcapital), docker exec Postgres (twenty)

Required env vars:
  SERVER_HOSTNAME, HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER
"""

import os
import sys
import subprocess
import json
import requests

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
    "HRMS_PORT": HRMS_PORT,
    "HRMS_CONTAINER": HRMS_CONTAINER,
    "HRMS_DB_CONTAINER": HRMS_DB_CONTAINER,
    "BIGCAPITAL_PORT": BIGCAPITAL_PORT,
    "BIGCAPITAL_CONTAINER": BIGCAPITAL_CONTAINER,
    "BIGCAPITAL_DB_CONTAINER": BIGCAPITAL_DB_CONTAINER,
    "TWENTY_PORT": TWENTY_PORT,
    "TWENTY_CONTAINER": TWENTY_CONTAINER,
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


_hrms_db_name: str | None = None


def _detect_hrms_db() -> str:
    """Find the Frappe site DB (has tabJob Applicant table)."""
    global _hrms_db_name
    if _hrms_db_name:
        return _hrms_db_name
    rc, out, err = docker_exec(
        HRMS_DB_CONTAINER,
        "mysql", "-uroot", f"-p{os.environ.get('HRMS_DB_ROOT_PASSWORD', 'hrms123456')}",
        "--default-character-set=utf8mb4", "-N", "-e",
        "SELECT TABLE_SCHEMA FROM information_schema.TABLES "
        "WHERE TABLE_NAME='tabJob Applicant' LIMIT 1",
    )
    if rc != 0:
        raise RuntimeError(f"mysql detect error: {err.strip()}")
    db = out.strip().split("\n")[0].strip()
    if not db:
        raise RuntimeError("no Frappe DB with tabJob Applicant found")
    _hrms_db_name = db
    return db


def hrms_sql(sql: str) -> str:
    """Query HRMS MariaDB, return raw stdout."""
    db = _detect_hrms_db()
    rc, out, err = docker_exec(
        HRMS_DB_CONTAINER,
        "mysql", "-uroot", f"-p{os.environ.get('HRMS_DB_ROOT_PASSWORD', 'hrms123456')}",
        "--default-character-set=utf8mb4",
        db, "-N", "-e", sql,
    )
    if rc != 0:
        raise RuntimeError(f"mysql error: {err.strip()}")
    return out.strip()


def twenty_sql(sql: str) -> str:
    """Query Twenty Postgres, return raw stdout."""
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c", sql,
    )
    if rc != 0:
        raise RuntimeError(f"psql error: {err.strip()}")
    return out.strip()


def get_twenty_workspace_schema() -> str:
    """Find the Twenty workspace schema name."""
    result = twenty_sql(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'workspace_%' ORDER BY schema_name LIMIT 1"
    )
    if not result:
        raise RuntimeError("no workspace schema found")
    return result.split("\n")[0].strip()


# ── BigCapital API helpers ────────────────────────────────────────────────────
_bc_token: str | None = None
_bc_org_id: str | None = None


def bigcapital_auth() -> tuple[str, str]:
    """Login to BigCapital, return (token, org_id). Cached."""
    global _bc_token, _bc_org_id
    if _bc_token and _bc_org_id:
        return _bc_token, _bc_org_id
    url = f"http://{HOST}:{BIGCAPITAL_PORT}/api/auth/signin"
    r = requests.post(url, json={
        "email": "admin@bigcapital.local",
        "password": "admin123",
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    _bc_token = data["access_token"]
    _bc_org_id = data.get("organization_id", "")
    return _bc_token, _bc_org_id


def bigcapital_get(path: str) -> dict:
    """GET a BigCapital API endpoint (auto-auth)."""
    token, org_id = bigcapital_auth()
    r = requests.get(
        f"http://{HOST}:{BIGCAPITAL_PORT}/api{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "organization-id": str(org_id),
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# ── HRMS Checks ───────────────────────────────────────────────────────────────

def check_1_job_requisition() -> None:
    """Job requisition submitted for Business Analyst in Human Resources - TVS"""
    try:
        result = hrms_sql(
            "SELECT name, docstatus, department, designation "
            "FROM `tabJob Requisition` "
            "WHERE designation='Business Analyst' "
            "AND department='Human Resources - TVS'"
        )
        if not result:
            check("1. Job requisition submitted", 1, False, "not found")
            return
        row = result.split("\n")[0].split("\t")
        docstatus = row[1] if len(row) > 1 else ""
        check("1. Job requisition submitted", 1, docstatus == "1",
              f"docstatus={docstatus}")
    except Exception as e:
        check("1. Job requisition submitted", 1, False, f"exception: {e}")


def check_2_job_opening() -> None:
    """Job opening for Business Analyst, status Open, 1 position"""
    try:
        result = hrms_sql(
            "SELECT name, status, vacancies "
            "FROM `tabJob Opening` "
            "WHERE designation='Business Analyst' "
            "AND department='Human Resources - TVS'"
        )
        if not result:
            check("2. Job opening Open with 1 position", 1, False, "not found")
            return
        row = result.split("\n")[0].split("\t")
        status = row[1].strip() if len(row) > 1 else ""
        vacancies = row[2].strip() if len(row) > 2 else ""
        passed = status == "Open" and str(vacancies) == "1"
        check("2. Job opening Open with 1 position", 1, passed,
              f"status={status}, vacancies={vacancies}")
    except Exception as e:
        check("2. Job opening Open with 1 position", 1, False, f"exception: {e}")


def check_3_karan_accepted() -> None:
    """Karan Mehta applicant status = Accepted"""
    try:
        result = hrms_sql(
            "SELECT status FROM `tabJob Applicant` "
            "WHERE email_id='karan.mehta@gmail.com'"
        )
        status = result.strip() if result else ""
        check("3. Karan Mehta status Accepted", 1, status == "Accepted",
              f"status={status!r}")
    except Exception as e:
        check("3. Karan Mehta status Accepted", 1, False, f"exception: {e}")


def check_4_divya_open() -> None:
    """Divya Pillai applicant status = Open"""
    try:
        result = hrms_sql(
            "SELECT status FROM `tabJob Applicant` "
            "WHERE email_id='divya.pillai@outlook.com'"
        )
        status = result.strip() if result else ""
        check("4. Divya Pillai status Open", 1, status == "Open",
              f"status={status!r}")
    except Exception as e:
        check("4. Divya Pillai status Open", 1, False, f"exception: {e}")


def check_5_rohit_rejected() -> None:
    """Rohit Nambiar applicant status = Rejected"""
    try:
        result = hrms_sql(
            "SELECT status FROM `tabJob Applicant` "
            "WHERE email_id='rohit.nambiar@yahoo.com'"
        )
        status = result.strip() if result else ""
        check("5. Rohit Nambiar status Rejected", 1, status == "Rejected",
              f"status={status!r}")
    except Exception as e:
        check("5. Rohit Nambiar status Rejected", 1, False, f"exception: {e}")


def check_6_interview_rounds() -> None:
    """Both interview rounds exist"""
    try:
        result = hrms_sql(
            "SELECT name FROM `tabInterview Round` "
            "WHERE name IN ('HR Analytical Skills Test','HR Director Final Round')"
        )
        found = set(line.strip() for line in result.split("\n")) if result else set()
        has_r1 = "HR Analytical Skills Test" in found
        has_r2 = "HR Director Final Round" in found
        check("6. Interview rounds exist", 1, has_r1 and has_r2,
              f"round1={has_r1}, round2={has_r2}")
    except Exception as e:
        check("6. Interview rounds exist", 1, False, f"exception: {e}")


def check_7_round1_feedback() -> None:
    """Round 1: 3 interviews with correct results (Karan Cleared, Divya Cleared, Rohit Rejected)"""
    try:
        result = hrms_sql(
            "SELECT ja.applicant_name, i.status "
            "FROM `tabInterview` i "
            "JOIN `tabJob Applicant` ja ON i.job_applicant = ja.name "
            "WHERE i.interview_round='HR Analytical Skills Test' "
            "AND i.docstatus=1"
        )
        if not result:
            check("7. Round 1 feedback", 2, False, "no interviews found")
            return
        interviews = {}
        for line in result.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                interviews[parts[0].strip()] = parts[1].strip()
        expected = {
            "Karan Mehta": "Cleared",
            "Divya Pillai": "Cleared",
            "Rohit Nambiar": "Rejected",
        }
        issues = []
        for name, exp_result in expected.items():
            actual = interviews.get(name)
            if actual is None:
                issues.append(f"{name} missing")
            elif actual != exp_result:
                issues.append(f"{name} status={actual}, expected {exp_result}")
        check("7. Round 1 feedback", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("7. Round 1 feedback", 2, False, f"exception: {e}")


def check_8_round2_feedback() -> None:
    """Round 2: Karan and Divya Cleared"""
    try:
        result = hrms_sql(
            "SELECT ja.applicant_name, i.status "
            "FROM `tabInterview` i "
            "JOIN `tabJob Applicant` ja ON i.job_applicant = ja.name "
            "WHERE i.interview_round='HR Director Final Round' "
            "AND i.docstatus=1"
        )
        if not result:
            check("8. Round 2 feedback", 2, False, "no interviews found")
            return
        interviews = {}
        for line in result.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                interviews[parts[0].strip()] = parts[1].strip()
        issues = []
        for name in ("Karan Mehta", "Divya Pillai"):
            actual = interviews.get(name)
            if actual is None:
                issues.append(f"{name} missing")
            elif actual != "Cleared":
                issues.append(f"{name} status={actual}, expected Cleared")
        check("8. Round 2 feedback", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("8. Round 2 feedback", 2, False, f"exception: {e}")


def check_9_job_offer() -> None:
    """Job offer for Karan Mehta with designation Business Analyst and base salary 1050000"""
    try:
        applicant_id = hrms_sql(
            "SELECT name FROM `tabJob Applicant` "
            "WHERE email_id='karan.mehta@gmail.com'"
        ).strip()
        if not applicant_id:
            check("9. Job offer for Karan Mehta", 2, False, "applicant not found")
            return
        result = hrms_sql(
            f"SELECT name, designation, offer_date "
            f"FROM `tabJob Offer` "
            f"WHERE job_applicant='{applicant_id}'"
        )
        if not result:
            check("9. Job offer for Karan Mehta", 2, False, "no job offer found")
            return
        row = result.split("\n")[0].split("\t")
        offer_name = row[0].strip()
        designation = row[1].strip() if len(row) > 1 else ""
        # Check offer terms for base salary
        terms = hrms_sql(
            f"SELECT offer_term, value FROM `tabJob Offer Term` "
            f"WHERE parent='{offer_name}'"
        )
        has_salary = "1050000" in terms if terms else False
        has_designation = designation == "Business Analyst"
        check("9. Job offer for Karan Mehta", 2, has_salary and has_designation,
              f"designation={designation}, salary_in_terms={has_salary}")
    except Exception as e:
        check("9. Job offer for Karan Mehta", 2, False, f"exception: {e}")


# ── BigCapital Checks (REST API) ─────────────────────────────────────────────

def check_10_expense_account() -> None:
    """HR Recruitment Cost Account exists as Expense type"""
    try:
        data = bigcapital_get("/accounts?search=HR+Recruitment+Cost+Account")
        accounts = data.get("accounts", [])
        found = None
        for a in accounts:
            if a.get("name") == "HR Recruitment Cost Account":
                found = a
                break
        if not found:
            check("10. Expense account exists", 1, False, "account not found")
            return
        acct_type = found.get("account_type", "")
        passed = acct_type.lower() == "expense"
        check("10. Expense account exists", 1, passed, f"type={acct_type}")
    except Exception as e:
        check("10. Expense account exists", 1, False, f"exception: {e}")


def check_11_expenses() -> None:
    """Two published expenses (4800 + 650 = 5450) under HR Recruitment Cost Account"""
    try:
        data = bigcapital_get("/expenses?page_size=200")
        expenses = data.get("expenses", [])
        relevant_amounts = []
        for exp in expenses:
            # Check if published
            if not exp.get("published_at") and not exp.get("is_published"):
                continue
            # Check categories for the HR Recruitment Cost Account
            for cat in exp.get("categories", []):
                acct = cat.get("expense_account", {})
                if acct.get("name") == "HR Recruitment Cost Account":
                    relevant_amounts.append(float(cat.get("amount", 0)))
        has_4800 = any(abs(a - 4800) < 0.01 for a in relevant_amounts)
        has_650 = any(abs(a - 650) < 0.01 for a in relevant_amounts)
        total = sum(relevant_amounts)
        passed = has_4800 and has_650 and abs(total - 5450) < 0.01
        check("11. Expense entries total 5450", 2, passed,
              f"amounts={relevant_amounts}, total={total}")
    except Exception as e:
        check("11. Expense entries total 5450", 2, False, f"exception: {e}")


# ── Twenty CRM Checks (docker exec Postgres) ─────────────────────────────────

def check_12_onboarding_task() -> None:
    """Task 'Onboard Karan Mehta - Business Analyst' with correct body content"""
    try:
        ws = get_twenty_workspace_schema()
        result = twenty_sql(
            f"SELECT title, \"bodyV2Markdown\" FROM \"{ws}\".task "
            f"WHERE title = 'Onboard Karan Mehta - Business Analyst' "
            f"AND \"deletedAt\" IS NULL"
        )
        if not result:
            check("12. Onboarding task", 2, False, "task not found")
            return
        # Result format: title|bodyV2Markdown
        parts = result.split("|", 1)
        body = parts[1] if len(parts) > 1 else ""
        body_lower = body.lower()
        has_offer = "offer accepted" in body_lower
        has_salary = "1050000" in body
        has_cost = "5450" in body
        passed = has_offer and has_salary and has_cost
        check("12. Onboarding task", 2, passed,
              f"offer_accepted={has_offer}, salary={has_salary}, cost={has_cost}")
    except Exception as e:
        check("12. Onboarding task", 2, False, f"exception: {e}")


def check_13_rejection_task() -> None:
    """Task 'Send rejection notifications - Business Analyst recruitment' with correct body"""
    try:
        ws = get_twenty_workspace_schema()
        result = twenty_sql(
            f"SELECT title, \"bodyV2Markdown\" FROM \"{ws}\".task "
            f"WHERE title = 'Send rejection notifications - Business Analyst recruitment' "
            f"AND \"deletedAt\" IS NULL"
        )
        if not result:
            check("13. Rejection notification task", 2, False, "task not found")
            return
        parts = result.split("|", 1)
        body = parts[1] if len(parts) > 1 else ""
        has_rohit = "rohit.nambiar@yahoo.com" in body.lower()
        has_divya = "divya pillai" in body.lower()
        passed = has_rohit and has_divya
        check("13. Rejection notification task", 2, passed,
              f"has_rohit={has_rohit}, has_divya={has_divya}")
    except Exception as e:
        check("13. Rejection notification task", 2, False, f"exception: {e}")


def check_14_recruitment_note() -> None:
    """Note 'Recruitment Summary - Business Analyst - 2026-08-19' with key content"""
    try:
        ws = get_twenty_workspace_schema()
        result = twenty_sql(
            f"SELECT title, \"bodyV2Markdown\" FROM \"{ws}\".note "
            f"WHERE title = 'Recruitment Summary - Business Analyst - 2026-08-19' "
            f"AND \"deletedAt\" IS NULL"
        )
        if not result:
            check("14. Recruitment summary note", 2, False, "note not found")
            return
        parts = result.split("|", 1)
        body = parts[1] if len(parts) > 1 else ""
        has_accepted = "ACCEPTED" in body
        has_waitlisted = "Waitlisted" in body
        has_rejected = "REJECTED" in body
        has_total = "5450" in body
        passed = has_accepted and has_waitlisted and has_rejected and has_total
        check("14. Recruitment summary note", 2, passed,
              f"ACCEPTED={has_accepted}, Waitlisted={has_waitlisted}, "
              f"REJECTED={has_rejected}, total_5450={has_total}")
    except Exception as e:
        check("14. Recruitment summary note", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_job_requisition()
    check_2_job_opening()
    check_3_karan_accepted()
    check_4_divya_open()
    check_5_rohit_rejected()
    check_6_interview_rounds()
    check_7_round1_feedback()
    check_8_round2_feedback()
    check_9_job_offer()
    check_10_expense_account()
    check_11_expenses()
    check_12_onboarding_task()
    check_13_rejection_task()
    check_14_recruitment_note()

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
