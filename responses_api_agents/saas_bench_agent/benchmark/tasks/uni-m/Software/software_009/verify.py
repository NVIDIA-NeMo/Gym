# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_009/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-009-I2: Establish ADR Governance Workflow

Checks: 13 weighted checks across code-server, baserow, openproject.
Strategy: docker exec (code-server filesystem, openproject DB), REST API (baserow)

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import sys
import subprocess
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_PORT = os.environ.get("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
OPENPROJECT_PORT = os.environ.get("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

_required = {
    "CODE_SERVER_PORT": CODE_SERVER_PORT,
    "CODE_SERVER_CONTAINER": CODE_SERVER_CONTAINER,
    "BASEROW_PORT": BASEROW_PORT,
    "BASEROW_CONTAINER": BASEROW_CONTAINER,
    "BASEROW_DB_CONTAINER": BASEROW_DB_CONTAINER,
    "OPENPROJECT_PORT": OPENPROJECT_PORT,
    "OPENPROJECT_CONTAINER": OPENPROJECT_CONTAINER,
}
for var, val in _required.items():
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"

# ── Slot values ───────────────────────────────────────────────────────────────
ADR_FILENAMES = [
    "005-secrets-management.md",
    "006-container-orchestration.md",
    "007-backup-strategy.md",
]
ADR_TITLES = [
    "Secrets Management Solution for Production Workloads",
    "Container Orchestration Platform Selection",
    "Backup and Disaster Recovery Strategy",
]
ADR_CONTEXTS = [
    "We need to centralize secret storage and rotation to eliminate plaintext credentials from source control and improve auditability.",
    "We need to select a container orchestration platform capable of running stateful and stateless workloads with automated scaling and self-healing.",
    "We need to define a comprehensive backup and disaster recovery strategy that meets our RTO and RPO targets across all tier-1 systems.",
]
ADR_NUMBERS = ["005", "006", "007"]
ADR_REVIEWERS = ["Emma Wilson", "Frank Nguyen", "Grace Patel"]
ADR_DATE = "2025-06-10"
ADR_AUTHOR = "DevOps Engineering Guild"
ADR_STATUS = "Review"

# Expected file contents (5 lines each)
EXPECTED_CONTENTS = {}
for i in range(3):
    EXPECTED_CONTENTS[ADR_FILENAMES[i]] = (
        f"# ADR-{ADR_NUMBERS[i]}: {ADR_TITLES[i]}\n"
        f"Status: {ADR_STATUS}\n"
        f"Date: {ADR_DATE}\n"
        f"Author: {ADR_AUTHOR}\n"
        f"{ADR_CONTEXTS[i]}"
    )


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


def baserow_auth() -> str:
    """Get Baserow JWT access_token."""
    resp = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    # Baserow returns both 'token' and 'access_token'; use access_token with JWT prefix
    return data.get("access_token", data.get("token", ""))


def baserow_get(path: str, token: str) -> requests.Response:
    return requests.get(
        f"{BASEROW_URL}/api{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )


def op_db_query(sql: str) -> str:
    """Run a SQL query against the OpenProject embedded Postgres DB."""
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER, "bash", "-c",
        f"PGPASSWORD=openproject psql -U openproject -d openproject -h 127.0.0.1 -t -A -c \"{sql}\"",
        timeout=20,
    )
    if rc != 0:
        raise RuntimeError(f"psql failed (rc={rc}): {err.strip()}")
    return out.strip()


# ── Individual checks ─────────────────────────────────────────────────────────

ADR_DIR = "/home/coder/project/devops-configs/docs/adr"


def check_1_adr_directory_exists() -> None:
    """Check that devops-configs/docs/adr/ directory exists in code-server."""
    try:
        rc, out, err = docker_exec(CODE_SERVER_CONTAINER, "test", "-d", ADR_DIR)
        check("1. ADR directory exists", 1, rc == 0,
              "directory not found" if rc != 0 else "")
    except Exception as e:
        check("1. ADR directory exists", 1, False, f"exception: {e}")


def check_2_file_005_content() -> None:
    """Check 005-secrets-management.md has correct 5-line content."""
    fname = ADR_FILENAMES[0]
    try:
        rc, out, err = docker_exec(CODE_SERVER_CONTAINER, "cat", f"{ADR_DIR}/{fname}")
        if rc != 0:
            check("2. 005-secrets-management.md content", 2, False, "file not found")
            return
        actual = out.rstrip("\n")
        expected = EXPECTED_CONTENTS[fname]
        passed = actual == expected
        detail = "" if passed else f"content mismatch: got {len(actual)} chars, expected {len(expected)}"
        check("2. 005-secrets-management.md content", 2, passed, detail)
    except Exception as e:
        check("2. 005-secrets-management.md content", 2, False, f"exception: {e}")


def check_3_file_006_content() -> None:
    """Check 006-container-orchestration.md has correct 5-line content."""
    fname = ADR_FILENAMES[1]
    try:
        rc, out, err = docker_exec(CODE_SERVER_CONTAINER, "cat", f"{ADR_DIR}/{fname}")
        if rc != 0:
            check("3. 006-container-orchestration.md content", 2, False, "file not found")
            return
        actual = out.rstrip("\n")
        expected = EXPECTED_CONTENTS[fname]
        passed = actual == expected
        detail = "" if passed else f"content mismatch: got {len(actual)} chars, expected {len(expected)}"
        check("3. 006-container-orchestration.md content", 2, passed, detail)
    except Exception as e:
        check("3. 006-container-orchestration.md content", 2, False, f"exception: {e}")


def check_4_file_007_content() -> None:
    """Check 007-backup-strategy.md has correct 5-line content."""
    fname = ADR_FILENAMES[2]
    try:
        rc, out, err = docker_exec(CODE_SERVER_CONTAINER, "cat", f"{ADR_DIR}/{fname}")
        if rc != 0:
            check("4. 007-backup-strategy.md content", 2, False, "file not found")
            return
        actual = out.rstrip("\n")
        expected = EXPECTED_CONTENTS[fname]
        passed = actual == expected
        detail = "" if passed else f"content mismatch: got {len(actual)} chars, expected {len(expected)}"
        check("4. 007-backup-strategy.md content", 2, passed, detail)
    except Exception as e:
        check("4. 007-backup-strategy.md content", 2, False, f"exception: {e}")


def check_5_exactly_3_files() -> None:
    """Check that adr directory contains exactly 3 .md files."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER, "bash", "-c",
            f"ls -1 {ADR_DIR}/*.md 2>/dev/null | wc -l"
        )
        if rc != 0:
            check("5. Exactly 3 .md files in adr dir", 1, False, "ls failed")
            return
        count = int(out.strip())
        check("5. Exactly 3 .md files in adr dir", 1, count == 3,
              f"found {count} files" if count != 3 else "")
    except Exception as e:
        check("5. Exactly 3 .md files in adr dir", 1, False, f"exception: {e}")


def check_6_baserow_database_exists() -> None:
    """Check Baserow database 'ADR Decision Registry' exists."""
    try:
        token = baserow_auth()
        resp = baserow_get("/applications/", token)
        resp.raise_for_status()
        apps = resp.json()
        found = any(
            a.get("name") == "ADR Decision Registry"
            for a in apps
        )
        check("6. Baserow DB 'ADR Decision Registry' exists", 1, found,
              "database not found" if not found else "")
    except Exception as e:
        check("6. Baserow DB 'ADR Decision Registry' exists", 1, False, f"exception: {e}")


def _get_baserow_table(token: str):
    """Find the ADR Registry table and return table_id or None."""
    resp = baserow_get("/applications/", token)
    resp.raise_for_status()
    apps = resp.json()
    for app in apps:
        if app.get("name") == "ADR Decision Registry":
            for table in app.get("tables", []):
                if table.get("name") == "ADR Registry":
                    return table["id"]
    return None


def check_7_baserow_table_fields() -> None:
    """Check ADR Registry table exists with correct field structure."""
    try:
        token = baserow_auth()
        table_id = _get_baserow_table(token)
        if table_id is None:
            check("7. Baserow table 'ADR Registry' with fields", 2, False,
                  "table not found")
            return
        resp = baserow_get(f"/database/fields/table/{table_id}/", token)
        resp.raise_for_status()
        fields = resp.json()
        field_names = {f["name"] for f in fields}
        expected_fields = {"ADR ID", "Title", "Status", "Author", "Reviewer",
                           "Created Date", "Review Duration Days"}
        missing = expected_fields - field_names
        passed = len(missing) == 0
        detail = f"missing fields: {missing}" if missing else ""
        check("7. Baserow table 'ADR Registry' with fields", 2, passed, detail)
    except Exception as e:
        check("7. Baserow table 'ADR Registry' with fields", 2, False, f"exception: {e}")


def check_8_baserow_rows_adr_ids_titles() -> None:
    """Check ADR Registry has exactly 3 rows with correct ADR IDs and titles."""
    try:
        token = baserow_auth()
        table_id = _get_baserow_table(token)
        if table_id is None:
            check("8. Baserow 3 rows with ADR IDs/Titles", 2, False, "table not found")
            return
        resp = baserow_get(f"/database/rows/table/{table_id}/?user_field_names=true", token)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("results", [])
        if len(rows) != 3:
            check("8. Baserow 3 rows with ADR IDs/Titles", 2, False,
                  f"expected 3 rows, got {len(rows)}")
            return
        expected_ids = {f"ADR-{n}" for n in ADR_NUMBERS}
        expected_titles = set(ADR_TITLES)
        actual_ids = set()
        actual_titles = set()
        for row in rows:
            adr_id = str(row.get("ADR ID", "")).strip()
            title = str(row.get("Title", "")).strip()
            actual_ids.add(adr_id)
            actual_titles.add(title)
        id_ok = expected_ids == actual_ids
        title_ok = expected_titles == actual_titles
        passed = id_ok and title_ok
        details = []
        if not id_ok:
            details.append(f"IDs: expected {expected_ids}, got {actual_ids}")
        if not title_ok:
            details.append(f"Titles: expected {expected_titles}, got {actual_titles}")
        check("8. Baserow 3 rows with ADR IDs/Titles", 2, passed,
              "; ".join(details) if details else "")
    except Exception as e:
        check("8. Baserow 3 rows with ADR IDs/Titles", 2, False, f"exception: {e}")


def check_9_baserow_rows_status_reviewer_date() -> None:
    """Check rows have Status=Review, correct Reviewers, Date=2025-06-10, Duration=0."""
    try:
        token = baserow_auth()
        table_id = _get_baserow_table(token)
        if table_id is None:
            check("9. Baserow row details (Status/Reviewer/Date/Duration)", 2, False,
                  "table not found")
            return
        resp = baserow_get(f"/database/rows/table/{table_id}/?user_field_names=true", token)
        resp.raise_for_status()
        rows = resp.json().get("results", [])
        issues = []
        expected_reviewers = set(ADR_REVIEWERS)
        actual_reviewers = set()
        for row in rows:
            # Status - single select field returns dict with value key
            status = row.get("Status", {})
            if isinstance(status, dict):
                status_val = status.get("value", "")
            else:
                status_val = str(status)
            if status_val != ADR_STATUS:
                issues.append(f"row Status={status_val}, expected {ADR_STATUS}")

            reviewer = str(row.get("Reviewer", "")).strip()
            actual_reviewers.add(reviewer)

            # Created Date
            date_val = str(row.get("Created Date", "")).strip()
            if not date_val.startswith(ADR_DATE):
                issues.append(f"row date={date_val}, expected {ADR_DATE}")

            # Review Duration Days
            duration = row.get("Review Duration Days")
            duration_str = str(duration).strip() if duration is not None else ""
            if duration_str not in ("0", "0.0", "0.00"):
                issues.append(f"row duration={duration}, expected 0")

        if expected_reviewers != actual_reviewers:
            issues.append(f"reviewers: expected {expected_reviewers}, got {actual_reviewers}")

        passed = len(issues) == 0
        check("9. Baserow row details (Status/Reviewer/Date/Duration)", 2, passed,
              "; ".join(issues[:3]) if issues else "")
    except Exception as e:
        check("9. Baserow row details (Status/Reviewer/Date/Duration)", 2, False,
              f"exception: {e}")


def check_10_openproject_project_exists() -> None:
    """Check OpenProject project 'DevOps Automation' exists via DB."""
    try:
        result = op_db_query("SELECT name FROM projects WHERE name = 'DevOps Automation'")
        found = result == "DevOps Automation"
        check("10. OpenProject project 'DevOps Automation' exists", 1, found,
              "project not found" if not found else "")
    except Exception as e:
        check("10. OpenProject project 'DevOps Automation' exists", 1, False,
              f"exception: {e}")


def _get_op_epics() -> list[dict]:
    """Get Epic work packages in 'DevOps Automation' project from DB.
    Returns list of dicts with subject, assigned_to_id, priority_name, description.
    """
    sql = (
        "SELECT wp.subject, wp.assigned_to_id, e.name AS priority, wp.description "
        "FROM work_packages wp "
        "JOIN projects p ON wp.project_id = p.id "
        "JOIN enumerations e ON wp.priority_id = e.id "
        "WHERE p.name = 'DevOps Automation' AND wp.type_id = 5"
    )
    result = op_db_query(sql)
    if not result:
        return []
    epics = []
    for line in result.split("\n"):
        parts = line.split("|", 3)
        if len(parts) >= 4:
            epics.append({
                "subject": parts[0],
                "assigned_to_id": parts[1],
                "priority": parts[2],
                "description": parts[3],
            })
    return epics


def check_11_openproject_3_epics_subjects() -> None:
    """Check 3 Epic work packages with correct subjects."""
    try:
        epics = _get_op_epics()
        if len(epics) != 3:
            check("11. 3 Epic WPs with correct subjects", 2, False,
                  f"expected 3 epics, got {len(epics)}")
            return
        expected_subjects = {
            f"Implement ADR-{ADR_NUMBERS[i]}: {ADR_TITLES[i]}" for i in range(3)
        }
        actual_subjects = {e["subject"] for e in epics}
        passed = expected_subjects == actual_subjects
        detail = ""
        if not passed:
            missing = expected_subjects - actual_subjects
            detail = f"missing subjects: {missing}" if missing else f"unexpected: {actual_subjects - expected_subjects}"
        check("11. 3 Epic WPs with correct subjects", 2, passed, detail)
    except Exception as e:
        check("11. 3 Epic WPs with correct subjects", 2, False, f"exception: {e}")


def check_12_openproject_epics_assignee_priority() -> None:
    """Check epics have assignee=OpenProject Admin (user id for 'admin') and priority=Normal."""
    try:
        # Get admin user id
        admin_id = op_db_query(
            "SELECT id FROM users WHERE login = 'admin' AND admin = true"
        )
        if not admin_id:
            check("12. Epics assignee/priority", 2, False, "admin user not found in DB")
            return

        epics = _get_op_epics()
        if not epics:
            check("12. Epics assignee/priority", 2, False, "no epics found")
            return
        issues = []
        for ep in epics:
            if ep["assigned_to_id"] != admin_id:
                issues.append(f"'{ep['subject'][:30]}...' assignee_id={ep['assigned_to_id']}, expected {admin_id}")
            if ep["priority"] != "Normal":
                issues.append(f"'{ep['subject'][:30]}...' priority={ep['priority']}")
        passed = len(issues) == 0
        check("12. Epics assignee/priority", 2, passed,
              "; ".join(issues[:3]) if issues else "")
    except Exception as e:
        check("12. Epics assignee/priority", 2, False, f"exception: {e}")


def check_13_openproject_epics_description() -> None:
    """Check epic descriptions contain 'Linked ADR file: devops-configs/docs/adr/<filename>'."""
    try:
        epics = _get_op_epics()
        if not epics:
            check("13. Epic descriptions contain ADR file path", 2, False,
                  "no epics found")
            return
        issues = []
        for ep in epics:
            subj = ep["subject"]
            desc = ep["description"] or ""
            matched = False
            for i in range(3):
                expected_subj = f"Implement ADR-{ADR_NUMBERS[i]}: {ADR_TITLES[i]}"
                if subj == expected_subj:
                    expected_path = f"Linked ADR file: devops-configs/docs/adr/{ADR_FILENAMES[i]}"
                    if expected_path not in desc:
                        issues.append(
                            f"'{ADR_FILENAMES[i]}' path not in description"
                        )
                    matched = True
                    break
            if not matched:
                issues.append(f"unrecognized epic subject: '{subj[:40]}...'")
        passed = len(issues) == 0
        check("13. Epic descriptions contain ADR file path", 2, passed,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("13. Epic descriptions contain ADR file path", 2, False,
              f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_adr_directory_exists()
    check_2_file_005_content()
    check_3_file_006_content()
    check_4_file_007_content()
    check_5_exactly_3_files()
    check_6_baserow_database_exists()
    check_7_baserow_table_fields()
    check_8_baserow_rows_adr_ids_titles()
    check_9_baserow_rows_status_reviewer_date()
    check_10_openproject_project_exists()
    check_11_openproject_3_epics_subjects()
    check_12_openproject_epics_assignee_priority()
    check_13_openproject_epics_description()

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
