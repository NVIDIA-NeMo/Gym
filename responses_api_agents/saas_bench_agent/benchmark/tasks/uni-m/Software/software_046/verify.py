# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_046/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-046-I4: Release candidate QA sign-off workflow for data-analyzer v3.2.0-rc1

Checks: 12 weighted checks across openproject, code-server, baserow.
Strategy: docker exec (OpenProject DB, code-server filesystem), REST API (Baserow)

Required env vars:
  SERVER_HOSTNAME, OPENPROJECT_PORT, OPENPROJECT_CONTAINER,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER
"""

import os
import sys
import subprocess
import json
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENPROJECT_PORT = os.getenv("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.getenv("OPENPROJECT_CONTAINER")
CODE_SERVER_PORT = os.getenv("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.getenv("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.getenv("BASEROW_PORT")
BASEROW_CONTAINER = os.getenv("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.getenv("BASEROW_DB_CONTAINER")

for var_name, var_val in [
    ("OPENPROJECT_PORT", OPENPROJECT_PORT),
    ("OPENPROJECT_CONTAINER", OPENPROJECT_CONTAINER),
    ("CODE_SERVER_PORT", CODE_SERVER_PORT),
    ("CODE_SERVER_CONTAINER", CODE_SERVER_CONTAINER),
    ("BASEROW_PORT", BASEROW_PORT),
    ("BASEROW_CONTAINER", BASEROW_CONTAINER),
    ("BASEROW_DB_CONTAINER", BASEROW_DB_CONTAINER),
]:
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
        sys.exit(1)

OPENPROJECT_BASE = f"http://{HOST}:{OPENPROJECT_PORT}"
BASEROW_BASE = f"http://{HOST}:{BASEROW_PORT}"

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


def op_db_query(sql: str, sep: str = "|") -> str:
    """Query OpenProject's embedded PostgreSQL."""
    r = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=openproject",
         OPENPROJECT_CONTAINER,
         "psql", "-U", "openproject", "-h", "127.0.0.1", "-d", "openproject",
         "-t", "-A", "-F", sep, "-c", sql],
        capture_output=True, text=True, timeout=15,
    )
    return r.stdout.strip()


def baserow_api_get(path: str, token: str) -> dict:
    r = requests.get(
        f"{BASEROW_BASE}/api/{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_baserow_token() -> str:
    r = requests.post(
        f"{BASEROW_BASE}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]


# ── Check 1: OpenProject version v3.2.0-rc1 ──────────────────────────────────
def check_1_op_version() -> None:
    """OpenProject version v3.2.0-rc1 with correct dates and description."""
    try:
        SEP = "^^^"
        row = op_db_query(
            "SELECT v.name, v.start_date, v.effective_date, v.description, v.status "
            "FROM versions v "
            "JOIN projects p ON p.id = v.project_id "
            "WHERE p.identifier = 'data-analytics-pipeline' "
            "AND v.name = 'v3.2.0-rc1'",
            sep=SEP,
        )
        if not row:
            check("1. OP version v3.2.0-rc1", 2, False, "version not found")
            return
        parts = row.split(SEP)
        name = parts[0]
        start = parts[1] if len(parts) > 1 else ""
        due = parts[2] if len(parts) > 2 else ""
        desc = parts[3] if len(parts) > 3 else ""
        status = parts[4] if len(parts) > 4 else ""
        ok = (
            name == "v3.2.0-rc1"
            and start == "2025-07-07"
            and due == "2025-07-16"
            and "Release candidate for v3.2.0" in desc
            and "sign-off window 2025-07-07 to 2025-07-16" in desc.lower()
            and status == "open"
        )
        check("1. OP version v3.2.0-rc1", 2, ok,
              f"start={start}, due={due}, status={status}, desc={desc[:60]}")
    except Exception as e:
        check("1. OP version v3.2.0-rc1", 2, False, f"exception: {e}")


# ── Check 2: Git commit in code-server ────────────────────────────────────────
def check_2_git_commit() -> None:
    """Commit with exact message 'chore(release): prepare 3.2.0-rc1 candidate' exists."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "bash", "-c",
            "cd /home/coder/workspace/data-analyzer && "
            "git log --oneline --all --grep='chore(release): prepare 3.2.0-rc1 candidate' --format='%H %s'",
            timeout=15,
        )
        lines = [l for l in out.strip().split("\n") if l.strip()]
        found = any("chore(release): prepare 3.2.0-rc1 candidate" in l for l in lines)
        check("2. Git commit message", 2, found,
              f"found {len(lines)} matching commit(s)" if found else "commit not found")
    except Exception as e:
        check("2. Git commit message", 2, False, f"exception: {e}")


# ── Check 3: pyproject.toml version ───────────────────────────────────────────
def check_3_pyproject_version() -> None:
    """Manifest file (pyproject.toml or setup.py) has version 3.2.0-rc1."""
    try:
        # Check pyproject.toml first, then setup.py
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "bash", "-c",
            "cd /home/coder/workspace/data-analyzer && "
            "{ grep -E 'version' pyproject.toml 2>/dev/null || grep -E 'version' setup.py 2>/dev/null; }",
            timeout=15,
        )
        has_rc = "3.2.0-rc1" in out
        check("3. Manifest version", 1, has_rc,
              f"line: {out.strip()}" if out.strip() else "version line not found")
    except Exception as e:
        check("3. Manifest version", 1, False, f"exception: {e}")


# ── Check 4: CHANGELOG.md entries ─────────────────────────────────────────────
def check_4_changelog() -> None:
    """CHANGELOG.md has the three new lines below ## [Unreleased]."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "bash", "-c",
            "cd /home/coder/workspace/data-analyzer && cat CHANGELOG.md",
            timeout=15,
        )
        has_header = "## [3.2.0-rc1] - 2025-07-07" in out
        has_candidate = "### Candidate for v3.2.0" in out
        has_window = "- Sign-off window: 2025-07-07 to 2025-07-16" in out
        all_ok = has_header and has_candidate and has_window
        missing = []
        if not has_header:
            missing.append("header line")
        if not has_candidate:
            missing.append("candidate line")
        if not has_window:
            missing.append("window line")
        check("4. CHANGELOG.md entries", 2, all_ok,
              "all 3 lines present" if all_ok else f"missing: {', '.join(missing)}")
    except Exception as e:
        check("4. CHANGELOG.md entries", 2, False, f"exception: {e}")


# ── Check 5: Baserow database exists ─────────────────────────────────────────
def check_5_baserow_db(token: str) -> int | None:
    """Database 'RC Sign-off v3.2.0' exists in Baserow."""
    try:
        apps = baserow_api_get("applications/", token)
        db_id = None
        for app in apps:
            if app.get("name") == "RC Sign-off v3.2.0" and app.get("type") == "database":
                db_id = app["id"]
                break
        check("5. Baserow DB exists", 1, db_id is not None,
              f"id={db_id}" if db_id else "database not found")
        return db_id
    except Exception as e:
        check("5. Baserow DB exists", 1, False, f"exception: {e}")
        return None


# ── Check 6: Sign-off Criteria table with 5 rows ─────────────────────────────
def check_6_criteria_table(token: str, db_id: int) -> tuple[int | None, list]:
    """Sign-off Criteria table has 5 rows with correct names and categories."""
    try:
        tables = baserow_api_get(f"database/tables/database/{db_id}/", token)
        criteria_table = None
        for t in tables:
            if t.get("name") == "Sign-off Criteria":
                criteria_table = t
                break
        if not criteria_table:
            check("6. Sign-off Criteria rows", 2, False, "table not found")
            return None, []
        table_id = criteria_table["id"]
        rows_resp = baserow_api_get(f"database/rows/table/{table_id}/?user_field_names=true&size=50", token)
        rows = rows_resp.get("results", [])
        expected_names = [
            "Data ingestion pipeline validated",
            "Analysis throughput >= 1M rows/min",
            "No secrets or credentials in repo",
            "User manual and API reference refreshed",
            "Database migration rollback rehearsed",
        ]
        found_names = []
        for row in rows:
            # The "Criterion Name" field or look for the name in any text field
            for key, val in row.items():
                if isinstance(val, str) and val in expected_names:
                    found_names.append(val)
                    break
        ok = len(rows) == 5 and len(found_names) == 5
        check("6. Sign-off Criteria rows", 2, ok,
              f"{len(rows)} rows, {len(found_names)}/5 names match")
        return table_id, rows
    except Exception as e:
        check("6. Sign-off Criteria rows", 2, False, f"exception: {e}")
        return None, []


# ── Check 7: Sign-off Criteria statuses ───────────────────────────────────────
def check_7_criteria_statuses(rows: list) -> None:
    """4 criteria Passed, 1 (Documentation) Failed."""
    try:
        # Expected: SC-01 Passed, SC-02 Passed, SC-03 Passed, SC-04 Failed, SC-05 Passed
        status_map = {}
        for row in rows:
            # Find the primary field (Criterion ID) and Status field
            crit_id = None
            status_val = None
            for key, val in row.items():
                if isinstance(val, str) and val.startswith("SC-"):
                    crit_id = val
                if isinstance(val, dict) and "value" in val:
                    # single-select field
                    status_val = val.get("value")
                elif key.lower().replace(" ", "") == "status" or "status" in key.lower():
                    if isinstance(val, str) and val in ("Pending", "Passed", "Failed"):
                        status_val = val
            if crit_id:
                status_map[crit_id] = status_val

        expected_statuses = {
            "SC-01": "Passed", "SC-02": "Passed", "SC-03": "Passed",
            "SC-04": "Failed", "SC-05": "Passed",
        }
        mismatches = []
        for cid, exp in expected_statuses.items():
            got = status_map.get(cid)
            if got != exp:
                mismatches.append(f"{cid}: expected {exp}, got {got}")
        ok = len(mismatches) == 0 and len(status_map) == 5
        check("7. Criteria statuses", 2, ok,
              "4 Passed, 1 Failed" if ok else f"mismatches: {'; '.join(mismatches)}")
    except Exception as e:
        check("7. Criteria statuses", 2, False, f"exception: {e}")


# ── Check 8: Stakeholder Approvals 6 rows ────────────────────────────────────
def check_8_approvals(token: str, db_id: int) -> tuple[int | None, list]:
    """Stakeholder Approvals table has 6 rows with correct approver data."""
    try:
        tables = baserow_api_get(f"database/tables/database/{db_id}/", token)
        approvals_table = None
        for t in tables:
            if t.get("name") == "Stakeholder Approvals":
                approvals_table = t
                break
        if not approvals_table:
            check("8. Stakeholder Approvals rows", 2, False, "table not found")
            return None, []
        table_id = approvals_table["id"]
        rows_resp = baserow_api_get(
            f"database/rows/table/{table_id}/?user_field_names=true&size=50", token
        )
        rows = rows_resp.get("results", [])

        expected_approvers = [
            "Paul Garcia", "Paul Garcia", "Thomas Nickson",
            "Nora Mott", "Michael Robicheaux", "Sandra Love",
        ]
        found_names = []
        for row in rows:
            for key, val in row.items():
                if isinstance(val, str) and val in expected_approvers:
                    found_names.append(val)
                    break

        ok = len(rows) == 6
        check("8. Stakeholder Approvals rows", 2, ok,
              f"{len(rows)} rows, names: {found_names}")
        return table_id, rows
    except Exception as e:
        check("8. Stakeholder Approvals rows", 2, False, f"exception: {e}")
        return None, []


# ── Check 9: Approval IDs and Submitted At ───────────────────────────────────
def check_9_approval_ids(rows: list) -> None:
    """Approval IDs AP-001..AP-006 and all Submitted At = 2025-07-14."""
    try:
        approval_ids = []
        dates_ok = True
        for row in rows:
            for key, val in row.items():
                if isinstance(val, str) and val.startswith("AP-"):
                    approval_ids.append(val)
            # Check Submitted At
            date_found = False
            for key, val in row.items():
                if ("submitted" in key.lower() or "date" in key.lower()) and isinstance(val, str):
                    if "2025-07-14" in val:
                        date_found = True
            if not date_found:
                dates_ok = False

        expected_ids = ["AP-001", "AP-002", "AP-003", "AP-004", "AP-005", "AP-006"]
        ids_ok = sorted(approval_ids) == expected_ids
        ok = ids_ok and dates_ok
        check("9. Approval IDs & dates", 1, ok,
              f"IDs={sorted(approval_ids)}, dates_ok={dates_ok}")
    except Exception as e:
        check("9. Approval IDs & dates", 1, False, f"exception: {e}")


# ── Check 10: Form view exists ────────────────────────────────────────────────
def check_10_form_view(token: str, approvals_table_id: int) -> None:
    """Form view 'RC Sign-off Form' exists on Stakeholder Approvals table."""
    try:
        views = baserow_api_get(f"database/views/table/{approvals_table_id}/", token)
        form_view = None
        for v in views:
            if v.get("type") == "form" and "RC Sign-off Form" in v.get("name", ""):
                form_view = v
                break
        ok = form_view is not None
        detail = f"view id={form_view['id']}" if form_view else "form view not found"
        check("10. Form view exists", 1, ok, detail)
    except Exception as e:
        check("10. Form view exists", 1, False, f"exception: {e}")


# ── Check 11: OpenProject Milestone WP ────────────────────────────────────────
def check_11_milestone_wp() -> None:
    """Milestone 'RC Sign-off: v3.2.0-rc1' exists with correct description."""
    try:
        SEP = "^^^"
        row = op_db_query(
            "SELECT wp.subject, wp.description, t.name as type_name "
            "FROM work_packages wp "
            "JOIN types t ON t.id = wp.type_id "
            "JOIN projects p ON p.id = wp.project_id "
            "WHERE p.identifier = 'data-analytics-pipeline' "
            "AND wp.subject = 'RC Sign-off: v3.2.0-rc1'",
            sep=SEP,
        )
        if not row:
            check("11. Milestone WP", 2, False, "work package not found")
            return
        parts = row.split(SEP)
        subject = parts[0]
        desc = parts[1] if len(parts) > 1 else ""
        type_name = parts[2] if len(parts) > 2 else ""
        is_milestone = "milestone" in type_name.lower()
        has_passed = "Passed: 4" in desc or "Passed:4" in desc
        has_failed = "Failed: 1" in desc or "Failed:1" in desc
        has_pending = "Pending: 0" in desc or "Pending:0" in desc
        desc_ok = has_passed and has_failed and has_pending
        ok = subject == "RC Sign-off: v3.2.0-rc1" and is_milestone and desc_ok
        check("11. Milestone WP", 2, ok,
              f"type={type_name}, desc has P/F/N={has_passed}/{has_failed}/{has_pending}")
    except Exception as e:
        check("11. Milestone WP", 2, False, f"exception: {e}")


# ── Check 12: Bug WP for failed criterion ────────────────────────────────────
def check_12_bug_wp() -> None:
    """Bug WP 'Fix before GA: User manual and API reference refreshed' is child of milestone."""
    try:
        SEP = "^^^"
        row = op_db_query(
            "SELECT wp.subject, wp.description, t.name as type_name, "
            "parent.subject as parent_subject, "
            "u.login as assignee_login "
            "FROM work_packages wp "
            "JOIN types t ON t.id = wp.type_id "
            "JOIN projects p ON p.id = wp.project_id "
            "LEFT JOIN work_packages parent ON parent.id = wp.parent_id "
            "LEFT JOIN users u ON u.id = wp.assigned_to_id "
            "WHERE p.identifier = 'data-analytics-pipeline' "
            "AND wp.subject = 'Fix before GA: User manual and API reference refreshed'",
            sep=SEP,
        )
        if not row:
            check("12. Bug WP for failed criterion", 2, False, "work package not found")
            return
        parts = row.split(SEP)
        subject = parts[0]
        desc = parts[1] if len(parts) > 1 else ""
        type_name = parts[2] if len(parts) > 2 else ""
        parent_subject = parts[3] if len(parts) > 3 else ""
        assignee = parts[4] if len(parts) > 4 else ""
        is_bug = "bug" in type_name.lower()
        parent_ok = parent_subject == "RC Sign-off: v3.2.0-rc1"
        assignee_ok = "michael.robicheaux" in assignee
        desc_has_category = "Category: Documentation" in desc
        desc_has_target = "docs/ site builds and CHANGELOG is current" in desc
        desc_has_role = "Required Approver Role: ProductOwner" in desc
        ok = is_bug and parent_ok and assignee_ok and desc_has_category and desc_has_target and desc_has_role
        check("12. Bug WP for failed criterion", 2, ok,
              f"type={type_name}, parent={parent_subject}, assignee={assignee}, "
              f"desc_ok={desc_has_category and desc_has_target and desc_has_role}")
    except Exception as e:
        check("12. Bug WP for failed criterion", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # OpenProject checks
    check_1_op_version()

    # code-server checks
    check_2_git_commit()
    check_3_pyproject_version()
    check_4_changelog()

    # Baserow checks
    try:
        br_token = get_baserow_token()
    except Exception as e:
        check("5. Baserow DB exists", 1, False, f"auth failed: {e}")
        check("6. Sign-off Criteria rows", 2, False, "skipped (no auth)")
        check("7. Criteria statuses", 2, False, "skipped (no auth)")
        check("8. Stakeholder Approvals rows", 2, False, "skipped (no auth)")
        check("9. Approval IDs & dates", 1, False, "skipped (no auth)")
        check("10. Form view exists", 1, False, "skipped (no auth)")
        # Continue to OpenProject checks
        check_11_milestone_wp()
        check_12_bug_wp()
        # Score
        total = sum(w for _, w, _, _ in _checks)
        earned = sum(w for _, w, p, _ in _checks if p)
        all_pass = all(p for _, _, p, _ in _checks) and bool(_checks)
        score = (earned / total) if total else 0.0
        print(f"SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})", file=sys.stderr)
        sys.exit(0 if all_pass else 1)
        return

    db_id = check_5_baserow_db(br_token)
    if db_id:
        criteria_table_id, criteria_rows = check_6_criteria_table(br_token, db_id)
        if criteria_rows:
            check_7_criteria_statuses(criteria_rows)
        else:
            check("7. Criteria statuses", 2, False, "no criteria rows to check")

        approvals_table_id, approvals_rows = check_8_approvals(br_token, db_id)
        if approvals_rows:
            check_9_approval_ids(approvals_rows)
        else:
            check("9. Approval IDs & dates", 1, False, "no approval rows to check")

        if approvals_table_id:
            check_10_form_view(br_token, approvals_table_id)
        else:
            check("10. Form view exists", 1, False, "approvals table not found")
    else:
        check("6. Sign-off Criteria rows", 2, False, "skipped (no DB)")
        check("7. Criteria statuses", 2, False, "skipped (no DB)")
        check("8. Stakeholder Approvals rows", 2, False, "skipped (no DB)")
        check("9. Approval IDs & dates", 1, False, "skipped (no DB)")
        check("10. Form view exists", 1, False, "skipped (no DB)")

    # More OpenProject checks
    check_11_milestone_wp()
    check_12_bug_wp()

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
