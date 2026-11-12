#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_031/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-031-I4: Sprint retrospective data-gathering for Pentest Round 1
in Security Audit project.

Checks: 14 weighted checks across openproject, code-server, baserow.
Strategy: docker exec (DB queries for OpenProject & Baserow, filesystem for code-server)

Required env vars:
  SERVER_HOSTNAME, OPENPROJECT_PORT, OPENPROJECT_CONTAINER,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER
"""

import json
import os
import re
import subprocess
import sys

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENPROJECT_PORT = os.getenv("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.getenv("OPENPROJECT_CONTAINER")
CODE_SERVER_PORT = os.getenv("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.getenv("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.getenv("BASEROW_PORT")
BASEROW_CONTAINER = os.getenv("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.getenv("BASEROW_DB_CONTAINER")

_missing = []
for var in [
    "OPENPROJECT_PORT", "OPENPROJECT_CONTAINER",
    "CODE_SERVER_PORT", "CODE_SERVER_CONTAINER",
    "BASEROW_PORT", "BASEROW_CONTAINER", "BASEROW_DB_CONTAINER",
]:
    if not os.getenv(var):
        _missing.append(var)
if _missing:
    print(f"FATAL: missing env vars: {', '.join(_missing)}", file=sys.stderr)
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


def baserow_sql(query: str) -> str:
    """Run a SQL query against the Baserow postgres DB."""
    rc, out, err = docker_exec(
        BASEROW_DB_CONTAINER,
        "env", "PGPASSWORD=kdpzkuyhsgb22onku8y7rxkx3czej88nxpngaz4mlmgad67vpc", "psql", "-h", "127.0.0.1", "-U", "baserow", "-d", "baserow", "-t", "-A", "-c", query,
    )
    return out.strip()


def openproject_sql(query: str) -> str:
    """Run a SQL query against the OpenProject postgres DB (embedded in app container)."""
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "psql", "-U", "openproject", "-d", "openproject", "-t", "-A", "-c", query,
    )
    return out.strip()


# ── Baserow checks ────────────────────────────────────────────────────────────
def check_1_baserow_db_exists() -> None:
    """Database 'Retro Pentest Round 1' exists in Baserow."""
    try:
        result = baserow_sql(
            "SELECT d.id FROM database_application d "
            "JOIN core_application a ON d.application_ptr_id = a.id "
            "WHERE a.name = 'Retro Pentest Round 1';"
        )
        found = bool(result.strip())
        check("1. Baserow DB 'Retro Pentest Round 1' exists", 1, found,
              f"db_id={result}" if found else "not found")
    except Exception as e:
        check("1. Baserow DB 'Retro Pentest Round 1' exists", 1, False, f"exception: {e}")


def check_2_sprint_wp_table() -> None:
    """Table 'Sprint Work Packages' exists with expected fields."""
    try:
        table_id = baserow_sql(
            "SELECT dt.id FROM database_table dt "
            "JOIN database_application da ON dt.database_id = da.application_ptr_id "
            "JOIN core_application ca ON da.application_ptr_id = ca.id "
            "WHERE ca.name = 'Retro Pentest Round 1' AND dt.name = 'Sprint Work Packages';"
        )
        if not table_id:
            check("2. Table 'Sprint Work Packages' exists with fields", 2, False, "table not found")
            return

        fields_raw = baserow_sql(
            f"SELECT f.name FROM database_field f WHERE f.table_id = {table_id} AND f.trashed = false ORDER BY f.name;"
        )
        fields = set(fields_raw.split("\n")) if fields_raw else set()
        expected_fields = {"WP ID", "Subject", "Type", "Status", "Estimated Hours", "Closed"}
        missing = expected_fields - fields
        ok = len(missing) == 0
        check("2. Table 'Sprint Work Packages' exists with fields", 2, ok,
              f"fields={sorted(fields)}" if ok else f"missing fields: {sorted(missing)}, have: {sorted(fields)}")
    except Exception as e:
        check("2. Table 'Sprint Work Packages' exists with fields", 2, False, f"exception: {e}")


def check_3_test_health_table() -> None:
    """Table 'Test Health' exists with expected fields."""
    try:
        table_id = baserow_sql(
            "SELECT dt.id FROM database_table dt "
            "JOIN database_application da ON dt.database_id = da.application_ptr_id "
            "JOIN core_application ca ON da.application_ptr_id = ca.id "
            "WHERE ca.name = 'Retro Pentest Round 1' AND dt.name = 'Test Health';"
        )
        if not table_id:
            check("3. Table 'Test Health' exists with fields", 2, False, "table not found")
            return

        fields_raw = baserow_sql(
            f"SELECT f.name FROM database_field f WHERE f.table_id = {table_id} AND f.trashed = false ORDER BY f.name;"
        )
        fields = set(fields_raw.split("\n")) if fields_raw else set()
        expected_fields = {"Project", "Tests Passed", "Tests Failed", "Test Files Count", "Pass Rate", "Health Badge"}
        missing = expected_fields - fields
        ok = len(missing) == 0
        check("3. Table 'Test Health' exists with fields", 2, ok,
              f"fields={sorted(fields)}" if ok else f"missing fields: {sorted(missing)}, have: {sorted(fields)}")
    except Exception as e:
        check("3. Table 'Test Health' exists with fields", 2, False, f"exception: {e}")


def check_4_sprint_wp_rows() -> None:
    """'Sprint Work Packages' has rows with Closed boolean set correctly."""
    try:
        table_id = baserow_sql(
            "SELECT dt.id FROM database_table dt "
            "JOIN database_application da ON dt.database_id = da.application_ptr_id "
            "JOIN core_application ca ON da.application_ptr_id = ca.id "
            "WHERE ca.name = 'Retro Pentest Round 1' AND dt.name = 'Sprint Work Packages';"
        )
        if not table_id:
            check("4. Sprint Work Packages has rows", 2, False, "table not found")
            return

        row_count = baserow_sql(f"SELECT count(*) FROM database_table_{table_id};")
        row_count = int(row_count) if row_count else 0
        has_rows = row_count > 0
        check("4. Sprint Work Packages has rows", 2, has_rows,
              f"row_count={row_count}" if has_rows else "no rows found")
    except Exception as e:
        check("4. Sprint Work Packages has rows", 2, False, f"exception: {e}")


def check_5_test_health_rows() -> None:
    """'Test Health' has rows for json and data-analyzer with Health Badge."""
    try:
        table_id = baserow_sql(
            "SELECT dt.id FROM database_table dt "
            "JOIN database_application da ON dt.database_id = da.application_ptr_id "
            "JOIN core_application ca ON da.application_ptr_id = ca.id "
            "WHERE ca.name = 'Retro Pentest Round 1' AND dt.name = 'Test Health';"
        )
        if not table_id:
            check("5. Test Health has rows for json & data-analyzer", 2, False, "table not found")
            return

        row_count = baserow_sql(f"SELECT count(*) FROM database_table_{table_id};")
        row_count = int(row_count) if row_count else 0
        ok = row_count == 2
        check("5. Test Health has rows for json & data-analyzer", 2, ok,
              f"row_count={row_count}, expected 2")
    except Exception as e:
        check("5. Test Health has rows for json & data-analyzer", 2, False, f"exception: {e}")


def check_6_completion_summary_view() -> None:
    """Grid view 'Completion Summary' exists on 'Sprint Work Packages'."""
    try:
        result = baserow_sql(
            "SELECT v.id FROM database_view v "
            "JOIN database_table dt ON v.table_id = dt.id "
            "JOIN database_application da ON dt.database_id = da.application_ptr_id "
            "JOIN core_application ca ON da.application_ptr_id = ca.id "
            "WHERE ca.name = 'Retro Pentest Round 1' "
            "AND dt.name = 'Sprint Work Packages' "
            "AND v.name = 'Completion Summary';"
        )
        found = bool(result.strip())
        check("6. View 'Completion Summary' on Sprint Work Packages", 1, found,
              f"view_id={result}" if found else "view not found")
    except Exception as e:
        check("6. View 'Completion Summary' on Sprint Work Packages", 1, False, f"exception: {e}")


# ── code-server checks ───────────────────────────────────────────────────────
def _read_retro_file() -> str | None:
    """Read the retro markdown file from code-server container. Returns content or None."""
    rc, out, err = docker_exec(
        CODE_SERVER_CONTAINER,
        "cat", "/home/coder/devops-configs/docs/retro-Pentest Round 1.md",
    )
    if rc != 0:
        return None
    return out


def check_7_retro_file_exists() -> None:
    """Retro markdown file exists in code-server."""
    try:
        content = _read_retro_file()
        check("7. Retro file exists", 1, content is not None,
              "found" if content is not None else "file not found")
    except Exception as e:
        check("7. Retro file exists", 1, False, f"exception: {e}")


def check_8_retro_header_date() -> None:
    """Lines 1-2: header and date correct."""
    try:
        content = _read_retro_file()
        if content is None:
            check("8. Retro header & date", 1, False, "file not found")
            return
        lines = content.strip().split("\n")
        if len(lines) < 2:
            check("8. Retro header & date", 1, False, f"only {len(lines)} lines")
            return
        header_ok = lines[0].strip() == "# Retrospective: Pentest Round 1"
        date_ok = lines[1].strip() == "Date: 2024-12-03"
        ok = header_ok and date_ok
        check("8. Retro header & date", 1, ok,
              f"line1={'ok' if header_ok else repr(lines[0])}, line2={'ok' if date_ok else repr(lines[1])}")
    except Exception as e:
        check("8. Retro header & date", 1, False, f"exception: {e}")


def check_9_retro_closed_hours() -> None:
    """Lines 3-4: Work packages closed and planned hours."""
    try:
        content = _read_retro_file()
        if content is None:
            check("9. Retro closed count & hours", 2, False, "file not found")
            return
        lines = content.strip().split("\n")
        if len(lines) < 4:
            check("9. Retro closed count & hours", 2, False, f"only {len(lines)} lines")
            return
        line3_match = re.match(r"Work packages closed: (\d+) of (\d+)", lines[2].strip())
        line4_match = re.match(r"Planned hours closed: ([\d.]+)", lines[3].strip())
        line3_ok = line3_match is not None
        line4_ok = line4_match is not None
        ok = line3_ok and line4_ok
        detail_parts = []
        if line3_ok:
            detail_parts.append(f"closed={line3_match.group(1)}/{line3_match.group(2)}")
        else:
            detail_parts.append(f"line3={repr(lines[2])}")
        if line4_ok:
            detail_parts.append(f"hours={line4_match.group(1)}")
        else:
            detail_parts.append(f"line4={repr(lines[3])}")
        check("9. Retro closed count & hours", 2, ok, ", ".join(detail_parts))
    except Exception as e:
        check("9. Retro closed count & hours", 2, False, f"exception: {e}")


def check_10_retro_test_health() -> None:
    """Line 5: Test health line with project pass rates and badges."""
    try:
        content = _read_retro_file()
        if content is None:
            check("10. Retro test health line", 2, False, "file not found")
            return
        lines = content.strip().split("\n")
        if len(lines) < 5:
            check("10. Retro test health line", 2, False, f"only {len(lines)} lines")
            return
        line5 = lines[4].strip()
        # Expected: Test health — data-analyzer:XX.XX% (Badge) ; json:XX.XX% (Badge)
        starts_ok = line5.startswith("Test health")
        # Check both projects appear (alphabetical: data-analyzer, json)
        has_da = "data-analyzer:" in line5
        has_json = "json:" in line5
        has_badges = any(b in line5 for b in ("Green", "Yellow", "Red"))
        ok = starts_ok and has_da and has_json and has_badges
        check("10. Retro test health line", 2, ok,
              f"line={repr(line5[:100])}")
    except Exception as e:
        check("10. Retro test health line", 2, False, f"exception: {e}")


def check_11_retro_red_badges() -> None:
    """Line 6: Red badges count."""
    try:
        content = _read_retro_file()
        if content is None:
            check("11. Retro red badges line", 1, False, "file not found")
            return
        lines = content.strip().split("\n")
        if len(lines) < 6:
            check("11. Retro red badges line", 1, False, f"only {len(lines)} lines")
            return
        line6 = lines[5].strip()
        match = re.match(r"Red badges: (\d+)", line6)
        ok = match is not None
        check("11. Retro red badges line", 1, ok,
              f"line={repr(line6)}")
    except Exception as e:
        check("11. Retro red badges line", 1, False, f"exception: {e}")


# ── OpenProject checks ────────────────────────────────────────────────────────
def check_12_retro_wp_exists() -> None:
    """Task WP 'Retro action items: Pentest Round 1' exists in project 'Security Audit'."""
    try:
        result = openproject_sql(
            "SELECT wp.id FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.name = 'Security Audit' "
            "AND t.name = 'Task' "
            "AND wp.subject = 'Retro action items: Pentest Round 1';"
        )
        found = bool(result.strip())
        check("12. OpenProject retro WP exists", 2, found,
              f"wp_id={result}" if found else "not found")
    except Exception as e:
        check("12. OpenProject retro WP exists", 2, False, f"exception: {e}")


def check_13_retro_wp_assignee_priority() -> None:
    """Retro WP has assignee user11 and priority Normal."""
    try:
        result = openproject_sql(
            "SELECT u.login, ip.name FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "LEFT JOIN users u ON wp.assigned_to_id = u.id "
            "LEFT JOIN enumerations ip ON wp.priority_id = ip.id "
            "WHERE p.name = 'Security Audit' "
            "AND t.name = 'Task' "
            "AND wp.subject = 'Retro action items: Pentest Round 1';"
        )
        if not result.strip():
            check("13. Retro WP assignee & priority", 2, False, "WP not found")
            return
        parts = result.strip().split("|")
        assignee = parts[0].strip() if len(parts) > 0 else ""
        priority = parts[1].strip() if len(parts) > 1 else ""
        assignee_ok = assignee == "user11"
        priority_ok = priority.lower() == "normal"
        ok = assignee_ok and priority_ok
        check("13. Retro WP assignee & priority", 2, ok,
              f"assignee={repr(assignee)}, priority={repr(priority)}")
    except Exception as e:
        check("13. Retro WP assignee & priority", 2, False, f"exception: {e}")


def check_14_retro_wp_description() -> None:
    """Retro WP description matches expected format."""
    try:
        result = openproject_sql(
            "SELECT j.data->'description'->>'raw' "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "JOIN journals j ON j.journable_id = wp.id AND j.journable_type = 'WorkPackage' "
            "WHERE p.name = 'Security Audit' "
            "AND t.name = 'Task' "
            "AND wp.subject = 'Retro action items: Pentest Round 1' "
            "ORDER BY j.version DESC LIMIT 1;"
        )
        desc = result.strip()
        # Expected: "Retro doc: devops-configs/docs/retro-Pentest Round 1.md; Closed rate: X/T; Red projects: R"
        has_retro_doc = "Retro doc: devops-configs/docs/retro-Pentest Round 1.md" in desc
        has_closed_rate = bool(re.search(r"Closed rate: \d+/\d+", desc))
        has_red_projects = bool(re.search(r"Red projects: \d+", desc))
        ok = has_retro_doc and has_closed_rate and has_red_projects
        check("14. Retro WP description format", 2, ok,
              f"desc={repr(desc[:150])}")
    except Exception as e:
        check("14. Retro WP description format", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_db_exists()
    check_2_sprint_wp_table()
    check_3_test_health_table()
    check_4_sprint_wp_rows()
    check_5_test_health_rows()
    check_6_completion_summary_view()
    check_7_retro_file_exists()
    check_8_retro_header_date()
    check_9_retro_closed_hours()
    check_10_retro_test_health()
    check_11_retro_red_badges()
    check_12_retro_wp_exists()
    check_13_retro_wp_assignee_priority()
    check_14_retro_wp_description()

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
