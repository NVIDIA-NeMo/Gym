# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_032/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-032-I1: Kick off Sprint 2025-04 for E-Commerce Platform

Checks: 16 weighted checks across openproject, code-server, baserow.
Strategy: docker exec (OpenProject embedded DB, code-server filesystem), REST API (Baserow)

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

OP_PORT = os.environ.get("OPENPROJECT_PORT")
OP_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")
CS_PORT = os.environ.get("CODE_SERVER_PORT")
CS_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BR_PORT = os.environ.get("BASEROW_PORT")
BR_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BR_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")

for _var_name, _var_val in [
    ("OPENPROJECT_PORT", OP_PORT), ("OPENPROJECT_CONTAINER", OP_CONTAINER),
    ("CODE_SERVER_PORT", CS_PORT), ("CODE_SERVER_CONTAINER", CS_CONTAINER),
    ("BASEROW_PORT", BR_PORT), ("BASEROW_CONTAINER", BR_CONTAINER),
    ("BASEROW_DB_CONTAINER", BR_DB_CONTAINER),
]:
    if not _var_val:
        print(f"FATAL: {_var_name} not set", file=sys.stderr)
        sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
BACKLOG = [
    {"subject": "Fix race condition in cart merge on login", "type": "Bug", "priority": "High", "estimated_hours": 6.0, "assignee": "Judith Triplett", "target_module": "blog-engine/src/routes/api.js"},
    {"subject": "Add structured logging to checkout routes", "type": "Task", "priority": "Normal", "estimated_hours": 4.5, "assignee": "John Marshall", "target_module": "todo-api/tests/test_categories.py"},
    {"subject": "Slug generation helper supports unicode", "type": "Feature", "priority": "Normal", "estimated_hours": 8.0, "assignee": "Lena Hogan", "target_module": "blog-engine/src/utils/slugify.js"},
    {"subject": "Export analyzer summary as JSON", "type": "Feature", "priority": "Low", "estimated_hours": 5.0, "assignee": "Jane Dradder", "target_module": "data-analyzer/src/analyzer.py"},
    {"subject": "Fix ItemList pagination double-fetch", "type": "Bug", "priority": "Immediate", "estimated_hours": 3.0, "assignee": "Latisha Mazon", "target_module": "vue-hackernews-2.0/src/views/ItemList.vue"},
]

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


def op_query(sql: str) -> str:
    """Run a psql query inside the OpenProject container (embedded Postgres)."""
    rc, out, err = docker_exec(
        OP_CONTAINER, "psql", "-U", "openproject", "-d", "openproject",
        "-t", "-A", "-c", sql
    )
    return out.strip()


def op_query_rows(sql: str) -> list[list[str]]:
    raw = op_query(sql)
    if not raw:
        return []
    return [line.split("|") for line in raw.split("\n") if line.strip()]


# ── OpenProject checks ───────────────────────────────────────────────────────
def check_1_version():
    """Version Sprint 2025-04 exists with correct dates, status, description."""
    try:
        rows = op_query_rows(
            "SELECT v.name, v.start_date, v.effective_date, v.status, v.description "
            "FROM versions v JOIN projects p ON v.project_id = p.id "
            "WHERE p.name = 'E-Commerce Platform' AND v.name = 'Sprint 2025-04'"
        )
        if not rows:
            check("1. Version Sprint 2025-04", 2, False, "version not found")
            return
        row = rows[0]
        name, start, due, status, desc = row[0], row[1], row[2], row[3], row[4] if len(row) > 4 else ""
        issues = []
        if start != "2025-04-07":
            issues.append(f"start_date={start}")
        if due != "2025-04-18":
            issues.append(f"due_date={due}")
        if status != "open":
            issues.append(f"status={status}")
        expected_desc = "Sprint goal: Stabilize checkout and improve cart conversion"
        if desc.strip() != expected_desc:
            issues.append(f"description mismatch: '{desc.strip()[:60]}...'")
        check("1. Version Sprint 2025-04", 2, not issues,
              "; ".join(issues) if issues else "all fields correct")
    except Exception as e:
        check("1. Version Sprint 2025-04", 2, False, f"exception: {e}")


def check_2_board():
    """Board 'Sprint 2025-04 Board' exists in the project."""
    try:
        # OpenProject stores boards in the grids table with type containing 'Board'
        result = op_query(
            "SELECT name FROM grids "
            "WHERE project_id = (SELECT id FROM projects WHERE name = 'E-Commerce Platform') "
            "AND name = 'Sprint 2025-04 Board'"
        )
        if result:
            check("2. Board Sprint 2025-04 Board", 1, True, "found in grids")
            return
        # Fallback: try boards table (some OP versions)
        result = op_query(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='boards')"
        )
        if result == "t":
            result = op_query(
                "SELECT name FROM boards "
                "WHERE project_id = (SELECT id FROM projects WHERE name = 'E-Commerce Platform') "
                "AND name = 'Sprint 2025-04 Board'"
            )
            if result:
                check("2. Board Sprint 2025-04 Board", 1, True, "found in boards")
                return
        check("2. Board Sprint 2025-04 Board", 1, False, "board not found")
    except Exception as e:
        check("2. Board Sprint 2025-04 Board", 1, False, f"exception: {e}")


def check_3_work_packages_exist():
    """5 work packages with correct subjects assigned to Sprint 2025-04."""
    try:
        rows = op_query_rows(
            "SELECT wp.subject "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.name = 'E-Commerce Platform' "
            "AND wp.version_id = ("
            "  SELECT id FROM versions WHERE name = 'Sprint 2025-04' "
            "  AND project_id = (SELECT id FROM projects WHERE name = 'E-Commerce Platform')"
            ") ORDER BY wp.subject"
        )
        found_subjects = {r[0] for r in rows}
        expected_subjects = {item["subject"] for item in BACKLOG}
        missing = expected_subjects - found_subjects
        passed = len(rows) == 5 and not missing
        detail = f"found {len(rows)} WPs"
        if missing:
            detail += f"; missing: {list(missing)[:3]}"
        check("3. 5 work packages with correct subjects", 2, passed, detail)
    except Exception as e:
        check("3. 5 work packages with correct subjects", 2, False, f"exception: {e}")


def check_4_wp_types_priorities():
    """Work packages have correct types and priorities."""
    try:
        rows = op_query_rows(
            "SELECT wp.subject, t.name, e.name "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "LEFT JOIN enumerations e ON wp.priority_id = e.id "
            "WHERE p.name = 'E-Commerce Platform' "
            "AND wp.version_id = ("
            "  SELECT id FROM versions WHERE name = 'Sprint 2025-04' "
            "  AND project_id = (SELECT id FROM projects WHERE name = 'E-Commerce Platform')"
            ")"
        )
        expected = {item["subject"]: (item["type"], item["priority"]) for item in BACKLOG}
        issues = []
        matched = 0
        for row in rows:
            subj, typ, pri = row[0], row[1], row[2] if len(row) > 2 else ""
            if subj in expected:
                exp_type, exp_pri = expected[subj]
                if typ != exp_type:
                    issues.append(f"'{subj[:30]}': type={typ}")
                elif pri != exp_pri:
                    issues.append(f"'{subj[:30]}': priority={pri}")
                else:
                    matched += 1
        passed = matched == 5 and not issues
        check("4. WP types and priorities", 2, passed,
              f"{matched}/5 correct" + (f"; {'; '.join(issues[:3])}" if issues else ""))
    except Exception as e:
        check("4. WP types and priorities", 2, False, f"exception: {e}")


def check_5_wp_hours_assignees():
    """Work packages have correct estimated hours and assignees."""
    try:
        rows = op_query_rows(
            "SELECT wp.subject, wp.estimated_hours, "
            "COALESCE(u.firstname || ' ' || u.lastname, '') "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "LEFT JOIN users u ON wp.assigned_to_id = u.id "
            "WHERE p.name = 'E-Commerce Platform' "
            "AND wp.version_id = ("
            "  SELECT id FROM versions WHERE name = 'Sprint 2025-04' "
            "  AND project_id = (SELECT id FROM projects WHERE name = 'E-Commerce Platform')"
            ")"
        )
        expected = {item["subject"]: (item["estimated_hours"], item["assignee"]) for item in BACKLOG}
        issues = []
        matched = 0
        for row in rows:
            subj = row[0]
            hours_str = row[1] if len(row) > 1 else ""
            assignee = row[2].strip() if len(row) > 2 else ""
            if subj in expected:
                exp_hours, exp_assignee = expected[subj]
                try:
                    hours = float(hours_str) if hours_str else 0.0
                except ValueError:
                    hours = 0.0
                if abs(hours - exp_hours) > 0.01:
                    issues.append(f"'{subj[:30]}': hours={hours}, expected {exp_hours}")
                elif assignee != exp_assignee:
                    issues.append(f"'{subj[:30]}': assignee='{assignee}', expected '{exp_assignee}'")
                else:
                    matched += 1
        passed = matched == 5 and not issues
        check("5. WP estimated hours and assignees", 2, passed,
              f"{matched}/5 correct" + (f"; {'; '.join(issues[:3])}" if issues else ""))
    except Exception as e:
        check("5. WP estimated hours and assignees", 2, False, f"exception: {e}")


def check_6_meeting():
    """Meeting 'Sprint Planning: Sprint 2025-04' exists with correct date/time."""
    try:
        rows = op_query_rows(
            "SELECT m.title, m.start_time "
            "FROM meetings m JOIN projects p ON m.project_id = p.id "
            "WHERE p.name = 'E-Commerce Platform' "
            "AND m.title = 'Sprint Planning: Sprint 2025-04'"
        )
        if not rows:
            check("6. Meeting Sprint Planning", 2, False, "meeting not found")
            return
        title, start_time = rows[0][0], rows[0][1]
        passed = "2025-04-07" in start_time and "10:00" in start_time
        check("6. Meeting Sprint Planning", 2, passed, f"start_time={start_time}")
    except Exception as e:
        check("6. Meeting Sprint Planning", 2, False, f"exception: {e}")


def check_7_agenda_items():
    """Meeting has 5 agenda items with correct titles in order."""
    try:
        rows = op_query_rows(
            "SELECT mai.title "
            "FROM meeting_agenda_items mai "
            "JOIN meetings m ON mai.meeting_id = m.id "
            "JOIN projects p ON m.project_id = p.id "
            "WHERE p.name = 'E-Commerce Platform' "
            "AND m.title = 'Sprint Planning: Sprint 2025-04' "
            "ORDER BY mai.position"
        )
        expected_titles = [f"Review: {item['subject']}" for item in BACKLOG]
        found_titles = [r[0] for r in rows]

        if len(found_titles) != 5:
            check("7. Meeting agenda items", 2, False,
                  f"found {len(found_titles)} items, expected 5")
            return

        issues = []
        for i, (found, expected) in enumerate(zip(found_titles, expected_titles)):
            if found != expected:
                issues.append(f"item {i+1}: '{found[:40]}' != '{expected[:40]}'")
        passed = not issues
        check("7. Meeting agenda items", 2, passed,
              "all 5 in correct order" if passed else "; ".join(issues[:3]))
    except Exception as e:
        check("7. Meeting agenda items", 2, False, f"exception: {e}")


# ── code-server checks ───────────────────────────────────────────────────────
def _find_file_in_container(target_module: str) -> str:
    """Locate a file by its relative path inside the code-server container."""
    rc, out, err = docker_exec(
        CS_CONTAINER, "find", "/home", "-path", f"*/{target_module}", "-type", "f",
        timeout=10,
    )
    if rc == 0 and out.strip():
        return out.strip().split("\n")[0]
    return ""


def _check_todo_comment(check_num: int, item: dict) -> None:
    """Verify that the TODO comment line exists in the target file."""
    subject = item["subject"]
    target = item["target_module"]
    ext = target.rsplit(".", 1)[-1] if "." in target else ""

    if ext in ("js", "ts", "tsx", "vue"):
        prefix = "//"
    else:
        prefix = "#"

    expected_comment = f"{prefix} TODO [Sprint 2025-04]: {subject}"

    try:
        filepath = _find_file_in_container(target)
        if not filepath:
            check(f"{check_num}. TODO in {target}", 1, False, "file not found in container")
            return

        rc, out, err = docker_exec(CS_CONTAINER, "grep", "-F", expected_comment, filepath)
        passed = rc == 0 and expected_comment in out
        check(f"{check_num}. TODO in {target}", 1, passed,
              "found" if passed else "comment not found")
    except Exception as e:
        check(f"{check_num}. TODO in {target}", 1, False, f"exception: {e}")


def check_8_todo_api_js():
    _check_todo_comment(8, BACKLOG[0])


def check_9_todo_test_categories():
    _check_todo_comment(9, BACKLOG[1])


def check_10_todo_slugify():
    _check_todo_comment(10, BACKLOG[2])


def check_11_todo_analyzer():
    _check_todo_comment(11, BACKLOG[3])


def check_12_todo_itemlist():
    _check_todo_comment(12, BACKLOG[4])


# ── Baserow checks (REST API) ────────────────────────────────────────────────
def _baserow_auth() -> str:
    """Authenticate to Baserow and return the JWT access token."""
    r = requests.post(
        f"http://{HOST}:{BR_PORT}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _baserow_headers(token: str) -> dict:
    return {"Authorization": f"JWT {token}"}


def _find_baserow_table(token: str):
    """Find the Sprint Backlog table. Returns (db_id, table_id) or (None, None)."""
    headers = _baserow_headers(token)
    r = requests.get(f"http://{HOST}:{BR_PORT}/api/applications/",
                     headers=headers, timeout=10)
    r.raise_for_status()
    apps = r.json()

    db_id = None
    for app in apps:
        if app.get("name") == "Sprint 2025-04 Tracking" and app.get("type") == "database":
            db_id = app["id"]
            break
    if not db_id:
        return None, None

    r = requests.get(f"http://{HOST}:{BR_PORT}/api/database/tables/database/{db_id}/",
                     headers=headers, timeout=10)
    r.raise_for_status()
    for t in r.json():
        if t.get("name") == "Sprint Backlog":
            return db_id, t["id"]
    return db_id, None


def check_13_baserow_db_table():
    """Baserow database 'Sprint 2025-04 Tracking' and table 'Sprint Backlog' exist."""
    try:
        token = _baserow_auth()
        db_id, table_id = _find_baserow_table(token)
        if not db_id:
            check("13. Baserow DB and Sprint Backlog table", 1, False, "database not found")
        elif not table_id:
            check("13. Baserow DB and Sprint Backlog table", 1, False,
                  f"db found (id={db_id}) but table 'Sprint Backlog' not found")
        else:
            check("13. Baserow DB and Sprint Backlog table", 1, True,
                  f"db_id={db_id}, table_id={table_id}")
    except Exception as e:
        check("13. Baserow DB and Sprint Backlog table", 1, False, f"exception: {e}")


def check_14_baserow_rows_subjects():
    """5 rows with correct subjects in backlog order."""
    try:
        token = _baserow_auth()
        headers = _baserow_headers(token)
        db_id, table_id = _find_baserow_table(token)
        if not table_id:
            check("14. Baserow 5 rows with correct subjects", 2, False, "table not found")
            return

        # Get fields to find Subject field
        r = requests.get(f"http://{HOST}:{BR_PORT}/api/database/fields/table/{table_id}/",
                         headers=headers, timeout=10)
        r.raise_for_status()
        fields = r.json()
        subject_field = None
        for f in fields:
            if f.get("name") == "Subject":
                subject_field = f"field_{f['id']}"
                break

        # Get rows (ordered by id to preserve insertion order)
        r = requests.get(
            f"http://{HOST}:{BR_PORT}/api/database/rows/table/{table_id}/?size=100",
            headers=headers, timeout=10,
        )
        r.raise_for_status()
        rows = r.json().get("results", [])

        if len(rows) != 5:
            check("14. Baserow 5 rows with correct subjects", 2, False,
                  f"found {len(rows)} rows, expected 5")
            return

        expected_subjects = [item["subject"] for item in BACKLOG]
        found_subjects = []
        for row in rows:
            if subject_field:
                found_subjects.append(row.get(subject_field, ""))
            else:
                found_subjects.append("(Subject field not found)")

        issues = []
        for i, (found, expected) in enumerate(zip(found_subjects, expected_subjects)):
            if found != expected:
                issues.append(f"row {i+1}: '{found[:40]}' != '{expected[:40]}'")
        passed = not issues and subject_field is not None
        check("14. Baserow 5 rows with correct subjects", 2, passed,
              "all subjects match in order" if passed else "; ".join(issues[:3]))
    except Exception as e:
        check("14. Baserow 5 rows with correct subjects", 2, False, f"exception: {e}")


def check_15_baserow_row_fields():
    """Rows have correct Type, Priority, Estimated Hours, Assignee, Target Module."""
    try:
        token = _baserow_auth()
        headers = _baserow_headers(token)
        db_id, table_id = _find_baserow_table(token)
        if not table_id:
            check("15. Baserow row field values", 2, False, "table not found")
            return

        # Get field mapping
        r = requests.get(f"http://{HOST}:{BR_PORT}/api/database/fields/table/{table_id}/",
                         headers=headers, timeout=10)
        r.raise_for_status()
        field_map = {}
        for f in r.json():
            field_map[f["name"]] = f"field_{f['id']}"

        # Get rows
        r = requests.get(
            f"http://{HOST}:{BR_PORT}/api/database/rows/table/{table_id}/?size=100",
            headers=headers, timeout=10,
        )
        r.raise_for_status()
        rows = r.json().get("results", [])

        issues = []
        for i, item in enumerate(BACKLOG):
            if i >= len(rows):
                issues.append(f"row {i+1} missing")
                continue
            row = rows[i]

            # Type (single-select → dict with "value")
            type_key = field_map.get("Type")
            if type_key:
                type_val = row.get(type_key)
                if isinstance(type_val, dict):
                    type_val = type_val.get("value", "")
                if str(type_val) != item["type"]:
                    issues.append(f"row {i+1} Type: '{type_val}' != '{item['type']}'")

            # Priority (single-select)
            pri_key = field_map.get("Priority")
            if pri_key:
                pri_val = row.get(pri_key)
                if isinstance(pri_val, dict):
                    pri_val = pri_val.get("value", "")
                if str(pri_val) != item["priority"]:
                    issues.append(f"row {i+1} Priority: '{pri_val}' != '{item['priority']}'")

            # Estimated Hours (number)
            hours_key = field_map.get("Estimated Hours")
            if hours_key:
                hours_val = row.get(hours_key)
                try:
                    if abs(float(hours_val or 0) - item["estimated_hours"]) > 0.01:
                        issues.append(f"row {i+1} Hours: {hours_val} != {item['estimated_hours']}")
                except (ValueError, TypeError):
                    issues.append(f"row {i+1} Hours: '{hours_val}' invalid")

            # Assignee
            assignee_key = field_map.get("Assignee")
            if assignee_key:
                if row.get(assignee_key, "") != item["assignee"]:
                    issues.append(f"row {i+1} Assignee: '{row.get(assignee_key)}' != '{item['assignee']}'")

            # Target Module
            tm_key = field_map.get("Target Module")
            if tm_key:
                if row.get(tm_key, "") != item["target_module"]:
                    issues.append(f"row {i+1} Target: '{row.get(tm_key)}' != '{item['target_module']}'")

        passed = not issues
        check("15. Baserow row field values", 2, passed,
              "all fields correct" if passed else "; ".join(issues[:5]))
    except Exception as e:
        check("15. Baserow row field values", 2, False, f"exception: {e}")


def check_16_baserow_kanban_view():
    """Kanban view 'By Priority' exists on Sprint Backlog table."""
    try:
        token = _baserow_auth()
        headers = _baserow_headers(token)
        db_id, table_id = _find_baserow_table(token)
        if not table_id:
            check("16. Baserow Kanban view By Priority", 1, False, "table not found")
            return

        r = requests.get(f"http://{HOST}:{BR_PORT}/api/database/views/table/{table_id}/",
                         headers=headers, timeout=10)
        r.raise_for_status()
        views = r.json()

        kanban = None
        for v in views:
            if v.get("name") == "By Priority" and v.get("type") == "kanban":
                kanban = v
                break

        passed = kanban is not None
        check("16. Baserow Kanban view By Priority", 1, passed,
              "found" if passed else "kanban view 'By Priority' not found")
    except Exception as e:
        check("16. Baserow Kanban view By Priority", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_version()
    check_2_board()
    check_3_work_packages_exist()
    check_4_wp_types_priorities()
    check_5_wp_hours_assignees()
    check_6_meeting()
    check_7_agenda_items()
    check_8_todo_api_js()
    check_9_todo_test_categories()
    check_10_todo_slugify()
    check_11_todo_analyzer()
    check_12_todo_itemlist()
    check_13_baserow_db_table()
    check_14_baserow_rows_subjects()
    check_15_baserow_row_fields()
    check_16_baserow_kanban_view()

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
