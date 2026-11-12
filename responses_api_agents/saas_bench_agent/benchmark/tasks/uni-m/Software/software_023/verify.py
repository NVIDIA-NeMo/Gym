#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_023/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-023-I4: TypeScript upgrade campaign across 4 projects with
Baserow inventory and OpenProject Epic.

Checks: 9 weighted checks across code-server, baserow, openproject.
Strategy: Baserow API, code-server docker exec (filesystem), OpenProject docker exec (DB).

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import json
import os
import re
import subprocess
import sys

import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

_required_vars = {
    "CODE_SERVER_PORT": None, "CODE_SERVER_CONTAINER": None,
    "BASEROW_PORT": None, "BASEROW_CONTAINER": None, "BASEROW_DB_CONTAINER": None,
    "OPENPROJECT_PORT": None, "OPENPROJECT_CONTAINER": None,
}
for var in _required_vars:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    _required_vars[var] = val

CODE_SERVER_CONTAINER = _required_vars["CODE_SERVER_CONTAINER"]
BASEROW_PORT = _required_vars["BASEROW_PORT"]
BASEROW_DB_CONTAINER = _required_vars["BASEROW_DB_CONTAINER"]
OPENPROJECT_CONTAINER = _required_vars["OPENPROJECT_CONTAINER"]

BASEROW_BASE = f"http://{HOST}:{BASEROW_PORT}"
BASEROW_EMAIL = "admin@example.com"
BASEROW_PASS = "Admin1234"

OP_DB = "openproject"
OP_USER = "openproject"
OP_PASS = "openproject"

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


def op_sql(query: str) -> str:
    """Run a SQL query against OpenProject's embedded Postgres."""
    r = subprocess.run(
        ["docker", "exec", "-i", OPENPROJECT_CONTAINER,
         "bash", "-c",
         f"PGPASSWORD={OP_PASS} psql -h 127.0.0.1 -U {OP_USER} -d {OP_DB} -t -A"],
        input=query,
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"psql error: {r.stderr.strip()}")
    return r.stdout.strip()


def baserow_auth() -> str:
    """Get Baserow JWT token."""
    r = requests.post(
        f"{BASEROW_BASE}/api/user/token-auth/",
        json={"email": BASEROW_EMAIL, "password": BASEROW_PASS},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def baserow_get(path: str, token: str) -> dict | list:
    r = requests.get(
        f"{BASEROW_BASE}/api/{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# ── Shared state across checks ───────────────────────────────────────────────
_br_token: str = ""
_br_table_id: int | None = None
_br_field_map: dict[str, int] = {}  # field name -> field id
_br_rows: list[dict] = []


def _init_baserow() -> bool:
    """Authenticate and locate the database/table. Returns True on success."""
    global _br_token, _br_table_id, _br_field_map, _br_rows
    try:
        _br_token = baserow_auth()
    except Exception as e:
        return False

    # Find database
    apps = baserow_get("applications/", _br_token)
    db_id = None
    for app in apps:
        if app.get("name") == "TypeScript Upgrade Campaign July 2026" and app.get("type") == "database":
            db_id = app["id"]
            break
    if db_id is None:
        return False

    # Find table
    tables = baserow_get(f"database/tables/database/{db_id}/", _br_token)
    for t in tables:
        if t.get("name") == "Upgrade Inventory":
            _br_table_id = t["id"]
            break
    if _br_table_id is None:
        return False

    # Load fields
    fields = baserow_get(f"database/fields/table/{_br_table_id}/", _br_token)
    _br_field_map = {f["name"]: f["id"] for f in fields}

    # Load rows
    resp = baserow_get(f"database/rows/table/{_br_table_id}/?user_field_names=true", _br_token)
    _br_rows = resp.get("results", []) if isinstance(resp, dict) else resp

    return True


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_database_exists() -> None:
    """Baserow database 'TypeScript Upgrade Campaign July 2026' exists."""
    try:
        if not _br_token:
            raise RuntimeError("Baserow auth failed")
        apps = baserow_get("applications/", _br_token)
        found = any(
            a.get("name") == "TypeScript Upgrade Campaign July 2026" and a.get("type") == "database"
            for a in apps
        )
        check("1. Baserow DB exists", 1, found,
              "TypeScript Upgrade Campaign July 2026" if found else "database not found")
    except Exception as e:
        check("1. Baserow DB exists", 1, False, f"exception: {e}")


def check_2_table_and_fields() -> None:
    """Table 'Upgrade Inventory' with correct field schema."""
    try:
        if _br_table_id is None:
            check("2. Table & fields", 2, False, "table not found")
            return
        expected_fields = {
            "Project", "Manifest Path", "Current Version", "Target Version",
            "Migration Complexity", "Status", "Captured At",
        }
        actual = set(_br_field_map.keys())
        missing = expected_fields - actual
        check("2. Table & fields", 2, not missing,
              f"all fields present" if not missing else f"missing fields: {missing}")
    except Exception as e:
        check("2. Table & fields", 2, False, f"exception: {e}")


def check_3_row_projects() -> None:
    """4 rows with correct Project names in alphabetical order."""
    try:
        projects = [r.get("Project", "") for r in _br_rows]
        expected = ["blog-engine", "tabler", "todo-api", "weather-dashboard"]
        ok = projects == expected
        check("3. Row projects (alpha order)", 2, ok,
              f"expected {expected}, got {projects}")
    except Exception as e:
        check("3. Row projects (alpha order)", 2, False, f"exception: {e}")


def _get_select_value(row: dict, field_name: str) -> str:
    """Extract the display value from a single-select field."""
    val = row.get(field_name, "")
    if isinstance(val, dict):
        return val.get("value", "")
    return str(val) if val else ""


def check_4_row_field_values() -> None:
    """Target Version, Migration Complexity, Captured At are correct per row."""
    try:
        if not _br_rows:
            check("4. Row field values", 2, False, "no rows found")
            return
        complexity_map = {
            "tabler": "High", "weather-dashboard": "Medium",
            "todo-api": "Low", "blog-engine": "Low",
        }
        issues = []
        for row in _br_rows:
            proj = row.get("Project", "")
            tv = row.get("Target Version", "")
            if tv != "5.4.5":
                issues.append(f"{proj}: Target Version={tv!r}, expected '5.4.5'")
            mc = _get_select_value(row, "Migration Complexity")
            expected_mc = complexity_map.get(proj, "?")
            if mc != expected_mc:
                issues.append(f"{proj}: Migration Complexity={mc!r}, expected {expected_mc!r}")
            ca = row.get("Captured At", "")
            if isinstance(ca, str) and not ca.startswith("2026-07-08"):
                issues.append(f"{proj}: Captured At={ca!r}, expected 2026-07-08")
        check("4. Row field values", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("4. Row field values", 2, False, f"exception: {e}")


def check_5_status_values() -> None:
    """tabler, todo-api, weather-dashboard Status=Pending."""
    try:
        if not _br_rows:
            check("5. Status values", 1, False, "no rows found")
            return
        issues = []
        for row in _br_rows:
            proj = row.get("Project", "")
            status = _get_select_value(row, "Status")
            if proj == "blog-engine":
                continue
            else:
                if status != "Pending":
                    issues.append(f"{proj}: Status={status!r}, expected 'Pending'")
        check("5. Status values", 1, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("5. Status values", 1, False, f"exception: {e}")


def check_6_high_complexity_view() -> None:
    """'High Complexity' view exists with Migration Complexity=High filter."""
    try:
        if _br_table_id is None:
            check("6. High Complexity view", 2, False, "table not found")
            return
        views = baserow_get(f"database/views/table/{_br_table_id}/", _br_token)
        view_id = None
        for v in views:
            if v.get("name") == "High Complexity":
                view_id = v["id"]
                break
        if view_id is None:
            check("6. High Complexity view", 2, False, "view 'High Complexity' not found")
            return

        filters = baserow_get(f"database/views/{view_id}/filters/", _br_token)
        # filters could be a list or dict with results
        filter_list = filters if isinstance(filters, list) else filters.get("results", filters)
        mc_field_id = _br_field_map.get("Migration Complexity")
        has_filter = False
        for f in filter_list:
            if f.get("field") == mc_field_id and "High" in str(f.get("value", "")):
                has_filter = True
                break
        check("6. High Complexity view", 2, has_filter,
              "view + filter OK" if has_filter else f"filter not found (filters={filter_list})")
    except Exception as e:
        check("6. High Complexity view", 2, False, f"exception: {e}")


def check_8_openproject_epic() -> None:
    """Epic 'Upgrade typescript to 5.4.5' in 'Mobile App Redesign' with correct description."""
    try:
        row = op_sql(
            "SELECT wp.subject, wp.description "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.name = 'Mobile App Redesign' "
            "AND t.name = 'Epic' "
            "AND wp.subject = 'Upgrade typescript to 5.4.5'"
        )
        if not row:
            check("8. OpenProject Epic", 2, False, "Epic not found")
            return

        parts = row.split("|")
        subject = parts[0].strip() if parts else ""
        description = parts[1].strip() if len(parts) > 1 else ""

        # Description should contain "Campaign Date: 2026-07-08; Target: 5.4.5; Projects: 4"
        desc_ok = (
            "Campaign Date: 2026-07-08" in description
            and "Target: 5.4.5" in description
            and "Projects: 4" in description
        )
        check("8. OpenProject Epic", 2, desc_ok,
              f"description OK" if desc_ok else f"description={description!r}")
    except Exception as e:
        check("8. OpenProject Epic", 2, False, f"exception: {e}")


def check_9_openproject_tasks() -> None:
    """3 Task children under Epic with correct subjects, assignee=OpenProject Admin, priority."""
    try:
        # Get Epic ID
        epic_id = op_sql(
            "SELECT wp.id FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.name = 'Mobile App Redesign' "
            "AND t.name = 'Epic' "
            "AND wp.subject = 'Upgrade typescript to 5.4.5'"
        )
        if not epic_id:
            check("9. OpenProject Tasks", 3, False, "parent Epic not found")
            return

        # Get child tasks
        rows = op_sql(
            f"SELECT wp.subject, u.login, u.firstname, u.lastname, "
            f"e.name AS priority_name "
            f"FROM work_packages wp "
            f"JOIN types t ON wp.type_id = t.id "
            f"LEFT JOIN users u ON wp.assigned_to_id = u.id "
            f"LEFT JOIN enumerations e ON wp.priority_id = e.id "
            f"WHERE wp.parent_id = {epic_id.strip()} "
            f"AND t.name = 'Task' "
            f"ORDER BY wp.subject"
        )
        if not rows:
            check("9. OpenProject Tasks", 3, False, "no child Tasks found")
            return

        tasks = []
        for line in rows.strip().splitlines():
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 5:
                tasks.append({
                    "subject": cols[0],
                    "login": cols[1],
                    "firstname": cols[2],
                    "lastname": cols[3],
                    "priority": cols[4],
                })

        # We expect 3 tasks (blog-engine is Done, so only tabler, todo-api, weather-dashboard)
        # Expected subjects pattern: [<Project>] Bump typescript <Current Version> → 5.4.5
        # We don't know Current Version at verify time, so check pattern
        issues = []
        if len(tasks) != 3:
            issues.append(f"expected 3 tasks, found {len(tasks)}")

        expected_projects = {"tabler", "todo-api", "weather-dashboard"}
        found_projects = set()
        for task in tasks:
            subj = task["subject"]
            # Extract project name from [<Project>]
            m = re.match(r"\[([^\]]+)\]", subj)
            if m:
                proj = m.group(1)
                found_projects.add(proj)

                # Check assignee is admin
                assignee = task.get("login", "")
                if assignee != "admin":
                    assignee_name = f"{task.get('firstname', '')} {task.get('lastname', '')}".strip()
                    if "admin" not in assignee_name.lower() and "openproject" not in assignee_name.lower():
                        issues.append(f"[{proj}] assignee={assignee!r}, expected admin")

                # Check priority: High for tabler (Migration Complexity=High), Normal for others
                pri = task.get("priority", "")
                if proj == "tabler":
                    if pri != "High":
                        issues.append(f"[tabler] priority={pri!r}, expected 'High'")
                else:
                    if pri != "Normal":
                        issues.append(f"[{proj}] priority={pri!r}, expected 'Normal'")

                # Check subject contains "5.4.5" and "Bump typescript"
                if "5.4.5" not in subj or "Bump typescript" not in subj:
                    issues.append(f"[{proj}] subject format wrong: {subj!r}")
            else:
                issues.append(f"subject does not match pattern: {subj!r}")

        missing_projects = expected_projects - found_projects
        if missing_projects:
            issues.append(f"missing projects: {missing_projects}")

        check("9. OpenProject Tasks", 3, not issues,
              "all 3 tasks correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("9. OpenProject Tasks", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Initialize Baserow connection
    br_ok = _init_baserow()
    if not br_ok:
        print("WARNING: Baserow init failed; Baserow checks will fail", file=sys.stderr)

    check_1_database_exists()
    check_2_table_and_fields()
    check_3_row_projects()
    check_4_row_field_values()
    check_5_status_values()
    check_6_high_complexity_view()
    check_8_openproject_epic()
    check_9_openproject_tasks()

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
