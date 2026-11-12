#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_004/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-004-I1: Plan synchronized Backend/Data sprint across
OpenProject, code-server, and Baserow.

Checks: 10 weighted checks (20 pts) across openproject, code-server, baserow.
Strategy: docker exec (DB queries + filesystem) for all checks.

Required env vars:
  SERVER_HOSTNAME,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER,
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

OPENPROJECT_PORT = os.environ.get("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")
CODE_SERVER_PORT = os.environ.get("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")

_missing = []
for var in [
    "OPENPROJECT_PORT", "OPENPROJECT_CONTAINER",
    "CODE_SERVER_PORT", "CODE_SERVER_CONTAINER",
    "BASEROW_PORT", "BASEROW_CONTAINER", "BASEROW_DB_CONTAINER",
]:
    if not os.environ.get(var):
        _missing.append(var)
if _missing:
    print(f"FATAL: missing env vars: {', '.join(_missing)}", file=sys.stderr)
    sys.exit(1)

# ── Slot values (from instance) ──────────────────────────────────────────────
SPRINT_NAME = "Sprint Synchronize 2025-W10"
TEAM_A = "Backend"
TEAM_B = "Data"
SPRINT_START = "2025-03-03"
SPRINT_END = "2025-03-17"
NUM_WP_PER_TEAM = 4
TEAM_A_HOURS = 32
TEAM_B_HOURS = 28
OP_PROJECT = "Data Analytics Pipeline"
BASEROW_DB_NAME = "Sprint Capacity Planner"
BASEROW_TABLE_NAME = "Sprint Capacity"
COMMENT_LINE = "# Sprint Sprint Synchronize 2025-W10: integration touchpoint"

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
    """Run a SQL query against the embedded OpenProject Postgres DB."""
    r = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=openproject",
         OPENPROJECT_CONTAINER,
         "psql", "-U", "openproject", "-h", "127.0.0.1", "-d", "openproject",
         "-t", "-A", "-c", query],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"op_sql failed (rc={r.returncode}): {r.stderr.strip()}")
    return r.stdout.strip()


def baserow_sql(query: str) -> str:
    """Run a SQL query against the Baserow Postgres DB."""
    rc, out, err = docker_exec(
        BASEROW_DB_CONTAINER,
        "env", "PGPASSWORD=kdpzkuyhsgb22onku8y7rxkx3czej88nxpngaz4mlmgad67vpc", "psql", "-h", "127.0.0.1", "-U", "baserow", "-d", "baserow",
        "-t", "-A", "-c", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"baserow_sql failed (rc={rc}): {err.strip()}")
    return out.strip()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_version_exists() -> None:
    """Version 'Sprint Synchronize 2025-W10' exists in project with correct dates and status."""
    try:
        row = op_sql(
            f"SELECT v.name, v.start_date, v.effective_date, v.status "
            f"FROM versions v "
            f"JOIN projects p ON v.project_id = p.id "
            f"WHERE p.name = '{OP_PROJECT}' AND v.name = '{SPRINT_NAME}'"
        )
        if not row:
            check("1. Version exists", 2, False, "version not found")
            return
        parts = row.split("|")
        name = parts[0]
        start = parts[1] if len(parts) > 1 else ""
        end = parts[2] if len(parts) > 2 else ""
        status = parts[3] if len(parts) > 3 else ""
        ok = (
            name == SPRINT_NAME
            and start == SPRINT_START
            and end == SPRINT_END
            and status == "open"
        )
        check("1. Version exists", 2, ok,
              f"name={name}, start={start}, end={end}, status={status}")
    except Exception as e:
        check("1. Version exists", 2, False, f"exception: {e}")


def _get_team_wps(prefix: str) -> list[dict]:
    """Get work packages whose subject starts with the given prefix, assigned to the sprint version."""
    rows = op_sql(
        f"SELECT wp.id, wp.subject, t.name AS type_name, wp.estimated_hours "
        f"FROM work_packages wp "
        f"JOIN projects p ON wp.project_id = p.id "
        f"JOIN types t ON wp.type_id = t.id "
        f"LEFT JOIN versions v ON wp.version_id = v.id "
        f"WHERE p.name = '{OP_PROJECT}' "
        f"AND wp.subject LIKE '[{prefix}]%' "
        f"AND v.name = '{SPRINT_NAME}'"
    )
    wps = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        wps.append({
            "id": int(parts[0]),
            "subject": parts[1],
            "type": parts[2] if len(parts) > 2 else "",
            "hours": float(parts[3]) if len(parts) > 3 and parts[3] else 0.0,
        })
    return wps


def check_2_backend_wp_count() -> None:
    """Exactly 4 Feature work packages with [Backend] prefix assigned to the version."""
    try:
        wps = _get_team_wps(TEAM_A)
        count = len(wps)
        all_feature = all(wp["type"] == "Feature" for wp in wps)
        ok = count == NUM_WP_PER_TEAM and all_feature
        types = set(wp["type"] for wp in wps)
        check("2. Backend WP count", 2, ok,
              f"count={count}, types={types}")
    except Exception as e:
        check("2. Backend WP count", 2, False, f"exception: {e}")


def check_3_data_wp_count() -> None:
    """Exactly 4 Feature work packages with [Data] prefix assigned to the version."""
    try:
        wps = _get_team_wps(TEAM_B)
        count = len(wps)
        all_feature = all(wp["type"] == "Feature" for wp in wps)
        ok = count == NUM_WP_PER_TEAM and all_feature
        types = set(wp["type"] for wp in wps)
        check("3. Data WP count", 2, ok,
              f"count={count}, types={types}")
    except Exception as e:
        check("3. Data WP count", 2, False, f"exception: {e}")


def check_4_backend_hours() -> None:
    """Backend work packages total estimated hours = 32."""
    try:
        wps = _get_team_wps(TEAM_A)
        total = sum(wp["hours"] for wp in wps)
        ok = abs(total - TEAM_A_HOURS) < 0.01
        check("4. Backend total hours", 2, ok,
              f"expected={TEAM_A_HOURS}, got={total}")
    except Exception as e:
        check("4. Backend total hours", 2, False, f"exception: {e}")


def check_5_data_hours() -> None:
    """Data work packages total estimated hours = 28."""
    try:
        wps = _get_team_wps(TEAM_B)
        total = sum(wp["hours"] for wp in wps)
        ok = abs(total - TEAM_B_HOURS) < 0.01
        check("5. Data total hours", 2, ok,
              f"expected={TEAM_B_HOURS}, got={total}")
    except Exception as e:
        check("5. Data total hours", 2, False, f"exception: {e}")


def check_6_follows_relation() -> None:
    """Exactly one 'follows' relation linking a Backend WP to a Data WP (or vice versa)."""
    try:
        backend_wps = _get_team_wps(TEAM_A)
        data_wps = _get_team_wps(TEAM_B)
        backend_ids = {wp["id"] for wp in backend_wps}
        data_ids = {wp["id"] for wp in data_wps}
        all_ids = backend_ids | data_ids

        if not all_ids:
            check("6. Cross-team follows relation", 3, False, "no WPs found")
            return

        id_list = ",".join(str(i) for i in all_ids)
        rows = op_sql(
            f"SELECT r.from_id, r.to_id, r.relation_type "
            f"FROM relations r "
            f"WHERE r.relation_type = 'follows' "
            f"AND (r.from_id IN ({id_list}) OR r.to_id IN ({id_list}))"
        )
        cross_team = []
        for line in rows.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            from_id = int(parts[0])
            to_id = int(parts[1])
            from_backend = from_id in backend_ids
            from_data = from_id in data_ids
            to_backend = to_id in backend_ids
            to_data = to_id in data_ids
            if (from_backend and to_data) or (from_data and to_backend):
                cross_team.append((from_id, to_id))

        ok = len(cross_team) == 1
        check("6. Cross-team follows relation", 3, ok,
              f"cross-team follows count={len(cross_team)}")
    except Exception as e:
        check("6. Cross-team follows relation", 3, False, f"exception: {e}")


def check_7_code_server_comment() -> None:
    """todo-api/app.py has comment '# Sprint Sprint Synchronize 2025-W10: integration touchpoint' at the top."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "head", "-n", "5", "/home/coder/project/todo-api/app.py",
            timeout=10,
        )
        if rc != 0:
            # Try alternative paths
            rc, out, err = docker_exec(
                CODE_SERVER_CONTAINER,
                "find", "/home/coder", "-maxdepth", "4", "-name", "app.py", "-path", "*/todo-api/*",
                timeout=10,
            )
            if rc == 0 and out.strip():
                filepath = out.strip().splitlines()[0]
                rc, out, err = docker_exec(
                    CODE_SERVER_CONTAINER,
                    "head", "-n", "5", filepath,
                    timeout=10,
                )

        found = COMMENT_LINE in out if rc == 0 else False
        # Check it's at the top (first non-empty line or within first few lines)
        at_top = False
        if found:
            for line in out.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                at_top = stripped == COMMENT_LINE.strip()
                break

        ok = found and at_top
        detail = "comment at top" if ok else f"found={found}, at_top={at_top}"
        if rc != 0:
            detail = f"file read failed: {err.strip()}"
        check("7. Code-server comment", 2, ok, detail)
    except Exception as e:
        check("7. Code-server comment", 2, False, f"exception: {e}")


def check_8_baserow_db_table_exist() -> None:
    """Baserow has database 'Sprint Capacity Planner' with table 'Sprint Capacity'."""
    try:
        # Find the database by name (database_database.application_ptr_id == core_application.id)
        db_row = baserow_sql(
            f"SELECT a.id FROM core_application a "
            f"JOIN database_database d ON d.application_ptr_id = a.id "
            f"WHERE a.name = '{BASEROW_DB_NAME}'"
        )
        if not db_row.strip():
            check("8. Baserow DB+table exist", 1, False, "database not found")
            return

        db_id = db_row.strip().splitlines()[0]

        # Find the table by name within that database
        tbl_row = baserow_sql(
            f"SELECT t.id FROM database_table t "
            f"WHERE t.database_id = {db_id} AND t.name = '{BASEROW_TABLE_NAME}'"
        )
        if not tbl_row.strip():
            check("8. Baserow DB+table exist", 1, False,
                  f"database found (id={db_id}) but table '{BASEROW_TABLE_NAME}' not found")
            return

        check("8. Baserow DB+table exist", 1, True, f"db_id={db_id}, table_id={tbl_row.strip()}")
    except Exception as e:
        check("8. Baserow DB+table exist", 1, False, f"exception: {e}")


def _get_baserow_table_id() -> int | None:
    """Return the Baserow internal table ID for 'Sprint Capacity'."""
    db_row = baserow_sql(
        f"SELECT a.id FROM core_application a "
        f"JOIN database_database d ON d.application_ptr_id = a.id "
        f"WHERE a.name = '{BASEROW_DB_NAME}'"
    )
    if not db_row.strip():
        return None
    db_id = db_row.strip().splitlines()[0]
    tbl_row = baserow_sql(
        f"SELECT t.id FROM database_table t "
        f"WHERE t.database_id = {db_id} AND t.name = '{BASEROW_TABLE_NAME}'"
    )
    if not tbl_row.strip():
        return None
    return int(tbl_row.strip().splitlines()[0])


def check_9_baserow_row_count() -> None:
    """Baserow 'Sprint Capacity' table has exactly 2 rows."""
    try:
        table_id = _get_baserow_table_id()
        if table_id is None:
            check("9. Baserow row count", 1, False, "table not found")
            return

        count_str = baserow_sql(
            f"SELECT COUNT(*) FROM database_table_{table_id}"
        )
        count = int(count_str.strip())
        ok = count == 2
        check("9. Baserow row count", 1, ok, f"expected=2, got={count}")
    except Exception as e:
        check("9. Baserow row count", 1, False, f"exception: {e}")


def check_10_baserow_row_data() -> None:
    """Baserow rows have correct Team, Planned Hours, Work Package Count, Has Cross-Team Dep."""
    try:
        table_id = _get_baserow_table_id()
        if table_id is None:
            check("10. Baserow row data", 3, False, "table not found")
            return

        # Get field mappings for the table
        fields_raw = baserow_sql(
            f"SELECT f.id, f.name FROM database_field f "
            f"WHERE f.table_id = {table_id} ORDER BY f.id"
        )
        field_map = {}  # name -> field_<id> column name
        for line in fields_raw.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            fid = parts[0].strip()
            fname = parts[1].strip() if len(parts) > 1 else ""
            field_map[fname] = f"field_{fid}"

        team_col = field_map.get("Team", "")
        hours_col = field_map.get("Planned Hours", "")
        count_col = field_map.get("Work Package Count", "")
        dep_col = field_map.get("Has Cross-Team Dep", "")

        if not all([team_col, hours_col, count_col, dep_col]):
            check("10. Baserow row data", 3, False,
                  f"missing fields: team={team_col}, hours={hours_col}, count={count_col}, dep={dep_col}")
            return

        rows_raw = baserow_sql(
            f"SELECT {team_col}, {hours_col}, {count_col}, {dep_col} "
            f"FROM database_table_{table_id} ORDER BY {team_col}"
        )
        rows = []
        for line in rows_raw.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            rows.append({
                "team": parts[0].strip(),
                "hours": float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0,
                "count": int(float(parts[2].strip())) if len(parts) > 2 and parts[2].strip() else 0,
                "dep": parts[3].strip().lower() in ("t", "true", "1") if len(parts) > 3 else False,
            })

        if len(rows) != 2:
            check("10. Baserow row data", 3, False, f"expected 2 rows, parsed {len(rows)}")
            return

        # Build expected: one team has dep=true, the other false
        row_by_team = {r["team"]: r for r in rows}
        backend = row_by_team.get(TEAM_A)
        data = row_by_team.get(TEAM_B)

        issues = []
        if backend is None:
            issues.append(f"no row for team '{TEAM_A}'")
        else:
            if abs(backend["hours"] - TEAM_A_HOURS) > 0.01:
                issues.append(f"Backend hours: expected {TEAM_A_HOURS}, got {backend['hours']}")
            if backend["count"] != NUM_WP_PER_TEAM:
                issues.append(f"Backend WP count: expected {NUM_WP_PER_TEAM}, got {backend['count']}")

        if data is None:
            issues.append(f"no row for team '{TEAM_B}'")
        else:
            if abs(data["hours"] - TEAM_B_HOURS) > 0.01:
                issues.append(f"Data hours: expected {TEAM_B_HOURS}, got {data['hours']}")
            if data["count"] != NUM_WP_PER_TEAM:
                issues.append(f"Data WP count: expected {NUM_WP_PER_TEAM}, got {data['count']}")

        # Exactly one team should have Has Cross-Team Dep = true
        if backend is not None and data is not None:
            dep_count = sum(1 for r in rows if r["dep"])
            if dep_count != 1:
                issues.append(f"expected exactly 1 row with dep=true, got {dep_count}")

        ok = not issues
        check("10. Baserow row data", 3, ok,
              "all fields correct" if ok else "; ".join(issues))
    except Exception as e:
        check("10. Baserow row data", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_version_exists()
    check_2_backend_wp_count()
    check_3_data_wp_count()
    check_4_backend_hours()
    check_5_data_hours()
    check_6_follows_relation()
    check_7_code_server_comment()
    check_8_baserow_db_table_exist()
    check_9_baserow_row_count()
    check_10_baserow_row_data()

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
