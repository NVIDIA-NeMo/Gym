# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_030/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-030-I1: Code Complexity Audit Across Three Workspace Projects

Checks: 15 weighted checks across code-server, baserow, openproject.
Strategy: Baserow via REST API, code-server via docker exec filesystem,
          OpenProject via docker exec embedded postgres.

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER.
"""

import os
import sys
import subprocess
import json
import re
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

_required = {
    "CODE_SERVER_CONTAINER": CODE_SERVER_CONTAINER,
    "BASEROW_PORT": BASEROW_PORT,
    "BASEROW_CONTAINER": BASEROW_CONTAINER,
    "BASEROW_DB_CONTAINER": BASEROW_DB_CONTAINER,
    "OPENPROJECT_CONTAINER": OPENPROJECT_CONTAINER,
}
for var, val in _required.items():
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"

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
    """Get Baserow JWT access token."""
    resp = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    # Baserow returns access_token (JWT) or token depending on version
    return data.get("access_token") or data.get("token", "")


def baserow_get(path: str, token: str, params: dict | None = None) -> dict | list:
    resp = requests.get(
        f"{BASEROW_URL}/api{path}",
        headers={"Authorization": f"JWT {token}"},
        params=params or {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def op_db_query(sql: str) -> str:
    """Query OpenProject embedded postgres."""
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "psql", "-U", "openproject", "-d", "openproject",
        "-t", "-A", "-c", sql,
        timeout=15,
    )
    return out.strip()


# ── Shared state for cross-check consistency ──────────────────────────────────
_baserow_rows = []  # populated by check_3
_baserow_table_id = None  # populated by check_2


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_baserow_database_exists() -> dict | None:
    """Baserow database 'Code Complexity Audit Q2 2025' exists."""
    try:
        token = baserow_auth()
        # List all applications (databases)
        apps = baserow_get("/applications/", token)
        target_db = None
        for app in apps:
            if app.get("name") == "Code Complexity Audit Q2 2025" and app.get("type") == "database":
                target_db = app
                break
        check("1. Baserow DB 'Code Complexity Audit Q2 2025' exists", 1,
              target_db is not None,
              f"found DB id={target_db['id']}" if target_db else "database not found")
        return {"token": token, "db": target_db} if target_db else {"token": token, "db": None}
    except Exception as e:
        check("1. Baserow DB 'Code Complexity Audit Q2 2025' exists", 1, False, f"exception: {e}")
        return None


def check_2_table_and_fields(ctx: dict | None) -> dict | None:
    """Table 'Complexity Metrics' exists with required fields."""
    global _baserow_table_id
    if not ctx or not ctx.get("db"):
        check("2. Table 'Complexity Metrics' with required fields", 2, False, "no DB context")
        return ctx
    try:
        token = ctx["token"]
        db_id = ctx["db"]["id"]
        tables = baserow_get(f"/database/tables/database/{db_id}/", token)
        target_table = None
        for t in tables:
            if t.get("name") == "Complexity Metrics":
                target_table = t
                break
        if not target_table:
            check("2. Table 'Complexity Metrics' with required fields", 2, False, "table not found")
            return ctx
        table_id = target_table["id"]
        _baserow_table_id = table_id
        # Get fields
        fields = baserow_get(f"/database/fields/table/{table_id}/", token)
        field_names = {f["name"] for f in fields}
        required_fields = {"Metric ID", "Project", "File Path", "Lines Of Code",
                           "Function Count", "Avg Function Length", "Complexity Band", "Captured At"}
        missing = required_fields - field_names
        passed = len(missing) == 0
        detail = f"table id={table_id}, fields OK" if passed else f"missing fields: {missing}"
        check("2. Table 'Complexity Metrics' with required fields", 2, passed, detail)
        ctx["table_id"] = table_id
        ctx["fields"] = {f["name"]: f for f in fields}
        return ctx
    except Exception as e:
        check("2. Table 'Complexity Metrics' with required fields", 2, False, f"exception: {e}")
        return ctx


def check_3_rows_cover_all_projects(ctx: dict | None) -> None:
    """Rows exist for all 3 projects: blog-engine, data-analyzer, todo-api."""
    global _baserow_rows
    if not ctx or "table_id" not in ctx:
        check("3. Rows cover all 3 projects", 2, False, "no table context")
        return
    try:
        token = ctx["token"]
        table_id = ctx["table_id"]
        # Fetch all rows (expect not too many, <200)
        page = 1
        all_rows = []
        while True:
            data = baserow_get(f"/database/rows/table/{table_id}/",
                               token, params={"size": 200, "page": page})
            all_rows.extend(data.get("results", []))
            if not data.get("next"):
                break
            page += 1
        _baserow_rows = all_rows

        # Extract project values — could be dict (single_select) or string
        projects_found = set()
        for row in all_rows:
            proj_val = row.get("Project") or row.get("field_Project")
            # Single-select fields return {"id": ..., "value": "..."} or just a string
            if isinstance(proj_val, dict):
                proj_val = proj_val.get("value", "")
            if proj_val:
                projects_found.add(proj_val)

        # If field names are by field_id, try to find them
        if not projects_found and all_rows:
            # Fields might be keyed by field_<id>
            fields = ctx.get("fields", {})
            proj_field = fields.get("Project", {})
            proj_field_key = f"field_{proj_field.get('id', '')}"
            for row in all_rows:
                proj_val = row.get(proj_field_key)
                if isinstance(proj_val, dict):
                    proj_val = proj_val.get("value", "")
                if proj_val:
                    projects_found.add(proj_val)

        expected_projects = {"blog-engine", "data-analyzer", "todo-api"}
        missing = expected_projects - projects_found
        passed = len(missing) == 0 and len(all_rows) > 0
        check("3. Rows cover all 3 projects", 2, passed,
              f"{len(all_rows)} rows, projects={projects_found}" if passed
              else f"{len(all_rows)} rows, missing projects: {missing}")
    except Exception as e:
        check("3. Rows cover all 3 projects", 2, False, f"exception: {e}")


def _get_field_value(row: dict, field_name: str, fields: dict) -> object:
    """Get a field value from a row, trying both name-based and field_id-based keys."""
    val = row.get(field_name)
    if val is not None:
        return val
    field_info = fields.get(field_name, {})
    field_key = f"field_{field_info.get('id', '')}"
    return row.get(field_key)


def check_4_metric_ids_sequential(ctx: dict | None) -> None:
    """Metric IDs follow CM-NNN format starting at CM-001."""
    if not _baserow_rows or not ctx:
        check("4. Metric IDs follow CM-NNN format sequentially", 1, False, "no rows")
        return
    try:
        fields = ctx.get("fields", {})
        ids = []
        for row in _baserow_rows:
            mid = _get_field_value(row, "Metric ID", fields)
            if isinstance(mid, dict):
                mid = mid.get("value", "")
            ids.append(str(mid) if mid else "")

        # Check format CM-NNN
        pattern = re.compile(r"^CM-(\d{3})$")
        valid = all(pattern.match(i) for i in ids if i)
        # Check sequential from 001
        nums = []
        for i in ids:
            m = pattern.match(i) if i else None
            if m:
                nums.append(int(m.group(1)))
        expected_seq = list(range(1, len(nums) + 1))
        sequential = nums == expected_seq
        passed = valid and sequential and len(nums) > 0
        check("4. Metric IDs follow CM-NNN format sequentially", 1, passed,
              f"{len(nums)} IDs, first={ids[0] if ids else '?'}, last={ids[-1] if ids else '?'}"
              if passed else f"valid={valid}, sequential={sequential}, ids_sample={ids[:3]}")
    except Exception as e:
        check("4. Metric IDs follow CM-NNN format sequentially", 1, False, f"exception: {e}")


def check_5_complexity_band_correct(ctx: dict | None) -> None:
    """Complexity Band correctly assigned per LOC/avg-fn-length thresholds."""
    if not _baserow_rows or not ctx:
        check("5. Complexity Band assigned correctly per thresholds", 2, False, "no rows")
        return
    try:
        fields = ctx.get("fields", {})
        mismatches = []
        for row in _baserow_rows:
            loc_val = _get_field_value(row, "Lines Of Code", fields)
            avg_val = _get_field_value(row, "Avg Function Length", fields)
            band_val = _get_field_value(row, "Complexity Band", fields)
            mid_val = _get_field_value(row, "Metric ID", fields)
            if isinstance(mid_val, dict):
                mid_val = mid_val.get("value", "")

            loc = int(loc_val) if loc_val is not None else 0
            try:
                avg = float(avg_val) if avg_val is not None else 0.0
            except (ValueError, TypeError):
                avg = 0.0

            if isinstance(band_val, dict):
                band = band_val.get("value", "")
            else:
                band = str(band_val) if band_val else ""

            # Determine expected band
            if loc >= 1000 or avg >= 40:
                expected = "Critical"
            elif loc >= 500 or avg >= 25:
                expected = "High"
            elif loc >= 200:
                expected = "Medium"
            else:
                expected = "Low"

            if band != expected:
                mismatches.append(f"{mid_val}: got '{band}' expected '{expected}' (LOC={loc}, avg={avg})")

        passed = len(mismatches) == 0 and len(_baserow_rows) > 0
        check("5. Complexity Band assigned correctly per thresholds", 2, passed,
              f"all {len(_baserow_rows)} rows correct" if passed
              else f"{len(mismatches)} mismatches: {mismatches[:3]}")
    except Exception as e:
        check("5. Complexity Band assigned correctly per thresholds", 2, False, f"exception: {e}")


def check_6_captured_at_date(ctx: dict | None) -> None:
    """All rows have Captured At = 2025-05-15."""
    if not _baserow_rows or not ctx:
        check("6. All rows have Captured At = 2025-05-15", 1, False, "no rows")
        return
    try:
        fields = ctx.get("fields", {})
        wrong = 0
        for row in _baserow_rows:
            cap_val = _get_field_value(row, "Captured At", fields)
            date_str = str(cap_val) if cap_val else ""
            if "2025-05-15" not in date_str:
                wrong += 1
        passed = wrong == 0 and len(_baserow_rows) > 0
        check("6. All rows have Captured At = 2025-05-15", 1, passed,
              f"all {len(_baserow_rows)} rows OK" if passed else f"{wrong} rows with wrong date")
    except Exception as e:
        check("6. All rows have Captured At = 2025-05-15", 1, False, f"exception: {e}")


def check_7_row_ordering(ctx: dict | None) -> None:
    """Rows ordered by Project ascending, then File Path ascending."""
    if not _baserow_rows or not ctx:
        check("7. Rows ordered by Project asc, File Path asc", 2, False, "no rows")
        return
    try:
        fields = ctx.get("fields", {})
        pairs = []
        for row in _baserow_rows:
            proj_val = _get_field_value(row, "Project", fields)
            fp_val = _get_field_value(row, "File Path", fields)
            if isinstance(proj_val, dict):
                proj_val = proj_val.get("value", "")
            proj = str(proj_val) if proj_val else ""
            fp = str(fp_val) if fp_val else ""
            pairs.append((proj, fp))
        sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
        passed = pairs == sorted_pairs and len(pairs) > 0
        check("7. Rows ordered by Project asc, File Path asc", 2, passed,
              f"{len(pairs)} rows in correct order" if passed
              else f"order mismatch at first diff")
    except Exception as e:
        check("7. Rows ordered by Project asc, File Path asc", 2, False, f"exception: {e}")


def check_8_top_offenders_view(ctx: dict | None) -> None:
    """'Top Offenders' Grid view exists with filter on High/Critical and sort by LOC desc."""
    if not ctx or "table_id" not in ctx:
        check("8. 'Top Offenders' Grid view exists", 2, False, "no table context")
        return
    try:
        token = ctx["token"]
        table_id = ctx["table_id"]
        views = baserow_get(f"/database/views/table/{table_id}/", token)
        target = None
        for v in views:
            if v.get("name") == "Top Offenders":
                target = v
                break
        if not target:
            check("8. 'Top Offenders' Grid view exists", 2, False, "view not found")
            return
        is_grid = target.get("type") == "grid"
        # Check filters and sorts via view detail
        view_id = target["id"]
        # Get filters
        filters_data = baserow_get(f"/database/views/{view_id}/filters/", token)
        # Get sorts
        sorts_data = baserow_get(f"/database/views/{view_id}/sortings/", token)

        has_filter = len(filters_data) > 0 if isinstance(filters_data, list) else False
        has_sort = len(sorts_data) > 0 if isinstance(sorts_data, list) else False

        passed = is_grid and (has_filter or has_sort)
        check("8. 'Top Offenders' Grid view exists", 2, passed,
              f"grid={is_grid}, filters={len(filters_data) if isinstance(filters_data, list) else '?'}, "
              f"sorts={len(sorts_data) if isinstance(sorts_data, list) else '?'}")
    except Exception as e:
        check("8. 'Top Offenders' Grid view exists", 2, False, f"exception: {e}")


def check_9_by_band_kanban_view(ctx: dict | None) -> None:
    """'By Band' Kanban view exists stacked by Complexity Band."""
    if not ctx or "table_id" not in ctx:
        check("9. 'By Band' Kanban view exists", 1, False, "no table context")
        return
    try:
        token = ctx["token"]
        table_id = ctx["table_id"]
        views = baserow_get(f"/database/views/table/{table_id}/", token)
        target = None
        for v in views:
            if v.get("name") == "By Band":
                target = v
                break
        if not target:
            check("9. 'By Band' Kanban view exists", 1, False, "view not found")
            return
        is_kanban = target.get("type") == "kanban"
        check("9. 'By Band' Kanban view exists", 1, is_kanban,
              f"type={target.get('type')}")
    except Exception as e:
        check("9. 'By Band' Kanban view exists", 1, False, f"exception: {e}")


def check_10_audit_file_exists() -> list[str]:
    """File devops-configs/docs/complexity-audit-2025-05-15.md exists in code-server."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER, "cat",
            "/home/coder/project/devops-configs/docs/complexity-audit-2025-05-15.md",
            timeout=10,
        )
        if rc != 0:
            # Try alternate path
            rc, out, err = docker_exec(
                CODE_SERVER_CONTAINER, "cat",
                "/home/coder/devops-configs/docs/complexity-audit-2025-05-15.md",
                timeout=10,
            )
        lines = out.strip().split("\n") if rc == 0 and out.strip() else []
        check("10. Audit markdown file exists in code-server", 1, rc == 0 and len(lines) >= 5,
              f"{len(lines)} lines" if rc == 0 else f"file not found (rc={rc})")
        return lines
    except Exception as e:
        check("10. Audit markdown file exists in code-server", 1, False, f"exception: {e}")
        return []


def check_11_audit_file_header(lines: list[str]) -> None:
    """Lines 1-2: heading and sorted project list."""
    if len(lines) < 2:
        check("11. Audit file header and project list (lines 1-2)", 2, False, "fewer than 2 lines")
        return
    try:
        line1_ok = "# Complexity Audit" in lines[0] and "2025-05-15" in lines[0]
        line2 = lines[1] if len(lines) > 1 else ""
        # Projects should be sorted alphabetically: blog-engine, data-analyzer, todo-api
        projects_ok = ("blog-engine" in line2 and "data-analyzer" in line2 and "todo-api" in line2)
        passed = line1_ok and projects_ok
        check("11. Audit file header and project list (lines 1-2)", 2, passed,
              f"line1_ok={line1_ok}, projects_ok={projects_ok}, line1='{lines[0][:60]}', line2='{line2[:60]}'")
    except Exception as e:
        check("11. Audit file header and project list (lines 1-2)", 2, False, f"exception: {e}")


def check_12_audit_file_body(lines: list[str]) -> None:
    """Lines 3-5: total files, band counts, top file."""
    if len(lines) < 5:
        check("12. Audit file body lines 3-5 (counts and top file)", 2, False,
              f"only {len(lines)} lines, need 5")
        return
    try:
        line3 = lines[2]
        line4 = lines[3]
        line5 = lines[4]

        # Line 3: "Total files measured: <N>"
        line3_ok = "Total files measured:" in line3
        # Line 4: "Critical: <C>; High: <H>; Medium: <M>; Low: <L>"
        line4_ok = all(band in line4 for band in ["Critical:", "High:", "Medium:", "Low:"])
        # Line 5: "Top file: <path> (<project>, <LOC> LOC)"
        line5_ok = "Top file:" in line5 and "LOC" in line5

        passed = line3_ok and line4_ok and line5_ok
        check("12. Audit file body lines 3-5 (counts and top file)", 2, passed,
              f"line3_ok={line3_ok}, line4_ok={line4_ok}, line5_ok={line5_ok}")
    except Exception as e:
        check("12. Audit file body lines 3-5 (counts and top file)", 2, False, f"exception: {e}")


def check_13_op_work_packages_exist() -> list[dict]:
    """OpenProject: Task work packages exist in 'security-audit' project for High/Critical rows."""
    try:
        # Find project id for security-audit
        proj_row = op_db_query(
            "SELECT id FROM projects WHERE identifier = 'security-audit' LIMIT 1;"
        )
        if not proj_row:
            check("13. Work packages exist in 'security-audit' project", 2, False,
                  "project 'security-audit' not found")
            return []
        project_id = int(proj_row.strip())

        # Get Task type id
        type_row = op_db_query("SELECT id FROM types WHERE name = 'Task' LIMIT 1;")
        task_type_id = int(type_row.strip()) if type_row.strip() else None

        # Get work packages in this project
        type_filter = f" AND type_id = {task_type_id}" if task_type_id else ""
        wp_rows = op_db_query(
            f"SELECT wp.id, wp.subject, wp.description, "
            f"u.login AS assignee_login, "
            f"s.name AS status_name, "
            f"e.name AS priority_name "
            f"FROM work_packages wp "
            f"LEFT JOIN users u ON wp.assigned_to_id = u.id "
            f"LEFT JOIN statuses s ON wp.status_id = s.id "
            f"LEFT JOIN enumerations e ON wp.priority_id = e.id "
            f"WHERE wp.project_id = {project_id}{type_filter} "
            f"ORDER BY wp.id;"
        )
        wps = []
        if wp_rows:
            for line in wp_rows.split("\n"):
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) >= 6:
                    wps.append({
                        "id": parts[0].strip(),
                        "subject": parts[1].strip(),
                        "description": parts[2].strip(),
                        "assignee": parts[3].strip(),
                        "status": parts[4].strip(),
                        "priority": parts[5].strip(),
                    })

        # Should have up to 10 work packages
        passed = 1 <= len(wps) <= 10
        check("13. Work packages exist in 'security-audit' project", 2, passed,
              f"{len(wps)} Task work packages found")
        return wps
    except Exception as e:
        check("13. Work packages exist in 'security-audit' project", 2, False, f"exception: {e}")
        return []


def check_14_wp_subject_format(wps: list[dict]) -> None:
    """Work package subjects match 'Refactor: <path> (<LOC> LOC, <avg> avg fn length)'."""
    if not wps:
        check("14. Work package subjects match format", 2, False, "no work packages")
        return
    try:
        pattern = re.compile(r"^Refactor: .+ \(\d+ LOC, [\d.]+ avg fn length\)$")
        matching = sum(1 for wp in wps if pattern.match(wp["subject"]))
        passed = matching == len(wps)
        check("14. Work package subjects match format", 2, passed,
              f"{matching}/{len(wps)} match pattern"
              + (f", sample='{wps[0]['subject'][:70]}'" if wps and not passed else ""))
    except Exception as e:
        check("14. Work package subjects match format", 2, False, f"exception: {e}")


def check_15_wp_assignee_admin(wps: list[dict]) -> None:
    """All work packages assigned to admin."""
    if not wps:
        check("15. Work packages assigned to admin", 1, False, "no work packages")
        return
    try:
        admin_count = sum(1 for wp in wps if wp.get("assignee") == "admin")
        passed = admin_count == len(wps)
        check("15. Work packages assigned to admin", 1, passed,
              f"{admin_count}/{len(wps)} assigned to admin")
    except Exception as e:
        check("15. Work packages assigned to admin", 1, False, f"exception: {e}")


def check_16_wp_priority_mapping(wps: list[dict]) -> None:
    """Priority: High for Critical band, Normal for High band (from description)."""
    if not wps:
        check("16. Work package priority mapping correct", 2, False, "no work packages")
        return
    try:
        mismatches = []
        for wp in wps:
            desc = wp.get("description", "")
            priority = wp.get("priority", "")
            # Extract band from description: "Band: <band>"
            band_match = re.search(r"Band:\s*(Critical|High)", desc)
            if band_match:
                band = band_match.group(1)
                expected_priority = "High" if band == "Critical" else "Normal"
                if priority != expected_priority:
                    mismatches.append(
                        f"subject='{wp['subject'][:40]}': band={band}, "
                        f"priority={priority}, expected={expected_priority}")
            else:
                mismatches.append(f"subject='{wp['subject'][:40]}': band not found in description")

        passed = len(mismatches) == 0 and len(wps) > 0
        check("16. Work package priority mapping correct", 2, passed,
              f"all {len(wps)} correct" if passed else f"{len(mismatches)} issues: {mismatches[:2]}")
    except Exception as e:
        check("16. Work package priority mapping correct", 2, False, f"exception: {e}")


def check_17_wp_description_format(wps: list[dict]) -> None:
    """Description: 'Project: <P>; Function Count: <FC>; Band: <B>; Audit: 2025-05-15'."""
    if not wps:
        check("17. Work package descriptions match format", 2, False, "no work packages")
        return
    try:
        pattern = re.compile(
            r"Project: .+; Function Count: \d+; Band: (Critical|High); Audit: 2025-05-15"
        )
        matching = sum(1 for wp in wps if pattern.search(wp.get("description", "")))
        passed = matching == len(wps)
        check("17. Work package descriptions match format", 2, passed,
              f"{matching}/{len(wps)} match"
              + (f", sample='{wps[0].get('description', '')[:70]}'" if wps and not passed else ""))
    except Exception as e:
        check("17. Work package descriptions match format", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Baserow checks
    ctx = check_1_baserow_database_exists()
    ctx = check_2_table_and_fields(ctx)
    check_3_rows_cover_all_projects(ctx)
    check_4_metric_ids_sequential(ctx)
    check_5_complexity_band_correct(ctx)
    check_6_captured_at_date(ctx)
    check_7_row_ordering(ctx)
    check_8_top_offenders_view(ctx)
    check_9_by_band_kanban_view(ctx)

    # code-server checks
    lines = check_10_audit_file_exists()
    check_11_audit_file_header(lines)
    check_12_audit_file_body(lines)

    # OpenProject checks
    wps = check_13_op_work_packages_exist()
    check_14_wp_subject_format(wps)
    check_15_wp_assignee_admin(wps)
    check_16_wp_priority_mapping(wps)
    check_17_wp_description_format(wps)

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
