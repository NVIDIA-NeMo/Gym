# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_005/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-005-I2: Dependency Audit Across blog-engine and weather-dashboard Projects

Checks: 13 weighted checks across code-server, baserow, metabase, openproject.
Strategy: docker exec (code-server, baserow DB, openproject DB) + REST API (metabase)

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  METABASE_PORT, METABASE_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import sys
import subprocess
import json
import re
import shlex
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

_required = {
    "CODE_SERVER_CONTAINER": None,
    "BASEROW_DB_CONTAINER": None,
    "METABASE_PORT": None,
    "OPENPROJECT_CONTAINER": None,
}
for var in _required:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    _required[var] = val

CODE_SERVER_CONTAINER = _required["CODE_SERVER_CONTAINER"]
BASEROW_DB_CONTAINER = _required["BASEROW_DB_CONTAINER"]
METABASE_PORT = _required["METABASE_PORT"]
OPENPROJECT_CONTAINER = _required["OPENPROJECT_CONTAINER"]

METABASE_BASE = f"http://{HOST}:{METABASE_PORT}"

# Slot values
BASEROW_DB_NAME = "Frontend Dependency Audit 2026"
TABLE_NAME = "Dependency Inventory"
AUDIT_DATE = "2026-04-15"
STALE_MAJOR_THRESHOLD = 3
KNOWN_STALE_LIST = ["express", "ejs", "react"]
METABASE_COLLECTION = "Frontend Audit Insights"
DASHBOARD_NAME = "Frontend Dependency Health"
OP_PROJECT = "Marketing Website"
WP_SUBJECT = "Upgrade stale dependencies: 2026-04-15"

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
    """Run a SQL query against the Baserow Postgres DB."""
    rc, out, err = docker_exec(
        BASEROW_DB_CONTAINER,
        "env", "PGPASSWORD=kdpzkuyhsgb22onku8y7rxkx3czej88nxpngaz4mlmgad67vpc", "psql", "-h", "127.0.0.1", "-U", "baserow", "-d", "baserow", "-t", "-A", "-c", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"psql error: {err.strip()}")
    return out.strip()


def openproject_sql(query: str) -> str:
    """Run a SQL query against the OpenProject embedded Postgres DB."""
    cmd = f"PGPASSWORD=openproject psql -h 127.0.0.1 -U openproject -d openproject -t -A -c {shlex.quote(query)}"
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "bash", "-c", cmd,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"psql error: {err.strip()}")
    return out.strip()


def metabase_session() -> str:
    """Get a Metabase session token."""
    r = requests.post(
        f"{METABASE_BASE}/api/session",
        json={"username": "admin@metabase.local", "password": "mw-admin-123"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["id"]


def metabase_get(path: str, token: str) -> dict:
    r = requests.get(
        f"{METABASE_BASE}/api/{path}",
        headers={"X-Metabase-Session": token},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ── Ground truth: read package.json files from code-server ────────────────────
def get_ground_truth_deps() -> dict[str, dict[str, str]]:
    """Returns {project_name: {dep_name: version_string}} from code-server."""
    deps = {}
    for project in ("blog-engine", "weather-dashboard"):
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "cat", f"/home/coder/{project}/package.json",
            timeout=10,
        )
        if rc != 0:
            # Try alternate path
            rc, out, err = docker_exec(
                CODE_SERVER_CONTAINER,
                "cat", f"/home/coder/project/{project}/package.json",
                timeout=10,
            )
        if rc != 0:
            # Try workspace root
            rc, out, err = docker_exec(
                CODE_SERVER_CONTAINER,
                "bash", "-c", f"find /home -name package.json -path '*{project}*' 2>/dev/null | head -1 | xargs cat",
                timeout=10,
            )
        if rc == 0 and out.strip():
            pkg = json.loads(out)
            project_deps = {}
            for section in ("dependencies",):
                if section in pkg:
                    project_deps.update(pkg[section])
            deps[project] = project_deps
        else:
            deps[project] = {}
    return deps


def is_stale(dep_name: str, version_str: str) -> bool:
    """Determine if a dependency is stale per the task rules."""
    if dep_name.lower() in [s.lower() for s in KNOWN_STALE_LIST]:
        return True
    # Extract major version from version string (strip ^, ~, etc.)
    clean = re.sub(r'^[^0-9]*', '', version_str)
    match = re.match(r'(\d+)', clean)
    if match:
        major = int(match.group(1))
        if major < STALE_MAJOR_THRESHOLD:
            return True
    return False


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_package_json_files() -> dict[str, dict[str, str]]:
    """Check that package.json files are readable from code-server."""
    try:
        deps = get_ground_truth_deps()
        total_deps = sum(len(v) for v in deps.values())
        has_both = len(deps.get("blog-engine", {})) > 0 and len(deps.get("weather-dashboard", {})) > 0
        check("1. package.json files readable", 1, has_both,
              f"blog-engine={len(deps.get('blog-engine', {}))} deps, weather-dashboard={len(deps.get('weather-dashboard', {}))} deps")
        return deps
    except Exception as e:
        check("1. package.json files readable", 1, False, f"exception: {e}")
        return {}


def check_2_baserow_database_exists() -> int | None:
    """Check Baserow database 'Frontend Dependency Audit 2026' exists."""
    try:
        row = baserow_sql(
            f"SELECT a.id FROM core_application a "
            f"JOIN database_database d ON d.application_ptr_id = a.id "
            f"WHERE a.name = '{BASEROW_DB_NAME}'"
        )
        db_id = int(row.split('\n')[0]) if row else None
        check("2. Baserow database exists", 1, db_id is not None,
              f"db_id={db_id}" if db_id else "not found")
        return db_id
    except Exception as e:
        check("2. Baserow database exists", 1, False, f"exception: {e}")
        return None


def check_3_baserow_table_exists(db_id: int | None) -> int | None:
    """Check table 'Dependency Inventory' exists with required fields."""
    try:
        if db_id is None:
            check("3. Baserow table with fields", 1, False, "no database found")
            return None
        row = baserow_sql(
            f"SELECT t.id FROM database_table t WHERE t.database_id = {db_id} AND t.name = '{TABLE_NAME}'"
        )
        table_id = int(row.split('\n')[0]) if row else None
        if table_id is None:
            check("3. Baserow table with fields", 1, False, "table not found")
            return None

        # Check fields exist
        fields_raw = baserow_sql(
            f"SELECT f.name FROM database_field f WHERE f.table_id = {table_id} AND f.trashed = false ORDER BY f.name"
        )
        fields = set(f.strip() for f in fields_raw.split('\n') if f.strip())
        required = {"Project", "Dependency Name", "Current Version", "Manifest File", "Captured At", "Stale"}
        missing = required - fields
        check("3. Baserow table with fields", 1, not missing,
              f"fields={sorted(fields)}" if missing else "all fields present")
        return table_id
    except Exception as e:
        check("3. Baserow table with fields", 1, False, f"exception: {e}")
        return None


def check_4_baserow_row_count(table_id: int | None, ground_truth: dict[str, dict[str, str]]) -> None:
    """Check row count matches total dependencies."""
    try:
        if table_id is None:
            check("4. Correct row count", 2, False, "no table found")
            return
        expected_count = sum(len(v) for v in ground_truth.values())
        actual = baserow_sql(f"SELECT count(*) FROM database_table_{table_id}")
        actual_count = int(actual) if actual else 0
        check("4. Correct row count", 2, actual_count == expected_count,
              f"expected={expected_count}, actual={actual_count}")
    except Exception as e:
        check("4. Correct row count", 2, False, f"exception: {e}")


def check_5_baserow_captured_at(table_id: int | None) -> None:
    """Check all rows have Captured At = 2026-04-15."""
    try:
        if table_id is None:
            check("5. Captured At dates", 2, False, "no table found")
            return

        # Find the Captured At field column
        field_info = baserow_sql(
            f"SELECT f.id, f.db_column FROM database_field f "
            f"WHERE f.table_id = {table_id} AND f.name = 'Captured At' AND f.trashed = false"
        )
        if not field_info:
            check("5. Captured At dates", 2, False, "Captured At field not found")
            return
        db_column = field_info.split('|')[1].strip() if '|' in field_info else f"field_{field_info.split('|')[0].strip()}"

        total = baserow_sql(f"SELECT count(*) FROM database_table_{table_id}")
        correct = baserow_sql(
            f"SELECT count(*) FROM database_table_{table_id} WHERE {db_column}::text LIKE '2026-04-15%'"
        )
        total_n = int(total)
        correct_n = int(correct)
        check("5. Captured At dates", 2, total_n > 0 and total_n == correct_n,
              f"{correct_n}/{total_n} rows have 2026-04-15")
    except Exception as e:
        check("5. Captured At dates", 2, False, f"exception: {e}")


def check_6_baserow_stale_boolean(table_id: int | None, ground_truth: dict[str, dict[str, str]]) -> None:
    """Check Stale boolean is correctly computed for each dependency."""
    try:
        if table_id is None or not ground_truth:
            check("6. Stale boolean correctness", 3, False, "no table or ground truth")
            return

        # Get field column mappings
        fields_raw = baserow_sql(
            f"SELECT f.name, f.db_column FROM database_field f "
            f"WHERE f.table_id = {table_id} AND f.trashed = false"
        )
        field_map = {}
        for line in fields_raw.split('\n'):
            if '|' in line:
                name, col = line.split('|', 1)
                field_map[name.strip()] = col.strip()

        dep_col = field_map.get("Dependency Name", "")
        proj_col = field_map.get("Project", "")
        ver_col = field_map.get("Current Version", "")
        stale_col = field_map.get("Stale", "")

        if not all([dep_col, proj_col, ver_col, stale_col]):
            check("6. Stale boolean correctness", 3, False,
                  f"missing columns: dep={dep_col}, proj={proj_col}, ver={ver_col}, stale={stale_col}")
            return

        rows_raw = baserow_sql(
            f"SELECT {proj_col}, {dep_col}, {ver_col}, {stale_col} "
            f"FROM database_table_{table_id} ORDER BY {proj_col}, {dep_col}"
        )

        wrong = []
        for line in rows_raw.split('\n'):
            if not line.strip():
                continue
            parts = line.split('|')
            if len(parts) < 4:
                continue
            proj = parts[0].strip()
            dep = parts[1].strip()
            ver = parts[2].strip()
            stale_val = parts[3].strip().lower()
            actual_stale = stale_val in ('true', 't', '1', 'yes')

            expected_stale = is_stale(dep, ver)
            if actual_stale != expected_stale:
                wrong.append(f"{dep}(expected={expected_stale}, got={actual_stale})")

        check("6. Stale boolean correctness", 3, not wrong,
              "all correct" if not wrong else f"wrong: {wrong[:5]}")
    except Exception as e:
        check("6. Stale boolean correctness", 3, False, f"exception: {e}")


def check_7_metabase_collection(token: str) -> int | None:
    """Check Metabase collection 'Frontend Audit Insights' exists."""
    try:
        collections = metabase_get("collection", token)
        coll = None
        for c in collections:
            if c.get("name") == METABASE_COLLECTION:
                coll = c
                break
        check("7. Metabase collection exists", 1, coll is not None,
              f"id={coll['id']}" if coll else "not found")
        return coll["id"] if coll else None
    except Exception as e:
        check("7. Metabase collection exists", 1, False, f"exception: {e}")
        return None


def check_8_metabase_bar_chart(token: str, coll_id: int | None) -> int | None:
    """Check 'Dependencies by Project' bar chart question exists."""
    try:
        if coll_id is None:
            check("8. Bar chart question", 2, False, "no collection found")
            return None
        items = metabase_get(f"collection/{coll_id}/items", token)
        card = None
        for item in items.get("data", items) if isinstance(items, dict) else items:
            if item.get("name") == "Dependencies by Project" and item.get("model") == "card":
                card = item
                break
        if card is None:
            check("8. Bar chart question", 2, False, "question not found")
            return None
        # Fetch card detail to check display type
        card_detail = metabase_get(f"card/{card['id']}", token)
        display = card_detail.get("display", "")
        is_bar = display == "bar"
        check("8. Bar chart question", 2, is_bar,
              f"display={display}")
        return card["id"]
    except Exception as e:
        check("8. Bar chart question", 2, False, f"exception: {e}")
        return None


def check_9_metabase_pie_chart(token: str, coll_id: int | None) -> int | None:
    """Check 'Stale vs Current' pie chart question exists."""
    try:
        if coll_id is None:
            check("9. Pie chart question", 2, False, "no collection found")
            return None
        items = metabase_get(f"collection/{coll_id}/items", token)
        card = None
        for item in items.get("data", items) if isinstance(items, dict) else items:
            if item.get("name") == "Stale vs Current" and item.get("model") == "card":
                card = item
                break
        if card is None:
            check("9. Pie chart question", 2, False, "question not found")
            return None
        card_detail = metabase_get(f"card/{card['id']}", token)
        display = card_detail.get("display", "")
        is_pie = display == "pie"
        check("9. Pie chart question", 2, is_pie,
              f"display={display}")
        return card["id"]
    except Exception as e:
        check("9. Pie chart question", 2, False, f"exception: {e}")
        return None


def check_10_metabase_dashboard(token: str, coll_id: int | None, bar_id: int | None, pie_id: int | None) -> None:
    """Check dashboard 'Frontend Dependency Health' exists with both cards."""
    try:
        if coll_id is None:
            check("10. Metabase dashboard with cards", 2, False, "no collection found")
            return
        items = metabase_get(f"collection/{coll_id}/items", token)
        dash = None
        for item in items.get("data", items) if isinstance(items, dict) else items:
            if item.get("name") == DASHBOARD_NAME and item.get("model") == "dashboard":
                dash = item
                break
        if dash is None:
            check("10. Metabase dashboard with cards", 2, False, "dashboard not found")
            return

        # Fetch dashboard detail to check cards
        dash_detail = metabase_get(f"dashboard/{dash['id']}", token)
        card_ids_on_dash = set()
        for dc in dash_detail.get("dashcards", dash_detail.get("ordered_cards", [])):
            cid = dc.get("card_id") or (dc.get("card") or {}).get("id")
            if cid:
                card_ids_on_dash.add(cid)

        has_bar = bar_id is not None and bar_id in card_ids_on_dash
        has_pie = pie_id is not None and pie_id in card_ids_on_dash
        check("10. Metabase dashboard with cards", 2, has_bar and has_pie,
              f"bar={'yes' if has_bar else 'no'}, pie={'yes' if has_pie else 'no'}, cards_on_dash={card_ids_on_dash}")
    except Exception as e:
        check("10. Metabase dashboard with cards", 2, False, f"exception: {e}")


def check_11_openproject_wp_exists() -> int | None:
    """Check work package exists with correct subject and Task type in Marketing Website project."""
    try:
        row = openproject_sql(
            f"SELECT wp.id FROM work_packages wp "
            f"JOIN projects p ON p.id = wp.project_id "
            f"JOIN types t ON t.id = wp.type_id "
            f"WHERE p.name = '{OP_PROJECT}' "
            f"AND wp.subject = '{WP_SUBJECT}' "
            f"AND t.name = 'Task'"
        )
        wp_id = int(row.split('\n')[0]) if row else None
        check("11. OpenProject work package exists", 1, wp_id is not None,
              f"wp_id={wp_id}" if wp_id else "not found")
        return wp_id
    except Exception as e:
        check("11. OpenProject work package exists", 1, False, f"exception: {e}")
        return None


def check_12_openproject_priority(wp_id: int | None) -> None:
    """Check work package has priority High."""
    try:
        if wp_id is None:
            check("12. Work package priority High", 1, False, "no work package found")
            return
        row = openproject_sql(
            f"SELECT e.name FROM enumerations e "
            f"JOIN work_packages wp ON wp.priority_id = e.id "
            f"WHERE wp.id = {wp_id}"
        )
        priority = row.strip() if row else ""
        check("12. Work package priority High", 1, priority.lower() == "high",
              f"priority={priority}")
    except Exception as e:
        check("12. Work package priority High", 1, False, f"exception: {e}")


def check_13_openproject_description(wp_id: int | None, ground_truth: dict[str, dict[str, str]]) -> None:
    """Check work package description lists all stale dependencies."""
    try:
        if wp_id is None:
            check("13. WP description lists stale deps", 2, False, "no work package found")
            return

        row = openproject_sql(
            f"SELECT j.data->>'description'->>'raw' FROM journals j "
            f"WHERE j.journable_id = {wp_id} AND j.journable_type = 'WorkPackage' "
            f"ORDER BY j.id DESC LIMIT 1"
        )
        # Fallback: read description directly from work_packages table
        # OpenProject stores description as a serialized object; let's try reading the raw text
        desc_raw = openproject_sql(
            f"SELECT wp.description FROM work_packages wp WHERE wp.id = {wp_id}"
        )
        description = desc_raw.strip() if desc_raw else ""

        # Compute expected stale deps
        stale_deps = []
        for project, deps in sorted(ground_truth.items()):
            for dep_name, version in sorted(deps.items()):
                if is_stale(dep_name, version):
                    stale_deps.append((project, dep_name, version))

        if not stale_deps:
            check("13. WP description lists stale deps", 2, True, "no stale deps expected")
            return

        missing = []
        for proj, dep, ver in stale_deps:
            # Check if the description contains the pattern "project / dep @ ver"
            # Be flexible with exact formatting
            if dep.lower() not in description.lower() or ver not in description:
                missing.append(f"{proj}/{dep}@{ver}")

        check("13. WP description lists stale deps", 2, not missing,
              f"all {len(stale_deps)} listed" if not missing else f"missing: {missing[:5]}")
    except Exception as e:
        check("13. WP description lists stale deps", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Step 1: Get ground truth from code-server
    ground_truth = check_1_package_json_files()

    # Steps 2-6: Baserow checks
    db_id = check_2_baserow_database_exists()
    table_id = check_3_baserow_table_exists(db_id)
    check_4_baserow_row_count(table_id, ground_truth)
    check_5_baserow_captured_at(table_id)
    check_6_baserow_stale_boolean(table_id, ground_truth)

    # Steps 7-10: Metabase checks
    try:
        token = metabase_session()
    except Exception as e:
        print(f"FATAL: cannot get Metabase session: {e}", file=sys.stderr)
        check("7. Metabase collection exists", 1, False, f"auth failed: {e}")
        check("8. Bar chart question", 2, False, "auth failed")
        check("9. Pie chart question", 2, False, "auth failed")
        check("10. Metabase dashboard with cards", 2, False, "auth failed")
        token = None

    if token:
        coll_id = check_7_metabase_collection(token)
        bar_id = check_8_metabase_bar_chart(token, coll_id)
        pie_id = check_9_metabase_pie_chart(token, coll_id)
        check_10_metabase_dashboard(token, coll_id, bar_id, pie_id)

    # Steps 11-13: OpenProject checks
    wp_id = check_11_openproject_wp_exists()
    check_12_openproject_priority(wp_id)
    check_13_openproject_description(wp_id, ground_truth)

    # Summary
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
