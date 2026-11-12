# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_021/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-021-I1: Build Engineering Hiring Pipeline Analytics
across Baserow, Metabase, and OpenProject.

Checks: 11 weighted checks (18 total points).
Strategy: Baserow API, Metabase API, OpenProject API.

Required env vars:
  SERVER_HOSTNAME, BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  METABASE_PORT, METABASE_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER.
"""

import os
import sys
import json
import subprocess
import requests
from datetime import date

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

REQUIRED_VARS = {
    "BASEROW_PORT": None,
    "BASEROW_CONTAINER": None,
    "METABASE_PORT": None,
    "METABASE_CONTAINER": None,
    "OPENPROJECT_PORT": None,
    "OPENPROJECT_CONTAINER": None,
}
for var in REQUIRED_VARS:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    REQUIRED_VARS[var] = val

BASEROW_PORT = REQUIRED_VARS["BASEROW_PORT"]
METABASE_PORT = REQUIRED_VARS["METABASE_PORT"]
OPENPROJECT_PORT = REQUIRED_VARS["OPENPROJECT_PORT"]

BASEROW_BASE = f"http://{HOST}:{BASEROW_PORT}"
METABASE_BASE = f"http://{HOST}:{METABASE_PORT}"
OPENPROJECT_BASE = f"http://{HOST}:{OPENPROJECT_PORT}"

# Login credentials
BASEROW_EMAIL = "admin@example.com"
BASEROW_PASSWORD = "Admin1234"
METABASE_EMAIL = "admin@metabase.local"
METABASE_PASSWORD = "mw-admin-123"
OPENPROJECT_USER = "admin"
OPENPROJECT_PASS = "AdminPass123!"

# Expected data
REPORT_DATE = date(2026, 3, 15)

POSITIONS_DATA = [
    {"pos_id": "POS-001", "role": "Senior Backend Engineer", "level": "Senior", "manager": "Lane Mahon", "team": "Platform", "opened": "2026-01-12", "approved": True},
    {"pos_id": "POS-002", "role": "Staff Frontend Engineer", "level": "Staff", "manager": "Elizabeth Cunningham", "team": "Frontend", "opened": "2026-01-20", "approved": True},
    {"pos_id": "POS-003", "role": "Mid Data Engineer", "level": "Mid", "manager": "Jane Dradder", "team": "Data", "opened": "2026-02-03", "approved": True},
    {"pos_id": "POS-004", "role": "Principal Security Engineer", "level": "Principal", "manager": "Latisha Mazon", "team": "Security", "opened": "2026-02-10", "approved": False},
    {"pos_id": "POS-005", "role": "Junior DevOps Engineer", "level": "Junior", "manager": "John Marshall", "team": "DevOps", "opened": "2026-02-18", "approved": True},
    {"pos_id": "POS-006", "role": "Senior Platform Engineer", "level": "Senior", "manager": "Lane Mahon", "team": "Platform", "opened": "2026-02-25", "approved": False},
]

CANDIDATES_DATA = [
    {"stage": "Onsite", "sourced": "2026-02-15", "days": 28, "offer": "None"},
    {"stage": "Offer", "sourced": "2026-02-01", "days": 42, "offer": "Pending"},
    {"stage": "Technical", "sourced": "2026-02-20", "days": 23, "offer": "None"},
    {"stage": "Screen", "sourced": "2026-03-01", "days": 14, "offer": "None"},
    {"stage": "Hired", "sourced": "2026-02-05", "days": 38, "offer": "Accepted"},
    {"stage": "Rejected", "sourced": "2026-02-10", "days": 33, "offer": "Declined"},
    {"stage": "Sourced", "sourced": "2026-03-05", "days": 10, "offer": "None"},
    {"stage": "Screen", "sourced": "2026-02-28", "days": 15, "offer": "None"},
    {"stage": "Technical", "sourced": "2026-02-22", "days": 21, "offer": "None"},
    {"stage": "Sourced", "sourced": "2026-03-08", "days": 7, "offer": "None"},
]

APPROVED_POSITIONS = [p for p in POSITIONS_DATA if p["approved"]]


# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Auth helpers ──────────────────────────────────────────────────────────────
_baserow_token = None
_metabase_session = None


def baserow_auth() -> str:
    global _baserow_token
    if _baserow_token:
        return _baserow_token
    r = requests.post(
        f"{BASEROW_BASE}/api/user/token-auth/",
        json={"email": BASEROW_EMAIL, "password": BASEROW_PASSWORD},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    # Baserow returns both 'token' and 'access_token'; use access_token with JWT auth
    _baserow_token = data.get("access_token", data.get("token", ""))
    return _baserow_token


def baserow_get(path: str) -> dict | list:
    token = baserow_auth()
    r = requests.get(
        f"{BASEROW_BASE}/api/{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def metabase_auth() -> str:
    global _metabase_session
    if _metabase_session:
        return _metabase_session
    r = requests.post(
        f"{METABASE_BASE}/api/session",
        json={"username": METABASE_EMAIL, "password": METABASE_PASSWORD},
        timeout=15,
    )
    r.raise_for_status()
    _metabase_session = r.json()["id"]
    return _metabase_session


def metabase_get(path: str) -> dict | list:
    session = metabase_auth()
    r = requests.get(
        f"{METABASE_BASE}/api/{path}",
        headers={"X-Metabase-Session": session},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def openproject_get(path: str) -> dict:
    """OpenProject API with Host header and basic auth."""
    host_header = f"{HOST}:{OPENPROJECT_PORT}"
    r = requests.get(
        f"{OPENPROJECT_BASE}/api/v3/{path}",
        auth=("apikey", OPENPROJECT_PASS),
        headers={"Host": host_header},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def docker_exec(container: str, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    r = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


def op_db_query(query: str) -> str:
    """Run a SQL query against the OpenProject embedded Postgres."""
    container = REQUIRED_VARS["OPENPROJECT_CONTAINER"]
    rc, out, err = docker_exec(
        container, "bash", "-c",
        f"PGPASSWORD=openproject psql -U openproject -d openproject -h 127.0.0.1 -t -A -c \"{query}\"",
    )
    if rc != 0:
        raise RuntimeError(f"op_db_query failed: {err.strip()}")
    return out.strip()


# ── Shared state (populated by early checks, used by later ones) ─────────────
_baserow_db_id = None
_positions_table_id = None
_candidates_table_id = None


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_baserow_database() -> None:
    """Database 'Engineering Hiring Pipeline' exists in Baserow."""
    global _baserow_db_id
    try:
        apps = baserow_get("applications/")
        for app in apps:
            if app.get("name") == "Engineering Hiring Pipeline" and app.get("type") == "database":
                _baserow_db_id = app["id"]
                break
        check("1. Baserow database exists", 1, _baserow_db_id is not None,
              f"db_id={_baserow_db_id}" if _baserow_db_id else "not found")
    except Exception as e:
        check("1. Baserow database exists", 1, False, f"exception: {e}")


def check_2_open_positions_table() -> None:
    """'Open Positions' table has 6 rows with correct role data."""
    global _positions_table_id
    try:
        if not _baserow_db_id:
            check("2. Open Positions table", 2, False, "no database found")
            return
        tables = baserow_get(f"database/tables/database/{_baserow_db_id}/")
        for t in tables:
            if t["name"] == "Open Positions":
                _positions_table_id = t["id"]
                break
        if not _positions_table_id:
            check("2. Open Positions table", 2, False, "table not found")
            return

        rows_resp = baserow_get(f"database/rows/table/{_positions_table_id}/?user_field_names=true&size=200")
        rows = rows_resp.get("results", [])
        if len(rows) != 6:
            check("2. Open Positions table", 2, False, f"expected 6 rows, got {len(rows)}")
            return

        expected_roles = {p["role"] for p in POSITIONS_DATA}
        found_roles = set()
        for row in rows:
            role = row.get("Role Title", "")
            found_roles.add(role)

        missing = expected_roles - found_roles
        check("2. Open Positions table", 2, not missing,
              f"6 rows, all roles present" if not missing else f"missing roles: {missing}")
    except Exception as e:
        check("2. Open Positions table", 2, False, f"exception: {e}")


def check_3_candidate_pipeline_table() -> None:
    """'Candidate Pipeline' table has 10 rows with correct stages."""
    global _candidates_table_id
    try:
        if not _baserow_db_id:
            check("3. Candidate Pipeline table", 2, False, "no database found")
            return
        tables = baserow_get(f"database/tables/database/{_baserow_db_id}/")
        for t in tables:
            if t["name"] == "Candidate Pipeline":
                _candidates_table_id = t["id"]
                break
        if not _candidates_table_id:
            check("3. Candidate Pipeline table", 2, False, "table not found")
            return

        rows_resp = baserow_get(f"database/rows/table/{_candidates_table_id}/?user_field_names=true&size=200")
        rows = rows_resp.get("results", [])
        if len(rows) != 10:
            check("3. Candidate Pipeline table", 2, False, f"expected 10 rows, got {len(rows)}")
            return

        expected_stages = sorted([c["stage"] for c in CANDIDATES_DATA])
        found_stages = sorted([
            (row.get("Stage") or {}).get("value", row.get("Stage", ""))
            if isinstance(row.get("Stage"), dict) else str(row.get("Stage", ""))
            for row in rows
        ])
        check("3. Candidate Pipeline table", 2, expected_stages == found_stages,
              f"10 rows, stages match" if expected_stages == found_stages
              else f"expected stages {expected_stages}, got {found_stages}")
    except Exception as e:
        check("3. Candidate Pipeline table", 2, False, f"exception: {e}")


def check_4_days_in_pipeline() -> None:
    """Days In Pipeline correctly computed relative to 2026-03-15."""
    try:
        if not _candidates_table_id:
            check("4. Days In Pipeline computed", 2, False, "no candidates table")
            return

        rows_resp = baserow_get(f"database/rows/table/{_candidates_table_id}/?user_field_names=true&size=200")
        rows = rows_resp.get("results", [])

        expected_days = sorted([c["days"] for c in CANDIDATES_DATA])
        found_days = []
        for row in rows:
            val = row.get("Days In Pipeline")
            if val is not None:
                try:
                    found_days.append(int(float(str(val))))
                except (ValueError, TypeError):
                    found_days.append(-1)
            else:
                found_days.append(-1)
        found_days.sort()

        check("4. Days In Pipeline computed", 2, expected_days == found_days,
              f"days match" if expected_days == found_days
              else f"expected {expected_days}, got {found_days}")
    except Exception as e:
        check("4. Days In Pipeline computed", 2, False, f"exception: {e}")


def check_5_kanban_view() -> None:
    """Kanban view 'Funnel' exists on Candidate Pipeline, stacked by Stage."""
    try:
        if not _candidates_table_id:
            check("5. Kanban view Funnel", 1, False, "no candidates table")
            return

        views = baserow_get(f"database/views/table/{_candidates_table_id}/")
        funnel = None
        for v in views:
            if v.get("name") == "Funnel" and v.get("type") == "kanban":
                funnel = v
                break

        check("5. Kanban view Funnel", 1, funnel is not None,
              "found kanban view" if funnel else "not found or not kanban type")
    except Exception as e:
        check("5. Kanban view Funnel", 1, False, f"exception: {e}")


def check_6_metabase_collection() -> None:
    """Metabase collection 'Hiring Analytics' exists."""
    try:
        collections = metabase_get("collection")
        found = any(c.get("name") == "Hiring Analytics" for c in collections)
        check("6. Metabase collection", 1, found,
              "found" if found else "collection 'Hiring Analytics' not found")
    except Exception as e:
        check("6. Metabase collection", 1, False, f"exception: {e}")


def _find_collection_id(name: str) -> int | None:
    collections = metabase_get("collection")
    for c in collections:
        if c.get("name") == name:
            return c["id"]
    return None


def check_7_metabase_questions() -> None:
    """Three saved questions exist with correct names in 'Hiring Analytics'."""
    try:
        coll_id = _find_collection_id("Hiring Analytics")
        if not coll_id:
            check("7. Metabase questions", 2, False, "collection not found")
            return

        items = metabase_get(f"collection/{coll_id}/items?models=card")
        card_names = {item["name"] for item in items.get("data", items) if item.get("model") == "card"}

        expected = {"Funnel by Stage", "Average Days-In-Pipeline by Team", "Open Positions by Level"}
        missing = expected - card_names
        check("7. Metabase questions", 2, not missing,
              "all 3 questions found" if not missing else f"missing: {missing}")
    except Exception as e:
        check("7. Metabase questions", 2, False, f"exception: {e}")


def check_8_metabase_dashboard_exists() -> None:
    """Dashboard 'Hiring Funnel Snapshot' exists with correct description."""
    try:
        coll_id = _find_collection_id("Hiring Analytics")
        if not coll_id:
            check("8. Metabase dashboard", 1, False, "collection not found")
            return

        items = metabase_get(f"collection/{coll_id}/items?models=dashboard")
        data = items.get("data", items) if isinstance(items, dict) else items
        dash = None
        for item in data:
            if item.get("name") == "Hiring Funnel Snapshot":
                dash = item
                break

        if not dash:
            check("8. Metabase dashboard", 1, False, "dashboard not found in collection")
            return

        dash_detail = metabase_get(f"dashboard/{dash['id']}")
        desc = dash_detail.get("description", "")
        expected_desc = "Hiring funnel snapshot as of 2026-03-15"
        check("8. Metabase dashboard", 1, desc == expected_desc,
              f"found, desc matches" if desc == expected_desc else f"desc={desc!r}")
    except Exception as e:
        check("8. Metabase dashboard", 1, False, f"exception: {e}")


def check_9_dashboard_cards() -> None:
    """Dashboard has all 3 questions as cards."""
    try:
        coll_id = _find_collection_id("Hiring Analytics")
        if not coll_id:
            check("9. Dashboard cards", 2, False, "collection not found")
            return

        items = metabase_get(f"collection/{coll_id}/items?models=dashboard")
        data = items.get("data", items) if isinstance(items, dict) else items
        dash_id = None
        for item in data:
            if item.get("name") == "Hiring Funnel Snapshot":
                dash_id = item["id"]
                break

        if not dash_id:
            check("9. Dashboard cards", 2, False, "dashboard not found")
            return

        dash_detail = metabase_get(f"dashboard/{dash_id}")
        cards = dash_detail.get("dashcards", dash_detail.get("ordered_cards", []))
        card_names = set()
        for card in cards:
            c = card.get("card", {})
            if c and c.get("name"):
                card_names.add(c["name"])

        expected = {"Funnel by Stage", "Average Days-In-Pipeline by Team", "Open Positions by Level"}
        missing = expected - card_names
        check("9. Dashboard cards", 2, not missing,
              f"all 3 cards present" if not missing else f"missing cards: {missing}, found: {card_names}")
    except Exception as e:
        check("9. Dashboard cards", 2, False, f"exception: {e}")


def check_10_openproject_work_packages() -> None:
    """4 Task work packages with correct subjects in 'Internal Tools' project."""
    try:
        # Find project ID
        proj_row = op_db_query("SELECT id FROM projects WHERE name = 'Internal Tools' LIMIT 1")
        if not proj_row:
            check("10. OpenProject work packages", 2, False, "project 'Internal Tools' not found")
            return
        proj_id = proj_row.strip()

        # type_id=1 is 'Task' (confirmed from schema)
        rows = op_db_query(
            f"SELECT subject FROM work_packages WHERE project_id = {proj_id} AND type_id = 1"
        )
        found_subjects = {line.strip() for line in rows.splitlines() if line.strip()}

        expected_subjects = {
            f"Recruit: {p['role']} ({p['level']})" for p in APPROVED_POSITIONS
        }

        missing = expected_subjects - found_subjects
        check("10. OpenProject work packages", 2, not missing,
              f"all 4 found" if not missing else f"missing: {missing}")
    except Exception as e:
        check("10. OpenProject work packages", 2, False, f"exception: {e}")


def check_11_wp_details() -> None:
    """Work packages have correct assignee (Donald Wright), priority (Normal), and descriptions."""
    try:
        proj_row = op_db_query("SELECT id FROM projects WHERE name = 'Internal Tools' LIMIT 1")
        if not proj_row:
            check("11. WP assignee & description", 2, False, "project not found")
            return
        proj_id = proj_row.strip()

        expected_map = {}
        for p in APPROVED_POSITIONS:
            subj = f"Recruit: {p['role']} ({p['level']})"
            desc = f"Team: {p['team']}; Hiring Manager: {p['manager']}; Opened: {p['opened']}"
            expected_map[subj] = desc

        # Query work packages with assignee and priority info
        # priority_id=8 is 'Normal', type_id=1 is 'Task'
        rows = op_db_query(
            f"SELECT wp.subject, wp.description, wp.priority_id, wp.assigned_to_id, "
            f"u.firstname, u.lastname "
            f"FROM work_packages wp "
            f"LEFT JOIN users u ON wp.assigned_to_id = u.id "
            f"WHERE wp.project_id = {proj_id} AND wp.type_id = 1"
        )

        issues = []
        matched = 0
        for line in rows.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 6:
                continue
            subj = parts[0].strip()
            desc_text = parts[1].strip()
            priority_id = parts[2].strip()
            firstname = parts[4].strip()
            lastname = parts[5].strip()

            if subj not in expected_map:
                continue
            matched += 1

            # Check assignee
            assignee_full = f"{firstname} {lastname}".strip()
            if assignee_full != "Donald Wright":
                issues.append(f"{subj}: assignee={assignee_full!r}")

            # Check priority (8 = Normal)
            if priority_id != "8":
                issues.append(f"{subj}: priority_id={priority_id} (expected 8/Normal)")

            # Check description
            expected_desc = expected_map[subj]
            if expected_desc not in desc_text:
                issues.append(f"{subj}: desc mismatch, got {desc_text!r}")

        if matched < len(expected_map):
            issues.append(f"only {matched}/{len(expected_map)} matching WPs found")

        check("11. WP assignee & description", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("11. WP assignee & description", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_database()
    check_2_open_positions_table()
    check_3_candidate_pipeline_table()
    check_4_days_in_pipeline()
    check_5_kanban_view()
    check_6_metabase_collection()
    check_7_metabase_questions()
    check_8_metabase_dashboard_exists()
    check_9_dashboard_cards()
    check_10_openproject_work_packages()
    check_11_wp_details()

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
