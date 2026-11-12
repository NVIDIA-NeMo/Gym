# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_008/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-008-I2: Engineering OKR Tracker across Baserow, Metabase, OpenProject

Checks: 12 weighted checks (19 total points) across baserow, metabase, openproject.
Strategy: Baserow API, Metabase API, OpenProject embedded DB.

Required env vars:
  SERVER_HOSTNAME, BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  METABASE_PORT, METABASE_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import sys
import json
import subprocess
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
METABASE_PORT = os.environ.get("METABASE_PORT")
METABASE_CONTAINER = os.environ.get("METABASE_CONTAINER")
OPENPROJECT_PORT = os.environ.get("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

_missing = []
for var in [
    "BASEROW_PORT", "BASEROW_CONTAINER", "BASEROW_DB_CONTAINER",
    "METABASE_PORT", "METABASE_CONTAINER",
    "OPENPROJECT_PORT", "OPENPROJECT_CONTAINER",
]:
    if not os.environ.get(var):
        _missing.append(var)
if _missing:
    print(f"FATAL: missing env vars: {', '.join(_missing)}", file=sys.stderr)
    sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"
METABASE_URL = f"http://{HOST}:{METABASE_PORT}"

# ── Expected data ─────────────────────────────────────────────────────────────
OBJECTIVES = [
    ("OBJ-1", "Scale observability coverage", "Priya Patel", "Q3-2026"),
    ("OBJ-2", "Improve checkout conversion", "Marco Rossi", "Q3-2026"),
    ("OBJ-3", "Harden data pipeline quality", "Elena Volkov", "Q3-2026"),
    ("OBJ-4", "Modernize mobile experience", "Jamal Harris", "Q3-2026"),
    ("OBJ-5", "Grow API gateway adoption", "Sophie Laurent", "Q3-2026"),
]

KEY_RESULTS = [
    ("KR-1", "OBJ-1", "Instrument 100% of production services with tracing", 100, 95),
    ("KR-2", "OBJ-1", "Reduce alert noise by 40%", 40, 10),
    ("KR-3", "OBJ-2", "Lift checkout conversion to 4.5%", 450, 420),
    ("KR-4", "OBJ-2", "Reduce cart abandonment to 55%", 55, 68),
    ("KR-5", "OBJ-3", "Achieve 99% data freshness SLA", 99, 97),
    ("KR-6", "OBJ-3", "Resolve 30 data quality incidents", 30, 12),
    ("KR-7", "OBJ-4", "Ship 15 redesigned mobile screens", 15, 14),
    ("KR-8", "OBJ-4", "Improve mobile crash-free rate to 99.8%", 998, 994),
    ("KR-9", "OBJ-5", "Migrate 25 services behind the API gateway", 25, 8),
    ("KR-10", "OBJ-5", "Onboard 10 external API consumers", 10, 2),
]

# Precompute expected Progress Pct and Status
EXPECTED_KR = {}
for kr_id, obj_id, desc, target, current in KEY_RESULTS:
    pct = round(current / target * 100, 1)
    if pct >= 75:
        status = "OnTrack"
    elif pct >= 40:
        status = "AtRisk"
    else:
        status = "OffTrack"
    EXPECTED_KR[kr_id] = {
        "obj_id": obj_id, "desc": desc, "target": target,
        "current": current, "pct": pct, "status": status,
    }

OFFTRACK_KRS = {k: v for k, v in EXPECTED_KR.items() if v["status"] == "OffTrack"}

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


def baserow_auth() -> dict:
    """Authenticate to Baserow API and return headers with token."""
    resp = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    return {"Authorization": f"JWT {token}"}


def metabase_auth() -> str:
    """Authenticate to Metabase and return session token."""
    resp = requests.post(
        f"{METABASE_URL}/api/session",
        json={"username": "admin@metabase.local", "password": "mw-admin-123"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["id"]


# ── Individual checks ─────────────────────────────────────────────────────────

# -- Baserow checks (API) --

_baserow_headers = None
_baserow_db_id = None
_obj_table_id = None
_kr_table_id = None
_obj_rows = None
_kr_rows = None


def _init_baserow():
    """Fetch Baserow state once: database, tables, rows."""
    global _baserow_headers, _baserow_db_id, _obj_table_id, _kr_table_id
    global _obj_rows, _kr_rows
    if _baserow_headers is not None:
        return
    _baserow_headers = baserow_auth()

    # Find database
    resp = requests.get(
        f"{BASEROW_URL}/api/applications/",
        headers=_baserow_headers, timeout=15,
    )
    resp.raise_for_status()
    apps = resp.json()
    for app in apps:
        if app.get("name") == "Engineering OKRs Q3 2026":
            _baserow_db_id = app["id"]
            break

    if _baserow_db_id is None:
        return

    # Find tables
    resp = requests.get(
        f"{BASEROW_URL}/api/database/tables/database/{_baserow_db_id}/",
        headers=_baserow_headers, timeout=15,
    )
    resp.raise_for_status()
    tables = resp.json()
    for t in tables:
        if t["name"] == "Objectives":
            _obj_table_id = t["id"]
        elif t["name"] == "Key Results":
            _kr_table_id = t["id"]

    # Fetch rows
    if _obj_table_id:
        resp = requests.get(
            f"{BASEROW_URL}/api/database/rows/table/{_obj_table_id}/"
            f"?user_field_names=true&size=100",
            headers=_baserow_headers, timeout=15,
        )
        resp.raise_for_status()
        _obj_rows = resp.json().get("results", [])

    if _kr_table_id:
        resp = requests.get(
            f"{BASEROW_URL}/api/database/rows/table/{_kr_table_id}/"
            f"?user_field_names=true&size=100",
            headers=_baserow_headers, timeout=15,
        )
        resp.raise_for_status()
        _kr_rows = resp.json().get("results", [])


def check_1_baserow_db_and_tables() -> None:
    """Database 'Engineering OKRs Q3 2026' exists with Objectives and Key Results tables."""
    try:
        _init_baserow()
        has_db = _baserow_db_id is not None
        has_obj = _obj_table_id is not None
        has_kr = _kr_table_id is not None
        ok = has_db and has_obj and has_kr
        detail = f"db={has_db}, objectives_tbl={has_obj}, key_results_tbl={has_kr}"
        check("1. Baserow DB and tables exist", 1, ok, detail)
    except Exception as e:
        check("1. Baserow DB and tables exist", 1, False, f"exception: {e}")


def check_2_objectives_data() -> None:
    """Objectives table has 5 rows with correct OBJ IDs, Titles, Owners, Quarter."""
    try:
        _init_baserow()
        if not _obj_rows:
            check("2. Objectives data (5 rows)", 2, False, "no rows found")
            return

        if len(_obj_rows) != 5:
            check("2. Objectives data (5 rows)", 2, False,
                  f"expected 5 rows, got {len(_obj_rows)}")
            return

        # Build lookup by Objective ID
        found = {}
        for row in _obj_rows:
            obj_id = str(row.get("Objective ID", "")).strip()
            found[obj_id] = row

        mismatches = []
        for obj_id, title, owner, quarter in OBJECTIVES:
            if obj_id not in found:
                mismatches.append(f"{obj_id} missing")
                continue
            r = found[obj_id]
            if str(r.get("Title", "")).strip() != title:
                mismatches.append(f"{obj_id} title mismatch")
            if str(r.get("Owner", "")).strip() != owner:
                mismatches.append(f"{obj_id} owner mismatch")
            if str(r.get("Quarter", "")).strip() != quarter:
                mismatches.append(f"{obj_id} quarter mismatch")

        ok = len(mismatches) == 0
        detail = "; ".join(mismatches) if mismatches else "all 5 match"
        check("2. Objectives data (5 rows)", 2, ok, detail)
    except Exception as e:
        check("2. Objectives data (5 rows)", 2, False, f"exception: {e}")


def check_3_kr_rows_and_ids() -> None:
    """Key Results table has 10 rows with correct KR IDs and descriptions."""
    try:
        _init_baserow()
        if not _kr_rows:
            check("3. Key Results rows and IDs", 2, False, "no rows found")
            return

        if len(_kr_rows) != 10:
            check("3. Key Results rows and IDs", 2, False,
                  f"expected 10 rows, got {len(_kr_rows)}")
            return

        found = {}
        for row in _kr_rows:
            kr_id = str(row.get("KR ID", "")).strip()
            found[kr_id] = row

        mismatches = []
        for kr_id, _, desc, _, _ in KEY_RESULTS:
            if kr_id not in found:
                mismatches.append(f"{kr_id} missing")
                continue
            actual_desc = str(found[kr_id].get("Description", "")).strip()
            if actual_desc != desc:
                mismatches.append(f"{kr_id} desc mismatch")

        ok = len(mismatches) == 0
        detail = "; ".join(mismatches) if mismatches else "all 10 match"
        check("3. Key Results rows and IDs", 2, ok, detail)
    except Exception as e:
        check("3. Key Results rows and IDs", 2, False, f"exception: {e}")


def check_4_kr_target_current() -> None:
    """Key Results Target and Current values match expected."""
    try:
        _init_baserow()
        if not _kr_rows:
            check("4. Key Results Target/Current values", 2, False, "no rows")
            return

        found = {}
        for row in _kr_rows:
            kr_id = str(row.get("KR ID", "")).strip()
            found[kr_id] = row

        mismatches = []
        for kr_id in EXPECTED_KR:
            exp = EXPECTED_KR[kr_id]
            if kr_id not in found:
                mismatches.append(f"{kr_id} missing")
                continue
            r = found[kr_id]
            # Target and Current may be int or string
            try:
                actual_target = float(r.get("Target", 0))
            except (ValueError, TypeError):
                actual_target = None
            try:
                actual_current = float(r.get("Current", 0))
            except (ValueError, TypeError):
                actual_current = None

            if actual_target != float(exp["target"]):
                mismatches.append(
                    f"{kr_id} target: expected {exp['target']}, got {actual_target}")
            if actual_current != float(exp["current"]):
                mismatches.append(
                    f"{kr_id} current: expected {exp['current']}, got {actual_current}")

        ok = len(mismatches) == 0
        detail = "; ".join(mismatches) if mismatches else "all match"
        check("4. Key Results Target/Current values", 2, ok, detail)
    except Exception as e:
        check("4. Key Results Target/Current values", 2, False, f"exception: {e}")


def check_5_kr_progress_pct() -> None:
    """Key Results Progress Pct computed correctly as round(Current/Target*100, 1)."""
    try:
        _init_baserow()
        if not _kr_rows:
            check("5. Key Results Progress Pct", 2, False, "no rows")
            return

        found = {}
        for row in _kr_rows:
            kr_id = str(row.get("KR ID", "")).strip()
            found[kr_id] = row

        mismatches = []
        for kr_id, exp in EXPECTED_KR.items():
            if kr_id not in found:
                mismatches.append(f"{kr_id} missing")
                continue
            try:
                actual_pct = float(found[kr_id].get("Progress Pct", -1))
            except (ValueError, TypeError):
                actual_pct = None
            if actual_pct is None or abs(actual_pct - exp["pct"]) > 0.15:
                mismatches.append(
                    f"{kr_id}: expected {exp['pct']}, got {actual_pct}")

        ok = len(mismatches) == 0
        detail = "; ".join(mismatches) if mismatches else "all correct"
        check("5. Key Results Progress Pct", 2, ok, detail)
    except Exception as e:
        check("5. Key Results Progress Pct", 2, False, f"exception: {e}")


def check_6_kr_status() -> None:
    """Key Results Status assigned correctly (OnTrack/AtRisk/OffTrack)."""
    try:
        _init_baserow()
        if not _kr_rows:
            check("6. Key Results Status values", 1, False, "no rows")
            return

        found = {}
        for row in _kr_rows:
            kr_id = str(row.get("KR ID", "")).strip()
            found[kr_id] = row

        mismatches = []
        for kr_id, exp in EXPECTED_KR.items():
            if kr_id not in found:
                mismatches.append(f"{kr_id} missing")
                continue
            # Status may be a dict (single-select) or string
            raw_status = found[kr_id].get("Status", "")
            if isinstance(raw_status, dict):
                actual_status = raw_status.get("value", "")
            else:
                actual_status = str(raw_status).strip()
            if actual_status != exp["status"]:
                mismatches.append(
                    f"{kr_id}: expected {exp['status']}, got {actual_status}")

        ok = len(mismatches) == 0
        detail = "; ".join(mismatches) if mismatches else "all correct"
        check("6. Key Results Status values", 1, ok, detail)
    except Exception as e:
        check("6. Key Results Status values", 1, False, f"exception: {e}")


# -- Metabase checks (API) --

def check_7_metabase_collection() -> None:
    """Metabase collection 'Engineering OKRs Q3 2026' exists."""
    try:
        session_id = metabase_auth()
        headers = {"X-Metabase-Session": session_id}
        resp = requests.get(
            f"{METABASE_URL}/api/collection", headers=headers, timeout=15)
        resp.raise_for_status()
        collections = resp.json()
        found = any(
            c.get("name") == "Engineering OKRs Q3 2026" for c in collections
        )
        check("7. Metabase collection exists", 1, found,
              "found" if found else "not found")
    except Exception as e:
        check("7. Metabase collection exists", 1, False, f"exception: {e}")


def check_8_metabase_questions() -> None:
    """Three questions exist: KR Status Breakdown, Average Progress by Objective, Off-Track KRs."""
    try:
        session_id = metabase_auth()
        headers = {"X-Metabase-Session": session_id}

        # Find collection ID
        resp = requests.get(
            f"{METABASE_URL}/api/collection", headers=headers, timeout=15)
        resp.raise_for_status()
        coll_id = None
        for c in resp.json():
            if c.get("name") == "Engineering OKRs Q3 2026":
                coll_id = c["id"]
                break

        if coll_id is None:
            check("8. Metabase questions (3)", 2, False, "collection not found")
            return

        # Get items in collection
        resp = requests.get(
            f"{METABASE_URL}/api/collection/{coll_id}/items",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("data", resp.json()) if isinstance(resp.json(), dict) else resp.json()
        if isinstance(items, dict):
            items = items.get("data", [])

        expected_names = {
            "KR Status Breakdown",
            "Average Progress by Objective",
            "Off-Track KRs",
        }
        found_names = set()
        for item in items:
            name = item.get("name", "")
            if name in expected_names:
                found_names.add(name)

        missing = expected_names - found_names
        ok = len(missing) == 0
        detail = f"missing: {missing}" if missing else "all 3 found"
        check("8. Metabase questions (3)", 2, ok, detail)
    except Exception as e:
        check("8. Metabase questions (3)", 2, False, f"exception: {e}")


def check_9_metabase_dashboard() -> None:
    """Dashboard 'Engineering OKRs Q3 2026 Dashboard' exists with 3 cards."""
    try:
        session_id = metabase_auth()
        headers = {"X-Metabase-Session": session_id}

        # Search for dashboard
        resp = requests.get(
            f"{METABASE_URL}/api/search",
            params={"q": "Engineering OKRs Q3 2026 Dashboard", "models": "dashboard"},
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("data", [])

        dash_id = None
        for r in results:
            if r.get("name") == "Engineering OKRs Q3 2026 Dashboard":
                dash_id = r["id"]
                break

        if dash_id is None:
            check("9. Metabase dashboard with 3 cards", 2, False,
                  "dashboard not found")
            return

        # Get dashboard details
        resp = requests.get(
            f"{METABASE_URL}/api/dashboard/{dash_id}",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        dash = resp.json()

        # Count question cards (exclude text/heading cards)
        cards = [
            c for c in dash.get("dashcards", dash.get("ordered_cards", []))
            if c.get("card_id") or c.get("card", {}).get("id")
        ]
        num_cards = len(cards)
        ok = num_cards >= 3
        check("9. Metabase dashboard with 3 cards", 2, ok,
              f"{num_cards} question card(s)")
    except Exception as e:
        check("9. Metabase dashboard with 3 cards", 2, False, f"exception: {e}")


# -- OpenProject checks (embedded DB) --

def _op_query(sql: str) -> str:
    """Run a SQL query against OpenProject's embedded Postgres."""
    r = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=openproject",
         OPENPROJECT_CONTAINER,
         "psql", "-h", "127.0.0.1", "-U", "openproject", "-d", "openproject",
         "-t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"psql failed (rc={r.returncode}): {r.stderr.strip()}")
    return r.stdout.strip()


def check_10_op_recover_kr_tasks() -> None:
    """OpenProject 'API Gateway' has 3 Task WPs for OffTrack KRs with correct subjects."""
    try:
        # Get project ID
        project_id = _op_query(
            "SELECT id FROM projects WHERE name = 'API Gateway' LIMIT 1"
        )
        if not project_id:
            check("10. OpenProject Recover KR tasks (3)", 2, False,
                  "project 'API Gateway' not found")
            return

        # Get type ID for 'Task'
        task_type_id = _op_query(
            "SELECT id FROM types WHERE name = 'Task' LIMIT 1"
        )

        # Find work packages with subject matching 'Recover KR: KR-*'
        type_clause = f"AND type_id = {task_type_id}" if task_type_id else ""
        rows = _op_query(
            f"SELECT subject FROM work_packages "
            f"WHERE project_id = {project_id} {type_clause} "
            f"AND subject LIKE 'Recover KR:%'"
        )
        found_subjects = set(rows.splitlines()) if rows else set()

        expected_subjects = {f"Recover KR: {kr_id}" for kr_id in OFFTRACK_KRS}
        missing = expected_subjects - found_subjects
        extra = found_subjects - expected_subjects

        ok = missing == set() and len(found_subjects) >= len(expected_subjects)
        detail_parts = []
        if missing:
            detail_parts.append(f"missing: {missing}")
        if extra:
            detail_parts.append(f"extra: {extra}")
        if not detail_parts:
            detail_parts.append(f"all {len(expected_subjects)} found")
        check("10. OpenProject Recover KR tasks (3)", 2, ok,
              "; ".join(detail_parts))
    except Exception as e:
        check("10. OpenProject Recover KR tasks (3)", 2, False,
              f"exception: {e}")


def check_11_op_priority_high() -> None:
    """Recover KR work packages have High priority."""
    try:
        project_id = _op_query(
            "SELECT id FROM projects WHERE name = 'API Gateway' LIMIT 1"
        )
        if not project_id:
            check("11. OpenProject WP priority High", 1, False,
                  "project not found")
            return

        rows = _op_query(
            f"SELECT wp.subject, p.name AS priority "
            f"FROM work_packages wp "
            f"JOIN enumerations p ON wp.priority_id = p.id "
            f"WHERE wp.project_id = {project_id} "
            f"AND wp.subject LIKE 'Recover KR:%'"
        )
        if not rows:
            check("11. OpenProject WP priority High", 1, False, "no WPs found")
            return

        non_high = []
        for line in rows.splitlines():
            parts = line.split("|")
            if len(parts) == 2:
                subj, pri = parts[0].strip(), parts[1].strip()
                if pri != "High":
                    non_high.append(f"{subj} has priority {pri}")

        ok = len(non_high) == 0
        detail = "; ".join(non_high) if non_high else "all High"
        check("11. OpenProject WP priority High", 1, ok, detail)
    except Exception as e:
        check("11. OpenProject WP priority High", 1, False, f"exception: {e}")


def check_12_op_description() -> None:
    """Recover KR work packages have description with Target, Current, Progress Pct."""
    try:
        project_id = _op_query(
            "SELECT id FROM projects WHERE name = 'API Gateway' LIMIT 1"
        )
        if not project_id:
            check("12. OpenProject WP descriptions", 1, False,
                  "project not found")
            return

        rows = _op_query(
            f"SELECT subject || '|||' || COALESCE(description, '') "
            f"FROM work_packages "
            f"WHERE project_id = {project_id} "
            f"AND subject LIKE 'Recover KR:%'"
        )
        if not rows:
            check("12. OpenProject WP descriptions", 1, False, "no WPs found")
            return

        mismatches = []
        for line in rows.splitlines():
            if "|||" not in line:
                continue
            subj, desc = line.split("|||", 1)
            subj = subj.strip()
            # Extract KR ID from subject
            kr_id = subj.replace("Recover KR:", "").strip()
            if kr_id not in EXPECTED_KR:
                continue
            exp = EXPECTED_KR[kr_id]
            # Check description contains Target, Current, Progress Pct
            if str(exp["target"]) not in desc:
                mismatches.append(f"{kr_id}: Target {exp['target']} missing in desc")
            elif str(exp["current"]) not in desc:
                mismatches.append(f"{kr_id}: Current {exp['current']} missing in desc")
            elif str(exp["pct"]) not in desc:
                mismatches.append(f"{kr_id}: Progress {exp['pct']}% missing in desc")

        ok = len(mismatches) == 0
        detail = "; ".join(mismatches) if mismatches else "all correct"
        check("12. OpenProject WP descriptions", 1, ok, detail)
    except Exception as e:
        check("12. OpenProject WP descriptions", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_db_and_tables()
    check_2_objectives_data()
    check_3_kr_rows_and_ids()
    check_4_kr_target_current()
    check_5_kr_progress_pct()
    check_6_kr_status()
    check_7_metabase_collection()
    check_8_metabase_questions()
    check_9_metabase_dashboard()
    check_10_op_recover_kr_tasks()
    check_11_op_priority_high()
    check_12_op_description()

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
