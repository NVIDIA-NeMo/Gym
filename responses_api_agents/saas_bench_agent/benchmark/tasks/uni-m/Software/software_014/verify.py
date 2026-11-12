# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_014/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-014-I1: Inventory and Govern Feature Flags Across todo-api and blog-engine

Checks: 15 weighted checks across code-server, baserow, metabase, openproject.
Strategy: docker exec DB for Baserow structure + OpenProject, Baserow API for row data,
          Metabase API for collection/questions/dashboard, docker exec for code-server.

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  METABASE_PORT, METABASE_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import sys
import subprocess
import json
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_PORT = os.environ.get("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")

BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")

METABASE_PORT = os.environ.get("METABASE_PORT")
METABASE_CONTAINER = os.environ.get("METABASE_CONTAINER")

OPENPROJECT_PORT = os.environ.get("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

_required = {
    "CODE_SERVER_PORT": CODE_SERVER_PORT,
    "CODE_SERVER_CONTAINER": CODE_SERVER_CONTAINER,
    "BASEROW_PORT": BASEROW_PORT,
    "BASEROW_CONTAINER": BASEROW_CONTAINER,
    "BASEROW_DB_CONTAINER": BASEROW_DB_CONTAINER,
    "METABASE_PORT": METABASE_PORT,
    "METABASE_CONTAINER": METABASE_CONTAINER,
    "OPENPROJECT_PORT": OPENPROJECT_PORT,
    "OPENPROJECT_CONTAINER": OPENPROJECT_CONTAINER,
}
for _var, _val in _required.items():
    if not _val:
        print(f"FATAL: {_var} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"
METABASE_URL = f"http://{HOST}:{METABASE_PORT}"
OPENPROJECT_URL = f"http://{HOST}:{OPENPROJECT_PORT}"

# ── Expected data ─────────────────────────────────────────────────────────────
FLAG_METADATA = {
    "NEW_CHECKOUT": {"default_state": "Enabled", "owner": "alice@example.com", "target_removal_date": "2026-05-15"},
    "DARK_MODE": {"default_state": "Enabled", "owner": "bob@example.com", "target_removal_date": "2026-08-30"},
    "LEGACY_AUTH": {"default_state": "Disabled", "owner": "carol@example.com", "target_removal_date": "2026-04-10"},
    "BETA_COMMENTS": {"default_state": "Disabled", "owner": "dave@example.com", "target_removal_date": "2026-12-01"},
    "EXPERIMENTAL_SEARCH": {"default_state": "Enabled", "owner": "eve@example.com", "target_removal_date": "2026-06-20"},
}
SUNSET_CUTOFF = "2026-07-01"
SUNSET_FLAGS = {k for k, v in FLAG_METADATA.items() if v["target_removal_date"] <= SUNSET_CUTOFF}
# Expected: NEW_CHECKOUT, LEGACY_AUTH, EXPERIMENTAL_SEARCH

EXPECTED_FIELDS = [
    "Flag Name", "Project", "File Path", "Line Number",
    "Default State", "Owner", "Target Removal Date", "Status",
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


def baserow_db_query(sql: str) -> str:
    rc, out, err = docker_exec(
        BASEROW_DB_CONTAINER,
        "env", "PGPASSWORD=kdpzkuyhsgb22onku8y7rxkx3czej88nxpngaz4mlmgad67vpc", "psql", "-h", "127.0.0.1", "-U", "baserow", "-d", "baserow", "-t", "-A", "-c", sql,
    )
    return out.strip()


def openproject_db_query(sql: str) -> str:
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "psql", "-U", "openproject", "-d", "openproject", "-t", "-A", "-c", sql,
    )
    return out.strip()


def baserow_api_auth() -> tuple[str, dict]:
    """Get Baserow JWT token and return (token, headers)."""
    resp = requests.post(f"{BASEROW_URL}/api/user/token-auth/", json={
        "email": "admin@example.com", "password": "Admin1234",
    }, timeout=10)
    resp.raise_for_status()
    token = resp.json()["token"]
    return token, {"Authorization": f"JWT {token}"}


def metabase_api_auth() -> tuple[str, dict]:
    """Get Metabase session token and return (session_id, headers)."""
    resp = requests.post(f"{METABASE_URL}/api/session", json={
        "username": "admin@metabase.local", "password": "mw-admin-123",
    }, timeout=10)
    resp.raise_for_status()
    sid = resp.json()["id"]
    return sid, {"X-Metabase-Session": sid}


def _get_baserow_table_id() -> str:
    """Return the Baserow table ID for 'Feature Flags' or empty string."""
    return baserow_db_query(
        "SELECT dt.id FROM database_table dt "
        "JOIN core_application ca ON dt.database_id = ca.id "
        "WHERE ca.name = 'Feature Flag Governance' AND dt.name = 'Feature Flags' LIMIT 1;"
    ).strip()


def _get_baserow_rows_and_field_map() -> tuple[list, dict]:
    """Fetch all rows and build a field_name -> field_key map via Baserow API."""
    token, headers = baserow_api_auth()
    table_id = _get_baserow_table_id()
    if not table_id:
        return [], {}
    resp = requests.get(
        f"{BASEROW_URL}/api/database/rows/table/{table_id}/?size=200",
        headers=headers, timeout=15,
    )
    resp.raise_for_status()
    rows = resp.json().get("results", [])
    fields_resp = requests.get(
        f"{BASEROW_URL}/api/database/fields/table/{table_id}/",
        headers=headers, timeout=15,
    )
    fields_resp.raise_for_status()
    field_map = {f["name"]: f"field_{f['id']}" for f in fields_resp.json()}
    return rows, field_map


def _select_value(val) -> str:
    """Extract the display value from a Baserow single-select cell."""
    if isinstance(val, dict):
        return val.get("value", "")
    if val is None:
        return ""
    return str(val)


def _get_metabase_collection_id(headers: dict) -> int | None:
    resp = requests.get(f"{METABASE_URL}/api/collection", headers=headers, timeout=15)
    resp.raise_for_status()
    for c in resp.json():
        if c.get("name") == "Feature Flag Governance Q2 2026":
            return c["id"]
    return None


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_code_server_flag_files():
    """Verify that feature flag patterns exist on code-server filesystem."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER, "bash", "-c",
            "grep -r -E 'FEATURE_FLAG_[A-Z0-9_]+\\s*=' /home/coder/project/todo-api/ "
            "/home/coder/project/blog-engine/ 2>/dev/null || true",
            timeout=20,
        )
        lines = [l for l in out.strip().split("\n") if l.strip()]
        flag_names = set()
        import re
        for line in lines:
            m = re.search(r"FEATURE_FLAG_([A-Z0-9_]+)\s*=", line)
            if m:
                flag_names.add(m.group(1))
        passed = len(flag_names) >= 3  # at least some flags found
        check("1. Feature flag patterns on code-server", 1, passed,
              f"found {len(flag_names)} flags: {sorted(flag_names)}" if flag_names else "no flags found")
    except Exception as e:
        check("1. Feature flag patterns on code-server", 1, False, f"exception: {e}")


def check_2_baserow_database_exists():
    """Baserow database 'Feature Flag Governance' exists."""
    try:
        result = baserow_db_query(
            "SELECT ca.id FROM core_application ca "
            "JOIN django_content_type ct ON ca.content_type_id = ct.id "
            "WHERE ca.name = 'Feature Flag Governance' "
            "AND ct.app_label = 'database' AND ct.model = 'database' LIMIT 1;"
        )
        passed = bool(result.strip())
        check("2. Baserow DB 'Feature Flag Governance' exists", 1, passed,
              f"id={result}" if passed else "database not found")
    except Exception as e:
        check("2. Baserow DB 'Feature Flag Governance' exists", 1, False, f"exception: {e}")


def check_3_baserow_table_exists():
    """Baserow table 'Feature Flags' exists."""
    try:
        table_id = _get_baserow_table_id()
        passed = bool(table_id)
        check("3. Table 'Feature Flags' exists", 1, passed,
              f"table_id={table_id}" if passed else "table not found")
    except Exception as e:
        check("3. Table 'Feature Flags' exists", 1, False, f"exception: {e}")


def check_4_baserow_fields():
    """Table has all required fields."""
    try:
        result = baserow_db_query(
            "SELECT df.name FROM database_field df "
            "JOIN database_table dt ON df.table_id = dt.id "
            "JOIN core_application ca ON dt.database_id = ca.id "
            "WHERE ca.name = 'Feature Flag Governance' AND dt.name = 'Feature Flags' "
            "ORDER BY df.name;"
        )
        found_fields = [f.strip() for f in result.split("\n") if f.strip()]
        missing = [f for f in EXPECTED_FIELDS if f not in found_fields]
        passed = len(missing) == 0
        check("4. Table has required fields", 2, passed,
              f"missing: {missing}" if not passed else f"all {len(EXPECTED_FIELDS)} fields present")
    except Exception as e:
        check("4. Table has required fields", 2, False, f"exception: {e}")


def check_5_baserow_flags_present():
    """All 5 expected feature flags present as rows."""
    try:
        rows, field_map = _get_baserow_rows_and_field_map()
        fn_key = field_map.get("Flag Name")
        if not fn_key:
            check("5. All 5 flags present as rows", 2, False, "Flag Name field not found")
            return
        found_flags = set()
        for row in rows:
            val = row.get(fn_key, "")
            if val:
                found_flags.add(str(val).strip())
        expected = set(FLAG_METADATA.keys())
        missing = expected - found_flags
        passed = len(missing) == 0
        detail = f"found {len(found_flags)} flags"
        if missing:
            detail += f", missing: {sorted(missing)}"
        check("5. All 5 flags present as rows", 2, passed, detail)
    except Exception as e:
        check("5. All 5 flags present as rows", 2, False, f"exception: {e}")


def check_6_flag_metadata_correct():
    """Default State, Owner, Target Removal Date match expected values."""
    try:
        rows, field_map = _get_baserow_rows_and_field_map()
        fn_key = field_map.get("Flag Name")
        ds_key = field_map.get("Default State")
        owner_key = field_map.get("Owner")
        trd_key = field_map.get("Target Removal Date")
        if not all([fn_key, ds_key, owner_key, trd_key]):
            check("6. Flag metadata correct", 2, False,
                  f"missing field keys: fn={fn_key}, ds={ds_key}, owner={owner_key}, trd={trd_key}")
            return
        errors = []
        for row in rows:
            flag_name = str(row.get(fn_key, "")).strip()
            if flag_name not in FLAG_METADATA:
                continue
            exp = FLAG_METADATA[flag_name]
            ds_val = _select_value(row.get(ds_key))
            if ds_val != exp["default_state"]:
                errors.append(f"{flag_name}: Default State='{ds_val}' expected '{exp['default_state']}'")
            owner_val = str(row.get(owner_key, "")).strip()
            if owner_val != exp["owner"]:
                errors.append(f"{flag_name}: Owner='{owner_val}' expected '{exp['owner']}'")
            trd_val = str(row.get(trd_key, "")).strip()[:10]
            if trd_val != exp["target_removal_date"]:
                errors.append(f"{flag_name}: TRD='{trd_val}' expected '{exp['target_removal_date']}'")
        passed = len(errors) == 0
        check("6. Flag metadata correct", 2, passed,
              "; ".join(errors) if errors else "all metadata matches")
    except Exception as e:
        check("6. Flag metadata correct", 2, False, f"exception: {e}")


def check_7_status_correct():
    """Status field: Sunset if date <= 2026-07-01, else Active."""
    try:
        rows, field_map = _get_baserow_rows_and_field_map()
        fn_key = field_map.get("Flag Name")
        status_key = field_map.get("Status")
        if not all([fn_key, status_key]):
            check("7. Status correctly computed", 2, False, "missing fields")
            return
        errors = []
        for row in rows:
            flag_name = str(row.get(fn_key, "")).strip()
            if flag_name not in FLAG_METADATA:
                continue
            expected_status = "Sunset" if flag_name in SUNSET_FLAGS else "Active"
            status_val = _select_value(row.get(status_key))
            if status_val != expected_status:
                errors.append(f"{flag_name}: status='{status_val}' expected '{expected_status}'")
        passed = len(errors) == 0
        check("7. Status correctly computed", 2, passed,
              "; ".join(errors) if errors else "all statuses correct")
    except Exception as e:
        check("7. Status correctly computed", 2, False, f"exception: {e}")


def check_8_baserow_kanban_view():
    """Kanban view exists on Feature Flags table."""
    try:
        result = baserow_db_query(
            "SELECT dv.id FROM database_view dv "
            "JOIN database_table dt ON dv.table_id = dt.id "
            "JOIN core_application ca ON dt.database_id = ca.id "
            "WHERE ca.name = 'Feature Flag Governance' AND dt.name = 'Feature Flags' "
            "AND dv.type = 'kanban' LIMIT 1;"
        )
        passed = bool(result.strip())
        check("8. Kanban view on Feature Flags table", 1, passed,
              f"view_id={result}" if passed else "no kanban view found")
    except Exception as e:
        check("8. Kanban view on Feature Flags table", 1, False, f"exception: {e}")


def check_9_metabase_collection():
    """Metabase collection 'Feature Flag Governance Q2 2026' exists."""
    try:
        _, headers = metabase_api_auth()
        coll_id = _get_metabase_collection_id(headers)
        passed = coll_id is not None
        check("9. Metabase collection exists", 1, passed,
              f"id={coll_id}" if passed else "collection not found")
    except Exception as e:
        check("9. Metabase collection exists", 1, False, f"exception: {e}")


def check_10_metabase_question_bar_chart():
    """Question 'Flags by Project and State' exists as bar chart in collection."""
    try:
        _, headers = metabase_api_auth()
        coll_id = _get_metabase_collection_id(headers)
        if coll_id is None:
            check("10. Question 'Flags by Project and State'", 2, False, "collection not found")
            return
        resp = requests.get(f"{METABASE_URL}/api/card", headers=headers, timeout=15)
        resp.raise_for_status()
        cards = resp.json()
        found = [c for c in cards
                 if c.get("name") == "Flags by Project and State"
                 and c.get("collection_id") == coll_id]
        passed = len(found) > 0
        detail = ""
        if passed:
            display = found[0].get("display", "unknown")
            detail = f"display={display}"
            if display != "bar":
                detail += " (expected bar chart)"
        else:
            detail = "question not found in collection"
        check("10. Question 'Flags by Project and State'", 2, passed, detail)
    except Exception as e:
        check("10. Question 'Flags by Project and State'", 2, False, f"exception: {e}")


def check_11_metabase_question_sunset():
    """Question 'Sunset Flag Schedule' exists as table in collection."""
    try:
        _, headers = metabase_api_auth()
        coll_id = _get_metabase_collection_id(headers)
        if coll_id is None:
            check("11. Question 'Sunset Flag Schedule'", 2, False, "collection not found")
            return
        resp = requests.get(f"{METABASE_URL}/api/card", headers=headers, timeout=15)
        resp.raise_for_status()
        cards = resp.json()
        found = [c for c in cards
                 if c.get("name") == "Sunset Flag Schedule"
                 and c.get("collection_id") == coll_id]
        passed = len(found) > 0
        detail = ""
        if passed:
            display = found[0].get("display", "unknown")
            detail = f"display={display}"
        else:
            detail = "question not found in collection"
        check("11. Question 'Sunset Flag Schedule'", 2, passed, detail)
    except Exception as e:
        check("11. Question 'Sunset Flag Schedule'", 2, False, f"exception: {e}")


def check_12_metabase_dashboard_exists():
    """Dashboard 'Feature Flag Sunset Dashboard' exists in collection."""
    try:
        _, headers = metabase_api_auth()
        coll_id = _get_metabase_collection_id(headers)
        if coll_id is None:
            check("12. Dashboard 'Feature Flag Sunset Dashboard'", 1, False, "collection not found")
            return
        resp = requests.get(f"{METABASE_URL}/api/dashboard", headers=headers, timeout=15)
        resp.raise_for_status()
        dashboards = resp.json()
        found = [d for d in dashboards
                 if d.get("name") == "Feature Flag Sunset Dashboard"
                 and d.get("collection_id") == coll_id]
        passed = len(found) > 0
        check("12. Dashboard 'Feature Flag Sunset Dashboard'", 1, passed,
              f"id={found[0]['id']}" if passed else "dashboard not found in collection")
    except Exception as e:
        check("12. Dashboard 'Feature Flag Sunset Dashboard'", 1, False, f"exception: {e}")


def check_13_metabase_dashboard_cards():
    """Dashboard contains both saved question cards."""
    try:
        _, headers = metabase_api_auth()
        coll_id = _get_metabase_collection_id(headers)
        if coll_id is None:
            check("13. Dashboard has both question cards", 2, False, "collection not found")
            return
        resp = requests.get(f"{METABASE_URL}/api/dashboard", headers=headers, timeout=15)
        resp.raise_for_status()
        dashboards = resp.json()
        found = [d for d in dashboards
                 if d.get("name") == "Feature Flag Sunset Dashboard"
                 and d.get("collection_id") == coll_id]
        if not found:
            check("13. Dashboard has both question cards", 2, False, "dashboard not found")
            return
        dash_id = found[0]["id"]
        resp2 = requests.get(f"{METABASE_URL}/api/dashboard/{dash_id}", headers=headers, timeout=15)
        resp2.raise_for_status()
        dash_data = resp2.json()
        cards_list = dash_data.get("ordered_cards") or dash_data.get("dashcards") or []
        card_names = []
        for c in cards_list:
            card = c.get("card") or {}
            if card.get("name"):
                card_names.append(card["name"])
        has_bar = any("Flags by Project" in n for n in card_names)
        has_sunset = any("Sunset Flag" in n for n in card_names)
        passed = has_bar and has_sunset
        detail = f"cards: {card_names}"
        if not has_bar:
            detail += "; missing 'Flags by Project and State'"
        if not has_sunset:
            detail += "; missing 'Sunset Flag Schedule'"
        check("13. Dashboard has both question cards", 2, passed, detail)
    except Exception as e:
        check("13. Dashboard has both question cards", 2, False, f"exception: {e}")


def check_14_openproject_sunset_work_packages():
    """OpenProject has work packages for each sunset flag with correct subjects."""
    try:
        expected_subjects = {f"Remove feature flag: {flag}" for flag in SUNSET_FLAGS}
        result = openproject_db_query(
            "SELECT wp.subject FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.identifier = 'demo-project' "
            "AND wp.subject LIKE 'Remove feature flag:%';"
        )
        found_subjects = {s.strip() for s in result.split("\n") if s.strip()}
        missing = expected_subjects - found_subjects
        passed = len(missing) == 0
        detail = f"found {len(found_subjects)} WPs"
        if missing:
            detail += f", missing: {sorted(missing)}"
        check("14. OpenProject sunset work packages exist", 2, passed, detail)
    except Exception as e:
        check("14. OpenProject sunset work packages exist", 2, False, f"exception: {e}")


def check_15_openproject_wp_details():
    """Work packages have correct assignee email, due date, and priority Normal."""
    try:
        result = openproject_db_query(
            "SELECT wp.subject, u.mail, wp.due_date::text, "
            "  (SELECT e.name FROM enumerations e WHERE e.id = wp.priority_id), "
            "  (SELECT t.name FROM types t WHERE t.id = wp.type_id) "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "LEFT JOIN users u ON wp.assigned_to_id = u.id "
            "WHERE p.identifier = 'demo-project' "
            "AND wp.subject LIKE 'Remove feature flag:%';"
        )
        errors = []
        found_count = 0
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            subject, assignee_email, due_date, priority, wp_type = parts[:5]
            flag_name = subject.replace("Remove feature flag:", "").strip()
            found_count += 1
            if flag_name in FLAG_METADATA:
                exp = FLAG_METADATA[flag_name]
                if assignee_email != exp["owner"]:
                    errors.append(f"{flag_name}: assignee='{assignee_email}' expected '{exp['owner']}'")
                if due_date and due_date[:10] != exp["target_removal_date"]:
                    errors.append(f"{flag_name}: due={due_date[:10]} expected '{exp['target_removal_date']}'")
                if priority.lower() != "normal":
                    errors.append(f"{flag_name}: priority='{priority}' expected 'Normal'")
                if wp_type.lower() != "task":
                    errors.append(f"{flag_name}: type='{wp_type}' expected 'Task'")
        if found_count == 0:
            errors.append("no matching work packages found")
        passed = len(errors) == 0 and found_count == len(SUNSET_FLAGS)
        if found_count != len(SUNSET_FLAGS) and found_count > 0:
            errors.append(f"expected {len(SUNSET_FLAGS)} WPs, found {found_count}")
        check("15. WP details (assignee, due date, priority, type)", 2, passed,
              "; ".join(errors) if errors else "all details match")
    except Exception as e:
        check("15. WP details (assignee, due date, priority, type)", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_code_server_flag_files()
    check_2_baserow_database_exists()
    check_3_baserow_table_exists()
    check_4_baserow_fields()
    check_5_baserow_flags_present()
    check_6_flag_metadata_correct()
    check_7_status_correct()
    check_8_baserow_kanban_view()
    check_9_metabase_collection()
    check_10_metabase_question_bar_chart()
    check_11_metabase_question_sunset()
    check_12_metabase_dashboard_exists()
    check_13_metabase_dashboard_cards()
    check_14_openproject_sunset_work_packages()
    check_15_openproject_wp_details()

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
