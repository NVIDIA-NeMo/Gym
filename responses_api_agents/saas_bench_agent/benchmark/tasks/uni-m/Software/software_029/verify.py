# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_029/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-029-I4: Data Platform Migration Portfolio Analysis Q4 2026

Checks: 16 weighted checks across baserow, code-server, metabase, openproject.
Strategy: Baserow API, docker exec filesystem, Metabase API, OpenProject DB.

Required env vars:
  SERVER_HOSTNAME, BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  METABASE_PORT, METABASE_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
CODE_SERVER_PORT = os.environ.get("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
METABASE_PORT = os.environ.get("METABASE_PORT")
METABASE_CONTAINER = os.environ.get("METABASE_CONTAINER")
OPENPROJECT_PORT = os.environ.get("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

_required = {
    "BASEROW_PORT": BASEROW_PORT,
    "BASEROW_CONTAINER": BASEROW_CONTAINER,
    "BASEROW_DB_CONTAINER": BASEROW_DB_CONTAINER,
    "CODE_SERVER_PORT": CODE_SERVER_PORT,
    "CODE_SERVER_CONTAINER": CODE_SERVER_CONTAINER,
    "METABASE_PORT": METABASE_PORT,
    "METABASE_CONTAINER": METABASE_CONTAINER,
    "OPENPROJECT_PORT": OPENPROJECT_PORT,
    "OPENPROJECT_CONTAINER": OPENPROJECT_CONTAINER,
}
for var_name, var_val in _required.items():
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"
METABASE_URL = f"http://{HOST}:{METABASE_PORT}"
OPENPROJECT_URL = f"http://{HOST}:{OPENPROJECT_PORT}"

# ── Expected data ─────────────────────────────────────────────────────────────
EXPECTED_CANDIDATES = [
    {"id": "MC-01", "name": "Migrate Hadoop Cluster to EMR Serverless",
     "current": 450000.00, "projected": 270000.00, "savings": 180000.00,
     "effort": 15.0, "risk": 5.5, "alignment": "High", "roi": 1.71, "decision": "Approve"},
    {"id": "MC-02", "name": "Replace Talend ETL with dbt Cloud",
     "current": 210000.00, "projected": 96000.00, "savings": 114000.00,
     "effort": 9.0, "risk": 4.0, "alignment": "High", "roi": 1.81, "decision": "Approve"},
    {"id": "MC-03", "name": "Migrate Tableau Server to Tableau Cloud",
     "current": 156000.00, "projected": 120000.00, "savings": 36000.00,
     "effort": 5.0, "risk": 3.5, "alignment": "Medium", "roi": 0.86, "decision": "Defer"},
    {"id": "MC-04", "name": "Retire Legacy SSIS Packages",
     "current": 78000.00, "projected": 72000.00, "savings": 6000.00,
     "effort": 7.0, "risk": 8.8, "alignment": "Low", "roi": 0.08, "decision": "Reject"},
    {"id": "MC-05", "name": "Move Airflow Self-hosted to MWAA",
     "current": 132000.00, "projected": 78000.00, "savings": 54000.00,
     "effort": 6.0, "risk": 4.5, "alignment": "Medium", "roi": 1.07, "decision": "Defer"},
    {"id": "MC-06", "name": "Consolidate Data Catalogs onto AWS Glue",
     "current": 96000.00, "projected": 84000.00, "savings": 12000.00,
     "effort": 8.0, "risk": 7.0, "alignment": "Low", "roi": 0.14, "decision": "Reject"},
]

APPROVED = [c for c in EXPECTED_CANDIDATES if c["decision"] == "Approve"]

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


def http_request(url: str, method: str = "GET", data: dict | None = None,
                 headers: dict | None = None, timeout: int = 15) -> tuple[int, dict | str]:
    """Make an HTTP request and return (status_code, parsed_json_or_text)."""
    hdrs = headers or {}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def baserow_auth() -> str:
    """Authenticate to Baserow API and return JWT token."""
    status, resp = http_request(
        f"{BASEROW_URL}/api/user/token-auth/",
        method="POST",
        data={"email": "admin@example.com", "password": "Admin1234"},
    )
    if status != 200:
        raise RuntimeError(f"Baserow auth failed: {status} {resp}")
    return resp["access_token"]


def metabase_auth() -> str:
    """Authenticate to Metabase API and return session token."""
    status, resp = http_request(
        f"{METABASE_URL}/api/session",
        method="POST",
        data={"username": "admin@metabase.local", "password": "mw-admin-123"},
    )
    if status != 200:
        raise RuntimeError(f"Metabase auth failed: {status} {resp}")
    return resp["id"]


def op_db_query(sql: str) -> str:
    """Query OpenProject embedded Postgres via TCP with password auth."""
    r = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=openproject",
         OPENPROJECT_CONTAINER,
         "psql", "-h", "127.0.0.1", "-U", "openproject", "-d", "openproject",
         "-t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"OpenProject DB query failed: {r.stderr.strip()}")
    return r.stdout.strip()


# ── Baserow checks ───────────────────────────────────────────────────────────
_baserow_token = None
_baserow_table_id = None
_baserow_rows = None


def _get_baserow_token():
    global _baserow_token
    if _baserow_token is None:
        _baserow_token = baserow_auth()
    return _baserow_token


def _baserow_get(path: str) -> tuple[int, dict | str]:
    token = _get_baserow_token()
    return http_request(
        f"{BASEROW_URL}{path}",
        headers={"Authorization": f"JWT {token}"},
    )


def check_1_baserow_database_exists() -> None:
    """Check that the Baserow database 'Data Platform Migration Portfolio Q4 2026' exists."""
    try:
        status, apps = _baserow_get("/api/applications/")
        if status != 200:
            check("1. Baserow database exists", 1, False, f"API returned {status}")
            return
        db_name = "Data Platform Migration Portfolio Q4 2026"
        found = [a for a in apps if a.get("name") == db_name and a.get("type") == "database"]
        if found:
            check("1. Baserow database exists", 1, True)
        else:
            names = [a.get("name") for a in apps if a.get("type") == "database"]
            check("1. Baserow database exists", 1, False, f"not found among {names}")
    except Exception as e:
        check("1. Baserow database exists", 1, False, f"exception: {e}")


def _find_table_id() -> int | None:
    """Find the table ID for 'Migration Candidates' in the target database."""
    global _baserow_table_id
    if _baserow_table_id is not None:
        return _baserow_table_id
    status, apps = _baserow_get("/api/applications/")
    if status != 200:
        return None
    db_name = "Data Platform Migration Portfolio Q4 2026"
    for app in apps:
        if app.get("name") == db_name and app.get("type") == "database":
            for tbl in app.get("tables", []):
                if tbl.get("name") == "Migration Candidates":
                    _baserow_table_id = tbl["id"]
                    return _baserow_table_id
    return None


def _get_rows() -> list[dict] | None:
    """Fetch all rows from the Migration Candidates table with user field names."""
    global _baserow_rows
    if _baserow_rows is not None:
        return _baserow_rows
    table_id = _find_table_id()
    if table_id is None:
        return None
    status, resp = _baserow_get(f"/api/database/rows/table/{table_id}/?user_field_names=true&size=100")
    if status != 200:
        return None
    _baserow_rows = resp.get("results", [])
    return _baserow_rows


def check_2_table_and_row_count() -> None:
    """Check table exists with exactly 6 rows."""
    try:
        table_id = _find_table_id()
        if table_id is None:
            check("2. Table 'Migration Candidates' with 6 rows", 1, False, "table not found")
            return
        rows = _get_rows()
        if rows is None:
            check("2. Table 'Migration Candidates' with 6 rows", 1, False, "could not fetch rows")
            return
        count = len(rows)
        check("2. Table 'Migration Candidates' with 6 rows", 1, count == 6,
              f"found {count} rows")
    except Exception as e:
        check("2. Table 'Migration Candidates' with 6 rows", 1, False, f"exception: {e}")


def _match_row(rows: list[dict], candidate_id: str) -> dict | None:
    """Find a row matching the given candidate ID."""
    for row in rows:
        # Try common field name variants
        for key in ["Candidate ID", "candidate_id", "Candidate Id"]:
            if row.get(key) == candidate_id:
                return row
        # Also check the primary field (first visible text) — Baserow might use 'order' key
        # or the value might be in the first text column
        for key, val in row.items():
            if isinstance(val, str) and val.strip() == candidate_id:
                return row
    return None


def _get_field_value(row: dict, *field_names: str) -> str | float | None:
    """Get field value trying multiple name variants."""
    for name in field_names:
        if name in row:
            val = row[name]
            # Single-select fields in Baserow API return {"id": ..., "value": ..., "color": ...}
            if isinstance(val, dict) and "value" in val:
                return val["value"]
            return val
    return None


def check_3_annual_savings() -> None:
    """Verify Annual Savings = Current - Projected for all 6 rows."""
    try:
        rows = _get_rows()
        if not rows:
            check("3. Annual Savings correct", 3, False, "no rows available")
            return
        errors = []
        for cand in EXPECTED_CANDIDATES:
            row = _match_row(rows, cand["id"])
            if row is None:
                errors.append(f"{cand['id']} not found")
                continue
            savings = _get_field_value(row, "Annual Savings", "annual_savings")
            if savings is None:
                errors.append(f"{cand['id']}: field not found")
                continue
            try:
                actual = float(savings)
            except (ValueError, TypeError):
                errors.append(f"{cand['id']}: non-numeric '{savings}'")
                continue
            if abs(actual - cand["savings"]) > 0.1:
                errors.append(f"{cand['id']}: expected {cand['savings']}, got {actual}")
        passed = len(errors) == 0
        check("3. Annual Savings correct", 3, passed,
              "; ".join(errors) if errors else "all 6 correct")
    except Exception as e:
        check("3. Annual Savings correct", 3, False, f"exception: {e}")


def check_4_roi_score() -> None:
    """Verify ROI Score for all 6 rows."""
    try:
        rows = _get_rows()
        if not rows:
            check("4. ROI Score correct", 3, False, "no rows available")
            return
        errors = []
        for cand in EXPECTED_CANDIDATES:
            row = _match_row(rows, cand["id"])
            if row is None:
                errors.append(f"{cand['id']} not found")
                continue
            roi = _get_field_value(row, "ROI Score", "roi_score")
            if roi is None:
                errors.append(f"{cand['id']}: field not found")
                continue
            try:
                actual = float(roi)
            except (ValueError, TypeError):
                errors.append(f"{cand['id']}: non-numeric '{roi}'")
                continue
            if abs(actual - cand["roi"]) > 0.02:
                errors.append(f"{cand['id']}: expected {cand['roi']}, got {actual}")
        passed = len(errors) == 0
        check("4. ROI Score correct", 3, passed,
              "; ".join(errors) if errors else "all 6 correct")
    except Exception as e:
        check("4. ROI Score correct", 3, False, f"exception: {e}")


def check_5_decision() -> None:
    """Verify Decision field for all 6 rows."""
    try:
        rows = _get_rows()
        if not rows:
            check("5. Decision correct", 2, False, "no rows available")
            return
        errors = []
        for cand in EXPECTED_CANDIDATES:
            row = _match_row(rows, cand["id"])
            if row is None:
                errors.append(f"{cand['id']} not found")
                continue
            decision = _get_field_value(row, "Decision", "decision")
            if decision is None:
                errors.append(f"{cand['id']}: field not found")
                continue
            actual = str(decision).strip()
            if actual.lower() != cand["decision"].lower():
                errors.append(f"{cand['id']}: expected {cand['decision']}, got {actual}")
        passed = len(errors) == 0
        check("5. Decision correct", 2, passed,
              "; ".join(errors) if errors else "all 6 correct")
    except Exception as e:
        check("5. Decision correct", 2, False, f"exception: {e}")


def check_6_ranked_candidates_view() -> None:
    """Check that a Grid view named 'Ranked Candidates' exists."""
    try:
        table_id = _find_table_id()
        if table_id is None:
            check("6. View 'Ranked Candidates' exists", 1, False, "table not found")
            return
        status, views = _baserow_get(f"/api/database/views/table/{table_id}/")
        if status != 200:
            check("6. View 'Ranked Candidates' exists", 1, False, f"API returned {status}")
            return
        found = any(v.get("name") == "Ranked Candidates" for v in views)
        check("6. View 'Ranked Candidates' exists", 1, found,
              "" if found else f"views: {[v.get('name') for v in views]}")
    except Exception as e:
        check("6. View 'Ranked Candidates' exists", 1, False, f"exception: {e}")


# ── Code-server checks ───────────────────────────────────────────────────────
def _find_alertmanager_file() -> str | None:
    """Find the alertmanager.yml file path in the code-server container."""
    rc, out, _ = docker_exec(
        CODE_SERVER_CONTAINER,
        "find", "/home", "-maxdepth", "6", "-path", "*/devops-configs/monitoring/alertmanager.yml",
        "-type", "f",
        timeout=10,
    )
    if rc == 0 and out.strip():
        return out.strip().split("\n")[0]
    # Also try /config/workspace or other common paths
    rc2, out2, _ = docker_exec(
        CODE_SERVER_CONTAINER,
        "find", "/", "-maxdepth", "6", "-path", "*/devops-configs/monitoring/alertmanager.yml",
        "-type", "f",
        timeout=10,
    )
    if rc2 == 0 and out2.strip():
        return out2.strip().split("\n")[0]
    return None


def check_7_section_marker() -> None:
    """Check alertmanager.yml contains the section marker."""
    try:
        filepath = _find_alertmanager_file()
        if filepath is None:
            check("7. Section marker in alertmanager.yml", 1, False, "file not found")
            return
        rc, content, _ = docker_exec(CODE_SERVER_CONTAINER, "cat", filepath)
        if rc != 0:
            check("7. Section marker in alertmanager.yml", 1, False, "cannot read file")
            return
        marker = "# === DATA PLATFORM MIGRATION Q4 NOTES ==="
        found = marker in content
        check("7. Section marker in alertmanager.yml", 1, found,
              "" if found else "marker line not found")
    except Exception as e:
        check("7. Section marker in alertmanager.yml", 1, False, f"exception: {e}")


def check_8_comment_lines() -> None:
    """Check 6 correctly formatted migration comment lines below the marker."""
    try:
        filepath = _find_alertmanager_file()
        if filepath is None:
            check("8. 6 migration comment lines", 2, False, "file not found")
            return
        rc, content, _ = docker_exec(CODE_SERVER_CONTAINER, "cat", filepath)
        if rc != 0:
            check("8. 6 migration comment lines", 2, False, "cannot read file")
            return
        marker = "# === DATA PLATFORM MIGRATION Q4 NOTES ==="
        lines = content.split("\n")
        marker_idx = None
        for i, line in enumerate(lines):
            if marker in line:
                marker_idx = i
                break
        if marker_idx is None:
            check("8. 6 migration comment lines", 2, False, "marker not found")
            return
        # Get lines after marker, skipping blanks
        after_lines = []
        for line in lines[marker_idx + 1:]:
            stripped = line.strip()
            if stripped.startswith("# MIGRATION-CANDIDATE"):
                after_lines.append(stripped)
            elif stripped and not stripped.startswith("#"):
                break  # Stop at non-comment content
            elif len(after_lines) >= 6:
                break
        errors = []
        if len(after_lines) != 6:
            errors.append(f"expected 6 lines, found {len(after_lines)}")
        for cand in EXPECTED_CANDIDATES:
            # Build expected pattern — allow both em dash and double hyphen
            expected_fragments = [
                f"# MIGRATION-CANDIDATE {cand['id']}:",
                cand["name"],
                f"Effort {cand['effort']}w",
                f"ROI {cand['roi']}",
                f"Decision {cand['decision']}",
            ]
            found_line = False
            for line in after_lines:
                if all(frag in line for frag in expected_fragments):
                    found_line = True
                    break
            if not found_line:
                errors.append(f"{cand['id']} line missing or malformed")
        passed = len(errors) == 0
        check("8. 6 migration comment lines", 2, passed,
              "; ".join(errors) if errors else "all 6 correct")
    except Exception as e:
        check("8. 6 migration comment lines", 2, False, f"exception: {e}")


# ── Metabase checks ──────────────────────────────────────────────────────────
_metabase_session = None
_metabase_collection_id = None


def _get_metabase_session():
    global _metabase_session
    if _metabase_session is None:
        _metabase_session = metabase_auth()
    return _metabase_session


def _metabase_get(path: str) -> tuple[int, dict | str | list]:
    session = _get_metabase_session()
    return http_request(
        f"{METABASE_URL}{path}",
        headers={"X-Metabase-Session": session},
    )


def _find_metabase_collection() -> int | None:
    global _metabase_collection_id
    if _metabase_collection_id is not None:
        return _metabase_collection_id
    status, collections = _metabase_get("/api/collection")
    if status != 200:
        return None
    target = "Data Platform Migration Analysis Q4 2026"
    for c in collections:
        if c.get("name") == target:
            _metabase_collection_id = c["id"]
            return _metabase_collection_id
    return None


def check_9_metabase_collection() -> None:
    """Check that the Metabase collection exists."""
    try:
        cid = _find_metabase_collection()
        check("9. Metabase collection exists", 1, cid is not None,
              "" if cid else "collection not found")
    except Exception as e:
        check("9. Metabase collection exists", 1, False, f"exception: {e}")


def check_10_metabase_questions() -> None:
    """Check 3 questions exist with correct names in the collection."""
    try:
        cid = _find_metabase_collection()
        if cid is None:
            check("10. 3 Metabase questions exist", 2, False, "collection not found")
            return
        status, items = _metabase_get(f"/api/collection/{cid}/items?models=card")
        if status != 200:
            check("10. 3 Metabase questions exist", 2, False, f"API returned {status}")
            return
        item_data = items.get("data", items) if isinstance(items, dict) else items
        card_names = [item.get("name") for item in item_data if item.get("model") == "card"]
        expected_names = [
            "Effort vs ROI",
            "Decisions Breakdown",
            "Total Projected Annual Savings (Approved)",
        ]
        missing = [n for n in expected_names if n not in card_names]
        check("10. 3 Metabase questions exist", 2, len(missing) == 0,
              f"missing: {missing}" if missing else f"all 3 found among {card_names}")
    except Exception as e:
        check("10. 3 Metabase questions exist", 2, False, f"exception: {e}")


def check_11_metabase_dashboard() -> None:
    """Check dashboard exists with correct name and description."""
    try:
        cid = _find_metabase_collection()
        if cid is None:
            check("11. Metabase dashboard exists with description", 2, False, "collection not found")
            return
        status, items = _metabase_get(f"/api/collection/{cid}/items?models=dashboard")
        if status != 200:
            check("11. Metabase dashboard exists with description", 2, False, f"API returned {status}")
            return
        item_data = items.get("data", items) if isinstance(items, dict) else items
        dashboards = [d for d in item_data if d.get("model") == "dashboard"]
        target_name = "Data Platform Migration Dashboard Q4 2026"
        target_desc = "Platform migration portfolio as of 2026-10-08"
        found_dash = None
        for d in dashboards:
            if d.get("name") == target_name:
                found_dash = d
                break
        if found_dash is None:
            check("11. Metabase dashboard exists with description", 2, False,
                  f"dashboard not found; found: {[d.get('name') for d in dashboards]}")
            return
        # Fetch full dashboard to get description
        dash_id = found_dash["id"]
        status2, dash_detail = _metabase_get(f"/api/dashboard/{dash_id}")
        if status2 != 200:
            check("11. Metabase dashboard exists with description", 2, False, f"cannot fetch dashboard detail: {status2}")
            return
        actual_desc = (dash_detail.get("description") or "").strip()
        passed = actual_desc == target_desc
        check("11. Metabase dashboard exists with description", 2, passed,
              f"description: '{actual_desc}'" if not passed else "")
    except Exception as e:
        check("11. Metabase dashboard exists with description", 2, False, f"exception: {e}")


def check_12_dashboard_cards() -> None:
    """Check dashboard has 3 cards."""
    try:
        cid = _find_metabase_collection()
        if cid is None:
            check("12. Dashboard has 3 cards", 1, False, "collection not found")
            return
        status, items = _metabase_get(f"/api/collection/{cid}/items?models=dashboard")
        if status != 200:
            check("12. Dashboard has 3 cards", 1, False, f"API returned {status}")
            return
        item_data = items.get("data", items) if isinstance(items, dict) else items
        target_name = "Data Platform Migration Dashboard Q4 2026"
        dash_id = None
        for d in item_data:
            if d.get("name") == target_name and d.get("model") == "dashboard":
                dash_id = d["id"]
                break
        if dash_id is None:
            check("12. Dashboard has 3 cards", 1, False, "dashboard not found")
            return
        status2, dash_detail = _metabase_get(f"/api/dashboard/{dash_id}")
        if status2 != 200:
            check("12. Dashboard has 3 cards", 1, False, f"cannot fetch dashboard: {status2}")
            return
        # Count cards that are question cards (not text/heading cards)
        cards = dash_detail.get("dashcards", dash_detail.get("ordered_cards", []))
        question_cards = [c for c in cards if c.get("card_id") or c.get("card", {}).get("id")]
        count = len(question_cards)
        check("12. Dashboard has 3 cards", 1, count >= 3,
              f"found {count} question cards")
    except Exception as e:
        check("12. Dashboard has 3 cards", 1, False, f"exception: {e}")


# ── OpenProject checks ───────────────────────────────────────────────────────
def check_13_op_version() -> None:
    """Check version 'Migration-Portfolio-2026-10-08' exists with correct dates."""
    try:
        sql = (
            "SELECT v.name, v.start_date, v.effective_date, v.status "
            "FROM versions v "
            "JOIN projects p ON v.project_id = p.id "
            "WHERE p.identifier = 'data-analytics-pipeline' "
            "AND v.name = 'Migration-Portfolio-2026-10-08'"
        )
        result = op_db_query(sql)
        if not result:
            check("13. OpenProject version exists", 2, False, "version not found")
            return
        parts = result.split("|")
        errors = []
        if len(parts) >= 3:
            start_date = parts[1].strip()
            due_date = parts[2].strip()
            if start_date != "2026-10-08":
                errors.append(f"start_date={start_date}, expected 2026-10-08")
            if due_date != "2027-03-31":
                errors.append(f"due_date={due_date}, expected 2027-03-31")
        else:
            errors.append(f"unexpected format: {result}")
        if len(parts) >= 4:
            status_val = parts[3].strip()
            if status_val != "open":
                errors.append(f"status={status_val}, expected open")
        passed = len(errors) == 0
        check("13. OpenProject version exists", 2, passed,
              "; ".join(errors) if errors else "")
    except Exception as e:
        check("13. OpenProject version exists", 2, False, f"exception: {e}")


def check_14_op_epic_subjects() -> None:
    """Check 2 Epic work packages exist for approved candidates."""
    try:
        sql = (
            "SELECT wp.subject "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.identifier = 'data-analytics-pipeline' "
            "AND t.name = 'Epic' "
            "ORDER BY wp.subject"
        )
        result = op_db_query(sql)
        if not result:
            check("14. 2 Epic work packages for approved candidates", 2, False, "no epics found")
            return
        subjects = [s.strip() for s in result.split("\n") if s.strip()]
        expected_subjects = [
            "Migrate: Migrate Hadoop Cluster to EMR Serverless",
            "Migrate: Replace Talend ETL with dbt Cloud",
        ]
        missing = [s for s in expected_subjects if s not in subjects]
        extra = [s for s in subjects if s not in expected_subjects]
        errors = []
        if missing:
            errors.append(f"missing: {missing}")
        if extra:
            errors.append(f"extra epics: {extra}")
        passed = len(missing) == 0 and len(subjects) == 2
        check("14. 2 Epic work packages for approved candidates", 2, passed,
              "; ".join(errors) if errors else f"found: {subjects}")
    except Exception as e:
        check("14. 2 Epic work packages for approved candidates", 2, False, f"exception: {e}")


def check_15_op_descriptions() -> None:
    """Check work package descriptions contain correct values."""
    try:
        sql = (
            "SELECT wp.subject, wp.description "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.identifier = 'data-analytics-pipeline' "
            "AND t.name = 'Epic' "
            "AND (wp.subject LIKE 'Migrate:%')"
        )
        result = op_db_query(sql)
        if not result:
            check("15. Epic descriptions correct", 2, False, "no epics found")
            return
        rows = [r.strip() for r in result.split("\n") if r.strip()]
        errors = []
        for cand in APPROVED:
            expected_subject = f"Migrate: {cand['name']}"
            found = False
            for row in rows:
                if expected_subject in row:
                    found = True
                    desc = row.split("|", 1)[1] if "|" in row else ""
                    # Check key values in description
                    checks_list = [
                        (f"{cand['effort']}", "effort"),
                        (f"{cand['savings']}", "savings"),
                        (f"{cand['roi']}", "roi"),
                        (cand["alignment"], "alignment"),
                    ]
                    for val, label in checks_list:
                        if str(val) not in desc:
                            # Try alternate formats
                            if label == "savings":
                                alt = f"{int(cand['savings'])}"
                                if alt not in desc:
                                    errors.append(f"{cand['id']}: {label} value {val} not in description")
                            else:
                                errors.append(f"{cand['id']}: {label} value {val} not in description")
                    break
            if not found:
                errors.append(f"{cand['id']}: epic '{expected_subject}' not found")
        passed = len(errors) == 0
        check("15. Epic descriptions correct", 2, passed,
              "; ".join(errors) if errors else "")
    except Exception as e:
        check("15. Epic descriptions correct", 2, False, f"exception: {e}")


def check_16_op_priorities() -> None:
    """Check work package priorities: High for Hadoop (risk>=5.0), Normal for Talend."""
    try:
        sql = (
            "SELECT wp.subject, e.name AS priority "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "JOIN enumerations e ON wp.priority_id = e.id "
            "WHERE p.identifier = 'data-analytics-pipeline' "
            "AND t.name = 'Epic' "
            "AND (wp.subject LIKE 'Migrate:%')"
        )
        result = op_db_query(sql)
        if not result:
            check("16. Epic priorities correct", 1, False, "no epics found")
            return
        rows = [r.strip() for r in result.split("\n") if r.strip()]
        errors = []
        expected_priorities = {
            "Migrate: Migrate Hadoop Cluster to EMR Serverless": "High",
            "Migrate: Replace Talend ETL with dbt Cloud": "Normal",
        }
        for subj, expected_pri in expected_priorities.items():
            found = False
            for row in rows:
                if subj in row:
                    found = True
                    parts = row.split("|")
                    actual_pri = parts[-1].strip() if len(parts) >= 2 else "unknown"
                    if actual_pri.lower() != expected_pri.lower():
                        errors.append(f"'{subj}': expected {expected_pri}, got {actual_pri}")
                    break
            if not found:
                errors.append(f"'{subj}' not found")
        passed = len(errors) == 0
        check("16. Epic priorities correct", 1, passed,
              "; ".join(errors) if errors else "")
    except Exception as e:
        check("16. Epic priorities correct", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_database_exists()
    check_2_table_and_row_count()
    check_3_annual_savings()
    check_4_roi_score()
    check_5_decision()
    check_6_ranked_candidates_view()
    check_7_section_marker()
    check_8_comment_lines()
    check_9_metabase_collection()
    check_10_metabase_questions()
    check_11_metabase_dashboard()
    check_12_dashboard_cards()
    check_13_op_version()
    check_14_op_epic_subjects()
    check_15_op_descriptions()
    check_16_op_priorities()

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
