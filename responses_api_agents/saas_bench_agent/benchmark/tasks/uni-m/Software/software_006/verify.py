# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_006/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-006-I2: Build SLO Registry and Error-Budget Dashboard

Checks: 13 weighted checks across baserow, code-server, metabase, openproject.
Strategy: Baserow API, code-server docker exec, Metabase API, OpenProject DB.

Required env vars:
  SERVER_HOSTNAME, {BASEROW,CODE_SERVER,METABASE,OPENPROJECT}_PORT,
  {BASEROW,CODE_SERVER,METABASE,OPENPROJECT}_CONTAINER,
  BASEROW_DB_CONTAINER, OPENPROJECT uses embedded DB.
"""

import json
import os
import re
import subprocess
import sys

import requests

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
for var, val in _required.items():
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_BASE = f"http://{HOST}:{BASEROW_PORT}"
METABASE_BASE = f"http://{HOST}:{METABASE_PORT}"

# ── Expected data ─────────────────────────────────────────────────────────────
SERVICES = ["payments-gateway", "auth-service", "inventory-api"]
SLO_DATA = [
    ("Availability", 99.95, 99.88),
    ("Latency", 180.00, 165.40),
    ("ErrorRate", 0.50, 0.72),
]

EXPECTED_ROWS = []
for svc, (slo_type, target, current) in zip(SERVICES, SLO_DATA):
    if slo_type == "Availability":
        budget = round(target - current, 2)
    else:
        budget = round(current - target, 2)
    breaching = budget < 0
    EXPECTED_ROWS.append({
        "service": svc,
        "slo_type": slo_type,
        "target": target,
        "current": current,
        "budget": budget,
        "breaching": breaching,
    })
# payments-gateway: 0.07, false; auth-service: -14.60, true; inventory-api: 0.22, false

BREACHING_ROWS = [r for r in EXPECTED_ROWS if r["breaching"]]

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
    """Get Baserow JWT token."""
    r = requests.post(
        f"{BASEROW_BASE}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["token"]


def baserow_get(path: str, token: str) -> dict:
    r = requests.get(
        f"{BASEROW_BASE}/api/{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def metabase_auth() -> str:
    """Get Metabase session token."""
    r = requests.post(
        f"{METABASE_BASE}/api/session",
        json={"username": "admin@metabase.local", "password": "mw-admin-123"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["id"]


def metabase_get(path: str, token: str) -> dict | list:
    r = requests.get(
        f"{METABASE_BASE}/api/{path}",
        headers={"X-Metabase-Session": token},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def op_db_query(sql: str) -> str:
    """Query OpenProject embedded Postgres DB."""
    rc, stdout, stderr = docker_exec(
        OPENPROJECT_CONTAINER,
        "psql", "-U", "openproject", "-d", "openproject",
        "-t", "-A", "-c", sql,
        timeout=15,
    )
    return stdout.strip()


# ── Baserow checks ───────────────────────────────────────────────────────────

def check_1_table_exists() -> dict | None:
    """Check that database 'Platform SLO Registry' with table 'Service SLOs' exists."""
    try:
        token = baserow_auth()
        apps = baserow_get("applications/", token)
        db_id = None
        for app in apps:
            if app.get("name") == "Platform SLO Registry" and app.get("type") == "database":
                db_id = app["id"]
                break
        if db_id is None:
            check("1. Baserow table exists", 1, False, "database 'Platform SLO Registry' not found")
            return None

        tables = baserow_get(f"database/tables/database/{db_id}/", token)
        table_id = None
        for t in tables:
            if t.get("name") == "Service SLOs":
                table_id = t["id"]
                break
        if table_id is None:
            check("1. Baserow table exists", 1, False, "table 'Service SLOs' not found in database")
            return None

        check("1. Baserow table exists", 1, True, f"db_id={db_id}, table_id={table_id}")
        return {"token": token, "table_id": table_id}
    except Exception as e:
        check("1. Baserow table exists", 1, False, f"exception: {e}")
        return None


def check_2_row_count(ctx: dict | None) -> list | None:
    """Check exactly 3 rows in Service SLOs."""
    if ctx is None:
        check("2. Exactly 3 rows", 1, False, "skipped: table not found")
        return None
    try:
        data = baserow_get(
            f"database/rows/table/{ctx['table_id']}/?user_field_names=true&size=100",
            ctx["token"],
        )
        rows = data.get("results", [])
        check("2. Exactly 3 rows", 1, len(rows) == 3, f"found {len(rows)} rows")
        return rows
    except Exception as e:
        check("2. Exactly 3 rows", 1, False, f"exception: {e}")
        return None


def _find_row(rows: list, service_name: str) -> dict | None:
    for r in rows:
        if r.get("Service") == service_name:
            return r
    return None


def _check_row(check_num: int, rows: list | None, expected: dict) -> None:
    """Check a single service row's values."""
    label = f"{check_num}. Row '{expected['service']}'"
    if rows is None:
        check(label, 2, False, "skipped: no rows")
        return
    try:
        row = _find_row(rows, expected["service"])
        if row is None:
            check(label, 2, False, f"row not found for service '{expected['service']}'")
            return

        # SLO Type: single-select field returns dict with "value" key
        slo_type_raw = row.get("SLO Type")
        if isinstance(slo_type_raw, dict):
            slo_type = slo_type_raw.get("value", "")
        else:
            slo_type = str(slo_type_raw or "")

        target = row.get("Target")
        current = row.get("Current")
        budget = row.get("Budget Remaining")
        breaching = row.get("Breaching")

        issues = []
        if slo_type != expected["slo_type"]:
            issues.append(f"SLO Type: expected '{expected['slo_type']}', got '{slo_type}'")

        def approx(a, b, tol=0.015):
            try:
                return abs(float(a) - float(b)) < tol
            except (TypeError, ValueError):
                return False

        if not approx(target, expected["target"]):
            issues.append(f"Target: expected {expected['target']}, got {target}")
        if not approx(current, expected["current"]):
            issues.append(f"Current: expected {expected['current']}, got {current}")
        if not approx(budget, expected["budget"]):
            issues.append(f"Budget: expected {expected['budget']}, got {budget}")
        if bool(breaching) != expected["breaching"]:
            issues.append(f"Breaching: expected {expected['breaching']}, got {breaching}")

        if issues:
            check(label, 2, False, "; ".join(issues))
        else:
            check(label, 2, True, f"all fields correct")
    except Exception as e:
        check(label, 2, False, f"exception: {e}")


def check_3_row_payments(rows: list | None) -> None:
    _check_row(3, rows, EXPECTED_ROWS[0])


def check_4_row_auth(rows: list | None) -> None:
    _check_row(4, rows, EXPECTED_ROWS[1])


def check_5_row_inventory(rows: list | None) -> None:
    _check_row(5, rows, EXPECTED_ROWS[2])


# ── code-server check ────────────────────────────────────────────────────────


# ── Metabase checks ──────────────────────────────────────────────────────────

def check_7_metabase_collection() -> int | None:
    """Check collection 'Platform Reliability' exists in Metabase."""
    try:
        token = metabase_auth()
        collections = metabase_get("collection", token)
        coll_id = None
        for c in collections:
            if c.get("name") == "Platform Reliability":
                coll_id = c["id"]
                break
        if coll_id is None:
            check("7. Metabase collection exists", 1, False, "'Platform Reliability' not found")
            return None
        check("7. Metabase collection exists", 1, True, f"id={coll_id}")
        return coll_id
    except Exception as e:
        check("7. Metabase collection exists", 1, False, f"exception: {e}")
        return None


def _metabase_collection_items(coll_id: int, token: str) -> list:
    """Get items in a Metabase collection."""
    data = metabase_get(f"collection/{coll_id}/items", token)
    if isinstance(data, dict):
        return data.get("data", [])
    return data


def check_8_metabase_question(coll_id: int | None) -> int | None:
    """Check saved question 'Platform SLO Targets vs Current' exists in collection."""
    if coll_id is None:
        check("8. Metabase saved question", 2, False, "skipped: collection not found")
        return None
    try:
        token = metabase_auth()
        items = _metabase_collection_items(coll_id, token)
        question_id = None
        for item in items:
            if item.get("model") in ("card", "question") and item.get("name") == "Platform SLO Targets vs Current":
                question_id = item["id"]
                break
        if question_id is None:
            # Also search all cards
            cards = metabase_get("card", token)
            for card in cards:
                if card.get("name") == "Platform SLO Targets vs Current" and card.get("collection_id") == coll_id:
                    question_id = card["id"]
                    break
        if question_id is None:
            check("8. Metabase saved question", 2, False, "question not found in collection")
            return None
        check("8. Metabase saved question", 2, True, f"card_id={question_id}")
        return question_id
    except Exception as e:
        check("8. Metabase saved question", 2, False, f"exception: {e}")
        return None


def check_9_metabase_dashboard(coll_id: int | None) -> int | None:
    """Check dashboard 'Platform Error Budget Tracker' exists in collection."""
    if coll_id is None:
        check("9. Metabase dashboard", 2, False, "skipped: collection not found")
        return None
    try:
        token = metabase_auth()
        items = _metabase_collection_items(coll_id, token)
        dash_id = None
        for item in items:
            if item.get("model") == "dashboard" and item.get("name") == "Platform Error Budget Tracker":
                dash_id = item["id"]
                break
        if dash_id is None:
            # Search all dashboards
            dashboards = metabase_get("dashboard", token)
            for d in dashboards:
                if d.get("name") == "Platform Error Budget Tracker" and d.get("collection_id") == coll_id:
                    dash_id = d["id"]
                    break
        if dash_id is None:
            check("9. Metabase dashboard", 2, False, "dashboard not found in collection")
            return None
        check("9. Metabase dashboard", 2, True, f"dash_id={dash_id}")
        return dash_id
    except Exception as e:
        check("9. Metabase dashboard", 2, False, f"exception: {e}")
        return None


def check_10_dashboard_has_card(dash_id: int | None, question_id: int | None) -> None:
    """Check dashboard contains the saved question as a card."""
    if dash_id is None:
        check("10. Dashboard contains question card", 1, False, "skipped: dashboard not found")
        return
    try:
        token = metabase_auth()
        dash = metabase_get(f"dashboard/{dash_id}", token)
        cards = dash.get("dashcards", dash.get("ordered_cards", []))
        if not cards:
            check("10. Dashboard contains question card", 1, False, "no cards on dashboard")
            return
        if question_id is not None:
            found = any(c.get("card_id") == question_id or (c.get("card") or {}).get("id") == question_id for c in cards)
            if found:
                check("10. Dashboard contains question card", 1, True, "question card found on dashboard")
            else:
                check("10. Dashboard contains question card", 1, False, f"question card_id={question_id} not among dashboard cards")
        else:
            # At least one card exists
            check("10. Dashboard contains question card", 1, len(cards) > 0, f"{len(cards)} card(s) on dashboard")
    except Exception as e:
        check("10. Dashboard contains question card", 1, False, f"exception: {e}")


# ── OpenProject checks ───────────────────────────────────────────────────────

def check_11_op_bug_exists() -> None:
    """Check Bug work package 'SLO breach: auth-service (Latency)' exists with High priority in project 'Infrastructure Upgrade'."""
    try:
        sql = """
            SELECT wp.id, wp.subject, s.name AS status, t.name AS type, p2.name AS priority
            FROM work_packages wp
            JOIN projects p ON wp.project_id = p.id
            JOIN types t ON wp.type_id = t.id
            JOIN enumerations p2 ON wp.priority_id = p2.id
            WHERE p.name = 'Infrastructure Upgrade'
              AND wp.subject = 'SLO breach: auth-service (Latency)'
              AND t.name = 'Bug'
        """
        result = op_db_query(sql)
        if not result:
            check("11. OpenProject bug WP exists", 2, False, "work package not found")
            return
        row = result.split("\n")[0]
        cols = row.split("|")
        priority = cols[4].strip() if len(cols) > 4 else ""
        is_high = priority.lower() == "high"
        check("11. OpenProject bug WP exists", 2, is_high, f"priority={priority}")
    except Exception as e:
        check("11. OpenProject bug WP exists", 2, False, f"exception: {e}")


def check_12_op_bug_description() -> None:
    """Check bug work package description contains correct metric values."""
    try:
        sql = """
            SELECT wp.description
            FROM work_packages wp
            JOIN projects p ON wp.project_id = p.id
            JOIN types t ON wp.type_id = t.id
            WHERE p.name = 'Infrastructure Upgrade'
              AND wp.subject = 'SLO breach: auth-service (Latency)'
              AND t.name = 'Bug'
        """
        result = op_db_query(sql)
        if not result:
            check("12. Bug WP description", 2, False, "work package not found")
            return

        desc = result.strip()
        issues = []
        # Expected description contains: Current=165.40, Target=180.00, Budget Remaining=-14.60
        # Allow for minor formatting variations
        if "165.4" not in desc:
            issues.append("missing Current=165.40")
        if "180.0" not in desc:
            issues.append("missing Target=180.00")
        if "-14.6" not in desc:
            issues.append("missing Budget Remaining=-14.60")

        check("12. Bug WP description", 2, not issues,
              "all values present" if not issues else "; ".join(issues))
    except Exception as e:
        check("12. Bug WP description", 2, False, f"exception: {e}")


def check_13_op_no_extra_bugs() -> None:
    """Check no extra SLO breach bug work packages beyond expected ones."""
    try:
        sql = """
            SELECT wp.subject
            FROM work_packages wp
            JOIN projects p ON wp.project_id = p.id
            JOIN types t ON wp.type_id = t.id
            WHERE p.name = 'Infrastructure Upgrade'
              AND t.name = 'Bug'
              AND wp.subject LIKE 'SLO breach:%'
        """
        result = op_db_query(sql)
        rows = [r.strip() for r in result.strip().split("\n") if r.strip()] if result.strip() else []
        expected_count = len(BREACHING_ROWS)  # 1
        check("13. No extra SLO breach bugs", 1, len(rows) == expected_count,
              f"expected {expected_count} SLO breach bug(s), found {len(rows)}")
    except Exception as e:
        check("13. No extra SLO breach bugs", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Baserow
    ctx = check_1_table_exists()
    rows = check_2_row_count(ctx)
    check_3_row_payments(rows)
    check_4_row_auth(rows)
    check_5_row_inventory(rows)

    # Metabase
    coll_id = check_7_metabase_collection()
    question_id = check_8_metabase_question(coll_id)
    dash_id = check_9_metabase_dashboard(coll_id)
    check_10_dashboard_has_card(dash_id, question_id)

    # OpenProject
    check_11_op_bug_exists()
    check_12_op_bug_description()
    check_13_op_no_extra_bugs()

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
