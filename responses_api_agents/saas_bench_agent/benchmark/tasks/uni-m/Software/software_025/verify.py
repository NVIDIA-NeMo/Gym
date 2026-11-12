# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_025/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-025-I1: Cross-project lint audit with Baserow import,
Metabase dashboard, and OpenProject remediation tasks.

Checks: 14 weighted checks across code-server, baserow, metabase, openproject.
Strategy: docker exec (code-server filesystem, baserow DB, openproject DB),
          REST API (baserow views, metabase).

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  METABASE_PORT, METABASE_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import re
import shlex
import subprocess
import sys

import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")


def require_env(name: str) -> str:
    val = os.getenv(name, "")
    if not val:
        print(f"FATAL: {name} not set", file=sys.stderr)
        sys.exit(1)
    return val


CODE_SERVER_CONTAINER = require_env("CODE_SERVER_CONTAINER")
CODE_SERVER_PORT = require_env("CODE_SERVER_PORT")

BASEROW_PORT = require_env("BASEROW_PORT")
BASEROW_CONTAINER = require_env("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = require_env("BASEROW_DB_CONTAINER")

METABASE_PORT = require_env("METABASE_PORT")
METABASE_CONTAINER = require_env("METABASE_CONTAINER")

OPENPROJECT_PORT = require_env("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = require_env("OPENPROJECT_CONTAINER")

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
    """Query Baserow's Postgres DB."""
    rc, out, err = docker_exec(
        BASEROW_DB_CONTAINER, "env", "PGPASSWORD=kdpzkuyhsgb22onku8y7rxkx3czej88nxpngaz4mlmgad67vpc", "psql", "-h", "127.0.0.1", "-U", "baserow", "-d", "baserow",
        "-t", "-A", "-c", sql,
    )
    return out.strip()


def openproject_db_query(sql: str) -> str:
    """Query OpenProject's embedded Postgres DB."""
    escaped = shlex.quote(sql)
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER, "bash", "-c",
        f"PGPASSWORD=openproject psql -h localhost -U openproject -d openproject -t -A -c {escaped}",
    )
    return out.strip()


_metabase_token: str | None = None


def get_metabase_token() -> str:
    global _metabase_token
    if _metabase_token:
        return _metabase_token
    base = f"http://{HOST}:{METABASE_PORT}"
    r = requests.post(
        f"{base}/api/session",
        json={"username": "admin@metabase.local", "password": "mw-admin-123"},
        timeout=10,
    )
    r.raise_for_status()
    _metabase_token = r.json()["id"]
    return _metabase_token


def metabase_get(path: str):
    base = f"http://{HOST}:{METABASE_PORT}"
    token = get_metabase_token()
    r = requests.get(
        f"{base}{path}", headers={"X-Metabase-Session": token}, timeout=15,
    )
    r.raise_for_status()
    return r.json()


_baserow_token: str | None = None


def get_baserow_token() -> str:
    global _baserow_token
    if _baserow_token:
        return _baserow_token
    base = f"http://{HOST}:{BASEROW_PORT}"
    r = requests.post(
        f"{base}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=10,
    )
    r.raise_for_status()
    _baserow_token = r.json()["token"]
    return _baserow_token


def baserow_api_get(path: str):
    base = f"http://{HOST}:{BASEROW_PORT}"
    token = get_baserow_token()
    r = requests.get(
        f"{base}{path}", headers={"Authorization": f"JWT {token}"}, timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _get_baserow_table_id() -> str | None:
    """Return the Baserow internal table id for 'Lint Violations'."""
    result = baserow_db_query(
        "SELECT dt.id FROM database_table dt "
        "JOIN core_application ca ON dt.database_id = ca.id "
        "WHERE ca.name = 'Code Quality Audit Q2 2025' "
        "AND dt.name = 'Lint Violations'"
    )
    return result.split("\n")[0].strip() if result else None


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_csv_exists() -> None:
    """lint_violations.csv exists somewhere under /home in code-server."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER, "bash", "-c",
            "find /home -name 'lint_violations.csv' -type f 2>/dev/null | head -5",
        )
        found = out.strip()
        check("1. CSV file exists in code-server", 1, bool(found),
              found.split("\n")[0] if found else "lint_violations.csv not found")
    except Exception as e:
        check("1. CSV file exists in code-server", 1, False, f"exception: {e}")


def check_2_csv_columns() -> None:
    """CSV has columns Project, File Path, Rule ID, Severity and ≥1 data row."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER, "bash", "-c",
            "find /home -name 'lint_violations.csv' -type f -exec head -1 {} \\; 2>/dev/null | head -1",
        )
        header = out.strip().lower()
        has_project = "project" in header
        has_filepath = ("file path" in header or "file_path" in header
                        or "filepath" in header)
        has_ruleid = ("rule id" in header or "rule_id" in header
                      or "ruleid" in header)
        has_severity = "severity" in header
        all_cols = has_project and has_filepath and has_ruleid and has_severity

        rc2, out2, _ = docker_exec(
            CODE_SERVER_CONTAINER, "bash", "-c",
            "find /home -name 'lint_violations.csv' -type f -exec wc -l {} \\; 2>/dev/null | head -1",
        )
        parts = out2.strip().split()
        row_count = int(parts[0]) - 1 if parts and parts[0].isdigit() else 0

        passed = all_cols and row_count > 0
        check("2. CSV columns and rows", 2, passed,
              f"header='{out.strip()}', data_rows={row_count}")
    except Exception as e:
        check("2. CSV columns and rows", 2, False, f"exception: {e}")


def check_3_baserow_database() -> None:
    """Baserow database 'Code Quality Audit Q2 2025' exists."""
    try:
        result = baserow_db_query(
            "SELECT ca.id FROM core_application ca "
            "WHERE ca.name = 'Code Quality Audit Q2 2025'"
        )
        check("3. Baserow database exists", 1, bool(result),
              f"app_id={result}" if result else "database not found")
    except Exception as e:
        check("3. Baserow database exists", 1, False, f"exception: {e}")


def check_4_baserow_table_rows() -> None:
    """Table 'Lint Violations' exists with imported rows."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("4. Baserow table with rows", 2, False, "table not found")
            return
        row_count = baserow_db_query(
            f"SELECT count(*) FROM database_table_{table_id}"
        )
        count = int(row_count) if row_count else 0
        check("4. Baserow table with rows", 2, count > 0,
              f"table_id={table_id}, rows={count}")
    except Exception as e:
        check("4. Baserow table with rows", 2, False, f"exception: {e}")


def check_5_violation_ids() -> None:
    """Violation IDs formatted LV-NNN sequentially from LV-001."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("5. Violation IDs sequential LV-NNN", 2, False, "table not found")
            return

        primary_field = baserow_db_query(
            f"SELECT df.id FROM database_field df "
            f"WHERE df.table_id = {table_id} AND df.\"primary\" = true"
        ).strip()
        if not primary_field:
            check("5. Violation IDs sequential LV-NNN", 2, False, "primary field not found")
            return

        rows = baserow_db_query(
            f"SELECT field_{primary_field} FROM database_table_{table_id} "
            f"ORDER BY \"order\" ASC, id ASC"
        )
        if not rows:
            check("5. Violation IDs sequential LV-NNN", 2, False, "no rows")
            return

        ids = [r.strip() for r in rows.split("\n") if r.strip()]
        mismatch_detail = ""
        all_match = True
        for i, vid in enumerate(ids):
            expected = f"LV-{i + 1:03d}"
            if vid != expected:
                all_match = False
                mismatch_detail = f"index {i}: got '{vid}', expected '{expected}'"
                break

        check("5. Violation IDs sequential LV-NNN", 2, all_match,
              f"count={len(ids)}, first={ids[0]}, last={ids[-1]}"
              + (f", MISMATCH {mismatch_detail}" if not all_match else ""))
    except Exception as e:
        check("5. Violation IDs sequential LV-NNN", 2, False, f"exception: {e}")


def check_6_captured_at() -> None:
    """Captured At = 2025-05-14 for every row."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("6. Captured At date correct", 1, False, "table not found")
            return

        cap_field = baserow_db_query(
            f"SELECT df.id FROM database_field df "
            f"WHERE df.table_id = {table_id} AND df.name = 'Captured At'"
        ).strip()
        if not cap_field:
            check("6. Captured At date correct", 1, False, "field not found")
            return

        total = baserow_db_query(
            f"SELECT count(*) FROM database_table_{table_id}"
        )
        wrong = baserow_db_query(
            f"SELECT count(*) FROM database_table_{table_id} "
            f"WHERE field_{cap_field}::text NOT LIKE '2025-05-14%'"
        )
        check("6. Captured At date correct", 1, wrong == "0",
              f"total={total}, wrong_date={wrong}")
    except Exception as e:
        check("6. Captured At date correct", 1, False, f"exception: {e}")


def check_7_top_offenders_view() -> None:
    """'Top Offenders' grid view with group-by File Path and filter Severity=Error."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("7. Top Offenders view", 2, False, "table not found")
            return

        views = baserow_api_get(f"/api/database/views/table/{table_id}/")
        top_view = None
        for v in views:
            if v.get("name") == "Top Offenders":
                top_view = v
                break

        if not top_view:
            check("7. Top Offenders view", 2, False, "view not found")
            return

        view_id = top_view["id"]

        # Check filter: Severity = Error
        filters = baserow_api_get(f"/api/database/views/{view_id}/filters/")
        has_severity_filter = False
        for f in filters:
            field_id = f.get("field")
            if field_id:
                fname = baserow_db_query(
                    f"SELECT df.name FROM database_field df WHERE df.id = {field_id}"
                ).strip()
                if fname == "Severity" and "Error" in str(f.get("value", "")):
                    has_severity_filter = True

        # Check group-by: File Path (try DB table for group_bys)
        has_grouping = False
        try:
            group_rows = baserow_db_query(
                f"SELECT df.name FROM database_viewgroupby vg "
                f"JOIN database_field df ON vg.field_id = df.id "
                f"WHERE vg.view_id = {view_id}"
            ).strip()
            if "File Path" in group_rows:
                has_grouping = True
        except Exception:
            pass

        # Fallback: check via API view detail for group_bys
        if not has_grouping:
            try:
                detail = baserow_api_get(f"/api/database/views/{view_id}/")
                for gb in detail.get("group_bys", []):
                    fid = gb.get("field")
                    if fid:
                        fname = baserow_db_query(
                            f"SELECT df.name FROM database_field df WHERE df.id = {fid}"
                        ).strip()
                        if fname == "File Path":
                            has_grouping = True
            except Exception:
                pass

        check("7. Top Offenders view", 2, has_severity_filter and has_grouping,
              f"severity_error_filter={has_severity_filter}, grouped_file_path={has_grouping}")
    except Exception as e:
        check("7. Top Offenders view", 2, False, f"exception: {e}")


def check_8_metabase_collection() -> None:
    """Collection 'Lint Audit Q2 2025' exists."""
    try:
        collections = metabase_get("/api/collection")
        found = any(c.get("name") == "Lint Audit Q2 2025" for c in collections)
        check("8. Metabase collection exists", 1, found,
              "found" if found else "not found")
    except Exception as e:
        check("8. Metabase collection exists", 1, False, f"exception: {e}")


def _find_collection_id() -> int | None:
    collections = metabase_get("/api/collection")
    for c in collections:
        if c.get("name") == "Lint Audit Q2 2025":
            return c["id"]
    return None


def _get_collection_cards(col_id: int) -> list:
    """Return card items in a collection."""
    resp = metabase_get(f"/api/collection/{col_id}/items?models=card")
    if isinstance(resp, dict):
        return resp.get("data", [])
    return resp


def check_9_violations_by_project() -> None:
    """'Violations by Project' bar chart question in collection."""
    try:
        col_id = _find_collection_id()
        if not col_id:
            check("9. Violations by Project question", 2, False, "collection not found")
            return

        cards = _get_collection_cards(col_id)
        found = next((c for c in cards if c.get("name") == "Violations by Project"), None)
        if not found:
            check("9. Violations by Project question", 2, False, "question not found")
            return

        detail = metabase_get(f"/api/card/{found['id']}")
        display = detail.get("display", "")
        is_bar = display == "bar"
        check("9. Violations by Project question", 2, is_bar,
              f"display={display}")
    except Exception as e:
        check("9. Violations by Project question", 2, False, f"exception: {e}")


def check_10_rule_frequency() -> None:
    """'Rule Frequency' table question (top 10) in collection."""
    try:
        col_id = _find_collection_id()
        if not col_id:
            check("10. Rule Frequency question", 2, False, "collection not found")
            return

        cards = _get_collection_cards(col_id)
        found = next((c for c in cards if c.get("name") == "Rule Frequency"), None)
        if not found:
            check("10. Rule Frequency question", 2, False, "question not found")
            return

        detail = metabase_get(f"/api/card/{found['id']}")
        display = detail.get("display", "")
        is_table = display == "table"

        # Check for limit 10 in structured or native query
        dq = detail.get("dataset_query", {})
        limit = dq.get("query", {}).get("limit")
        has_limit = limit == 10
        if not has_limit and "native" in dq:
            native_q = dq["native"].get("query", "")
            has_limit = bool(re.search(r"(?i)\blimit\s+10\b", native_q))

        check("10. Rule Frequency question", 2, is_table and has_limit,
              f"display={display}, limit={limit if limit else 'native-check=' + str(has_limit)}")
    except Exception as e:
        check("10. Rule Frequency question", 2, False, f"exception: {e}")


def check_11_metabase_dashboard() -> None:
    """Dashboard 'Code Quality Audit Dashboard' with correct description in collection."""
    try:
        col_id = _find_collection_id()
        if not col_id:
            check("11. Metabase dashboard", 2, False, "collection not found")
            return

        dashboards = metabase_get("/api/dashboard")
        found = next(
            (d for d in dashboards if d.get("name") == "Code Quality Audit Dashboard"),
            None,
        )
        if not found:
            check("11. Metabase dashboard", 2, False, "dashboard not found")
            return

        dash = metabase_get(f"/api/dashboard/{found['id']}")
        desc = dash.get("description", "") or ""
        expected_desc = "Lint audit 2025-05-14 across 4 projects"
        desc_ok = desc.strip() == expected_desc
        in_col = dash.get("collection_id") == col_id

        check("11. Metabase dashboard", 2, desc_ok and in_col,
              f"desc='{desc}', in_collection={in_col}")
    except Exception as e:
        check("11. Metabase dashboard", 2, False, f"exception: {e}")


def check_12_dashboard_cards() -> None:
    """Dashboard contains both question cards."""
    try:
        dashboards = metabase_get("/api/dashboard")
        found = next(
            (d for d in dashboards if d.get("name") == "Code Quality Audit Dashboard"),
            None,
        )
        if not found:
            check("12. Dashboard has both cards", 1, False, "dashboard not found")
            return

        dash = metabase_get(f"/api/dashboard/{found['id']}")
        dc = dash.get("dashcards", dash.get("ordered_cards", []))
        card_names = []
        for c in dc:
            card = c.get("card", {})
            if card and card.get("name"):
                card_names.append(card["name"])

        has_v = "Violations by Project" in card_names
        has_r = "Rule Frequency" in card_names
        check("12. Dashboard has both cards", 1, has_v and has_r,
              f"cards={card_names}")
    except Exception as e:
        check("12. Dashboard has both cards", 1, False, f"exception: {e}")


def check_13_openproject_work_packages() -> None:
    """'Security Audit' project has exactly 5 Task-type work packages."""
    try:
        project_id = openproject_db_query(
            "SELECT id FROM projects WHERE name = 'Security Audit'"
        ).strip()
        if not project_id:
            check("13. OpenProject 5 Task WPs", 2, False, "project not found")
            return

        task_type_id = openproject_db_query(
            "SELECT id FROM types WHERE name = 'Task'"
        ).strip()
        if not task_type_id:
            check("13. OpenProject 5 Task WPs", 2, False, "Task type not found")
            return

        count = openproject_db_query(
            f"SELECT count(*) FROM work_packages "
            f"WHERE project_id = {project_id} AND type_id = {task_type_id}"
        ).strip()
        check("13. OpenProject 5 Task WPs", 2, count == "5",
              f"count={count}")
    except Exception as e:
        check("13. OpenProject 5 Task WPs", 2, False, f"exception: {e}")


def check_14_wp_details() -> None:
    """Work packages have correct subject, assignee (admin), priority (High), description."""
    try:
        project_id = openproject_db_query(
            "SELECT id FROM projects WHERE name = 'Security Audit'"
        ).strip()
        task_type_id = openproject_db_query(
            "SELECT id FROM types WHERE name = 'Task'"
        ).strip()
        admin_id = openproject_db_query(
            "SELECT id FROM users WHERE login = 'admin'"
        ).strip()
        high_pri_id = openproject_db_query(
            "SELECT id FROM enumerations WHERE name = 'High' AND type = 'IssuePriority'"
        ).strip()

        if not all([project_id, task_type_id, admin_id, high_pri_id]):
            check("14. WP details correct", 3, False,
                  f"proj={project_id}, type={task_type_id}, admin={admin_id}, pri={high_pri_id}")
            return

        rows_raw = openproject_db_query(
            f"SELECT subject || '|||' || COALESCE(assigned_to_id::text, 'NULL') "
            f"|| '|||' || COALESCE(priority_id::text, 'NULL') "
            f"|| '|||' || COALESCE(description, '') "
            f"FROM work_packages "
            f"WHERE project_id = {project_id} AND type_id = {task_type_id} "
            f"ORDER BY subject"
        )
        if not rows_raw:
            check("14. WP details correct", 3, False, "no work packages found")
            return

        issues: list[str] = []
        for line in rows_raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|||")
            if len(parts) < 4:
                issues.append(f"parse error: {line[:80]}")
                continue
            subject = parts[0].strip()
            assignee_id = parts[1].strip()
            priority_id = parts[2].strip()
            desc = parts[3].strip()

            if not re.match(r"^Fix lint errors: .+ \(\d+ errors?\)$", subject):
                issues.append(f"bad subject format: {subject}")
            if assignee_id != admin_id:
                issues.append(f"wrong assignee for '{subject[:40]}': {assignee_id}")
            if priority_id != high_pri_id:
                issues.append(f"wrong priority for '{subject[:40]}': {priority_id}")
            if not re.search(r"Project: .+;\s*Top rules: .+", desc):
                issues.append(f"bad description for '{subject[:40]}'")

        check("14. WP details correct", 3, not issues,
              "all correct" if not issues else "; ".join(issues[:3]))
    except Exception as e:
        check("14. WP details correct", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_csv_exists()
    check_2_csv_columns()
    check_3_baserow_database()
    check_4_baserow_table_rows()
    check_5_violation_ids()
    check_6_captured_at()
    check_7_top_offenders_view()
    check_8_metabase_collection()
    check_9_violations_by_project()
    check_10_rule_frequency()
    check_11_metabase_dashboard()
    check_12_dashboard_cards()
    check_13_openproject_work_packages()
    check_14_wp_details()

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
