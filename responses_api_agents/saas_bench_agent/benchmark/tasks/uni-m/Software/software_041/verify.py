# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_041/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-041-I4: QA Regression Suite Registry for Sprint-2026-06

Checks: 16 weighted checks across code-server, baserow, openproject.
Strategy: docker exec (code-server filesystem), Baserow REST API, OpenProject embedded DB.

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import re
import sys
import subprocess
import json

try:
    import requests
except ImportError:
    print("FATAL: requests library not available", file=sys.stderr)
    sys.exit(1)

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_PORT = os.environ.get("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
OPENPROJECT_PORT = os.environ.get("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

_required = {
    "CODE_SERVER_PORT": CODE_SERVER_PORT,
    "CODE_SERVER_CONTAINER": CODE_SERVER_CONTAINER,
    "BASEROW_PORT": BASEROW_PORT,
    "BASEROW_CONTAINER": BASEROW_CONTAINER,
    "BASEROW_DB_CONTAINER": BASEROW_DB_CONTAINER,
    "OPENPROJECT_PORT": OPENPROJECT_PORT,
    "OPENPROJECT_CONTAINER": OPENPROJECT_CONTAINER,
}
for _var, _val in _required.items():
    if not _val:
        print(f"FATAL: {_var} not set", file=sys.stderr)
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


def op_db_query(sql: str, timeout: int = 15) -> str:
    """Query OpenProject's embedded PostgreSQL."""
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "psql", "-U", "openproject", "-d", "openproject", "-t", "-A", "-c", sql,
        timeout=timeout,
    )
    return out.strip()


def baserow_auth() -> dict:
    """Authenticate to Baserow API and return auth headers."""
    resp = requests.post(f"{BASEROW_URL}/api/user/token-auth/", json={
        "email": "admin@example.com",
        "password": "Admin1234",
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token") or data.get("access_token", "")
    # Try JWT prefix (standard Baserow), fall back to Token
    headers = {"Authorization": f"JWT {token}"}
    test = requests.get(f"{BASEROW_URL}/api/applications/", headers=headers, timeout=10)
    if test.status_code == 401:
        headers = {"Authorization": f"Token {token}"}
    return headers


def baserow_get(path: str, headers: dict, params: dict = None) -> dict:
    resp = requests.get(f"{BASEROW_URL}{path}", headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── Shared Baserow state ─────────────────────────────────────────────────────
_br_headers = None
_test_cases_table_id = None
_summary_table_id = None
_db_id = None
_test_cases_fields: dict = {}
_summary_fields: dict = {}
_test_cases_rows: list = []
_summary_rows: list = []


def _init_baserow():
    """Load Baserow database, tables, fields, and rows into module state."""
    global _br_headers, _test_cases_table_id, _summary_table_id, _db_id
    global _test_cases_fields, _summary_fields, _test_cases_rows, _summary_rows

    _br_headers = baserow_auth()

    # Find database
    apps = baserow_get("/api/applications/", _br_headers)
    for app in apps:
        if app.get("name") == "QA Regression Registry 2026-06":
            _db_id = app["id"]
            break

    if not _db_id:
        return

    # Find tables
    tables = baserow_get(f"/api/database/tables/database/{_db_id}/", _br_headers)
    for t in tables:
        if t["name"] == "Test Cases":
            _test_cases_table_id = t["id"]
        elif t["name"] == "Project Run Summary":
            _summary_table_id = t["id"]

    # Load fields and rows for Test Cases
    if _test_cases_table_id:
        fields = baserow_get(f"/api/database/fields/table/{_test_cases_table_id}/", _br_headers)
        _test_cases_fields = {f["name"]: f for f in fields}
        page = 1
        while True:
            data = baserow_get(
                f"/api/database/rows/table/{_test_cases_table_id}/",
                _br_headers, params={"size": 200, "page": page},
            )
            _test_cases_rows.extend(data.get("results", []))
            if not data.get("next"):
                break
            page += 1

    # Load fields and rows for Project Run Summary
    if _summary_table_id:
        fields = baserow_get(f"/api/database/fields/table/{_summary_table_id}/", _br_headers)
        _summary_fields = {f["name"]: f for f in fields}
        data = baserow_get(
            f"/api/database/rows/table/{_summary_table_id}/",
            _br_headers, params={"size": 200},
        )
        _summary_rows.extend(data.get("results", []))


def _get_field_value(row: dict, field_name: str, fields_map: dict):
    """Get the value of a named field from a Baserow row."""
    field = fields_map.get(field_name)
    if not field:
        return None
    field_key = f"field_{field['id']}"
    val = row.get(field_key)
    if isinstance(val, dict) and "value" in val:
        return val["value"]
    return val


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_baserow_db_exists():
    """Baserow database 'QA Regression Registry 2026-06' exists."""
    try:
        _init_baserow()
        check("1. Baserow DB 'QA Regression Registry 2026-06' exists", 1,
              _db_id is not None,
              "" if _db_id else "database not found")
    except Exception as e:
        check("1. Baserow DB 'QA Regression Registry 2026-06' exists", 1, False, f"exception: {e}")


def check_2_test_cases_table():
    """Test Cases table exists in the database."""
    try:
        check("2. 'Test Cases' table exists", 1,
              _test_cases_table_id is not None,
              "" if _test_cases_table_id else "table not found")
    except Exception as e:
        check("2. 'Test Cases' table exists", 1, False, f"exception: {e}")


def check_3_summary_table():
    """Project Run Summary table exists."""
    try:
        check("3. 'Project Run Summary' table exists", 1,
              _summary_table_id is not None,
              "" if _summary_table_id else "table not found")
    except Exception as e:
        check("3. 'Project Run Summary' table exists", 1, False, f"exception: {e}")


def check_4_test_cases_projects():
    """Test Cases has rows for both projects with TC-NNNN IDs."""
    try:
        if not _test_cases_rows:
            check("4. Test Cases rows with correct projects and TC-IDs", 2, False, "no rows found")
            return

        projects_found = set()
        tc_ids = []
        for row in _test_cases_rows:
            proj = _get_field_value(row, "Project", _test_cases_fields)
            tc_id = _get_field_value(row, "Test ID", _test_cases_fields)
            if proj:
                projects_found.add(str(proj))
            if tc_id:
                tc_ids.append(str(tc_id))

        has_both = "tabler" in projects_found and "json" in projects_found
        tc_format_ok = all(re.match(r"^TC-\d{4}$", tid) for tid in tc_ids) if tc_ids else False

        passed = has_both and tc_format_ok and len(_test_cases_rows) > 0
        detail = f"{len(_test_cases_rows)} rows, projects={projects_found}"
        if not tc_format_ok:
            detail += ", TC-ID format issues"
        check("4. Test Cases rows with correct projects and TC-IDs", 2, passed, detail)
    except Exception as e:
        check("4. Test Cases rows with correct projects and TC-IDs", 2, False, f"exception: {e}")


def check_5_suite_category():
    """Suite Category correctly assigned based on prefix mapping."""
    try:
        if not _test_cases_rows:
            check("5. Suite Category assignment correct", 2, False, "no rows")
            return

        prefix_map = [
            ("tests/src/integration-", "Integration"),
            ("tests/src/regression-", "Regression"),
            ("tests/src/unit-", "Unit"),
            ("preview/", "Smoke"),
            ("core/", "Unit"),
        ]

        errors = 0
        sample_error = ""
        for row in _test_cases_rows:
            test_file = str(_get_field_value(row, "Test File", _test_cases_fields) or "")
            category = str(_get_field_value(row, "Suite Category", _test_cases_fields) or "")

            expected_cat = "Unit"  # default
            for prefix, cat in prefix_map:
                if test_file.startswith(prefix):
                    expected_cat = cat
                    break

            if category != expected_cat:
                errors += 1
                if not sample_error:
                    sample_error = f"file={test_file}, expected={expected_cat}, got={category}"

        passed = errors == 0
        detail = f"{errors} mismatches" + (f"; first: {sample_error}" if sample_error else "")
        check("5. Suite Category assignment correct", 2, passed, detail)
    except Exception as e:
        check("5. Suite Category assignment correct", 2, False, f"exception: {e}")


def check_6_flaky_flags():
    """Flaky flags set correctly for the 3 known flaky tests."""
    try:
        if not _test_cases_rows:
            check("6. Flaky flags correct", 2, False, "no rows")
            return

        known_flaky = {
            "tabler::test_modal_focus_trap",
            "json::test_bson_roundtrip_large",
            "tabler::test_tooltip_positioning",
        }

        errors = 0
        sample_error = ""
        for row in _test_cases_rows:
            proj = str(_get_field_value(row, "Project", _test_cases_fields) or "")
            test_name = str(_get_field_value(row, "Test Name", _test_cases_fields) or "")
            flaky = _get_field_value(row, "Flaky", _test_cases_fields)

            key = f"{proj}::{test_name}"
            expected_flaky = key in known_flaky
            actual_flaky = bool(flaky)

            if actual_flaky != expected_flaky:
                errors += 1
                if not sample_error:
                    sample_error = f"{key}: expected={expected_flaky}, got={actual_flaky}"

        passed = errors == 0
        detail = f"{errors} mismatches" + (f"; first: {sample_error}" if sample_error else "")
        check("6. Flaky flags correct", 2, passed, detail)
    except Exception as e:
        check("6. Flaky flags correct", 2, False, f"exception: {e}")


def check_7_last_run_date():
    """Last Run = 2026-06-10 for all test case rows."""
    try:
        if not _test_cases_rows:
            check("7. Last Run date correct", 1, False, "no rows")
            return

        errors = 0
        for row in _test_cases_rows:
            last_run = _get_field_value(row, "Last Run", _test_cases_fields)
            if last_run and isinstance(last_run, str):
                if not last_run.startswith("2026-06-10"):
                    errors += 1
            else:
                errors += 1

        passed = errors == 0
        detail = f"{errors}/{len(_test_cases_rows)} rows with wrong date" if errors else ""
        check("7. Last Run date correct", 1, passed, detail)
    except Exception as e:
        check("7. Last Run date correct", 1, False, f"exception: {e}")


def check_8_summary_values():
    """Project Run Summary has correct computed values (Pass Rate, Verdict)."""
    try:
        if not _summary_rows:
            check("8. Project Run Summary computed values", 2, False, "no summary rows")
            return

        if not _test_cases_rows:
            check("8. Project Run Summary computed values", 2, False, "no test case rows to verify against")
            return

        # Compute expected values from test case rows
        project_stats: dict = {}
        for row in _test_cases_rows:
            proj = str(_get_field_value(row, "Project", _test_cases_fields) or "")
            status = str(_get_field_value(row, "Status", _test_cases_fields) or "")
            flaky = bool(_get_field_value(row, "Flaky", _test_cases_fields))

            if proj not in project_stats:
                project_stats[proj] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "flaky": 0}
            project_stats[proj]["total"] += 1
            if status == "Pass":
                project_stats[proj]["passed"] += 1
            elif status == "Fail":
                project_stats[proj]["failed"] += 1
            elif status == "Skipped":
                project_stats[proj]["skipped"] += 1
            if flaky:
                project_stats[proj]["flaky"] += 1

        errors = 0
        sample_error = ""
        summary_projects = set()
        for srow in _summary_rows:
            proj = str(_get_field_value(srow, "Project", _summary_fields) or "")
            summary_projects.add(proj)
            total = _get_field_value(srow, "Total Tests", _summary_fields)
            pass_rate = _get_field_value(srow, "Pass Rate Pct", _summary_fields)
            verdict = str(_get_field_value(srow, "Run Verdict", _summary_fields) or "")

            if proj not in project_stats:
                errors += 1
                if not sample_error:
                    sample_error = f"unknown project '{proj}'"
                continue

            exp = project_stats[proj]

            # Check total
            try:
                total_int = int(float(str(total))) if total is not None else -1
            except (ValueError, TypeError):
                total_int = -1
            if total_int != exp["total"]:
                errors += 1
                if not sample_error:
                    sample_error = f"{proj}: total expected={exp['total']}, got={total}"

            # Check pass rate
            if exp["total"] > 0:
                expected_rate = round(exp["passed"] / exp["total"] * 100, 2)
                try:
                    actual_rate = float(str(pass_rate)) if pass_rate is not None else -1
                except (ValueError, TypeError):
                    actual_rate = -1
                if abs(actual_rate - expected_rate) > 0.05:
                    errors += 1
                    if not sample_error:
                        sample_error = f"{proj}: pass_rate expected={expected_rate}, got={actual_rate}"

            # Check verdict
            if exp["total"] > 0:
                expected_rate_val = round(exp["passed"] / exp["total"] * 100, 2)
                flaky_count = exp["flaky"]
                if expected_rate_val >= 96.50 and flaky_count <= 3:
                    expected_verdict = "Green"
                elif expected_rate_val < 85.00:
                    expected_verdict = "Red"
                else:
                    expected_verdict = "Yellow"

                if verdict != expected_verdict:
                    errors += 1
                    if not sample_error:
                        sample_error = f"{proj}: verdict expected={expected_verdict}, got={verdict}"

        if "tabler" not in summary_projects or "json" not in summary_projects:
            errors += 1
            if not sample_error:
                sample_error = f"missing projects, found: {summary_projects}"

        passed = errors == 0
        detail = f"{errors} issues" + (f"; first: {sample_error}" if sample_error else "")
        check("8. Project Run Summary computed values", 2, passed, detail)
    except Exception as e:
        check("8. Project Run Summary computed values", 2, False, f"exception: {e}")


def check_9_failures_only_view():
    """'Failures Only' view exists on Test Cases."""
    try:
        if not _test_cases_table_id or not _br_headers:
            check("9. 'Failures Only' view exists", 1, False, "no Test Cases table or auth")
            return

        views = baserow_get(f"/api/database/views/table/{_test_cases_table_id}/", _br_headers)
        view_names = [v["name"] for v in views]
        found = "Failures Only" in view_names
        check("9. 'Failures Only' view exists", 1, found,
              "" if found else f"views found: {view_names}")
    except Exception as e:
        check("9. 'Failures Only' view exists", 1, False, f"exception: {e}")


def check_10_flaky_tests_view():
    """'Flaky Tests' view exists on Test Cases."""
    try:
        if not _test_cases_table_id or not _br_headers:
            check("10. 'Flaky Tests' view exists", 1, False, "no Test Cases table or auth")
            return

        views = baserow_get(f"/api/database/views/table/{_test_cases_table_id}/", _br_headers)
        view_names = [v["name"] for v in views]
        found = "Flaky Tests" in view_names
        check("10. 'Flaky Tests' view exists", 1, found,
              "" if found else f"views found: {view_names}")
    except Exception as e:
        check("10. 'Flaky Tests' view exists", 1, False, f"exception: {e}")


def check_11_qa_report_exists():
    """File devops-configs/docs/qa-report-2026-06.md exists in code-server."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "find", "/home", "-name", "qa-report-2026-06.md", "-type", "f",
            timeout=10,
        )
        found = bool(out.strip())
        check("11. qa-report-2026-06.md exists", 1, found,
              f"found at: {out.strip().split(chr(10))[0]}" if found else "file not found")
    except Exception as e:
        check("11. qa-report-2026-06.md exists", 1, False, f"exception: {e}")


def check_12_qa_report_content():
    """qa-report-2026-06.md has correct header and content format."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "find", "/home", "-name", "qa-report-2026-06.md", "-type", "f",
            timeout=10,
        )
        filepath = out.strip().split("\n")[0] if out.strip() else ""

        if not filepath:
            check("12. qa-report content correct", 2, False, "file not found")
            return

        rc, content, err = docker_exec(CODE_SERVER_CONTAINER, "cat", filepath)
        lines = content.strip().split("\n")

        errors = []
        if len(lines) < 6:
            errors.append(f"only {len(lines)} lines, expected >= 6")
        else:
            if lines[0].strip() != "# QA Regression Report: Sprint-2026-06":
                errors.append(f"line 1 mismatch: '{lines[0].strip()[:60]}'")
            if lines[1].strip() != "Run Date: 2026-06-10":
                errors.append(f"line 2 mismatch: '{lines[1].strip()[:60]}'")
            if not re.match(r"Total tests: \d+; Passed: \d+; Failed: \d+; Skipped: \d+", lines[2].strip()):
                errors.append(f"line 3 format wrong: '{lines[2].strip()[:60]}'")
            if not re.match(r"Flaky count: \d+", lines[3].strip()):
                errors.append(f"line 4 format wrong: '{lines[3].strip()[:60]}'")
            project_lines = [l for l in lines[4:] if l.strip().startswith("- ")]
            if len(project_lines) < 2:
                errors.append(f"expected 2 project lines, found {len(project_lines)}")
            else:
                for pl in project_lines:
                    if not re.match(r"- \w+: [\d.]+% \(\w+\)", pl.strip()):
                        errors.append(f"project line format wrong: '{pl.strip()[:60]}'")
                        break

        passed = len(errors) == 0
        detail = "; ".join(errors) if errors else ""
        check("12. qa-report content correct", 2, passed, detail)
    except Exception as e:
        check("12. qa-report content correct", 2, False, f"exception: {e}")


def check_13_op_epic_exists():
    """OpenProject Epic 'QA Regression: Sprint-2026-06' exists in demo-project with correct assignee."""
    try:
        sql = """
        SELECT wp.id, u.login AS assignee
        FROM work_packages wp
        JOIN projects p ON wp.project_id = p.id
        JOIN types t ON wp.type_id = t.id
        LEFT JOIN users u ON wp.assigned_to_id = u.id
        WHERE p.identifier = 'demo-project'
          AND t.name = 'Epic'
          AND wp.subject = 'QA Regression: Sprint-2026-06'
        """
        result = op_db_query(sql)

        if not result:
            check("13. Epic 'QA Regression: Sprint-2026-06' exists", 2, False, "not found")
            return

        parts = result.split("\n")[0].split("|")
        epic_id = parts[0].strip() if parts else ""
        assignee = parts[1].strip() if len(parts) > 1 else ""

        assignee_ok = assignee == "qa.manager"
        passed = bool(epic_id) and assignee_ok
        detail = ""
        if not assignee_ok:
            detail = f"assignee='{assignee}', expected='qa.manager'"
        check("13. Epic 'QA Regression: Sprint-2026-06' exists", 2, passed, detail)
    except Exception as e:
        check("13. Epic 'QA Regression: Sprint-2026-06' exists", 2, False, f"exception: {e}")


def check_14_op_epic_description():
    """Epic description has correct format with computed totals."""
    try:
        sql = """
        SELECT wp.description
        FROM work_packages wp
        JOIN projects p ON wp.project_id = p.id
        JOIN types t ON wp.type_id = t.id
        WHERE p.identifier = 'demo-project'
          AND t.name = 'Epic'
          AND wp.subject = 'QA Regression: Sprint-2026-06'
        """
        result = op_db_query(sql)

        if not result:
            check("14. Epic description correct", 2, False, "epic not found")
            return

        desc = result.strip()
        errors = []

        if "Run Date: 2026-06-10" not in desc:
            errors.append("missing 'Run Date: 2026-06-10'")
        if "devops-configs/docs/qa-report-2026-06.md" not in desc:
            errors.append("missing report path")
        if not re.search(r"Total: \d+", desc):
            errors.append("missing 'Total: <N>'")
        if not re.search(r"Pass Rate: [\d.]+%", desc):
            errors.append("missing 'Pass Rate: <N>%'")

        check("14. Epic description correct", 2, len(errors) == 0,
              "; ".join(errors) if errors else "")
    except Exception as e:
        check("14. Epic description correct", 2, False, f"exception: {e}")


def check_15_op_child_bugs():
    """Child Bug work packages exist under the Epic for each failed test."""
    try:
        epic_id = op_db_query("""
        SELECT wp.id
        FROM work_packages wp
        JOIN projects p ON wp.project_id = p.id
        JOIN types t ON wp.type_id = t.id
        WHERE p.identifier = 'demo-project'
          AND t.name = 'Epic'
          AND wp.subject = 'QA Regression: Sprint-2026-06'
        """).strip()

        if not epic_id:
            check("15. Child Bug work packages exist", 2, False, "epic not found")
            return

        # Count failed tests from Baserow data
        failed_count = 0
        for row in _test_cases_rows:
            status = str(_get_field_value(row, "Status", _test_cases_fields) or "")
            if status == "Fail":
                failed_count += 1

        bug_count_str = op_db_query(f"""
        SELECT COUNT(*)
        FROM work_packages wp
        JOIN types t ON wp.type_id = t.id
        WHERE wp.parent_id = {epic_id}
          AND t.name = 'Bug'
        """).strip()

        try:
            bug_count = int(bug_count_str)
        except ValueError:
            bug_count = 0

        if failed_count == 0:
            check("15. Child Bug work packages exist", 2, bug_count > 0,
                  f"found {bug_count} bugs (no Baserow test data to cross-check count)")
        else:
            passed = bug_count == failed_count
            check("15. Child Bug work packages exist", 2, passed,
                  f"expected {failed_count} bugs, found {bug_count}")
    except Exception as e:
        check("15. Child Bug work packages exist", 2, False, f"exception: {e}")


def check_16_op_bug_priorities():
    """Bug priorities correct: High when not flaky, Normal when flaky."""
    try:
        epic_id = op_db_query("""
        SELECT wp.id
        FROM work_packages wp
        JOIN projects p ON wp.project_id = p.id
        JOIN types t ON wp.type_id = t.id
        WHERE p.identifier = 'demo-project'
          AND t.name = 'Epic'
          AND wp.subject = 'QA Regression: Sprint-2026-06'
        """).strip()

        if not epic_id:
            check("16. Bug priorities correct", 2, False, "epic not found")
            return

        result = op_db_query(f"""
        SELECT wp.subject, e.name AS priority
        FROM work_packages wp
        JOIN types t ON wp.type_id = t.id
        JOIN enumerations e ON wp.priority_id = e.id
        WHERE wp.parent_id = {epic_id}
          AND t.name = 'Bug'
        """)

        if not result:
            check("16. Bug priorities correct", 2, False, "no child bugs found")
            return

        known_flaky_tests = {
            "test_modal_focus_trap",
            "test_bson_roundtrip_large",
            "test_tooltip_positioning",
        }

        errors = 0
        sample_error = ""
        for line in result.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            subject = parts[0].strip()
            priority = parts[1].strip()

            match = re.search(r"Test fail: (.+)$", subject)
            test_name = match.group(1).strip() if match else ""

            is_flaky = test_name in known_flaky_tests
            expected_priority = "Normal" if is_flaky else "High"

            if priority != expected_priority:
                errors += 1
                if not sample_error:
                    sample_error = f"'{subject}': expected={expected_priority}, got={priority}"

        passed = errors == 0
        detail = f"{errors} wrong priorities" + (f"; first: {sample_error}" if sample_error else "")
        check("16. Bug priorities correct", 2, passed, detail)
    except Exception as e:
        check("16. Bug priorities correct", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_db_exists()
    check_2_test_cases_table()
    check_3_summary_table()
    check_4_test_cases_projects()
    check_5_suite_category()
    check_6_flaky_flags()
    check_7_last_run_date()
    check_8_summary_values()
    check_9_failures_only_view()
    check_10_flaky_tests_view()
    check_11_qa_report_exists()
    check_12_qa_report_content()
    check_13_op_epic_exists()
    check_14_op_epic_description()
    check_15_op_child_bugs()
    check_16_op_bug_priorities()

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
