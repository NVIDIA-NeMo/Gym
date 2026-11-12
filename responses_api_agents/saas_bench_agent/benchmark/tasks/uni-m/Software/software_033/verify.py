# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_033/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-033-I3: Audit devops-configs CI workflow and track missing stages

Checks: 12 weighted checks across code-server, baserow, openproject.
Strategy: docker exec (filesystem + DB) for code-server and openproject;
          Baserow REST API for baserow (dynamic table schema).

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import json
import os
import re
import subprocess
import sys

import requests

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
OPENPROJECT_URL = f"http://{HOST}:{OPENPROJECT_PORT}"

# Slot values
CI_WORKFLOW_PATH = "devops-configs/.github/workflows/deploy.yml"
OLD_RUNNER = "ubuntu-18.04"
NEW_RUNNER = "ubuntu-22.04"
BASEROW_DB_NAME = "CI Workflow Remediation Tracker"
COMMIT_MSG = "ci: upgrade runner to ubuntu-22.04"
STAGE_MAP = {
    "docker-build": "Build", "npm-build": "Build",
    "jest": "Test", "e2e": "Test",
    "prettier": "Lint", "tflint": "Lint",
    "deploy-staging": "Deploy", "deploy-prod": "Deploy",
    "notify": "Other",
}
REQUIRED_STAGES = ["Build", "Test", "Lint", "Deploy"]
OP_PROJECT = "devops-automation"
CI_OWNER = "Paul Harris"

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
    resp = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def baserow_get(path: str, token: str, params: dict = None) -> dict:
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


# ── Check 1: deploy.yml has no old runner ─────────────────────────────────────
def check_1_no_old_runner() -> None:
    """Verify deploy.yml contains no 'ubuntu-18.04'."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "grep", "-c", OLD_RUNNER, f"/home/coder/{CI_WORKFLOW_PATH}",
        )
        # grep -c returns 0 if matches found, 1 if no matches
        has_old = (rc == 0 and out.strip() != "0")
        check("1. deploy.yml no old runner", 1, not has_old,
              f"found {out.strip()} occurrences of '{OLD_RUNNER}'" if has_old else "")
    except Exception as e:
        check("1. deploy.yml no old runner", 1, False, f"exception: {e}")


# ── Check 2: deploy.yml has new runner ────────────────────────────────────────
def check_2_has_new_runner() -> None:
    """Verify deploy.yml contains 'ubuntu-22.04'."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "grep", "-c", NEW_RUNNER, f"/home/coder/{CI_WORKFLOW_PATH}",
        )
        count = int(out.strip()) if rc == 0 else 0
        check("2. deploy.yml has new runner", 1, count > 0,
              f"{count} occurrences of '{NEW_RUNNER}'" if count > 0 else f"'{NEW_RUNNER}' not found")
    except Exception as e:
        check("2. deploy.yml has new runner", 1, False, f"exception: {e}")


# ── Check 3: Git commit message ───────────────────────────────────────────────
def check_3_commit_message() -> None:
    """Verify a git commit with exact message 'ci: upgrade runner to ubuntu-22.04' exists."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "bash", "-c",
            f"cd /home/coder/devops-configs && git log --oneline --all --format='%s'",
        )
        messages = [m.strip() for m in out.strip().split("\n") if m.strip()]
        found = COMMIT_MSG in messages
        check("3. Git commit message exact", 2, found,
              "" if found else f"'{COMMIT_MSG}' not in git log ({len(messages)} commits found)")
    except Exception as e:
        check("3. Git commit message exact", 2, False, f"exception: {e}")


# ── Check 4: Baserow database exists ─────────────────────────────────────────
def check_4_baserow_db() -> None:
    """Verify Baserow database 'CI Workflow Remediation Tracker' exists."""
    try:
        token = baserow_auth()
        apps = baserow_get("/applications/", token)
        # apps is a list of applications
        found = any(
            app.get("name") == BASEROW_DB_NAME
            for app in apps
        )
        check("4. Baserow DB exists", 1, found,
              "" if found else f"'{BASEROW_DB_NAME}' not found among {len(apps)} applications")
    except Exception as e:
        check("4. Baserow DB exists", 1, False, f"exception: {e}")


# ── Check 5: CI Jobs table with correct fields ───────────────────────────────
def _get_ci_jobs_table(token: str):
    """Find the CI Jobs table and return (table_id, fields_dict)."""
    apps = baserow_get("/applications/", token)
    for app in apps:
        if app.get("name") == BASEROW_DB_NAME:
            tables = app.get("tables", [])
            for tbl in tables:
                if tbl.get("name") == "CI Jobs":
                    table_id = tbl["id"]
                    fields_resp = baserow_get(f"/database/fields/table/{table_id}/", token)
                    fields = {f["name"]: f for f in fields_resp}
                    return table_id, fields
    return None, {}


def check_5_table_fields() -> None:
    """Verify CI Jobs table has required fields with correct types."""
    try:
        token = baserow_auth()
        table_id, fields = _get_ci_jobs_table(token)
        if table_id is None:
            check("5. CI Jobs table + fields", 2, False, "table 'CI Jobs' not found")
            return

        expected_fields = {
            "Job Name": "text",
            "Runs On": "text",
            "Has Dependencies": "boolean",
            "Stage Category": "single_select",
            "Missing Stage": "boolean",
        }
        missing = []
        wrong_type = []
        for fname, ftype in expected_fields.items():
            if fname not in fields:
                missing.append(fname)
            elif fields[fname]["type"] != ftype:
                wrong_type.append(f"{fname}: expected {ftype}, got {fields[fname]['type']}")

        ok = not missing and not wrong_type
        detail = ""
        if missing:
            detail += f"missing fields: {missing}"
        if wrong_type:
            detail += f"; wrong types: {wrong_type}" if detail else f"wrong types: {wrong_type}"
        check("5. CI Jobs table + fields", 2, ok, detail)
    except Exception as e:
        check("5. CI Jobs table + fields", 2, False, f"exception: {e}")


# ── Check 6: Job rows with CJ-NN IDs ─────────────────────────────────────────
def _get_all_rows(token: str, table_id: int) -> list[dict]:
    """Fetch all rows from a Baserow table."""
    rows = []
    page = 1
    while True:
        resp = baserow_get(
            f"/database/rows/table/{table_id}/",
            token,
            params={"page": page, "size": 200, "user_field_names": "true"},
        )
        rows.extend(resp.get("results", []))
        if resp.get("next") is None:
            break
        page += 1
    return rows


def check_6_job_id_format() -> None:
    """Verify rows have Job ID in CJ-NN format."""
    try:
        token = baserow_auth()
        table_id, fields = _get_ci_jobs_table(token)
        if table_id is None:
            check("6. Job rows with CJ-NN IDs", 2, False, "table not found")
            return

        rows = _get_all_rows(token, table_id)
        if not rows:
            check("6. Job rows with CJ-NN IDs", 2, False, "no rows found")
            return

        # The primary field is "Job ID" — it's the first field (primary text)
        # In user_field_names mode, the primary field name might be the actual field name
        # Find the primary field name
        primary_field = None
        for f in fields.values():
            if f.get("primary"):
                primary_field = f["name"]
                break
        if primary_field is None:
            # Fallback: check if "Job ID" key exists in rows
            primary_field = "Job ID"

        pattern = re.compile(r"^CJ-\d{2}$")
        valid = 0
        total = len(rows)
        for row in rows:
            val = str(row.get(primary_field, "")).strip()
            if pattern.match(val):
                valid += 1

        check("6. Job rows with CJ-NN IDs", 2, valid == total,
              f"{valid}/{total} rows have valid CJ-NN format")
    except Exception as e:
        check("6. Job rows with CJ-NN IDs", 2, False, f"exception: {e}")


# ── Check 7: Stage Category assignments ──────────────────────────────────────
def check_7_stage_categories() -> None:
    """Verify Stage Category is correctly assigned per the stage map."""
    try:
        token = baserow_auth()
        table_id, fields = _get_ci_jobs_table(token)
        if table_id is None:
            check("7. Stage Category assignments", 2, False, "table not found")
            return

        rows = _get_all_rows(token, table_id)
        non_missing = [r for r in rows if not str(r.get("Job Name", "")).startswith("MISSING:")]
        if not non_missing:
            check("7. Stage Category assignments", 2, False, "no non-MISSING rows found")
            return

        wrong = []
        for row in non_missing:
            job_name = str(row.get("Job Name", "")).strip()
            stage_cat = row.get("Stage Category")
            # Stage Category is a single_select field — value is dict with "value" key
            if isinstance(stage_cat, dict):
                stage_cat = stage_cat.get("value", "")
            elif stage_cat is None:
                stage_cat = ""
            expected = STAGE_MAP.get(job_name, "Other")
            if stage_cat != expected:
                wrong.append(f"{job_name}: expected '{expected}', got '{stage_cat}'")

        check("7. Stage Category assignments", 2, len(wrong) == 0,
              f"{len(wrong)} wrong: {'; '.join(wrong[:3])}" if wrong else f"{len(non_missing)} job rows checked")
    except Exception as e:
        check("7. Stage Category assignments", 2, False, f"exception: {e}")


# ── Check 8: MISSING stage rows ──────────────────────────────────────────────
def check_8_missing_rows() -> None:
    """Verify placeholder MISSING rows exist for uncovered required stages."""
    try:
        token = baserow_auth()
        table_id, fields = _get_ci_jobs_table(token)
        if table_id is None:
            check("8. MISSING stage placeholder rows", 2, False, "table not found")
            return

        rows = _get_all_rows(token, table_id)

        # Determine which required stages ARE covered by non-MISSING rows
        covered = set()
        for row in rows:
            job_name = str(row.get("Job Name", "")).strip()
            if job_name.startswith("MISSING:"):
                continue
            stage_cat = row.get("Stage Category")
            if isinstance(stage_cat, dict):
                stage_cat = stage_cat.get("value", "")
            if stage_cat in REQUIRED_STAGES:
                covered.add(stage_cat)

        expected_missing = set(REQUIRED_STAGES) - covered

        # Check that MISSING:<stage> rows exist for each expected missing stage
        found_missing = set()
        for row in rows:
            job_name = str(row.get("Job Name", "")).strip()
            if job_name.startswith("MISSING:"):
                stage = job_name.replace("MISSING:", "").strip()
                missing_flag = row.get("Missing Stage")
                if isinstance(missing_flag, dict):
                    missing_flag = missing_flag.get("value", False)
                if missing_flag:
                    found_missing.add(stage)

        ok = found_missing == expected_missing
        detail = ""
        if not ok:
            if expected_missing - found_missing:
                detail += f"missing MISSING rows for: {expected_missing - found_missing}"
            if found_missing - expected_missing:
                detail += f"; unexpected MISSING rows for: {found_missing - expected_missing}"
        else:
            detail = f"MISSING rows found for {found_missing}" if found_missing else "no missing stages needed"
        check("8. MISSING stage placeholder rows", 2, ok, detail)
    except Exception as e:
        check("8. MISSING stage placeholder rows", 2, False, f"exception: {e}")


# ── Check 9: Gaps view ───────────────────────────────────────────────────────
def check_9_gaps_view() -> None:
    """Verify 'Gaps' grid view exists on the CI Jobs table, filtered to Missing Stage=true."""
    try:
        token = baserow_auth()
        table_id, fields = _get_ci_jobs_table(token)
        if table_id is None:
            check("9. Gaps view exists", 2, False, "table not found")
            return

        views_resp = baserow_get(f"/database/views/table/{table_id}/", token)
        gaps_view = None
        for v in views_resp:
            if v.get("name") == "Gaps":
                gaps_view = v
                break

        if gaps_view is None:
            check("9. Gaps view exists", 2, False, "'Gaps' view not found")
            return

        # Check that view type is grid
        is_grid = gaps_view.get("type") == "grid"

        # Check filters
        view_id = gaps_view["id"]
        filters_resp = baserow_get(f"/database/views/{view_id}/filters/", token)
        # Look for a filter on the Missing Stage field
        missing_stage_field_id = None
        for f in fields.values():
            if f["name"] == "Missing Stage":
                missing_stage_field_id = f["id"]
                break

        has_filter = False
        if missing_stage_field_id:
            for flt in filters_resp:
                if flt.get("field") == missing_stage_field_id:
                    # Boolean filter: type "boolean" with value "true" or "1"
                    if str(flt.get("value", "")).lower() in ("true", "1"):
                        has_filter = True
                        break

        ok = is_grid and has_filter
        detail_parts = []
        if not is_grid:
            detail_parts.append(f"view type is '{gaps_view.get('type')}', expected 'grid'")
        if not has_filter:
            detail_parts.append("no filter on Missing Stage=true found")
        check("9. Gaps view exists", 2, ok,
              "; ".join(detail_parts) if detail_parts else "grid view with correct filter")
    except Exception as e:
        check("9. Gaps view exists", 2, False, f"exception: {e}")


# ── Check 10: OpenProject work packages exist ────────────────────────────────
def check_10_op_work_packages() -> None:
    """Verify Task work packages with subject 'Add CI stage: <Category>' exist in devops-automation."""
    try:
        # Find project id
        project_id = op_db_query(
            f"SELECT id FROM projects WHERE identifier='{OP_PROJECT}'"
        )
        if not project_id:
            check("10. OP work packages exist", 2, False,
                  f"project '{OP_PROJECT}' not found")
            return

        # Find Task type id
        task_type_id = op_db_query(
            "SELECT id FROM types WHERE LOWER(name)='task' LIMIT 1"
        )

        # Get work packages with subject matching pattern
        wps_raw = op_db_query(
            f"SELECT subject FROM work_packages "
            f"WHERE project_id={project_id} "
            f"AND subject LIKE 'Add CI stage:%'"
            + (f" AND type_id={task_type_id}" if task_type_id else "")
        )
        found_subjects = [s.strip() for s in wps_raw.split("\n") if s.strip()] if wps_raw else []

        # We need at least 1 WP per missing stage — we don't know exact count without
        # re-deriving from the file, so just check that at least one exists and they
        # match the expected pattern
        ok = len(found_subjects) > 0
        check("10. OP work packages exist", 2, ok,
              f"found {len(found_subjects)} work packages: {found_subjects}" if found_subjects
              else "no 'Add CI stage:' work packages found")
    except Exception as e:
        check("10. OP work packages exist", 2, False, f"exception: {e}")


# ── Check 11: OP assignee and priority ────────────────────────────────────────
def check_11_op_assignee_priority() -> None:
    """Verify work packages are assigned to Paul Harris with High priority."""
    try:
        project_id = op_db_query(
            f"SELECT id FROM projects WHERE identifier='{OP_PROJECT}'"
        )
        if not project_id:
            check("11. OP assignee + priority", 2, False, "project not found")
            return

        # Get user id for Paul Harris
        paul_id = op_db_query(
            "SELECT id FROM users WHERE CONCAT(firstname, ' ', lastname) = 'Paul Harris' LIMIT 1"
        )

        # Get priority id for High
        high_priority_id = op_db_query(
            "SELECT id FROM enumerations WHERE name='High' AND type='IssuePriority' LIMIT 1"
        )

        # Count WPs with correct assignee and priority
        conditions = f"project_id={project_id} AND subject LIKE 'Add CI stage:%'"
        total = int(op_db_query(f"SELECT COUNT(*) FROM work_packages WHERE {conditions}") or "0")

        if total == 0:
            check("11. OP assignee + priority", 2, False, "no work packages found")
            return

        assignee_ok = 0
        priority_ok = 0
        if paul_id:
            assignee_ok = int(op_db_query(
                f"SELECT COUNT(*) FROM work_packages WHERE {conditions} AND assigned_to_id={paul_id}"
            ) or "0")
        if high_priority_id:
            priority_ok = int(op_db_query(
                f"SELECT COUNT(*) FROM work_packages WHERE {conditions} AND priority_id={high_priority_id}"
            ) or "0")

        ok = (assignee_ok == total) and (priority_ok == total)
        detail = f"assignee correct: {assignee_ok}/{total}, priority correct: {priority_ok}/{total}"
        if not paul_id:
            detail += "; Paul Harris user not found"
        if not high_priority_id:
            detail += "; High priority not found"
        check("11. OP assignee + priority", 2, ok, detail)
    except Exception as e:
        check("11. OP assignee + priority", 2, False, f"exception: {e}")


# ── Check 12: OP description format ──────────────────────────────────────────
def check_12_op_description() -> None:
    """Verify work package descriptions match the expected format."""
    try:
        project_id = op_db_query(
            f"SELECT id FROM projects WHERE identifier='{OP_PROJECT}'"
        )
        if not project_id:
            check("12. OP description format", 2, False, "project not found")
            return

        # Fetch WPs with descriptions
        raw = op_db_query(
            f"SELECT subject || '|||' || COALESCE(description, '') "
            f"FROM work_packages WHERE project_id={project_id} "
            f"AND subject LIKE 'Add CI stage:%'"
        )
        if not raw:
            check("12. OP description format", 2, False, "no work packages found")
            return

        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        correct = 0
        total = len(lines)
        for line in lines:
            parts = line.split("|||", 1)
            if len(parts) < 2:
                continue
            subject, desc = parts
            # Extract category from subject
            m = re.match(r"Add CI stage:\s*(\w+)", subject)
            if not m:
                continue
            category = m.group(1)
            # Description should contain: "Add a job of category <Cat>"
            # and "devops-configs/.github/workflows/deploy.yml"
            if (f"Add a job of category {category}" in desc
                    and "devops-configs/.github/workflows/deploy.yml" in desc
                    and "current jobs:" in desc):
                correct += 1

        ok = correct == total and total > 0
        check("12. OP description format", 2, ok,
              f"{correct}/{total} descriptions match expected format")
    except Exception as e:
        check("12. OP description format", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_no_old_runner()
    check_2_has_new_runner()
    check_3_commit_message()
    check_4_baserow_db()
    check_5_table_fields()
    check_6_job_id_format()
    check_7_stage_categories()
    check_8_missing_rows()
    check_9_gaps_view()
    check_10_op_work_packages()
    check_11_op_assignee_priority()
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
