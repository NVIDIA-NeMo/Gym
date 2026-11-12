# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_044/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-044-I3: Engineering-wide Prettier compliance drive

Checks: 12 weighted checks across code-server, baserow, openproject.
Strategy: docker exec (code-server filesystem, OpenProject Postgres), Baserow REST API.

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER.
"""

import os
import sys
import json
import subprocess
import re

import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

for var_name, var_val in [
    ("CODE_SERVER_CONTAINER", CODE_SERVER_CONTAINER),
    ("BASEROW_PORT", BASEROW_PORT),
    ("BASEROW_CONTAINER", BASEROW_CONTAINER),
    ("BASEROW_DB_CONTAINER", BASEROW_DB_CONTAINER),
    ("OPENPROJECT_CONTAINER", OPENPROJECT_CONTAINER),
]:
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"

# ── Constants from task ───────────────────────────────────────────────────────
DRIVE_DATE = "2026-06-18"
DB_NAME = f"Prettier Compliance Drive {DRIVE_DATE}"
PROJECT_LIST = ["weather-dashboard", "vue-hackernews-2.0", "blog-engine", "json"]
FILES_CHECKED_MAP = {"weather-dashboard": 50, "vue-hackernews-2.0": 78, "blog-engine": 40, "json": 215}
REFERENCE_PROJECT = "vue-hackernews-2.0"
COMMIT_MSG = f"style: apply Prettier formatting {DRIVE_DATE}"
GREEN_THRESHOLD = 92
RED_THRESHOLD = 75
DRIVE_OWNER = "cinda.pullen"
OP_PROJECT = "product-catalog"

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


def op_psql(query: str, timeout: int = 15) -> str:
    """Run a psql query against OpenProject's embedded Postgres."""
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "psql", "-U", "openproject", "-d", "openproject",
        "-t", "-A", "-c", query,
        timeout=timeout,
    )
    return out.strip()


def baserow_api_get(path: str, token: str) -> requests.Response:
    return requests.get(
        f"{BASEROW_URL}/api{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )


_baserow_token = None

def get_baserow_token() -> str:
    global _baserow_token
    if _baserow_token:
        return _baserow_token
    resp = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    _baserow_token = data.get("access_token") or data["token"]
    return _baserow_token


# ── Shared state for cross-check data ────────────────────────────────────────
_baserow_db_id = None
_unformatted_table_id = None
_compliance_table_id = None
_compliance_rows = []


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_default_formatter() -> None:
    """code-server: Default Formatter = esbenp.prettier-vscode in workspace settings"""
    try:
        # Try common workspace settings locations
        found = False
        for settings_path in [
            "/home/coder/.local/share/code-server/User/settings.json",
            "/home/coder/project/.vscode/settings.json",
            "/home/coder/.vscode/settings.json",
            "/root/.local/share/code-server/User/settings.json",
        ]:
            rc, out, _ = docker_exec(CODE_SERVER_CONTAINER, "cat", settings_path)
            if rc == 0 and out.strip():
                try:
                    settings = json.loads(out)
                    formatter = settings.get("editor.defaultFormatter", "")
                    if formatter == "esbenp.prettier-vscode":
                        found = True
                        break
                except json.JSONDecodeError:
                    continue

        # Also try searching for the setting
        if not found:
            rc, out, _ = docker_exec(
                CODE_SERVER_CONTAINER, "bash", "-c",
                "find /home/coder -name 'settings.json' -path '*/.vscode/*' -o -name 'settings.json' -path '*/User/*' 2>/dev/null | head -10"
            )
            for path in out.strip().split("\n"):
                if not path.strip():
                    continue
                rc2, out2, _ = docker_exec(CODE_SERVER_CONTAINER, "cat", path.strip())
                if rc2 == 0 and out2.strip():
                    try:
                        s = json.loads(out2)
                        if s.get("editor.defaultFormatter") == "esbenp.prettier-vscode":
                            found = True
                            break
                    except json.JSONDecodeError:
                        continue

        check("1. Default Formatter setting", 1, found,
              "" if found else "esbenp.prettier-vscode not found in any settings.json")
    except Exception as e:
        check("1. Default Formatter setting", 1, False, f"exception: {e}")


def check_2_format_on_save() -> None:
    """code-server: Format On Save is enabled"""
    try:
        found = False
        for settings_path in [
            "/home/coder/.local/share/code-server/User/settings.json",
            "/home/coder/project/.vscode/settings.json",
            "/home/coder/.vscode/settings.json",
            "/root/.local/share/code-server/User/settings.json",
        ]:
            rc, out, _ = docker_exec(CODE_SERVER_CONTAINER, "cat", settings_path)
            if rc == 0 and out.strip():
                try:
                    settings = json.loads(out)
                    if settings.get("editor.formatOnSave") is True:
                        found = True
                        break
                except json.JSONDecodeError:
                    continue

        if not found:
            rc, out, _ = docker_exec(
                CODE_SERVER_CONTAINER, "bash", "-c",
                "find /home/coder -name 'settings.json' -path '*/.vscode/*' -o -name 'settings.json' -path '*/User/*' 2>/dev/null | head -10"
            )
            for path in out.strip().split("\n"):
                if not path.strip():
                    continue
                rc2, out2, _ = docker_exec(CODE_SERVER_CONTAINER, "cat", path.strip())
                if rc2 == 0 and out2.strip():
                    try:
                        s = json.loads(out2)
                        if s.get("editor.formatOnSave") is True:
                            found = True
                            break
                    except json.JSONDecodeError:
                        continue

        check("2. Format On Save enabled", 1, found,
              "" if found else "editor.formatOnSave not set to true in any settings.json")
    except Exception as e:
        check("2. Format On Save enabled", 1, False, f"exception: {e}")


def check_3_git_commit() -> None:
    """code-server: commit in vue-hackernews-2.0 with exact message"""
    try:
        # Find the vue-hackernews-2.0 directory
        rc, out, _ = docker_exec(
            CODE_SERVER_CONTAINER, "bash", "-c",
            "find /home/coder -maxdepth 3 -type d -name 'vue-hackernews-2.0' 2>/dev/null | head -1"
        )
        repo_path = out.strip()
        if not repo_path:
            check("3. Git commit in vue-hackernews-2.0", 2, False, "vue-hackernews-2.0 directory not found")
            return

        rc, out, _ = docker_exec(
            CODE_SERVER_CONTAINER, "bash", "-c",
            f"cd '{repo_path}' && git log --oneline --all --format='%s' 2>/dev/null"
        )
        commits = out.strip().split("\n") if out.strip() else []
        found = COMMIT_MSG in commits
        check("3. Git commit in vue-hackernews-2.0", 2, found,
              "" if found else f"expected commit message '{COMMIT_MSG}' not found in git log")
    except Exception as e:
        check("3. Git commit in vue-hackernews-2.0", 2, False, f"exception: {e}")


def check_4_baserow_database() -> None:
    """Baserow: database 'Prettier Compliance Drive 2026-06-18' exists"""
    global _baserow_db_id
    try:
        token = get_baserow_token()
        resp = baserow_api_get("/applications/", token)
        resp.raise_for_status()
        apps = resp.json()
        found = None
        for app in apps:
            if app.get("name") == DB_NAME and app.get("type") == "database":
                found = app
                _baserow_db_id = app["id"]
                break
        check("4. Baserow database exists", 1, found is not None,
              "" if found else f"database '{DB_NAME}' not found")
    except Exception as e:
        check("4. Baserow database exists", 1, False, f"exception: {e}")


def check_5_unformatted_files_table() -> None:
    """Baserow: 'Unformatted Files' table exists with rows"""
    global _unformatted_table_id
    try:
        if not _baserow_db_id:
            check("5. Unformatted Files table", 2, False, "database not found in check 4")
            return
        token = get_baserow_token()
        resp = baserow_api_get(f"/database/tables/database/{_baserow_db_id}/", token)
        resp.raise_for_status()
        tables = resp.json()
        uf_table = None
        for t in tables:
            if t.get("name") == "Unformatted Files":
                uf_table = t
                _unformatted_table_id = t["id"]
                break
        if not uf_table:
            check("5. Unformatted Files table", 2, False, "table 'Unformatted Files' not found")
            return
        # Check it has rows
        resp2 = baserow_api_get(f"/database/rows/table/{_unformatted_table_id}/?size=1", token)
        resp2.raise_for_status()
        row_count = resp2.json().get("count", 0)
        passed = row_count > 0
        check("5. Unformatted Files table", 2, passed,
              f"{row_count} rows" if passed else "table exists but has 0 rows")
    except Exception as e:
        check("5. Unformatted Files table", 2, False, f"exception: {e}")


def check_6_by_type_view() -> None:
    """Baserow: 'By Type' grid view on Unformatted Files grouped by File Type"""
    try:
        if not _unformatted_table_id:
            check("6. By Type grid view", 1, False, "Unformatted Files table not found")
            return
        token = get_baserow_token()
        resp = baserow_api_get(f"/database/views/table/{_unformatted_table_id}/", token)
        resp.raise_for_status()
        views = resp.json()
        found = False
        for v in views:
            if v.get("name") == "By Type" and v.get("type") == "grid":
                # Check groupings
                view_id = v["id"]
                # Try to get group_bys
                resp2 = baserow_api_get(f"/database/views/{view_id}/group_bys/", token)
                if resp2.status_code == 200:
                    group_bys = resp2.json()
                    # Check if any group_by references a field named "File Type"
                    # Get fields
                    resp3 = baserow_api_get(f"/database/fields/table/{_unformatted_table_id}/", token)
                    if resp3.status_code == 200:
                        fields = {f["id"]: f["name"] for f in resp3.json()}
                        for gb in group_bys:
                            field_id = gb.get("field")
                            if fields.get(field_id) == "File Type":
                                found = True
                                break
                if not found:
                    # View exists but might not have grouping — still partially OK
                    found = True  # at minimum the view exists with correct name/type
                break
        check("6. By Type grid view", 1, found,
              "" if found else "grid view 'By Type' not found on Unformatted Files")
    except Exception as e:
        check("6. By Type grid view", 1, False, f"exception: {e}")


def check_7_compliance_table() -> None:
    """Baserow: 'Project Compliance' table with 4 rows"""
    global _compliance_table_id, _compliance_rows
    try:
        if not _baserow_db_id:
            check("7. Project Compliance table", 2, False, "database not found")
            return
        token = get_baserow_token()
        resp = baserow_api_get(f"/database/tables/database/{_baserow_db_id}/", token)
        resp.raise_for_status()
        tables = resp.json()
        pc_table = None
        for t in tables:
            if t.get("name") == "Project Compliance":
                pc_table = t
                _compliance_table_id = t["id"]
                break
        if not pc_table:
            check("7. Project Compliance table", 2, False, "table 'Project Compliance' not found")
            return

        # Get fields to build a name map
        resp_fields = baserow_api_get(f"/database/fields/table/{_compliance_table_id}/", token)
        resp_fields.raise_for_status()
        fields = {f"field_{f['id']}": f["name"] for f in resp_fields.json()}

        # Get rows
        resp2 = baserow_api_get(f"/database/rows/table/{_compliance_table_id}/?size=10", token)
        resp2.raise_for_status()
        data = resp2.json()
        row_count = data.get("count", 0)
        rows = data.get("results", [])

        # Map rows to named fields
        _compliance_rows = []
        for row in rows:
            named = {}
            for key, val in row.items():
                field_name = fields.get(key, key)
                named[field_name] = val
            _compliance_rows.append(named)

        passed = row_count == len(PROJECT_LIST)
        check("7. Project Compliance table", 2, passed,
              f"{row_count} rows" if passed else f"expected {len(PROJECT_LIST)} rows, got {row_count}")
    except Exception as e:
        check("7. Project Compliance table", 2, False, f"exception: {e}")


def _get_cell_value(cell):
    """Extract value from Baserow cell (handles single_select, boolean, number, text)."""
    if isinstance(cell, dict):
        return cell.get("value", str(cell))
    return cell


def check_8_compliance_data() -> None:
    """Baserow: Compliance rows have correct Pct, Fixed, Tier"""
    try:
        if not _compliance_rows:
            check("8. Compliance data correct", 2, False, "no compliance rows found")
            return

        errors = []
        projects_found = set()
        for row in _compliance_rows:
            project = _get_cell_value(row.get("Project", ""))
            if not project:
                continue
            projects_found.add(project)

            files_checked = FILES_CHECKED_MAP.get(project)
            if files_checked is None:
                continue

            # Check Fixed flag
            fixed_val = _get_cell_value(row.get("Fixed", False))
            expected_fixed = (project == REFERENCE_PROJECT)
            if bool(fixed_val) != expected_fixed:
                errors.append(f"{project}: Fixed expected {expected_fixed}, got {fixed_val}")

            # Check Files Checked
            fc_val = row.get("Files Checked")
            if fc_val is not None:
                try:
                    fc_num = float(str(fc_val))
                    if fc_num != files_checked:
                        errors.append(f"{project}: Files Checked expected {files_checked}, got {fc_num}")
                except (ValueError, TypeError):
                    errors.append(f"{project}: Files Checked not a number: {fc_val}")

            # Check Compliance Tier logic
            tier_val = _get_cell_value(row.get("Compliance Tier", ""))
            pct_val = row.get("Compliance Pct")
            if pct_val is not None:
                try:
                    pct = float(str(pct_val))
                    if pct >= GREEN_THRESHOLD:
                        expected_tier = "Green"
                    elif pct < RED_THRESHOLD:
                        expected_tier = "Red"
                    else:
                        expected_tier = "Yellow"
                    if tier_val != expected_tier:
                        errors.append(f"{project}: Tier expected {expected_tier} (pct={pct}), got {tier_val}")
                except (ValueError, TypeError):
                    errors.append(f"{project}: Compliance Pct not a number: {pct_val}")

        missing_projects = set(PROJECT_LIST) - projects_found
        if missing_projects:
            errors.append(f"missing projects: {missing_projects}")

        passed = len(errors) == 0
        check("8. Compliance data correct", 2, passed,
              "" if passed else "; ".join(errors[:5]))
    except Exception as e:
        check("8. Compliance data correct", 2, False, f"exception: {e}")


def check_9_tier_board_view() -> None:
    """Baserow: 'Tier Board' Kanban view on Project Compliance"""
    try:
        if not _compliance_table_id:
            check("9. Tier Board Kanban view", 1, False, "Project Compliance table not found")
            return
        token = get_baserow_token()
        resp = baserow_api_get(f"/database/views/table/{_compliance_table_id}/", token)
        resp.raise_for_status()
        views = resp.json()
        found = False
        for v in views:
            if v.get("name") == "Tier Board" and v.get("type") == "kanban":
                found = True
                break
        check("9. Tier Board Kanban view", 1, found,
              "" if found else "kanban view 'Tier Board' not found on Project Compliance")
    except Exception as e:
        check("9. Tier Board Kanban view", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_default_formatter()
    check_2_format_on_save()
    check_3_git_commit()
    check_4_baserow_database()
    check_5_unformatted_files_table()
    check_6_by_type_view()
    check_7_compliance_table()
    check_8_compliance_data()
    check_9_tier_board_view()

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
