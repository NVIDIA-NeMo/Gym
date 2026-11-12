# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_027/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-027-I2: Cross-Team Dependency Map for Frontend & Infra Services

Checks: 13 weighted checks across code-server, baserow, openproject.
Strategy: code-server via docker exec filesystem; baserow via REST API; openproject via REST API.

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER.
"""

import os
import re
import sys
import json
import subprocess
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_PORT = os.getenv("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.getenv("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.getenv("BASEROW_PORT")
BASEROW_CONTAINER = os.getenv("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.getenv("BASEROW_DB_CONTAINER")
OPENPROJECT_PORT = os.getenv("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.getenv("OPENPROJECT_CONTAINER")

_required = {
    "CODE_SERVER_PORT": CODE_SERVER_PORT,
    "CODE_SERVER_CONTAINER": CODE_SERVER_CONTAINER,
    "BASEROW_PORT": BASEROW_PORT,
    "BASEROW_CONTAINER": BASEROW_CONTAINER,
    "BASEROW_DB_CONTAINER": BASEROW_DB_CONTAINER,
    "OPENPROJECT_PORT": OPENPROJECT_PORT,
    "OPENPROJECT_CONTAINER": OPENPROJECT_CONTAINER,
}
for var, val in _required.items():
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"
OPENPROJECT_URL = f"http://{HOST}:{OPENPROJECT_PORT}"

# ── Task constants ────────────────────────────────────────────────────────────
PROJECTS = ["tabler", "vue-hackernews-2.0", "json", "devops-configs", "todo-api"]
SCAN_ROOTS = {
    "tabler": "tabler/core",
    "vue-hackernews-2.0": "vue-hackernews-2.0/src",
    "json": "json/include",
    "devops-configs": "devops-configs",
    "todo-api": "todo-api/app",
}
OWNERSHIP = {
    "tabler": {"owning_team": "Frontend", "tech_lead": "Karen Brown"},
    "vue-hackernews-2.0": {"owning_team": "Frontend", "tech_lead": "Liam Robinson"},
    "json": {"owning_team": "Core Libraries", "tech_lead": "Frank Nguyen"},
    "devops-configs": {"owning_team": "Infrastructure", "tech_lead": "Noah Taylor"},
    "todo-api": {"owning_team": "Backend", "tech_lead": "Grace Patel"},
}
BASEROW_DB_NAME = "Service Coupling Atlas Q3"
OP_PROJECT = "Infrastructure Upgrade"
EPIC_SUBJECT = "Dependency map snapshot: 2025-07-22"

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
    """Authenticate to Baserow, return JWT token."""
    r = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["token"]


def baserow_get(path: str, token: str) -> dict:
    r = requests.get(
        f"{BASEROW_URL}/api/{path}",
        headers={"Authorization": f"Token {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def op_get(path: str) -> dict:
    """OpenProject API GET with basic auth."""
    r = requests.get(
        f"{OPENPROJECT_URL}{path}",
        auth=("admin", "AdminPass123!"),
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# ── Check 1: code-server project directories exist ───────────────────────────
def check_1_project_dirs() -> None:
    """Verify that all 5 project directories exist on the code-server filesystem."""
    try:
        missing = []
        for proj in PROJECTS:
            rc, out, err = docker_exec(
                CODE_SERVER_CONTAINER, "test", "-d", f"/home/coder/project/{proj}"
            )
            if rc != 0:
                # Try alternate paths
                rc2, _, _ = docker_exec(
                    CODE_SERVER_CONTAINER, "test", "-d", f"/home/coder/{proj}"
                )
                if rc2 != 0:
                    missing.append(proj)
        check("1. Project dirs on code-server", 1, len(missing) == 0,
              "all 5 exist" if not missing else f"missing: {missing}")
    except Exception as e:
        check("1. Project dirs on code-server", 1, False, f"exception: {e}")


# ── Baserow checks ───────────────────────────────────────────────────────────
_baserow_token = None
_baserow_db_id = None
_team_ownership_table_id = None
_dep_edges_table_id = None
_team_ownership_rows = []
_dep_edges_rows = []
_dep_edges_fields = []


def _init_baserow() -> bool:
    """Authenticate and find the database + tables. Returns True if successful."""
    global _baserow_token, _baserow_db_id, _team_ownership_table_id, _dep_edges_table_id
    global _team_ownership_rows, _dep_edges_rows, _dep_edges_fields
    try:
        _baserow_token = baserow_auth()

        # Find the database
        apps = baserow_get("applications/", _baserow_token)
        for app in apps:
            if app.get("name") == BASEROW_DB_NAME and app.get("type") == "database":
                _baserow_db_id = app["id"]
                break

        if not _baserow_db_id:
            return False

        # Find tables
        tables = baserow_get(f"database/tables/database/{_baserow_db_id}/", _baserow_token)
        for t in tables:
            if t["name"] == "Team Ownership":
                _team_ownership_table_id = t["id"]
            elif t["name"] == "Dependency Edges":
                _dep_edges_table_id = t["id"]

        # Load rows if tables exist
        if _team_ownership_table_id:
            resp = baserow_get(
                f"database/rows/table/{_team_ownership_table_id}/?user_field_names=true&size=100",
                _baserow_token,
            )
            _team_ownership_rows = resp.get("results", [])

        if _dep_edges_table_id:
            _dep_edges_fields = baserow_get(
                f"database/fields/table/{_dep_edges_table_id}/", _baserow_token
            )
            resp = baserow_get(
                f"database/rows/table/{_dep_edges_table_id}/?user_field_names=true&size=200",
                _baserow_token,
            )
            _dep_edges_rows = resp.get("results", [])

        return True
    except Exception:
        return False


def check_2_baserow_db_exists() -> None:
    """Database 'Service Coupling Atlas Q3' exists in Baserow."""
    try:
        ok = _init_baserow()
        check("2. Baserow DB exists", 1, _baserow_db_id is not None,
              f"db_id={_baserow_db_id}" if _baserow_db_id else "database not found")
    except Exception as e:
        check("2. Baserow DB exists", 1, False, f"exception: {e}")


def check_3_team_ownership_rows() -> None:
    """Team Ownership table has exactly 5 rows with correct project names."""
    try:
        if not _team_ownership_table_id:
            check("3. Team Ownership rows", 2, False, "table not found")
            return

        # Find which field is the primary (Project) field
        row_projects = set()
        for row in _team_ownership_rows:
            # The primary field is typically named "Project" or is the first text field
            proj = row.get("Project") or row.get("Name") or ""
            if isinstance(proj, str) and proj:
                row_projects.add(proj)

        expected = set(PROJECTS)
        missing = expected - row_projects
        extra = row_projects - expected
        ok = len(_team_ownership_rows) == 5 and not missing
        detail = f"{len(_team_ownership_rows)} rows"
        if missing:
            detail += f", missing: {sorted(missing)}"
        if extra:
            detail += f", extra: {sorted(extra)}"
        check("3. Team Ownership rows", 2, ok, detail)
    except Exception as e:
        check("3. Team Ownership rows", 2, False, f"exception: {e}")


def check_4_team_ownership_fields() -> None:
    """Team Ownership rows have correct Owning Team and Tech Lead values."""
    try:
        if not _team_ownership_rows:
            check("4. Team Ownership fields", 2, False, "no rows to check")
            return

        issues = []
        for row in _team_ownership_rows:
            proj = row.get("Project") or row.get("Name") or ""
            if proj not in OWNERSHIP:
                continue
            expected = OWNERSHIP[proj]

            # Owning Team may be a dict (single-select) or string
            owning_team_raw = row.get("Owning Team")
            if isinstance(owning_team_raw, dict):
                owning_team = owning_team_raw.get("value", "")
            elif isinstance(owning_team_raw, list) and owning_team_raw:
                owning_team = owning_team_raw[0].get("value", "") if isinstance(owning_team_raw[0], dict) else str(owning_team_raw[0])
            else:
                owning_team = str(owning_team_raw or "")

            tech_lead = str(row.get("Tech Lead", ""))

            if owning_team != expected["owning_team"]:
                issues.append(f"{proj}: team={owning_team!r} expected {expected['owning_team']!r}")
            if tech_lead != expected["tech_lead"]:
                issues.append(f"{proj}: lead={tech_lead!r} expected {expected['tech_lead']!r}")

        ok = len(issues) == 0 and len(_team_ownership_rows) == 5
        check("4. Team Ownership fields", 2, ok,
              "all correct" if ok else f"issues: {issues[:5]}")
    except Exception as e:
        check("4. Team Ownership fields", 2, False, f"exception: {e}")


def check_5_dep_edges_table() -> None:
    """Dependency Edges table exists with rows having DE-NNN Edge IDs."""
    try:
        if not _dep_edges_table_id:
            check("5. Dependency Edges table", 2, False, "table not found")
            return

        # Check Edge IDs follow DE-NNN pattern
        edge_ids = []
        for row in _dep_edges_rows:
            eid = row.get("Edge ID") or row.get("Name") or ""
            edge_ids.append(str(eid))

        de_pattern = re.compile(r"^DE-\d{3}$")
        valid_ids = [eid for eid in edge_ids if de_pattern.match(eid)]
        has_rows = len(_dep_edges_rows) > 0
        all_valid = len(valid_ids) == len(_dep_edges_rows) and has_rows

        check("5. Dependency Edges table", 2, all_valid,
              f"{len(_dep_edges_rows)} rows, {len(valid_ids)} valid DE-NNN IDs")
    except Exception as e:
        check("5. Dependency Edges table", 2, False, f"exception: {e}")


def check_6_dep_edges_links() -> None:
    """Dependency Edges rows have Source Project and Target Project link fields."""
    try:
        if not _dep_edges_fields:
            check("6. Dependency Edges link fields", 2, False, "no fields loaded")
            return

        field_names = {f["name"]: f["type"] for f in _dep_edges_fields}
        has_source = "Source Project" in field_names and "link" in field_names.get("Source Project", "")
        has_target = "Target Project" in field_names and "link" in field_names.get("Target Project", "")

        # Also check rows actually have linked values
        rows_with_links = 0
        for row in _dep_edges_rows:
            sp = row.get("Source Project")
            tp = row.get("Target Project")
            if sp and tp:
                rows_with_links += 1

        ok = has_source and has_target and rows_with_links == len(_dep_edges_rows) and len(_dep_edges_rows) > 0
        check("6. Dependency Edges link fields", 2, ok,
              f"Source={'link_row' if has_source else 'missing'}, Target={'link_row' if has_target else 'missing'}, "
              f"{rows_with_links}/{len(_dep_edges_rows)} rows linked")
    except Exception as e:
        check("6. Dependency Edges link fields", 2, False, f"exception: {e}")


def check_7_cross_team_flag() -> None:
    """Cross Team boolean is set correctly based on team ownership."""
    try:
        if not _dep_edges_rows or not _team_ownership_rows:
            check("7. Cross Team flag", 2, False, "no data to verify")
            return

        # Build project→team map from Team Ownership rows
        proj_team = {}
        for row in _team_ownership_rows:
            proj = row.get("Project") or row.get("Name") or ""
            ot = row.get("Owning Team")
            if isinstance(ot, dict):
                team = ot.get("value", "")
            elif isinstance(ot, list) and ot:
                team = ot[0].get("value", "") if isinstance(ot[0], dict) else str(ot[0])
            else:
                team = str(ot or "")
            if proj:
                proj_team[proj] = team

        issues = []
        for row in _dep_edges_rows:
            eid = row.get("Edge ID") or row.get("Name") or ""
            cross = row.get("Cross Team")
            # cross might be bool or truthy value
            is_cross = bool(cross)

            # Resolve source/target project names from link field
            sp_raw = row.get("Source Project")
            tp_raw = row.get("Target Project")
            sp_name = ""
            tp_name = ""
            if isinstance(sp_raw, list) and sp_raw:
                sp_name = sp_raw[0].get("value", "") if isinstance(sp_raw[0], dict) else str(sp_raw[0])
            elif isinstance(sp_raw, str):
                sp_name = sp_raw
            if isinstance(tp_raw, list) and tp_raw:
                tp_name = tp_raw[0].get("value", "") if isinstance(tp_raw[0], dict) else str(tp_raw[0])
            elif isinstance(tp_raw, str):
                tp_name = tp_raw

            if sp_name in proj_team and tp_name in proj_team:
                expected_cross = proj_team[sp_name] != proj_team[tp_name]
                if is_cross != expected_cross:
                    issues.append(f"{eid}: cross={is_cross}, expected={expected_cross}")

        ok = len(issues) == 0 and len(_dep_edges_rows) > 0
        check("7. Cross Team flag", 2, ok,
              "all correct" if ok else f"mismatches: {issues[:5]}")
    except Exception as e:
        check("7. Cross Team flag", 2, False, f"exception: {e}")


def check_8_cross_team_view() -> None:
    """'Cross-Team Edges' grid view exists on Dependency Edges table."""
    try:
        if not _dep_edges_table_id:
            check("8. Cross-Team Edges view", 1, False, "table not found")
            return

        views = baserow_get(
            f"database/views/table/{_dep_edges_table_id}/", _baserow_token
        )
        view_names = [v["name"] for v in views]
        found = "Cross-Team Edges" in view_names
        check("8. Cross-Team Edges view", 1, found,
              f"views: {view_names}" if not found else "view exists")
    except Exception as e:
        check("8. Cross-Team Edges view", 1, False, f"exception: {e}")


# ── OpenProject checks ───────────────────────────────────────────────────────
_op_epic = None
_op_children = []


def _init_openproject() -> bool:
    """Find the Infrastructure Upgrade project and the epic."""
    global _op_epic, _op_children
    try:
        # Find project
        projects = op_get("/api/v3/projects")
        proj_id = None
        for p in projects.get("_embedded", {}).get("elements", []):
            if p.get("name") == OP_PROJECT:
                proj_id = p["id"]
                break

        if not proj_id:
            return False

        # Find work packages in project
        wps = op_get(f"/api/v3/projects/{proj_id}/work_packages?pageSize=100")
        elements = wps.get("_embedded", {}).get("elements", [])

        for wp in elements:
            subj = wp.get("subject", "")
            if subj == EPIC_SUBJECT:
                _op_epic = wp
                break

        if _op_epic:
            epic_id = _op_epic["id"]
            # Find children: work packages whose parent is the epic
            children_resp = op_get(
                f"/api/v3/projects/{proj_id}/work_packages?pageSize=100"
                f"&filters=[{{\"parent\":{{\"operator\":\"=\",\"values\":[\"{epic_id}\"]}}}}"
                "]"
            )
            _op_children = children_resp.get("_embedded", {}).get("elements", [])

        return True
    except Exception:
        return False


def check_9_op_epic_exists() -> None:
    """Epic 'Dependency map snapshot: 2025-07-22' exists in Infrastructure Upgrade."""
    try:
        _init_openproject()
        ok = _op_epic is not None
        detail = f"id={_op_epic['id']}" if ok else "epic not found"
        # Also check type is Epic
        if ok:
            wp_type = _op_epic.get("_links", {}).get("type", {}).get("title", "")
            if "epic" not in wp_type.lower():
                detail += f", type={wp_type!r} (expected Epic)"
                # Still pass if subject matches — type naming may vary
        check("9. OpenProject Epic exists", 2, ok, detail)
    except Exception as e:
        check("9. OpenProject Epic exists", 2, False, f"exception: {e}")


def check_10_op_epic_description() -> None:
    """Epic description matches 'Total edges: <E>; Cross-team edges: <X>' format."""
    try:
        if not _op_epic:
            check("10. Epic description", 2, False, "epic not found")
            return

        desc = _op_epic.get("description", {})
        if isinstance(desc, dict):
            desc_text = desc.get("raw", "")
        else:
            desc_text = str(desc or "")

        pattern = r"Total edges:\s*\d+;\s*Cross-team edges:\s*\d+"
        match = re.search(pattern, desc_text)
        check("10. Epic description", 2, match is not None,
              f"desc={desc_text!r:.100}")
    except Exception as e:
        check("10. Epic description", 2, False, f"exception: {e}")


def check_11_op_child_tasks_subjects() -> None:
    """Child Tasks under Epic have correct 'Sync: TeamA ↔ TeamB (K coupling points)' subjects."""
    try:
        if not _op_epic:
            check("11. Child Task subjects", 2, False, "epic not found")
            return

        if not _op_children:
            check("11. Child Task subjects", 2, False, "no child work packages found")
            return

        sync_pattern = re.compile(
            r"^Sync:\s+(.+?)\s+[\u2194↔]\s+(.+?)\s+\((\d+)\s+coupling points?\)$"
        )
        valid = 0
        for child in _op_children:
            subj = child.get("subject", "")
            m = sync_pattern.match(subj)
            if m:
                team_a, team_b = m.group(1), m.group(2)
                # TeamA should be alphabetically before TeamB
                if team_a < team_b:
                    valid += 1

        ok = valid == len(_op_children) and len(_op_children) > 0
        subjects = [c.get("subject", "") for c in _op_children]
        check("11. Child Task subjects", 2, ok,
              f"{valid}/{len(_op_children)} valid; subjects={subjects}")
    except Exception as e:
        check("11. Child Task subjects", 2, False, f"exception: {e}")


def check_12_op_child_assignee() -> None:
    """Child Tasks are assigned to OpenProject Admin."""
    try:
        if not _op_children:
            check("12. Child Task assignees", 1, False, "no children")
            return

        issues = []
        for child in _op_children:
            assignee_link = child.get("_links", {}).get("assignee", {})
            assignee_title = assignee_link.get("title", "")
            if "admin" not in assignee_title.lower() and "openproject" not in assignee_title.lower():
                issues.append(f"{child.get('subject','')}: assignee={assignee_title!r}")

        ok = len(issues) == 0 and len(_op_children) > 0
        check("12. Child Task assignees", 1, ok,
              "all assigned to admin" if ok else f"issues: {issues[:3]}")
    except Exception as e:
        check("12. Child Task assignees", 1, False, f"exception: {e}")


def check_13_op_child_priorities() -> None:
    """Child Tasks have priority High when K>=2, Normal otherwise."""
    try:
        if not _op_children:
            check("13. Child Task priorities", 1, False, "no children")
            return

        sync_pattern = re.compile(
            r"Sync:\s+.+?\s+[\u2194↔]\s+.+?\s+\((\d+)\s+coupling points?\)"
        )
        issues = []
        for child in _op_children:
            subj = child.get("subject", "")
            m = sync_pattern.search(subj)
            if not m:
                issues.append(f"cannot parse K from {subj!r}")
                continue
            k = int(m.group(1))
            priority_title = child.get("_links", {}).get("priority", {}).get("title", "")
            expected_prio = "High" if k >= 2 else "Normal"
            if priority_title.lower() != expected_prio.lower():
                issues.append(f"K={k}: prio={priority_title!r} expected {expected_prio!r}")

        ok = len(issues) == 0 and len(_op_children) > 0
        check("13. Child Task priorities", 1, ok,
              "all correct" if ok else f"issues: {issues[:3]}")
    except Exception as e:
        check("13. Child Task priorities", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_project_dirs()
    check_2_baserow_db_exists()
    check_3_team_ownership_rows()
    check_4_team_ownership_fields()
    check_5_dep_edges_table()
    check_6_dep_edges_links()
    check_7_cross_team_flag()
    check_8_cross_team_view()
    check_9_op_epic_exists()
    check_10_op_epic_description()
    check_11_op_child_tasks_subjects()
    check_12_op_child_assignee()
    check_13_op_child_priorities()

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
