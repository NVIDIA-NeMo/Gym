# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_042/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-042-I2: Branching strategy governance across repos

Checks: 11 weighted checks across code-server, baserow, openproject.
Strategy: docker exec (filesystem + DB queries)

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

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

for var_name, val in [
    ("CODE_SERVER_CONTAINER", CODE_SERVER_CONTAINER),
    ("BASEROW_DB_CONTAINER", BASEROW_DB_CONTAINER),
    ("OPENPROJECT_CONTAINER", OPENPROJECT_CONTAINER),
]:
    if not val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
        sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
POLICY_DATE = "2026-05-15"
POLICY_OWNER = "OpenProject Admin"
BRANCH_PREFIXES = ["feature", "bugfix", "release", "docs"]
PREFIX_DESCRIPTIONS = {
    "feature": "Long-lived feature work integrated via PR",
    "bugfix": "Non-urgent defect remediation branches",
    "release": "Release stabilization branches cut from main",
    "docs": "Documentation-only changes",
}
PROJECT_LIST = ["vue-hackernews-2.0", "json", "tabler"]
ALL_PROJECTS_ALPHA = sorted(PROJECT_LIST + ["devops-configs"])  # alphabetical
NEW_BRANCH = "docs/branching-policy-rollout"
COMMIT_MSG = "docs: formalize branching strategy 2026-05-15"
BASEROW_DB_NAME = "Branch Strategy Governance Hub"
TABLE_NAME = "Repo Branch Audit"
OP_PROJECT = "scrum-project"

EXPECTED_FILE_CONTENT = (
    "# Branching Strategy\n"
    f"Effective Date: {POLICY_DATE}\n"
    f"Owner: {POLICY_OWNER}\n"
    "\n"
    "## Allowed Branch Prefixes\n"
)
for prefix in BRANCH_PREFIXES:
    EXPECTED_FILE_CONTENT += f"- {prefix}/: {PREFIX_DESCRIPTIONS[prefix]}\n"

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


def baserow_sql(query: str, timeout: int = 15) -> str:
    """Run a psql query against the Baserow DB and return stdout."""
    rc, out, err = docker_exec(
        BASEROW_DB_CONTAINER,
        "env", "PGPASSWORD=kdpzkuyhsgb22onku8y7rxkx3czej88nxpngaz4mlmgad67vpc", "psql", "-h", "127.0.0.1", "-U", "baserow", "-d", "baserow", "-t", "-A", "-c", query,
        timeout=timeout,
    )
    return out.strip()


def openproject_sql(query: str, timeout: int = 15) -> str:
    """Run a psql query against the OpenProject DB (embedded) and return stdout."""
    r = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=openproject", OPENPROJECT_CONTAINER,
         "psql", "-U", "openproject", "-d", "openproject", "-h", "127.0.0.1",
         "-t", "-A", "-c", query],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout.strip()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_branching_strategy_file() -> None:
    """Verify devops-configs/docs/BRANCHING_STRATEGY.md exists with exact content."""
    try:
        # Try common code-server workspace paths
        for base in ["/home/coder/project", "/home/coder", "/config/workspace", "/home/coder/workspace"]:
            path = f"{base}/devops-configs/docs/BRANCHING_STRATEGY.md"
            rc, out, err = docker_exec(CODE_SERVER_CONTAINER, "cat", path)
            if rc == 0:
                break
        if rc != 0:
            # Try find
            rc2, found, _ = docker_exec(CODE_SERVER_CONTAINER, "find", "/", "-path",
                                         "*/devops-configs/docs/BRANCHING_STRATEGY.md",
                                         "-type", "f", timeout=30)
            if rc2 == 0 and found.strip():
                first = found.strip().split("\n")[0]
                rc, out, err = docker_exec(CODE_SERVER_CONTAINER, "cat", first)

        if rc != 0:
            check("1. BRANCHING_STRATEGY.md exists with correct content", 2, False,
                  "file not found in container")
            return

        actual = out
        # Normalize: ensure trailing newline for comparison
        expected = EXPECTED_FILE_CONTENT
        if not actual.endswith("\n"):
            actual += "\n"

        passed = actual == expected
        detail = "" if passed else f"content mismatch; got {len(actual)} chars"
        check("1. BRANCHING_STRATEGY.md exists with correct content", 2, passed, detail)
    except Exception as e:
        check("1. BRANCHING_STRATEGY.md exists with correct content", 2, False, f"exception: {e}")


def check_2_commit_message() -> None:
    """Verify commit with exact message exists in devops-configs repo."""
    try:
        # Find the devops-configs repo
        for base in ["/home/coder/project", "/home/coder", "/config/workspace", "/home/coder/workspace"]:
            rc, out, err = docker_exec(
                CODE_SERVER_CONTAINER, "git", "-C", f"{base}/devops-configs",
                "log", "--oneline", "--all", "--grep", COMMIT_MSG, "--format=%s"
            )
            if rc == 0:
                break

        if rc != 0:
            # Try find
            rc2, found, _ = docker_exec(CODE_SERVER_CONTAINER, "find", "/", "-path",
                                         "*/devops-configs/.git", "-type", "d", timeout=30)
            if rc2 == 0 and found.strip():
                repo_git = found.strip().split("\n")[0]
                repo_dir = repo_git.rsplit("/.git", 1)[0]
                rc, out, err = docker_exec(
                    CODE_SERVER_CONTAINER, "git", "-C", repo_dir,
                    "log", "--oneline", "--all", "--grep", COMMIT_MSG, "--format=%s"
                )

        if rc != 0:
            check("2. Commit with exact message in devops-configs", 2, False,
                  "could not access devops-configs git repo")
            return

        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        passed = COMMIT_MSG in lines
        detail = "" if passed else f"commit message not found; got {lines[:3]}"
        check("2. Commit with exact message in devops-configs", 2, passed, detail)
    except Exception as e:
        check("2. Commit with exact message in devops-configs", 2, False, f"exception: {e}")


def check_3_branch_in_all_projects() -> None:
    """Verify branch docs/branching-policy-rollout exists in all 4 projects."""
    try:
        all_repos = PROJECT_LIST + ["devops-configs"]
        missing = []
        found_count = 0

        for proj in all_repos:
            branch_found = False
            for base in ["/home/coder/project", "/home/coder", "/config/workspace", "/home/coder/workspace"]:
                rc, out, err = docker_exec(
                    CODE_SERVER_CONTAINER, "git", "-C", f"{base}/{proj}",
                    "branch", "--list", NEW_BRANCH
                )
                if rc == 0 and NEW_BRANCH in out:
                    branch_found = True
                    found_count += 1
                    break
                # Also check if currently on that branch
                if rc == 0:
                    rc2, out2, _ = docker_exec(
                        CODE_SERVER_CONTAINER, "git", "-C", f"{base}/{proj}",
                        "rev-parse", "--abbrev-ref", "HEAD"
                    )
                    if rc2 == 0 and out2.strip() == NEW_BRANCH:
                        branch_found = True
                        found_count += 1
                        break
            if not branch_found:
                missing.append(proj)

        passed = len(missing) == 0
        detail = "" if passed else f"missing in: {missing}"
        check("3. Branch docs/branching-policy-rollout in all 4 projects", 2, passed, detail)
    except Exception as e:
        check("3. Branch docs/branching-policy-rollout in all 4 projects", 2, False, f"exception: {e}")


def check_4_baserow_database_exists() -> None:
    """Verify Baserow database 'Branch Strategy Governance Hub' exists."""
    try:
        result = baserow_sql(
            f"SELECT id FROM database_application da "
            f"JOIN core_application ca ON ca.id = da.application_ptr_id "
            f"WHERE ca.name = '{BASEROW_DB_NAME}';"
        )
        if not result:
            # Try alternate schema
            result = baserow_sql(
                f"SELECT id FROM core_application WHERE name = '{BASEROW_DB_NAME}';"
            )
        passed = bool(result and result.strip())
        detail = "" if passed else "database not found"
        check("4. Baserow database exists", 1, passed, detail)
    except Exception as e:
        check("4. Baserow database exists", 1, False, f"exception: {e}")


def _get_baserow_table_id() -> str | None:
    """Get the Baserow table ID for 'Repo Branch Audit'."""
    # Try joining with core_application
    result = baserow_sql(
        f"SELECT dt.id FROM database_table dt "
        f"JOIN core_application ca ON ca.id = dt.database_id "
        f"WHERE dt.name = '{TABLE_NAME}' AND ca.name = '{BASEROW_DB_NAME}';"
    )
    if result and result.strip():
        return result.strip().split("\n")[0]
    # Fallback: just find by table name
    result = baserow_sql(
        f"SELECT id FROM database_table WHERE name = '{TABLE_NAME}';"
    )
    if result and result.strip():
        return result.strip().split("\n")[0]
    return None


def _get_field_map(table_id: str) -> dict:
    """Return {field_name: (field_id, field_type)} for the given table."""
    result = baserow_sql(
        f"SELECT f.id, f.name, ct.model "
        f"FROM database_field f "
        f"JOIN django_content_type ct ON ct.id = f.content_type_id "
        f"WHERE f.table_id = {table_id} AND f.trashed = false "
        f"ORDER BY f.order;"
    )
    fields = {}
    for line in result.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            fid, fname, ftype = parts[0].strip(), parts[1].strip(), parts[2].strip()
            fields[fname] = (fid, ftype)
    return fields


def check_5_baserow_table_fields() -> None:
    """Verify table 'Repo Branch Audit' exists with expected fields."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("5. Table Repo Branch Audit with correct fields", 2, False, "table not found")
            return

        field_map = _get_field_map(table_id)
        expected_fields = ["Repo ID", "Project", "Current Branch", "New Branch Created",
                           "Prefix Used", "Compliant", "Audited At"]
        missing = [f for f in expected_fields if f not in field_map]
        passed = len(missing) == 0
        detail = "" if passed else f"missing fields: {missing}; found: {list(field_map.keys())}"
        check("5. Table Repo Branch Audit with correct fields", 2, passed, detail)
    except Exception as e:
        check("5. Table Repo Branch Audit with correct fields", 2, False, f"exception: {e}")


def check_6_baserow_rows_correct_projects() -> None:
    """Verify exactly 4 rows with correct Project values in alphabetical order."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("6. Exactly 4 rows with correct Projects (alphabetical)", 2, False, "table not found")
            return

        field_map = _get_field_map(table_id)
        if "Project" not in field_map:
            check("6. Exactly 4 rows with correct Projects (alphabetical)", 2, False, "Project field not found")
            return

        project_fid = field_map["Project"][0]
        # For single-select, value is stored as a foreign key; need to get the select option value
        # Query the dynamic table
        row_count_str = baserow_sql(f"SELECT count(*) FROM database_table_{table_id};")
        row_count = int(row_count_str.strip()) if row_count_str.strip() else 0

        if row_count != 4:
            check("6. Exactly 4 rows with correct Projects (alphabetical)", 2, False,
                  f"expected 4 rows, got {row_count}")
            return

        # Get Project values - for single_select, the column stores the option id
        # We need to join with database_selectoption
        result = baserow_sql(
            f"SELECT so.value FROM database_table_{table_id} r "
            f"JOIN database_selectoption so ON so.id = r.field_{project_fid} "
            f"ORDER BY r.\"order\", r.id;"
        )
        projects = [l.strip() for l in result.strip().split("\n") if l.strip()]

        passed = projects == ALL_PROJECTS_ALPHA
        detail = "" if passed else f"expected {ALL_PROJECTS_ALPHA}, got {projects}"
        check("6. Exactly 4 rows with correct Projects (alphabetical)", 2, passed, detail)
    except Exception as e:
        check("6. Exactly 4 rows with correct Projects (alphabetical)", 2, False, f"exception: {e}")


def check_7_baserow_new_branch_and_date() -> None:
    """Verify New Branch Created and Audited At values are correct for all rows."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("7. New Branch Created & Audited At correct", 2, False, "table not found")
            return

        field_map = _get_field_map(table_id)
        if "New Branch Created" not in field_map or "Audited At" not in field_map:
            check("7. New Branch Created & Audited At correct", 2, False,
                  f"missing fields; have: {list(field_map.keys())}")
            return

        nbc_fid = field_map["New Branch Created"][0]
        aud_fid = field_map["Audited At"][0]

        result = baserow_sql(
            f"SELECT field_{nbc_fid}, field_{aud_fid}::text FROM database_table_{table_id};"
        )
        issues = []
        for line in result.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            nbc_val = parts[0].strip() if len(parts) > 0 else ""
            aud_val = parts[1].strip() if len(parts) > 1 else ""
            if nbc_val != NEW_BRANCH:
                issues.append(f"New Branch Created={nbc_val!r}")
            if not aud_val.startswith(POLICY_DATE):
                issues.append(f"Audited At={aud_val!r}")

        passed = len(issues) == 0
        detail = "" if passed else "; ".join(issues[:5])
        check("7. New Branch Created & Audited At correct", 2, passed, detail)
    except Exception as e:
        check("7. New Branch Created & Audited At correct", 2, False, f"exception: {e}")


def check_8_baserow_compliant_logic() -> None:
    """Verify Compliant and Prefix Used are logically consistent."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("8. Compliant / Prefix Used logic correct", 2, False, "table not found")
            return

        field_map = _get_field_map(table_id)
        needed = ["Current Branch", "Prefix Used", "Compliant"]
        missing = [f for f in needed if f not in field_map]
        if missing:
            check("8. Compliant / Prefix Used logic correct", 2, False, f"missing fields: {missing}")
            return

        cb_fid = field_map["Current Branch"][0]
        pu_fid = field_map["Prefix Used"][0]
        comp_fid = field_map["Compliant"][0]

        # Prefix Used is single-select, Compliant is boolean
        result = baserow_sql(
            f"SELECT field_{cb_fid}, "
            f"(SELECT so.value FROM database_selectoption so WHERE so.id = r.field_{pu_fid}), "
            f"field_{comp_fid} "
            f"FROM database_table_{table_id} r;"
        )
        issues = []
        for line in result.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 3:
                issues.append(f"unexpected row format: {line}")
                continue
            cur_branch = parts[0].strip()
            prefix_used = parts[1].strip() if parts[1].strip() else None
            compliant_str = parts[2].strip().lower()
            compliant = compliant_str in ("t", "true", "1")

            # Determine expected prefix
            expected_prefix = None
            for p in BRANCH_PREFIXES:
                if cur_branch.startswith(p + "/"):
                    expected_prefix = p
                    break

            if prefix_used != expected_prefix and not (prefix_used is None and expected_prefix is None):
                if not (prefix_used == "" and expected_prefix is None):
                    issues.append(f"branch={cur_branch}: prefix_used={prefix_used!r} expected={expected_prefix!r}")

            expected_compliant = expected_prefix is not None
            if compliant != expected_compliant:
                issues.append(f"branch={cur_branch}: compliant={compliant} expected={expected_compliant}")

        passed = len(issues) == 0
        detail = "" if passed else "; ".join(issues[:5])
        check("8. Compliant / Prefix Used logic correct", 2, passed, detail)
    except Exception as e:
        check("8. Compliant / Prefix Used logic correct", 2, False, f"exception: {e}")


def check_9_baserow_view_filter() -> None:
    """Verify Grid view 'Non-Compliant Repos' filtered to Compliant=false."""
    try:
        table_id = _get_baserow_table_id()
        if not table_id:
            check("9. Grid view Non-Compliant Repos with filter", 1, False, "table not found")
            return

        # Find the view
        view_result = baserow_sql(
            f"SELECT v.id, ct.model FROM database_view v "
            f"JOIN django_content_type ct ON ct.id = v.content_type_id "
            f"WHERE v.table_id = {table_id} AND v.name = 'Non-Compliant Repos';"
        )
        if not view_result.strip():
            check("9. Grid view Non-Compliant Repos with filter", 1, False, "view not found")
            return

        view_id = view_result.strip().split("|")[0].strip()

        # Check for a filter on Compliant field
        field_map = _get_field_map(table_id)
        comp_fid = field_map.get("Compliant", (None, None))[0]

        filter_result = baserow_sql(
            f"SELECT field_id, type, value FROM database_viewfilter WHERE view_id = {view_id};"
        )

        # Check that there's a filter on the Compliant field
        has_compliant_filter = False
        if filter_result.strip():
            for line in filter_result.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 3:
                    fid = parts[0].strip()
                    if comp_fid and fid == comp_fid:
                        has_compliant_filter = True

        passed = has_compliant_filter
        detail = "" if passed else f"no filter on Compliant field (field_id={comp_fid}); filters: {filter_result.strip()}"
        check("9. Grid view Non-Compliant Repos with filter", 1, passed, detail)
    except Exception as e:
        check("9. Grid view Non-Compliant Repos with filter", 1, False, f"exception: {e}")


def check_10_openproject_wiki_page() -> None:
    """Verify wiki page 'Branching Strategy' in scrum-project with correct body."""
    try:
        # Get the wiki page content (text is directly on wiki_pages in OpenProject)
        result = openproject_sql(
            f"SELECT wp.text FROM wiki_pages wp "
            f"JOIN wikis w ON w.id = wp.wiki_id "
            f"JOIN projects p ON p.id = w.project_id "
            f"WHERE p.identifier = '{OP_PROJECT}' AND wp.title = 'Branching Strategy';"
        )
        if not result.strip():
            # Try with slug
            result = openproject_sql(
                f"SELECT wp.text FROM wiki_pages wp "
                f"JOIN wikis w ON w.id = wp.wiki_id "
                f"JOIN projects p ON p.id = w.project_id "
                f"WHERE p.identifier = '{OP_PROJECT}' AND wp.slug = 'branching-strategy';"
            )
        if not result.strip():
            check("10. OpenProject wiki page Branching Strategy", 2, False, "wiki page not found")
            return

        body = result.strip()
        # Expected lines:
        # Line A: "Policy document: devops-configs/docs/BRANCHING_STRATEGY.md"
        # Line B: "Effective: 2026-05-15"
        # Line C: "Repos audited: 4" (R = 4 total rows)
        issues = []
        if "Policy document: devops-configs/docs/BRANCHING_STRATEGY.md" not in body:
            issues.append("missing line A (Policy document)")
        if f"Effective: {POLICY_DATE}" not in body:
            issues.append("missing line B (Effective date)")
        if "Repos audited: 4" not in body:
            issues.append("missing line C (Repos audited: 4)")

        passed = len(issues) == 0
        detail = "" if passed else "; ".join(issues) + f"; body={body[:200]!r}"
        check("10. OpenProject wiki page Branching Strategy", 2, passed, detail)
    except Exception as e:
        check("10. OpenProject wiki page Branching Strategy", 2, False, f"exception: {e}")


def check_11_openproject_work_package() -> None:
    """Verify Task work package with correct subject, assignee, description."""
    try:
        # Find the work package
        result = openproject_sql(
            f"SELECT wp.id, wp.subject, wp.description, t.name as type_name, "
            f"u.login as assignee_login, u.firstname, u.lastname "
            f"FROM work_packages wp "
            f"JOIN types t ON t.id = wp.type_id "
            f"LEFT JOIN users u ON u.id = wp.assigned_to_id "
            f"WHERE wp.subject = 'Enforce branching policy: {POLICY_DATE}';"
        )
        if not result.strip():
            check("11. OpenProject work package with correct fields", 2, False, "work package not found")
            return

        parts = result.strip().split("|")
        issues = []

        if len(parts) >= 4:
            type_name = parts[3].strip()
            if type_name.lower() != "task":
                issues.append(f"type={type_name!r}, expected Task")

        if len(parts) >= 7:
            firstname = parts[5].strip()
            lastname = parts[6].strip()
            full_name = f"{firstname} {lastname}"
            if "OpenProject" not in full_name and "Admin" not in full_name:
                issues.append(f"assignee={full_name!r}, expected OpenProject Admin")

        if len(parts) >= 3:
            desc = parts[2].strip()
            # Expected: "Policy: devops-configs/docs/BRANCHING_STRATEGY.md; Repos: 4; Non-compliant: <NC>"
            if "Policy: devops-configs/docs/BRANCHING_STRATEGY.md" not in desc:
                issues.append("description missing Policy path")
            if "Repos: 4" not in desc:
                issues.append("description missing Repos: 4")
            # NC could vary, but check format
            nc_match = re.search(r"Non-compliant:\s*(\d+)", desc)
            if not nc_match:
                issues.append("description missing Non-compliant count")

        passed = len(issues) == 0
        detail = "" if passed else "; ".join(issues)
        check("11. OpenProject work package with correct fields", 2, passed, detail)
    except Exception as e:
        check("11. OpenProject work package with correct fields", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_branching_strategy_file()
    check_2_commit_message()
    check_3_branch_in_all_projects()
    check_4_baserow_database_exists()
    check_5_baserow_table_fields()
    check_6_baserow_rows_correct_projects()
    check_7_baserow_new_branch_and_date()
    check_8_baserow_compliant_logic()
    check_9_baserow_view_filter()
    check_10_openproject_wiki_page()
    check_11_openproject_work_package()

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
