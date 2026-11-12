# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_040/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-040-I1: Build tabler frontend component inventory across
code-server, Baserow, and OpenProject.

Checks: 15 weighted checks across code-server, baserow, openproject.
Strategy: docker exec (DB for baserow/openproject, filesystem for code-server).

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


def baserow_sql(query: str) -> str:
    """Run a psql query against the Baserow database."""
    rc, out, err = docker_exec(
        BASEROW_DB_CONTAINER,
        "env", "PGPASSWORD=kdpzkuyhsgb22onku8y7rxkx3czej88nxpngaz4mlmgad67vpc", "psql", "-h", "127.0.0.1", "-U", "baserow", "-d", "baserow", "-t", "-A", "-c", query,
    )
    if rc != 0:
        raise RuntimeError(f"baserow psql failed: {err.strip()}")
    return out.strip()


def openproject_sql(query: str) -> str:
    """Run a psql query against the OpenProject database (embedded)."""
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "bash", "-c",
        f"PGPASSWORD=openproject psql -h 127.0.0.1 -U openproject -d openproject -t -A -c {repr(query)}",
    )
    if rc != 0:
        raise RuntimeError(f"openproject psql failed: {err.strip()}")
    return out.strip()


# ── Baserow checks ───────────────────────────────────────────────────────────

def check_1_baserow_database_exists() -> None:
    """Verify Baserow database 'Tabler Component Audit' exists."""
    try:
        result = baserow_sql(
            "SELECT COUNT(*) FROM database_database dd "
            "JOIN core_application ca ON dd.application_ptr_id = ca.id "
            "WHERE ca.name = 'Tabler Component Audit';"
        )
        count = int(result.strip().split('\n')[-1])
        check("1. Baserow database 'Tabler Component Audit' exists", 1,
              count >= 1, f"found {count}")
    except Exception as e:
        check("1. Baserow database 'Tabler Component Audit' exists", 1, False, f"exception: {e}")


def check_2_baserow_table_exists() -> None:
    """Verify table 'Frontend Components' exists in the database."""
    try:
        result = baserow_sql(
            "SELECT dt.id FROM database_table dt "
            "JOIN core_application ca ON dt.database_id = ca.id "
            "WHERE ca.name = 'Tabler Component Audit' "
            "AND dt.name = 'Frontend Components';"
        )
        lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
        found = len(lines) >= 1
        check("2. Table 'Frontend Components' exists", 1, found,
              f"table_id={lines[0]}" if found else "not found")
    except Exception as e:
        check("2. Table 'Frontend Components' exists", 1, False, f"exception: {e}")


def _get_table_id() -> str | None:
    """Get the Baserow table ID for 'Frontend Components'."""
    result = baserow_sql(
        "SELECT dt.id FROM database_table dt "
        "JOIN core_application ca ON dt.database_id = ca.id "
        "WHERE ca.name = 'Tabler Component Audit' "
        "AND dt.name = 'Frontend Components' LIMIT 1;"
    )
    lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
    return lines[0] if lines else None


def check_3_rows_have_valid_component_ids() -> None:
    """Verify rows have Component ID in FC-NNN format."""
    try:
        table_id = _get_table_id()
        if not table_id:
            check("3. Rows have valid FC-NNN Component IDs", 1, False, "table not found")
            return
        # The primary field in Baserow dynamic row tables is typically in a field column
        # Get field ID for the primary field (Component ID)
        field_result = baserow_sql(
            f"SELECT df.id, df.name FROM database_field df "
            f"WHERE df.table_id = {table_id} AND df.name = 'Component ID';"
        )
        if not field_result.strip():
            # Try checking primary field
            field_result = baserow_sql(
                f"SELECT df.id, df.name FROM database_field df "
                f"WHERE df.table_id = {table_id} AND df.primary IS TRUE;"
            )
        if not field_result.strip():
            check("3. Rows have valid FC-NNN Component IDs", 1, False, "Component ID field not found")
            return
        field_id = field_result.strip().split('\n')[0].split('|')[0].strip()

        # Query the dynamic row table
        row_result = baserow_sql(
            f"SELECT field_{field_id} FROM database_table_{table_id} "
            f"WHERE trashed = false ORDER BY \"order\" ASC;"
        )
        rows = [r.strip() for r in row_result.strip().split('\n') if r.strip()]
        if not rows:
            check("3. Rows have valid FC-NNN Component IDs", 1, False, "no rows found")
            return
        pattern = re.compile(r'^FC-\d{3}$')
        valid = sum(1 for r in rows if pattern.match(r))
        check("3. Rows have valid FC-NNN Component IDs", 1,
              valid == len(rows), f"{valid}/{len(rows)} valid")
    except Exception as e:
        check("3. Rows have valid FC-NNN Component IDs", 1, False, f"exception: {e}")


def check_4_rows_have_component_name() -> None:
    """Verify rows have non-empty Component Name."""
    try:
        table_id = _get_table_id()
        if not table_id:
            check("4. Rows have non-empty Component Name", 1, False, "table not found")
            return
        field_result = baserow_sql(
            f"SELECT df.id FROM database_field df "
            f"WHERE df.table_id = {table_id} AND df.name = 'Component Name';"
        )
        if not field_result.strip():
            check("4. Rows have non-empty Component Name", 1, False, "field not found")
            return
        field_id = field_result.strip().split('\n')[0].strip()
        row_result = baserow_sql(
            f"SELECT COUNT(*) FROM database_table_{table_id} "
            f"WHERE trashed = false AND field_{field_id} IS NOT NULL "
            f"AND field_{field_id} != '';"
        )
        non_empty = int(row_result.strip().split('\n')[-1])
        total_result = baserow_sql(
            f"SELECT COUNT(*) FROM database_table_{table_id} WHERE trashed = false;"
        )
        total = int(total_result.strip().split('\n')[-1])
        check("4. Rows have non-empty Component Name", 1,
              non_empty == total and total > 0, f"{non_empty}/{total} non-empty")
    except Exception as e:
        check("4. Rows have non-empty Component Name", 1, False, f"exception: {e}")


def check_5_category_values_correct() -> None:
    """Verify Category field uses only allowed single-select values."""
    allowed = {"Layout", "Form", "Display", "Navigation", "Chart", "Utility"}
    try:
        table_id = _get_table_id()
        if not table_id:
            check("5. Category values are valid", 2, False, "table not found")
            return
        field_result = baserow_sql(
            f"SELECT df.id FROM database_field df "
            f"WHERE df.table_id = {table_id} AND df.name = 'Category';"
        )
        if not field_result.strip():
            check("5. Category values are valid", 2, False, "Category field not found")
            return
        field_id = field_result.strip().split('\n')[0].strip()

        # Single-select stores option IDs; get the option values
        options_result = baserow_sql(
            f"SELECT so.id, so.value FROM database_selectoption so "
            f"WHERE so.field_id = {field_id};"
        )
        option_map = {}
        for line in options_result.strip().split('\n'):
            if '|' in line:
                parts = line.split('|')
                option_map[parts[0].strip()] = parts[1].strip()

        # Check that all option values are in allowed set
        all_valid = all(v in allowed for v in option_map.values())
        # Check that rows use these options
        row_result = baserow_sql(
            f"SELECT DISTINCT field_{field_id} FROM database_table_{table_id} "
            f"WHERE trashed = false AND field_{field_id} IS NOT NULL;"
        )
        used_ids = [r.strip() for r in row_result.strip().split('\n') if r.strip()]
        used_values = [option_map.get(uid, f"unknown({uid})") for uid in used_ids]
        invalid = [v for v in used_values if v not in allowed and not v.startswith("unknown")]
        check("5. Category values are valid", 2,
              all_valid and len(invalid) == 0,
              f"options={list(option_map.values())}, used={used_values}")
    except Exception as e:
        check("5. Category values are valid", 2, False, f"exception: {e}")


def check_6_deprecation_candidate_logic() -> None:
    """Verify Deprecation Candidate=true iff Usage Count <= 1."""
    try:
        table_id = _get_table_id()
        if not table_id:
            check("6. Deprecation Candidate logic correct", 2, False, "table not found")
            return
        uc_field = baserow_sql(
            f"SELECT df.id FROM database_field df "
            f"WHERE df.table_id = {table_id} AND df.name = 'Usage Count';"
        ).strip().split('\n')[0].strip()
        dc_field = baserow_sql(
            f"SELECT df.id FROM database_field df "
            f"WHERE df.table_id = {table_id} AND df.name = 'Deprecation Candidate';"
        ).strip().split('\n')[0].strip()
        if not uc_field or not dc_field:
            check("6. Deprecation Candidate logic correct", 2, False, "fields not found")
            return
        # Check rows where deprecation candidate doesn't match usage count logic
        # Deprecation Candidate is boolean; Usage Count is number
        # True when usage_count <= 1
        wrong_result = baserow_sql(
            f"SELECT COUNT(*) FROM database_table_{table_id} "
            f"WHERE trashed = false AND ("
            f"  (COALESCE(field_{uc_field}::numeric, 0) <= 1 AND field_{dc_field} IS NOT TRUE) OR "
            f"  (COALESCE(field_{uc_field}::numeric, 0) > 1 AND field_{dc_field} IS TRUE)"
            f");"
        )
        wrong_count = int(wrong_result.strip().split('\n')[-1])
        total_result = baserow_sql(
            f"SELECT COUNT(*) FROM database_table_{table_id} WHERE trashed = false;"
        )
        total = int(total_result.strip().split('\n')[-1])
        check("6. Deprecation Candidate logic correct", 2,
              wrong_count == 0 and total > 0,
              f"{wrong_count} incorrect out of {total}")
    except Exception as e:
        check("6. Deprecation Candidate logic correct", 2, False, f"exception: {e}")


def check_7_captured_at_date() -> None:
    """Verify Captured At is 2026-03-20 for all rows."""
    try:
        table_id = _get_table_id()
        if not table_id:
            check("7. Captured At = 2026-03-20", 1, False, "table not found")
            return
        field_result = baserow_sql(
            f"SELECT df.id FROM database_field df "
            f"WHERE df.table_id = {table_id} AND df.name = 'Captured At';"
        )
        if not field_result.strip():
            check("7. Captured At = 2026-03-20", 1, False, "field not found")
            return
        field_id = field_result.strip().split('\n')[0].strip()
        # Date field in Baserow stores as date type
        wrong = baserow_sql(
            f"SELECT COUNT(*) FROM database_table_{table_id} "
            f"WHERE trashed = false AND "
            f"(field_{field_id} IS NULL OR field_{field_id}::date != '2026-03-20');"
        )
        wrong_count = int(wrong.strip().split('\n')[-1])
        check("7. Captured At = 2026-03-20", 1,
              wrong_count == 0, f"{wrong_count} rows with wrong date")
    except Exception as e:
        check("7. Captured At = 2026-03-20", 1, False, f"exception: {e}")


def check_8_grid_view_high_impact() -> None:
    """Verify Grid view 'High-Impact Components' exists."""
    try:
        table_id = _get_table_id()
        if not table_id:
            check("8. Grid view 'High-Impact Components' exists", 1, False, "table not found")
            return
        result = baserow_sql(
            f"SELECT COUNT(*) FROM database_view "
            f"WHERE table_id = {table_id} AND name = 'High-Impact Components';"
        )
        count = int(result.strip().split('\n')[-1])
        check("8. Grid view 'High-Impact Components' exists", 1,
              count >= 1, f"found {count}")
    except Exception as e:
        check("8. Grid view 'High-Impact Components' exists", 1, False, f"exception: {e}")


def check_9_kanban_view_by_category() -> None:
    """Verify Kanban view 'By Category' exists."""
    try:
        table_id = _get_table_id()
        if not table_id:
            check("9. Kanban view 'By Category' exists", 1, False, "table not found")
            return
        result = baserow_sql(
            f"SELECT COUNT(*) FROM database_view "
            f"WHERE table_id = {table_id} AND name = 'By Category';"
        )
        count = int(result.strip().split('\n')[-1])
        check("9. Kanban view 'By Category' exists", 1,
              count >= 1, f"found {count}")
    except Exception as e:
        check("9. Kanban view 'By Category' exists", 1, False, f"exception: {e}")


# ── code-server checks ───────────────────────────────────────────────────────

def check_10_components_md_exists() -> None:
    """Verify tabler/docs/COMPONENTS.md exists in code-server."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "test", "-f", "/home/coder/tabler/docs/COMPONENTS.md",
        )
        check("10. COMPONENTS.md exists", 1, rc == 0,
              "file found" if rc == 0 else "file not found")
    except Exception as e:
        check("10. COMPONENTS.md exists", 1, False, f"exception: {e}")


def check_11_components_md_header() -> None:
    """Verify COMPONENTS.md has correct header lines (line 1-2)."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "head", "-n", "2", "/home/coder/tabler/docs/COMPONENTS.md",
        )
        if rc != 0:
            check("11. COMPONENTS.md header lines correct", 2, False, "cannot read file")
            return
        lines = out.split('\n')
        line1_ok = len(lines) >= 1 and lines[0].strip() == "# tabler Component Inventory"
        line2_ok = len(lines) >= 2 and lines[1].strip() == "Captured: 2026-03-20"
        check("11. COMPONENTS.md header lines correct", 2,
              line1_ok and line2_ok,
              f"line1={'OK' if line1_ok else repr(lines[0] if lines else '')}, "
              f"line2={'OK' if line2_ok else repr(lines[1] if len(lines) > 1 else '')}")
    except Exception as e:
        check("11. COMPONENTS.md header lines correct", 2, False, f"exception: {e}")


def check_12_components_md_counts() -> None:
    """Verify COMPONENTS.md lines 3-4 have Total components and Deprecation candidates."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "head", "-n", "4", "/home/coder/tabler/docs/COMPONENTS.md",
        )
        if rc != 0:
            check("12. COMPONENTS.md count lines correct", 2, False, "cannot read file")
            return
        lines = out.split('\n')
        line3_ok = len(lines) >= 3 and re.match(r'^Total components: \d+$', lines[2].strip())
        line4_ok = len(lines) >= 4 and re.match(r'^Deprecation candidates: \d+$', lines[3].strip())
        check("12. COMPONENTS.md count lines correct", 2,
              line3_ok and line4_ok,
              f"line3={'OK' if line3_ok else repr(lines[2] if len(lines) > 2 else '')}, "
              f"line4={'OK' if line4_ok else repr(lines[3] if len(lines) > 3 else '')}")
    except Exception as e:
        check("12. COMPONENTS.md count lines correct", 2, False, f"exception: {e}")


def check_13_components_md_entries_format() -> None:
    """Verify component entry lines follow the expected format."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "cat", "/home/coder/tabler/docs/COMPONENTS.md",
            timeout=15,
        )
        if rc != 0:
            check("13. COMPONENTS.md entry lines formatted correctly", 2, False, "cannot read file")
            return
        lines = out.strip().split('\n')
        # Lines 5+ (index 4+) should be component entries
        entry_lines = [l for l in lines[4:] if l.strip()]
        if not entry_lines:
            check("13. COMPONENTS.md entry lines formatted correctly", 2, False, "no entry lines found")
            return
        # Expected: "- <Name> (<Category>, used <N>x) — <Path>:<Line>"
        # Use a flexible pattern
        pattern = re.compile(
            r'^- .+ \((Layout|Form|Display|Navigation|Chart|Utility), used \d+x\) — .+:\d+$'
        )
        valid = sum(1 for l in entry_lines if pattern.match(l.strip()))
        check("13. COMPONENTS.md entry lines formatted correctly", 2,
              valid == len(entry_lines),
              f"{valid}/{len(entry_lines)} match expected format")
    except Exception as e:
        check("13. COMPONENTS.md entry lines formatted correctly", 2, False, f"exception: {e}")


def check_14_git_commit_message() -> None:
    """Verify git commit with exact message 'docs: add component inventory 2026-03-20'."""
    try:
        rc, out, err = docker_exec(
            CODE_SERVER_CONTAINER,
            "bash", "-c",
            "cd /home/coder/tabler && git log --oneline --all --grep='docs: add component inventory 2026-03-20' --format='%s'",
            timeout=15,
        )
        commits = [l.strip() for l in out.strip().split('\n') if l.strip()]
        exact = any(c == "docs: add component inventory 2026-03-20" for c in commits)
        check("14. Git commit with exact message", 2,
              exact, f"found commits: {commits[:3]}")
    except Exception as e:
        check("14. Git commit with exact message", 2, False, f"exception: {e}")


# ── OpenProject checks ───────────────────────────────────────────────────────

def check_15_openproject_tasks_exist() -> None:
    """Verify OpenProject has Task work packages with 'Review deprecation:' subjects in demo-project."""
    try:
        result = openproject_sql(
            "SELECT wp.subject FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.identifier = 'demo-project' "
            "AND t.name = 'Task' "
            "AND wp.subject LIKE 'Review deprecation:%';"
        )
        subjects = [l.strip() for l in result.strip().split('\n') if l.strip()]
        check("15. OpenProject 'Review deprecation' tasks exist", 2,
              len(subjects) >= 1,
              f"found {len(subjects)} tasks")
    except Exception as e:
        check("15. OpenProject 'Review deprecation' tasks exist", 2, False, f"exception: {e}")


def check_16_openproject_task_assignee() -> None:
    """Verify OpenProject tasks are assigned to OpenProject Admin with Normal priority."""
    try:
        # Get tasks assigned to admin with priority Normal
        result = openproject_sql(
            "SELECT wp.subject, u.login, e.name AS priority "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "LEFT JOIN users u ON wp.assigned_to_id = u.id "
            "LEFT JOIN enumerations e ON wp.priority_id = e.id "
            "WHERE p.identifier = 'demo-project' "
            "AND t.name = 'Task' "
            "AND wp.subject LIKE 'Review deprecation:%';"
        )
        lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
        if not lines:
            check("16. OpenProject tasks: assignee=admin, priority=Normal", 2, False, "no tasks found")
            return
        all_ok = True
        details = []
        for line in lines:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                assignee_ok = parts[1] == "admin"
                priority_ok = parts[2] == "Normal"
                if not (assignee_ok and priority_ok):
                    all_ok = False
                    details.append(f"assignee={parts[1]}, priority={parts[2]}")
            else:
                all_ok = False
                details.append(f"unexpected format: {line}")
        check("16. OpenProject tasks: assignee=admin, priority=Normal", 2,
              all_ok, "; ".join(details) if details else f"{len(lines)} tasks OK")
    except Exception as e:
        check("16. OpenProject tasks: assignee=admin, priority=Normal", 2, False, f"exception: {e}")


def check_17_openproject_task_descriptions() -> None:
    """Verify task descriptions match expected format."""
    try:
        result = openproject_sql(
            "SELECT wp.subject, wp.description "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.identifier = 'demo-project' "
            "AND t.name = 'Task' "
            "AND wp.subject LIKE 'Review deprecation:%';"
        )
        lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
        if not lines:
            check("17. OpenProject task descriptions match format", 2, False, "no tasks found")
            return
        # Expected format: "File: <path>:<line>; Usage: <N>; Category: <cat>; Captured: 2026-03-20"
        pattern = re.compile(r'File: .+:\d+; Usage: \d+; Category: \w+; Captured: 2026-03-20')
        matched = sum(1 for l in lines if pattern.search(l))
        check("17. OpenProject task descriptions match format", 2,
              matched == len(lines),
              f"{matched}/{len(lines)} match expected format")
    except Exception as e:
        check("17. OpenProject task descriptions match format", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_database_exists()
    check_2_baserow_table_exists()
    check_3_rows_have_valid_component_ids()
    check_4_rows_have_component_name()
    check_5_category_values_correct()
    check_6_deprecation_candidate_logic()
    check_7_captured_at_date()
    check_8_grid_view_high_impact()
    check_9_kanban_view_by_category()
    check_10_components_md_exists()
    check_11_components_md_header()
    check_12_components_md_counts()
    check_13_components_md_entries_format()
    check_14_git_commit_message()
    check_15_openproject_tasks_exist()
    check_16_openproject_task_assignee()
    check_17_openproject_task_descriptions()

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
