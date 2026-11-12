# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_039/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-039-I5: Cross-team integration testing infrastructure for Sprint-2026-Q4-W1

Checks: 12 weighted checks across code-server, openproject, baserow.
Strategy: docker exec (filesystem for code-server, DB for openproject/baserow)

Required env vars:
  SERVER_HOSTNAME,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER
"""

import os
import sys
import subprocess
import json

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")

for var in ["CODE_SERVER_CONTAINER", "OPENPROJECT_CONTAINER", "BASEROW_CONTAINER", "BASEROW_DB_CONTAINER"]:
    if not os.environ.get(var):
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)

for var in ["CODE_SERVER_PORT", "OPENPROJECT_PORT", "BASEROW_PORT"]:
    if not os.environ.get(var):
        print(f"FATAL: {var} not set", file=sys.stderr)
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


def op_sql(query: str) -> str:
    """Run SQL against OpenProject's embedded Postgres (peer auth via su - postgres)."""
    r = subprocess.run(
        ["docker", "exec", "-i", OPENPROJECT_CONTAINER,
         "su", "-", "postgres", "-c", "psql -d openproject -t -A"],
        input=query, capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"OpenProject psql failed: {r.stderr.strip()}")
    return r.stdout.strip()


def baserow_sql(query: str) -> str:
    """Run SQL against Baserow's Postgres DB container."""
    rc, out, err = docker_exec(
        BASEROW_DB_CONTAINER,
        "env", "PGPASSWORD=kdpzkuyhsgb22onku8y7rxkx3czej88nxpngaz4mlmgad67vpc", "psql", "-h", "127.0.0.1", "-U", "baserow", "-d", "baserow", "-t", "-A", "-c", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"Baserow psql failed: {err.strip()}")
    return out.strip()


def find_file_in_container(container: str, filename: str, search_root: str = "/") -> str:
    """Find a file path inside a container."""
    rc, out, err = docker_exec(
        container, "find", search_root, "-path", f"*/{filename}", "-type", "f",
        timeout=30,
    )
    paths = [p.strip() for p in out.strip().splitlines() if p.strip()]
    return paths[0] if paths else ""


# ── Constants from task ───────────────────────────────────────────────────────
SPRINT = "Sprint-2026-Q4-W1"
COMMIT_MSG = "docs: mark integration point for Sprint-2026-Q4-W1"
EXTENSIONS_COMMENTS = [
    f"# INTEGRATION-POINT {SPRINT}: consumed by Insights Engineering Team/data-analyzer",
    "# Contract owner: Yuki Tanaka",
    "# Review cadence: every sprint",
]
RUN_ANALYSIS_COMMENTS = [
    f"# INTEGRATION-POINT {SPRINT}: consumes Producer API Team/todo-api",
    "# Contract owner: Olivia Bennett",
]
WP_SPECS = [
    {"subject": "Publish reminder-events topic [CTR-REMIND-EVT]", "hours": 13, "assignee": "Yuki Tanaka"},
    {"subject": "Expose subtask-graph API [CTR-SUBTASK-GR]", "hours": 11, "assignee": "Yuki Tanaka"},
    {"subject": "Provide attachment-index export [CTR-ATTACH-IDX]", "hours": 8, "assignee": "Yuki Tanaka"},
    {"subject": "Consume reminder-events topic for engagement metrics [CTR-REMIND-EVT]", "hours": 12, "assignee": "Olivia Bennett"},
    {"subject": "Ingest subtask-graph API for dependency analytics [CTR-SUBTASK-GR]", "hours": 10, "assignee": "Olivia Bennett"},
]
AGENDA_ITEMS = [
    "Contract changes since last sync",
    "Integration test results",
    "Blockers and escalations",
]


# ── Individual checks ─────────────────────────────────────────────────────────
def check_1_extensions_comments() -> None:
    """Verify extensions.py has the 3 integration-point comment lines after anchor."""
    try:
        path = find_file_in_container(
            CODE_SERVER_CONTAINER, "todo-api/app/extensions.py",
            search_root="/home",
        )
        if not path:
            # Try broader search
            path = find_file_in_container(
                CODE_SERVER_CONTAINER, "extensions.py",
                search_root="/",
            )
        if not path:
            check("1. extensions.py integration comments", 2, False, "file not found in container")
            return
        rc, content, _ = docker_exec(CODE_SERVER_CONTAINER, "cat", path)
        if rc != 0:
            check("1. extensions.py integration comments", 2, False, f"cannot read {path}")
            return
        missing = [c for c in EXTENSIONS_COMMENTS if c not in content]
        if missing:
            check("1. extensions.py integration comments", 2, False,
                  f"missing {len(missing)} comment(s): {missing[0][:60]}...")
        else:
            # Check they appear after anchor line
            anchor_pos = content.find("migrate = Migrate()")
            if anchor_pos < 0:
                check("1. extensions.py integration comments", 2, False, "anchor line not found")
                return
            first_comment_pos = content.find(EXTENSIONS_COMMENTS[0])
            if first_comment_pos > anchor_pos:
                check("1. extensions.py integration comments", 2, True)
            else:
                check("1. extensions.py integration comments", 2, False,
                      "comments not after anchor line")
    except Exception as e:
        check("1. extensions.py integration comments", 2, False, f"exception: {e}")


def check_2_run_analysis_comments() -> None:
    """Verify run_analysis.py has the 2 integration-point comment lines after anchor."""
    try:
        path = find_file_in_container(
            CODE_SERVER_CONTAINER, "data-analyzer/scripts/run_analysis.py",
            search_root="/home",
        )
        if not path:
            path = find_file_in_container(
                CODE_SERVER_CONTAINER, "run_analysis.py",
                search_root="/",
            )
        if not path:
            check("2. run_analysis.py integration comments", 2, False, "file not found in container")
            return
        rc, content, _ = docker_exec(CODE_SERVER_CONTAINER, "cat", path)
        if rc != 0:
            check("2. run_analysis.py integration comments", 2, False, f"cannot read {path}")
            return
        missing = [c for c in RUN_ANALYSIS_COMMENTS if c not in content]
        if missing:
            check("2. run_analysis.py integration comments", 2, False,
                  f"missing {len(missing)} comment(s): {missing[0][:60]}...")
        else:
            anchor_pos = content.find("def run_pipeline(input_path):")
            if anchor_pos < 0:
                check("2. run_analysis.py integration comments", 2, False, "anchor line not found")
                return
            first_comment_pos = content.find(RUN_ANALYSIS_COMMENTS[0])
            if first_comment_pos > anchor_pos:
                check("2. run_analysis.py integration comments", 2, True)
            else:
                check("2. run_analysis.py integration comments", 2, False,
                      "comments not after anchor line")
    except Exception as e:
        check("2. run_analysis.py integration comments", 2, False, f"exception: {e}")


def check_3_git_commits() -> None:
    """Verify two separate commits with exact message in the repos."""
    try:
        # Find git repos for todo-api and data-analyzer
        commit_count = 0
        for project in ["todo-api", "data-analyzer"]:
            # Find the project directory
            rc, out, _ = docker_exec(
                CODE_SERVER_CONTAINER,
                "find", "/home", "-type", "d", "-name", project, "-maxdepth", "4",
                timeout=15,
            )
            dirs = [d.strip() for d in out.strip().splitlines() if d.strip()]
            if not dirs:
                # Try root
                rc, out, _ = docker_exec(
                    CODE_SERVER_CONTAINER,
                    "find", "/", "-type", "d", "-name", project, "-maxdepth", "4",
                    timeout=15,
                )
                dirs = [d.strip() for d in out.strip().splitlines() if d.strip()]
            if not dirs:
                continue
            proj_dir = dirs[0]
            rc, log_out, _ = docker_exec(
                CODE_SERVER_CONTAINER,
                "bash", "-c", f"cd {proj_dir} && git log --all --oneline --format='%s'",
                timeout=15,
            )
            if COMMIT_MSG in log_out:
                commit_count += 1

        if commit_count >= 2:
            check("3. Git commits with correct message", 2, True, f"found in {commit_count} repos")
        elif commit_count == 1:
            check("3. Git commits with correct message", 2, False, "found in only 1 repo, expected 2")
        else:
            check("3. Git commits with correct message", 2, False, "commit message not found in any repo")
    except Exception as e:
        check("3. Git commits with correct message", 2, False, f"exception: {e}")


def check_4_op_version() -> None:
    """Verify OpenProject version Sprint-2026-Q4-W1 with correct dates and description."""
    try:
        row = op_sql(
            "SELECT v.name, v.description, v.start_date, v.effective_date, v.status "
            "FROM versions v "
            "JOIN projects p ON v.project_id = p.id "
            "WHERE p.identifier = 'demo-project' "
            f"AND v.name = '{SPRINT}'"
        )
        if not row:
            check("4. Version Sprint-2026-Q4-W1", 2, False, "version not found")
            return
        parts = row.split("|")
        issues = []
        if len(parts) < 5:
            check("4. Version Sprint-2026-Q4-W1", 2, False, f"unexpected row format: {row[:100]}")
            return
        name, desc, start, due, status = parts[0], parts[1], parts[2], parts[3], parts[4]
        expected_desc = "Cross-team sprint: Producer API Team ↔ Insights Engineering Team"
        if expected_desc not in desc and "Producer API Team" not in desc:
            issues.append(f"description mismatch: '{desc[:80]}'")
        if "2026-10-05" not in start:
            issues.append(f"start_date={start}, expected 2026-10-05")
        if "2026-10-16" not in due:
            issues.append(f"due_date={due}, expected 2026-10-16")
        if status and status != "open":
            issues.append(f"status={status}, expected open")
        if issues:
            check("4. Version Sprint-2026-Q4-W1", 2, False, "; ".join(issues))
        else:
            check("4. Version Sprint-2026-Q4-W1", 2, True)
    except Exception as e:
        check("4. Version Sprint-2026-Q4-W1", 2, False, f"exception: {e}")


def check_5_wp_subjects() -> None:
    """Verify 5 Feature work packages with correct subjects in version."""
    try:
        rows = op_sql(
            "SELECT wp.subject "
            "FROM work_packages wp "
            "JOIN types t ON wp.type_id = t.id "
            "JOIN versions v ON wp.version_id = v.id "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.identifier = 'demo-project' "
            f"AND v.name = '{SPRINT}' "
            "AND t.name = 'Feature' "
            "ORDER BY wp.subject"
        )
        found_subjects = set(r.strip() for r in rows.splitlines() if r.strip())
        expected_subjects = set(s["subject"] for s in WP_SPECS)
        missing = expected_subjects - found_subjects
        extra = found_subjects - expected_subjects
        if not missing and len(found_subjects) == 5:
            check("5. Five Feature WPs with correct subjects", 2, True)
        else:
            detail = f"found {len(found_subjects)}/5"
            if missing:
                detail += f"; missing: {list(missing)[0][:50]}..."
            check("5. Five Feature WPs with correct subjects", 2, False, detail)
    except Exception as e:
        check("5. Five Feature WPs with correct subjects", 2, False, f"exception: {e}")


def check_6_wp_estimated_hours() -> None:
    """Verify WPs have correct estimated hours."""
    try:
        rows = op_sql(
            "SELECT wp.subject, wp.estimated_hours "
            "FROM work_packages wp "
            "JOIN types t ON wp.type_id = t.id "
            "JOIN versions v ON wp.version_id = v.id "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.identifier = 'demo-project' "
            f"AND v.name = '{SPRINT}' "
            "AND t.name = 'Feature'"
        )
        wp_hours = {}
        for line in rows.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("|", 1)
            if len(parts) == 2:
                subj = parts[0].strip()
                hrs = parts[1].strip()
                wp_hours[subj] = hrs

        issues = []
        for spec in WP_SPECS:
            subj = spec["subject"]
            expected_h = spec["hours"]
            actual_h = wp_hours.get(subj, None)
            if actual_h is None:
                issues.append(f"WP '{subj[:30]}' not found")
            else:
                try:
                    if abs(float(actual_h) - expected_h) > 0.01:
                        issues.append(f"'{subj[:30]}': {actual_h}h, expected {expected_h}h")
                except ValueError:
                    issues.append(f"'{subj[:30]}': hours='{actual_h}'")

        if issues:
            check("6. WP estimated hours", 2, False, "; ".join(issues[:3]))
        else:
            check("6. WP estimated hours", 2, True)
    except Exception as e:
        check("6. WP estimated hours", 2, False, f"exception: {e}")


def check_7_wp_assignees() -> None:
    """Verify WPs have correct assignees."""
    try:
        rows = op_sql(
            "SELECT wp.subject, u.firstname || ' ' || u.lastname AS assignee "
            "FROM work_packages wp "
            "JOIN types t ON wp.type_id = t.id "
            "JOIN versions v ON wp.version_id = v.id "
            "JOIN projects p ON wp.project_id = p.id "
            "LEFT JOIN users u ON wp.assigned_to_id = u.id "
            "WHERE p.identifier = 'demo-project' "
            f"AND v.name = '{SPRINT}' "
            "AND t.name = 'Feature'"
        )
        wp_assignees = {}
        for line in rows.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("|", 1)
            if len(parts) == 2:
                wp_assignees[parts[0].strip()] = parts[1].strip()

        issues = []
        for spec in WP_SPECS:
            subj = spec["subject"]
            expected_a = spec["assignee"]
            actual_a = wp_assignees.get(subj, "")
            if expected_a.lower() not in actual_a.lower():
                issues.append(f"'{subj[:30]}': assignee='{actual_a}', expected '{expected_a}'")

        if issues:
            check("7. WP assignees", 2, False, "; ".join(issues[:3]))
        else:
            check("7. WP assignees", 2, True)
    except Exception as e:
        check("7. WP assignees", 2, False, f"exception: {e}")


def check_8_follows_relations() -> None:
    """Verify follows relations between matching producer/consumer WP pairs."""
    try:
        # Get WP IDs and subjects
        rows = op_sql(
            "SELECT wp.id, wp.subject "
            "FROM work_packages wp "
            "JOIN versions v ON wp.version_id = v.id "
            "JOIN projects p ON wp.project_id = p.id "
            "WHERE p.identifier = 'demo-project' "
            f"AND v.name = '{SPRINT}'"
        )
        wp_map = {}
        for line in rows.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                wp_map[parts[1].strip()] = parts[0].strip()

        # Check relations for CTR-REMIND-EVT and CTR-SUBTASK-GR pairs
        expected_relations = 0
        found_relations = 0
        contract_ids = ["CTR-REMIND-EVT", "CTR-SUBTASK-GR"]

        for cid in contract_ids:
            producer_id = None
            consumer_id = None
            for subj, wpid in wp_map.items():
                if cid in subj:
                    # Producer = Producer API Team (Yuki Tanaka's WPs)
                    if any(s["subject"] == subj and s["assignee"] == "Yuki Tanaka" for s in WP_SPECS):
                        producer_id = wpid
                    elif any(s["subject"] == subj and s["assignee"] == "Olivia Bennett" for s in WP_SPECS):
                        consumer_id = wpid
            if producer_id and consumer_id:
                expected_relations += 1
                # "follows" relation: consumer follows producer
                # In OpenProject, relation_type could be 'follows' or 'precedes'
                # "follows" from WP_B to WP_A means WP_B.from_id follows WP_A.to_id
                rel = op_sql(
                    "SELECT count(*) FROM relations "
                    f"WHERE ((from_id = {consumer_id} AND to_id = {producer_id} AND relation_type = 'follows') "
                    f"OR (from_id = {producer_id} AND to_id = {consumer_id} AND relation_type = 'precedes'))"
                )
                if rel and int(rel) > 0:
                    found_relations += 1

        if expected_relations == 0:
            check("8. Follows relations", 2, False, "could not identify WP pairs")
        elif found_relations == expected_relations:
            check("8. Follows relations", 2, True, f"{found_relations}/{expected_relations}")
        else:
            check("8. Follows relations", 2, False,
                  f"found {found_relations}/{expected_relations} relations")
    except Exception as e:
        check("8. Follows relations", 2, False, f"exception: {e}")


def check_9_meeting_and_agenda() -> None:
    """Verify recurring meeting with title and 3 agenda items in order."""
    try:
        meeting_title = f"Cross-team sync: {SPRINT}"
        # Try to find the meeting
        meeting_row = op_sql(
            "SELECT m.id, m.title "
            "FROM meetings m "
            "JOIN projects p ON m.project_id = p.id "
            "WHERE p.identifier = 'demo-project' "
            f"AND m.title = '{meeting_title}' "
            "LIMIT 1"
        )
        if not meeting_row:
            # Try partial match
            meeting_row = op_sql(
                "SELECT m.id, m.title "
                "FROM meetings m "
                "JOIN projects p ON m.project_id = p.id "
                "WHERE p.identifier = 'demo-project' "
                f"AND m.title LIKE '%{SPRINT}%' "
                "LIMIT 1"
            )
        if not meeting_row:
            check("9. Meeting and agenda items", 2, False, "meeting not found")
            return

        parts = meeting_row.split("|", 1)
        meeting_id = parts[0].strip()

        # Check agenda items
        agenda_rows = op_sql(
            "SELECT title FROM meeting_agenda_items "
            f"WHERE meeting_id = {meeting_id} "
            "ORDER BY position ASC, id ASC"
        )
        if not agenda_rows:
            # Try meeting_contents or structured_meeting_agenda_items
            agenda_rows = op_sql(
                "SELECT title FROM meeting_agenda_items "
                f"WHERE meeting_id IN (SELECT id FROM meetings WHERE "
                f"title LIKE '%{SPRINT}%') "
                "ORDER BY position ASC, id ASC"
            )

        found_items = [r.strip() for r in agenda_rows.splitlines() if r.strip()]
        issues = []

        if len(found_items) != 3:
            issues.append(f"found {len(found_items)} agenda items, expected 3")
        else:
            for i, expected in enumerate(AGENDA_ITEMS):
                if found_items[i] != expected:
                    issues.append(f"item {i+1}: '{found_items[i][:40]}', expected '{expected}'")

        if issues:
            check("9. Meeting and agenda items", 2, False, "; ".join(issues))
        else:
            check("9. Meeting and agenda items", 2, True)
    except Exception as e:
        check("9. Meeting and agenda items", 2, False, f"exception: {e}")


def check_10_baserow_db_and_table() -> None:
    """Verify Baserow database and table exist."""
    try:
        db_name = "Q4W1 Integration Contracts Workspace"
        table_name = "Integration Contracts"

        # Find the database (application) in Baserow
        db_row = baserow_sql(
            f"SELECT id FROM core_application WHERE name = '{db_name}'"
        )
        if not db_row:
            check("10. Baserow DB and table exist", 1, False, f"database '{db_name}' not found")
            return

        db_id = db_row.strip().splitlines()[0].strip()

        # Find the table
        table_row = baserow_sql(
            f"SELECT id FROM database_table WHERE database_id = {db_id} "
            f"AND name = '{table_name}'"
        )
        if not table_row:
            check("10. Baserow DB and table exist", 1, False, f"table '{table_name}' not found")
            return

        check("10. Baserow DB and table exist", 1, True)
    except Exception as e:
        check("10. Baserow DB and table exist", 1, False, f"exception: {e}")


def check_11_baserow_rows() -> None:
    """Verify table has 2 rows with correct contract data."""
    try:
        db_name = "Q4W1 Integration Contracts Workspace"
        table_name = "Integration Contracts"

        # Get the table ID
        table_id_raw = baserow_sql(
            "SELECT dt.id FROM database_table dt "
            "JOIN core_application ca ON dt.database_id = ca.id "
            f"WHERE ca.name = '{db_name}' AND dt.name = '{table_name}'"
        )
        if not table_id_raw:
            check("11. Baserow contract rows", 2, False, "table not found")
            return

        table_id = table_id_raw.strip().splitlines()[0].strip()

        # Count rows in the dynamic row table
        row_count_raw = baserow_sql(
            f"SELECT count(*) FROM database_table_{table_id}"
        )
        row_count = int(row_count_raw.strip())

        if row_count != 2:
            check("11. Baserow contract rows", 2, False, f"found {row_count} rows, expected 2")
            return

        # Get field info to find field IDs
        fields_raw = baserow_sql(
            f"SELECT df.id, df.name FROM database_field df "
            f"WHERE df.table_id = {table_id} ORDER BY df.id"
        )
        field_map = {}
        for line in fields_raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                field_map[parts[1].strip()] = parts[0].strip()

        # Check primary field (Contract ID) values
        contract_id_field = field_map.get("Contract ID")
        status_field = field_map.get("Status")

        issues = []
        if contract_id_field:
            cid_vals = baserow_sql(
                f"SELECT field_{contract_id_field} FROM database_table_{table_id} "
                f"ORDER BY field_{contract_id_field} ASC"
            )
            cid_list = [r.strip() for r in cid_vals.splitlines() if r.strip()]
            if "IC-01" not in cid_list or "IC-02" not in cid_list:
                issues.append(f"Contract IDs: {cid_list}, expected IC-01, IC-02")
        else:
            issues.append("Contract ID field not found")

        if status_field:
            status_vals = baserow_sql(
                f"SELECT field_{status_field} FROM database_table_{table_id}"
            )
            for line in status_vals.splitlines():
                val = line.strip()
                if val and "Planned" not in val:
                    issues.append(f"Status value '{val}', expected 'Planned'")
                    break

        if issues:
            check("11. Baserow contract rows", 2, False, "; ".join(issues))
        else:
            check("11. Baserow contract rows", 2, True)
    except Exception as e:
        check("11. Baserow contract rows", 2, False, f"exception: {e}")


def check_12_baserow_grid_view() -> None:
    """Verify Grid view 'Contract Matrix' exists."""
    try:
        db_name = "Q4W1 Integration Contracts Workspace"
        table_name = "Integration Contracts"

        table_id_raw = baserow_sql(
            "SELECT dt.id FROM database_table dt "
            "JOIN core_application ca ON dt.database_id = ca.id "
            f"WHERE ca.name = '{db_name}' AND dt.name = '{table_name}'"
        )
        if not table_id_raw:
            check("12. Grid view Contract Matrix", 1, False, "table not found")
            return

        table_id = table_id_raw.strip().splitlines()[0].strip()

        view_row = baserow_sql(
            f"SELECT v.id, v.name FROM database_view v "
            f"WHERE v.table_id = {table_id} AND v.name = 'Contract Matrix'"
        )
        if not view_row:
            check("12. Grid view Contract Matrix", 1, False, "view 'Contract Matrix' not found")
        else:
            check("12. Grid view Contract Matrix", 1, True)
    except Exception as e:
        check("12. Grid view Contract Matrix", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_extensions_comments()
    check_2_run_analysis_comments()
    check_3_git_commits()
    check_4_op_version()
    check_5_wp_subjects()
    check_6_wp_estimated_hours()
    check_7_wp_assignees()
    check_8_follows_relations()
    check_9_meeting_and_agenda()
    check_10_baserow_db_and_table()
    check_11_baserow_rows()
    check_12_baserow_grid_view()

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
