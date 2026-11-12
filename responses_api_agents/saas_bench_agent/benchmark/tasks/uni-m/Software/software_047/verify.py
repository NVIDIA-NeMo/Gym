# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_047/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-047-I2: Docker Compliance Audit for Multi-Service Deployables (2026-04-12)

Checks: 15 weighted checks across code-server, baserow, openproject.
Strategy: Baserow REST API, code-server docker exec, OpenProject REST API.

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

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

# Credentials
BASEROW_EMAIL = "admin@example.com"
BASEROW_PASSWORD = "Admin1234"

# Task constants
AUDIT_DATE = "2026-04-12"
DB_NAME = "Container Security Review 2026-04-12"
SERVICES = ["blog-engine", "devops-configs", "tabler", "todo-api"]
OP_PROJECT = "devops-automation"
SECURITY_OWNER = "sandra.love"
PLATFORM_OWNER = "richard.rethman"

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


_baserow_token: str | None = None


def baserow_auth() -> str:
    global _baserow_token
    if _baserow_token:
        return _baserow_token
    r = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": BASEROW_EMAIL, "password": BASEROW_PASSWORD},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    _baserow_token = data.get("access_token") or data.get("token")
    return _baserow_token


def baserow_get(path: str) -> dict | list:
    token = baserow_auth()
    r = requests.get(
        f"{BASEROW_URL}/api/{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def op_db_query(sql: str) -> str:
    """Query OpenProject's embedded Postgres via docker exec."""
    r = subprocess.run(
        [
            "docker", "exec",
            "-e", "PGPASSWORD=openproject",
            OPENPROJECT_CONTAINER,
            "psql", "-U", "openproject", "-d", "openproject",
            "-h", "127.0.0.1", "-t", "-A", "-c", sql,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"psql failed: {r.stderr.strip()}")
    return r.stdout.strip()


# ── Baserow helpers ───────────────────────────────────────────────────────────
_br_db_id: int | None = None
_br_tables: dict[str, int] = {}  # table name -> table id


def find_baserow_db() -> int | None:
    """Find the database by name, return its ID."""
    global _br_db_id
    if _br_db_id is not None:
        return _br_db_id
    apps = baserow_get("applications/")
    for app in apps:
        if app.get("name") == DB_NAME and app.get("type") == "database":
            _br_db_id = app["id"]
            return _br_db_id
    return None


def find_baserow_table(table_name: str) -> int | None:
    """Find a table by name within the target database."""
    if table_name in _br_tables:
        return _br_tables[table_name]
    db_id = find_baserow_db()
    if not db_id:
        return None
    tables = baserow_get(f"database/tables/database/{db_id}/")
    for t in tables:
        _br_tables[t["name"]] = t["id"]
    return _br_tables.get(table_name)


# ── Check 1: Baserow database exists ─────────────────────────────────────────
def check_1_baserow_db_exists() -> None:
    """Baserow database 'Container Security Review 2026-04-12' exists."""
    try:
        db_id = find_baserow_db()
        check("1. Baserow database exists", 2, db_id is not None,
              f"db_id={db_id}" if db_id else "database not found")
    except Exception as e:
        check("1. Baserow database exists", 2, False, f"exception: {e}")


# ── Check 2: Dockerfile Audit table exists with correct fields ────────────────
def check_2_dockerfile_audit_table() -> None:
    """Table 'Dockerfile Audit' exists with expected fields."""
    try:
        table_id = find_baserow_table("Dockerfile Audit")
        if not table_id:
            check("2. Dockerfile Audit table", 2, False, "table not found")
            return
        fields = baserow_get(f"database/fields/table/{table_id}/")
        field_names = {f["name"] for f in fields}
        expected_fields = {
            "Audit ID", "Service", "Dockerfile Path", "Base Image",
            "Uses Latest Tag", "Runs As Root", "Has Healthcheck",
            "Multistage Build", "Run Instruction Count", "Captured At",
            "Compliance Score",
        }
        missing = expected_fields - field_names
        check("2. Dockerfile Audit table", 2, not missing,
              f"missing fields: {missing}" if missing else "all fields present")
    except Exception as e:
        check("2. Dockerfile Audit table", 2, False, f"exception: {e}")


# ── Check 3: Dockerfile Audit has 4 rows with correct services ───────────────
def check_3_audit_rows() -> None:
    """Exactly 4 rows, one per service in alphabetical order."""
    try:
        table_id = find_baserow_table("Dockerfile Audit")
        if not table_id:
            check("3. Audit rows (4 services)", 2, False, "table not found")
            return
        data = baserow_get(f"database/rows/table/{table_id}/?size=100")
        rows = data.get("results", [])
        # Extract service values from rows - field may be named differently
        fields = baserow_get(f"database/fields/table/{table_id}/")
        service_field_id = None
        for f in fields:
            if f["name"] == "Service":
                service_field_id = f["id"]
                break
        services_found = []
        for row in rows:
            val = row.get(f"field_{service_field_id}", "")
            if isinstance(val, dict):
                val = val.get("value", "")
            services_found.append(str(val))
        has_all = set(SERVICES).issubset(set(services_found))
        correct_count = len(rows) == 4
        check("3. Audit rows (4 services)", 2,
              correct_count and has_all,
              f"rows={len(rows)}, services={services_found}")
    except Exception as e:
        check("3. Audit rows (4 services)", 2, False, f"exception: {e}")


# ── Check 4: Compliance Scores are correctly computed ─────────────────────────
def check_4_compliance_scores() -> None:
    """Compliance Score follows the deduction formula for each service."""
    try:
        table_id = find_baserow_table("Dockerfile Audit")
        if not table_id:
            check("4. Compliance scores", 3, False, "table not found")
            return
        fields = baserow_get(f"database/fields/table/{table_id}/")
        fmap = {f["name"]: f["id"] for f in fields}
        data = baserow_get(f"database/rows/table/{table_id}/?size=100")
        rows = data.get("results", [])

        def get_val(row, name):
            fid = fmap.get(name)
            if fid is None:
                return None
            return row.get(f"field_{fid}")

        def to_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes")
            return bool(v)

        issues = []
        for row in rows:
            svc = get_val(row, "Service")
            if isinstance(svc, dict):
                svc = svc.get("value", "")
            latest = to_bool(get_val(row, "Uses Latest Tag"))
            root = to_bool(get_val(row, "Runs As Root"))
            health = to_bool(get_val(row, "Has Healthcheck"))
            multi = to_bool(get_val(row, "Multistage Build"))
            run_count_raw = get_val(row, "Run Instruction Count")
            run_count = int(run_count_raw) if run_count_raw is not None else 0
            score_raw = get_val(row, "Compliance Score")
            score = float(score_raw) if score_raw is not None else -1

            expected = 100
            if latest:
                expected -= 25
            if root:
                expected -= 25
            if not health:
                expected -= 15
            if not multi:
                expected -= 10
            expected -= 5 * max(0, run_count - 6)
            expected = max(0, expected)

            if abs(score - expected) > 0.5:
                issues.append(f"{svc}: got {score}, expected {expected}")

        check("4. Compliance scores", 3, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("4. Compliance scores", 3, False, f"exception: {e}")


# ── Check 5: Hardcoded Secrets table exists ───────────────────────────────────
def check_5_secrets_table() -> None:
    """Table 'Hardcoded Secrets' exists with expected fields."""
    try:
        table_id = find_baserow_table("Hardcoded Secrets")
        if not table_id:
            check("5. Hardcoded Secrets table", 1, False, "table not found")
            return
        fields = baserow_get(f"database/fields/table/{table_id}/")
        field_names = {f["name"] for f in fields}
        expected_fields = {
            "Finding ID", "Service", "File Path", "Line Number",
            "Pattern Name", "Severity", "Detected At",
        }
        missing = expected_fields - field_names
        check("5. Hardcoded Secrets table", 1, not missing,
              f"missing: {missing}" if missing else "all fields present")
    except Exception as e:
        check("5. Hardcoded Secrets table", 1, False, f"exception: {e}")


# ── Check 6: Secrets rows have correct Severity mapping ──────────────────────
def check_6_secrets_severity() -> None:
    """Severity=Critical for AWSKey/APIToken, High otherwise."""
    try:
        table_id = find_baserow_table("Hardcoded Secrets")
        if not table_id:
            check("6. Secrets severity mapping", 2, False, "table not found")
            return
        fields = baserow_get(f"database/fields/table/{table_id}/")
        fmap = {f["name"]: f["id"] for f in fields}
        data = baserow_get(f"database/rows/table/{table_id}/?size=200")
        rows = data.get("results", [])
        if not rows:
            check("6. Secrets severity mapping", 2, False, "no rows in Hardcoded Secrets")
            return

        issues = []
        for row in rows:
            pattern_raw = row.get(f"field_{fmap.get('Pattern Name', 0)}", "")
            severity_raw = row.get(f"field_{fmap.get('Severity', 0)}", "")
            pattern = pattern_raw.get("value", "") if isinstance(pattern_raw, dict) else str(pattern_raw)
            severity = severity_raw.get("value", "") if isinstance(severity_raw, dict) else str(severity_raw)
            finding_id = row.get(f"field_{fmap.get('Finding ID', 0)}", "?")

            if pattern in ("AWSKey", "APIToken"):
                if severity != "Critical":
                    issues.append(f"{finding_id}: {pattern} should be Critical, got {severity}")
            else:
                if severity != "High":
                    issues.append(f"{finding_id}: {pattern} should be High, got {severity}")

        check("6. Secrets severity mapping", 2, not issues,
              f"{len(rows)} rows, all correct" if not issues else "; ".join(issues[:3]))
    except Exception as e:
        check("6. Secrets severity mapping", 2, False, f"exception: {e}")


# ── Check 7: Compliance Ranking view exists ───────────────────────────────────
def check_7_compliance_ranking_view() -> None:
    """Grid view 'Compliance Ranking' exists on Dockerfile Audit, sorted by Compliance Score asc."""
    try:
        table_id = find_baserow_table("Dockerfile Audit")
        if not table_id:
            check("7. Compliance Ranking view", 1, False, "table not found")
            return
        views = baserow_get(f"database/views/table/{table_id}/")
        found = None
        for v in views:
            if v.get("name") == "Compliance Ranking":
                found = v
                break
        if not found:
            check("7. Compliance Ranking view", 1, False, "view not found")
            return
        is_grid = found.get("type") == "grid"
        # Check sortings
        sortings = baserow_get(f"database/views/{found['id']}/sortings/")
        has_sort = False
        fields = baserow_get(f"database/fields/table/{table_id}/")
        score_field_id = None
        for f in fields:
            if f["name"] == "Compliance Score":
                score_field_id = f["id"]
                break
        if isinstance(sortings, list):
            for s in sortings:
                if s.get("field") == score_field_id and s.get("order") == "ASC":
                    has_sort = True
        check("7. Compliance Ranking view", 1, is_grid and has_sort,
              f"grid={is_grid}, sorted_asc={has_sort}")
    except Exception as e:
        check("7. Compliance Ranking view", 1, False, f"exception: {e}")


# ── Check 8: By Pattern Kanban view exists ────────────────────────────────────
def check_8_by_pattern_view() -> None:
    """Kanban view 'By Pattern' exists on Hardcoded Secrets, stacked by Pattern Name."""
    try:
        table_id = find_baserow_table("Hardcoded Secrets")
        if not table_id:
            check("8. By Pattern Kanban view", 1, False, "table not found")
            return
        views = baserow_get(f"database/views/table/{table_id}/")
        found = None
        for v in views:
            if v.get("name") == "By Pattern":
                found = v
                break
        if not found:
            check("8. By Pattern Kanban view", 1, False, "view not found")
            return
        is_kanban = found.get("type") == "kanban"
        check("8. By Pattern Kanban view", 1, is_kanban,
              f"type={found.get('type')}")
    except Exception as e:
        check("8. By Pattern Kanban view", 1, False, f"exception: {e}")


# ── Check 9: Audit markdown file exists with correct format ──────────────────
def check_9_audit_md_exists() -> None:
    """devops-configs/docs/docker-audit-2026-04-12.md exists with 5 lines, correct header."""
    try:
        rc, stdout, stderr = docker_exec(
            CODE_SERVER_CONTAINER,
            "cat", "/home/coder/project/devops-configs/docs/docker-audit-2026-04-12.md",
        )
        if rc != 0:
            check("9. Audit markdown file", 2, False, "file not found")
            return
        lines = stdout.strip().split("\n")
        line_count_ok = len(lines) == 5
        header_ok = lines[0].strip() == "# Docker Audit \u2014 2026-04-12" if lines else False
        services_line_ok = False
        if len(lines) >= 2:
            services_line_ok = lines[1].strip().startswith("Services scanned:")
            # Check alphabetical services
            expected_svc = "blog-engine, devops-configs, tabler, todo-api"
            if expected_svc in lines[1]:
                services_line_ok = True

        check("9. Audit markdown file", 2,
              line_count_ok and header_ok and services_line_ok,
              f"lines={len(lines)}, header={'ok' if header_ok else 'wrong'}, services={'ok' if services_line_ok else 'wrong'}")
    except Exception as e:
        check("9. Audit markdown file", 2, False, f"exception: {e}")


# ── Check 10: Audit markdown values are internally consistent ────────────────
def check_10_audit_md_values() -> None:
    """Lines 3-5 contain avg_score, failing_count, secret_count that are plausible numbers."""
    try:
        rc, stdout, _ = docker_exec(
            CODE_SERVER_CONTAINER,
            "cat", "/home/coder/project/devops-configs/docs/docker-audit-2026-04-12.md",
        )
        if rc != 0:
            check("10. Audit md values", 2, False, "file not found")
            return
        lines = stdout.strip().split("\n")
        if len(lines) < 5:
            check("10. Audit md values", 2, False, f"only {len(lines)} lines")
            return

        # Line 3: "Avg compliance score: <float>"
        avg_match = re.search(r"Avg compliance score:\s*([\d.]+)", lines[2])
        # Line 4: "Services below 75: <int>"
        fail_match = re.search(r"Services below 75:\s*(\d+)", lines[3])
        # Line 5: "Hardcoded secrets found: <int> (Critical: <int>; High: <int>)"
        secret_match = re.search(
            r"Hardcoded secrets found:\s*(\d+)\s*\(Critical:\s*(\d+);\s*High:\s*(\d+)\)",
            lines[4],
        )

        avg_ok = avg_match is not None
        fail_ok = fail_match is not None
        secret_ok = secret_match is not None

        # Cross-validate: secret_count should equal critical + high
        if secret_match:
            total_s = int(secret_match.group(1))
            crit = int(secret_match.group(2))
            high = int(secret_match.group(3))
            if total_s != crit + high:
                secret_ok = False

        check("10. Audit md values", 2, avg_ok and fail_ok and secret_ok,
              f"avg={'ok' if avg_ok else 'missing'}, fail={'ok' if fail_ok else 'missing'}, secrets={'ok' if secret_ok else 'missing/inconsistent'}")
    except Exception as e:
        check("10. Audit md values", 2, False, f"exception: {e}")


# ── Check 11: Git commit in devops-configs repo ──────────────────────────────
def check_11_git_commit() -> None:
    """Commit with message 'audit: docker compliance 2026-04-12' exists in devops-configs."""
    try:
        rc, stdout, stderr = docker_exec(
            CODE_SERVER_CONTAINER,
            "git", "-C", "/home/coder/project/devops-configs",
            "log", "--oneline", "--all", "-50",
        )
        if rc != 0:
            check("11. Git commit", 2, False, f"git log failed: {stderr.strip()}")
            return
        target_msg = "audit: docker compliance 2026-04-12"
        found = target_msg in stdout
        check("11. Git commit", 2, found,
              "commit found" if found else f"not found in last 50 commits")
    except Exception as e:
        check("11. Git commit", 2, False, f"exception: {e}")


# ── Check 12: OpenProject 'Immediate' priority exists ────────────────────────
def check_12_op_immediate_priority() -> None:
    """'Immediate' priority enumeration exists in OpenProject."""
    try:
        result = op_db_query("SELECT name FROM enumerations WHERE type='IssuePriority' AND name='Immediate'")
        found = "Immediate" in result
        check("12. OP Immediate priority", 1, found,
              "found" if found else "not found in enumerations")
    except Exception as e:
        check("12. OP Immediate priority", 1, False, f"exception: {e}")


# ── Check 13: Bug work packages for Hardcoded Secrets rows ───────────────────
def check_13_op_bug_wps() -> None:
    """One Bug WP per Hardcoded Secrets row with correct subject pattern."""
    try:
        # Find project id
        proj_id = op_db_query("SELECT id FROM projects WHERE identifier='devops-automation'")
        if not proj_id:
            check("13. OP Bug work packages", 2, False, "project not found")
            return
        # Find Bug type id
        bug_type_id = op_db_query("SELECT id FROM types WHERE name='Bug'")
        if not bug_type_id:
            check("13. OP Bug work packages", 2, False, "Bug type not found")
            return
        # Query work packages
        result = op_db_query(
            f"SELECT subject FROM work_packages WHERE project_id={proj_id} AND type_id={bug_type_id} AND subject LIKE 'SECRET LEAK%'"
        )
        subjects = [s for s in result.split("\n") if s.strip()]
        pattern = re.compile(r"SECRET LEAK \[.+\]: .+ at .+:\d+")
        valid = [s for s in subjects if pattern.match(s)]
        has_any = len(valid) > 0
        check("13. OP Bug work packages", 2, has_any,
              f"found {len(valid)} Bug WPs with correct subject pattern")
    except Exception as e:
        check("13. OP Bug work packages", 2, False, f"exception: {e}")


# ── Check 14: Bug WP assignee and priority mapping ───────────────────────────
def check_14_op_bug_assignee_priority() -> None:
    """Bug WPs assigned to sandra.love with correct priority mapping (Immediate or High)."""
    try:
        proj_id = op_db_query("SELECT id FROM projects WHERE identifier='devops-automation'")
        bug_type_id = op_db_query("SELECT id FROM types WHERE name='Bug'")
        if not proj_id or not bug_type_id:
            check("14. Bug WP assignee/priority", 2, False, "project or Bug type not found")
            return

        result = op_db_query(
            f"SELECT wp.subject, u.login, e.name AS priority "
            f"FROM work_packages wp "
            f"LEFT JOIN users u ON wp.assigned_to_id = u.id "
            f"LEFT JOIN enumerations e ON wp.priority_id = e.id "
            f"WHERE wp.project_id={proj_id} AND wp.type_id={bug_type_id} "
            f"AND wp.subject LIKE 'SECRET LEAK%'"
        )
        rows = [r for r in result.split("\n") if r.strip()]
        if not rows:
            check("14. Bug WP assignee/priority", 2, False, "no SECRET LEAK WPs found")
            return

        issues = []
        for row in rows:
            parts = row.split("|")
            if len(parts) < 3:
                continue
            subject = parts[0].strip()
            login = parts[1].strip()
            priority = parts[2].strip()
            # Check assignee matches sandra.love (login may vary)
            if "sandra" not in login.lower() and "love" not in login.lower():
                issues.append(f"{subject[:40]}: assignee={login}")
            if priority not in ("Immediate", "High"):
                issues.append(f"{subject[:40]}: priority={priority}")

        check("14. Bug WP assignee/priority", 2, not issues,
              f"{len(rows)} WPs checked, all OK" if not issues else "; ".join(issues[:3]))
    except Exception as e:
        check("14. Bug WP assignee/priority", 2, False, f"exception: {e}")


# ── Check 15: Task WPs for low-score services ────────────────────────────────
def check_15_op_task_wps() -> None:
    """Task WPs exist for Dockerfile Audit rows with Compliance Score < 75."""
    try:
        proj_id = op_db_query("SELECT id FROM projects WHERE identifier='devops-automation'")
        task_type_id = op_db_query("SELECT id FROM types WHERE name='Task'")
        if not proj_id or not task_type_id:
            check("15. OP Task WPs (harden)", 2, False, "project or Task type not found")
            return

        result = op_db_query(
            f"SELECT wp.subject, u.login, e.name AS priority "
            f"FROM work_packages wp "
            f"LEFT JOIN users u ON wp.assigned_to_id = u.id "
            f"LEFT JOIN enumerations e ON wp.priority_id = e.id "
            f"WHERE wp.project_id={proj_id} AND wp.type_id={task_type_id} "
            f"AND wp.subject LIKE 'Harden Dockerfile:%'"
        )
        rows = [r for r in result.split("\n") if r.strip()]

        pattern = re.compile(r"Harden Dockerfile: .+ \(score \d+\)")
        issues = []
        valid_count = 0
        for row in rows:
            parts = row.split("|")
            if len(parts) < 3:
                continue
            subject = parts[0].strip()
            login = parts[1].strip()
            priority = parts[2].strip()
            if not pattern.match(subject):
                continue
            valid_count += 1
            if "richard" not in login.lower() and "rethman" not in login.lower():
                issues.append(f"{subject}: assignee={login}")
            if priority != "High":
                issues.append(f"{subject}: priority={priority}")

        has_any = valid_count > 0
        check("15. OP Task WPs (harden)", 2, has_any and not issues,
              f"{valid_count} Task WPs" + (f", issues: {'; '.join(issues[:3])}" if issues else ", all OK"))
    except Exception as e:
        check("15. OP Task WPs (harden)", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_db_exists()
    check_2_dockerfile_audit_table()
    check_3_audit_rows()
    check_4_compliance_scores()
    check_5_secrets_table()
    check_6_secrets_severity()
    check_7_compliance_ranking_view()
    check_8_by_pattern_view()
    check_9_audit_md_exists()
    check_10_audit_md_values()
    check_11_git_commit()
    check_12_op_immediate_priority()
    check_13_op_bug_wps()
    check_14_op_bug_assignee_priority()
    check_15_op_task_wps()

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
