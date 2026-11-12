# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_026/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-026-I1: Q4-2024 Engineering Investment Portfolio Review

Checks: 12 weighted checks across openproject, baserow, code-server.
Strategy: OpenProject embedded Postgres, Baserow REST API, code-server docker exec.

Required env vars:
  SERVER_HOSTNAME, OPENPROJECT_PORT, OPENPROJECT_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  CODE_SERVER_PORT, CODE_SERVER_CONTAINER
"""

import os
import sys
import subprocess
import json
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OP_PORT = os.environ.get("OPENPROJECT_PORT")
OP_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

BR_PORT = os.environ.get("BASEROW_PORT")
BR_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BR_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")

CS_PORT = os.environ.get("CODE_SERVER_PORT")
CS_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")

_required = {
    "OPENPROJECT_PORT": OP_PORT, "OPENPROJECT_CONTAINER": OP_CONTAINER,
    "BASEROW_PORT": BR_PORT, "BASEROW_CONTAINER": BR_CONTAINER,
    "BASEROW_DB_CONTAINER": BR_DB_CONTAINER, "CODE_SERVER_CONTAINER": CS_CONTAINER,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    print(f"FATAL: {', '.join(_missing)} not set", file=sys.stderr)
    sys.exit(1)

# ── Classification constants ──────────────────────────────────────────────────
SECURITY_KW = ['security', 'auth', 'vulnerability', 'sso', 'saml', 'encrypt', 'secure']
RELIABILITY_KW = ['reliability', 'sla', 'alert', 'monitor', 'latency', 'timeout',
                  '502', 'error', 'uptime', 'availability']
TECHDEBT_KW = ['refactor', 'cleanup', 'migrate', 'upgrade', 'debt', 'legacy', 'tuning']

ASSIGNEE_TEAM = {
    'David Kim': 'Platform', 'Frank Nguyen': 'Product', 'Grace Patel': 'Product',
    'Henry Johnson': 'Data', 'James Lee': 'Platform', 'Liam Robinson': 'Product',
    'Mia Anderson': 'Platform', 'Paul Harris': 'Platform', 'Samuel Clark': 'Security',
    'OpenProject Admin': 'Platform',
}
TARGET_PCT = {'NewFeature': 50.0, 'TechDebt': 25.0, 'Reliability': 15.0, 'Security': 10.0}
BUCKET_ORDER = ['NewFeature', 'TechDebt', 'Reliability', 'Security']

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


def op_sql(sql: str) -> str:
    """Query OpenProject embedded Postgres."""
    rc, out, err = docker_exec(
        OP_CONTAINER, "psql", "-U", "openproject", "-d", "openproject",
        "-t", "-A", "-c", sql, timeout=15,
    )
    return out.strip()


def classify_bucket(subject: str, wp_type: str) -> str:
    subj_lower = subject.lower()
    if any(kw in subj_lower for kw in SECURITY_KW):
        return 'Security'
    if any(kw in subj_lower for kw in RELIABILITY_KW):
        return 'Reliability'
    if wp_type == 'Bug' or any(kw in subj_lower for kw in TECHDEBT_KW):
        return 'TechDebt'
    return 'NewFeature'


def get_assignee_team(assignee: str) -> str:
    return ASSIGNEE_TEAM.get(assignee, 'Platform')


# ── Ground truth from OpenProject ─────────────────────────────────────────────
_gt = None
_gt_err = None


def compute_ground_truth():
    """Query OpenProject for closed WPs and compute expected Baserow data."""
    global _gt, _gt_err
    if _gt is not None or _gt_err is not None:
        return

    project_id = op_sql("SELECT id FROM projects WHERE name = 'API Gateway' LIMIT 1")
    if not project_id:
        _gt_err = "Project 'API Gateway' not found"
        return

    closed_id = op_sql("SELECT id FROM statuses WHERE name = 'Closed' LIMIT 1")
    if not closed_id:
        _gt_err = "Status 'Closed' not found"
        return

    rows = op_sql(f"""
        SELECT wp.id, wp.subject, t.name,
               COALESCE(u.firstname || ' ' || u.lastname, 'Unassigned'),
               wp.updated_at::date
        FROM work_packages wp
        JOIN types t ON t.id = wp.type_id
        LEFT JOIN users u ON u.id = wp.assigned_to_id
        WHERE wp.project_id = {project_id}
          AND wp.status_id = {closed_id}
          AND wp.updated_at >= '2024-10-01'
          AND wp.updated_at < '2025-01-01'
        ORDER BY wp.id ASC
        LIMIT 30
    """)

    if not rows:
        _gt_err = "No closed WPs found in date range"
        return

    wps = []
    for line in rows.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split('|')
        if len(parts) < 5:
            continue
        wp_id, subject, wp_type, assignee, closed_date = parts[0], parts[1], parts[2], parts[3], parts[4]
        bucket = classify_bucket(subject, wp_type)
        team = get_assignee_team(assignee)
        wps.append({
            'wp_id': int(wp_id), 'subject': subject, 'type': wp_type,
            'assignee': assignee, 'bucket': bucket, 'team': team,
            'closed_date': closed_date,
        })

    total = len(wps)
    bucket_counts = {b: 0 for b in BUCKET_ORDER}
    for wp in wps:
        bucket_counts[wp['bucket']] += 1

    bucket_totals = []
    for b in BUCKET_ORDER:
        count = bucket_counts[b]
        share = round(count / total * 100, 1) if total else 0.0
        target = TARGET_PCT[b]
        gap = round(target - share, 1)
        bucket_totals.append({
            'bucket': b, 'count': count,
            'share_pct': share, 'target_pct': target, 'gap_pct': gap,
        })

    _gt = {'wps': wps, 'total': total, 'bucket_totals': bucket_totals}


# ── Baserow API helpers ───────────────────────────────────────────────────────
_br_token = None


def baserow_token() -> str:
    global _br_token
    if _br_token:
        return _br_token
    r = requests.post(
        f"http://{HOST}:{BR_PORT}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"}, timeout=10,
    )
    r.raise_for_status()
    _br_token = r.json()["access_token"]
    return _br_token


def br_get(path: str) -> dict | list:
    r = requests.get(
        f"http://{HOST}:{BR_PORT}/api/{path}",
        headers={"Authorization": f"JWT {baserow_token()}"}, timeout=15,
    )
    r.raise_for_status()
    return r.json()


def find_baserow_db(name: str) -> int | None:
    """Find a Baserow database application by name, return its ID."""
    apps = br_get("applications/")
    for a in apps:
        if a.get("name") == name and a.get("type") == "database":
            return a["id"]
    return None


def find_baserow_table(db_id: int, table_name: str) -> int | None:
    """Find a table by name in a Baserow database."""
    tables = br_get(f"database/tables/database/{db_id}/")
    for t in tables:
        if t.get("name") == table_name:
            return t["id"]
    return None


def get_baserow_rows(table_id: int) -> list[dict]:
    """Get all rows from a Baserow table using human-readable field names."""
    rows = []
    url = f"database/rows/table/{table_id}/?user_field_names=true&size=200"
    data = br_get(url)
    rows.extend(data.get("results", []))
    while data.get("next"):
        # Parse next URL path
        next_url = data["next"]
        # Extract path after /api/
        path = next_url.split("/api/", 1)[-1]
        data = br_get(path)
        rows.extend(data.get("results", []))
    return rows


def select_value(val) -> str:
    """Extract value from a Baserow single-select field."""
    if isinstance(val, dict):
        return val.get("value", "")
    return str(val or "")


# ── Cached Baserow state ─────────────────────────────────────────────────────
_br_db_id = None
_br_cwp_rows = None  # Closed Work Packages rows
_br_bt_rows = None   # Bucket Totals rows


def load_baserow_state():
    """Load Baserow database, tables and rows into cache."""
    global _br_db_id, _br_cwp_rows, _br_bt_rows
    if _br_db_id is not None:
        return

    db_id = find_baserow_db("Portfolio Review Q4-2024")
    if db_id is None:
        _br_db_id = -1
        return
    _br_db_id = db_id

    cwp_tid = find_baserow_table(db_id, "Closed Work Packages")
    if cwp_tid:
        _br_cwp_rows = get_baserow_rows(cwp_tid)

    bt_tid = find_baserow_table(db_id, "Bucket Totals")
    if bt_tid:
        _br_bt_rows = get_baserow_rows(bt_tid)


# ── Individual checks ─────────────────────────────────────────────────────────
def check_1_baserow_db_exists() -> None:
    """Baserow database 'Portfolio Review Q4-2024' exists."""
    try:
        load_baserow_state()
        found = _br_db_id is not None and _br_db_id > 0
        check("1. Baserow DB exists", 1, found,
              f"db_id={_br_db_id}" if found else "not found")
    except Exception as e:
        check("1. Baserow DB exists", 1, False, f"exception: {e}")


def check_2_cwp_table_row_count() -> None:
    """'Closed Work Packages' table exists with correct row count."""
    try:
        compute_ground_truth()
        load_baserow_state()
        if _gt_err:
            check("2. CWP table row count", 2, False, f"ground truth error: {_gt_err}")
            return
        if _br_cwp_rows is None:
            check("2. CWP table row count", 2, False, "table not found")
            return
        expected = len(_gt['wps'])
        actual = len(_br_cwp_rows)
        ok = actual == expected
        check("2. CWP table row count", 2, ok,
              f"expected={expected}, actual={actual}")
    except Exception as e:
        check("2. CWP table row count", 2, False, f"exception: {e}")


def check_3_cwp_bucket_classification() -> None:
    """Closed WPs have correct Investment Bucket classification."""
    try:
        compute_ground_truth()
        load_baserow_state()
        if _gt_err or _br_cwp_rows is None:
            check("3. CWP bucket classification", 2, False,
                  _gt_err or "table not found")
            return

        # Build expected bucket by WP ID
        expected_buckets = {wp['wp_id']: wp['bucket'] for wp in _gt['wps']}

        wrong = []
        for row in _br_cwp_rows:
            wp_id_val = row.get("WP ID")
            if wp_id_val is None:
                continue
            wp_id = int(wp_id_val) if not isinstance(wp_id_val, int) else wp_id_val
            actual_bucket = select_value(row.get("Investment Bucket", ""))
            expected_bucket = expected_buckets.get(wp_id)
            if expected_bucket and actual_bucket != expected_bucket:
                wrong.append(f"WP#{wp_id}: expected={expected_bucket}, got={actual_bucket}")

        ok = len(wrong) == 0 and len(_br_cwp_rows) > 0
        check("3. CWP bucket classification", 2, ok,
              f"{len(wrong)} wrong" if wrong else f"all {len(_br_cwp_rows)} correct")
        if wrong:
            for w in wrong[:5]:
                print(f"  detail: {w}", file=sys.stderr)
    except Exception as e:
        check("3. CWP bucket classification", 2, False, f"exception: {e}")


def check_4_cwp_team_assignment() -> None:
    """Closed WPs have correct Team assignments."""
    try:
        compute_ground_truth()
        load_baserow_state()
        if _gt_err or _br_cwp_rows is None:
            check("4. CWP team assignment", 2, False,
                  _gt_err or "table not found")
            return

        expected_teams = {wp['wp_id']: wp['team'] for wp in _gt['wps']}

        wrong = []
        for row in _br_cwp_rows:
            wp_id_val = row.get("WP ID")
            if wp_id_val is None:
                continue
            wp_id = int(wp_id_val) if not isinstance(wp_id_val, int) else wp_id_val
            actual_team = select_value(row.get("Team", ""))
            expected_team = expected_teams.get(wp_id)
            if expected_team and actual_team != expected_team:
                wrong.append(f"WP#{wp_id}: expected={expected_team}, got={actual_team}")

        ok = len(wrong) == 0 and len(_br_cwp_rows) > 0
        check("4. CWP team assignment", 2, ok,
              f"{len(wrong)} wrong" if wrong else f"all {len(_br_cwp_rows)} correct")
    except Exception as e:
        check("4. CWP team assignment", 2, False, f"exception: {e}")


def check_5_bucket_totals_exists() -> None:
    """'Bucket Totals' table exists with 4 rows in correct order."""
    try:
        load_baserow_state()
        if _br_bt_rows is None:
            check("5. Bucket Totals table", 1, False, "table not found")
            return
        count = len(_br_bt_rows)
        if count != 4:
            check("5. Bucket Totals table", 1, False, f"expected 4 rows, got {count}")
            return
        actual_order = [select_value(r.get("Bucket", "")) for r in _br_bt_rows]
        ok = actual_order == BUCKET_ORDER
        check("5. Bucket Totals table", 1, ok,
              f"order={actual_order}")
    except Exception as e:
        check("5. Bucket Totals table", 1, False, f"exception: {e}")


def check_6_bucket_totals_count_share() -> None:
    """Bucket Totals Count and Share Pct match ground truth."""
    try:
        compute_ground_truth()
        load_baserow_state()
        if _gt_err or _br_bt_rows is None:
            check("6. Bucket Count & Share Pct", 3, False,
                  _gt_err or "table not found")
            return

        expected_by_bucket = {bt['bucket']: bt for bt in _gt['bucket_totals']}
        wrong = []
        for row in _br_bt_rows:
            bucket = select_value(row.get("Bucket", ""))
            exp = expected_by_bucket.get(bucket)
            if not exp:
                wrong.append(f"{bucket}: unexpected bucket")
                continue

            actual_count = row.get("Count")
            if isinstance(actual_count, str):
                actual_count = float(actual_count)
            if actual_count is not None:
                actual_count = int(actual_count)

            actual_share = row.get("Share Pct")
            if isinstance(actual_share, str):
                actual_share = float(actual_share)

            if actual_count != exp['count']:
                wrong.append(f"{bucket} Count: expected={exp['count']}, got={actual_count}")
            if actual_share is not None and abs(float(actual_share) - exp['share_pct']) > 0.15:
                wrong.append(f"{bucket} Share: expected={exp['share_pct']}, got={actual_share}")

        ok = len(wrong) == 0 and len(_br_bt_rows) > 0
        check("6. Bucket Count & Share Pct", 3, ok,
              "all correct" if ok else "; ".join(wrong[:5]))
    except Exception as e:
        check("6. Bucket Count & Share Pct", 3, False, f"exception: {e}")


def check_7_bucket_totals_target_gap() -> None:
    """Bucket Totals Target Pct and Gap Pct match expected values."""
    try:
        compute_ground_truth()
        load_baserow_state()
        if _gt_err or _br_bt_rows is None:
            check("7. Bucket Target & Gap Pct", 2, False,
                  _gt_err or "table not found")
            return

        expected_by_bucket = {bt['bucket']: bt for bt in _gt['bucket_totals']}
        wrong = []
        for row in _br_bt_rows:
            bucket = select_value(row.get("Bucket", ""))
            exp = expected_by_bucket.get(bucket)
            if not exp:
                continue

            actual_target = row.get("Target Pct")
            if isinstance(actual_target, str):
                actual_target = float(actual_target)
            actual_gap = row.get("Gap Pct")
            if isinstance(actual_gap, str):
                actual_gap = float(actual_gap)

            if actual_target is not None and abs(float(actual_target) - exp['target_pct']) > 0.15:
                wrong.append(f"{bucket} Target: expected={exp['target_pct']}, got={actual_target}")
            if actual_gap is not None and abs(float(actual_gap) - exp['gap_pct']) > 0.15:
                wrong.append(f"{bucket} Gap: expected={exp['gap_pct']}, got={actual_gap}")

        ok = len(wrong) == 0 and len(_br_bt_rows) > 0
        check("7. Bucket Target & Gap Pct", 2, ok,
              "all correct" if ok else "; ".join(wrong[:5]))
    except Exception as e:
        check("7. Bucket Target & Gap Pct", 2, False, f"exception: {e}")


def check_8_codeserver_file_exists() -> None:
    """Markdown file exists in code-server at the expected path."""
    try:
        rc, out, err = docker_exec(
            CS_CONTAINER, "test", "-f",
            "/home/coder/project/devops-configs/docs/portfolio-review-Q4-2024.md",
        )
        ok = rc == 0
        check("8. Code-server file exists", 1, ok,
              "found" if ok else "file not found")
    except Exception as e:
        check("8. Code-server file exists", 1, False, f"exception: {e}")


def check_9_codeserver_file_content() -> None:
    """Markdown file has the five expected lines with correct data."""
    try:
        compute_ground_truth()
        rc, out, err = docker_exec(
            CS_CONTAINER, "cat",
            "/home/coder/project/devops-configs/docs/portfolio-review-Q4-2024.md",
        )
        if rc != 0:
            check("9. Code-server file content", 2, False, "cannot read file")
            return

        lines = [l for l in out.strip().split('\n') if l.strip()]

        issues = []
        # Line 1: title
        if not lines or "Engineering Investment Portfolio" not in lines[0]:
            issues.append("line 1 missing/wrong title")
        # Line 2: window
        if len(lines) < 2 or "2024-10-01" not in lines[1] or "2024-12-31" not in lines[1]:
            issues.append("line 2 missing/wrong window dates")
        # Line 3: total closed
        if _gt and len(lines) >= 3:
            expected_total = str(_gt['total'])
            if expected_total not in lines[2]:
                issues.append(f"line 3 expected total={expected_total}, got: {lines[2]}")
        elif len(lines) < 3:
            issues.append("line 3 missing")
        # Line 4: actual shares
        if _gt and len(lines) >= 4:
            for bt in _gt['bucket_totals']:
                share_str = str(bt['share_pct'])
                if share_str not in lines[3]:
                    issues.append(f"line 4 missing {bt['bucket']} share {share_str}")
                    break
        elif len(lines) < 4:
            issues.append("line 4 missing")
        # Line 5: target shares
        if len(lines) >= 5:
            for b, t in TARGET_PCT.items():
                if str(t) not in lines[4]:
                    issues.append(f"line 5 missing {b} target {t}")
                    break
        elif len(lines) < 5:
            issues.append("line 5 missing")

        ok = len(issues) == 0
        check("9. Code-server file content", 2, ok,
              "all 5 lines correct" if ok else "; ".join(issues[:3]))
    except Exception as e:
        check("9. Code-server file content", 2, False, f"exception: {e}")


def check_10_rebalance_wp_count() -> None:
    """OpenProject has correct number of rebalance WPs for under-invested buckets."""
    try:
        compute_ground_truth()
        if _gt_err:
            check("10. Rebalance WP count", 2, False, _gt_err)
            return

        expected_count = sum(1 for bt in _gt['bucket_totals'] if bt['gap_pct'] > 0)

        project_id = op_sql("SELECT id FROM projects WHERE name = 'API Gateway' LIMIT 1")
        count = op_sql(f"""
            SELECT COUNT(*) FROM work_packages wp
            JOIN types t ON t.id = wp.type_id
            WHERE wp.project_id = {project_id}
              AND t.name = 'Task'
              AND wp.subject LIKE 'Rebalance next quarter:%'
        """)
        actual = int(count) if count else 0
        ok = actual == expected_count
        check("10. Rebalance WP count", 2, ok,
              f"expected={expected_count}, actual={actual}")
    except Exception as e:
        check("10. Rebalance WP count", 2, False, f"exception: {e}")


def check_11_rebalance_wp_subjects() -> None:
    """Rebalance WPs have correct subject format with bucket name and gap percentage."""
    try:
        compute_ground_truth()
        if _gt_err:
            check("11. Rebalance WP subjects", 2, False, _gt_err)
            return

        project_id = op_sql("SELECT id FROM projects WHERE name = 'API Gateway' LIMIT 1")
        rows = op_sql(f"""
            SELECT wp.subject FROM work_packages wp
            JOIN types t ON t.id = wp.type_id
            WHERE wp.project_id = {project_id}
              AND t.name = 'Task'
              AND wp.subject LIKE 'Rebalance next quarter:%'
        """)

        actual_subjects = set()
        if rows:
            for line in rows.split('\n'):
                line = line.strip()
                if line:
                    actual_subjects.add(line)

        expected_subjects = set()
        for bt in _gt['bucket_totals']:
            if bt['gap_pct'] > 0:
                expected_subjects.add(
                    f"Rebalance next quarter: {bt['bucket']} (+{bt['gap_pct']}%)"
                )

        missing = expected_subjects - actual_subjects
        extra = actual_subjects - expected_subjects
        ok = not missing and not extra
        detail = "all correct" if ok else ""
        if missing:
            detail += f"missing: {missing}"
        if extra:
            detail += f" extra: {extra}"
        check("11. Rebalance WP subjects", 2, ok, detail.strip())
    except Exception as e:
        check("11. Rebalance WP subjects", 2, False, f"exception: {e}")


def check_12_rebalance_wp_details() -> None:
    """Rebalance WPs have correct assignee, priority, and description."""
    try:
        compute_ground_truth()
        if _gt_err:
            check("12. Rebalance WP details", 2, False, _gt_err)
            return

        project_id = op_sql("SELECT id FROM projects WHERE name = 'API Gateway' LIMIT 1")
        # Get admin user ID
        admin_id = op_sql(
            "SELECT id FROM users WHERE login = 'admin' LIMIT 1"
        )

        rows = op_sql(f"""
            SELECT wp.subject, wp.assigned_to_id, wp.description,
                   p.name AS priority_name
            FROM work_packages wp
            JOIN types t ON t.id = wp.type_id
            LEFT JOIN enumerations p ON p.id = wp.priority_id
            WHERE wp.project_id = {project_id}
              AND t.name = 'Task'
              AND wp.subject LIKE 'Rebalance next quarter:%'
        """)

        if not rows:
            check("12. Rebalance WP details", 2, False, "no rebalance WPs found")
            return

        expected_by_bucket = {bt['bucket']: bt for bt in _gt['bucket_totals']}
        issues = []
        for line in rows.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 4:
                continue
            subject, assigned_id, description, priority = [p.strip() for p in parts]

            # Check assignee is admin
            if admin_id and assigned_id != admin_id:
                issues.append(f"{subject}: wrong assignee (id={assigned_id}, expected={admin_id})")

            # Check priority is Normal
            if priority.lower() != 'normal':
                issues.append(f"{subject}: priority={priority}, expected=Normal")

            # Check description format
            # Extract bucket from subject: "Rebalance next quarter: <Bucket> (+<Gap>%)"
            import re
            m = re.search(r'Rebalance next quarter: (\w+)', subject)
            if m:
                bucket = m.group(1)
                bt = expected_by_bucket.get(bucket)
                if bt:
                    expected_desc = (
                        f"Current: {bt['share_pct']}%; "
                        f"Target: {bt['target_pct']}%; "
                        f"Quarter under review: Q4-2024"
                    )
                    # Description may contain HTML or markdown formatting
                    desc_clean = description.replace('\n', ' ').strip()
                    if expected_desc not in desc_clean and \
                       f"Current: {bt['share_pct']}%" not in desc_clean:
                        issues.append(
                            f"{bucket}: desc mismatch, expected contains '{expected_desc}'"
                        )

        ok = len(issues) == 0 and rows.strip() != ""
        check("12. Rebalance WP details", 2, ok,
              "all correct" if ok else "; ".join(issues[:3]))
    except Exception as e:
        check("12. Rebalance WP details", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_baserow_db_exists()
    check_2_cwp_table_row_count()
    check_3_cwp_bucket_classification()
    check_4_cwp_team_assignment()
    check_5_bucket_totals_exists()
    check_6_bucket_totals_count_share()
    check_7_bucket_totals_target_gap()
    check_8_codeserver_file_exists()
    check_9_codeserver_file_content()
    check_10_rebalance_wp_count()
    check_11_rebalance_wp_subjects()
    check_12_rebalance_wp_details()

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
