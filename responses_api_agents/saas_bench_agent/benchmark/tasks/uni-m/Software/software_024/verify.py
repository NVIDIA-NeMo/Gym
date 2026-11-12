# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_024/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-024-I5: Data Platform RFC workflow — files, Baserow registry, OpenProject Epics

Checks: 13 weighted checks across code-server, baserow, openproject.
Strategy: docker exec filesystem (code-server), REST API (baserow), docker exec psql (openproject).

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
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

# ── Task data ─────────────────────────────────────────────────────────────────
RFC_FILENAMES = [
    "rfc-001-kafka-schema-registry.md",
    "rfc-002-dbt-incremental-models.md",
    "rfc-003-airflow-on-kubernetes.md",
    "rfc-004-data-lake-iceberg.md",
    "rfc-005-feature-store-rollout.md",
]
RFC_TITLES = [
    "Kafka Schema Registry Adoption",
    "dbt Incremental Models Migration",
    "Airflow Deployment on Kubernetes",
    "Data Lake Migration to Apache Iceberg",
    "Feature Store Rollout for ML Pipelines",
]
RFC_AUTHOR = "Leila Farahani"
RFC_CREATED = "2025-10-06"
RFC_REVIEWERS = {
    "001": "Takeshi Morimoto",
    "002": "Beatriz Cardoso",
    "003": "Henrik Johansson",
    "004": "Chidera Obi",
    "005": "Valentina Russo",
}
RFC_DECISIONS = [
    "Approved: deploy Confluent Schema Registry with Avro contracts enforced at producer level for all Kafka topics.",
    "Under review: benchmark incremental materialization strategies on representative fact tables before commitment.",
    "Approved: migrate Airflow to KubernetesExecutor on dedicated EKS node group with per-task pod isolation.",
    "Approved: adopt Apache Iceberg on S3 with AWS Glue catalog for all analytics tables within Q4.",
    "Deferred: revisit after feast vs. tecton vendor evaluation and cost analysis is complete.",
]
# 1-indexed: RFC-001, RFC-003, RFC-004 are approved
APPROVED_INDICES = {1, 3, 4}
DECISION_DATE = "2025-10-30"
# Days between 2025-10-06 and 2025-10-30 = 24
REVIEW_DURATION_DAYS = 24

RFC_DIR = "devops-configs/docs/rfc"

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


def read_file_in_container(container: str, path: str) -> str | None:
    """Read a file from a docker container, return contents or None."""
    rc, out, _ = docker_exec(container, "cat", path)
    return out if rc == 0 else None


def baserow_auth() -> str:
    """Authenticate to Baserow and return JWT token."""
    r = requests.post(
        f"{BASEROW_URL}/api/user/token-auth/",
        json={"email": "admin@example.com", "password": "Admin1234"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def baserow_get(path: str, token: str) -> dict:
    r = requests.get(
        f"{BASEROW_URL}/api/{path}",
        headers={"Authorization": f"JWT {token}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# ── Check 1: All 5 RFC files exist ───────────────────────────────────────────
def check_1_files_exist() -> None:
    """Verify all 5 RFC markdown files exist in code-server."""
    try:
        rc, out, _ = docker_exec(
            CODE_SERVER_CONTAINER, "ls", f"/home/coder/project/{RFC_DIR}/"
        )
        if rc != 0:
            check("1. RFC files exist", 1, False, f"directory not found: {RFC_DIR}")
            return
        files = out.strip().split("\n") if out.strip() else []
        missing = [f for f in RFC_FILENAMES if f not in files]
        check("1. RFC files exist", 1, not missing,
              f"all 5 present" if not missing else f"missing: {missing}")
    except Exception as e:
        check("1. RFC files exist", 1, False, f"exception: {e}")


# ── Check 2: RFC file headings ───────────────────────────────────────────────
def check_2_headings() -> None:
    """Verify line 1 of each RFC file matches '# RFC-NNN: Title'."""
    try:
        issues = []
        for i, (fname, title) in enumerate(zip(RFC_FILENAMES, RFC_TITLES)):
            num = f"{i+1:03d}"
            expected_heading = f"# RFC-{num}: {title}"
            content = read_file_in_container(
                CODE_SERVER_CONTAINER, f"/home/coder/project/{RFC_DIR}/{fname}"
            )
            if content is None:
                issues.append(f"{fname}: file not found")
                continue
            lines = content.split("\n")
            if not lines or lines[0].strip() != expected_heading:
                issues.append(f"{fname}: got {lines[0].strip()!r}")
        check("2. RFC file headings", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("2. RFC file headings", 2, False, f"exception: {e}")


# ── Check 3: Author, date, reviewer lines ────────────────────────────────────
def check_3_metadata_lines() -> None:
    """Verify lines 3 (Author), 4 (Created), 5 (Reviewer) in each RFC file."""
    try:
        issues = []
        for i, fname in enumerate(RFC_FILENAMES):
            num = f"{i+1:03d}"
            content = read_file_in_container(
                CODE_SERVER_CONTAINER, f"/home/coder/project/{RFC_DIR}/{fname}"
            )
            if content is None:
                issues.append(f"{fname}: file not found")
                continue
            lines = content.split("\n")
            if len(lines) < 5:
                issues.append(f"{fname}: too few lines ({len(lines)})")
                continue
            # Line 3 (index 2): Author
            if lines[2].strip() != f"Author: {RFC_AUTHOR}":
                issues.append(f"{fname}: author={lines[2].strip()!r}")
            # Line 4 (index 3): Created
            if lines[3].strip() != f"Created: {RFC_CREATED}":
                issues.append(f"{fname}: created={lines[3].strip()!r}")
            # Line 5 (index 4): Reviewer
            expected_reviewer = f"Reviewer: {RFC_REVIEWERS[num]}"
            if lines[4].strip() != expected_reviewer:
                issues.append(f"{fname}: reviewer={lines[4].strip()!r}")
        check("3. Author/date/reviewer lines", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("3. Author/date/reviewer lines", 2, False, f"exception: {e}")


# ── Check 4: Decision summary lines ──────────────────────────────────────────
def check_4_decisions() -> None:
    """Verify line 6 (decision) of each RFC file."""
    try:
        issues = []
        for i, (fname, decision) in enumerate(zip(RFC_FILENAMES, RFC_DECISIONS)):
            content = read_file_in_container(
                CODE_SERVER_CONTAINER, f"/home/coder/project/{RFC_DIR}/{fname}"
            )
            if content is None:
                issues.append(f"{fname}: file not found")
                continue
            lines = content.split("\n")
            if len(lines) < 6:
                issues.append(f"{fname}: too few lines ({len(lines)})")
                continue
            if lines[5].strip() != decision:
                issues.append(f"{fname}: decision mismatch")
        check("4. Decision summary lines", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("4. Decision summary lines", 2, False, f"exception: {e}")


# ── Check 5: Status lines ────────────────────────────────────────────────────
def check_5_status_lines() -> None:
    """Verify Status: Approved for indices 1,3,4; Status: Draft for 2,5."""
    try:
        issues = []
        for i, fname in enumerate(RFC_FILENAMES):
            idx = i + 1  # 1-indexed
            expected_status = "Status: Approved" if idx in APPROVED_INDICES else "Status: Draft"
            content = read_file_in_container(
                CODE_SERVER_CONTAINER, f"/home/coder/project/{RFC_DIR}/{fname}"
            )
            if content is None:
                issues.append(f"{fname}: file not found")
                continue
            lines = content.split("\n")
            if len(lines) < 2:
                issues.append(f"{fname}: too few lines")
                continue
            if lines[1].strip() != expected_status:
                issues.append(f"{fname}: expected {expected_status!r}, got {lines[1].strip()!r}")
        check("5. Status lines correct", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("5. Status lines correct", 2, False, f"exception: {e}")


# ── Check 6: Baserow database and table exist ────────────────────────────────
_baserow_token: str | None = None
_baserow_table_id: int | None = None


def check_6_baserow_db_table() -> None:
    """Verify Baserow has database 'Data Platform RFC Registry' with table 'RFC Registry'."""
    global _baserow_token, _baserow_table_id
    try:
        _baserow_token = baserow_auth()
        # List all applications (databases)
        apps = baserow_get("applications/", _baserow_token)
        db = None
        for app in apps:
            if app.get("name") == "Data Platform RFC Registry" and app.get("type") == "database":
                db = app
                break
        if db is None:
            check("6. Baserow DB and table exist", 1, False, "database not found")
            return
        # Find table within database
        tables = baserow_get(f"database/tables/database/{db['id']}/", _baserow_token)
        table = None
        for t in tables:
            if t.get("name") == "RFC Registry":
                table = t
                break
        if table is None:
            check("6. Baserow DB and table exist", 1, False, "table 'RFC Registry' not found")
            return
        _baserow_table_id = table["id"]
        check("6. Baserow DB and table exist", 1, True, f"db_id={db['id']}, table_id={table['id']}")
    except Exception as e:
        check("6. Baserow DB and table exist", 1, False, f"exception: {e}")


# ── Check 7: Baserow rows with correct RFC IDs and titles ────────────────────
_baserow_rows: list[dict] = []
_baserow_field_map: dict[str, str] = {}


def _load_baserow_rows() -> None:
    """Load rows and build field name->key mapping."""
    global _baserow_rows, _baserow_field_map
    if not _baserow_token or not _baserow_table_id:
        return
    # Get fields to map field names to field_<id> keys
    fields = baserow_get(f"database/fields/table/{_baserow_table_id}/", _baserow_token)
    _baserow_field_map = {}
    for f in fields:
        _baserow_field_map[f["name"]] = f"field_{f['id']}"
    # Also include primary field
    # Get rows
    rows_data = baserow_get(f"database/rows/table/{_baserow_table_id}/?size=100", _baserow_token)
    _baserow_rows = rows_data.get("results", [])


def check_7_baserow_rows() -> None:
    """Verify 5 rows exist with correct RFC IDs and titles."""
    try:
        if not _baserow_token or not _baserow_table_id:
            check("7. Baserow rows (IDs & titles)", 2, False, "no table found in check 6")
            return
        _load_baserow_rows()
        if len(_baserow_rows) != 5:
            check("7. Baserow rows (IDs & titles)", 2, False,
                  f"expected 5 rows, got {len(_baserow_rows)}")
            return
        # Map rows by RFC ID
        rfc_id_field = _baserow_field_map.get("RFC ID", "")
        title_field = _baserow_field_map.get("Title", "")
        if not rfc_id_field:
            # Primary field might not be in fields list; try first visible field
            # Baserow returns primary field value in a special way
            check("7. Baserow rows (IDs & titles)", 2, False, "RFC ID field not found in field map")
            return
        issues = []
        found_ids = set()
        for row in _baserow_rows:
            rfc_id = row.get(rfc_id_field, "")
            found_ids.add(rfc_id)
        expected_ids = {f"RFC-{i+1:03d}" for i in range(5)}
        missing_ids = expected_ids - found_ids
        extra_ids = found_ids - expected_ids
        if missing_ids:
            issues.append(f"missing IDs: {missing_ids}")
        if extra_ids:
            issues.append(f"extra IDs: {extra_ids}")
        # Check titles
        if title_field:
            for row in _baserow_rows:
                rfc_id = row.get(rfc_id_field, "")
                title = row.get(title_field, "")
                # Match RFC ID to expected title
                m = re.match(r"RFC-(\d+)", rfc_id)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(RFC_TITLES) and title != RFC_TITLES[idx]:
                        issues.append(f"{rfc_id}: title={title!r}")
        check("7. Baserow rows (IDs & titles)", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("7. Baserow rows (IDs & titles)", 2, False, f"exception: {e}")


# ── Check 8: Baserow row fields (Status, Decision Date, Duration) ────────────
def check_8_baserow_row_fields() -> None:
    """Verify Status, Decision Date, Review Duration Days for each row."""
    try:
        if not _baserow_rows or not _baserow_field_map:
            check("8. Baserow row fields", 2, False, "no rows loaded")
            return
        rfc_id_field = _baserow_field_map.get("RFC ID", "")
        status_field = _baserow_field_map.get("Status", "")
        decision_date_field = _baserow_field_map.get("Decision Date", "")
        duration_field = _baserow_field_map.get("Review Duration Days", "")
        issues = []
        for row in _baserow_rows:
            rfc_id = row.get(rfc_id_field, "")
            m = re.match(r"RFC-(\d+)", rfc_id)
            if not m:
                continue
            idx = int(m.group(1))
            is_approved = idx in APPROVED_INDICES
            # Status (single-select returns dict with value)
            status_val = row.get(status_field, "")
            if isinstance(status_val, dict):
                status_val = status_val.get("value", "")
            expected_status = "Approved" if is_approved else "Draft"
            if status_val != expected_status:
                issues.append(f"{rfc_id}: status={status_val!r}, expected {expected_status!r}")
            # Decision Date
            dd = row.get(decision_date_field, None)
            if is_approved:
                if dd is None or DECISION_DATE not in str(dd):
                    issues.append(f"{rfc_id}: decision_date={dd!r}, expected {DECISION_DATE}")
            else:
                if dd is not None and dd != "":
                    issues.append(f"{rfc_id}: decision_date should be null, got {dd!r}")
            # Review Duration Days
            dur = row.get(duration_field, None)
            expected_dur = REVIEW_DURATION_DAYS if is_approved else 0
            if dur is not None:
                dur_str = str(dur).strip()
                try:
                    dur_int = int(float(dur_str)) if dur_str else 0
                except (ValueError, TypeError):
                    dur_int = -1
                if dur_int != expected_dur:
                    issues.append(f"{rfc_id}: duration={dur}, expected {expected_dur}")
            elif expected_dur != 0:
                issues.append(f"{rfc_id}: duration is null, expected {expected_dur}")
        check("8. Baserow row fields", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("8. Baserow row fields", 2, False, f"exception: {e}")


# ── Check 9: Baserow "Approved RFCs" view ────────────────────────────────────
def check_9_baserow_view() -> None:
    """Verify 'Approved RFCs' grid view exists on the RFC Registry table."""
    try:
        if not _baserow_token or not _baserow_table_id:
            check("9. Baserow 'Approved RFCs' view", 2, False, "no table found")
            return
        views = baserow_get(f"database/views/table/{_baserow_table_id}/", _baserow_token)
        view = None
        for v in views:
            if v.get("name") == "Approved RFCs":
                view = v
                break
        if view is None:
            check("9. Baserow 'Approved RFCs' view", 2, False, "view not found")
            return
        # View exists — check it's a grid type
        is_grid = view.get("type") == "grid"
        check("9. Baserow 'Approved RFCs' view", 2, is_grid,
              f"type={view.get('type')}" if not is_grid else "view exists")
    except Exception as e:
        check("9. Baserow 'Approved RFCs' view", 2, False, f"exception: {e}")


# ── Check 10: OpenProject project exists ──────────────────────────────────────
def op_psql(sql: str) -> str:
    """Run a psql query in the OpenProject container (via postgres peer auth)."""
    import shlex
    inner_cmd = f"psql -d openproject -t -A -c {shlex.quote(sql)}"
    outer_cmd = f"su - postgres -c {shlex.quote(inner_cmd)}"
    rc, out, err = docker_exec(
        OPENPROJECT_CONTAINER,
        "bash", "-c", outer_cmd,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"psql failed: {err.strip()}")
    return out.strip()


_op_project_id: int | None = None


def check_10_op_project() -> None:
    """Verify OpenProject project 'Data Analytics Pipeline' exists."""
    global _op_project_id
    try:
        result = op_psql("SELECT id, name FROM projects WHERE name = 'Data Analytics Pipeline';")
        if not result:
            check("10. OP project exists", 1, False, "project not found")
            return
        parts = result.split("|")
        _op_project_id = int(parts[0])
        check("10. OP project exists", 1, True, f"project_id={_op_project_id}")
    except Exception as e:
        check("10. OP project exists", 1, False, f"exception: {e}")


# ── Check 11: OpenProject Epics with correct subjects ────────────────────────
_op_epics: list[dict] = []


def check_11_op_epic_subjects() -> None:
    """Verify 3 Epic work packages with correct subjects."""
    global _op_epics
    try:
        if _op_project_id is None:
            check("11. OP Epic subjects", 2, False, "no project found")
            return
        # Find the type ID for "Epic"
        type_result = op_psql("SELECT id FROM types WHERE name = 'Epic';")
        if not type_result:
            check("11. OP Epic subjects", 2, False, "Epic type not found in OpenProject")
            return
        epic_type_id = type_result.split("\n")[0].strip()
        # Get work packages
        result = op_psql(
            f"SELECT wp.id, wp.subject FROM work_packages wp "
            f"WHERE wp.project_id = {_op_project_id} AND wp.type_id = {epic_type_id};"
        )
        rows = [r for r in result.split("\n") if r.strip()] if result else []
        _op_epics = []
        for row in rows:
            parts = row.split("|", 1)
            if len(parts) == 2:
                _op_epics.append({"id": int(parts[0]), "subject": parts[1]})
        expected_subjects = []
        for i in sorted(APPROVED_INDICES):
            title = RFC_TITLES[i - 1]
            expected_subjects.append(f"Implement RFC-{i:03d}: {title}")
        found_subjects = {e["subject"] for e in _op_epics}
        missing = [s for s in expected_subjects if s not in found_subjects]
        check("11. OP Epic subjects", 2, not missing and len(_op_epics) >= 3,
              f"found {len(_op_epics)} epics" + (f", missing: {missing}" if missing else ""))
    except Exception as e:
        check("11. OP Epic subjects", 2, False, f"exception: {e}")


# ── Check 12: Epic assignee and priority ──────────────────────────────────────
def check_12_op_epic_assignee_priority() -> None:
    """Verify each Epic is assigned to OpenProject Admin with Normal priority."""
    try:
        if not _op_epics:
            check("12. OP Epic assignee & priority", 2, False, "no epics found")
            return
        epic_ids = ",".join(str(e["id"]) for e in _op_epics)
        # Get assignee and priority for each epic
        result = op_psql(
            f"SELECT wp.id, u.login, u.firstname, u.lastname, "
            f"(SELECT name FROM enumerations WHERE id = wp.priority_id) as priority "
            f"FROM work_packages wp "
            f"LEFT JOIN users u ON u.id = wp.assigned_to_id "
            f"WHERE wp.id IN ({epic_ids});"
        )
        issues = []
        rows = [r for r in result.split("\n") if r.strip()] if result else []
        for row in rows:
            parts = row.split("|")
            if len(parts) >= 5:
                wp_id = parts[0].strip()
                login = parts[1].strip()
                fname = parts[2].strip()
                lname = parts[3].strip()
                priority = parts[4].strip()
                # Check assignee is admin (OpenProject Admin)
                if login != "admin":
                    issues.append(f"wp {wp_id}: assignee={login}, expected admin")
                if priority != "Normal":
                    issues.append(f"wp {wp_id}: priority={priority}, expected Normal")
        check("12. OP Epic assignee & priority", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("12. OP Epic assignee & priority", 2, False, f"exception: {e}")


# ── Check 13: Epic descriptions ──────────────────────────────────────────────
def check_13_op_epic_descriptions() -> None:
    """Verify each Epic description matches expected format."""
    try:
        if not _op_epics:
            check("13. OP Epic descriptions", 2, False, "no epics found")
            return
        issues = []
        for epic in _op_epics:
            # Get description from journals or directly
            desc_result = op_psql(
                f"SELECT description FROM work_packages WHERE id = {epic['id']};"
            )
            desc = desc_result.strip() if desc_result else ""
            subject = epic["subject"]
            # Parse RFC number from subject
            m = re.match(r"Implement RFC-(\d+):", subject)
            if not m:
                issues.append(f"wp {epic['id']}: can't parse RFC number from subject")
                continue
            idx = int(m.group(1))
            fname = RFC_FILENAMES[idx - 1]
            reviewer = RFC_REVIEWERS[f"{idx:03d}"]
            expected_desc = (
                f"Linked RFC file: {RFC_DIR}/{fname}; "
                f"Decision Date: {DECISION_DATE}; "
                f"Reviewer: {reviewer}"
            )
            # OpenProject may wrap description in HTML; check content
            # Strip HTML tags for comparison
            desc_text = re.sub(r"<[^>]+>", "", desc).strip()
            if desc_text != expected_desc:
                issues.append(
                    f"RFC-{idx:03d}: desc mismatch — "
                    f"expected {expected_desc!r}, got {desc_text!r}"
                )
        check("13. OP Epic descriptions", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("13. OP Epic descriptions", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # code-server checks
    check_1_files_exist()
    check_2_headings()
    check_3_metadata_lines()
    check_4_decisions()
    check_5_status_lines()
    # Baserow checks
    check_6_baserow_db_table()
    check_7_baserow_rows()
    check_8_baserow_row_fields()
    check_9_baserow_view()
    # OpenProject checks
    check_10_op_project()
    check_11_op_epic_subjects()
    check_12_op_epic_assignee_priority()
    check_13_op_epic_descriptions()

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
