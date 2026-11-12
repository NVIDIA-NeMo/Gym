# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Software/software_038/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Software-038-I1: Build Code-to-Test Ratio Engineering Metrics Dashboard

Checks: 16 weighted checks across code-server, baserow, metabase, openproject.
Strategy: docker exec (code-server filesystem, openproject DB), REST API (baserow, metabase)

Required env vars:
  SERVER_HOSTNAME, CODE_SERVER_PORT, CODE_SERVER_CONTAINER,
  BASEROW_PORT, BASEROW_CONTAINER, BASEROW_DB_CONTAINER,
  METABASE_PORT, METABASE_CONTAINER,
  OPENPROJECT_PORT, OPENPROJECT_CONTAINER
"""

import os
import sys
import subprocess
import json
import re

import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

CODE_SERVER_PORT = os.environ.get("CODE_SERVER_PORT")
CODE_SERVER_CONTAINER = os.environ.get("CODE_SERVER_CONTAINER")
BASEROW_PORT = os.environ.get("BASEROW_PORT")
BASEROW_CONTAINER = os.environ.get("BASEROW_CONTAINER")
BASEROW_DB_CONTAINER = os.environ.get("BASEROW_DB_CONTAINER")
METABASE_PORT = os.environ.get("METABASE_PORT")
METABASE_CONTAINER = os.environ.get("METABASE_CONTAINER")
OPENPROJECT_PORT = os.environ.get("OPENPROJECT_PORT")
OPENPROJECT_CONTAINER = os.environ.get("OPENPROJECT_CONTAINER")

for _var in [
    "CODE_SERVER_PORT", "CODE_SERVER_CONTAINER",
    "BASEROW_PORT", "BASEROW_CONTAINER", "BASEROW_DB_CONTAINER",
    "METABASE_PORT", "METABASE_CONTAINER",
    "OPENPROJECT_PORT", "OPENPROJECT_CONTAINER",
]:
    if not os.environ.get(_var):
        print(f"FATAL: {_var} not set", file=sys.stderr)
        sys.exit(1)

BASEROW_URL = f"http://{HOST}:{BASEROW_PORT}"
METABASE_URL = f"http://{HOST}:{METABASE_PORT}"

PROJECTS = ["blog-engine", "data-analyzer", "json", "todo-api"]  # alphabetical

COMMANDS = {
    "todo-api": "echo \"$(find app -type f -name '*.py' | wc -l) $(find tests -type f -name 'test_*.py' | wc -l)\"",
    "blog-engine": "echo \"$(find src -type f -name '*.js' | wc -l) $(find tests -type f -name '*.test.js' 2>/dev/null | wc -l)\"",
    "data-analyzer": "echo \"$(find src -type f -name '*.py' | wc -l) $(find tests -type f -name 'test_*.py' | wc -l)\"",
    "json": "echo \"$(find include -type f -name '*.hpp' | wc -l) $(find tests/src -type f -name 'unit-*.cpp' | wc -l)\"",
}

# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Helpers ───────────────────────────────────────────────────────────────────
def docker_exec(container: str, *args: str, timeout: int = 15,
                env_vars: dict[str, str] | None = None) -> tuple[int, str, str]:
    cmd = ["docker", "exec"]
    for k, v in (env_vars or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [container, *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


_baserow_token: str | None = None


def baserow_headers() -> dict:
    global _baserow_token
    if _baserow_token is None:
        resp = requests.post(
            f"{BASEROW_URL}/api/user/token-auth/",
            json={"email": "admin@example.com", "password": "Admin1234"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _baserow_token = data.get("access_token") or data.get("token")
    return {"Authorization": f"JWT {_baserow_token}"}


_metabase_session: str | None = None


def metabase_headers() -> dict:
    global _metabase_session
    if _metabase_session is None:
        resp = requests.post(
            f"{METABASE_URL}/api/session",
            json={"username": "admin@metabase.local", "password": "mw-admin-123"},
            timeout=15,
        )
        resp.raise_for_status()
        _metabase_session = resp.json()["id"]
    return {"X-Metabase-Session": _metabase_session}


def compute_tier(ratio: float) -> str:
    if ratio >= 0.8:
        return "Gold"
    if ratio >= 0.5:
        return "Silver"
    if ratio >= 0.2:
        return "Bronze"
    return "AtRisk"


# ── Shared state across checks ───────────────────────────────────────────────
ref_data: dict[str, tuple[int, int]] = {}  # project -> (S, T)
_baserow_table_id: int | None = None
_baserow_fields: dict = {}   # field_name -> field dict
_baserow_rows: list = []
_baserow_views: list = []
_mb_collection_id: int | None = None
_mb_dashboard_id: int | None = None


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_code_server_projects() -> None:
    """Gather reference S, T from code-server project directories."""
    try:
        # Locate workspace root by finding one of the known project dirs
        rc, out, _ = docker_exec(
            CODE_SERVER_CONTAINER, "bash", "-c",
            "find / -maxdepth 4 -type d -name 'todo-api' 2>/dev/null | head -1",
            timeout=30,
        )
        root = ""
        if rc == 0 and out.strip():
            # root is parent of todo-api
            todo_path = out.strip().rstrip("/")
            root = todo_path.rsplit("/", 1)[0] if "/" in todo_path else ""

        if not root:
            root = "/home/coder"

        for proj in PROJECTS:
            cmd = f"cd {root}/{proj} && {COMMANDS[proj]}"
            rc2, out2, _ = docker_exec(
                CODE_SERVER_CONTAINER, "bash", "-c", cmd, timeout=30
            )
            if rc2 == 0 and out2.strip():
                parts = out2.strip().split()
                if len(parts) >= 2:
                    try:
                        ref_data[proj] = (int(parts[0]), int(parts[1]))
                    except ValueError:
                        pass

        check(
            "1. Code-server project dirs & file counts", 1,
            len(ref_data) == 4,
            f"found {len(ref_data)}/4: {ref_data}",
        )
    except Exception as e:
        check("1. Code-server project dirs & file counts", 1, False, f"exception: {e}")


def check_2_baserow_database() -> None:
    """Verify Baserow database 'Engineering Quality Metrics' exists."""
    try:
        headers = baserow_headers()
        resp = requests.get(f"{BASEROW_URL}/api/applications/", headers=headers, timeout=15)
        resp.raise_for_status()
        apps = resp.json()
        if isinstance(apps, dict):
            apps = apps.get("results", apps.get("applications", []))

        found = any(a.get("name") == "Engineering Quality Metrics" for a in apps)
        check(
            "2. Baserow DB 'Engineering Quality Metrics'", 1, found,
            f"databases: {[a.get('name') for a in apps]}",
        )
    except Exception as e:
        check("2. Baserow DB 'Engineering Quality Metrics'", 1, False, f"exception: {e}")


def check_3_baserow_table_fields() -> None:
    """Verify table 'Project Metrics' exists with required fields."""
    global _baserow_table_id, _baserow_fields
    try:
        headers = baserow_headers()
        resp = requests.get(f"{BASEROW_URL}/api/applications/", headers=headers, timeout=15)
        resp.raise_for_status()
        apps = resp.json()
        if isinstance(apps, dict):
            apps = apps.get("results", apps.get("applications", []))

        db_id = None
        for a in apps:
            if a.get("name") == "Engineering Quality Metrics":
                db_id = a["id"]
                break
        if not db_id:
            check("3. Table 'Project Metrics' with fields", 2, False, "database not found")
            return

        resp = requests.get(
            f"{BASEROW_URL}/api/database/tables/database/{db_id}/",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        tables = resp.json()

        for t in tables:
            if t.get("name") == "Project Metrics":
                _baserow_table_id = t["id"]
                break
        if not _baserow_table_id:
            check(
                "3. Table 'Project Metrics' with fields", 2, False,
                f"table not found; tables={[t.get('name') for t in tables]}",
            )
            return

        resp = requests.get(
            f"{BASEROW_URL}/api/database/fields/table/{_baserow_table_id}/",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        fields = resp.json()
        _baserow_fields = {f["name"]: f for f in fields}

        required = {"Project", "Source Files", "Test Files", "Test Coverage Ratio",
                     "Quality Tier", "Measured At"}
        missing = required - set(_baserow_fields.keys())
        check(
            "3. Table 'Project Metrics' with fields", 2,
            len(missing) == 0,
            f"missing: {missing}" if missing else f"all {len(required)} fields present",
        )
    except Exception as e:
        check("3. Table 'Project Metrics' with fields", 2, False, f"exception: {e}")


def _field_key(name: str) -> str | None:
    f = _baserow_fields.get(name)
    return f"field_{f['id']}" if f else None


def _row_value(row: dict, field_name: str):
    """Extract value from a Baserow row for the given field name."""
    key = _field_key(field_name)
    if not key:
        return None
    val = row.get(key)
    if isinstance(val, dict):
        return val.get("value", "")
    return val


def check_4_baserow_rows() -> None:
    """Verify exactly 4 rows, one per project."""
    global _baserow_rows
    try:
        if not _baserow_table_id:
            check("4. Four project rows", 2, False, "table not found")
            return
        headers = baserow_headers()
        resp = requests.get(
            f"{BASEROW_URL}/api/database/rows/table/{_baserow_table_id}/",
            headers=headers, params={"size": 100}, timeout=15,
        )
        resp.raise_for_status()
        _baserow_rows = resp.json().get("results", [])

        row_projects = sorted(str(_row_value(r, "Project") or "") for r in _baserow_rows)
        check(
            "4. Four project rows", 2,
            len(_baserow_rows) == 4 and row_projects == PROJECTS,
            f"expected {PROJECTS}, got {row_projects}",
        )
    except Exception as e:
        check("4. Four project rows", 2, False, f"exception: {e}")


def check_5_source_test_counts() -> None:
    """Verify Source Files and Test Files match code-server output."""
    try:
        if not _baserow_rows or not ref_data:
            check("5. Source/Test file counts match", 2, False, "no rows or ref data")
            return

        mismatches = []
        for r in _baserow_rows:
            proj = str(_row_value(r, "Project") or "")
            if proj not in ref_data:
                mismatches.append(f"{proj}: no ref data")
                continue
            exp_s, exp_t = ref_data[proj]
            try:
                got_s = int(float(str(_row_value(r, "Source Files") or 0)))
                got_t = int(float(str(_row_value(r, "Test Files") or 0)))
            except (ValueError, TypeError):
                mismatches.append(f"{proj}: non-numeric values")
                continue
            if got_s != exp_s or got_t != exp_t:
                mismatches.append(f"{proj}: S={got_s}(exp {exp_s}), T={got_t}(exp {exp_t})")

        check(
            "5. Source/Test file counts match", 2,
            len(mismatches) == 0,
            "; ".join(mismatches) if mismatches else "all counts match",
        )
    except Exception as e:
        check("5. Source/Test file counts match", 2, False, f"exception: {e}")


def check_6_ratios_tiers() -> None:
    """Verify Test Coverage Ratio and Quality Tier correctness."""
    try:
        if not _baserow_rows or not ref_data:
            check("6. Ratios & tiers correct", 3, False, "no rows or ref data")
            return

        mismatches = []
        for r in _baserow_rows:
            proj = str(_row_value(r, "Project") or "")
            if proj not in ref_data:
                continue
            S, T = ref_data[proj]
            exp_ratio = round(T / S, 3) if S > 0 else 0.0
            exp_tier = compute_tier(exp_ratio)

            try:
                got_ratio = float(str(_row_value(r, "Test Coverage Ratio") or 0))
            except (ValueError, TypeError):
                got_ratio = None
            got_tier = str(_row_value(r, "Quality Tier") or "")

            ratio_ok = got_ratio is not None and abs(got_ratio - exp_ratio) < 0.0005
            tier_ok = got_tier == exp_tier
            if not ratio_ok or not tier_ok:
                mismatches.append(
                    f"{proj}: ratio={got_ratio}(exp {exp_ratio}), tier={got_tier}(exp {exp_tier})"
                )

        check(
            "6. Ratios & tiers correct", 3,
            len(mismatches) == 0,
            "; ".join(mismatches) if mismatches else "all correct",
        )
    except Exception as e:
        check("6. Ratios & tiers correct", 3, False, f"exception: {e}")


def check_7_measured_at() -> None:
    """Verify Measured At = 2026-04-01 for all rows."""
    try:
        if not _baserow_rows:
            check("7. Measured At dates", 1, False, "no rows")
            return
        dates = [str(_row_value(r, "Measured At") or "") for r in _baserow_rows]
        ok = all("2026-04-01" in d for d in dates)
        check("7. Measured At dates", 1, ok, f"dates={dates}")
    except Exception as e:
        check("7. Measured At dates", 1, False, f"exception: {e}")


def check_8_grid_view() -> None:
    """Verify Grid view 'Quality Ranking' exists."""
    global _baserow_views
    try:
        if not _baserow_table_id:
            check("8. Grid view 'Quality Ranking'", 1, False, "table not found")
            return
        headers = baserow_headers()
        resp = requests.get(
            f"{BASEROW_URL}/api/database/views/table/{_baserow_table_id}/",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        _baserow_views = resp.json()
        found = any(
            v.get("name") == "Quality Ranking" and v.get("type") == "grid"
            for v in _baserow_views
        )
        check(
            "8. Grid view 'Quality Ranking'", 1, found,
            f"views={[(v.get('name'), v.get('type')) for v in _baserow_views]}",
        )
    except Exception as e:
        check("8. Grid view 'Quality Ranking'", 1, False, f"exception: {e}")


def check_9_kanban_view() -> None:
    """Verify Kanban view 'By Tier' exists."""
    try:
        views = _baserow_views
        if not views and _baserow_table_id:
            headers = baserow_headers()
            resp = requests.get(
                f"{BASEROW_URL}/api/database/views/table/{_baserow_table_id}/",
                headers=headers, timeout=15,
            )
            resp.raise_for_status()
            views = resp.json()
        found = any(
            v.get("name") == "By Tier" and v.get("type") == "kanban"
            for v in views
        )
        check(
            "9. Kanban view 'By Tier'", 1, found,
            f"views={[(v.get('name'), v.get('type')) for v in views]}",
        )
    except Exception as e:
        check("9. Kanban view 'By Tier'", 1, False, f"exception: {e}")


def check_10_metabase_collection() -> None:
    """Verify Metabase collection 'Code Quality Audit Q2 2026' exists."""
    global _mb_collection_id
    try:
        headers = metabase_headers()
        resp = requests.get(f"{METABASE_URL}/api/collection", headers=headers, timeout=15)
        resp.raise_for_status()
        collections = resp.json()
        for c in collections:
            if c.get("name") == "Code Quality Audit Q2 2026":
                _mb_collection_id = c["id"]
                break
        check(
            "10. Metabase collection", 1,
            _mb_collection_id is not None,
            f"id={_mb_collection_id}" if _mb_collection_id
            else f"not found in {[c.get('name') for c in collections]}",
        )
    except Exception as e:
        check("10. Metabase collection", 1, False, f"exception: {e}")


def check_11_metabase_questions() -> None:
    """Verify 3 questions with correct names in the collection."""
    try:
        if _mb_collection_id is None:
            check("11. Three Metabase questions", 2, False, "collection not found")
            return
        headers = metabase_headers()
        resp = requests.get(
            f"{METABASE_URL}/api/collection/{_mb_collection_id}/items",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        items = resp.json()
        item_list = items.get("data", []) if isinstance(items, dict) else items

        question_names = {i["name"] for i in item_list if i.get("model") == "card"}
        expected = {"Source vs Test File Counts", "Tier Distribution", "Ratio Ranking"}
        missing = expected - question_names
        check(
            "11. Three Metabase questions", 2,
            len(missing) == 0,
            f"missing: {missing}" if missing else "all 3 found",
        )
    except Exception as e:
        check("11. Three Metabase questions", 2, False, f"exception: {e}")


def check_12_metabase_dashboard() -> None:
    """Verify dashboard name and description."""
    global _mb_dashboard_id
    try:
        if _mb_collection_id is None:
            check("12. Metabase dashboard", 2, False, "collection not found")
            return
        headers = metabase_headers()
        resp = requests.get(
            f"{METABASE_URL}/api/collection/{_mb_collection_id}/items",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        items = resp.json()
        item_list = items.get("data", []) if isinstance(items, dict) else items

        for d in item_list:
            if d.get("model") == "dashboard" and d.get("name") == "Code-to-Test Coverage Audit":
                _mb_dashboard_id = d["id"]
                break

        if _mb_dashboard_id is None:
            dash_names = [d.get("name") for d in item_list if d.get("model") == "dashboard"]
            check("12. Metabase dashboard", 2, False, f"not found; dashboards={dash_names}")
            return

        resp = requests.get(
            f"{METABASE_URL}/api/dashboard/{_mb_dashboard_id}",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        dash = resp.json()
        desc = (dash.get("description") or "").strip()
        expected_desc = "Code-to-test ratio audit 2026-04-01 across 4 projects"
        check(
            "12. Metabase dashboard", 2,
            desc == expected_desc,
            f"desc='{desc}'" if desc != expected_desc else "name & description correct",
        )
    except Exception as e:
        check("12. Metabase dashboard", 2, False, f"exception: {e}")


def check_13_dashboard_cards() -> None:
    """Verify dashboard has 3 question cards."""
    try:
        if _mb_dashboard_id is None:
            check("13. Dashboard has 3 cards", 1, False, "dashboard not found")
            return
        headers = metabase_headers()
        resp = requests.get(
            f"{METABASE_URL}/api/dashboard/{_mb_dashboard_id}",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        dash = resp.json()
        cards = dash.get("ordered_cards", dash.get("dashcards", []))
        real_cards = [c for c in cards if c.get("card_id")]
        check(
            "13. Dashboard has 3 cards", 1,
            len(real_cards) == 3,
            f"found {len(real_cards)} question cards",
        )
    except Exception as e:
        check("13. Dashboard has 3 cards", 1, False, f"exception: {e}")


def _bronze_atrisk() -> dict[str, tuple[int, int, float, str]]:
    """Return {project: (S, T, ratio, tier)} for Bronze/AtRisk projects."""
    result = {}
    for proj in PROJECTS:
        if proj in ref_data:
            S, T = ref_data[proj]
            ratio = round(T / S, 3) if S > 0 else 0.0
            tier = compute_tier(ratio)
            if tier in ("Bronze", "AtRisk"):
                result[proj] = (S, T, ratio, tier)
    return result


def check_14_op_work_packages() -> None:
    """Verify OpenProject work packages exist for Bronze/AtRisk projects."""
    try:
        ba = _bronze_atrisk()
        if not ref_data:
            check("14. OP work packages exist", 2, False, "no ref data")
            return

        rc, out, err = docker_exec(
            OPENPROJECT_CONTAINER,
            "psql", "-h", "127.0.0.1", "-U", "openproject", "-d", "openproject",
            "-t", "-A", "-c",
            "SELECT wp.subject, t.name "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.name = 'Customer Portal Redesign' "
            "AND t.name = 'Task' "
            "AND wp.subject LIKE 'Raise test coverage:%'",
            env_vars={"PGPASSWORD": "openproject"},
        )
        if rc != 0:
            check("14. OP work packages exist", 2, False, f"psql error: {err.strip()}")
            return

        found_subjects = [
            line.split("|")[0]
            for line in out.strip().split("\n")
            if line.strip()
        ]
        expected_subjects = [
            f"Raise test coverage: {proj} (ratio {ratio:.3f})"
            for proj, (_, _, ratio, _) in sorted(ba.items())
        ]

        missing = [s for s in expected_subjects if s not in found_subjects]
        extra = [s for s in found_subjects if s not in expected_subjects]
        ok = len(missing) == 0 and len(extra) == 0 and len(expected_subjects) == len(found_subjects)

        details = []
        if missing:
            details.append(f"missing: {missing}")
        if extra:
            details.append(f"extra: {extra}")
        if not details:
            details.append(f"{len(found_subjects)} WPs match")
        check("14. OP work packages exist", 2, ok, "; ".join(details))
    except Exception as e:
        check("14. OP work packages exist", 2, False, f"exception: {e}")


def check_15_op_assignee_priority() -> None:
    """Verify work package assignee=qa_lead and correct priority."""
    try:
        ba = _bronze_atrisk()
        rc, out, err = docker_exec(
            OPENPROJECT_CONTAINER,
            "psql", "-h", "127.0.0.1", "-U", "openproject", "-d", "openproject",
            "-t", "-A", "-c",
            "SELECT wp.subject, u.login, e.name "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "LEFT JOIN users u ON wp.assigned_to_id = u.id "
            "LEFT JOIN enumerations e ON wp.priority_id = e.id "
            "WHERE p.name = 'Customer Portal Redesign' "
            "AND t.name = 'Task' "
            "AND wp.subject LIKE 'Raise test coverage:%'",
            env_vars={"PGPASSWORD": "openproject"},
        )
        if rc != 0:
            check("15. WP assignee & priority", 2, False, f"psql error: {err.strip()}")
            return

        issues = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            subject, assignee, priority = parts[0], parts[1], parts[2]

            if assignee != "qa_lead":
                issues.append(f"assignee={assignee} for '{subject}'")

            for proj, (_, _, ratio, tier) in ba.items():
                exp_subj = f"Raise test coverage: {proj} (ratio {ratio:.3f})"
                if subject == exp_subj:
                    exp_prio = "High" if tier == "AtRisk" else "Normal"
                    if priority != exp_prio:
                        issues.append(f"{proj}: priority={priority}(exp {exp_prio})")

        check(
            "15. WP assignee & priority", 2,
            len(issues) == 0,
            "; ".join(issues) if issues else "all correct",
        )
    except Exception as e:
        check("15. WP assignee & priority", 2, False, f"exception: {e}")


def check_16_op_description() -> None:
    """Verify work package description format."""
    try:
        ba = _bronze_atrisk()
        rc, out, err = docker_exec(
            OPENPROJECT_CONTAINER,
            "psql", "-h", "127.0.0.1", "-U", "openproject", "-d", "openproject",
            "-t", "-A", "-c",
            "SELECT wp.subject, wp.description "
            "FROM work_packages wp "
            "JOIN projects p ON wp.project_id = p.id "
            "JOIN types t ON wp.type_id = t.id "
            "WHERE p.name = 'Customer Portal Redesign' "
            "AND t.name = 'Task' "
            "AND wp.subject LIKE 'Raise test coverage:%'",
            env_vars={"PGPASSWORD": "openproject"},
        )
        if rc != 0:
            check("16. WP descriptions", 2, False, f"psql error: {err.strip()}")
            return

        issues = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|", 1)
            if len(parts) < 2:
                continue
            subject, desc_raw = parts[0], parts[1]
            # Strip HTML tags (OpenProject stores rich text)
            desc_text = re.sub(r"<[^>]+>", "", desc_raw).strip()

            for proj, (S, T, ratio, tier) in ba.items():
                exp_subj = f"Raise test coverage: {proj} (ratio {ratio:.3f})"
                if subject == exp_subj:
                    exp_desc = f"Source: {S}; Tests: {T}; Tier: {tier}; Measured: 2026-04-01"
                    if desc_text != exp_desc:
                        issues.append(f"{proj}: got '{desc_text}', exp '{exp_desc}'")

        check(
            "16. WP descriptions", 2,
            len(issues) == 0,
            "; ".join(issues) if issues else "all descriptions correct",
        )
    except Exception as e:
        check("16. WP descriptions", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_code_server_projects()
    check_2_baserow_database()
    check_3_baserow_table_fields()
    check_4_baserow_rows()
    check_5_source_test_counts()
    check_6_ratios_tiers()
    check_7_measured_at()
    check_8_grid_view()
    check_9_kanban_view()
    check_10_metabase_collection()
    check_11_metabase_questions()
    check_12_metabase_dashboard()
    check_13_dashboard_cards()
    check_14_op_work_packages()
    check_15_op_assignee_priority()
    check_16_op_description()

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
