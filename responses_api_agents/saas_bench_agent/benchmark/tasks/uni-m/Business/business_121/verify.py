# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_121/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-121-I3: Q3 2026 Quarter-End Operations Review Across Four Apps

Checks: 15 weighted checks across twenty, bigcapital, hrms, pretix.
Strategy: docker exec (DB queries) for all checks.

Required env vars:
  SERVER_HOSTNAME,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER,
  HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER,
  PRETIX_PORT, PRETIX_CONTAINER, PRETIX_DB_CONTAINER
"""

import os
import sys
import subprocess

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

TWENTY_PORT = os.getenv("TWENTY_PORT")
TWENTY_CONTAINER = os.getenv("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = os.getenv("TWENTY_DB_CONTAINER")
BIGCAPITAL_PORT = os.getenv("BIGCAPITAL_PORT")
BIGCAPITAL_CONTAINER = os.getenv("BIGCAPITAL_CONTAINER")
HRMS_PORT = os.getenv("HRMS_PORT")
HRMS_CONTAINER = os.getenv("HRMS_CONTAINER")
HRMS_DB_CONTAINER = os.getenv("HRMS_DB_CONTAINER")
PRETIX_PORT = os.getenv("PRETIX_PORT")
PRETIX_CONTAINER = os.getenv("PRETIX_CONTAINER")
PRETIX_DB_CONTAINER = os.getenv("PRETIX_DB_CONTAINER")

_required = {
    "TWENTY_PORT": TWENTY_PORT, "TWENTY_CONTAINER": TWENTY_CONTAINER,
    "TWENTY_DB_CONTAINER": TWENTY_DB_CONTAINER,
    "BIGCAPITAL_PORT": BIGCAPITAL_PORT, "BIGCAPITAL_CONTAINER": BIGCAPITAL_CONTAINER,
    "HRMS_PORT": HRMS_PORT, "HRMS_CONTAINER": HRMS_CONTAINER,
    "HRMS_DB_CONTAINER": HRMS_DB_CONTAINER,
    "PRETIX_PORT": PRETIX_PORT, "PRETIX_CONTAINER": PRETIX_CONTAINER,
    "PRETIX_DB_CONTAINER": PRETIX_DB_CONTAINER,
}
for var_name, var_val in _required.items():
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
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


def twenty_psql(query: str, timeout: int = 15) -> str:
    """Run a psql query against Twenty's Postgres DB (user=postgres, db=default)."""
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default",
        "-t", "-A", "-c", query,
        timeout=timeout,
    )
    if rc != 0:
        raise RuntimeError(f"twenty psql error: {err.strip()}")
    return out.strip()


def bigcapital_mysql(query: str, db: str = "", timeout: int = 15) -> str:
    """Run a mysql query against BigCapital's embedded MariaDB (user=root, in BIGCAPITAL_CONTAINER)."""
    cmd = ["mysql", "-u", "root", "--default-character-set=utf8mb4", "-N", "-B"]
    if db:
        cmd += ["-D", db]
    cmd += ["-e", query]
    rc, out, err = docker_exec(BIGCAPITAL_CONTAINER, *cmd, timeout=timeout)
    if rc != 0:
        raise RuntimeError(f"bigcapital mysql error: {err.strip()}")
    return out.strip()


def pretix_psql(query: str, timeout: int = 15) -> str:
    rc, out, err = docker_exec(
        PRETIX_DB_CONTAINER,
        "psql", "-U", "pretix", "-d", "pretix",
        "-t", "-A", "-c", query,
        timeout=timeout,
    )
    if rc != 0:
        raise RuntimeError(f"pretix psql error: {err.strip()}")
    return out.strip()


def hrms_mysql(query: str, db: str = "", timeout: int = 15) -> str:
    """Run a mysql query against HRMS MariaDB. Discovers the frappe bench DB dynamically."""
    cmd = [
        "mysql", "-u", "root", "-phrms123456",
        "--default-character-set=utf8mb4", "-N", "-B",
    ]
    if db:
        cmd += ["-D", db]
    cmd += ["-e", query]
    rc, out, err = docker_exec(HRMS_DB_CONTAINER, *cmd, timeout=timeout)
    if rc != 0:
        raise RuntimeError(f"hrms mysql error: {err.strip()}")
    return out.strip()


# ── Twenty workspace schema discovery ─────────────────────────────────────────
_ws_schema: str | None = None


def get_twenty_schema() -> str:
    global _ws_schema
    if _ws_schema is None:
        raw = twenty_psql(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'workspace_%' LIMIT 1;"
        )
        if not raw:
            raise RuntimeError("No Twenty workspace schema found")
        _ws_schema = raw.split("\n")[0].strip()
    return _ws_schema


def twenty_ws_query(query: str) -> str:
    """Run a query against the Twenty workspace schema.
    Replaces unqualified table names 'note' and 'task' with schema-qualified versions.
    """
    schema = get_twenty_schema()
    # Replace table references with schema-qualified names
    q = query.replace(" note ", f' "{schema}".note ')
    q = q.replace(" note\n", f' "{schema}".note\n')
    q = q.replace("FROM note", f'FROM "{schema}".note')
    q = q.replace(" task ", f' "{schema}".task ')
    q = q.replace(" task\n", f' "{schema}".task\n')
    q = q.replace("FROM task", f'FROM "{schema}".task')
    return twenty_psql(q)


# ── BigCapital tenant DB discovery ────────────────────────────────────────────
_bc_tenant_db: str | None = None


def get_bigcapital_tenant_db() -> str:
    global _bc_tenant_db
    if _bc_tenant_db is None:
        raw = bigcapital_mysql(
            "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
            "WHERE SCHEMA_NAME LIKE 'bigcapital_tenant_%' LIMIT 1;"
        )
        if not raw:
            raise RuntimeError("No BigCapital tenant DB found")
        _bc_tenant_db = raw.split("\n")[0].strip()
    return _bc_tenant_db


def bigcapital_tenant_query(query: str) -> str:
    db = get_bigcapital_tenant_db()
    return bigcapital_mysql(query, db=db)


# ── HRMS frappe bench DB discovery ────────────────────────────────────────────
_hrms_bench_db: str | None = None


def get_hrms_bench_db() -> str:
    global _hrms_bench_db
    if _hrms_bench_db is None:
        # Find the DB that contains tabCompany (the frappe bench DB)
        dbs = hrms_mysql(
            "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
            "WHERE SCHEMA_NAME NOT IN ('information_schema','mysql','performance_schema','sys','hrms') "
            "AND SCHEMA_NAME LIKE '\\_%';"
        )
        for db_name in dbs.strip().split("\n"):
            db_name = db_name.strip()
            if not db_name:
                continue
            try:
                result = hrms_mysql(
                    "SELECT COUNT(*) FROM `tabCompany`;", db=db_name
                )
                if result.strip().isdigit():
                    _hrms_bench_db = db_name
                    break
            except RuntimeError:
                continue
        if _hrms_bench_db is None:
            raise RuntimeError("No HRMS frappe bench DB found")
    return _hrms_bench_db


def hrms_bench_query(query: str) -> str:
    db = get_hrms_bench_db()
    return hrms_mysql(query, db=db)


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_pipeline_summary_note_exists() -> None:
    """Q3 Pipeline Summary note exists in Twenty."""
    try:
        rows = twenty_ws_query(
            "SELECT id, title FROM note "
            "WHERE title LIKE '%Q3 Pipeline Summary%2026-09-30%' "
            "AND \"deletedAt\" IS NULL;"
        )
        found = bool(rows)
        check("1. Twenty: Q3 Pipeline Summary note exists", 2, found,
              f"found={rows[:120]}" if found else "note not found")
    except Exception as e:
        check("1. Twenty: Q3 Pipeline Summary note exists", 2, False, f"exception: {e}")


def check_2_pipeline_summary_content() -> None:
    """Q3 Pipeline Summary note contains key pipeline data."""
    try:
        body = twenty_ws_query(
            "SELECT COALESCE(\"bodyV2Markdown\", '') FROM note "
            "WHERE title LIKE '%Q3 Pipeline Summary%2026-09-30%' "
            "AND \"deletedAt\" IS NULL LIMIT 1;"
        )
        body_lower = body.lower()
        has_won = "won" in body_lower
        has_lost = "lost" in body_lower
        has_win_rate = "win rate" in body_lower
        has_open = "open" in body_lower or "screening" in body_lower
        all_ok = has_won and has_lost and has_win_rate and has_open
        missing = []
        if not has_won: missing.append("won")
        if not has_lost: missing.append("lost")
        if not has_win_rate: missing.append("win rate")
        if not has_open: missing.append("open/screening")
        check("2. Twenty: Pipeline Summary has pipeline data", 2, all_ok,
              "all sections present" if all_ok else f"missing: {missing}")
    except Exception as e:
        check("2. Twenty: Pipeline Summary has pipeline data", 2, False, f"exception: {e}")


def check_3_ops_review_note_exists() -> None:
    """Q3 Operations Review complete note exists in Twenty."""
    try:
        rows = twenty_ws_query(
            "SELECT id, title FROM note "
            "WHERE title LIKE '%Q3 Operations Review%Complete%2026-09-30%' "
            "AND \"deletedAt\" IS NULL;"
        )
        found = bool(rows)
        check("3. Twenty: Q3 Ops Review note exists", 2, found,
              f"found={rows[:120]}" if found else "note not found")
    except Exception as e:
        check("3. Twenty: Q3 Ops Review note exists", 2, False, f"exception: {e}")


def check_4_ops_review_financial() -> None:
    """Ops Review note contains financial summary (P&L, BS, Cash Flow, A/R, A/P)."""
    try:
        body = twenty_ws_query(
            "SELECT COALESCE(\"bodyV2Markdown\", '') FROM note "
            "WHERE title LIKE '%Q3 Operations Review%Complete%2026-09-30%' "
            "AND \"deletedAt\" IS NULL LIMIT 1;"
        )
        body_lower = body.lower()
        has_pl = "revenue" in body_lower and ("expense" in body_lower) and "net income" in body_lower
        has_bs = "assets" in body_lower and "liabilities" in body_lower and "equity" in body_lower
        has_ar = "a/r" in body_lower or "receivable" in body_lower
        has_ap = "a/p" in body_lower or "payable" in body_lower
        has_cash = "cash" in body_lower
        all_ok = has_pl and has_bs and has_ar and has_ap and has_cash
        missing = []
        if not has_pl: missing.append("P&L")
        if not has_bs: missing.append("Balance Sheet")
        if not has_ar: missing.append("A/R")
        if not has_ap: missing.append("A/P")
        if not has_cash: missing.append("Cash Flow")
        check("4. Twenty: Ops Review has financial summary", 3, all_ok,
              "all financial sections" if all_ok else f"missing: {missing}")
    except Exception as e:
        check("4. Twenty: Ops Review has financial summary", 3, False, f"exception: {e}")


def check_5_ops_review_crm() -> None:
    """Ops Review note contains CRM pipeline section."""
    try:
        body = twenty_ws_query(
            "SELECT COALESCE(\"bodyV2Markdown\", '') FROM note "
            "WHERE title LIKE '%Q3 Operations Review%Complete%2026-09-30%' "
            "AND \"deletedAt\" IS NULL LIMIT 1;"
        )
        body_lower = body.lower()
        has_won = "won" in body_lower
        has_lost = "lost" in body_lower
        has_win_rate = "win rate" in body_lower
        all_ok = has_won and has_lost and has_win_rate
        check("5. Twenty: Ops Review has CRM pipeline section", 2, all_ok,
              "CRM section present" if all_ok else f"won={has_won} lost={has_lost} winrate={has_win_rate}")
    except Exception as e:
        check("5. Twenty: Ops Review has CRM pipeline section", 2, False, f"exception: {e}")


def check_6_ops_review_hr() -> None:
    """Ops Review note contains HR metrics section."""
    try:
        body = twenty_ws_query(
            "SELECT COALESCE(\"bodyV2Markdown\", '') FROM note "
            "WHERE title LIKE '%Q3 Operations Review%Complete%2026-09-30%' "
            "AND \"deletedAt\" IS NULL LIMIT 1;"
        )
        body_lower = body.lower()
        has_headcount = "headcount" in body_lower
        has_absence = "absence" in body_lower
        has_expense = "expense" in body_lower or "unpaid" in body_lower
        has_leave = "leave" in body_lower or "sick" in body_lower
        all_ok = has_headcount and has_absence and has_expense and has_leave
        missing = []
        if not has_headcount: missing.append("headcount")
        if not has_absence: missing.append("absence")
        if not has_expense: missing.append("expense claims")
        if not has_leave: missing.append("leave liability")
        check("6. Twenty: Ops Review has HR metrics", 2, all_ok,
              "HR section present" if all_ok else f"missing: {missing}")
    except Exception as e:
        check("6. Twenty: Ops Review has HR metrics", 2, False, f"exception: {e}")


def check_7_ops_review_event() -> None:
    """Ops Review note contains event performance section (Hamilton)."""
    try:
        body = twenty_ws_query(
            "SELECT COALESCE(\"bodyV2Markdown\", '') FROM note "
            "WHERE title LIKE '%Q3 Operations Review%Complete%2026-09-30%' "
            "AND \"deletedAt\" IS NULL LIMIT 1;"
        )
        body_lower = body.lower()
        has_hamilton = "hamilton" in body_lower
        has_balcony = "balcony" in body_lower
        has_playbill = "playbill" in body_lower
        all_ok = has_hamilton and has_balcony and has_playbill
        missing = []
        if not has_hamilton: missing.append("Hamilton")
        if not has_balcony: missing.append("Balcony")
        if not has_playbill: missing.append("Playbill")
        check("7. Twenty: Ops Review has event performance", 2, all_ok,
              "event section present" if all_ok else f"missing: {missing}")
    except Exception as e:
        check("7. Twenty: Ops Review has event performance", 2, False, f"exception: {e}")


def check_8_presentation_task() -> None:
    """Presentation task exists with correct due date 2026-10-22."""
    try:
        rows = twenty_ws_query(
            "SELECT title, \"dueAt\"::text FROM task "
            "WHERE title LIKE '%Present Q3 operations review to leadership%' "
            "AND \"deletedAt\" IS NULL;"
        )
        found = bool(rows)
        has_due = "2026-10-22" in rows if found else False
        ok = found and has_due
        check("8. Twenty: Presentation task exists", 2, ok,
              "found, due date correct" if ok else
              f"found={found}, due_date_ok={has_due}, rows={rows[:120]}")
    except Exception as e:
        check("8. Twenty: Presentation task exists", 2, False, f"exception: {e}")


def check_9_followup_tasks() -> None:
    """At least one follow-up task for overdue receivables with due 2026-10-15."""
    try:
        rows = twenty_ws_query(
            "SELECT title, \"dueAt\"::text FROM task "
            "WHERE title LIKE '%Follow up on overdue receivable%' "
            "AND \"deletedAt\" IS NULL;"
        )
        found = bool(rows)
        has_due = "2026-10-15" in rows if found else False
        count = len([l for l in rows.strip().split("\n") if l.strip()]) if found else 0
        ok = found and has_due
        check("9. Twenty: Follow-up tasks for overdue clients", 2, ok,
              f"{count} task(s) found with due 2026-10-15" if ok else
              f"found={found}, due_ok={has_due}")
    except Exception as e:
        check("9. Twenty: Follow-up tasks for overdue clients", 2, False, f"exception: {e}")


def check_10_pretix_hamilton_event() -> None:
    """Hamilton event exists under broadway-group in Pretix."""
    try:
        rows = pretix_psql(
            "SELECT e.slug, e.live FROM pretixbase_event e "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'broadway-group' "
            "AND (e.slug ILIKE '%hamilton%' OR e.name::text ILIKE '%Hamilton%');"
        )
        found = bool(rows)
        check("10. Pretix: Hamilton event exists", 1, found,
              f"found={rows[:120]}" if found else "event not found")
    except Exception as e:
        check("10. Pretix: Hamilton event exists", 1, False, f"exception: {e}")


def check_11_pretix_products() -> None:
    """Balcony and Playbill Program products exist for Hamilton event."""
    try:
        rows = pretix_psql(
            "SELECT i.name::text FROM pretixbase_item i "
            "JOIN pretixbase_event e ON i.event_id = e.id "
            "JOIN pretixbase_organizer o ON e.organizer_id = o.id "
            "WHERE o.slug = 'broadway-group' "
            "AND (e.slug ILIKE '%hamilton%' OR e.name::text ILIKE '%Hamilton%');"
        )
        rows_lower = rows.lower()
        has_balcony = "balcony" in rows_lower
        has_playbill = "playbill" in rows_lower
        ok = has_balcony and has_playbill
        missing = []
        if not has_balcony: missing.append("Balcony")
        if not has_playbill: missing.append("Playbill Program")
        check("11. Pretix: Balcony & Playbill products exist", 1, ok,
              "both found" if ok else f"missing: {missing}, items: {rows[:200]}")
    except Exception as e:
        check("11. Pretix: Balcony & Playbill products exist", 1, False, f"exception: {e}")


def check_12_hrms_company() -> None:
    """TechVista Solutions Pvt. Ltd. company exists in HRMS."""
    try:
        rows = hrms_bench_query(
            "SELECT name FROM `tabCompany` "
            "WHERE name LIKE '%TechVista%' LIMIT 1;"
        )
        found = bool(rows)
        check("12. HRMS: TechVista company exists", 1, found,
              f"found={rows[:80]}" if found else "company not found")
    except Exception as e:
        check("12. HRMS: TechVista company exists", 1, False, f"exception: {e}")


def check_13_hrms_employees() -> None:
    """Active employees exist in HRMS for attendance verification."""
    try:
        rows = hrms_bench_query(
            "SELECT COUNT(*) FROM `tabEmployee` WHERE status = 'Active';"
        )
        count = int(rows.strip()) if rows.strip().isdigit() else 0
        ok = count > 0
        check("13. HRMS: Active employees exist", 1, ok,
              f"{count} active employees" if ok else "no active employees found")
    except Exception as e:
        check("13. HRMS: Active employees exist", 1, False, f"exception: {e}")


def check_14_bigcapital_customers() -> None:
    """Customers exist in BigCapital for A/R aging data."""
    try:
        rows = bigcapital_tenant_query(
            "SELECT COUNT(*) FROM CONTACTS WHERE CONTACT_SERVICE = 'customer';"
        )
        count = int(rows.strip()) if rows.strip().isdigit() else 0
        ok = count > 0
        check("14. BigCapital: Customers exist", 1, ok,
              f"{count} customers" if ok else "no customers found")
    except Exception as e:
        check("14. BigCapital: Customers exist", 1, False, f"exception: {e}")



# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_pipeline_summary_note_exists()
    check_2_pipeline_summary_content()
    check_3_ops_review_note_exists()
    check_4_ops_review_financial()
    check_5_ops_review_crm()
    check_6_ops_review_hr()
    check_7_ops_review_event()
    check_8_presentation_task()
    check_9_followup_tasks()
    check_10_pretix_hamilton_event()
    check_11_pretix_products()
    check_12_hrms_company()
    check_13_hrms_employees()
    check_14_bigcapital_customers()

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
