#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_023/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-023-I1: Process Expense Reimbursement for Mohammed Farooq
Across HRMS, BigCapital, and Twenty CRM.

Checks: 11 weighted checks (20 points total) across hrms, bigcapital, twenty.
Strategy: docker exec MariaDB for HRMS and BigCapital, docker exec Postgres for Twenty.

Required env vars:
  SERVER_HOSTNAME, HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  TWENTY_PORT, TWENTY_CONTAINER, TWENTY_DB_CONTAINER
"""

import os
import sys
import subprocess
import json


# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

HRMS_PORT = os.environ.get("HRMS_PORT")
HRMS_CONTAINER = os.environ.get("HRMS_CONTAINER")
HRMS_DB_CONTAINER = os.environ.get("HRMS_DB_CONTAINER")

BC_PORT = os.environ.get("BIGCAPITAL_PORT")
BC_CONTAINER = os.environ.get("BIGCAPITAL_CONTAINER")
BC_DB_CONTAINER = os.environ.get("BIGCAPITAL_DB_CONTAINER")

TWENTY_PORT = os.environ.get("TWENTY_PORT")
TWENTY_CONTAINER = os.environ.get("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = os.environ.get("TWENTY_DB_CONTAINER")

_required = {
    "HRMS_PORT": HRMS_PORT, "HRMS_CONTAINER": HRMS_CONTAINER,
    "HRMS_DB_CONTAINER": HRMS_DB_CONTAINER,
    "BIGCAPITAL_PORT": BC_PORT, "BIGCAPITAL_CONTAINER": BC_CONTAINER,
    "BIGCAPITAL_DB_CONTAINER": BC_DB_CONTAINER,
    "TWENTY_PORT": TWENTY_PORT, "TWENTY_CONTAINER": TWENTY_CONTAINER,
    "TWENTY_DB_CONTAINER": TWENTY_DB_CONTAINER,
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


# ── HRMS DB helpers ───────────────────────────────────────────────────────────
_hrms_db_name: str = ""
_hrms_db_password: str = ""


def _discover_hrms_db() -> None:
    """Read Frappe site_config.json to get DB name and password."""
    global _hrms_db_name, _hrms_db_password
    # Get default site name
    rc, out, err = docker_exec(
        HRMS_CONTAINER, "cat",
        "/home/frappe/frappe-bench/sites/common_site_config.json",
    )
    if rc != 0:
        raise RuntimeError(f"Cannot read common_site_config: {err.strip()}")
    common = json.loads(out)
    site = common.get("default_site", "hrms.localhost")
    # Get site-specific config
    rc, out, err = docker_exec(
        HRMS_CONTAINER, "cat",
        f"/home/frappe/frappe-bench/sites/{site}/site_config.json",
    )
    if rc != 0:
        raise RuntimeError(f"Cannot read site_config for {site}: {err.strip()}")
    site_cfg = json.loads(out)
    _hrms_db_name = site_cfg["db_name"]
    _hrms_db_password = site_cfg.get("db_password", "")


def hrms_query(sql: str) -> str:
    """Run a MariaDB query on the HRMS database."""
    if not _hrms_db_name:
        _discover_hrms_db()
    # Use the site-specific DB user (same as db_name in Frappe) with its password
    args = [
        "mysql", "--default-character-set=utf8mb4",
        "-u", _hrms_db_name,
    ]
    if _hrms_db_password:
        args.append(f"-p{_hrms_db_password}")
    args += ["-D", _hrms_db_name, "-N", "-B", "-e", sql]
    rc, out, err = docker_exec(HRMS_DB_CONTAINER, *args)
    if rc != 0:
        raise RuntimeError(f"mysql error: {err.strip()}")
    return out.strip()


# ── BigCapital DB helpers ─────────────────────────────────────────────────────
_bc_tenant_db: str = ""


def _discover_bc_tenant_db() -> None:
    """Find BigCapital's tenant database name."""
    global _bc_tenant_db
    rc, out, err = docker_exec(
        BC_CONTAINER, "mysql",
        "-u", "bigcapital", "-pbigcapital123",
        "-N", "-B", "-e", "SHOW DATABASES LIKE 'bigcapital_tenant_%';",
    )
    if rc != 0:
        # Fall back: try the separate DB container
        rc, out, err = docker_exec(
            BC_DB_CONTAINER, "mysql",
            "-u", "bigcapital", "-pbigcapital123",
            "-N", "-B", "-e", "SHOW DATABASES LIKE 'bigcapital_tenant_%';",
        )
        if rc != 0:
            raise RuntimeError(f"Cannot list BigCapital DBs: {err.strip()}")
    dbs = [d.strip() for d in out.strip().split("\n") if d.strip()]
    if not dbs:
        raise RuntimeError("No BigCapital tenant database found")
    _bc_tenant_db = dbs[0]


def bc_query(sql: str) -> str:
    """Run a MariaDB query on BigCapital's tenant database."""
    if not _bc_tenant_db:
        _discover_bc_tenant_db()
    # Try embedded DB first (BC_CONTAINER), then separate (BC_DB_CONTAINER)
    for container in (BC_CONTAINER, BC_DB_CONTAINER):
        rc, out, err = docker_exec(
            container, "mysql",
            "-u", "bigcapital", "-pbigcapital123",
            "-D", _bc_tenant_db,
            "-N", "-B", "-e", sql,
        )
        if rc == 0:
            return out.strip()
    raise RuntimeError(f"BigCapital mysql error: {err.strip()}")


# ── Twenty DB helpers ─────────────────────────────────────────────────────────
_twenty_schema: str = ""


def twenty_psql(sql: str) -> str:
    """Run a Postgres query on the Twenty database."""
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default",
        "-t", "-A", "-c", sql,
    )
    if rc != 0:
        raise RuntimeError(f"psql error: {err.strip()}")
    return out.strip()


def get_twenty_schema() -> str:
    global _twenty_schema
    if _twenty_schema:
        return _twenty_schema
    result = twenty_psql(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'workspace_%' LIMIT 1;"
    )
    if not result:
        raise RuntimeError("No workspace schema found in Twenty DB")
    _twenty_schema = result.split("\n")[0].strip()
    return _twenty_schema


def twenty_ws(sql: str) -> str:
    """Run a query in the Twenty workspace schema."""
    schema = get_twenty_schema()
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default",
        "-t", "-A",
        "-c", f'SET search_path TO "{schema}";',
        "-c", sql,
    )
    if rc != 0:
        raise RuntimeError(f"psql error: {err.strip()}")
    # Filter out the "SET" acknowledgment line from the first -c command
    lines = out.strip().split("\n")
    filtered = [ln for ln in lines if ln.strip() != "SET"]
    return "\n".join(filtered).strip()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_hrms_claim_approved() -> None:
    """Expense claim HR-EXP-2026-00006 is Approved with correct total (10350)."""
    try:
        result = hrms_query(
            "SELECT approval_status, total_claimed_amount "
            "FROM `tabExpense Claim` WHERE name='HR-EXP-2026-00006';"
        )
        if not result:
            check("1. HRMS claim approved", 2, False, "claim not found")
            return
        parts = result.split("\t")
        status = parts[0].strip()
        amount = float(parts[1]) if len(parts) > 1 else 0.0
        ok = status == "Approved" and abs(amount - 10350.0) < 0.01
        check("1. HRMS claim approved", 2, ok, f"status={status}, total={amount}")
    except Exception as e:
        check("1. HRMS claim approved", 2, False, f"exception: {e}")


def check_2_hrms_claim_line_items() -> None:
    """3 line items: Travel 8500, Food 1500, Calls 350."""
    try:
        result = hrms_query(
            "SELECT expense_type, amount FROM `tabExpense Claim Detail` "
            "WHERE parent='HR-EXP-2026-00006' ORDER BY idx;"
        )
        if not result:
            check("2. HRMS claim line items", 2, False, "no line items")
            return
        found: dict[str, float] = {}
        for line in result.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                found[parts[0].strip()] = float(parts[1])
        expected = {"Travel": 8500.0, "Food": 1500.0, "Calls": 350.0}
        ok = len(found) == 3 and all(
            abs(found.get(k, -1) - v) < 0.01 for k, v in expected.items()
        )
        check("2. HRMS claim line items", 2, ok, f"found={found}")
    except Exception as e:
        check("2. HRMS claim line items", 2, False, f"exception: {e}")


def check_3_bc_vendor_exists() -> None:
    """Vendor 'Mohammed Farooq Reimbursement' exists with correct email."""
    try:
        result = bc_query(
            "SELECT DISPLAY_NAME, EMAIL FROM CONTACTS "
            "WHERE DISPLAY_NAME = 'Mohammed Farooq Reimbursement' "
            "AND CONTACT_SERVICE = 'vendor' LIMIT 1;"
        )
        if not result:
            # Try without service filter
            result = bc_query(
                "SELECT DISPLAY_NAME, EMAIL FROM CONTACTS "
                "WHERE DISPLAY_NAME = 'Mohammed Farooq Reimbursement' LIMIT 1;"
            )
        ok = bool(result.strip())
        if ok:
            parts = result.split("\t")
            email = parts[1].strip() if len(parts) > 1 else ""
            check("3. BC vendor exists", 1, True, f"email={email}")
        else:
            check("3. BC vendor exists", 1, False, "vendor not found")
    except Exception as e:
        check("3. BC vendor exists", 1, False, f"exception: {e}")


def check_4_bc_items_exist() -> None:
    """Items 'Travel', 'Food', 'Calls' exist."""
    try:
        result = bc_query(
            "SELECT NAME FROM ITEMS WHERE NAME IN ('Travel', 'Food', 'Calls');"
        )
        found = {r.strip() for r in result.split("\n") if r.strip()} if result else set()
        required = {"Travel", "Food", "Calls"}
        missing = required - found
        check("4. BC items exist", 1, not missing,
              f"found={found}" if not missing else f"missing={missing}")
    except Exception as e:
        check("4. BC items exist", 1, False, f"exception: {e}")


def check_5_bc_bill_exists() -> None:
    """Bill dated 2026-03-20 for vendor with total ~10350 exists."""
    try:
        result = bc_query(
            "SELECT b.ID, b.BILL_DATE, b.AMOUNT, b.STATUS, c.DISPLAY_NAME "
            "FROM BILLS b "
            "LEFT JOIN CONTACTS c ON b.VENDOR_ID = c.ID "
            "WHERE c.DISPLAY_NAME = 'Mohammed Farooq Reimbursement' "
            "AND b.BILL_DATE = '2026-03-20' "
            "LIMIT 1;"
        )
        if not result:
            # Broader search
            result = bc_query(
                "SELECT b.ID, b.BILL_DATE, b.AMOUNT, b.STATUS "
                "FROM BILLS b WHERE ABS(b.AMOUNT - 10350) < 1 LIMIT 1;"
            )
        if not result:
            check("5. BC bill exists", 2, False, "no matching bill")
            return
        parts = result.split("\t")
        bill_date = parts[1].strip() if len(parts) > 1 else ""
        amount = float(parts[2]) if len(parts) > 2 else 0.0
        status = parts[3].strip() if len(parts) > 3 else ""
        ok = "2026-03-20" in bill_date and abs(amount - 10350.0) < 1.0
        check("5. BC bill exists", 2, ok,
              f"date={bill_date}, amount={amount}, status={status}")
    except Exception as e:
        check("5. BC bill exists", 2, False, f"exception: {e}")


def _find_bc_bill_id() -> str | None:
    """Find the ID of the target bill."""
    result = bc_query(
        "SELECT b.ID FROM BILLS b "
        "LEFT JOIN CONTACTS c ON b.VENDOR_ID = c.ID "
        "WHERE c.DISPLAY_NAME = 'Mohammed Farooq Reimbursement' "
        "AND ABS(b.AMOUNT - 10350) < 1 LIMIT 1;"
    )
    if result and result.strip():
        return result.strip().split("\n")[0].split("\t")[0].strip()
    # Broader search
    result = bc_query(
        "SELECT ID FROM BILLS WHERE ABS(AMOUNT - 10350) < 1 LIMIT 1;"
    )
    if result and result.strip():
        return result.strip().split("\n")[0].strip()
    return None


def check_6_bc_bill_line_items() -> None:
    """Bill has 3 entries: Travel 8500, Food 1500, Calls 350."""
    try:
        bill_id = _find_bc_bill_id()
        if not bill_id:
            check("6. BC bill line items", 2, False, "bill not found")
            return
        result = bc_query(
            f"SELECT i.NAME, ie.RATE, ie.QUANTITY "
            f"FROM ITEMS_ENTRIES ie "
            f"LEFT JOIN ITEMS i ON ie.ITEM_ID = i.ID "
            f"WHERE ie.REFERENCE_TYPE = 'Bill' AND ie.REFERENCE_ID = '{bill_id}';"
        )
        if not result:
            check("6. BC bill line items", 2, False, "no line items found")
            return
        expected = {"Travel": 8500.0, "Food": 1500.0, "Calls": 350.0}
        found: dict[str, float] = {}
        for line in result.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 3:
                name = parts[0].strip()
                rate = float(parts[1])
                qty = float(parts[2])
                found[name] = rate * qty
        ok = all(abs(found.get(k, -1) - v) < 1.0 for k, v in expected.items())
        check("6. BC bill line items", 2, ok, f"found={found}")
    except Exception as e:
        check("6. BC bill line items", 2, False, f"exception: {e}")


def check_7_bc_payment_recorded() -> None:
    """Payment of 10350 from Bank Account dated 2026-04-05."""
    try:
        result = bc_query(
            "SELECT bp.AMOUNT, bp.PAYMENT_DATE, a.NAME AS ACCT_NAME "
            "FROM BILLS_PAYMENTS bp "
            "LEFT JOIN ACCOUNTS a ON bp.PAYMENT_ACCOUNT_ID = a.ID "
            "WHERE ABS(bp.AMOUNT - 10350) < 1 "
            "AND bp.PAYMENT_DATE = '2026-04-05' "
            "LIMIT 1;"
        )
        if not result:
            check("7. BC payment recorded", 2, False, "payment not found")
            return
        parts = result.split("\t")
        amount = float(parts[0]) if parts else 0.0
        date = parts[1].strip() if len(parts) > 1 else ""
        acct = parts[2].strip() if len(parts) > 2 else ""
        ok = abs(amount - 10350.0) < 1.0 and "2026-04-05" in date
        check("7. BC payment recorded", 2, ok,
              f"amount={amount}, date={date}, account={acct}")
    except Exception as e:
        check("7. BC payment recorded", 2, False, f"exception: {e}")


def check_8_bc_bill_fully_paid() -> None:
    """Bill is fully paid — payment_amount matches amount (zero balance)."""
    try:
        result = bc_query(
            "SELECT b.AMOUNT, b.PAYMENT_AMOUNT, b.STATUS "
            "FROM BILLS b "
            "LEFT JOIN CONTACTS c ON b.VENDOR_ID = c.ID "
            "WHERE c.DISPLAY_NAME = 'Mohammed Farooq Reimbursement' "
            "AND ABS(b.AMOUNT - 10350) < 1 LIMIT 1;"
        )
        if not result:
            result = bc_query(
                "SELECT AMOUNT, PAYMENT_AMOUNT, STATUS FROM BILLS "
                "WHERE ABS(AMOUNT - 10350) < 1 LIMIT 1;"
            )
        if not result:
            check("8. BC bill fully paid", 3, False, "bill not found")
            return
        parts = result.split("\t")
        amount = float(parts[0])
        paid = float(parts[1]) if len(parts) > 1 else 0.0
        status = parts[2].strip() if len(parts) > 2 else ""
        due = amount - paid
        ok = abs(due) < 0.01 or status.lower() == "paid"
        check("8. BC bill fully paid", 3, ok,
              f"amount={amount}, paid={paid}, due={due:.2f}, status={status}")
    except Exception as e:
        check("8. BC bill fully paid", 3, False, f"exception: {e}")


def check_9_twenty_task_exists() -> None:
    """Task titled 'Expense reimbursement processed — Mohammed Farooq' exists."""
    try:
        result = twenty_ws(
            "SELECT id, title FROM task "
            "WHERE \"deletedAt\" IS NULL "
            "AND title LIKE '%Expense reimbursement processed%Mohammed Farooq%' "
            "LIMIT 1;"
        )
        ok = bool(result.strip())
        check("9. Twenty task exists", 1, ok,
              result.strip()[:120] if ok else "task not found")
    except Exception as e:
        check("9. Twenty task exists", 1, False, f"exception: {e}")


def check_10_twenty_task_completed() -> None:
    """Task is completed (DONE) with due date 2026-04-05."""
    try:
        result = twenty_ws(
            "SELECT status, \"dueAt\"::text FROM task "
            "WHERE \"deletedAt\" IS NULL "
            "AND title LIKE '%Expense reimbursement processed%Mohammed Farooq%' "
            "LIMIT 1;"
        )
        if not result.strip():
            check("10. Twenty task completed + due date", 2, False, "task not found")
            return
        parts = result.split("|")
        status = parts[0].strip()
        due_at = parts[1].strip() if len(parts) > 1 else ""
        ok = status == "DONE" and "2026-04-05" in due_at
        check("10. Twenty task completed + due date", 2, ok,
              f"status={status}, dueAt={due_at}")
    except Exception as e:
        check("10. Twenty task completed + due date", 2, False, f"exception: {e}")


def check_11_twenty_task_body() -> None:
    """Task body contains key expense reimbursement details."""
    try:
        # Try bodyV2Markdown first, fall back to bodyV2Blocknote
        result = twenty_ws(
            "SELECT COALESCE(\"bodyV2Markdown\", \"bodyV2Blocknote\"::text, '') FROM task "
            "WHERE \"deletedAt\" IS NULL "
            "AND title LIKE '%Expense reimbursement processed%Mohammed Farooq%' "
            "LIMIT 1;"
        )
        if not result.strip():
            check("11. Twenty task body", 2, False, "task not found")
            return
        body = result.lower()
        required = ["hr-exp-2026-00006", "travel", "food", "calls", "bank account"]
        has_amount = "10,350" in body or "10350" in body
        missing = [f for f in required if f not in body]
        if not has_amount:
            missing.append("amount 10350")
        ok = not missing
        check("11. Twenty task body", 2, ok,
              "all key details present" if ok else f"missing: {missing}")
    except Exception as e:
        check("11. Twenty task body", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_hrms_claim_approved()
    check_2_hrms_claim_line_items()
    check_3_bc_vendor_exists()
    check_4_bc_items_exist()
    check_5_bc_bill_exists()
    check_6_bc_bill_line_items()
    check_7_bc_payment_recorded()
    check_8_bc_bill_fully_paid()
    check_9_twenty_task_exists()
    check_10_twenty_task_completed()
    check_11_twenty_task_body()

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
