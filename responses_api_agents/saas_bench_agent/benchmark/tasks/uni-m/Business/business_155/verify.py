# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Business/business_155/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Business-155-I1: Employee Grievance Handling Workflow

Checks: 18 weighted checks across hrms, bigcapital, pretix, twenty.
Strategy: docker exec DB queries for all sites.

Required env vars:
  SERVER_HOSTNAME,
  HRMS_PORT, HRMS_CONTAINER, HRMS_DB_CONTAINER,
  BIGCAPITAL_PORT, BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER,
  PRETIX_PORT, PRETIX_CONTAINER, PRETIX_DB_CONTAINER,
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

BIGCAPITAL_PORT = os.environ.get("BIGCAPITAL_PORT")
BIGCAPITAL_CONTAINER = os.environ.get("BIGCAPITAL_CONTAINER")
BIGCAPITAL_DB_CONTAINER = os.environ.get("BIGCAPITAL_DB_CONTAINER")

PRETIX_PORT = os.environ.get("PRETIX_PORT")
PRETIX_CONTAINER = os.environ.get("PRETIX_CONTAINER")
PRETIX_DB_CONTAINER = os.environ.get("PRETIX_DB_CONTAINER")

TWENTY_PORT = os.environ.get("TWENTY_PORT")
TWENTY_CONTAINER = os.environ.get("TWENTY_CONTAINER")
TWENTY_DB_CONTAINER = os.environ.get("TWENTY_DB_CONTAINER")

_required = {
    "HRMS_PORT": HRMS_PORT, "HRMS_CONTAINER": HRMS_CONTAINER, "HRMS_DB_CONTAINER": HRMS_DB_CONTAINER,
    "BIGCAPITAL_PORT": BIGCAPITAL_PORT, "BIGCAPITAL_CONTAINER": BIGCAPITAL_CONTAINER,
    "BIGCAPITAL_DB_CONTAINER": BIGCAPITAL_DB_CONTAINER,
    "PRETIX_PORT": PRETIX_PORT, "PRETIX_CONTAINER": PRETIX_CONTAINER, "PRETIX_DB_CONTAINER": PRETIX_DB_CONTAINER,
    "TWENTY_PORT": TWENTY_PORT, "TWENTY_CONTAINER": TWENTY_CONTAINER, "TWENTY_DB_CONTAINER": TWENTY_DB_CONTAINER,
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


_hrms_db_cache: str | None = None


def _hrms_find_db() -> str:
    """Find the Frappe site DB — the underscore-prefixed DB that contains tabDocType."""
    global _hrms_db_cache
    if _hrms_db_cache:
        return _hrms_db_cache
    rc, out, err = docker_exec(
        HRMS_DB_CONTAINER,
        "mysql", "-u", "root", "-phrms123456",
        "--default-character-set=utf8mb4", "-N", "-e",
        "SHOW DATABASES LIKE '\\_%'"
    )
    dbs = [line.strip() for line in out.strip().splitlines() if line.strip()]
    for db in dbs:
        rc2, out2, _ = docker_exec(
            HRMS_DB_CONTAINER,
            "mysql", "-u", "root", "-phrms123456",
            "--default-character-set=utf8mb4", db, "-N", "-e",
            "SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_schema='{db}' AND table_name='tabEmployee'"
        )
        if out2.strip() == "1":
            _hrms_db_cache = db
            return db
    _hrms_db_cache = dbs[0] if dbs else "_frappe_bench"
    return _hrms_db_cache


def hrms_sql(query: str) -> str:
    """Run SQL on the HRMS MariaDB. Auto-discovers the Frappe site DB."""
    db_name = _hrms_find_db()
    rc, out, err = docker_exec(
        HRMS_DB_CONTAINER,
        "mysql", "-u", "root", "-phrms123456",
        "--default-character-set=utf8mb4", db_name, "-N", "-e", query
    )
    if rc != 0:
        raise RuntimeError(f"HRMS SQL error: {err.strip()}")
    return out.strip()


_bc_db_cache: str | None = None


def _bc_find_tenant_db() -> str:
    """Find the BigCapital tenant DB name."""
    global _bc_db_cache
    if _bc_db_cache:
        return _bc_db_cache
    # BigCapital embeds MariaDB in the app container OR uses a separate DB container.
    # Try app container first, then DB container.
    for container in (BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER):
        rc, out, err = docker_exec(
            container,
            "mysql", "-u", "root", "-N", "-e",
            "SHOW DATABASES LIKE 'bigcapital_tenant_%'"
        )
        dbs = [line.strip() for line in out.strip().splitlines() if line.strip()]
        if dbs:
            _bc_db_cache = dbs[0]
            return _bc_db_cache
    _bc_db_cache = "bigcapital"
    return _bc_db_cache


def _bc_container() -> str:
    """Return the container that has a working mysql client with the tenant DB."""
    db = _bc_find_tenant_db()
    for container in (BIGCAPITAL_CONTAINER, BIGCAPITAL_DB_CONTAINER):
        rc, out, err = docker_exec(
            container,
            "mysql", "-u", "root", "-N", db, "-e", "SELECT 1"
        )
        if rc == 0:
            return container
    return BIGCAPITAL_CONTAINER


def bigcapital_sql(query: str) -> str:
    """Run SQL on BigCapital MariaDB. Auto-discovers tenant DB and container."""
    db_name = _bc_find_tenant_db()
    container = _bc_container()
    rc, out, err = docker_exec(
        container,
        "mysql", "-u", "root", "-N", db_name, "-e", query
    )
    if rc != 0:
        raise RuntimeError(f"BigCapital SQL error: {err.strip()}")
    return out.strip()


def pretix_sql(query: str) -> str:
    rc, out, err = docker_exec(
        PRETIX_DB_CONTAINER,
        "psql", "-U", "pretix", "-d", "pretix", "-t", "-A", "-c", query
    )
    if rc != 0:
        raise RuntimeError(f"Pretix SQL error: {err.strip()}")
    return out.strip()


def twenty_sql(query: str) -> str:
    """Run SQL on Twenty Postgres. Auto-discovers workspace schema."""
    find_schema = (
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'workspace_%' LIMIT 1"
    )
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c", find_schema
    )
    schema = out.strip().split("\n")[0].strip() if out.strip() else "public"
    full_query = f'SET search_path TO "{schema}"; {query}'
    rc, out, err = docker_exec(
        TWENTY_DB_CONTAINER,
        "psql", "-U", "postgres", "-d", "default", "-t", "-A", "-c", full_query
    )
    if rc != 0:
        raise RuntimeError(f"Twenty SQL error: {err.strip()}")
    # Strip the "SET" line from SET search_path output
    lines = out.strip().splitlines()
    result_lines = [l for l in lines if l.strip() and l.strip() != "SET"]
    return "\n".join(result_lines)


# ── HRMS Checks ───────────────────────────────────────────────────────────────
def check_1_grievance_types() -> None:
    """Grievance types 'Workplace Harassment' and 'Retaliation' exist."""
    try:
        out = hrms_sql(
            "SELECT name FROM `tabGrievance Type` "
            "WHERE name IN ('Workplace Harassment', 'Retaliation')"
        )
        found = set(line.strip() for line in out.splitlines() if line.strip())
        has_wh = "Workplace Harassment" in found
        has_ret = "Retaliation" in found
        check("1. Grievance types exist", 1, has_wh and has_ret,
              f"found={found}")
    except Exception as e:
        check("1. Grievance types exist", 1, False, f"exception: {e}")


def check_2_employee_grievance() -> None:
    """Employee Grievance for Pooja Malhotra against Arjun Nair, type Workplace Harassment, status Open."""
    try:
        out = hrms_sql(
            "SELECT grievance_type, subject, status, grievance_against_party, grievance_against "
            "FROM `tabEmployee Grievance` "
            "WHERE raised_by='HR-EMP-00008' "
            "LIMIT 5"
        )
        lines = [l for l in out.splitlines() if l.strip()]
        found_match = False
        detail = f"rows={len(lines)}"
        for line in lines:
            cols = line.split("\t")
            if len(cols) >= 5:
                g_type, subject, status, party_type, against = [c.strip() for c in cols]
                if (g_type == "Workplace Harassment"
                        and "hostile behavior" in subject.lower()
                        and status == "Open"
                        and "HR-EMP-00011" in against):
                    found_match = True
                    detail = f"type={g_type}, status={status}, against={against}"
                    break
        check("2. Employee Grievance record", 2, found_match, detail)
    except Exception as e:
        check("2. Employee Grievance record", 2, False, f"exception: {e}")


def check_3_employee_transfer() -> None:
    """Submitted Employee Transfer for Arjun Nair, dept change to Customer Service."""
    try:
        out = hrms_sql(
            "SELECT t.name, t.docstatus, t.transfer_date, d.property, d.current, d.new "
            "FROM `tabEmployee Transfer` t "
            "LEFT JOIN `tabEmployee Property History` d ON d.parent = t.name AND d.parenttype = 'Employee Transfer' "
            "WHERE t.employee = 'HR-EMP-00011' "
            "ORDER BY t.creation DESC LIMIT 10"
        )
        lines = [l for l in out.splitlines() if l.strip()]
        found_ok = False
        detail = f"rows={len(lines)}"
        for line in lines:
            cols = line.split("\t")
            if len(cols) >= 6:
                name, docstatus, tdate, prop, current_val, new_val = [c.strip() for c in cols[:6]]
                if (docstatus == "1"
                        and "Customer Service" in new_val
                        and "2025-07-01" in tdate):
                    found_ok = True
                    detail = f"docstatus={docstatus}, date={tdate}, new_dept={new_val}"
                    break
        check("3. Employee Transfer submitted", 2, found_ok, detail)
    except Exception as e:
        check("3. Employee Transfer submitted", 2, False, f"exception: {e}")


def check_4_training_program() -> None:
    """Training Program 'Workplace Policy Compliance 2025' exists."""
    try:
        out = hrms_sql(
            "SELECT name FROM `tabTraining Program` "
            "WHERE name = 'Workplace Policy Compliance 2025'"
        )
        found = bool(out.strip())
        check("4. Training Program exists", 1, found, f"found={'yes' if found else 'no'}")
    except Exception as e:
        check("4. Training Program exists", 1, False, f"exception: {e}")


def check_5_training_event() -> None:
    """Training Event 'Policy Awareness Workshop - Q3 2025' with 3 participants."""
    try:
        out = hrms_sql(
            "SELECT te.event_name, te.type, te.training_program, "
            "GROUP_CONCAT(tee.employee SEPARATOR ',') as employees "
            "FROM `tabTraining Event` te "
            "LEFT JOIN `tabTraining Event Employee` tee ON tee.parent = te.name "
            "WHERE te.event_name LIKE '%Policy Awareness Workshop%' "
            "GROUP BY te.name LIMIT 5"
        )
        lines = [l for l in out.splitlines() if l.strip()]
        found_ok = False
        detail = f"rows={len(lines)}"
        for line in lines:
            cols = line.split("\t")
            if len(cols) >= 4:
                ename, etype, prog, emps = [c.strip() for c in cols[:4]]
                emp_list = [e.strip() for e in emps.split(",") if e.strip()] if emps else []
                is_workshop = etype == "Workshop"
                has_3 = len(emp_list) == 3
                detail = f"type={etype}, program={prog}, participants={len(emp_list)}"
                if is_workshop and has_3:
                    found_ok = True
                    break
        check("5. Training Event with 3 participants", 2, found_ok, detail)
    except Exception as e:
        check("5. Training Event with 3 participants", 2, False, f"exception: {e}")


# ── BigCapital Checks ─────────────────────────────────────────────────────────
def check_6_expense_account() -> None:
    """Expense account 'Legal and Advisory Fees' exists."""
    try:
        out = bigcapital_sql(
            "SELECT ID, NAME, ACCOUNT_TYPE FROM ACCOUNTS "
            "WHERE NAME = 'Legal and Advisory Fees' LIMIT 1"
        )
        found = bool(out.strip())
        check("6. Expense account exists", 1, found,
              f"found={'yes' if found else 'no'}")
    except Exception as e:
        check("6. Expense account exists", 1, False, f"exception: {e}")


def check_7_investigation_expense() -> None:
    """Expense: 2800 dated 2025-07-05, reference about investigation advisory."""
    try:
        out = bigcapital_sql(
            "SELECT e.TOTAL_AMOUNT, e.PAYMENT_DATE, e.REFERENCE_NO, e.PUBLISHED_AT "
            "FROM EXPENSES_TRANSACTIONS e "
            "JOIN EXPENSE_TRANSACTION_CATEGORIES c ON c.EXPENSE_ID = e.ID "
            "JOIN ACCOUNTS a ON a.ID = c.EXPENSE_ACCOUNT_ID "
            "WHERE a.NAME = 'Legal and Advisory Fees' "
            "AND e.PAYMENT_DATE = '2025-07-05' "
            "LIMIT 5"
        )
        lines = [l for l in out.splitlines() if l.strip()]
        found_ok = False
        detail = f"rows={len(lines)}"
        for line in lines:
            cols = line.split("\t")
            if len(cols) >= 4:
                amount_str, date, ref, published = [c.strip() for c in cols[:4]]
                try:
                    amount = float(amount_str)
                except ValueError:
                    continue
                if abs(amount - 2800) < 1:
                    found_ok = True
                    detail = f"amount={amount}, date={date}, ref={ref[:50]}, published={published}"
                    break
        check("7. Investigation expense (2800)", 2, found_ok, detail)
    except Exception as e:
        check("7. Investigation expense (2800)", 2, False, f"exception: {e}")


def check_8_mediation_expense() -> None:
    """Expense: 1700 dated 2025-07-20, reference about mediation."""
    try:
        out = bigcapital_sql(
            "SELECT e.TOTAL_AMOUNT, e.PAYMENT_DATE, e.REFERENCE_NO, e.PUBLISHED_AT "
            "FROM EXPENSES_TRANSACTIONS e "
            "JOIN EXPENSE_TRANSACTION_CATEGORIES c ON c.EXPENSE_ID = e.ID "
            "JOIN ACCOUNTS a ON a.ID = c.EXPENSE_ACCOUNT_ID "
            "WHERE a.NAME = 'Legal and Advisory Fees' "
            "AND e.PAYMENT_DATE = '2025-07-20' "
            "LIMIT 5"
        )
        lines = [l for l in out.splitlines() if l.strip()]
        found_ok = False
        detail = f"rows={len(lines)}"
        for line in lines:
            cols = line.split("\t")
            if len(cols) >= 4:
                amount_str, date, ref, published = [c.strip() for c in cols[:4]]
                try:
                    amount = float(amount_str)
                except ValueError:
                    continue
                if abs(amount - 1700) < 1:
                    found_ok = True
                    detail = f"amount={amount}, date={date}, ref={ref[:50]}, published={published}"
                    break
        check("8. Mediation expense (1700)", 2, found_ok, detail)
    except Exception as e:
        check("8. Mediation expense (1700)", 2, False, f"exception: {e}")


# ── Pretix Checks ─────────────────────────────────────────────────────────────
def check_9_pretix_event() -> None:
    """Event 'Workplace Policy Compliance Workshop' with slug 'policy-compliance-workshop'."""
    try:
        out = pretix_sql(
            "SELECT slug, date_from, currency FROM pretixbase_event "
            "WHERE slug = 'policy-compliance-workshop' LIMIT 1"
        )
        found = bool(out.strip())
        detail = out.strip() if found else "event not found"
        check("9. Pretix event exists", 1, found, detail)
    except Exception as e:
        check("9. Pretix event exists", 1, False, f"exception: {e}")


def check_10_pretix_live() -> None:
    """Event is set to live."""
    try:
        out = pretix_sql(
            "SELECT live FROM pretixbase_event "
            "WHERE slug = 'policy-compliance-workshop' LIMIT 1"
        )
        is_live = out.strip().lower() in ("t", "true", "1")
        check("10. Pretix event is live", 1, is_live, f"live={out.strip()}")
    except Exception as e:
        check("10. Pretix event is live", 1, False, f"exception: {e}")


def check_11_pretix_product() -> None:
    """Product 'Policy Training Admission' priced at 0."""
    try:
        out = pretix_sql(
            "SELECT i.default_price FROM pretixbase_item i "
            "JOIN pretixbase_event e ON e.id = i.event_id "
            "WHERE e.slug = 'policy-compliance-workshop' "
            "AND i.name::text LIKE '%Policy Training Admission%' "
            "LIMIT 1"
        )
        price_str = out.strip()
        try:
            price = float(price_str)
            is_free = abs(price) < 0.01
        except (ValueError, TypeError):
            is_free = False
        check("11. Product priced at 0", 1, is_free, f"price={price_str}")
    except Exception as e:
        check("11. Product priced at 0", 1, False, f"exception: {e}")


def check_12_pretix_quota() -> None:
    """Quota 'Training Capacity' size 50."""
    try:
        out = pretix_sql(
            "SELECT q.name, q.size FROM pretixbase_quota q "
            "JOIN pretixbase_event e ON e.id = q.event_id "
            "WHERE e.slug = 'policy-compliance-workshop' "
            "AND q.name LIKE '%Training Capacity%' LIMIT 1"
        )
        parts = out.strip().split("|")
        if len(parts) >= 2:
            size = int(parts[1].strip())
            ok = size == 50
            detail = f"size={size}"
        else:
            ok = False
            detail = f"raw={out.strip()}"
        check("12. Quota size 50", 1, ok, detail)
    except Exception as e:
        check("12. Quota size 50", 1, False, f"exception: {e}")


def check_13_pretix_questions() -> None:
    """Two custom questions: 'Employee ID' (text, required) and 'Department' (choice, required)."""
    try:
        out = pretix_sql(
            "SELECT q.question, q.type, q.required "
            "FROM pretixbase_question q "
            "JOIN pretixbase_event e ON e.id = q.event_id "
            "WHERE e.slug = 'policy-compliance-workshop'"
        )
        lines = [l for l in out.splitlines() if l.strip()]
        has_empid = False
        has_dept = False
        for line in lines:
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 3:
                qtext, qtype, required = cols[0], cols[1], cols[2]
                if "Employee ID" in qtext and qtype in ("S", "T") and required in ("t", "True", "1"):
                    has_empid = True
                if "Department" in qtext and qtype in ("C",) and required in ("t", "True", "1"):
                    has_dept = True
        check("13. Custom questions (Employee ID + Department)", 2,
              has_empid and has_dept,
              f"employee_id={'yes' if has_empid else 'no'}, department={'yes' if has_dept else 'no'}")
    except Exception as e:
        check("13. Custom questions (Employee ID + Department)", 2, False, f"exception: {e}")


def check_14_pretix_checkin() -> None:
    """Check-in list 'Training Attendance Check-in' exists."""
    try:
        out = pretix_sql(
            "SELECT cl.name FROM pretixbase_checkinlist cl "
            "JOIN pretixbase_event e ON e.id = cl.event_id "
            "WHERE e.slug = 'policy-compliance-workshop' "
            "AND cl.name LIKE '%Training Attendance Check-in%' LIMIT 1"
        )
        found = bool(out.strip())
        check("14. Check-in list exists", 1, found, f"found={'yes' if found else 'no'}")
    except Exception as e:
        check("14. Check-in list exists", 1, False, f"exception: {e}")


# ── Twenty CRM Checks ────────────────────────────────────────────────────────
def check_15_twenty_task_investigation() -> None:
    """Task: 'CONFIDENTIAL: Investigate grievance - Repeated hostile behavior in team meetings'."""
    try:
        out = twenty_sql(
            "SELECT title, \"dueAt\" FROM task "
            "WHERE title LIKE '%Investigate grievance%hostile behavior%' LIMIT 1"
        )
        if out.strip():
            parts = out.strip().split("|")
            title = parts[0].strip() if parts else ""
            due = parts[1].strip() if len(parts) > 1 else ""
            has_date = "2025-07-12" in due
            check("15. Twenty task: investigation", 2, has_date,
                  f"title={title[:60]}, due={due}")
        else:
            check("15. Twenty task: investigation", 2, False, "task not found")
    except Exception as e:
        check("15. Twenty task: investigation", 2, False, f"exception: {e}")


def check_16_twenty_task_mediation() -> None:
    """Task: 'CONFIDENTIAL: Mediation session - Pooja Malhotra and Arjun Nair'."""
    try:
        out = twenty_sql(
            "SELECT title, \"dueAt\" FROM task "
            "WHERE title LIKE '%Mediation session%Pooja Malhotra%' LIMIT 1"
        )
        if out.strip():
            parts = out.strip().split("|")
            title = parts[0].strip() if parts else ""
            due = parts[1].strip() if len(parts) > 1 else ""
            has_date = "2025-07-25" in due
            check("16. Twenty task: mediation", 2, has_date,
                  f"title={title[:60]}, due={due}")
        else:
            check("16. Twenty task: mediation", 2, False, "task not found")
    except Exception as e:
        check("16. Twenty task: mediation", 2, False, f"exception: {e}")


def check_17_twenty_task_training() -> None:
    """Task: 'Mandatory compliance training - Workplace Policy Compliance Workshop'."""
    try:
        out = twenty_sql(
            "SELECT title, \"dueAt\" FROM task "
            "WHERE title LIKE '%Mandatory compliance training%' LIMIT 1"
        )
        if out.strip():
            parts = out.strip().split("|")
            title = parts[0].strip() if parts else ""
            due = parts[1].strip() if len(parts) > 1 else ""
            has_date = "2025-07-15" in due
            check("17. Twenty task: compliance training", 2, has_date,
                  f"title={title[:60]}, due={due}")
        else:
            check("17. Twenty task: compliance training", 2, False, "task not found")
    except Exception as e:
        check("17. Twenty task: compliance training", 2, False, f"exception: {e}")


def check_18_twenty_note() -> None:
    """Note: 'Grievance Resolution Log - Repeated hostile behavior in team meetings - 2025-07-12'."""
    try:
        out = twenty_sql(
            "SELECT title, \"bodyV2Markdown\" FROM note "
            "WHERE title LIKE '%Grievance Resolution Log%hostile behavior%' LIMIT 1"
        )
        if out.strip():
            parts = out.strip().split("|", 1)
            title = parts[0].strip() if parts else ""
            body = parts[1].strip() if len(parts) > 1 else ""
            has_content = "Pooja Malhotra" in body and "Arjun Nair" in body
            check("18. Twenty note: grievance resolution log", 2, has_content,
                  f"title={title[:60]}, body_len={len(body)}")
        else:
            check("18. Twenty note: grievance resolution log", 2, False, "note not found")
    except Exception as e:
        check("18. Twenty note: grievance resolution log", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_grievance_types()
    check_2_employee_grievance()
    check_3_employee_transfer()
    check_4_training_program()
    check_5_training_event()
    check_6_expense_account()
    check_7_investigation_expense()
    check_8_mediation_expense()
    check_9_pretix_event()
    check_10_pretix_live()
    check_11_pretix_product()
    check_12_pretix_quota()
    check_13_pretix_questions()
    check_14_pretix_checkin()
    check_15_twenty_task_investigation()
    check_16_twenty_task_mediation()
    check_17_twenty_task_training()
    check_18_twenty_note()

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
