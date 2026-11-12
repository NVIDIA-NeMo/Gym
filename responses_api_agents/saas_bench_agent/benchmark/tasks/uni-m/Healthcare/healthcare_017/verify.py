# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_017/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-017-I5: Merge Duplicate Patient Records for Latoyia Kertzmann
and Compile Audit Documentation

Checks: 10 weighted checks across openemr, onlyoffice.
Strategy: docker exec (MariaDB for OpenEMR, MySQL for OnlyOffice)

Required env vars:
  SERVER_HOSTNAME,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENEMR_PORT = os.environ.get("OPENEMR_PORT")
OPENEMR_CONTAINER = os.environ.get("OPENEMR_CONTAINER")
OPENEMR_DB_CONTAINER = os.environ.get("OPENEMR_DB_CONTAINER")

ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.environ.get("ONLYOFFICE_DB_CONTAINER")

for var_name, var_val in [
    ("OPENEMR_PORT", OPENEMR_PORT),
    ("OPENEMR_CONTAINER", OPENEMR_CONTAINER),
    ("OPENEMR_DB_CONTAINER", OPENEMR_DB_CONTAINER),
    ("ONLYOFFICE_PORT", ONLYOFFICE_PORT),
    ("ONLYOFFICE_CONTAINER", ONLYOFFICE_CONTAINER),
    ("ONLYOFFICE_DB_CONTAINER", ONLYOFFICE_DB_CONTAINER),
]:
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
def openemr_sql(query: str, timeout: int = 15) -> str:
    """Run a SQL query against OpenEMR MariaDB and return stdout."""
    r = subprocess.run(
        [
            "docker", "exec", OPENEMR_DB_CONTAINER,
            "mysql", "--default-character-set=utf8mb4",
            "-u", "openemr", "-popenemr_pass", "-D", "openemr",
            "-sN", "-e", query,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout.strip()


def onlyoffice_sql(query: str, timeout: int = 15) -> str:
    """Run a SQL query against OnlyOffice MySQL and return stdout."""
    r = subprocess.run(
        [
            "docker", "exec", ONLYOFFICE_DB_CONTAINER,
            "mysql", "--default-character-set=utf8mb4",
            "-u", "onlyoffice_user", "-ponlyoffice_pass",
            "-D", "onlyoffice",
            "-sN", "-e", query,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout.strip()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_target_patient_exists() -> None:
    """Verify target patient Latoyia Kertzmann (pid 158) exists in patient_data."""
    try:
        row = openemr_sql(
            "SELECT fname, lname FROM patient_data WHERE pid = 158;"
        )
        if row:
            parts = row.split("\t")
            fname = parts[0].strip() if len(parts) > 0 else ""
            lname = parts[1].strip() if len(parts) > 1 else ""
            passed = "latoyia" in fname.lower() and "kertzmann" in lname.lower()
            check("1. Target patient pid 158 exists", 1, passed,
                  f"found: {fname} {lname}")
        else:
            check("1. Target patient pid 158 exists", 1, False,
                  "no patient_data row for pid 158")
    except Exception as e:
        check("1. Target patient pid 158 exists", 1, False, f"exception: {e}")


def check_2_source_patient_merged() -> None:
    """Verify source patient pid 249 has been merged (deleted or inactive)."""
    try:
        row = openemr_sql(
            "SELECT COUNT(*) FROM patient_data WHERE pid = 249;"
        )
        count = int(row.strip()) if row.strip() else -1
        # After merge, source patient should be deleted (count=0)
        # or if still present, check that lists have been transferred
        if count == 0:
            check("2. Source patient pid 249 merged away", 2, True,
                  "pid 249 no longer in patient_data")
        else:
            # Check if lists were at least transferred (no lists left on 249)
            list_count = openemr_sql(
                "SELECT COUNT(*) FROM lists WHERE pid = 249;"
            )
            lc = int(list_count.strip()) if list_count.strip() else -1
            if lc == 0:
                check("2. Source patient pid 249 merged away", 2, True,
                      "pid 249 still in patient_data but lists transferred (0 list entries)")
            else:
                check("2. Source patient pid 249 merged away", 2, False,
                      f"pid 249 still has {lc} list entries; merge may not have occurred")
    except Exception as e:
        check("2. Source patient pid 249 merged away", 2, False, f"exception: {e}")


def check_3_merged_has_problems() -> None:
    """Verify merged patient pid 158 has medical_problem entries in lists."""
    try:
        row = openemr_sql(
            "SELECT COUNT(*) FROM lists WHERE pid = 158 AND type = 'medical_problem';"
        )
        count = int(row.strip()) if row.strip() else 0
        check("3. Merged record has medical problems", 2, count > 0,
              f"found {count} medical_problem entries for pid 158")
    except Exception as e:
        check("3. Merged record has medical problems", 2, False, f"exception: {e}")


def check_4_merged_has_medications() -> None:
    """Verify merged patient pid 158 has medication entries in lists."""
    try:
        row = openemr_sql(
            "SELECT COUNT(*) FROM lists WHERE pid = 158 AND type = 'medication';"
        )
        count = int(row.strip()) if row.strip() else 0
        check("4. Merged record has medications", 2, count > 0,
              f"found {count} medication entries for pid 158")
    except Exception as e:
        check("4. Merged record has medications", 2, False, f"exception: {e}")


def check_5_merged_has_allergies() -> None:
    """Verify merged patient pid 158 has allergy entries in lists."""
    try:
        row = openemr_sql(
            "SELECT COUNT(*) FROM lists WHERE pid = 158 AND type = 'allergy';"
        )
        count = int(row.strip()) if row.strip() else 0
        check("5. Merged record has allergies", 2, count > 0,
              f"found {count} allergy entries for pid 158")
    except Exception as e:
        check("5. Merged record has allergies", 2, False, f"exception: {e}")


def check_6_merge_log_entry() -> None:
    """Verify system log contains a merge event entry."""
    try:
        # Search for merge-related events in the log table
        row = openemr_sql(
            "SELECT COUNT(*) FROM log WHERE event LIKE '%merge%' "
            "OR comments LIKE '%merge%' OR user_notes LIKE '%merge%';"
        )
        count = int(row.strip()) if row.strip() else 0
        if count > 0:
            check("6. Merge event in system log", 2, True,
                  f"found {count} merge-related log entries")
        else:
            # Also check base64-encoded comments for merge references
            row2 = openemr_sql(
                "SELECT COUNT(*) FROM log WHERE patient_id IN (158, 249);"
            )
            count2 = int(row2.strip()) if row2.strip() else 0
            check("6. Merge event in system log", 2, count2 > 0,
                  f"no 'merge' keyword in log; {count2} log entries for pid 158/249")
    except Exception as e:
        check("6. Merge event in system log", 2, False, f"exception: {e}")


def check_7_address_book_entry_exists() -> None:
    """Verify Address Book contains an entry for Dr. Yusuf Abdelrahman."""
    try:
        # Check users table (main address book in OpenEMR)
        row = openemr_sql(
            "SELECT id, fname, lname FROM users "
            "WHERE lname LIKE '%Abdelrahman%' AND fname LIKE '%Yusuf%' LIMIT 1;"
        )
        if row:
            check("7. Address book entry for Dr. Yusuf Abdelrahman", 1, True,
                  f"found in users table: {row}")
            return

        # Fallback: check misc_address_book
        row2 = openemr_sql(
            "SELECT id, fname, lname FROM misc_address_book "
            "WHERE lname LIKE '%Abdelrahman%' AND fname LIKE '%Yusuf%' LIMIT 1;"
        )
        if row2:
            check("7. Address book entry for Dr. Yusuf Abdelrahman", 1, True,
                  f"found in misc_address_book: {row2}")
            return

        check("7. Address book entry for Dr. Yusuf Abdelrahman", 1, False,
              "not found in users or misc_address_book")
    except Exception as e:
        check("7. Address book entry for Dr. Yusuf Abdelrahman", 1, False,
              f"exception: {e}")


def check_8_address_book_details() -> None:
    """Verify address book entry has correct specialty, phone, and address."""
    try:
        # Check users table first
        row = openemr_sql(
            "SELECT specialty, phonew1, street, city, state, zip FROM users "
            "WHERE lname LIKE '%Abdelrahman%' AND fname LIKE '%Yusuf%' LIMIT 1;"
        )
        if row:
            parts = row.split("\t")
            specialty = parts[0].strip() if len(parts) > 0 else ""
            phone = parts[1].strip() if len(parts) > 1 else ""
            street = parts[2].strip() if len(parts) > 2 else ""
            city = parts[3].strip() if len(parts) > 3 else ""
            state = parts[4].strip() if len(parts) > 4 else ""
            zipcode = parts[5].strip() if len(parts) > 5 else ""

            errors = []
            if "gastroenterology" not in specialty.lower():
                errors.append(f"specialty='{specialty}' expected 'Gastroenterology'")
            if "339-555-0617" not in phone and "3395550617" not in phone.replace("-", "").replace(" ", ""):
                errors.append(f"phone='{phone}' expected '339-555-0617'")
            # Check address components
            addr_combined = f"{street} {city} {state} {zipcode}".lower()
            if "centre" not in addr_combined and "center" not in addr_combined:
                errors.append(f"address missing 'Centre Street'")
            if "jamaica" not in addr_combined:
                errors.append(f"address missing 'Jamaica Plain'")

            passed = len(errors) == 0
            detail = "; ".join(errors) if errors else "all fields correct"
            check("8. Address book entry details correct", 2, passed, detail)
            return

        # Fallback: check misc_address_book
        row2 = openemr_sql(
            "SELECT phone, street, city, state, zip FROM misc_address_book "
            "WHERE lname LIKE '%Abdelrahman%' AND fname LIKE '%Yusuf%' LIMIT 1;"
        )
        if row2:
            parts = row2.split("\t")
            phone = parts[0].strip() if len(parts) > 0 else ""
            street = parts[1].strip() if len(parts) > 1 else ""
            city = parts[2].strip() if len(parts) > 2 else ""
            state = parts[3].strip() if len(parts) > 3 else ""
            zipcode = parts[4].strip() if len(parts) > 4 else ""

            errors = []
            if "339-555-0617" not in phone and "3395550617" not in phone.replace("-", "").replace(" ", ""):
                errors.append(f"phone='{phone}' expected '339-555-0617'")
            addr_combined = f"{street} {city} {state} {zipcode}".lower()
            if "centre" not in addr_combined and "center" not in addr_combined:
                errors.append(f"address missing 'Centre Street'")
            if "jamaica" not in addr_combined:
                errors.append(f"address missing 'Jamaica Plain'")
            # misc_address_book may not have specialty column
            passed = len(errors) == 0
            detail = "; ".join(errors) if errors else "phone/address correct (specialty not in misc_address_book)"
            check("8. Address book entry details correct", 2, passed, detail)
            return

        check("8. Address book entry details correct", 2, False,
              "entry not found in users or misc_address_book")
    except Exception as e:
        check("8. Address book entry details correct", 2, False, f"exception: {e}")


def check_9_audit_report_document() -> None:
    """Verify OnlyOffice has document 'Kertzmann Duplicate Patient Merge Audit Report - 2026-03-22'."""
    try:
        row = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Kertzmann%Duplicate%Merge%Audit%Report%2026-03-22%' "
            "AND current_version = 1 LIMIT 5;"
        )
        if row:
            check("9. Audit report document exists in OnlyOffice", 3, True,
                  f"found: {row.splitlines()[0]}")
        else:
            # Broader search
            row2 = onlyoffice_sql(
                "SELECT id, title FROM files_file "
                "WHERE title LIKE '%Kertzmann%Merge%' "
                "AND current_version = 1 LIMIT 5;"
            )
            if row2:
                check("9. Audit report document exists in OnlyOffice", 3, False,
                      f"partial match: {row2.splitlines()[0]}; expected title containing 'Kertzmann Duplicate Patient Merge Audit Report - 2026-03-22'")
            else:
                check("9. Audit report document exists in OnlyOffice", 3, False,
                      "no document matching 'Kertzmann...Merge...Audit Report...2026-03-22' found")
    except Exception as e:
        check("9. Audit report document exists in OnlyOffice", 3, False,
              f"exception: {e}")


def check_10_tracking_spreadsheet() -> None:
    """Verify OnlyOffice has spreadsheet 'Duplicate Patient Record Merge Audit Tracker - March 2026'."""
    try:
        row = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Duplicate%Patient%Merge%Audit%Tracker%March%2026%' "
            "AND current_version = 1 LIMIT 5;"
        )
        if row:
            check("10. Tracking spreadsheet exists in OnlyOffice", 2, True,
                  f"found: {row.splitlines()[0]}")
        else:
            # Broader search
            row2 = onlyoffice_sql(
                "SELECT id, title FROM files_file "
                "WHERE title LIKE '%Merge%Audit%Tracker%' "
                "AND current_version = 1 LIMIT 5;"
            )
            if row2:
                check("10. Tracking spreadsheet exists in OnlyOffice", 2, False,
                      f"partial match: {row2.splitlines()[0]}; expected title containing 'Duplicate Patient Record Merge Audit Tracker - March 2026'")
            else:
                check("10. Tracking spreadsheet exists in OnlyOffice", 2, False,
                      "no spreadsheet matching 'Duplicate Patient...Merge Audit Tracker...March 2026' found")
    except Exception as e:
        check("10. Tracking spreadsheet exists in OnlyOffice", 2, False,
              f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_target_patient_exists()
    check_2_source_patient_merged()
    check_3_merged_has_problems()
    check_4_merged_has_medications()
    check_5_merged_has_allergies()
    check_6_merge_log_entry()
    check_7_address_book_entry_exists()
    check_8_address_book_details()
    check_9_audit_report_document()
    check_10_tracking_spreadsheet()

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
