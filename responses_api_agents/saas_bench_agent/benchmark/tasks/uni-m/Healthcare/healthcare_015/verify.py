# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_015/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-015-I3: Care Transition Handover for Mora Ernser to Orthopedics

Checks: 14 weighted checks across openemr and onlyoffice.
Strategy: docker exec (MariaDB for OpenEMR, MySQL for OnlyOffice)

Required env vars:
  SERVER_HOSTNAME, OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENEMR_PORT = os.environ.get("OPENEMR_PORT")
OPENEMR_CONTAINER = os.environ.get("OPENEMR_CONTAINER")
OPENEMR_DB = os.environ.get("OPENEMR_DB_CONTAINER")
ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB = os.environ.get("ONLYOFFICE_DB_CONTAINER")

for _name, _val in [
    ("OPENEMR_PORT", OPENEMR_PORT),
    ("OPENEMR_CONTAINER", OPENEMR_CONTAINER),
    ("OPENEMR_DB_CONTAINER", OPENEMR_DB),
    ("ONLYOFFICE_PORT", ONLYOFFICE_PORT),
    ("ONLYOFFICE_CONTAINER", ONLYOFFICE_CONTAINER),
    ("ONLYOFFICE_DB_CONTAINER", ONLYOFFICE_DB),
]:
    if not _val:
        print(f"FATAL: {_name} not set", file=sys.stderr)
        sys.exit(1)

# ── Slot values ───────────────────────────────────────────────────────────────
PATIENT_FNAME = "Mora"
PATIENT_LNAME = "Ernser"
FUNCTIONAL_OBS = "Non-weight-bearing on right lower extremity, uses rolling walker, requires assistance with lower-body ADLs"
COGNITIVE_OBS = "Alert and oriented x4, MMSE 29/30, intact judgment and reasoning"
TREATMENT_GOALS = "Surgical evaluation for right hip osteoarthritis, pre-operative optimization, post-operative rehabilitation planning, and restoration of ambulatory independence"
TREATMENT_TARGET = "2026-11-30"
TRANSFER_REASON_SNIPPET = "progressive functional decline despite conservative management"
TRANSFER_FACILITY = "Brigham and Women's Hospital - Orthopedic Surgery Department"
ICD10_1 = "M16.11"
ICD10_2 = "Z47.1"
DISCLOSURE_RECIPIENT = "Dr. Gertrud Kuhic"
MSG_SUBJECT_KEY = "Care Transition Handover"
MSG_BODY_KEYS = [
    "transferring the care",
    "orthopedic surgery service",
    "total hip arthroplasty",
    "care transition summary document",
]
DOC_TITLE_KEY = "Care Transition Summary"
DOC_ID = "CTS-2026-0418-ERNSER-003"

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


def openemr_sql(query: str, timeout: int = 15) -> str:
    """Run SQL against OpenEMR MariaDB. Returns stdout."""
    rc, out, err = docker_exec(
        OPENEMR_DB,
        "mysql", "-u", "openemr", "-popenemr_pass",
        "--default-character-set=utf8mb4",
        "openemr", "-N", "-e", query,
        timeout=timeout,
    )
    return out


def onlyoffice_sql(query: str, timeout: int = 15) -> str:
    """Run SQL against OnlyOffice MySQL. Returns stdout."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB,
        "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "--default-character-set=utf8mb4",
        "onlyoffice", "-N", "-e", query,
        timeout=timeout,
    )
    return out


# ── Shared state ──────────────────────────────────────────────────────────────
_pid: str = ""
_enc: str = ""


# ── Individual checks ─────────────────────────────────────────────────────────

def check_01_patient_exists() -> None:
    """Patient Mora Ernser exists in patient_data."""
    global _pid
    try:
        out = openemr_sql(
            f"SELECT pid FROM patient_data "
            f"WHERE fname='{PATIENT_FNAME}' AND lname='{PATIENT_LNAME}' LIMIT 1;"
        )
        pid = out.strip().split("\n")[0].strip() if out.strip() else ""
        _pid = pid
        check("1. Patient Mora Ernser exists", 1, bool(pid), f"pid={pid}" if pid else "not found")
    except Exception as e:
        check("1. Patient Mora Ernser exists", 1, False, f"exception: {e}")


def check_02_new_encounter() -> None:
    """A new encounter exists for Mora Ernser (most recent)."""
    global _enc
    if not _pid:
        check("2. Transition encounter exists", 2, False, "no patient pid")
        return
    try:
        out = openemr_sql(
            f"SELECT encounter FROM form_encounter "
            f"WHERE pid={_pid} ORDER BY date DESC LIMIT 1;"
        )
        enc = out.strip().split("\n")[0].strip() if out.strip() else ""
        _enc = enc
        check("2. Transition encounter exists", 2, bool(enc), f"encounter={enc}" if enc else "none found")
    except Exception as e:
        check("2. Transition encounter exists", 2, False, f"exception: {e}")


def check_03_functional_status() -> None:
    """Functional status observation recorded in form_functional_cognitive_status."""
    if not _pid:
        check("3. Functional status observation", 2, False, "no patient pid")
        return
    try:
        enc_clause = f" AND encounter='{_enc}'" if _enc else ""
        out = openemr_sql(
            f"SELECT description FROM form_functional_cognitive_status "
            f"WHERE pid={_pid}{enc_clause};"
        )
        text = out.lower()
        found = "non-weight-bearing" in text and "rolling walker" in text
        check("3. Functional status observation", 2, found,
              "matches" if found else "key phrases not found in description")
    except Exception as e:
        check("3. Functional status observation", 2, False, f"exception: {e}")


def check_04_cognitive_status() -> None:
    """Cognitive status observation recorded in form_functional_cognitive_status."""
    if not _pid:
        check("4. Cognitive status observation", 2, False, "no patient pid")
        return
    try:
        enc_clause = f" AND encounter='{_enc}'" if _enc else ""
        out = openemr_sql(
            f"SELECT description FROM form_functional_cognitive_status "
            f"WHERE pid={_pid}{enc_clause};"
        )
        text = out.lower()
        found = "alert and oriented" in text and "mmse 29/30" in text
        check("4. Cognitive status observation", 2, found,
              "matches" if found else "key phrases not found in description")
    except Exception as e:
        check("4. Cognitive status observation", 2, False, f"exception: {e}")


def check_05_treatment_plan_goals() -> None:
    """Treatment Plan has correct goals in form_care_plan.description."""
    if not _pid:
        check("5. Treatment Plan goals", 2, False, "no patient pid")
        return
    try:
        enc_clause = f" AND encounter='{_enc}'" if _enc else ""
        out = openemr_sql(
            f"SELECT description FROM form_care_plan "
            f"WHERE pid={_pid}{enc_clause};"
        )
        text = out.lower()
        found = "surgical evaluation" in text and "pre-operative optimization" in text
        check("5. Treatment Plan goals", 2, found,
              "goals match" if found else "goal keywords not found")
    except Exception as e:
        check("5. Treatment Plan goals", 2, False, f"exception: {e}")


def check_06_treatment_plan_target() -> None:
    """Treatment Plan target date is 2026-11-30 (proposed_date or date_end or description)."""
    if not _pid:
        check("6. Treatment Plan target date", 1, False, "no patient pid")
        return
    try:
        enc_clause = f" AND encounter='{_enc}'" if _enc else ""
        out = openemr_sql(
            f"SELECT proposed_date, date_end, description FROM form_care_plan "
            f"WHERE pid={_pid}{enc_clause};"
        )
        found = "2026-11-30" in out
        check("6. Treatment Plan target date", 1, found,
              "date found" if found else "2026-11-30 not found in care plan")
    except Exception as e:
        check("6. Treatment Plan target date", 1, False, f"exception: {e}")


def _find_transfer_form_text() -> str:
    """Try to find transfer summary form content via forms table + form table or LBF."""
    if not _pid:
        return ""
    enc_clause = f" AND encounter={_enc}" if _enc else ""
    # Find forms with "transfer" in name or formdir
    rows = openemr_sql(
        f"SELECT form_id, formdir FROM forms "
        f"WHERE pid={_pid}{enc_clause} AND deleted=0 "
        f"AND (formdir LIKE '%transfer%' OR form_name LIKE '%Transfer%');"
    ).strip()
    if not rows:
        return ""
    first = rows.split("\n")[0]
    parts = first.split("\t")
    form_id = parts[0].strip()
    formdir = parts[1].strip() if len(parts) > 1 else ""
    # Try the form-specific table
    table = f"form_{formdir}" if formdir else "form_transfer_summary"
    try:
        out = openemr_sql(f"SELECT * FROM `{table}` WHERE id={form_id};")
        if out.strip():
            return out
    except Exception:
        pass
    # Fallback: LBF data
    try:
        out = openemr_sql(
            f"SELECT field_id, field_value FROM lbf_data WHERE form_id={form_id};"
        )
        if out.strip():
            return out
    except Exception:
        pass
    return ""


def check_07_transfer_summary_reason() -> None:
    """Transfer Summary contains the transfer reason."""
    try:
        text = _find_transfer_form_text().lower()
        found = "progressive functional decline" in text and "conservative management" in text
        check("7. Transfer Summary reason", 2, found,
              "reason matches" if found else "reason text not found")
    except Exception as e:
        check("7. Transfer Summary reason", 2, False, f"exception: {e}")


def check_08_transfer_summary_facility() -> None:
    """Transfer Summary contains receiving facility."""
    try:
        text = _find_transfer_form_text().lower()
        found = "brigham and women" in text and "orthopedic surgery" in text
        check("8. Transfer Summary facility", 1, found,
              "facility found" if found else "facility text not found")
    except Exception as e:
        check("8. Transfer Summary facility", 1, False, f"exception: {e}")


def check_09_icd10_codes() -> None:
    """ICD-10 codes M16.11 and Z47.1 exist in billing for the encounter."""
    if not _pid:
        check("9. ICD-10 codes M16.11 and Z47.1", 2, False, "no patient pid")
        return
    try:
        enc_clause = f" AND encounter={_enc}" if _enc else ""
        out = openemr_sql(
            f"SELECT code FROM billing "
            f"WHERE pid={_pid}{enc_clause} AND activity=1 "
            f"AND code IN ('{ICD10_1}','{ICD10_2}');"
        )
        codes = {line.strip() for line in out.strip().split("\n") if line.strip()}
        has_both = ICD10_1 in codes and ICD10_2 in codes
        check("9. ICD-10 codes M16.11 and Z47.1", 2, has_both,
              f"found: {codes}" if codes else "no matching codes in billing")
    except Exception as e:
        check("9. ICD-10 codes M16.11 and Z47.1", 2, False, f"exception: {e}")


def check_10_disclosure_record() -> None:
    """Disclosure record exists with recipient Dr. Gertrud Kuhic and Care Transition purpose."""
    if not _pid:
        check("10. Disclosure record", 2, False, "no patient pid")
        return
    try:
        out = openemr_sql(
            f"SELECT recipient, description, event FROM extended_log "
            f"WHERE patient_id='{_pid}' AND event='disclosure';"
        )
        text = out.lower()
        has_recipient = "kuhic" in text
        has_purpose = "care transition" in text
        passed = has_recipient and has_purpose
        detail_parts = []
        if not has_recipient:
            detail_parts.append("recipient missing")
        if not has_purpose:
            detail_parts.append("purpose missing")
        check("10. Disclosure record", 2, passed,
              "matches" if passed else "; ".join(detail_parts))
    except Exception as e:
        check("10. Disclosure record", 2, False, f"exception: {e}")


def check_11_message_subject() -> None:
    """Message to Dr. Gertrud Kuhic has correct subject containing 'Care Transition Handover'."""
    if not _pid:
        check("11. Message subject and recipient", 3, False, "no patient pid")
        return
    try:
        out = openemr_sql(
            f"SELECT title, assigned_to FROM pnotes "
            f"WHERE pid={_pid} AND deleted=0 ORDER BY id DESC;"
        )
        text = out.lower()
        has_subject = "care transition handover" in text and "orthopedic surgery referral" in text
        has_recipient = "kuhic" in text
        passed = has_subject and has_recipient
        detail_parts = []
        if not has_subject:
            detail_parts.append("subject mismatch")
        if not has_recipient:
            detail_parts.append("recipient not found")
        check("11. Message subject and recipient", 3, passed,
              "matches" if passed else "; ".join(detail_parts))
    except Exception as e:
        check("11. Message subject and recipient", 3, False, f"exception: {e}")


def check_12_message_body() -> None:
    """Message body contains expected key phrases about the care transition."""
    if not _pid:
        check("12. Message body content", 2, False, "no patient pid")
        return
    try:
        out = openemr_sql(
            f"SELECT body FROM pnotes "
            f"WHERE pid={_pid} AND deleted=0 ORDER BY id DESC LIMIT 5;"
        )
        body = out.lower()
        matches = sum(1 for kw in MSG_BODY_KEYS if kw.lower() in body)
        passed = matches >= 3
        check("12. Message body content", 2, passed,
              f"matched {matches}/{len(MSG_BODY_KEYS)} key phrases")
    except Exception as e:
        check("12. Message body content", 2, False, f"exception: {e}")


def check_13_onlyoffice_document() -> None:
    """OnlyOffice document with title matching 'Care Transition Summary ... Mora Ernser' exists."""
    try:
        out = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Care Transition Summary%Mora Ernser%';"
        )
        rows = out.strip()
        if rows:
            check("13. OnlyOffice document exists", 2, True, f"found: {rows[:100]}")
        else:
            # broader search for debugging
            out2 = onlyoffice_sql(
                "SELECT id, title FROM files_file "
                "WHERE title LIKE '%Care Transition%' OR title LIKE '%Mora%';"
            )
            detail = f"similar: {out2.strip()[:100]}" if out2.strip() else "no matching files"
            check("13. OnlyOffice document exists", 2, False, detail)
    except Exception as e:
        check("13. OnlyOffice document exists", 2, False, f"exception: {e}")


def check_14_onlyoffice_doc_content() -> None:
    """OnlyOffice document contains document ID CTS-2026-0418-ERNSER-003."""
    try:
        # Find document files on disk and grep for the document ID
        rc, out, _ = docker_exec(
            ONLYOFFICE_CONTAINER,
            "bash", "-c",
            "find /var/www/onlyoffice/Data/ -type f "
            "\\( -name '*.docx' -o -name '*.docxf' -o -name '*.odt' -o -name '*.txt' \\) "
            "2>/dev/null | head -30",
            timeout=15,
        )
        files = [f.strip() for f in out.strip().split("\n") if f.strip()]
        if not files:
            check("14. Document contains ID CTS-2026-0418-ERNSER-003", 1, False,
                  "no document files found on disk")
            return

        for fpath in files:
            rc2, out2, _ = docker_exec(
                ONLYOFFICE_CONTAINER,
                "bash", "-c",
                f"unzip -p '{fpath}' 2>/dev/null | strings | grep -i 'CTS-2026-0418-ERNSER-003'",
                timeout=15,
            )
            if out2.strip():
                check("14. Document contains ID CTS-2026-0418-ERNSER-003", 1, True,
                      f"found in {fpath}")
                return

        # fallback: try strings directly without unzip
        for fpath in files:
            rc3, out3, _ = docker_exec(
                ONLYOFFICE_CONTAINER,
                "bash", "-c",
                f"strings '{fpath}' 2>/dev/null | grep -i 'CTS-2026-0418-ERNSER-003'",
                timeout=15,
            )
            if out3.strip():
                check("14. Document contains ID CTS-2026-0418-ERNSER-003", 1, True,
                      f"found in {fpath}")
                return

        check("14. Document contains ID CTS-2026-0418-ERNSER-003", 1, False,
              f"ID not found in {len(files)} files")
    except Exception as e:
        check("14. Document contains ID CTS-2026-0418-ERNSER-003", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_01_patient_exists()
    check_02_new_encounter()
    check_03_functional_status()
    check_04_cognitive_status()
    check_05_treatment_plan_goals()
    check_06_treatment_plan_target()
    check_07_transfer_summary_reason()
    check_08_transfer_summary_facility()
    check_09_icd10_codes()
    check_10_disclosure_record()
    check_11_message_subject()
    check_12_message_body()
    check_13_onlyoffice_document()
    check_14_onlyoffice_doc_content()

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
