# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_007/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-007-I5: Prepare Discharge Summary and Follow-Up Plan for Adrianne Simonis

Checks: 13 weighted checks across openemr (8) and onlyoffice (5).
Strategy: docker exec (MariaDB for OpenEMR, MySQL for OnlyOffice) + filesystem inspection.

Required env vars:
  SERVER_HOSTNAME, OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess
import json
import re

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENEMR_PORT = os.getenv("OPENEMR_PORT")
OPENEMR_CONTAINER = os.getenv("OPENEMR_CONTAINER")
OPENEMR_DB_CONTAINER = os.getenv("OPENEMR_DB_CONTAINER")

ONLYOFFICE_PORT = os.getenv("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.getenv("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.getenv("ONLYOFFICE_DB_CONTAINER")

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
def docker_exec(container: str, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    r = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


def openemr_sql(query: str) -> str:
    """Run SQL against OpenEMR MariaDB and return stdout."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "openemr", "-popenemr_pass",
        "-D", "openemr",
        "-N", "-B", "-e", query,
    )
    return out.strip()


def onlyoffice_sql(query: str) -> str:
    """Run SQL against OnlyOffice MySQL and return stdout."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "-D", "onlyoffice",
        "-N", "-B", "-e", query,
    )
    return out.strip()


# ── Shared: find patient PID ─────────────────────────────────────────────────
_patient_pid = None


def get_patient_pid() -> int | None:
    global _patient_pid
    if _patient_pid is not None:
        return _patient_pid
    row = openemr_sql(
        "SELECT pid FROM patient_data "
        "WHERE fname LIKE '%Adrianne%' AND lname LIKE '%Simonis%' LIMIT 1;"
    )
    if row:
        _patient_pid = int(row.split("\t")[0].strip())
    return _patient_pid


# ── Check 1: Patient exists ──────────────────────────────────────────────────
def check_1_patient_exists() -> None:
    """Patient Adrianne Simonis exists in OpenEMR."""
    try:
        pid = get_patient_pid()
        check("1. Patient Adrianne Simonis exists in OpenEMR", 1, pid is not None,
              f"pid={pid}" if pid else "not found")
    except Exception as e:
        check("1. Patient Adrianne Simonis exists in OpenEMR", 1, False, f"exception: {e}")


# ── Check 2: Encounter exists ────────────────────────────────────────────────
_encounter_id = None


def check_2_encounter_exists() -> None:
    """A discharge encounter exists for Adrianne Simonis."""
    global _encounter_id
    try:
        pid = get_patient_pid()
        if not pid:
            check("2. Discharge encounter exists", 1, False, "patient not found")
            return
        # Get the most recent encounter for this patient
        row = openemr_sql(
            f"SELECT encounter FROM form_encounter WHERE pid={pid} "
            f"ORDER BY date DESC LIMIT 1;"
        )
        if row:
            _encounter_id = int(row.split("\t")[0].strip())
        check("2. Discharge encounter exists", 1, _encounter_id is not None,
              f"encounter={_encounter_id}" if _encounter_id else "no encounter found")
    except Exception as e:
        check("2. Discharge encounter exists", 1, False, f"exception: {e}")


# ── Check 3: Vitals ──────────────────────────────────────────────────────────
def check_3_vitals() -> None:
    """Vitals recorded: BP 118/74, pulse 72, temp 98.8, RR 14, O2 99%, weight 62."""
    try:
        pid = get_patient_pid()
        if not pid or not _encounter_id:
            check("3. Discharge vitals recorded", 2, False, "no encounter")
            return
        row = openemr_sql(
            f"SELECT bps, bpd, pulse, temperature, respiration, oxygen_saturation, weight "
            f"FROM form_vitals WHERE pid={pid} "
            f"ORDER BY date DESC LIMIT 1;"
        )
        if not row:
            check("3. Discharge vitals recorded", 2, False, "no vitals found")
            return
        parts = row.split("\t")
        if len(parts) < 7:
            check("3. Discharge vitals recorded", 2, False, f"unexpected format: {row}")
            return
        bps, bpd, pulse, temp, rr, o2, wt = [p.strip() for p in parts[:7]]
        issues = []
        if bps != "118":
            issues.append(f"bps={bps} expected 118")
        if bpd != "74":
            issues.append(f"bpd={bpd} expected 74")
        if pulse != "72":
            issues.append(f"pulse={pulse} expected 72")
        if not (temp.startswith("98.8") or temp == "98.8"):
            issues.append(f"temp={temp} expected 98.8")
        if rr != "14":
            issues.append(f"rr={rr} expected 14")
        # O2 sat may be stored as 99 or 99%
        if not o2.replace("%", "").strip().startswith("99"):
            issues.append(f"o2={o2} expected 99")
        if not wt.startswith("62"):
            issues.append(f"weight={wt} expected 62")
        check("3. Discharge vitals recorded", 2, not issues,
              "all vitals correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("3. Discharge vitals recorded", 2, False, f"exception: {e}")


# ── Check 4: Transfer Summary ────────────────────────────────────────────────
def check_4_transfer_summary() -> None:
    """Transfer Summary with correct reason and receiving facility."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("4. Transfer Summary form", 2, False, "patient not found")
            return
        # Check forms table for transfer summary linked to patient's encounters
        row = openemr_sql(
            f"SELECT f.form_id FROM forms f "
            f"WHERE f.pid={pid} AND f.formdir='transfer_summary' "
            f"AND f.deleted=0 ORDER BY f.date DESC LIMIT 1;"
        )
        if not row:
            check("4. Transfer Summary form", 2, False, "no transfer summary form found")
            return
        form_id = row.split("\t")[0].strip()
        # Query the transfer summary table
        detail_row = openemr_sql(
            f"SELECT reason, receiving_facility FROM form_transfer_summary "
            f"WHERE id={form_id} LIMIT 1;"
        )
        if not detail_row:
            check("4. Transfer Summary form", 2, False, "form record not found in form_transfer_summary")
            return
        parts = detail_row.split("\t")
        reason = parts[0].strip() if len(parts) > 0 else ""
        facility = parts[1].strip() if len(parts) > 1 else ""
        reason_ok = "resolved hypoxemia" in reason.lower() and "pneumonia" in reason.lower()
        facility_ok = facility.lower() == "home"
        check("4. Transfer Summary form", 2, reason_ok and facility_ok,
              f"reason_ok={reason_ok}, facility='{facility}'")
    except Exception as e:
        check("4. Transfer Summary form", 2, False, f"exception: {e}")


# ── Check 5: Clinical Instructions ───────────────────────────────────────────
def check_5_clinical_instructions() -> None:
    """Clinical Instructions form with correct discharge instructions text."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("5. Clinical Instructions form", 2, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT f.form_id FROM forms f "
            f"WHERE f.pid={pid} AND f.formdir='clinical_instructions' "
            f"AND f.deleted=0 ORDER BY f.date DESC LIMIT 1;"
        )
        if not row:
            check("5. Clinical Instructions form", 2, False, "no clinical instructions form found")
            return
        form_id = row.split("\t")[0].strip()
        detail_row = openemr_sql(
            f"SELECT clinical_instructions_text FROM form_clinical_instructions "
            f"WHERE id={form_id} LIMIT 1;"
        )
        if not detail_row:
            # Try alternate column name
            detail_row = openemr_sql(
                f"SELECT instruction FROM form_clinical_instructions "
                f"WHERE id={form_id} LIMIT 1;"
            )
        text = detail_row.strip().lower() if detail_row else ""
        has_antibiotics = "antibiotics" in text
        has_spirometer = "spirometer" in text
        has_hydration = "hydration" in text or "2 liters" in text
        has_fever = "101.5" in text or "fever" in text
        ok = has_antibiotics and has_spirometer and has_hydration and has_fever
        check("5. Clinical Instructions form", 2, ok,
              f"antibiotics={has_antibiotics}, spirometer={has_spirometer}, "
              f"hydration={has_hydration}, fever={has_fever}")
    except Exception as e:
        check("5. Clinical Instructions form", 2, False, f"exception: {e}")


# ── Check 6: Care Plan ───────────────────────────────────────────────────────
def check_6_care_plan() -> None:
    """Care Plan form with correct goal and instructions."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("6. Care Plan form", 2, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT f.form_id FROM forms f "
            f"WHERE f.pid={pid} AND f.formdir='care_plan' "
            f"AND f.deleted=0 ORDER BY f.date DESC LIMIT 1;"
        )
        if not row:
            check("6. Care Plan form", 2, False, "no care plan form found")
            return
        form_id = row.split("\t")[0].strip()
        detail_row = openemr_sql(
            f"SELECT care_plan_type, description, code, codetext "
            f"FROM form_care_plan WHERE id={form_id} LIMIT 5;"
        )
        if not detail_row:
            check("6. Care Plan form", 2, False, "no care plan data found")
            return
        text = detail_row.lower()
        has_goal = "resolution" in text and "pneumonia" in text and "pulmonary" in text
        has_instructions = ("azithromycin" in text and "spirometer" in text
                           and "x-ray" in text and "pneumococcal" in text)
        check("6. Care Plan form", 2, has_goal or has_instructions,
              f"goal_keywords={has_goal}, instruction_keywords={has_instructions}")
    except Exception as e:
        check("6. Care Plan form", 2, False, f"exception: {e}")


# ── Check 7: Fee Sheet ICD-10 codes ──────────────────────────────────────────
def check_7_fee_sheet() -> None:
    """Fee Sheet contains ICD-10 codes J18.9 and J96.01."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("7. Fee Sheet ICD-10 codes", 2, False, "patient not found")
            return
        # billing table stores fee sheet entries
        rows = openemr_sql(
            f"SELECT code FROM billing WHERE pid={pid} "
            f"AND code_type='ICD10' AND activity=1;"
        )
        if not rows:
            # Also try code_type='ICD10' in lists table
            rows = openemr_sql(
                f"SELECT diagnosis FROM billing WHERE pid={pid} AND activity=1;"
            )
        all_codes = rows.lower() if rows else ""
        has_j189 = "j18.9" in all_codes
        has_j9601 = "j96.01" in all_codes
        check("7. Fee Sheet ICD-10 codes", 2, has_j189 and has_j9601,
              f"J18.9={has_j189}, J96.01={has_j9601}")
    except Exception as e:
        check("7. Fee Sheet ICD-10 codes", 2, False, f"exception: {e}")


# ── Check 8: Follow-up appointment ───────────────────────────────────────────
def check_8_appointment() -> None:
    """Follow-up appointment on 2026-05-22 at 13:45 with Dr. Lorinda Pouros."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("8. Follow-up appointment", 3, False, "patient not found")
            return
        rows = openemr_sql(
            f"SELECT e.pc_eventDate, e.pc_startTime, e.pc_hometext, "
            f"e.pc_catid, u.fname, u.lname "
            f"FROM openemr_postcalendar_events e "
            f"LEFT JOIN users u ON e.pc_aid=u.id "
            f"WHERE e.pc_pid={pid} AND e.pc_eventDate='2026-05-22';"
        )
        if not rows:
            check("8. Follow-up appointment", 3, False,
                  "no appointment found for 2026-05-22")
            return
        # Parse rows — may be multiple, find the right one
        found_match = False
        detail_parts = []
        for line in rows.split("\n"):
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            evt_date, evt_time, comment, catid, pfname, plname = [p.strip() for p in parts[:6]]
            provider_name = f"{pfname} {plname}".strip().lower()
            date_ok = evt_date == "2026-05-22"
            time_ok = evt_time.startswith("13:45")
            provider_ok = "lorinda" in provider_name and "pouros" in provider_name
            comment_ok = "pneumonia" in comment.lower() if comment else False
            if date_ok and time_ok and provider_ok:
                found_match = True
                detail_parts.append(
                    f"date={evt_date}, time={evt_time}, provider={pfname} {plname}, "
                    f"comment_ok={comment_ok}"
                )
                break
            detail_parts.append(
                f"date={evt_date}, time={evt_time}, provider={pfname} {plname}"
            )
        check("8. Follow-up appointment", 3, found_match,
              detail_parts[0] if detail_parts else "no matching row")
    except Exception as e:
        check("8. Follow-up appointment", 3, False, f"exception: {e}")


# ── OnlyOffice checks ────────────────────────────────────────────────────────
EXPECTED_DOC_TITLE = "Discharge Summary - Adrianne Simonis - 2026-05-02"

_doc_content: str | None = None


def get_onlyoffice_doc_content() -> str | None:
    """Find the document in OnlyOffice DB and extract its text content."""
    global _doc_content
    if _doc_content is not None:
        return _doc_content

    # Search for the file in OnlyOffice DB
    row = onlyoffice_sql(
        f"SELECT id, content_length FROM files_file "
        f"WHERE title LIKE '%Discharge Summary%Adrianne Simonis%' LIMIT 1;"
    )
    if not row:
        # Try alternate table name
        row = onlyoffice_sql(
            f"SELECT id FROM files_file "
            f"WHERE title LIKE '%Discharge%Simonis%' LIMIT 1;"
        )

    # Try to find and read the document via the OnlyOffice container filesystem
    # OnlyOffice stores documents in /var/www/onlyoffice/Data/ or similar
    rc, out, err = docker_exec(
        ONLYOFFICE_CONTAINER,
        "bash", "-c",
        "find /var/www -name '*.docx' -o -name '*.odt' 2>/dev/null | head -50",
        timeout=30,
    )
    # Also check common OnlyOffice data paths
    rc2, out2, err2 = docker_exec(
        ONLYOFFICE_CONTAINER,
        "bash", "-c",
        "find /app/onlyoffice/data -type f -name '*.docx' 2>/dev/null; "
        "find /var/lib -type f -name '*.docx' 2>/dev/null; "
        "find /tmp -type f -name '*.docx' 2>/dev/null | head -50",
        timeout=30,
    )

    all_files = (out + "\n" + out2).strip().split("\n")
    all_files = [f for f in all_files if f.strip()]

    # Try to extract text from each docx and find the one with our content
    for fpath in all_files:
        fpath = fpath.strip()
        if not fpath:
            continue
        # Extract document.xml from docx (it's a zip)
        rc3, xml_out, _ = docker_exec(
            ONLYOFFICE_CONTAINER,
            "bash", "-c",
            f"unzip -p '{fpath}' word/document.xml 2>/dev/null || true",
            timeout=15,
        )
        if xml_out and "simonis" in xml_out.lower():
            # Strip XML tags to get text
            text = re.sub(r'<[^>]+>', ' ', xml_out)
            text = re.sub(r'\s+', ' ', text)
            _doc_content = text
            return _doc_content

    # If filesystem search didn't work, try OnlyOffice API
    try:
        import requests
        base_url = f"http://{HOST}:{ONLYOFFICE_PORT}"
        # Authenticate
        auth_resp = requests.post(
            f"{base_url}/api/2.0/authentication",
            json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
            timeout=15,
        )
        if auth_resp.status_code == 200:
            token_data = auth_resp.json()
            token = token_data.get("response", {}).get("token", "")
            headers = {"Authorization": token}
            # Search for file
            search_resp = requests.get(
                f"{base_url}/api/2.0/files/@search/Discharge Summary Adrianne Simonis",
                headers=headers,
                timeout=15,
            )
            if search_resp.status_code == 200:
                files = search_resp.json().get("response", [])
                if files:
                    file_id = files[0].get("id")
                    # Download file content
                    dl_resp = requests.get(
                        f"{base_url}/api/2.0/files/file/{file_id}/openedit",
                        headers=headers,
                        timeout=15,
                    )
                    # This might not give raw content — just note we found it
                    _doc_content = json.dumps(files[0])
                    return _doc_content
    except Exception:
        pass

    _doc_content = ""
    return _doc_content


def check_9_doc_exists() -> None:
    """Document titled 'Discharge Summary - Adrianne Simonis - 2026-05-02' exists in OnlyOffice."""
    try:
        # Check DB for file existence
        row = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Discharge Summary%Adrianne Simonis%2026-05-02%' LIMIT 1;"
        )
        if row:
            check("9. OnlyOffice document exists", 1, True, f"found: {row[:80]}")
            return
        # Fallback: broader search
        row = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Discharge%Simonis%' LIMIT 1;"
        )
        if row:
            check("9. OnlyOffice document exists", 1, True, f"partial match: {row[:80]}")
            return
        # Fallback: check via API
        content = get_onlyoffice_doc_content()
        found = content is not None and len(content) > 10
        check("9. OnlyOffice document exists", 1, found,
              "found via content search" if found else "document not found")
    except Exception as e:
        check("9. OnlyOffice document exists", 1, False, f"exception: {e}")


def check_10_doc_header_patient_info() -> None:
    """Document contains clinic name 'Pepperell Primary Care Clinic' and patient info."""
    try:
        content = get_onlyoffice_doc_content()
        if not content:
            check("10. Doc header & patient info", 2, False, "document content not accessible")
            return
        text = content.lower()
        has_clinic = "pepperell" in text and "primary care" in text
        has_patient = "adrianne" in text and "simonis" in text
        has_dob = "1964" in text and "11" in text and "15" in text
        check("10. Doc header & patient info", 2, has_clinic and has_patient,
              f"clinic={has_clinic}, patient={has_patient}, dob={has_dob}")
    except Exception as e:
        check("10. Doc header & patient info", 2, False, f"exception: {e}")


def check_11_doc_hospital_course() -> None:
    """Document contains hospital course narrative with key phrases."""
    try:
        content = get_onlyoffice_doc_content()
        if not content:
            check("11. Doc hospital course", 2, False, "document content not accessible")
            return
        text = content.lower()
        has_cough = "productive cough" in text
        has_pneumonia = "community-acquired pneumonia" in text or "community acquired pneumonia" in text
        has_strep = "streptococcus pneumoniae" in text
        has_defervesced = "defervesced" in text
        ok = has_cough and has_pneumonia and (has_strep or has_defervesced)
        check("11. Doc hospital course", 2, ok,
              f"cough={has_cough}, pneumonia={has_pneumonia}, "
              f"strep={has_strep}, defervesced={has_defervesced}")
    except Exception as e:
        check("11. Doc hospital course", 2, False, f"exception: {e}")


def check_12_doc_medications() -> None:
    """Document contains all three discharge medications."""
    try:
        content = get_onlyoffice_doc_content()
        if not content:
            check("12. Doc discharge medications", 2, False, "document content not accessible")
            return
        text = content.lower()
        has_azithromycin = "azithromycin" in text
        has_guaifenesin = "guaifenesin" in text
        has_acetaminophen = "acetaminophen" in text
        ok = has_azithromycin and has_guaifenesin and has_acetaminophen
        check("12. Doc discharge medications", 2, ok,
              f"azithromycin={has_azithromycin}, guaifenesin={has_guaifenesin}, "
              f"acetaminophen={has_acetaminophen}")
    except Exception as e:
        check("12. Doc discharge medications", 2, False, f"exception: {e}")


def check_13_doc_signature_condition() -> None:
    """Document contains discharge condition and signature block for Dr. Lorinda Pouros, MD."""
    try:
        content = get_onlyoffice_doc_content()
        if not content:
            check("13. Doc condition & signature", 1, False, "document content not accessible")
            return
        text = content.lower()
        has_condition = "afebrile" in text and "room air" in text
        has_signature = "lorinda pouros" in text
        ok = has_condition and has_signature
        check("13. Doc condition & signature", 1, ok,
              f"condition={has_condition}, signature={has_signature}")
    except Exception as e:
        check("13. Doc condition & signature", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_patient_exists()
    check_2_encounter_exists()
    check_3_vitals()
    check_4_transfer_summary()
    check_5_clinical_instructions()
    check_6_care_plan()
    check_7_fee_sheet()
    check_8_appointment()
    check_9_doc_exists()
    check_10_doc_header_patient_info()
    check_11_doc_hospital_course()
    check_12_doc_medications()
    check_13_doc_signature_condition()

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
