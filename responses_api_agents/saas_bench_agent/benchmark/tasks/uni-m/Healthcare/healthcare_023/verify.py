# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_023/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-023-I4: Preventive Health Counseling Visit with Personalized Letter for Norman Rath

Checks: 14 weighted checks across openemr (12) and onlyoffice (2).
Strategy: docker exec (MariaDB for OpenEMR, MySQL for OnlyOffice) + OnlyOffice REST API for doc content.

Required env vars:
  SERVER_HOSTNAME, OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER.
"""

import json
import os
import re
import subprocess
import sys

import requests

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

OPENEMR_BASE = f"http://{HOST}:{OPENEMR_PORT}"
ONLYOFFICE_BASE = f"http://{HOST}:{ONLYOFFICE_PORT}"

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
    """Run a SQL query against OpenEMR MariaDB and return stdout."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "openemr", "-popenemr_pass",
        "-D", "openemr",
        "-N", "-B", "-e", query,
    )
    return out.strip()


def onlyoffice_sql(query: str) -> str:
    """Run a SQL query against OnlyOffice MySQL and return stdout."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "-D", "onlyoffice",
        "-N", "-B", "-e", query,
    )
    return out.strip()


def get_patient_pid() -> str:
    """Get Norman Rath's pid from patient_data."""
    result = openemr_sql(
        "SELECT pid FROM patient_data WHERE fname='Norman' AND lname='Rath' LIMIT 1;"
    )
    return result.strip()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_social_history_tobacco() -> None:
    """Verify tobacco field updated in history_data for Norman Rath."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("1. Social history - tobacco", 1, False, "patient Norman Rath not found")
            return
        result = openemr_sql(
            f"SELECT tobacco FROM history_data WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        passed = "cigar" in result.lower() and "2-3" in result
        check("1. Social history - tobacco", 1, passed, f"got: {result[:80]}")
    except Exception as e:
        check("1. Social history - tobacco", 1, False, f"exception: {e}")


def check_2_social_history_alcohol() -> None:
    """Verify alcohol field updated in history_data for Norman Rath."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("2. Social history - alcohol", 1, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT alcohol FROM history_data WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        passed = "4-5 drinks" in result.lower() or "moderate" in result.lower()
        check("2. Social history - alcohol", 1, passed, f"got: {result[:80]}")
    except Exception as e:
        check("2. Social history - alcohol", 1, False, f"exception: {e}")


def check_3_social_history_exercise() -> None:
    """Verify exercise_patterns field updated in history_data for Norman Rath."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("3. Social history - exercise", 1, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT exercise_patterns FROM history_data WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        passed = "cycles" in result.lower() or "yard work" in result.lower()
        check("3. Social history - exercise", 1, passed, f"got: {result[:80]}")
    except Exception as e:
        check("3. Social history - exercise", 1, False, f"exception: {e}")


def check_4_encounter_exists() -> None:
    """Verify a new encounter exists for Norman Rath."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("4. Encounter exists", 1, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT COUNT(*) FROM form_encounter WHERE pid={pid};"
        )
        count = int(result.strip()) if result.strip().isdigit() else 0
        check("4. Encounter exists", 1, count > 0, f"encounter count: {count}")
    except Exception as e:
        check("4. Encounter exists", 1, False, f"exception: {e}")


def check_5_vitals() -> None:
    """Verify vitals recorded: BP 134/86, pulse 74, temp 98.5, RR 15, height 69, weight 195."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("5. Vitals recorded", 2, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT bps, bpd, pulse, temperature, respiration, height, weight "
            f"FROM form_vitals WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not result.strip():
            check("5. Vitals recorded", 2, False, "no vitals found")
            return
        parts = result.split("\t")
        if len(parts) < 7:
            check("5. Vitals recorded", 2, False, f"unexpected format: {result[:100]}")
            return

        bps, bpd, pulse, temp, resp, height, weight = [p.strip() for p in parts[:7]]
        issues = []
        # BP systolic
        if not _approx(bps, 134):
            issues.append(f"bps={bps}")
        # BP diastolic
        if not _approx(bpd, 86):
            issues.append(f"bpd={bpd}")
        # Pulse
        if not _approx(pulse, 74):
            issues.append(f"pulse={pulse}")
        # Temperature
        if not _approx(temp, 98.5, tol=0.5):
            issues.append(f"temp={temp}")
        # Respiration
        if not _approx(resp, 15):
            issues.append(f"resp={resp}")
        # Height 69 inches
        if not _approx(height, 69, tol=1):
            issues.append(f"height={height}")
        # Weight 195 lbs
        if not _approx(weight, 195, tol=1):
            issues.append(f"weight={weight}")

        check("5. Vitals recorded", 2, not issues,
              "all vitals match" if not issues else f"mismatches: {issues}")
    except Exception as e:
        check("5. Vitals recorded", 2, False, f"exception: {e}")


def _approx(val_str: str, expected: float, tol: float = 0.1) -> bool:
    """Check if a numeric string is approximately equal to expected."""
    try:
        return abs(float(val_str) - expected) <= tol
    except (ValueError, TypeError):
        return False


def check_6_ros_form() -> None:
    """Verify ROS form exists for Norman Rath's encounter."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("6. ROS form completed", 1, False, "patient not found")
            return
        # Check if form_ros has an entry linked to this patient's encounter
        result = openemr_sql(
            f"SELECT COUNT(*) FROM form_ros WHERE pid={pid};"
        )
        count = int(result.strip()) if result.strip().isdigit() else 0
        check("6. ROS form completed", 1, count > 0, f"form_ros count: {count}")
    except Exception as e:
        check("6. ROS form completed", 1, False, f"exception: {e}")


def check_7_pe_form() -> None:
    """Verify Physical Exam form exists for Norman Rath's encounter.

    OpenEMR may store PE data in form_observation or other form tables.
    We check the forms registry for any PE-type form linked to the patient.
    """
    try:
        pid = get_patient_pid()
        if not pid:
            check("7. Physical Exam form", 1, False, "patient not found")
            return
        # Check forms table for physical exam or clinical notes form
        encounter_ids = openemr_sql(
            f"SELECT encounter FROM form_encounter WHERE pid={pid} ORDER BY id DESC;"
        )
        if not encounter_ids.strip():
            check("7. Physical Exam form", 1, False, "no encounters found")
            return

        # Look for physical exam form in the forms registry
        enc_list = ",".join(encounter_ids.strip().split("\n"))
        result = openemr_sql(
            f"SELECT COUNT(*) FROM forms WHERE pid={pid} "
            f"AND encounter IN ({enc_list}) "
            f"AND (form_name LIKE '%Physical Exam%' OR form_name LIKE '%physical_exam%' "
            f"OR form_name LIKE '%Clinical%' OR form_name LIKE '%Exam%' "
            f"OR formdir LIKE '%physical%' OR formdir LIKE '%exam%');"
        )
        count = int(result.strip()) if result.strip().isdigit() else 0
        if count > 0:
            check("7. Physical Exam form", 1, True, f"PE form count: {count}")
        else:
            # Fallback: check form_observation
            obs_count = openemr_sql(
                f"SELECT COUNT(*) FROM form_observation WHERE pid={pid};"
            )
            obs_c = int(obs_count.strip()) if obs_count.strip().isdigit() else 0
            check("7. Physical Exam form", 1, obs_c > 0, f"form_observation count: {obs_c}")
    except Exception as e:
        check("7. Physical Exam form", 1, False, f"exception: {e}")


def check_8_soap_note() -> None:
    """Verify SOAP note with correct Subjective and Objective content."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("8. SOAP note (S/O)", 3, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT subjective, objective FROM form_soap WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not result.strip():
            check("8. SOAP note (S/O)", 3, False, "no SOAP note found")
            return

        parts = result.split("\t")
        subjective = parts[0] if len(parts) > 0 else ""
        objective = parts[1] if len(parts) > 1 else ""

        issues = []
        if "preventive health counseling" not in subjective.lower():
            issues.append("subjective missing key phrase")
        if "borderline" not in subjective.lower() and "cholesterol" not in subjective.lower():
            issues.append("subjective missing cholesterol reference")
        if "134/86" not in objective:
            issues.append("objective missing BP 134/86")

        check("8. SOAP note (S/O)", 3, not issues,
              "S/O content matches" if not issues else f"issues: {issues}")
    except Exception as e:
        check("8. SOAP note (S/O)", 3, False, f"exception: {e}")


def check_9_soap_assessment_plan() -> None:
    """Verify SOAP note Assessment and Plan content."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("9. SOAP note (A/P)", 2, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT assessment, plan FROM form_soap WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not result.strip():
            check("9. SOAP note (A/P)", 2, False, "no SOAP note found")
            return

        parts = result.split("\t")
        assessment = parts[0] if len(parts) > 0 else ""
        plan = parts[1] if len(parts) > 1 else ""

        issues = []
        if "hypertension" not in assessment.lower() and "pre-hypertension" not in assessment.lower():
            issues.append("assessment missing hypertension")
        if "mediterranean" not in plan.lower():
            issues.append("plan missing Mediterranean diet")
        if "cigar" not in plan.lower() and "cessation" not in plan.lower():
            issues.append("plan missing cigar cessation")

        check("9. SOAP note (A/P)", 2, not issues,
              "A/P content matches" if not issues else f"issues: {issues}")
    except Exception as e:
        check("9. SOAP note (A/P)", 2, False, f"exception: {e}")


def check_10_care_plan() -> None:
    """Verify Care Plan form with goal and instructions."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("10. Care Plan", 2, False, "patient not found")
            return
        # Care Plan may store goal/instructions in description or codetext fields
        result = openemr_sql(
            f"SELECT description, codetext FROM form_care_plan WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not result.strip():
            # Fallback: check via the forms registry and form_care_plan
            result2 = openemr_sql(
                f"SELECT COUNT(*) FROM forms WHERE pid={pid} "
                f"AND (form_name LIKE '%Care Plan%' OR formdir LIKE '%care_plan%');"
            )
            count = int(result2.strip()) if result2.strip().isdigit() else 0
            check("10. Care Plan", 2, count > 0, f"care plan form count: {count}, but no detail rows")
            return

        combined = result.lower()
        issues = []
        if "bp" not in combined and "130/80" not in combined:
            issues.append("missing BP goal")
        if "mediterranean" not in combined:
            issues.append("missing Mediterranean diet instruction")
        if "150 min" not in combined and "150 minutes" not in combined:
            issues.append("missing exercise instruction")

        check("10. Care Plan", 2, not issues,
              "care plan matches" if not issues else f"issues: {issues}")
    except Exception as e:
        check("10. Care Plan", 2, False, f"exception: {e}")


def check_11_fee_sheet_codes() -> None:
    """Verify ICD-10 Z71.3 and CPT 99403 on Fee Sheet (billing table)."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("11. Fee Sheet codes", 2, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT code_type, code FROM billing WHERE pid={pid} AND activity=1;"
        )
        if not result.strip():
            check("11. Fee Sheet codes", 2, False, "no billing records found")
            return

        codes = set()
        for line in result.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                codes.add((parts[0].strip(), parts[1].strip()))

        has_icd = any(c[1] == "Z71.3" for c in codes)
        has_cpt = any(c[1] == "99403" for c in codes)

        issues = []
        if not has_icd:
            issues.append("ICD-10 Z71.3 not found")
        if not has_cpt:
            issues.append("CPT 99403 not found")

        check("11. Fee Sheet codes", 2, not issues,
              f"found codes: {codes}" if not issues else f"issues: {issues}, found: {codes}")
    except Exception as e:
        check("11. Fee Sheet codes", 2, False, f"exception: {e}")


def check_12_followup_appointment() -> None:
    """Verify follow-up appointment on 2026-07-20 at 9:30 AM for Norman Rath."""
    try:
        pid = get_patient_pid()
        if not pid:
            check("12. Follow-up appointment", 2, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT pc_eventDate, pc_startTime, pc_title, pc_apptstatus "
            f"FROM openemr_postcalendar_events "
            f"WHERE pc_pid='{pid}' AND pc_eventDate='2026-07-20';"
        )
        if not result.strip():
            check("12. Follow-up appointment", 2, False, "no appointment found on 2026-07-20")
            return

        # Check time is 09:30
        has_correct_time = "09:30" in result
        check("12. Follow-up appointment", 2, has_correct_time,
              f"appointment data: {result[:120]}")
    except Exception as e:
        check("12. Follow-up appointment", 2, False, f"exception: {e}")


def check_13_onlyoffice_document_exists() -> None:
    """Verify OnlyOffice document titled 'Preventive Health Summary Letter - Norman Rath - April 2026' exists."""
    try:
        result = onlyoffice_sql(
            "SELECT id, title FROM files_file WHERE title LIKE "
            "'%Preventive Health Summary Letter%Norman Rath%';"
        )
        if not result.strip():
            # Try broader search
            result = onlyoffice_sql(
                "SELECT id, title FROM files_file WHERE title LIKE '%Norman Rath%';"
            )
        if not result.strip():
            check("13. OnlyOffice doc exists", 1, False, "document not found in files_file")
            return

        passed = "norman rath" in result.lower() and "preventive" in result.lower()
        check("13. OnlyOffice doc exists", 1, passed, f"found: {result[:100]}")
    except Exception as e:
        check("13. OnlyOffice doc exists", 1, False, f"exception: {e}")


def check_14_onlyoffice_document_content() -> None:
    """Verify OnlyOffice document contains key content: clinic name, recommendations, care plan."""
    try:
        # Authenticate with OnlyOffice API
        session = requests.Session()
        auth_resp = session.post(
            f"{ONLYOFFICE_BASE}/api/2.0/authentication",
            json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
            timeout=15,
        )
        if auth_resp.status_code not in (200, 201):
            check("14. OnlyOffice doc content", 2, False,
                  f"auth failed: {auth_resp.status_code}")
            return

        auth_data = auth_resp.json()
        token = auth_data.get("response", {}).get("token", "")
        if not token:
            check("14. OnlyOffice doc content", 2, False, "no auth token returned")
            return

        headers = {"Authorization": token}

        # Search for the document
        search_resp = session.get(
            f"{ONLYOFFICE_BASE}/api/2.0/files/@search/Preventive Health Summary Letter Norman Rath",
            headers=headers, timeout=15,
        )

        if search_resp.status_code != 200:
            check("14. OnlyOffice doc content", 2, False,
                  f"search failed: {search_resp.status_code}")
            return

        search_data = search_resp.json()
        resp = search_data.get("response", [])
        # Search response is a list of file objects
        files = resp if isinstance(resp, list) else resp.get("files", []) if isinstance(resp, dict) else []
        if not files:
            # fallback: list all files in my documents
            list_resp = session.get(
                f"{ONLYOFFICE_BASE}/api/2.0/files/@my",
                headers=headers, timeout=15,
            )
            if list_resp.status_code == 200:
                list_data = list_resp.json()
                lr = list_data.get("response", {})
                all_files = lr.get("files", []) if isinstance(lr, dict) else []
                files = [f for f in all_files
                         if "norman rath" in f.get("title", "").lower()]

        if not files:
            check("14. OnlyOffice doc content", 2, False, "document not found via API")
            return

        file_info = files[0]
        file_id = file_info.get("id")

        # Try to get document content via download and text extraction
        # Download the file
        dl_resp = session.get(
            f"{ONLYOFFICE_BASE}/api/2.0/files/file/{file_id}/open",
            headers=headers, timeout=30, allow_redirects=True,
        )

        # Alternative: try direct file content via docker exec
        # Find file on disk and extract text from DOCX
        rc, out, err = docker_exec(
            ONLYOFFICE_CONTAINER,
            "bash", "-c",
            "find /var/www/onlyoffice/Data/ -name '*.docx' -o -name '*.txt' 2>/dev/null | head -20",
            timeout=15,
        )

        # Try to extract text from the first matching docx
        content_text = ""
        if out.strip():
            for fpath in out.strip().split("\n"):
                fpath = fpath.strip()
                if not fpath:
                    continue
                # Try to extract text content from docx using python
                rc2, out2, err2 = docker_exec(
                    ONLYOFFICE_CONTAINER,
                    "bash", "-c",
                    f"python3 -c \""
                    f"import zipfile, xml.etree.ElementTree as ET; "
                    f"z=zipfile.ZipFile('{fpath}'); "
                    f"t=ET.fromstring(z.read('word/document.xml')); "
                    f"print(' '.join(n.text for n in t.iter() if n.text))\" 2>/dev/null || "
                    f"unzip -p '{fpath}' word/document.xml 2>/dev/null | "
                    f"sed -e 's/<[^>]*>//g' | head -200",
                    timeout=15,
                )
                if out2.strip():
                    content_text += out2
                    break

        if not content_text:
            # Can't extract content - just verify document exists (partial credit handled by check 13)
            check("14. OnlyOffice doc content", 2, False,
                  "could not extract document content for verification")
            return

        content_lower = content_text.lower()
        issues = []
        if "tewksbury" not in content_lower:
            issues.append("missing clinic name 'Tewksbury'")
        if "mediterranean" not in content_lower:
            issues.append("missing Mediterranean recommendation")
        if "134/86" not in content_text:
            issues.append("missing BP 134/86")
        if "978" not in content_text and "555-0400" not in content_text:
            issues.append("missing clinic phone")

        check("14. OnlyOffice doc content", 2, not issues,
              "key content found" if not issues else f"issues: {issues}")
    except Exception as e:
        check("14. OnlyOffice doc content", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_social_history_tobacco()
    check_2_social_history_alcohol()
    check_3_social_history_exercise()
    check_4_encounter_exists()
    check_5_vitals()
    check_6_ros_form()
    check_7_pe_form()
    check_8_soap_note()
    check_9_soap_assessment_plan()
    check_10_care_plan()
    check_11_fee_sheet_codes()
    check_12_followup_appointment()
    check_13_onlyoffice_document_exists()
    check_14_onlyoffice_document_content()

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
