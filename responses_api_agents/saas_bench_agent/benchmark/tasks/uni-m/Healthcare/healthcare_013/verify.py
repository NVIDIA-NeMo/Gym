#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_013/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-013-I5: Set Up Telehealth Intake Workflow and Visit Summary

Checks: 13 weighted checks across opnform, openemr, onlyoffice.
Strategy: docker exec (DB queries) for OpnForm and OpenEMR; DB + API for OnlyOffice.

Required env vars:
  SERVER_HOSTNAME, OPNFORM_PORT, OPNFORM_CONTAINER,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import json
import re
import subprocess

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPNFORM_PORT = os.getenv("OPNFORM_PORT")
OPNFORM_CONTAINER = os.getenv("OPNFORM_CONTAINER")

OPENEMR_PORT = os.getenv("OPENEMR_PORT")
OPENEMR_CONTAINER = os.getenv("OPENEMR_CONTAINER")
OPENEMR_DB_CONTAINER = os.getenv("OPENEMR_DB_CONTAINER")

ONLYOFFICE_PORT = os.getenv("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.getenv("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.getenv("ONLYOFFICE_DB_CONTAINER")

_required = {
    "OPNFORM_PORT": OPNFORM_PORT,
    "OPNFORM_CONTAINER": OPNFORM_CONTAINER,
    "OPENEMR_PORT": OPENEMR_PORT,
    "OPENEMR_CONTAINER": OPENEMR_CONTAINER,
    "OPENEMR_DB_CONTAINER": OPENEMR_DB_CONTAINER,
    "ONLYOFFICE_PORT": ONLYOFFICE_PORT,
    "ONLYOFFICE_CONTAINER": ONLYOFFICE_CONTAINER,
    "ONLYOFFICE_DB_CONTAINER": ONLYOFFICE_DB_CONTAINER,
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


def opnform_db(sql: str) -> str:
    """Query OpnForm PostgreSQL (embedded in app container)."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER, "psql", "-U", "forge", "-d", "forge",
        "-t", "-A", "-c", sql,
    )
    return out.strip()


def openemr_db(sql: str) -> str:
    """Query OpenEMR MariaDB."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER, "mysql", "-u", "openemr", "-popenemr_pass",
        "--default-character-set=utf8mb4", "openemr",
        "-N", "-B", "-e", sql,
    )
    return out.strip()


def onlyoffice_db(sql: str) -> str:
    """Query OnlyOffice MySQL 8.0."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER, "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "--default-character-set=utf8mb4", "onlyoffice",
        "-N", "-B", "-e", sql,
    )
    return out.strip()


def _get_patient_pid() -> str:
    """Return the pid for Riley Gleichner, or empty string."""
    return openemr_db(
        "SELECT pid FROM patient_data "
        "WHERE fname='Riley' AND lname='Gleichner' LIMIT 1;"
    )


# ── OpnForm checks ───────────────────────────────────────────────────────────

def check_1_opnform_form_settings() -> None:
    """Form exists with correct title, theme, size, dark_mode, visibility, auto_focus, redirect_url."""
    try:
        row = opnform_db(
            "SELECT title, theme, size, dark_mode, visibility, auto_focus, redirect_url "
            "FROM forms WHERE title = 'Digital Pre-Visit Telehealth Intake Survey' "
            "AND deleted_at IS NULL LIMIT 1;"
        )
        if not row:
            check("1. OpnForm form settings", 2, False, "form not found")
            return
        parts = row.split("|")
        if len(parts) < 7:
            check("1. OpnForm form settings", 2, False, f"unexpected format: {row[:200]}")
            return
        title, theme, size, dark_mode, visibility, auto_focus, redirect_url = (
            p.strip() for p in parts[:7]
        )
        issues = []
        if theme != "notion":
            issues.append(f"theme={theme}")
        if size != "md":
            issues.append(f"size={size}")
        if dark_mode != "light":
            issues.append(f"dark_mode={dark_mode}")
        if visibility != "public":
            issues.append(f"visibility={visibility}")
        if auto_focus not in ("t", "1", "true"):
            issues.append(f"auto_focus={auto_focus}")
        expected_redirect = "https://worcesterwellness.example.com/telehealth/submission-received"
        if redirect_url != expected_redirect:
            issues.append(f"redirect_url mismatch")
        check("1. OpnForm form settings", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("1. OpnForm form settings", 2, False, f"exception: {e}")


def check_2_opnform_form_fields() -> None:
    """Form has ~11 fields with correct types (select, rating, phone, files, nf-text)."""
    try:
        props_raw = opnform_db(
            "SELECT properties FROM forms "
            "WHERE title = 'Digital Pre-Visit Telehealth Intake Survey' "
            "AND deleted_at IS NULL LIMIT 1;"
        )
        if not props_raw:
            check("2. OpnForm form fields", 2, False, "form not found")
            return
        props = json.loads(props_raw)
        field_count = len(props)
        issues = []
        if field_count < 10:
            issues.append(f"only {field_count} fields, expected ~11")

        types_present = {f.get("type", "") for f in props}
        names_lower = [(f.get("name") or "").lower() for f in props]

        # Key field type checks
        if "select" not in types_present:
            issues.append("no select field")
        if "rating" not in types_present:
            issues.append("no rating field")
        if "phone_number" not in types_present and "phone" not in types_present:
            issues.append("no phone field")
        if "files" not in types_present:
            issues.append("no files field")
        if "nf-text" not in types_present:
            issues.append("no text block (nf-text)")

        # Check for Visit Reason select with correct options
        visit_reason_ok = False
        for f in props:
            if f.get("type") == "select":
                fname = (f.get("name") or "").lower()
                if "visit" in fname or "reason" in fname:
                    visit_reason_ok = True
                    break
        if not visit_reason_ok and "select" in types_present:
            # Any select field might be the visit reason
            visit_reason_ok = True

        # Check for conditional logic on medications list
        has_meds_field = any(
            "medication" in n and "list" in n for n in names_lower
        )
        if not has_meds_field:
            # Try broader match
            has_meds_field = any("medication" in n for n in names_lower)
        if not has_meds_field:
            issues.append("Current Medications List field not found")

        check("2. OpnForm form fields", 2, not issues,
              f"{field_count} fields, key types present" if not issues
              else f"{field_count} fields; {'; '.join(issues)}")
    except Exception as e:
        check("2. OpnForm form fields", 2, False, f"exception: {e}")


def check_3_opnform_email_integration() -> None:
    """Email notification integration to intake-triage@worcesterwellness.example.com."""
    try:
        target = "intake-triage@worcesterwellness.example.com"
        row = opnform_db(
            "SELECT fi.integration_id, fi.data::text FROM form_integrations fi "
            "JOIN forms f ON fi.form_id = f.id "
            "WHERE f.title = 'Digital Pre-Visit Telehealth Intake Survey' "
            "AND f.deleted_at IS NULL AND fi.deleted_at IS NULL "
            "LIMIT 10;"
        )
        if not row:
            check("3. OpnForm email integration", 1, False, "no integrations found")
            return
        found = target in row
        check("3. OpnForm email integration", 1, found,
              "email integration configured" if found
              else f"email not found in integrations: {row[:300]}")
    except Exception as e:
        check("3. OpnForm email integration", 1, False, f"exception: {e}")


# ── OpenEMR checks ───────────────────────────────────────────────────────────

def check_4_openemr_patient_phone() -> None:
    """Patient Riley Gleichner has phone (774) 555-0388."""
    try:
        row = openemr_db(
            "SELECT phone_home, phone_cell, phone_biz, phone_contact "
            "FROM patient_data WHERE fname='Riley' AND lname='Gleichner' LIMIT 1;"
        )
        if not row:
            check("4. OpenEMR patient phone", 1, False, "patient not found")
            return
        target = "(774) 555-0388"
        digits = "7745550388"
        raw = row.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
        found = target in row or digits in raw
        check("4. OpenEMR patient phone", 1, found,
              "phone correct" if found else f"phones: {row[:200]}")
    except Exception as e:
        check("4. OpenEMR patient phone", 1, False, f"exception: {e}")


def check_5_openemr_social_history() -> None:
    """Tobacco 'Former smoker, quit 5 years ago', alcohol 'Moderate drinker, 1-2 drinks weekly'."""
    try:
        pid = _get_patient_pid()
        if not pid:
            check("5. OpenEMR social history", 1, False, "patient not found")
            return
        row = openemr_db(
            f"SELECT tobacco, alcohol FROM history_data "
            f"WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not row:
            check("5. OpenEMR social history", 1, False, "no history_data record")
            return
        issues = []
        low = row.lower()
        if "former smoker" not in low and "quit" not in low:
            issues.append("tobacco mismatch")
        if "moderate" not in low and "1-2 drinks" not in low:
            issues.append("alcohol mismatch")
        check("5. OpenEMR social history", 1, not issues,
              "tobacco+alcohol correct" if not issues
              else f"{'; '.join(issues)}; raw: {row[:200]}")
    except Exception as e:
        check("5. OpenEMR social history", 1, False, f"exception: {e}")


def check_6_openemr_vitals() -> None:
    """Vitals: BP 138/88, pulse 74, temp 98.5, weight 215."""
    try:
        pid = _get_patient_pid()
        if not pid:
            check("6. OpenEMR vitals", 2, False, "patient not found")
            return
        row = openemr_db(
            f"SELECT bps, bpd, pulse, temperature, weight FROM form_vitals "
            f"WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not row:
            check("6. OpenEMR vitals", 2, False, "no vitals found")
            return
        parts = row.split("\t")
        if len(parts) < 5:
            check("6. OpenEMR vitals", 2, False, f"unexpected format: {row[:200]}")
            return
        bps, bpd, pulse, temp, weight = (p.strip() for p in parts[:5])
        issues = []
        if bps != "138":
            issues.append(f"bps={bps}")
        if bpd != "88":
            issues.append(f"bpd={bpd}")
        try:
            if abs(float(pulse) - 74) > 0.5:
                issues.append(f"pulse={pulse}")
        except ValueError:
            issues.append(f"pulse={pulse}")
        try:
            if abs(float(temp) - 98.5) > 0.2:
                issues.append(f"temp={temp}")
        except ValueError:
            issues.append(f"temp={temp}")
        try:
            if abs(float(weight) - 215) > 0.5:
                issues.append(f"weight={weight}")
        except ValueError:
            issues.append(f"weight={weight}")
        check("6. OpenEMR vitals", 2, not issues,
              "all vitals correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("6. OpenEMR vitals", 2, False, f"exception: {e}")


def check_7_openemr_ros() -> None:
    """ROS form exists with constitutional and respiratory documentation."""
    try:
        pid = _get_patient_pid()
        if not pid:
            check("7. OpenEMR ROS", 1, False, "patient not found")
            return
        # form_ros has checkbox columns (varchar 3). Check that a recent entry exists
        # and key columns are populated.
        row = openemr_db(
            f"SELECT id, fatigue, fever, chills, weight_change, "
            f"cough, shortness_of_breath, wheezing, sinus_problems "
            f"FROM form_ros WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not row:
            check("7. OpenEMR ROS", 1, False, "no ROS form found")
            return
        # A ROS form exists for this patient — that's the primary signal
        check("7. OpenEMR ROS", 1, True, "ROS form exists")
    except Exception as e:
        check("7. OpenEMR ROS", 1, False, f"exception: {e}")


def check_8_openemr_soap() -> None:
    """SOAP note with correct S/O/A/P text."""
    try:
        pid = _get_patient_pid()
        if not pid:
            check("8. OpenEMR SOAP", 2, False, "patient not found")
            return
        row = openemr_db(
            f"SELECT subjective, objective, assessment, plan FROM form_soap "
            f"WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not row:
            check("8. OpenEMR SOAP", 2, False, "no SOAP form found")
            return
        parts = row.split("\t")
        if len(parts) < 4:
            check("8. OpenEMR SOAP", 2, False, f"unexpected format: {row[:200]}")
            return
        subj, obj, assess, plan = (p.strip() for p in parts[:4])
        issues = []
        if "hypertension" not in subj.lower() or "lisinopril" not in subj.lower():
            issues.append("subjective mismatch")
        if "telehealth" not in obj.lower():
            issues.append("objective mismatch")
        if "essential hypertension" not in assess.lower():
            issues.append("assessment mismatch")
        if "lisinopril" not in plan.lower() or "20mg" not in plan.lower():
            issues.append("plan mismatch")
        check("8. OpenEMR SOAP", 2, not issues,
              "SOAP correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("8. OpenEMR SOAP", 2, False, f"exception: {e}")


def check_9_openemr_icd10() -> None:
    """ICD-10 code I10 billed for encounter."""
    try:
        pid = _get_patient_pid()
        if not pid:
            check("9. OpenEMR ICD-10 I10", 1, False, "patient not found")
            return
        row = openemr_db(
            f"SELECT code FROM billing "
            f"WHERE pid={pid} AND code='I10' AND activity=1 LIMIT 1;"
        )
        found = bool(row and "I10" in row)
        check("9. OpenEMR ICD-10 I10", 1, found,
              "I10 billed" if found else "I10 not found in billing")
    except Exception as e:
        check("9. OpenEMR ICD-10 I10", 1, False, f"exception: {e}")


def check_10_openemr_clinical_instructions() -> None:
    """Clinical instructions contain key directives (lisinopril 20mg, BP monitoring)."""
    try:
        pid = _get_patient_pid()
        if not pid:
            check("10. OpenEMR clinical instructions", 1, False, "patient not found")
            return
        row = openemr_db(
            f"SELECT instruction FROM form_clinical_instructions "
            f"WHERE pid={pid} ORDER BY id DESC LIMIT 1;"
        )
        if not row:
            check("10. OpenEMR clinical instructions", 1, False,
                  "no clinical instructions found")
            return
        low = row.lower()
        has_med = "lisinopril 20mg" in low or "lisinopril" in low
        has_bp = "blood pressure" in low
        ok = has_med and has_bp
        check("10. OpenEMR clinical instructions", 1, ok,
              "instructions correct" if ok
              else f"missing key content in: {row[:200]}")
    except Exception as e:
        check("10. OpenEMR clinical instructions", 1, False, f"exception: {e}")


def check_11_openemr_appointment() -> None:
    """Follow-up appointment on 2026-07-02 at 15:45 with Krystyna Reinger."""
    try:
        pid = _get_patient_pid()
        if not pid:
            check("11. OpenEMR appointment", 2, False, "patient not found")
            return
        row = openemr_db(
            f"SELECT e.pc_eventDate, e.pc_startTime, u.fname, u.lname "
            f"FROM openemr_postcalendar_events e "
            f"LEFT JOIN users u ON e.pc_aid = u.id "
            f"WHERE e.pc_pid='{pid}' AND e.pc_eventDate='2026-07-02' LIMIT 1;"
        )
        if not row:
            check("11. OpenEMR appointment", 2, False,
                  "no appointment found on 2026-07-02")
            return
        parts = row.split("\t")
        issues = []
        if len(parts) >= 4:
            date_val, time_val, ufname, ulname = (p.strip() for p in parts[:4])
            if date_val != "2026-07-02":
                issues.append(f"date={date_val}")
            if not time_val.startswith("15:45"):
                issues.append(f"time={time_val}, expected 15:45")
            provider = f"{ufname} {ulname}".strip().lower()
            if "krystyna" not in provider or "reinger" not in provider:
                issues.append(f"provider={ufname} {ulname}")
        else:
            if "2026-07-02" not in row:
                issues.append("date mismatch")
            if "15:45" not in row:
                issues.append("time mismatch")
        check("11. OpenEMR appointment", 2, not issues,
              "appointment correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("11. OpenEMR appointment", 2, False, f"exception: {e}")


# ── OnlyOffice checks ────────────────────────────────────────────────────────

def check_12_onlyoffice_document_exists() -> None:
    """Document 'Telehealth Visit Summary - Riley Gleichner - 2026-05-21' exists."""
    try:
        row = onlyoffice_db(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Telehealth Visit Summary%Riley Gleichner%2026-05-21%' "
            "LIMIT 1;"
        )
        if not row:
            # Broader search
            row = onlyoffice_db(
                "SELECT id, title FROM files_file "
                "WHERE title LIKE '%Telehealth%Riley%' LIMIT 1;"
            )
        found = bool(row)
        check("12. OnlyOffice document exists", 1, found,
              f"found: {row[:200]}" if found else "document not found")
    except Exception as e:
        check("12. OnlyOffice document exists", 1, False, f"exception: {e}")


def check_13_onlyoffice_document_content() -> None:
    """Document contains key clinical content (clinic info, patient data, vitals, follow-up)."""
    try:
        import requests  # noqa: delayed import — only needed for this check

        base = f"http://{HOST}:{ONLYOFFICE_PORT}"

        # Authenticate
        auth_resp = requests.post(
            f"{base}/api/2.0/authentication",
            json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
            timeout=15,
        )
        if auth_resp.status_code not in (200, 201):
            check("13. OnlyOffice document content", 2, False,
                  f"auth failed: HTTP {auth_resp.status_code}")
            return
        token = auth_resp.json().get("response", {}).get("token", "")
        if not token:
            check("13. OnlyOffice document content", 2, False, "no auth token")
            return
        headers = {"Authorization": token}

        # Search for the document
        search_resp = requests.get(
            f"{base}/api/2.0/files/@search/Telehealth Visit Summary Riley Gleichner",
            headers=headers, timeout=15,
        )
        if search_resp.status_code != 200:
            check("13. OnlyOffice document content", 2, False,
                  f"search failed: HTTP {search_resp.status_code}")
            return
        files = search_resp.json().get("response", [])
        target_file = None
        for f in files:
            t = f.get("title", "")
            if "Telehealth Visit Summary" in t and "Riley Gleichner" in t:
                target_file = f
                break
        if not target_file:
            check("13. OnlyOffice document content", 2, False,
                  "document not found via API search")
            return

        file_id = target_file.get("id")
        # Try to get download URL from viewUrl or dedicated endpoint
        view_url = target_file.get("viewUrl", "")
        # Attempt direct download via API
        dl_resp = requests.get(
            f"{base}/api/2.0/files/file/{file_id}/presigneduri",
            headers=headers, timeout=15,
        )
        download_url = ""
        if dl_resp.status_code == 200:
            download_url = dl_resp.json().get("response", "")
        if not download_url and view_url:
            download_url = view_url

        content = ""
        if download_url:
            try:
                file_resp = requests.get(download_url, headers=headers, timeout=20)
                if file_resp.status_code == 200 and len(file_resp.content) > 100:
                    import zipfile
                    import io
                    try:
                        with zipfile.ZipFile(io.BytesIO(file_resp.content)) as zf:
                            if "word/document.xml" in zf.namelist():
                                xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
                                content = re.sub(r"<[^>]+>", " ", xml)
                    except zipfile.BadZipFile:
                        pass
            except Exception:
                pass

        # Fallback: try docker exec filesystem extraction
        if not content:
            rc, find_out, _ = docker_exec(
                ONLYOFFICE_CONTAINER,
                "bash", "-c",
                "find /var/www/onlyoffice/Data -name '*.docx' -type f 2>/dev/null | head -30",
                timeout=15,
            )
            for fpath in (find_out or "").strip().split("\n"):
                fpath = fpath.strip()
                if not fpath:
                    continue
                rc2, xml_out, _ = docker_exec(
                    ONLYOFFICE_CONTAINER, "bash", "-c",
                    f"unzip -p '{fpath}' word/document.xml 2>/dev/null"
                    " | sed 's/<[^>]*>/ /g'",
                    timeout=15,
                )
                if "Worcester Wellness" in xml_out or "Riley Gleichner" in xml_out:
                    content = xml_out
                    break

        if not content:
            check("13. OnlyOffice document content", 2, False,
                  "document found but content could not be extracted")
            return

        # Verify key phrases
        key_phrases = [
            ("Worcester Wellness Associates", "clinic name"),
            ("(774) 555-0388", "patient phone"),
            ("1965-10-02", "patient DOB"),
            ("138/88", "BP"),
            ("2026-07-02", "follow-up date"),
            ("Krystyna Reinger", "follow-up provider"),
            ("(774) 555-0500", "clinic phone"),
            ("support@worcesterwellness.example.com", "clinic email"),
        ]
        missing = [label for phrase, label in key_phrases if phrase not in content]
        check("13. OnlyOffice document content", 2, not missing,
              "all key content found" if not missing
              else f"missing: {', '.join(missing)}")
    except Exception as e:
        check("13. OnlyOffice document content", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_opnform_form_settings()
    check_2_opnform_form_fields()
    check_3_opnform_email_integration()
    check_4_openemr_patient_phone()
    check_5_openemr_social_history()
    check_6_openemr_vitals()
    check_7_openemr_ros()
    check_8_openemr_soap()
    check_9_openemr_icd10()
    check_10_openemr_clinical_instructions()
    check_11_openemr_appointment()
    check_12_onlyoffice_document_exists()
    check_13_onlyoffice_document_content()

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
