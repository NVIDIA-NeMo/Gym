# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_002/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-002-I4: Compile Cardiology Referral for Dortha Brakus

Checks: 10 weighted checks across openemr, onlyoffice.
Strategy: docker exec (MariaDB) for OpenEMR; OnlyOffice REST API for document.

Required env vars:
  SERVER_HOSTNAME, OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import io
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

try:
    import requests
except ImportError:
    print("FATAL: requests library not available", file=sys.stderr)
    sys.exit(1)

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENEMR_PORT = os.environ.get("OPENEMR_PORT")
OPENEMR_CONTAINER = os.environ.get("OPENEMR_CONTAINER")
OPENEMR_DB_CONTAINER = os.environ.get("OPENEMR_DB_CONTAINER")

ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.environ.get("ONLYOFFICE_DB_CONTAINER")

for _var_name, _var_val in [
    ("OPENEMR_PORT", OPENEMR_PORT),
    ("OPENEMR_CONTAINER", OPENEMR_CONTAINER),
    ("OPENEMR_DB_CONTAINER", OPENEMR_DB_CONTAINER),
    ("ONLYOFFICE_PORT", ONLYOFFICE_PORT),
    ("ONLYOFFICE_CONTAINER", ONLYOFFICE_CONTAINER),
    ("ONLYOFFICE_DB_CONTAINER", ONLYOFFICE_DB_CONTAINER),
]:
    if not _var_val:
        print(f"FATAL: {_var_name} not set", file=sys.stderr)
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
    """Execute SQL on OpenEMR MariaDB and return stdout."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "openemr", "-popenemr_pass", "-D", "openemr",
        "-N", "-B", "-e", query,
        timeout=15,
    )
    return out.strip()


def onlyoffice_api(method: str, path: str, token: str, **kwargs) -> requests.Response:
    """Make an authenticated OnlyOffice API request."""
    url = f"http://{HOST}:{ONLYOFFICE_PORT}{path}"
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = token
    return requests.request(method, url, headers=headers, timeout=20, **kwargs)


def get_onlyoffice_token() -> str:
    """Authenticate to OnlyOffice and return the auth token."""
    url = f"http://{HOST}:{ONLYOFFICE_PORT}/api/2.0/authentication"
    resp = requests.post(url, json={
        "userName": "admin@onlyoffice.local",
        "password": "NewAdmin123!",
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", {}).get("token", "")


def extract_docx_text(content_bytes: bytes) -> str:
    """Extract plain text from a .docx file's word/document.xml."""
    buf = io.BytesIO(content_bytes)
    with zipfile.ZipFile(buf) as zf:
        if "word/document.xml" not in zf.namelist():
            return ""
        with zf.open("word/document.xml") as f:
            tree = ET.parse(f)
    # Extract all text nodes
    parts = []
    for elem in tree.iter():
        if elem.text:
            parts.append(elem.text)
        if elem.tail:
            parts.append(elem.tail)
    return " ".join(parts)


def _search_section(token: str, section: dict, depth: int = 0) -> tuple[bool, str, str] | None:
    """Search a files/folders section dict for the referral document. Max 2 levels deep."""
    files = section.get("files", [])
    folders = section.get("folders", [])

    for f in files:
        title = f.get("title", "")
        if "Cardiology" in title and "Brakus" in title:
            text = _download_doc(token, f)
            return True, title, text

    if depth < 2:
        for folder in folders:
            fid = folder.get("id")
            if fid is None:
                continue
            fresp = onlyoffice_api("GET", f"/api/2.0/files/{fid}", token)
            if fresp.status_code != 200:
                continue
            inner = fresp.json().get("response", {})
            if isinstance(inner, dict):
                result = _search_section(token, inner, depth + 1)
                if result:
                    return result
    return None


def find_and_download_onlyoffice_doc(token: str) -> tuple[bool, str, str]:
    """
    Search OnlyOffice for the referral document.
    Returns (found, title, extracted_text).
    """
    for scope in ("@my", "@root"):
        resp = onlyoffice_api("GET", f"/api/2.0/files/{scope}", token)
        if resp.status_code != 200:
            continue
        raw = resp.json().get("response", {})
        # @root returns a list of section dicts; @my returns a single section dict
        sections = raw if isinstance(raw, list) else [raw]
        for section in sections:
            if not isinstance(section, dict):
                continue
            result = _search_section(token, section)
            if result:
                return result

    return False, "", ""


def _download_doc(token: str, file_info: dict) -> str:
    """Download a document and extract text. Returns extracted text or empty string."""
    file_id = file_info.get("id")
    if file_id is None:
        return ""

    # Try the filehandler download endpoint
    dl_path = f"/products/files/httphandlers/filehandler.ashx?action=download&fileid={file_id}"
    resp = onlyoffice_api("GET", dl_path, token)
    if resp.status_code == 200 and len(resp.content) > 100:
        try:
            return extract_docx_text(resp.content)
        except Exception:
            pass

    # Fallback: try /api/2.0/files/file/{id}/open
    resp2 = onlyoffice_api("GET", f"/api/2.0/files/file/{file_id}/open", token)
    if resp2.status_code == 200 and len(resp2.content) > 100:
        try:
            return extract_docx_text(resp2.content)
        except Exception:
            pass

    return ""


# ── OpenEMR Checks ────────────────────────────────────────────────────────────

def check_1_patient_exists() -> None:
    """Verify patient Dortha Brakus exists in OpenEMR."""
    try:
        result = openemr_sql(
            "SELECT pid FROM patient_data "
            "WHERE fname='Dortha' AND lname='Brakus' LIMIT 1;"
        )
        passed = bool(result)
        check("1. Patient Dortha Brakus exists in OpenEMR", 1, passed,
              f"pid={result}" if passed else "patient not found")
    except Exception as e:
        check("1. Patient Dortha Brakus exists in OpenEMR", 1, False, f"exception: {e}")


def check_2_allergy_entry() -> None:
    """Verify allergy 'Iodinated contrast media' with correct reaction, severity, active."""
    try:
        result = openemr_sql(
            "SELECT l.title, l.reaction, l.severity_al, l.activity "
            "FROM lists l "
            "JOIN patient_data p ON l.pid = p.pid "
            "WHERE p.fname='Dortha' AND p.lname='Brakus' "
            "AND l.type='allergy' "
            "AND l.title LIKE '%Iodinated contrast media%';"
        )
        if not result:
            check("2. Allergy 'Iodinated contrast media'", 2, False, "allergy not found")
            return

        parts = result.split("\t")
        title = parts[0] if len(parts) > 0 else ""
        reaction = parts[1] if len(parts) > 1 else ""
        severity = parts[2] if len(parts) > 2 else ""
        activity = parts[3] if len(parts) > 3 else ""

        issues = []
        if "Iodinated contrast media" not in title:
            issues.append(f"title mismatch: '{title}'")
        if "anaphylactoid" not in reaction.lower():
            issues.append(f"reaction mismatch: '{reaction}'")
        if activity.strip() != "1":
            issues.append(f"not active: activity={activity}")

        detail = (f"title='{title}', reaction='{reaction}', "
                  f"severity='{severity}', activity={activity}")
        check("2. Allergy 'Iodinated contrast media'", 2, not issues,
              detail if not issues else f"{detail} | {'; '.join(issues)}")
    except Exception as e:
        check("2. Allergy 'Iodinated contrast media'", 2, False, f"exception: {e}")


def check_3_encounter_exists() -> None:
    """Verify at least one encounter exists for Dortha Brakus."""
    try:
        result = openemr_sql(
            "SELECT e.encounter, e.date FROM form_encounter e "
            "JOIN patient_data p ON e.pid = p.pid "
            "WHERE p.fname='Dortha' AND p.lname='Brakus' "
            "ORDER BY e.date DESC LIMIT 1;"
        )
        passed = bool(result)
        check("3. Encounter exists for Dortha Brakus", 1, passed,
              f"latest: {result}" if passed else "no encounters found")
    except Exception as e:
        check("3. Encounter exists for Dortha Brakus", 1, False, f"exception: {e}")


def check_4_clinical_notes() -> None:
    """Verify Clinical Notes form contains referral reason text about heart failure."""
    try:
        # Try form_clinical_notes first
        result = openemr_sql(
            "SELECT cn.description FROM form_clinical_notes cn "
            "WHERE cn.pid IN "
            "(SELECT pid FROM patient_data WHERE fname='Dortha' AND lname='Brakus') "
            "ORDER BY cn.date DESC;"
        )
        if not result:
            # Fallback: try the forms table to find clinical_notes entries
            result = openemr_sql(
                "SELECT f.form_name FROM forms f "
                "JOIN patient_data p ON f.pid = p.pid "
                "WHERE p.fname='Dortha' AND p.lname='Brakus' "
                "AND f.formdir = 'clinical_notes' "
                "ORDER BY f.date DESC LIMIT 5;"
            )

        text_lower = result.lower()
        has_referral_text = (
            "dyspnea" in text_lower
            or "ejection fraction" in text_lower
            or "heart failure" in text_lower
            or "cardiology" in text_lower
        )
        check("4. Clinical Notes contain referral reason", 2, has_referral_text,
              "referral keywords found" if has_referral_text
              else f"referral text not found; got: '{result[:200]}'")
    except Exception as e:
        check("4. Clinical Notes contain referral reason", 2, False, f"exception: {e}")


def check_5_fee_sheet_icd() -> None:
    """Verify Fee Sheet / billing has ICD-10 code I50.22."""
    try:
        result = openemr_sql(
            "SELECT b.code, b.code_type FROM billing b "
            "JOIN patient_data p ON b.pid = p.pid "
            "WHERE p.fname='Dortha' AND p.lname='Brakus' "
            "AND b.code = 'I50.22';"
        )
        if not result:
            # Broader search
            result = openemr_sql(
                "SELECT b.code, b.code_type FROM billing b "
                "WHERE b.pid IN "
                "(SELECT pid FROM patient_data WHERE fname='Dortha' AND lname='Brakus') "
                "AND b.code LIKE '%I50%';"
            )
        passed = "I50.22" in result
        check("5. Fee Sheet has ICD-10 code I50.22", 2, passed,
              f"found: {result}" if passed else "I50.22 not found in billing")
    except Exception as e:
        check("5. Fee Sheet has ICD-10 code I50.22", 2, False, f"exception: {e}")


# ── OnlyOffice Checks ─────────────────────────────────────────────────────────

# Cache the document lookup result so we only fetch once
_oo_cache: dict = {}


def _ensure_oo_doc() -> tuple[bool, str, str]:
    """Fetch and cache the OnlyOffice document. Returns (found, title, text)."""
    if "result" not in _oo_cache:
        try:
            token = get_onlyoffice_token()
            _oo_cache["result"] = find_and_download_onlyoffice_doc(token)
        except Exception as e:
            _oo_cache["result"] = (False, "", "")
            _oo_cache["error"] = str(e)
    return _oo_cache["result"]


def check_6_document_exists() -> None:
    """Verify OnlyOffice document 'Cardiology Referral — Dortha Brakus' exists."""
    try:
        found, title, _ = _ensure_oo_doc()
        check("6. OnlyOffice referral document exists", 1, found,
              f"title='{title}'" if found
              else f"document not found{'; error: ' + _oo_cache.get('error', '') if _oo_cache.get('error') else ''}")
    except Exception as e:
        check("6. OnlyOffice referral document exists", 1, False, f"exception: {e}")


def check_7_clinic_and_demographics() -> None:
    """Verify document contains clinic name and patient DOB."""
    try:
        found, _, text = _ensure_oo_doc()
        if not found or not text:
            check("7. Document has clinic name & patient DOB", 2, False,
                  "document not found or content not extractable")
            return

        text_lower = text.lower()
        has_clinic = "hingham senior care" in text_lower or "hingham" in text_lower
        has_dob = "1953-11-26" in text or "11/26/1953" in text or "november 26" in text_lower or "11-26-1953" in text
        has_name = "dortha" in text_lower and "brakus" in text_lower

        issues = []
        if not has_clinic:
            issues.append("clinic name missing")
        if not has_dob:
            issues.append("DOB missing")
        if not has_name:
            issues.append("patient name missing")

        check("7. Document has clinic name & patient demographics", 2, not issues,
              "all present" if not issues else "; ".join(issues))
    except Exception as e:
        check("7. Document has clinic name & patient demographics", 2, False, f"exception: {e}")


def check_8_referral_reason() -> None:
    """Verify document contains referral reason text."""
    try:
        found, _, text = _ensure_oo_doc()
        if not found or not text:
            check("8. Document has referral reason", 2, False,
                  "document not found or content not extractable")
            return

        text_lower = text.lower()
        has_reason = (
            ("dyspnea" in text_lower or "exertion" in text_lower)
            and ("ejection fraction" in text_lower or "heart failure" in text_lower)
        )
        check("8. Document has referral reason", 2, has_reason,
              "referral reason found" if has_reason else "referral reason text not found")
    except Exception as e:
        check("8. Document has referral reason", 2, False, f"exception: {e}")


def check_9_provider_names() -> None:
    """Verify document contains requesting and receiving provider names."""
    try:
        found, _, text = _ensure_oo_doc()
        if not found or not text:
            check("9. Document has provider names", 2, False,
                  "document not found or content not extractable")
            return

        text_lower = text.lower()
        has_requesting = "lindstrom" in text_lower or "rebecca" in text_lower
        has_receiving = "tanaka" in text_lower or "hiroshi" in text_lower

        issues = []
        if not has_requesting:
            issues.append("Dr. Rebecca Lindstrom not found")
        if not has_receiving:
            issues.append("Dr. Hiroshi Tanaka not found")

        check("9. Document has provider names", 2, not issues,
              "both providers found" if not issues else "; ".join(issues))
    except Exception as e:
        check("9. Document has provider names", 2, False, f"exception: {e}")


def check_10_allergy_in_doc() -> None:
    """Verify document mentions the allergy 'Iodinated contrast media'."""
    try:
        found, _, text = _ensure_oo_doc()
        if not found or not text:
            check("10. Document has allergy info", 1, False,
                  "document not found or content not extractable")
            return

        text_lower = text.lower()
        has_allergy = "iodinated contrast" in text_lower
        check("10. Document has allergy info", 1, has_allergy,
              "allergy mentioned" if has_allergy else "'Iodinated contrast media' not found")
    except Exception as e:
        check("10. Document has allergy info", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_patient_exists()
    check_2_allergy_entry()
    check_3_encounter_exists()
    check_4_clinical_notes()
    check_5_fee_sheet_icd()
    check_6_document_exists()
    check_7_clinic_and_demographics()
    check_8_referral_reason()
    check_9_provider_names()
    check_10_allergy_in_doc()

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
