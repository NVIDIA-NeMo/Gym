# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_026/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-026-I2: Prior Authorization Appeal Workflow for Lazaro Lang

Checks: 11 weighted checks across openemr and onlyoffice.
Strategy: docker exec DB (OpenEMR MariaDB), DB + API (OnlyOffice MySQL + REST)

Required env vars:
  SERVER_HOSTNAME, OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess
import re
import io
import zipfile

import requests
from urllib.parse import quote

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENEMR_PORT = os.environ.get("OPENEMR_PORT")
OPENEMR_CONTAINER = os.environ.get("OPENEMR_CONTAINER")
OPENEMR_DB_CONTAINER = os.environ.get("OPENEMR_DB_CONTAINER")

ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.environ.get("ONLYOFFICE_DB_CONTAINER")

for _var in [
    "OPENEMR_PORT", "OPENEMR_CONTAINER", "OPENEMR_DB_CONTAINER",
    "ONLYOFFICE_PORT", "ONLYOFFICE_CONTAINER", "ONLYOFFICE_DB_CONTAINER",
]:
    if not os.environ.get(_var):
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


def openemr_sql(query: str, timeout: int = 15) -> str:
    """Run SQL against OpenEMR MariaDB, return raw stdout."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "openemr", "-popenemr_pass", "-D", "openemr",
        "-N", "-B", "-e", query,
        timeout=timeout,
    )
    return out.strip()


def onlyoffice_sql(query: str, timeout: int = 15) -> str:
    """Run SQL against OnlyOffice MySQL."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass", "-D", "onlyoffice",
        "-N", "-B", "-e", query,
        timeout=timeout,
    )
    return out.strip()


# ── OnlyOffice API helpers ────────────────────────────────────────────────────
_oo_token: str | None = None


def oo_auth() -> str:
    global _oo_token
    if _oo_token:
        return _oo_token
    resp = requests.post(
        f"http://{HOST}:{ONLYOFFICE_PORT}/api/2.0/authentication",
        json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
        timeout=15,
    )
    resp.raise_for_status()
    _oo_token = resp.json()["response"]["token"]
    return _oo_token


def oo_download_file(file_id: int) -> bytes | None:
    """Download file from OnlyOffice via file handler endpoint."""
    token = oo_auth()
    try:
        resp = requests.get(
            f"http://{HOST}:{ONLYOFFICE_PORT}/products/files/httphandlers/filehandler.ashx",
            params={"action": "download", "fileid": str(file_id)},
            headers={"Authorization": token},
            cookies={"asc_auth_key": token},
            timeout=30,
            allow_redirects=True,
        )
        if resp.status_code == 200 and len(resp.content) > 100:
            return resp.content
    except Exception:
        pass
    return None


def extract_docx_text(data: bytes) -> str:
    """Extract plain text from a .docx file (ZIP with word/document.xml)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            text_parts = []
            for name in zf.namelist():
                if name in ("word/document.xml",) or name.startswith("word/header") or name.startswith("word/footer"):
                    xml = zf.read(name).decode("utf-8", errors="replace")
                    text_parts.append(re.sub(r"<[^>]+>", " ", xml))
            return " ".join(text_parts)
    except Exception:
        return ""


def extract_xlsx_info(data: bytes) -> tuple[list[str], str]:
    """Extract sheet names and cell text from an .xlsx file."""
    sheets: list[str] = []
    text = ""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if "xl/workbook.xml" in zf.namelist():
                wb = zf.read("xl/workbook.xml").decode("utf-8", errors="replace")
                sheets = re.findall(r'name="([^"]+)"', wb)
            if "xl/sharedStrings.xml" in zf.namelist():
                ss = zf.read("xl/sharedStrings.xml").decode("utf-8", errors="replace")
                text = re.sub(r"<[^>]+>", " ", ss)
            for name in zf.namelist():
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                    ws = zf.read(name).decode("utf-8", errors="replace")
                    text += " " + re.sub(r"<[^>]+>", " ", ws)
    except Exception:
        pass
    return sheets, text


# ── Shared lookups ────────────────────────────────────────────────────────────
_patient_pid: int | None = None
_newest_encounter: int | None = None


def get_patient_pid() -> int | None:
    global _patient_pid
    if _patient_pid is not None:
        return _patient_pid
    result = openemr_sql(
        "SELECT pid FROM patient_data WHERE fname='Lazaro' AND lname='Lang' LIMIT 1"
    )
    if result:
        _patient_pid = int(result.split("\t")[0].split("\n")[0])
    return _patient_pid


def get_newest_encounter() -> int | None:
    global _newest_encounter
    if _newest_encounter is not None:
        return _newest_encounter
    pid = get_patient_pid()
    if not pid:
        return None
    result = openemr_sql(
        f"SELECT encounter FROM form_encounter WHERE pid={pid} ORDER BY date DESC LIMIT 1"
    )
    if result:
        _newest_encounter = int(result.split("\t")[0].split("\n")[0])
    return _newest_encounter


def _find_oo_file_id(title_fragment: str) -> tuple[int | None, str]:
    """Find a file in OnlyOffice DB by title fragment. Returns (id, title)."""
    # Try files_file first, then files
    for table in ("files_file", "files"):
        try:
            row = onlyoffice_sql(
                f"SELECT id, title FROM {table} "
                f"WHERE title LIKE '%{title_fragment}%' LIMIT 1"
            )
            if row:
                parts = row.split("\t")
                return int(parts[0]), parts[1] if len(parts) > 1 else ""
        except Exception:
            continue
    return None, ""


# ── Cached OnlyOffice document content ────────────────────────────────────────
_appeal_letter_text: str | None = None
_tracking_sheets: list[str] | None = None
_tracking_text: str | None = None


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_patient_exists() -> None:
    """Patient Lazaro Lang exists in OpenEMR."""
    try:
        pid = get_patient_pid()
        check("1. Patient Lazaro Lang exists", 1, pid is not None,
              f"pid={pid}" if pid else "patient not found")
    except Exception as e:
        check("1. Patient Lazaro Lang exists", 1, False, f"exception: {e}")


def check_2_clinical_notes() -> None:
    """New encounter has Clinical Notes with medical necessity justification."""
    try:
        pid = get_patient_pid()
        enc = get_newest_encounter()
        if not pid or not enc:
            check("2. Clinical Notes with justification", 2, False,
                  "patient or encounter not found")
            return

        needle = "recurrent exertional chest pain"
        found = False

        # Try form_clinical_notes
        try:
            notes = openemr_sql(
                f"SELECT description FROM form_clinical_notes "
                f"WHERE pid={pid} AND encounter={enc} AND activity=1"
            )
            if needle.lower() in notes.lower():
                found = True
        except Exception:
            pass

        # Try broader search across common form tables
        if not found:
            for tbl in ("form_soap", "form_soap2", "form_note", "form_dictation",
                        "form_clinical_notes", "form_vitals"):
                try:
                    rows = openemr_sql(
                        f"SELECT * FROM `{tbl}` WHERE encounter={enc} AND activity=1",
                        timeout=5,
                    )
                    if needle.lower() in rows.lower():
                        found = True
                        break
                except Exception:
                    continue

        # Also search the forms registry for any form with the text
        if not found:
            try:
                form_ids = openemr_sql(
                    f"SELECT formdir, form_id FROM forms "
                    f"WHERE encounter={enc} AND deleted=0"
                )
                # Search each form table dynamically
                for line in form_ids.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    formdir = parts[0]
                    fid = parts[1]
                    try:
                        content = openemr_sql(
                            f"SELECT * FROM `form_{formdir}` WHERE id={fid}",
                            timeout=5,
                        )
                        if needle.lower() in content.lower():
                            found = True
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        check("2. Clinical Notes with justification", 2, found,
              f"enc={enc}, text_found={found}")
    except Exception as e:
        check("2. Clinical Notes with justification", 2, False, f"exception: {e}")


def check_3_billing_codes() -> None:
    """Fee Sheet has ICD-10 codes I20.9, R07.9 and CPT 78452."""
    try:
        pid = get_patient_pid()
        enc = get_newest_encounter()
        if not pid or not enc:
            check("3. Billing codes I20.9, R07.9, 78452", 2, False,
                  "patient or encounter not found")
            return

        billing = openemr_sql(
            f"SELECT code_type, code FROM billing "
            f"WHERE pid={pid} AND encounter={enc} AND activity=1"
        )
        has_i209 = "I20.9" in billing
        has_r079 = "R07.9" in billing
        has_78452 = "78452" in billing
        ok = has_i209 and has_r079 and has_78452
        check("3. Billing codes I20.9, R07.9, 78452", 2, ok,
              f"I20.9={'Y' if has_i209 else 'N'}, "
              f"R07.9={'Y' if has_r079 else 'N'}, "
              f"78452={'Y' if has_78452 else 'N'}")
    except Exception as e:
        check("3. Billing codes I20.9, R07.9, 78452", 2, False, f"exception: {e}")


def check_4_misc_billing() -> None:
    """Misc Billing Options: referring provider Cassi McClure, auth AUTH-2026-0322-LL-APL."""
    try:
        pid = get_patient_pid()
        enc = get_newest_encounter()
        if not pid or not enc:
            check("4. Misc billing (provider + auth)", 2, False,
                  "patient or encounter not found")
            return

        # Find the misc billing form for this encounter
        form_id = openemr_sql(
            f"SELECT form_id FROM forms "
            f"WHERE encounter={enc} AND formdir='misc_billing_options' AND deleted=0 "
            f"ORDER BY id DESC LIMIT 1"
        )
        if not form_id:
            check("4. Misc billing (provider + auth)", 2, False,
                  "no misc_billing_options form found")
            return
        fid = int(form_id.split("\t")[0].split("\n")[0])

        row = openemr_sql(
            f"SELECT * FROM form_misc_billing_options WHERE id={fid}"
        )

        # Check auth number
        has_auth = "AUTH-2026-0322-LL-APL" in row

        # Check referring provider — might be stored as name or as user ID
        has_provider = False
        if "Cassi" in row or "McClure" in row:
            has_provider = True
        else:
            # Try to resolve provider ID columns to user names
            try:
                cols = openemr_sql("SHOW COLUMNS FROM form_misc_billing_options")
                provider_cols = [
                    line.split("\t")[0]
                    for line in cols.split("\n")
                    if any(k in line.lower() for k in ("provider", "referring"))
                ]
                for col in provider_cols:
                    val = openemr_sql(
                        f"SELECT `{col}` FROM form_misc_billing_options WHERE id={fid}"
                    )
                    if val and val.strip().isdigit() and int(val.strip()) > 0:
                        user_name = openemr_sql(
                            f"SELECT CONCAT(fname,' ',lname) FROM users WHERE id={val.strip()}"
                        )
                        if "Cassi" in user_name or "McClure" in user_name:
                            has_provider = True
                            break
            except Exception:
                pass

        ok = has_auth and has_provider
        check("4. Misc billing (provider + auth)", 2, ok,
              f"auth={'Y' if has_auth else 'N'}, provider={'Y' if has_provider else 'N'}")
    except Exception as e:
        check("4. Misc billing (provider + auth)", 2, False, f"exception: {e}")


def check_5_appeal_letter_exists() -> None:
    """Appeal letter document exists in OnlyOffice with correct title."""
    global _appeal_letter_text
    try:
        fid, title = _find_oo_file_id("Prior Authorization Appeal Letter - Lang")
        if fid is None:
            # Fallback: try API search
            try:
                token = oo_auth()
                resp = requests.get(
                    f"http://{HOST}:{ONLYOFFICE_PORT}/api/2.0/files/@search/"
                    + quote("Prior Authorization Appeal Letter"),
                    headers={"Authorization": token},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json().get("response", {})
                files = data.get("files", data) if isinstance(data, dict) else data
                if isinstance(files, list):
                    for f in files:
                        if isinstance(f, dict) and "Prior Authorization Appeal Letter" in f.get("title", ""):
                            fid = f["id"]
                            title = f["title"]
                            break
            except Exception:
                pass

        expected = "Prior Authorization Appeal Letter - Lang, Lazaro - 2026-03-22"
        title_ok = expected.lower() in title.lower() if title else False

        # Download content for later checks
        if fid is not None:
            data = oo_download_file(fid)
            if data:
                _appeal_letter_text = extract_docx_text(data)

        check("5. Appeal letter exists", 1, fid is not None and title_ok,
              f"title='{title}'" if fid else "document not found")
    except Exception as e:
        check("5. Appeal letter exists", 1, False, f"exception: {e}")


def check_6_appeal_auth_details() -> None:
    """Appeal letter contains authorization/denial details."""
    try:
        text = _appeal_letter_text or ""
        if not text:
            check("6. Auth/denial details in letter", 2, False, "no document content available")
            return
        tl = text.lower()
        has_auth = "auth-2026-0301-ll" in tl
        has_date = "2026-03-05" in text
        has_reason = "medical necessity not established" in tl
        has_ref = "apl-2026-00305-ll" in tl
        ok = has_auth and has_date and has_reason
        check("6. Auth/denial details in letter", 2, ok,
              f"auth#={'Y' if has_auth else 'N'}, date={'Y' if has_date else 'N'}, "
              f"reason={'Y' if has_reason else 'N'}, ref={'Y' if has_ref else 'N'}")
    except Exception as e:
        check("6. Auth/denial details in letter", 2, False, f"exception: {e}")


def check_7_clinical_justification() -> None:
    """Appeal letter contains medical necessity narrative and active problems."""
    try:
        text = _appeal_letter_text or ""
        if not text:
            check("7. Clinical justification in letter", 2, False, "no document content")
            return
        tl = text.lower()
        has_narrative = "recurrent exertional chest pain" in tl
        has_gingivitis = "gingivitis" in tl
        has_policy = "pol908577" in tl
        has_dob = "1960-10-19" in text
        ok = has_narrative and has_gingivitis
        check("7. Clinical justification in letter", 2, ok,
              f"narrative={'Y' if has_narrative else 'N'}, "
              f"gingivitis={'Y' if has_gingivitis else 'N'}, "
              f"policy={'Y' if has_policy else 'N'}, "
              f"DOB={'Y' if has_dob else 'N'}")
    except Exception as e:
        check("7. Clinical justification in letter", 2, False, f"exception: {e}")


def check_8_evidence_and_codes() -> None:
    """Appeal letter contains supporting evidence, CPT/ICD, and NPI."""
    try:
        text = _appeal_letter_text or ""
        if not text:
            check("8. Evidence + codes + NPI in letter", 2, False, "no document content")
            return
        tl = text.lower()
        has_ev1 = "ekg" in tl or "electrocardiogram" in tl
        has_ev2 = "framingham" in tl or "class ii angina" in tl
        has_ev3 = "hba1c" in tl or "lipid panel" in tl
        has_cpt = "78452" in text
        has_npi = "1871574327" in text
        some_evidence = has_ev1 or has_ev2 or has_ev3
        ok = has_cpt and has_npi and some_evidence
        check("8. Evidence + codes + NPI in letter", 2, ok,
              f"CPT={'Y' if has_cpt else 'N'}, NPI={'Y' if has_npi else 'N'}, "
              f"evidence={'Y' if some_evidence else 'N'}")
    except Exception as e:
        check("8. Evidence + codes + NPI in letter", 2, False, f"exception: {e}")


def check_9_tracking_spreadsheet_exists() -> None:
    """Tracking spreadsheet exists in OnlyOffice."""
    global _tracking_sheets, _tracking_text
    try:
        fid, title = _find_oo_file_id("Claims and Appeal Tracking - Lang")
        if fid is None:
            try:
                token = oo_auth()
                resp = requests.get(
                    f"http://{HOST}:{ONLYOFFICE_PORT}/api/2.0/files/@search/"
                    + quote("Claims and Appeal Tracking"),
                    headers={"Authorization": token},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json().get("response", {})
                files = data.get("files", data) if isinstance(data, dict) else data
                if isinstance(files, list):
                    for f in files:
                        if isinstance(f, dict) and "Claims and Appeal Tracking" in f.get("title", ""):
                            fid = f["id"]
                            title = f["title"]
                            break
            except Exception:
                pass

        expected = "Claims and Appeal Tracking - Lang, Lazaro - 2026-03-22"
        title_ok = expected.lower() in title.lower() if title else False

        if fid is not None:
            data = oo_download_file(fid)
            if data:
                _tracking_sheets, _tracking_text = extract_xlsx_info(data)

        check("9. Tracking spreadsheet exists", 1, fid is not None and title_ok,
              f"title='{title}'" if fid else "spreadsheet not found")
    except Exception as e:
        check("9. Tracking spreadsheet exists", 1, False, f"exception: {e}")


def check_10_spreadsheet_sheets() -> None:
    """Spreadsheet has Claims Register and Authorization Timeline sheets."""
    try:
        sheets = _tracking_sheets or []
        if not sheets:
            check("10. Spreadsheet sheet names", 2, False, "no sheet data available")
            return
        sl = [s.lower() for s in sheets]
        has_claims = any("claims register" in s for s in sl)
        has_timeline = any("authorization timeline" in s for s in sl)
        ok = has_claims and has_timeline
        check("10. Spreadsheet sheet names", 2, ok,
              f"sheets={sheets}")
    except Exception as e:
        check("10. Spreadsheet sheet names", 2, False, f"exception: {e}")


def check_11_spreadsheet_data() -> None:
    """Spreadsheet contains claim status 'pending' and patient data."""
    try:
        text = _tracking_text or ""
        if not text:
            check("11. Spreadsheet claim data", 1, False, "no spreadsheet content")
            return
        tl = text.lower()
        has_pending = "pending" in tl
        has_patient = "lazaro" in tl or "lang" in tl
        ok = has_pending and has_patient
        check("11. Spreadsheet claim data", 1, ok,
              f"pending={'Y' if has_pending else 'N'}, patient={'Y' if has_patient else 'N'}")
    except Exception as e:
        check("11. Spreadsheet claim data", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_patient_exists()
    check_2_clinical_notes()
    check_3_billing_codes()
    check_4_misc_billing()
    check_5_appeal_letter_exists()
    check_6_appeal_auth_details()
    check_7_clinical_justification()
    check_8_evidence_and_codes()
    check_9_tracking_spreadsheet_exists()
    check_10_spreadsheet_sheets()
    check_11_spreadsheet_data()

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
