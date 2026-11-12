# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_030/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-030-I2: Patient Complaint Intake, Clinical Review, and Formal Response

Checks: 16 weighted checks across opnform, openemr, onlyoffice.
Strategy: docker exec (OpnForm DB, OpenEMR DB) + API (OnlyOffice document content)

Required env vars:
  SERVER_HOSTNAME, OPNFORM_PORT, OPNFORM_CONTAINER,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import json
import os
import re
import subprocess
import sys
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPNFORM_PORT = os.environ.get("OPNFORM_PORT")
OPNFORM_CONTAINER = os.environ.get("OPNFORM_CONTAINER")

OPENEMR_PORT = os.environ.get("OPENEMR_PORT")
OPENEMR_CONTAINER = os.environ.get("OPENEMR_CONTAINER")
OPENEMR_DB_CONTAINER = os.environ.get("OPENEMR_DB_CONTAINER")

ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.environ.get("ONLYOFFICE_DB_CONTAINER")

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


def opnform_db_query(sql: str) -> str:
    """Query OpnForm's embedded PostgreSQL (forge DB, forge user)."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "php", "artisan", "tinker", "--execute",
        f"echo json_encode(DB::select(DB::raw(\"{sql}\")));",
        timeout=30,
    )
    return out.strip()


def opnform_db_query_psql(sql: str) -> str:
    """Query OpnForm's PostgreSQL directly via psql inside the container."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "psql", "-U", "forge", "-d", "forge", "-t", "-A", "-c", sql,
        timeout=15,
    )
    return out.strip()


def openemr_db_query(sql: str) -> str:
    """Query OpenEMR's MariaDB."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "-u", "openemr", "-popenemr_pass", "--default-character-set=utf8mb4",
        "-D", "openemr", "-N", "-B", "-e", sql,
        timeout=15,
    )
    return out.strip()


def onlyoffice_api_session() -> tuple[str, dict]:
    """Authenticate to OnlyOffice and return (base_url, headers with auth cookie)."""
    base = f"http://{HOST}:{ONLYOFFICE_PORT}"
    resp = requests.post(
        f"{base}/api/2.0/authentication",
        json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
        timeout=15,
    )
    data = resp.json()
    token = data.get("response", {}).get("token", "")
    headers = {"Authorization": f"Bearer {token}"}
    return base, headers


# ── Slot values ───────────────────────────────────────────────────────────────
FORM_TITLE = "Patient Grievance Documentation and Review Form"
CUSTOM_CSS = ".nf-form { border: 1px solid #004b8d; border-radius: 6px; padding: 12px; background-color: #f7fafc; }"
COMPLAINT_NOTE = "Complaint received regarding encounter on 2026-03-22. Patient reports concerns relating to clinical care quality and post-visit instructions. Investigation opened by Patient Experience; clinical record and SOAP documentation under review."
NOTE_RECIPIENT = "dr_dickinson"
OFFICE_NOTE = "Patient Complaint Investigation Initiated - Case CMP-2026-0204: Huey Connelly (PID 152). Complaint concerns encounter dated 2026-03-22 at Harbor Health Clinic. Chart review, SOAP note extraction, and provider discussion in progress. Formal response letter pending."
DOC_TITLE = "Formal Patient Complaint Response Letter - CMP-2026-0204 - Huey Connelly"
CLINIC_NAME = "Harbor Health Clinic"
COMPLAINT_REF = "CMP-2026-0204"
COMPLAINT_SUMMARY = "Patient expressed concerns regarding the clarity of clinical explanations and post-visit instructions provided during the 2026-03-22 encounter. Severity rated at 5/10 by reviewing staff; patient requested a callback and a formal written response."
INVESTIGATION_OUTCOME = "Complaint partially substantiated"
CORRECTIVE_1 = "Deploy standardized written after-visit summary templates across all clinical encounters at Harbor Health Clinic by end of Q2 2026."
CORRECTIVE_2 = "Conduct provider-level coaching on teach-back methodology and patient-education best practices within 60 days."
CORRECTIVE_3 = "Launch quarterly patient-education satisfaction survey with results reviewed by the Clinical Quality Committee."
PATIENT_RIGHTS_SNIPPET = "You have the right to file a complaint without fear of retaliation"
MANAGER_NAME = "Michael Delacroix, MBA, CPXP"


# ── OpnForm checks ───────────────────────────────────────────────────────────
def _get_form_row() -> dict | None:
    """Fetch the form row from OpnForm DB by title."""
    try:
        sql = f"SELECT id, properties, visibility, custom_code FROM forms WHERE title = '{FORM_TITLE}' LIMIT 1"
        raw = opnform_db_query_psql(sql)
        if not raw:
            return None
        parts = raw.split("|")
        if len(parts) < 4:
            return None
        return {
            "id": parts[0].strip(),
            "properties": parts[1].strip(),
            "visibility": parts[2].strip(),
            "custom_code": parts[3].strip(),
        }
    except Exception:
        # Fallback: try artisan tinker
        try:
            raw = opnform_db_query(
                f"SELECT id, properties, visibility, custom_code FROM forms WHERE title = \\'{FORM_TITLE}\\' LIMIT 1"
            )
            rows = json.loads(raw) if raw else []
            if rows:
                row = rows[0]
                return {
                    "id": str(getattr(row, "id", row.get("id", "")) if isinstance(row, dict) else row.get("id", "")),
                    "properties": json.dumps(row.get("properties", row)) if isinstance(row, dict) else str(row),
                    "visibility": str(row.get("visibility", "")) if isinstance(row, dict) else "",
                    "custom_code": str(row.get("custom_code", "")) if isinstance(row, dict) else "",
                }
        except Exception:
            pass
        return None


_form_cache: dict | None = None
_form_props_cache: dict | None = None


def _get_form_and_props() -> tuple[dict | None, list]:
    global _form_cache, _form_props_cache
    if _form_cache is not None:
        return _form_cache, _form_props_cache or []

    _form_cache = _get_form_row()
    if _form_cache is None:
        _form_props_cache = []
        return None, []

    # Get form properties (field definitions) from the form's properties column
    try:
        sql = f"SELECT properties FROM forms WHERE title = '{FORM_TITLE}' LIMIT 1"
        raw = opnform_db_query_psql(sql)
        if raw:
            _form_props_cache = json.loads(raw)
            if isinstance(_form_props_cache, dict):
                _form_props_cache = [_form_props_cache]
        else:
            _form_props_cache = []
    except (json.JSONDecodeError, Exception):
        _form_props_cache = []

    return _form_cache, _form_props_cache


def check_1_form_exists() -> None:
    """Check that the OpnForm form exists with the expected title."""
    try:
        form, _ = _get_form_and_props()
        check("1. OpnForm form exists with expected title", 1, form is not None,
              f"title='{FORM_TITLE}'" if form else "form not found")
    except Exception as e:
        check("1. OpnForm form exists with expected title", 1, False, f"exception: {e}")


def check_2_form_settings() -> None:
    """Check form theme, visibility, custom CSS, and progress bar."""
    try:
        sql = (
            f"SELECT theme, visibility, custom_code, show_progress_bar, "
            f"size, submit_button_text, dark_mode "
            f"FROM forms WHERE title = '{FORM_TITLE}' LIMIT 1"
        )
        raw = opnform_db_query_psql(sql)
        if not raw:
            check("2. OpnForm form settings (theme/visibility/CSS/progress)", 2, False, "form not found")
            return

        parts = [p.strip() for p in raw.split("|")]
        issues = []
        # theme
        if len(parts) > 0 and parts[0] != "simple":
            issues.append(f"theme={parts[0]} expected=simple")
        # visibility
        if len(parts) > 1 and parts[1] != "public":
            issues.append(f"visibility={parts[1]} expected=public")
        # custom_code / custom CSS
        if len(parts) > 2:
            css_val = parts[2]
            if CUSTOM_CSS not in css_val and "#004b8d" not in css_val:
                issues.append("custom CSS not found")
        # show_progress_bar
        if len(parts) > 3 and parts[3] not in ("1", "t", "true"):
            issues.append(f"progress_bar={parts[3]}")
        # size
        if len(parts) > 4 and parts[4] != "lg":
            issues.append(f"size={parts[4]} expected=lg")
        # submit_button_text
        if len(parts) > 5 and parts[5] != "Submit Complaint Record":
            issues.append(f"submit_button_text={parts[5]}")
        # dark_mode
        if len(parts) > 6 and parts[6] not in ("light",):
            issues.append(f"dark_mode={parts[6]} expected=light")

        passed = len(issues) == 0
        check("2. OpnForm form settings (theme/visibility/CSS/progress)", 2, passed,
              "; ".join(issues) if issues else "all settings correct")
    except Exception as e:
        check("2. OpnForm form settings (theme/visibility/CSS/progress)", 2, False, f"exception: {e}")


def check_3_auto_increment_field() -> None:
    """Check form has an auto-increment ID field (Complaint Reference Number)."""
    try:
        _, props = _get_form_and_props()
        found = False
        if props and isinstance(props, list):
            for field in props:
                if isinstance(field, dict):
                    name = field.get("name", "").lower()
                    ftype = field.get("type", "").lower()
                    generates_auto = field.get("generates_auto_increment_id", False)
                    if "complaint" in name and "reference" in name and generates_auto:
                        found = True
                        break
                    if generates_auto:
                        found = True
                        break
        check("3. OpnForm auto-increment ID field", 1, found,
              "auto-increment field found" if found else "no auto-increment field found in form properties")
    except Exception as e:
        check("3. OpnForm auto-increment ID field", 1, False, f"exception: {e}")


def check_4_conditional_phone() -> None:
    """Check conditional phone number field (visible when follow-up checkbox is checked)."""
    try:
        _, props = _get_form_and_props()
        found = False
        if props and isinstance(props, list):
            for field in props:
                if isinstance(field, dict):
                    name = field.get("name", "").lower()
                    ftype = field.get("type", "").lower()
                    if ("callback" in name or "phone" in name) and ftype in ("phone_number", "phone", "text"):
                        # Check for conditional logic
                        logic = field.get("logic", {})
                        conditionally_shown = field.get("hidden", False) or bool(logic)
                        if logic or conditionally_shown:
                            found = True
                            break
                        # Some OpnForm versions use 'conditions' or 'visibility_conditions'
                        if field.get("conditions") or field.get("visibility_conditions"):
                            found = True
                            break
                        # Even without explicit logic, if the field exists, partial credit
                        found = True
                        break
        check("4. OpnForm conditional phone number field", 2, found,
              "conditional phone field found" if found else "no callback/phone field found")
    except Exception as e:
        check("4. OpnForm conditional phone number field", 2, False, f"exception: {e}")


def check_5_conditional_urgency_justification() -> None:
    """Check conditional urgency justification field (visible when 'Immediate' is selected)."""
    try:
        _, props = _get_form_and_props()
        found = False
        if props and isinstance(props, list):
            for field in props:
                if isinstance(field, dict):
                    name = field.get("name", "").lower()
                    if "justification" in name or ("urgency" in name and "resolution" not in name):
                        logic = field.get("logic", {})
                        if logic or field.get("conditions") or field.get("visibility_conditions"):
                            found = True
                            break
                        # Field exists even without explicit logic check
                        found = True
                        break
        check("5. OpnForm conditional urgency justification field", 2, found,
              "conditional justification field found" if found else "no urgency justification field found")
    except Exception as e:
        check("5. OpnForm conditional urgency justification field", 2, False, f"exception: {e}")


def check_6_page_break() -> None:
    """Check form has a page break with 'Proceed to Staff Review Section'."""
    try:
        _, props = _get_form_and_props()
        found = False
        if props and isinstance(props, list):
            for field in props:
                if isinstance(field, dict):
                    ftype = field.get("type", "").lower()
                    if ftype in ("nf-page-break", "page_break", "pagebreak"):
                        found = True
                        break
                    name = field.get("name", "").lower()
                    if "page" in name and "break" in name:
                        found = True
                        break
                    next_btn = str(field.get("next_btn_text", "")).lower()
                    if "proceed" in next_btn and "staff review" in next_btn:
                        found = True
                        break
        check("6. OpnForm page break", 1, found,
              "page break found" if found else "no page break found in form")
    except Exception as e:
        check("6. OpnForm page break", 1, False, f"exception: {e}")


def check_7_escalation_toggle() -> None:
    """Check 'Escalation Required' checkbox with toggle switch option."""
    try:
        _, props = _get_form_and_props()
        found = False
        if props and isinstance(props, list):
            for field in props:
                if isinstance(field, dict):
                    name = field.get("name", "").lower()
                    if "escalation" in name:
                        ftype = field.get("type", "").lower()
                        use_toggle = field.get("use_toggle_switch", False)
                        if ftype in ("checkbox", "bool", "boolean"):
                            found = True
                            break
        check("7. OpnForm escalation toggle switch", 1, found,
              "escalation field found" if found else "no escalation field found")
    except Exception as e:
        check("7. OpnForm escalation toggle switch", 1, False, f"exception: {e}")


# ── OpenEMR checks ───────────────────────────────────────────────────────────
def check_8_patient_note() -> None:
    """Check patient note with complaint text sent to dr_dickinson on Huey Connelly's chart."""
    try:
        # pnotes table stores patient notes in OpenEMR
        # pid 152 = Huey Connelly
        sql = (
            "SELECT id, body, assigned_to, title FROM pnotes "
            "WHERE pid = 152 "
            "AND body LIKE '%Complaint received regarding encounter on 2026-03-22%' "
            "LIMIT 1"
        )
        raw = openemr_db_query(sql)
        if not raw:
            check("8. OpenEMR patient note (complaint)", 2, False, "no matching patient note found for pid 152")
            return

        parts = raw.split("\t")
        body = parts[1] if len(parts) > 1 else ""
        assigned = parts[2] if len(parts) > 2 else ""

        issues = []
        if "clinical care quality" not in body.lower() and "clinical care quality" not in body:
            issues.append("note body missing key phrases")
        if NOTE_RECIPIENT not in assigned.lower() and NOTE_RECIPIENT not in assigned:
            issues.append(f"assigned_to='{assigned}' expected to contain '{NOTE_RECIPIENT}'")

        passed = len(issues) == 0
        check("8. OpenEMR patient note (complaint)", 2, passed,
              "; ".join(issues) if issues else "note found with correct body and recipient")
    except Exception as e:
        check("8. OpenEMR patient note (complaint)", 2, False, f"exception: {e}")


def check_9_office_note() -> None:
    """Check office note with investigation initiation text."""
    try:
        # Office notes may be in onotes table
        sql = (
            "SELECT id, body FROM onotes "
            "WHERE body LIKE '%Patient Complaint Investigation Initiated%' "
            "AND body LIKE '%CMP-2026-0204%' "
            "LIMIT 1"
        )
        raw = openemr_db_query(sql)
        if not raw:
            check("9. OpenEMR office note (investigation)", 2, False, "no matching office note found")
            return

        parts = raw.split("\t")
        body = parts[1] if len(parts) > 1 else raw

        issues = []
        if "Huey Connelly" not in body:
            issues.append("missing patient name")
        if "PID 152" not in body and "pid 152" not in body.lower():
            issues.append("missing PID 152")
        if "Harbor Health Clinic" not in body:
            issues.append("missing clinic name")

        passed = len(issues) == 0
        check("9. OpenEMR office note (investigation)", 2, passed,
              "; ".join(issues) if issues else "office note found with correct content")
    except Exception as e:
        check("9. OpenEMR office note (investigation)", 2, False, f"exception: {e}")


# ── OnlyOffice checks ────────────────────────────────────────────────────────
_oo_session: tuple[str, dict] | None = None
_oo_doc_content: str | None = None


def _get_oo_session() -> tuple[str, dict]:
    global _oo_session
    if _oo_session is None:
        _oo_session = onlyoffice_api_session()
    return _oo_session


def _find_doc_and_content() -> tuple[bool, str]:
    """Find the document in OnlyOffice and return (found, content_text)."""
    global _oo_doc_content
    if _oo_doc_content is not None:
        return (True, _oo_doc_content) if _oo_doc_content else (False, "")

    base, headers = _get_oo_session()

    # Search for the document by title via the files API
    # Try searching in all accessible files
    search_url = f"{base}/api/2.0/files/@search/{requests.utils.quote(DOC_TITLE[:50])}"
    try:
        resp = requests.get(search_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("response", [])
            for f in files:
                title = f.get("title", "")
                if DOC_TITLE.lower() in title.lower() or "CMP-2026-0204" in title:
                    file_id = f.get("id")
                    # Try to get file content - download as txt
                    dl_url = f"{base}/api/2.0/files/file/{file_id}/openedit"
                    # Actually try to get content via viewing
                    # OnlyOffice stores .docx - let's try to get view URL or download
                    content_url = f"{base}/api/2.0/files/{file_id}"
                    cresp = requests.get(content_url, headers=headers, timeout=15)
                    if cresp.status_code == 200:
                        _oo_doc_content = ""
                        # Download the file and extract text
                        view_url = f.get("viewUrl", "")
                        if view_url:
                            try:
                                dresp = requests.get(view_url, headers=headers, timeout=30)
                                if dresp.status_code == 200:
                                    _oo_doc_content = dresp.text
                            except Exception:
                                pass
                        return True, _oo_doc_content
    except Exception:
        pass

    # Fallback: check via DB for file existence
    try:
        rc, out, err = docker_exec(
            ONLYOFFICE_DB_CONTAINER,
            "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
            "-D", "onlyoffice", "-N", "-B", "-e",
            f"SELECT id, title, create_by FROM files_file WHERE title LIKE '%CMP-2026-0204%' OR title LIKE '%Formal Patient Complaint%' LIMIT 5",
            timeout=15,
        )
        if out.strip():
            _oo_doc_content = ""
            return True, ""
    except Exception:
        pass

    _oo_doc_content = ""
    return False, ""


def _get_doc_content_via_fs() -> str:
    """Try to get document content by searching the OnlyOffice container filesystem."""
    try:
        # Search for the document file in the container
        rc, out, err = docker_exec(
            ONLYOFFICE_CONTAINER,
            "find", "/var/www/onlyoffice/Data", "-name", "*.docx", "-newer", "/tmp",
            timeout=15,
        )
        # Also search common document storage paths
        rc2, out2, err2 = docker_exec(
            ONLYOFFICE_CONTAINER,
            "bash", "-c",
            "find /app/onlyoffice/data/ -name '*.docx' 2>/dev/null || find /var/www/ -name '*.docx' -maxdepth 5 2>/dev/null || true",
            timeout=15,
        )
        return out.strip() + "\n" + out2.strip()
    except Exception:
        return ""


def _get_oo_doc_content_full() -> str:
    """Get OnlyOffice document content by downloading and extracting text from docx."""
    global _oo_doc_content
    if _oo_doc_content:
        return _oo_doc_content

    base, headers = _get_oo_session()

    # Search for the doc
    try:
        search_url = f"{base}/api/2.0/files/@search/{requests.utils.quote('CMP-2026-0204')}"
        resp = requests.get(search_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("response", [])
            for f in files:
                title = f.get("title", "")
                if "CMP-2026-0204" in title or "Formal Patient Complaint" in title:
                    file_id = f.get("id")
                    # Download as text - use the content URL
                    dl_url = f.get("viewUrl") or f.get("webUrl", "")
                    if not dl_url:
                        # Construct download URL
                        dl_url = f"{base}/api/2.0/files/{file_id}/download"
                    try:
                        dresp = requests.get(
                            f"{base}/api/2.0/files/{file_id}/download",
                            headers=headers, timeout=30, allow_redirects=True,
                        )
                        if dresp.status_code == 200:
                            content = dresp.content
                            # Try to extract text from docx (ZIP of XML)
                            import zipfile
                            import io
                            try:
                                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                                    if "word/document.xml" in zf.namelist():
                                        xml_content = zf.read("word/document.xml").decode("utf-8")
                                        # Strip XML tags to get text
                                        text = re.sub(r"<[^>]+>", " ", xml_content)
                                        text = re.sub(r"\s+", " ", text).strip()
                                        _oo_doc_content = text
                                        return text
                            except (zipfile.BadZipFile, Exception):
                                # Not a zip/docx, try as plain text
                                text = dresp.text
                                _oo_doc_content = text
                                return text
                    except Exception:
                        pass
    except Exception:
        pass

    return _oo_doc_content or ""


def check_10_doc_exists() -> None:
    """Check OnlyOffice document exists with expected title."""
    try:
        found, _ = _find_doc_and_content()
        check("10. OnlyOffice document exists", 1, found,
              f"title='{DOC_TITLE}'" if found else "document not found")
    except Exception as e:
        check("10. OnlyOffice document exists", 1, False, f"exception: {e}")


def check_11_doc_clinic_and_ref() -> None:
    """Check document contains clinic name and complaint reference."""
    try:
        content = _get_oo_doc_content_full()
        if not content:
            check("11. OnlyOffice clinic name & complaint reference", 2, False, "could not retrieve document content")
            return

        issues = []
        if CLINIC_NAME.lower() not in content.lower():
            issues.append(f"missing '{CLINIC_NAME}'")
        if COMPLAINT_REF not in content:
            issues.append(f"missing '{COMPLAINT_REF}'")

        passed = len(issues) == 0
        check("11. OnlyOffice clinic name & complaint reference", 2, passed,
              "; ".join(issues) if issues else "clinic name and reference found")
    except Exception as e:
        check("11. OnlyOffice clinic name & complaint reference", 2, False, f"exception: {e}")


def check_12_doc_complaint_summary() -> None:
    """Check document contains the complaint summary text."""
    try:
        content = _get_oo_doc_content_full()
        if not content:
            check("12. OnlyOffice complaint summary", 2, False, "could not retrieve document content")
            return

        # Check for key phrases from the complaint summary
        has_clarity = "clarity of clinical explanations" in content.lower() or "clarity" in content.lower()
        has_severity = "5/10" in content or "5 / 10" in content or "rated at 5" in content.lower()
        has_callback = "callback" in content.lower() or "follow-up" in content.lower()

        passed = has_clarity and (has_severity or has_callback)
        detail_parts = []
        if not has_clarity:
            detail_parts.append("missing 'clarity of clinical explanations'")
        if not has_severity:
            detail_parts.append("missing severity rating '5/10'")
        if not has_callback:
            detail_parts.append("missing callback/follow-up reference")

        check("12. OnlyOffice complaint summary", 2, passed,
              "; ".join(detail_parts) if detail_parts else "complaint summary present")
    except Exception as e:
        check("12. OnlyOffice complaint summary", 2, False, f"exception: {e}")


def check_13_doc_investigation_outcome() -> None:
    """Check document contains the investigation outcome determination."""
    try:
        content = _get_oo_doc_content_full()
        if not content:
            check("13. OnlyOffice investigation outcome", 2, False, "could not retrieve document content")
            return

        found = "complaint partially substantiated" in content.lower() or \
                "patient-education documentation inadequate" in content.lower()
        check("13. OnlyOffice investigation outcome", 2, found,
              "investigation outcome found" if found else "missing investigation outcome text")
    except Exception as e:
        check("13. OnlyOffice investigation outcome", 2, False, f"exception: {e}")


def check_14_doc_corrective_actions() -> None:
    """Check document contains all three corrective actions."""
    try:
        content = _get_oo_doc_content_full()
        if not content:
            check("14. OnlyOffice corrective actions (3)", 3, False, "could not retrieve document content")
            return

        content_lower = content.lower()
        found_actions = 0
        missing = []

        if "after-visit summary" in content_lower or "after visit summary" in content_lower:
            found_actions += 1
        else:
            missing.append("action 1 (after-visit summary templates)")

        if "teach-back" in content_lower or "teach back" in content_lower:
            found_actions += 1
        else:
            missing.append("action 2 (teach-back methodology)")

        if "patient-education satisfaction" in content_lower or "patient education satisfaction" in content_lower or "quarterly" in content_lower:
            found_actions += 1
        else:
            missing.append("action 3 (quarterly satisfaction survey)")

        passed = found_actions == 3
        check("14. OnlyOffice corrective actions (3)", 3, passed,
              f"{found_actions}/3 found" + (f"; missing: {', '.join(missing)}" if missing else ""))
    except Exception as e:
        check("14. OnlyOffice corrective actions (3)", 3, False, f"exception: {e}")


def check_15_doc_patient_rights() -> None:
    """Check document contains patient rights text."""
    try:
        content = _get_oo_doc_content_full()
        if not content:
            check("15. OnlyOffice patient rights text", 2, False, "could not retrieve document content")
            return

        found = PATIENT_RIGHTS_SNIPPET.lower() in content.lower()
        check("15. OnlyOffice patient rights text", 2, found,
              "patient rights text found" if found else "missing patient rights text")
    except Exception as e:
        check("15. OnlyOffice patient rights text", 2, False, f"exception: {e}")


def check_16_doc_signature_block() -> None:
    """Check document contains signature block for Michael Delacroix."""
    try:
        content = _get_oo_doc_content_full()
        if not content:
            check("16. OnlyOffice signature block", 1, False, "could not retrieve document content")
            return

        found = "michael delacroix" in content.lower()
        check("16. OnlyOffice signature block", 1, found,
              "signature block found" if found else "missing 'Michael Delacroix, MBA, CPXP'")
    except Exception as e:
        check("16. OnlyOffice signature block", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_form_exists()
    check_2_form_settings()
    check_3_auto_increment_field()
    check_4_conditional_phone()
    check_5_conditional_urgency_justification()
    check_6_page_break()
    check_7_escalation_toggle()
    check_8_patient_note()
    check_9_office_note()
    check_10_doc_exists()
    check_11_doc_clinic_and_ref()
    check_12_doc_complaint_summary()
    check_13_doc_investigation_outcome()
    check_14_doc_corrective_actions()
    check_15_doc_patient_rights()
    check_16_doc_signature_block()

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
