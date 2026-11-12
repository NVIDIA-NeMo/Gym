# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_021/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-021-I2: Pediatric Well-Child Visit Workflow with Milestone Screening

Checks: 14 weighted checks across opnform, openemr, onlyoffice.
Strategy: API for OpnForm & OnlyOffice; docker exec MariaDB for OpenEMR.

Required env vars:
  SERVER_HOSTNAME, OPNFORM_PORT, OPNFORM_CONTAINER,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import io
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

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

for _var in [
    "OPNFORM_PORT", "OPNFORM_CONTAINER",
    "OPENEMR_PORT", "OPENEMR_CONTAINER", "OPENEMR_DB_CONTAINER",
    "ONLYOFFICE_PORT", "ONLYOFFICE_CONTAINER", "ONLYOFFICE_DB_CONTAINER",
]:
    if not os.environ.get(_var):
        print(f"FATAL: {_var} not set", file=sys.stderr)
        sys.exit(1)

OPNFORM_BASE = f"http://{HOST}:{OPNFORM_PORT}"
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


def openemr_db(sql: str) -> str:
    """Query OpenEMR MariaDB."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "-u", "openemr", "-popenemr_pass", "-D", "openemr",
        "--default-character-set=utf8mb4", "-N", "-e", sql,
        timeout=15,
    )
    return out.strip()


def get_patient_pid(fname: str, lname: str) -> str:
    return openemr_db(
        f"SELECT pid FROM patient_data WHERE fname='{fname}' AND lname='{lname}' LIMIT 1"
    ).strip()


# ── OpnForm Checks ───────────────────────────────────────────────────────────
_opnform_form = None


def _opnform_login() -> dict:
    r = requests.post(
        f"{OPNFORM_BASE}/api/login",
        json={"email": "seeded_admin@example.com", "password": "mw-admin-123"},
        timeout=15,
    )
    r.raise_for_status()
    token = r.json().get("token", "")
    return {"Authorization": f"Bearer {token}"}


def _find_opnform_form(headers: dict) -> dict | None:
    global _opnform_form
    if _opnform_form is not None:
        return _opnform_form

    r = requests.get(f"{OPNFORM_BASE}/api/open/forms", headers=headers, timeout=15)
    r.raise_for_status()
    forms_list = r.json()
    if isinstance(forms_list, dict):
        forms_list = forms_list.get("data", forms_list.get("response", []))
    if isinstance(forms_list, list):
        for f in forms_list:
            if f.get("title") == "Early Childhood Developmental Screening Form Q2-2026":
                _opnform_form = f
                return f
    return None


def check_1_opnform_form_exists() -> None:
    """Form exists with correct title and public visibility."""
    try:
        headers = _opnform_login()
        form = _find_opnform_form(headers)
        if not form:
            check("1. OpnForm form exists and is public", 2, False, "form not found")
            return
        vis = form.get("visibility", "")
        check("1. OpnForm form exists and is public", 2, vis == "public",
              f"visibility={vis}")
    except Exception as e:
        check("1. OpnForm form exists and is public", 2, False, f"exception: {e}")


def check_2_opnform_fields() -> None:
    """Form has all 13 required fields with correct types."""
    try:
        form = _opnform_form
        if not form:
            check("2. OpnForm required fields present", 2, False, "no form data")
            return

        props = form.get("properties", [])
        expected_names = [
            "Child's Name", "Date of Birth", "Age in Months",
            "Parent/Guardian Name", "Age Group", "Developmental Milestones",
            "Milestones Achieved Count", "Parent Concern Level",
            "Referral Requested by Parent", "Referral Reason",
            "Current Weight", "Current Height", "Head Circumference",
        ]
        prop_names_lower = [p.get("name", "").lower() for p in props]
        found = 0
        missing = []
        for en in expected_names:
            if any(en.lower() in pn for pn in prop_names_lower):
                found += 1
            else:
                missing.append(en)
        passed = found >= 10
        detail = f"found {found}/13"
        if missing:
            detail += f", missing: {missing[:4]}"
        check("2. OpnForm required fields present", 2, passed, detail)
    except Exception as e:
        check("2. OpnForm required fields present", 2, False, f"exception: {e}")


def check_3_opnform_matrix() -> None:
    """Matrix field 'Developmental Milestones' has correct rows and columns."""
    try:
        form = _opnform_form
        if not form:
            check("3. OpnForm matrix field config", 2, False, "no form data")
            return

        props = form.get("properties", [])
        matrix = None
        for p in props:
            if p.get("type") == "matrix":
                matrix = p
                break
        if not matrix:
            check("3. OpnForm matrix field config", 2, False, "no matrix field found")
            return

        def extract_names(items):
            names = []
            for item in (items or []):
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    names.append(item.get("name", item.get("label", item.get("value", ""))))
            return [n.lower() for n in names]

        row_names = extract_names(matrix.get("rows", []))
        col_names = extract_names(matrix.get("columns", []))

        exp_rows = ["walks independently", "stacks three or more blocks",
                     "uses two-word phrases", "points to named body parts",
                     "follows simple instructions", "scribbles with crayon"]
        exp_cols = ["achieved", "emerging", "not yet", "unable to assess"]

        rows_ok = sum(1 for er in exp_rows if any(er in rn for rn in row_names)) >= 5
        cols_ok = sum(1 for ec in exp_cols if any(ec in cn for cn in col_names)) >= 3
        check("3. OpnForm matrix field config", 2, rows_ok and cols_ok,
              f"rows={'OK' if rows_ok else 'MISSING'}({len(row_names)}), "
              f"cols={'OK' if cols_ok else 'MISSING'}({len(col_names)})")
    except Exception as e:
        check("3. OpnForm matrix field config", 2, False, f"exception: {e}")


def check_4_opnform_styling() -> None:
    """Form color=#2196F3, theme=default, border_radius=small, auto_save=on."""
    try:
        form = _opnform_form
        if not form:
            check("4. OpnForm form styling", 1, False, "no form data")
            return

        issues = []
        color = (form.get("color") or "").upper()
        if color != "#2196F3":
            issues.append(f"color={color}")
        if form.get("theme") != "default":
            issues.append(f"theme={form.get('theme')}")
        if form.get("border_radius") != "small":
            issues.append(f"border_radius={form.get('border_radius')}")
        if not form.get("auto_save"):
            issues.append("auto_save off")

        check("4. OpnForm form styling", 1, len(issues) == 0,
              "; ".join(issues) if issues else "all correct")
    except Exception as e:
        check("4. OpnForm form styling", 1, False, f"exception: {e}")


# ── OpenEMR — Connie McLaughlin ──────────────────────────────────────────────
def check_5_connie_family_history() -> None:
    """Family history updated with maternal aunt asthma etc."""
    try:
        pid = get_patient_pid("Connie", "McLaughlin")
        if not pid:
            check("5. Connie family history", 1, False, "patient not found")
            return
        # Family history is stored across multiple columns in history_data
        result = openemr_db(
            f"SELECT CONCAT_WS(' | ', history_mother, history_father, history_siblings, "
            f"history_offspring, history_spouse, additional_history, "
            f"relatives_cancer, relatives_diabetes, relatives_heart_problems) "
            f"FROM history_data WHERE pid='{pid}' ORDER BY id DESC LIMIT 1"
        )
        expected_fragment = "maternal aunt with asthma"
        passed = expected_fragment.lower() in result.lower() if result else False
        # If not found in concat, try a broader search
        if not passed and result:
            all_cols = openemr_db(
                f"SELECT * FROM history_data WHERE pid='{pid}' ORDER BY id DESC LIMIT 1"
            )
            passed = expected_fragment.lower() in all_cols.lower() if all_cols else False
        check("5. Connie family history", 1, passed,
              f"got: {result[:100]}" if result else "empty")
    except Exception as e:
        check("5. Connie family history", 1, False, f"exception: {e}")


def check_6_connie_vitals() -> None:
    """Vitals: weight 11.2, height 84.1, head_circ 47.0, temp 98.2."""
    try:
        pid = get_patient_pid("Connie", "McLaughlin")
        if not pid:
            check("6. Connie vitals", 2, False, "patient not found")
            return
        result = openemr_db(
            f"SELECT weight, height, head_circ, temperature FROM form_vitals "
            f"WHERE pid='{pid}' ORDER BY id DESC LIMIT 1"
        )
        if not result:
            check("6. Connie vitals", 2, False, "no vitals row")
            return
        parts = result.split("\t")
        issues = []
        if len(parts) >= 4:
            if "11.2" not in parts[0]:
                issues.append(f"weight={parts[0]}")
            if "84.1" not in parts[1]:
                issues.append(f"height={parts[1]}")
            if "47" not in parts[2]:
                issues.append(f"head_circ={parts[2]}")
            if "98.2" not in parts[3]:
                issues.append(f"temp={parts[3]}")
        else:
            issues.append(f"unexpected cols: {len(parts)}")
        check("6. Connie vitals", 2, len(issues) == 0,
              "; ".join(issues) if issues else "all match")
    except Exception as e:
        check("6. Connie vitals", 2, False, f"exception: {e}")


def check_8_connie_icd10() -> None:
    """ICD-10 Z00.129 on billing."""
    try:
        pid = get_patient_pid("Connie", "McLaughlin")
        if not pid:
            check("8. Connie ICD-10 Z00.129", 1, False, "patient not found")
            return
        result = openemr_db(
            f"SELECT code FROM billing WHERE pid='{pid}' AND code='Z00.129' LIMIT 1"
        )
        passed = "Z00.129" in result if result else False
        check("8. Connie ICD-10 Z00.129", 1, passed,
              f"found: {result}" if result else "not found in billing")
    except Exception as e:
        check("8. Connie ICD-10 Z00.129", 1, False, f"exception: {e}")


def check_9_connie_care_plan() -> None:
    """Care plan with goal about language development."""
    try:
        pid = get_patient_pid("Connie", "McLaughlin")
        if not pid:
            check("9. Connie care plan", 2, False, "patient not found")
            return
        # Check form_care_plan table
        result = openemr_db(
            f"SELECT care_plan_type, description FROM form_care_plan "
            f"WHERE pid='{pid}' ORDER BY id DESC LIMIT 5"
        )
        if not result:
            # Fallback: check forms table for care plan entry
            result = openemr_db(
                f"SELECT form_name FROM forms WHERE pid='{pid}' "
                f"AND (formdir LIKE '%care%plan%' OR form_name LIKE '%Care Plan%') "
                f"ORDER BY id DESC LIMIT 1"
            )
        expected = "language development"
        passed = expected.lower() in result.lower() if result else False
        check("9. Connie care plan", 2, passed,
              f"got: {result[:120]}" if result else "no care plan data")
    except Exception as e:
        check("9. Connie care plan", 2, False, f"exception: {e}")


# ── OpenEMR — Lekisha Bosco ──────────────────────────────────────────────────
def check_10_lekisha_vitals() -> None:
    """Vitals: weight 14.2, height 95.5, head_circ 49.1, temp 99.0."""
    try:
        pid = get_patient_pid("Lekisha", "Bosco")
        if not pid:
            check("10. Lekisha vitals", 2, False, "patient not found")
            return
        result = openemr_db(
            f"SELECT weight, height, head_circ, temperature FROM form_vitals "
            f"WHERE pid='{pid}' ORDER BY id DESC LIMIT 1"
        )
        if not result:
            check("10. Lekisha vitals", 2, False, "no vitals row")
            return
        parts = result.split("\t")
        issues = []
        if len(parts) >= 4:
            if "14.2" not in parts[0]:
                issues.append(f"weight={parts[0]}")
            if "95.5" not in parts[1]:
                issues.append(f"height={parts[1]}")
            if "49.1" not in parts[2]:
                issues.append(f"head_circ={parts[2]}")
            if "99" not in parts[3]:
                issues.append(f"temp={parts[3]}")
        else:
            issues.append(f"unexpected cols: {len(parts)}")
        check("10. Lekisha vitals", 2, len(issues) == 0,
              "; ".join(issues) if issues else "all match")
    except Exception as e:
        check("10. Lekisha vitals", 2, False, f"exception: {e}")


def check_11_lekisha_immunization() -> None:
    """MMR immunization: manufacturer Merck, lot LOT52918, route Subcutaneous."""
    try:
        pid = get_patient_pid("Lekisha", "Bosco")
        if not pid:
            check("11. Lekisha MMR immunization", 2, False, "patient not found")
            return
        result = openemr_db(
            f"SELECT immunization_id, manufacturer, lot_number, route "
            f"FROM immunizations WHERE patient_id='{pid}' "
            f"ORDER BY immunization_id DESC LIMIT 5"
        )
        if not result:
            check("11. Lekisha MMR immunization", 2, False, "no immunization records")
            return
        issues = []
        result_lower = result.lower()
        if "merck" not in result_lower:
            issues.append("manufacturer not Merck")
        if "lot52918" not in result_lower:
            issues.append("lot not LOT52918")
        if "subcutaneous" not in result_lower:
            issues.append("route not Subcutaneous")
        check("11. Lekisha MMR immunization", 2, len(issues) == 0,
              "; ".join(issues) if issues else "MMR record found with correct details")
    except Exception as e:
        check("11. Lekisha MMR immunization", 2, False, f"exception: {e}")


def check_13_lekisha_icd10() -> None:
    """ICD-10 Z00.129 on billing."""
    try:
        pid = get_patient_pid("Lekisha", "Bosco")
        if not pid:
            check("13. Lekisha ICD-10 Z00.129", 1, False, "patient not found")
            return
        result = openemr_db(
            f"SELECT code FROM billing WHERE pid='{pid}' AND code='Z00.129' LIMIT 1"
        )
        passed = "Z00.129" in result if result else False
        check("13. Lekisha ICD-10 Z00.129", 1, passed,
              f"found: {result}" if result else "not found in billing")
    except Exception as e:
        check("13. Lekisha ICD-10 Z00.129", 1, False, f"exception: {e}")


def check_14_lekisha_clinical_note() -> None:
    """Clinical note about 32-month well-child visit."""
    try:
        pid = get_patient_pid("Lekisha", "Bosco")
        if not pid:
            check("14. Lekisha clinical note", 2, False, "patient not found")
            return
        # Try form_clinical_notes
        result = openemr_db(
            f"SELECT description FROM form_clinical_notes "
            f"WHERE pid='{pid}' ORDER BY id DESC LIMIT 1"
        )
        if not result:
            # Fallback: check forms table
            result = openemr_db(
                f"SELECT form_name, form_id FROM forms WHERE pid='{pid}' "
                f"AND (formdir LIKE '%clinical%' OR form_name LIKE '%Clinical%') "
                f"ORDER BY id DESC LIMIT 1"
            )
        expected = "32-month well-child visit"
        passed = expected.lower() in result.lower() if result else False
        check("14. Lekisha clinical note", 2, passed,
              f"got: {result[:120]}" if result else "no clinical note")
    except Exception as e:
        check("14. Lekisha clinical note", 2, False, f"exception: {e}")


# ── OnlyOffice Checks ────────────────────────────────────────────────────────
def _onlyoffice_auth() -> dict:
    """Authenticate to OnlyOffice API, return headers."""
    r = requests.post(
        f"{ONLYOFFICE_BASE}/api/2.0/authentication",
        json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
        timeout=15,
    )
    r.raise_for_status()
    token = r.json().get("response", {}).get("token", "")
    return {"Authorization": token}


def _onlyoffice_find_file(headers: dict, title: str) -> dict | None:
    """Search for a file by title in OnlyOffice."""
    # Try searching via API
    r = requests.get(
        f"{ONLYOFFICE_BASE}/api/2.0/files/@search/{title}",
        headers=headers, timeout=15,
    )
    if r.ok:
        items = r.json().get("response", [])
        for item in items:
            if item.get("title", "") == title or title.lower() in item.get("title", "").lower():
                return item
    # Fallback: list My Documents
    r = requests.get(
        f"{ONLYOFFICE_BASE}/api/2.0/files/@my",
        headers=headers, timeout=15,
    )
    if r.ok:
        resp = r.json().get("response", {})
        files = resp.get("files", []) if isinstance(resp, dict) else []
        for f in files:
            if title.lower() in f.get("title", "").lower():
                return f
    return None


def check_15_onlyoffice_spreadsheet_exists() -> None:
    """Spreadsheet 'Pediatric Well-Child Visit Tracker Q2-2026' exists."""
    try:
        headers = _onlyoffice_auth()
        title = "Pediatric Well-Child Visit Tracker Q2-2026"
        f = _onlyoffice_find_file(headers, title)
        check("15. OnlyOffice spreadsheet exists", 1, f is not None,
              f"found: {f.get('title')}" if f else "not found")
    except Exception as e:
        check("15. OnlyOffice spreadsheet exists", 1, False, f"exception: {e}")


def check_16_onlyoffice_spreadsheet_sheets() -> None:
    """Spreadsheet has 3 sheets: Growth Vitals, Milestone Status, Visit Summary."""
    try:
        headers = _onlyoffice_auth()
        title = "Pediatric Well-Child Visit Tracker Q2-2026"
        f = _onlyoffice_find_file(headers, title)
        if not f:
            check("16. OnlyOffice spreadsheet sheets", 2, False, "file not found")
            return

        file_id = f.get("id")
        # Download the file
        dl_url = f"{ONLYOFFICE_BASE}/api/2.0/files/file/{file_id}/download"
        r = requests.get(dl_url, headers=headers, timeout=30)
        if not r.ok:
            check("16. OnlyOffice spreadsheet sheets", 2, False,
                  f"download failed: {r.status_code}")
            return

        # Parse xlsx (zip) to extract sheet names from workbook.xml
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            wb_xml = ET.parse(z.open("xl/workbook.xml"))
            sheets = [s.get("name") for s in wb_xml.findall(f".//{{{ns}}}sheet")]

        expected = ["Growth Vitals", "Milestone Status", "Visit Summary"]
        sheet_lower = [s.lower() for s in sheets]
        found = [e for e in expected if e.lower() in sheet_lower]
        passed = len(found) == 3
        check("16. OnlyOffice spreadsheet sheets", 2, passed,
              f"sheets={sheets}, expected={expected}")
    except Exception as e:
        check("16. OnlyOffice spreadsheet sheets", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_opnform_form_exists()
    check_2_opnform_fields()
    check_3_opnform_matrix()
    check_4_opnform_styling()
    check_5_connie_family_history()
    check_6_connie_vitals()
    check_8_connie_icd10()
    check_9_connie_care_plan()
    check_10_lekisha_vitals()
    check_11_lekisha_immunization()
    check_13_lekisha_icd10()
    check_14_lekisha_clinical_note()
    check_15_onlyoffice_spreadsheet_exists()
    check_16_onlyoffice_spreadsheet_sheets()

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
