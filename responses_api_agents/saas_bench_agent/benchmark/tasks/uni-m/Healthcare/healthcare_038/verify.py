# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_038/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-038-I1: HAI Surveillance — Exposure Form, Infection Documentation, Monthly Report

Checks: 13 weighted checks across opnform, openemr, onlyoffice.
Strategy: docker exec (DB queries) for all three sites.

Required env vars:
  SERVER_HOSTNAME,
  OPNFORM_PORT, OPNFORM_CONTAINER,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import json
import os
import subprocess
import sys

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
for _k, _v in _required.items():
    if not _v:
        print(f"FATAL: {_k} not set", file=sys.stderr)
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


def opnform_sql(query: str) -> str:
    """Query OpnForm's embedded Postgres (forge DB, forge user)."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "psql", "-U", "forge", "-d", "forge", "-t", "-A", "-c", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"opnform psql error: {err.strip()}")
    return out.strip()


def openemr_sql(query: str) -> str:
    """Query OpenEMR's MariaDB."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "openemr", "-popenemr_pass", "-D", "openemr",
        "-N", "-B", "-e", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"openemr mysql error: {err.strip()}")
    return out.strip()


def onlyoffice_sql(query: str) -> str:
    """Query OnlyOffice's MySQL 8.0."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "onlyoffice_user", "-ponlyoffice_pass", "-D", "onlyoffice",
        "-N", "-B", "-e", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"onlyoffice mysql error: {err.strip()}")
    return out.strip()


# ── Individual checks ─────────────────────────────────────────────────────────

# ---------- OpnForm ----------

def check_1_opnform_form_exists() -> None:
    """Form 'HAI Staff Exposure Incident Report - March 2026' exists."""
    try:
        row = opnform_sql(
            "SELECT id, title FROM forms "
            "WHERE title = 'HAI Staff Exposure Incident Report - March 2026' "
            "AND deleted_at IS NULL LIMIT 1;"
        )
        found = bool(row)
        check("1. OpnForm form exists with correct title", 1, found,
              f"got: {row!r}" if not found else "")
    except Exception as e:
        check("1. OpnForm form exists with correct title", 1, False, f"exception: {e}")


def check_2_opnform_form_settings() -> None:
    """Form theme='simple', color='#1E6091', presentation_style='classic',
    visibility='public', auto_save=true, submit_button_text='Submit Exposure Report'."""
    try:
        row = opnform_sql(
            "SELECT theme, color, presentation_style, visibility, auto_save, submit_button_text "
            "FROM forms "
            "WHERE title = 'HAI Staff Exposure Incident Report - March 2026' "
            "AND deleted_at IS NULL LIMIT 1;"
        )
        if not row:
            check("2. OpnForm form settings", 2, False, "form not found")
            return
        parts = row.split("|")
        if len(parts) < 6:
            check("2. OpnForm form settings", 2, False, f"unexpected row format: {row!r}")
            return
        theme, color, pres, vis, auto_save, submit_btn = [p.strip() for p in parts[:6]]
        errors = []
        if theme != "simple":
            errors.append(f"theme={theme!r}")
        if color.upper() != "#1E6091":
            errors.append(f"color={color!r}")
        if pres != "classic":
            errors.append(f"presentation_style={pres!r}")
        if vis != "public":
            errors.append(f"visibility={vis!r}")
        if auto_save not in ("t", "1", "true", "True"):
            errors.append(f"auto_save={auto_save!r}")
        if submit_btn != "Submit Exposure Report":
            errors.append(f"submit_button_text={submit_btn!r}")
        check("2. OpnForm form settings", 2, not errors,
              "; ".join(errors) if errors else "")
    except Exception as e:
        check("2. OpnForm form settings", 2, False, f"exception: {e}")


def check_3_opnform_closed_message() -> None:
    """closed_text matches expected message."""
    expected = ("This exposure reporting form is currently closed. "
                "For urgent exposures, please contact Employee Health Services "
                "directly at ext. 4357.")
    try:
        row = opnform_sql(
            "SELECT closed_text FROM forms "
            "WHERE title = 'HAI Staff Exposure Incident Report - March 2026' "
            "AND deleted_at IS NULL LIMIT 1;"
        )
        if not row:
            check("3. OpnForm closed message", 1, False, "form not found")
            return
        passed = expected in row
        check("3. OpnForm closed message", 1, passed,
              f"got: {row[:80]!r}..." if not passed else "")
    except Exception as e:
        check("3. OpnForm closed message", 1, False, f"exception: {e}")


def check_4_opnform_field_types() -> None:
    """Form properties contain required field types: text, select, multi_select,
    checkbox, date, number, rating, signature, nf-page-break, rich_text."""
    try:
        row = opnform_sql(
            "SELECT properties FROM forms "
            "WHERE title = 'HAI Staff Exposure Incident Report - March 2026' "
            "AND deleted_at IS NULL LIMIT 1;"
        )
        if not row:
            check("4. OpnForm field types present", 2, False, "form not found")
            return
        props = json.loads(row)
        types_present = {p.get("type", "") for p in props}
        # Also check for name matches
        names_present = {p.get("name", "").lower() for p in props}
        labels_present = {(p.get("label") or p.get("name") or "").lower() for p in props}

        required_types = {"text", "select", "date", "number", "checkbox"}
        # multi_select might be stored as "multi_select" or "multiselect"
        has_multi = any(t in types_present for t in ("multi_select", "multiselect", "multi-select"))
        has_rating = "rating" in types_present or "scale" in types_present
        has_signature = "signature" in types_present or any("signature" in l for l in labels_present)
        has_page_break = any(t in types_present for t in ("nf-page-break", "page_break", "pagebreak"))
        has_rich_text = any(t in types_present for t in ("rich_text", "richtext", "rich-text"))

        missing_types = required_types - types_present
        missing = []
        if missing_types:
            missing.append(f"missing types: {missing_types}")
        if not has_multi:
            missing.append("no multi_select field")
        if not has_rating:
            missing.append("no rating field")
        if not has_signature:
            missing.append("no signature field")
        if not has_page_break:
            missing.append("no page break")
        if not has_rich_text:
            missing.append("no rich_text field")

        check("4. OpnForm field types present", 2, not missing,
              "; ".join(missing) if missing else f"found {len(props)} fields")
    except Exception as e:
        check("4. OpnForm field types present", 2, False, f"exception: {e}")


def check_5_opnform_conditional_fields() -> None:
    """Conditional visibility on 'Other Exposure Details' (when Exposure Type='Other')
    and 'Source Patient Identifier' (when Source Patient Known='Yes')."""
    try:
        row = opnform_sql(
            "SELECT properties FROM forms "
            "WHERE title = 'HAI Staff Exposure Incident Report - March 2026' "
            "AND deleted_at IS NULL LIMIT 1;"
        )
        if not row:
            check("5. OpnForm conditional fields", 2, False, "form not found")
            return
        props = json.loads(row)
        found_other_cond = False
        found_source_cond = False
        for p in props:
            label = (p.get("label") or p.get("name") or "").lower()
            # Check for conditional logic — OpnForm stores it in various ways
            has_logic = bool(p.get("logic") or p.get("conditionalLogic") or p.get("hidden"))
            # Also check if the field is conditionally shown
            if "other exposure" in label or "other_exposure" in label:
                found_other_cond = has_logic or bool(p.get("conditions"))
            if "source patient identifier" in label or "source_patient_identifier" in label:
                found_source_cond = has_logic or bool(p.get("conditions"))

        errors = []
        if not found_other_cond:
            errors.append("'Other Exposure Details' conditional not found/configured")
        if not found_source_cond:
            errors.append("'Source Patient Identifier' conditional not found/configured")
        check("5. OpnForm conditional fields", 2, not errors,
              "; ".join(errors) if errors else "")
    except Exception as e:
        check("5. OpnForm conditional fields", 2, False, f"exception: {e}")


# ---------- OpenEMR ----------

def _get_patient_pid(patient_name: str) -> str:
    """Get pid for a patient by name parts."""
    parts = patient_name.split()
    fname, lname = parts[0], parts[-1]
    return openemr_sql(
        f"SELECT pid FROM patient_data "
        f"WHERE fname = '{fname}' AND lname = '{lname}' LIMIT 1;"
    )


def check_6_openemr_carolyne_encounter_notes() -> None:
    """Carolyne Schuster: encounter exists with Clinical Notes containing MRSA content."""
    try:
        pid = _get_patient_pid("Carolyne Schuster")
        if not pid:
            check("6. OpenEMR Carolyne encounter + clinical notes", 2, False,
                  "patient Carolyne Schuster not found")
            return
        notes = openemr_sql(
            f"SELECT description FROM form_clinical_notes "
            f"WHERE pid = {pid} AND activity = 1 "
            f"ORDER BY id DESC LIMIT 5;"
        )
        has_mrsa = "MRSA" in notes and "vancomycin" in notes.lower()
        check("6. OpenEMR Carolyne encounter + clinical notes", 2, has_mrsa,
              "MRSA clinical note found" if has_mrsa else f"notes: {notes[:100]!r}")
    except Exception as e:
        check("6. OpenEMR Carolyne encounter + clinical notes", 2, False, f"exception: {e}")


def check_7_openemr_hipolito_encounter_notes() -> None:
    """Hipolito Heller: encounter with Clinical Notes about C. diff."""
    try:
        pid = _get_patient_pid("Hipolito Heller")
        if not pid:
            check("7. OpenEMR Hipolito encounter + clinical notes", 2, False,
                  "patient Hipolito Heller not found")
            return
        notes = openemr_sql(
            f"SELECT description FROM form_clinical_notes "
            f"WHERE pid = {pid} AND activity = 1 "
            f"ORDER BY id DESC LIMIT 5;"
        )
        has_cdiff = "difficile" in notes.lower() or "c. diff" in notes.lower()
        check("7. OpenEMR Hipolito encounter + clinical notes", 2, has_cdiff,
              "C. diff clinical note found" if has_cdiff else f"notes: {notes[:100]!r}")
    except Exception as e:
        check("7. OpenEMR Hipolito encounter + clinical notes", 2, False, f"exception: {e}")


def check_8_openemr_carolyne_icd10() -> None:
    """Carolyne Schuster: billing has ICD-10 A49.02."""
    try:
        pid = _get_patient_pid("Carolyne Schuster")
        if not pid:
            check("8. OpenEMR Carolyne ICD-10 A49.02", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT code FROM billing "
            f"WHERE pid = {pid} AND code_type = 'ICD10' AND code = 'A49.02' "
            f"AND activity = 1 LIMIT 1;"
        )
        check("8. OpenEMR Carolyne ICD-10 A49.02", 1, bool(row),
              "" if row else "A49.02 not found in billing")
    except Exception as e:
        check("8. OpenEMR Carolyne ICD-10 A49.02", 1, False, f"exception: {e}")


def check_9_openemr_hipolito_icd10() -> None:
    """Hipolito Heller: billing has ICD-10 A04.7."""
    try:
        pid = _get_patient_pid("Hipolito Heller")
        if not pid:
            check("9. OpenEMR Hipolito ICD-10 A04.7", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT code FROM billing "
            f"WHERE pid = {pid} AND code_type = 'ICD10' AND code = 'A04.7' "
            f"AND activity = 1 LIMIT 1;"
        )
        check("9. OpenEMR Hipolito ICD-10 A04.7", 1, bool(row),
              "" if row else "A04.7 not found in billing")
    except Exception as e:
        check("9. OpenEMR Hipolito ICD-10 A04.7", 1, False, f"exception: {e}")


def check_10_openemr_carolyne_problem() -> None:
    """Carolyne Schuster: medical problem 'MRSA bloodstream infection (healthcare-associated)'."""
    try:
        pid = _get_patient_pid("Carolyne Schuster")
        if not pid:
            check("10. OpenEMR Carolyne medical problem", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT title FROM lists "
            f"WHERE pid = {pid} AND type = 'medical_problem' "
            f"AND title LIKE '%MRSA%bloodstream%' "
            f"AND activity = 1 LIMIT 1;"
        )
        check("10. OpenEMR Carolyne medical problem", 1, bool(row),
              "" if row else "MRSA bloodstream problem not found in lists")
    except Exception as e:
        check("10. OpenEMR Carolyne medical problem", 1, False, f"exception: {e}")


def check_11_openemr_hipolito_problem() -> None:
    """Hipolito Heller: medical problem 'Clostridioides difficile enterocolitis'."""
    try:
        pid = _get_patient_pid("Hipolito Heller")
        if not pid:
            check("11. OpenEMR Hipolito medical problem", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT title FROM lists "
            f"WHERE pid = {pid} AND type = 'medical_problem' "
            f"AND title LIKE '%difficile%enterocolitis%' "
            f"AND activity = 1 LIMIT 1;"
        )
        check("11. OpenEMR Hipolito medical problem", 1, bool(row),
              "" if row else "C. diff enterocolitis problem not found in lists")
    except Exception as e:
        check("11. OpenEMR Hipolito medical problem", 1, False, f"exception: {e}")


def check_12_openemr_procedure_order() -> None:
    """Carolyne Schuster: Procedure Order for 'Complete Blood Count', priority High."""
    try:
        pid = _get_patient_pid("Carolyne Schuster")
        if not pid:
            check("12. OpenEMR procedure order CBC High", 2, False, "patient not found")
            return
        # Check procedure_order joined with procedure_order_code
        row = openemr_sql(
            f"SELECT po.order_priority, poc.procedure_name "
            f"FROM procedure_order po "
            f"JOIN procedure_order_code poc ON po.procedure_order_id = poc.procedure_order_id "
            f"WHERE po.patient_id = {pid} "
            f"AND poc.procedure_name LIKE '%Complete Blood Count%' "
            f"LIMIT 1;"
        )
        if not row:
            # Try broader search
            row = openemr_sql(
                f"SELECT po.order_priority, poc.procedure_name "
                f"FROM procedure_order po "
                f"JOIN procedure_order_code poc ON po.procedure_order_id = poc.procedure_order_id "
                f"WHERE po.patient_id = {pid} "
                f"AND (poc.procedure_name LIKE '%CBC%' OR poc.procedure_name LIKE '%Blood Count%') "
                f"LIMIT 1;"
            )
        if not row:
            check("12. OpenEMR procedure order CBC High", 2, False,
                  "no CBC procedure order found")
            return
        parts = row.split("\t")
        priority = parts[0].strip().lower() if parts else ""
        is_high = priority in ("high", "high priority")
        check("12. OpenEMR procedure order CBC High", 2, is_high,
              "" if is_high else f"priority={parts[0]!r}")
    except Exception as e:
        check("12. OpenEMR procedure order CBC High", 2, False, f"exception: {e}")


# ---------- OnlyOffice ----------

def check_13_onlyoffice_spreadsheet_exists() -> None:
    """Spreadsheet 'HAI Surveillance Report - March 2026' exists."""
    try:
        row = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%HAI Surveillance Report%March 2026%' "
            "AND current_version = 1 "
            "LIMIT 1;"
        )
        if not row:
            # Broader search: may have .xlsx extension
            row = onlyoffice_sql(
                "SELECT id, title FROM files_file "
                "WHERE title LIKE '%HAI Surveillance%2026%' "
                "AND current_version = 1 "
                "LIMIT 1;"
            )
        check("13. OnlyOffice spreadsheet exists", 1, bool(row),
              "" if row else "spreadsheet not found")
    except Exception as e:
        check("13. OnlyOffice spreadsheet exists", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_opnform_form_exists()
    check_2_opnform_form_settings()
    check_3_opnform_closed_message()
    check_4_opnform_field_types()
    check_5_opnform_conditional_fields()
    check_6_openemr_carolyne_encounter_notes()
    check_7_openemr_hipolito_encounter_notes()
    check_8_openemr_carolyne_icd10()
    check_9_openemr_hipolito_icd10()
    check_10_openemr_carolyne_problem()
    check_11_openemr_hipolito_problem()
    check_12_openemr_procedure_order()
    check_13_onlyoffice_spreadsheet_exists()

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
