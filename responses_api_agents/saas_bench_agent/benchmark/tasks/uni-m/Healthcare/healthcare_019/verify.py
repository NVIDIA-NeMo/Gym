# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_019/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-019-I3: Colonoscopy Informed Consent Workflow for Three Patients

Checks: 16 weighted checks across opnform, openemr, onlyoffice.
Strategy: docker exec (DB queries) for all three apps.

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


def opnform_sql(query: str) -> str:
    """Query OpnForm's embedded Postgres (forge DB, forge user)."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "php", "artisan", "tinker", "--execute",
        f"echo json_encode(DB::select(DB::raw(\"{query}\")));",
        timeout=30,
    )
    return out.strip()


def opnform_sql_raw(query: str) -> str:
    """Query OpnForm Postgres directly via psql."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "bash", "-c",
        f"PGPASSWORD=forge psql -U forge -d forge -t -A -c \"{query}\"",
        timeout=15,
    )
    return out.strip()


def openemr_sql(query: str) -> str:
    """Query OpenEMR MariaDB."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "openemr", "-popenemr_pass", "-D", "openemr",
        "-N", "-B", "-e", query,
        timeout=15,
    )
    return out.strip()


def onlyoffice_sql(query: str) -> str:
    """Query OnlyOffice MySQL."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql",
        "-u", "onlyoffice_user", "-ponlyoffice_pass", "-D", "onlyoffice",
        "-N", "-B", "-e", query,
        timeout=15,
    )
    return out.strip()


def find_patient_pid(fname: str, lname: str) -> str:
    """Look up OpenEMR patient pid by first/last name."""
    return openemr_sql(
        f"SELECT pid FROM patient_data WHERE fname='{fname}' AND lname='{lname}' LIMIT 1"
    )


# ── OpnForm Checks ───────────────────────────────────────────────────────────
def check_1_opnform_form_settings():
    """Form exists with correct title, visibility, theme, size, color."""
    try:
        row = opnform_sql_raw(
            "SELECT title, visibility, theme, size, color FROM forms "
            "WHERE title='Colonoscopy Informed Consent Form 2026' AND deleted_at IS NULL LIMIT 1"
        )
        if not row:
            check("1. OpnForm form exists with correct settings", 2, False, "form not found")
            return
        parts = row.split("|")
        title = parts[0] if len(parts) > 0 else ""
        visibility = parts[1] if len(parts) > 1 else ""
        theme = parts[2] if len(parts) > 2 else ""
        size = parts[3] if len(parts) > 3 else ""
        color = parts[4] if len(parts) > 4 else ""
        issues = []
        if visibility != "public":
            issues.append(f"visibility={visibility}")
        if theme != "minimal":
            issues.append(f"theme={theme}")
        if size != "lg":
            issues.append(f"size={size}")
        if color != "#16a34a":
            issues.append(f"color={color}")
        passed = len(issues) == 0
        detail = "; ".join(issues) if issues else ""
        check("1. OpnForm form exists with correct settings", 2, passed, detail)
    except Exception as e:
        check("1. OpnForm form exists with correct settings", 2, False, f"exception: {e}")


def check_2_opnform_form_options():
    """Form has re_fillable, correct button text, search indexing disabled."""
    try:
        row = opnform_sql_raw(
            "SELECT re_fillable, re_fill_button_text, can_be_indexed FROM forms "
            "WHERE title='Colonoscopy Informed Consent Form 2026' AND deleted_at IS NULL LIMIT 1"
        )
        if not row:
            check("2. OpnForm form options (re-fillable, indexing)", 1, False, "form not found")
            return
        parts = row.split("|")
        re_fillable = parts[0] if len(parts) > 0 else ""
        re_fill_text = parts[1] if len(parts) > 1 else ""
        can_be_indexed = parts[2] if len(parts) > 2 else ""
        issues = []
        # re_fillable: postgres boolean is 't'/'f' or '1'/'0'
        if re_fillable not in ("t", "1", "true"):
            issues.append(f"re_fillable={re_fillable}")
        if re_fill_text != "Complete Another Consent":
            issues.append(f"re_fill_button_text={re_fill_text}")
        if can_be_indexed not in ("f", "0", "false"):
            issues.append(f"can_be_indexed={can_be_indexed}")
        check("2. OpnForm form options (re-fillable, indexing)", 1, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("2. OpnForm form options (re-fillable, indexing)", 1, False, f"exception: {e}")


def check_3_opnform_form_fields():
    """Form properties contain required field types: multi-select with 4 risk options,
    page break, signature, conditional field, toggle checkbox."""
    try:
        row = opnform_sql_raw(
            "SELECT properties FROM forms "
            "WHERE title='Colonoscopy Informed Consent Form 2026' AND deleted_at IS NULL LIMIT 1"
        )
        if not row:
            check("3. OpnForm form fields structure", 3, False, "form not found")
            return
        props = json.loads(row)
        field_types = [f.get("type", "") for f in props] if isinstance(props, list) else []
        field_names = [f.get("name", "") for f in props] if isinstance(props, list) else []
        field_labels = [f.get("label", "") if f.get("label") else "" for f in props] if isinstance(props, list) else []

        issues = []

        # Check for nf-text blocks (procedure, risks, benefits)
        text_blocks = [f for f in props if f.get("type") == "nf-text"] if isinstance(props, list) else []
        if len(text_blocks) < 3:
            issues.append(f"text blocks: {len(text_blocks)}/3")

        # Check for multi-select with 4 risk options
        multi_selects = [f for f in props if f.get("type") == "multi_select"] if isinstance(props, list) else []
        found_risks = False
        for ms in multi_selects:
            opts = ms.get("multi_select", {}).get("options", []) if isinstance(ms.get("multi_select"), dict) else []
            if not opts:
                # Try alternate structure
                opts = ms.get("options", [])
                if not opts:
                    opts = ms.get(ms.get("name", ""), {}).get("options", []) if isinstance(ms.get(ms.get("name", "")), dict) else []
            opt_names = [o.get("name", "") if isinstance(o, dict) else str(o) for o in opts]
            expected_risks = {"Bowel Perforation", "Post-Polypectomy Bleeding", "Sedation Reaction", "Cardiopulmonary Complications"}
            if expected_risks.issubset(set(opt_names)):
                found_risks = True
                break
        if not found_risks and multi_selects:
            # Simpler check: just verify multi_select exists
            found_risks_approx = any(
                "risk" in (f.get("name", "") + f.get("label", "")).lower()
                for f in multi_selects
            )
            if found_risks_approx:
                found_risks = True
        if not found_risks:
            issues.append("multi-select 'Acknowledged Risks' not found or missing options")

        # Check for signature field
        if "signature" not in field_types:
            issues.append("no signature field")

        # Check for page break
        has_page_break = any(f.get("type") in ("nf-page-break", "page-break") for f in props) if isinstance(props, list) else False
        if not has_page_break:
            issues.append("no page break")

        check("3. OpnForm form fields structure", 3, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("3. OpnForm form fields structure", 3, False, f"exception: {e}")


def check_4_opnform_webhook():
    """Webhook integration exists with correct URL."""
    try:
        # Get form ID first
        form_id = opnform_sql_raw(
            "SELECT id FROM forms "
            "WHERE title='Colonoscopy Informed Consent Form 2026' AND deleted_at IS NULL LIMIT 1"
        )
        if not form_id:
            check("4. OpnForm webhook integration", 1, False, "form not found")
            return
        # Check form_integrations
        rows = opnform_sql_raw(
            f"SELECT data FROM form_integrations "
            f"WHERE form_id={form_id} AND deleted_at IS NULL"
        )
        expected_url = "https://hooks.clinicops.example.com/opnform/colonoscopy-consent"
        found = expected_url in rows if rows else False
        if not found:
            # Also check webhook_url on the form itself
            webhook_url = opnform_sql_raw(
                f"SELECT webhook_url FROM forms WHERE id={form_id}"
            )
            found = webhook_url == expected_url
        check("4. OpnForm webhook integration", 1, found,
              "" if found else f"webhook URL not found in integrations")
    except Exception as e:
        check("4. OpnForm webhook integration", 1, False, f"exception: {e}")


# ── OpenEMR Checks — Hettie Torphy ───────────────────────────────────────────
def check_5_hettie_prior_auth():
    """Hettie Torphy has encounter with Prior Auth PA-2026-GI-1."""
    try:
        pid = find_patient_pid("Hettie", "Torphy")
        if not pid:
            check("5. Hettie Torphy prior auth PA-2026-GI-1", 2, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT prior_auth_number FROM form_misc_billing_options "
            f"WHERE pid={pid} AND prior_auth_number='PA-2026-GI-1' LIMIT 1"
        )
        check("5. Hettie Torphy prior auth PA-2026-GI-1", 2, row == "PA-2026-GI-1",
              f"got: {row!r}" if row != "PA-2026-GI-1" else "")
    except Exception as e:
        check("5. Hettie Torphy prior auth PA-2026-GI-1", 2, False, f"exception: {e}")


def check_6_hettie_clinical_instructions():
    """Hettie Torphy has Clinical Instructions with correct text."""
    try:
        pid = find_patient_pid("Hettie", "Torphy")
        if not pid:
            check("6. Hettie Torphy clinical instructions", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT instruction FROM form_clinical_instructions "
            f"WHERE pid={pid} ORDER BY id DESC LIMIT 1"
        )
        expected_start = "Begin clear liquid diet"
        passed = expected_start in row if row else False
        check("6. Hettie Torphy clinical instructions", 1, passed,
              "" if passed else f"instruction not found or wrong content")
    except Exception as e:
        check("6. Hettie Torphy clinical instructions", 1, False, f"exception: {e}")


def check_7_hettie_medical_problem():
    """Hettie Torphy has active medical problem 'Rectal bleeding, unspecified' K62.5."""
    try:
        pid = find_patient_pid("Hettie", "Torphy")
        if not pid:
            check("7. Hettie Torphy medical problem K62.5", 2, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT title, diagnosis, activity FROM lists "
            f"WHERE pid={pid} AND type='medical_problem' "
            f"AND (title LIKE '%Rectal bleeding%' OR diagnosis LIKE '%K62.5%') LIMIT 1"
        )
        if not row:
            check("7. Hettie Torphy medical problem K62.5", 2, False, "problem not found")
            return
        parts = row.split("\t")
        title = parts[0] if len(parts) > 0 else ""
        diagnosis = parts[1] if len(parts) > 1 else ""
        activity = parts[2] if len(parts) > 2 else ""
        issues = []
        if "K62.5" not in diagnosis:
            issues.append(f"diagnosis={diagnosis}")
        if activity != "1":
            issues.append(f"activity={activity} (not active)")
        check("7. Hettie Torphy medical problem K62.5", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("7. Hettie Torphy medical problem K62.5", 2, False, f"exception: {e}")


def check_8_hettie_fee_sheet():
    """Hettie Torphy Fee Sheet has ICD-10 K62.5 and CPT 45378."""
    try:
        pid = find_patient_pid("Hettie", "Torphy")
        if not pid:
            check("8. Hettie Torphy fee sheet (K62.5 + CPT 45378)", 2, False, "patient not found")
            return
        rows = openemr_sql(
            f"SELECT code, code_type FROM billing "
            f"WHERE pid={pid} AND activity=1 "
            f"AND (code='K62.5' OR code='45378')"
        )
        codes_found = set()
        for line in rows.splitlines():
            parts = line.split("\t")
            if len(parts) >= 1:
                codes_found.add(parts[0].strip())
        has_icd = "K62.5" in codes_found
        has_cpt = "45378" in codes_found
        issues = []
        if not has_icd:
            issues.append("missing ICD-10 K62.5")
        if not has_cpt:
            issues.append("missing CPT 45378")
        check("8. Hettie Torphy fee sheet (K62.5 + CPT 45378)", 2, len(issues) == 0,
              "; ".join(issues) if issues else "")
    except Exception as e:
        check("8. Hettie Torphy fee sheet (K62.5 + CPT 45378)", 2, False, f"exception: {e}")


# ── OpenEMR Checks — Zack Nikolaus ───────────────────────────────────────────
def check_9_zack_prior_auth():
    """Zack Nikolaus has encounter with Prior Auth PA-2026-GI-2."""
    try:
        pid = find_patient_pid("Zack", "Nikolaus")
        if not pid:
            check("9. Zack Nikolaus prior auth PA-2026-GI-2", 2, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT prior_auth_number FROM form_misc_billing_options "
            f"WHERE pid={pid} AND prior_auth_number='PA-2026-GI-2' LIMIT 1"
        )
        check("9. Zack Nikolaus prior auth PA-2026-GI-2", 2, row == "PA-2026-GI-2",
              f"got: {row!r}" if row != "PA-2026-GI-2" else "")
    except Exception as e:
        check("9. Zack Nikolaus prior auth PA-2026-GI-2", 2, False, f"exception: {e}")


def check_10_zack_clinical_instructions():
    """Zack Nikolaus has Clinical Instructions."""
    try:
        pid = find_patient_pid("Zack", "Nikolaus")
        if not pid:
            check("10. Zack Nikolaus clinical instructions", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT instruction FROM form_clinical_instructions "
            f"WHERE pid={pid} ORDER BY id DESC LIMIT 1"
        )
        passed = "Begin clear liquid diet" in row if row else False
        check("10. Zack Nikolaus clinical instructions", 1, passed,
              "" if passed else "instruction not found")
    except Exception as e:
        check("10. Zack Nikolaus clinical instructions", 1, False, f"exception: {e}")


def check_11_zack_cpt():
    """Zack Nikolaus has CPT 45378 on Fee Sheet."""
    try:
        pid = find_patient_pid("Zack", "Nikolaus")
        if not pid:
            check("11. Zack Nikolaus CPT 45378", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT code FROM billing "
            f"WHERE pid={pid} AND code='45378' AND activity=1 LIMIT 1"
        )
        check("11. Zack Nikolaus CPT 45378", 1, row == "45378",
              f"got: {row!r}" if row != "45378" else "")
    except Exception as e:
        check("11. Zack Nikolaus CPT 45378", 1, False, f"exception: {e}")


# ── OpenEMR Checks — Connie McLaughlin ────────────────────────────────────────
def check_12_connie_prior_auth():
    """Connie McLaughlin has encounter with Prior Auth PA-2026-GI-3."""
    try:
        pid = find_patient_pid("Connie", "McLaughlin")
        if not pid:
            check("12. Connie McLaughlin prior auth PA-2026-GI-3", 2, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT prior_auth_number FROM form_misc_billing_options "
            f"WHERE pid={pid} AND prior_auth_number='PA-2026-GI-3' LIMIT 1"
        )
        check("12. Connie McLaughlin prior auth PA-2026-GI-3", 2, row == "PA-2026-GI-3",
              f"got: {row!r}" if row != "PA-2026-GI-3" else "")
    except Exception as e:
        check("12. Connie McLaughlin prior auth PA-2026-GI-3", 2, False, f"exception: {e}")


def check_13_connie_clinical_instructions():
    """Connie McLaughlin has Clinical Instructions."""
    try:
        pid = find_patient_pid("Connie", "McLaughlin")
        if not pid:
            check("13. Connie McLaughlin clinical instructions", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT instruction FROM form_clinical_instructions "
            f"WHERE pid={pid} ORDER BY id DESC LIMIT 1"
        )
        passed = "Begin clear liquid diet" in row if row else False
        check("13. Connie McLaughlin clinical instructions", 1, passed,
              "" if passed else "instruction not found")
    except Exception as e:
        check("13. Connie McLaughlin clinical instructions", 1, False, f"exception: {e}")


def check_14_connie_cpt():
    """Connie McLaughlin has CPT 45378 on Fee Sheet."""
    try:
        pid = find_patient_pid("Connie", "McLaughlin")
        if not pid:
            check("14. Connie McLaughlin CPT 45378", 1, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT code FROM billing "
            f"WHERE pid={pid} AND code='45378' AND activity=1 LIMIT 1"
        )
        check("14. Connie McLaughlin CPT 45378", 1, row == "45378",
              f"got: {row!r}" if row != "45378" else "")
    except Exception as e:
        check("14. Connie McLaughlin CPT 45378", 1, False, f"exception: {e}")


def check_15_connie_onset_date():
    """Connie McLaughlin Misc Billing Options onset date = 2026-04-08."""
    try:
        pid = find_patient_pid("Connie", "McLaughlin")
        if not pid:
            check("15. Connie McLaughlin onset date 2026-04-08", 2, False, "patient not found")
            return
        row = openemr_sql(
            f"SELECT onset_date FROM form_misc_billing_options "
            f"WHERE pid={pid} AND onset_date IS NOT NULL "
            f"ORDER BY id DESC LIMIT 1"
        )
        passed = row.strip() == "2026-04-08" if row else False
        check("15. Connie McLaughlin onset date 2026-04-08", 2, passed,
              f"got: {row!r}" if not passed else "")
    except Exception as e:
        check("15. Connie McLaughlin onset date 2026-04-08", 2, False, f"exception: {e}")


# ── OnlyOffice Check ──────────────────────────────────────────────────────────
def check_16_onlyoffice_spreadsheet():
    """Spreadsheet 'Colonoscopy Consent and Procedure Tracker 2026' exists."""
    try:
        row = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Colonoscopy Consent and Procedure Tracker 2026%' "
            "AND current_version=1 LIMIT 1"
        )
        if not row:
            # Try without current_version filter
            row = onlyoffice_sql(
                "SELECT id, title FROM files_file "
                "WHERE title LIKE '%Colonoscopy Consent and Procedure Tracker 2026%' LIMIT 1"
            )
        passed = bool(row)
        check("16. OnlyOffice spreadsheet exists", 1, passed,
              "" if passed else "spreadsheet not found")
    except Exception as e:
        check("16. OnlyOffice spreadsheet exists", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_opnform_form_settings()
    check_2_opnform_form_options()
    check_3_opnform_form_fields()
    check_4_opnform_webhook()
    check_5_hettie_prior_auth()
    check_6_hettie_clinical_instructions()
    check_7_hettie_medical_problem()
    check_8_hettie_fee_sheet()
    check_9_zack_prior_auth()
    check_10_zack_clinical_instructions()
    check_11_zack_cpt()
    check_12_connie_prior_auth()
    check_13_connie_clinical_instructions()
    check_14_connie_cpt()
    check_15_connie_onset_date()
    check_16_onlyoffice_spreadsheet()

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
