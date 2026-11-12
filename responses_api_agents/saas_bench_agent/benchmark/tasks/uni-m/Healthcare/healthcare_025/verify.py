# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_025/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-025-I5: Reportable Disease Workflow - Suspected Mumps Case

Checks: 15 weighted checks across openemr, opnform, onlyoffice.
Strategy: docker exec (DB queries) for all three sites.

Required env vars:
  SERVER_HOSTNAME, OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  OPNFORM_PORT, OPNFORM_CONTAINER, ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER,
  ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess
import json

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENEMR_PORT = os.environ.get("OPENEMR_PORT")
OPENEMR_CONTAINER = os.environ.get("OPENEMR_CONTAINER")
OPENEMR_DB_CONTAINER = os.environ.get("OPENEMR_DB_CONTAINER")

OPNFORM_PORT = os.environ.get("OPNFORM_PORT")
OPNFORM_CONTAINER = os.environ.get("OPNFORM_CONTAINER")

ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.environ.get("ONLYOFFICE_DB_CONTAINER")

_required = {
    "OPENEMR_PORT": OPENEMR_PORT,
    "OPENEMR_CONTAINER": OPENEMR_CONTAINER,
    "OPENEMR_DB_CONTAINER": OPENEMR_DB_CONTAINER,
    "OPNFORM_PORT": OPNFORM_PORT,
    "OPNFORM_CONTAINER": OPNFORM_CONTAINER,
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


def openemr_sql(query: str) -> str:
    """Query OpenEMR MariaDB."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "openemr", "-popenemr_pass", "-D", "openemr",
        "-N", "-e", query,
        timeout=15,
    )
    return out.strip()


def opnform_sql(query: str) -> str:
    """Query OpnForm Postgres (embedded in app container)."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "psql", "-U", "forge", "-d", "forge",
        "-t", "-A", "-c", query,
        timeout=15,
    )
    return out.strip()


def onlyoffice_sql(query: str) -> str:
    """Query OnlyOffice MySQL."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "onlyoffice_user", "-ponlyoffice_pass", "-D", "onlyoffice",
        "-N", "-e", query,
        timeout=15,
    )
    return out.strip()


# ── Patient / encounter cache ─────────────────────────────────────────────────
_pid_cache: str | None = None
_enc_cache: str | None = None


def get_pid() -> str:
    global _pid_cache
    if _pid_cache is None:
        _pid_cache = openemr_sql(
            "SELECT pid FROM patient_data WHERE fname='Freeda' AND lname='Stamm' LIMIT 1"
        )
    return _pid_cache


def get_encounter() -> str:
    global _enc_cache
    if _enc_cache is None:
        pid = get_pid()
        if pid:
            _enc_cache = openemr_sql(
                f"SELECT encounter FROM form_encounter WHERE pid={pid} "
                f"ORDER BY encounter DESC LIMIT 1"
            )
        else:
            _enc_cache = ""
    return _enc_cache


# ── OpenEMR checks ────────────────────────────────────────────────────────────

def check_1_encounter() -> None:
    """Encounter exists for patient Freeda Stamm."""
    try:
        pid = get_pid()
        if not pid:
            check("1. Encounter for Freeda Stamm", 1, False, "patient not found")
            return
        enc = get_encounter()
        check("1. Encounter for Freeda Stamm", 1, bool(enc),
              f"encounter={enc}" if enc else "no encounter found")
    except Exception as e:
        check("1. Encounter for Freeda Stamm", 1, False, f"exception: {e}")


def check_2_soap_subjective_objective() -> None:
    """SOAP Subjective mentions bilateral parotid swelling; Objective has vitals."""
    try:
        pid, enc = get_pid(), get_encounter()
        if not pid or not enc:
            check("2. SOAP Subjective+Objective", 2, False, "no patient/encounter")
            return
        result = openemr_sql(
            f"SELECT fs.subjective, fs.objective FROM form_soap fs "
            f"INNER JOIN forms f ON f.form_id=fs.id AND f.formdir='soap' "
            f"WHERE f.pid={pid} AND f.encounter={enc} ORDER BY fs.id DESC LIMIT 1"
        )
        if not result:
            check("2. SOAP Subjective+Objective", 2, False, "no SOAP form found")
            return
        low = result.lower()
        s_ok = "bilateral parotid swelling" in low
        o_ok = "102.3" in result and "parotid" in low
        check("2. SOAP Subjective+Objective", 2, s_ok and o_ok,
              f"subj={'ok' if s_ok else 'missing key phrase'}, obj={'ok' if o_ok else 'missing key phrase'}")
    except Exception as e:
        check("2. SOAP Subjective+Objective", 2, False, f"exception: {e}")


def check_3_soap_assessment_plan() -> None:
    """SOAP Assessment mentions mumps/parotitis; Plan mentions droplet precautions."""
    try:
        pid, enc = get_pid(), get_encounter()
        if not pid or not enc:
            check("3. SOAP Assessment+Plan", 2, False, "no patient/encounter")
            return
        result = openemr_sql(
            f"SELECT fs.assessment, fs.plan FROM form_soap fs "
            f"INNER JOIN forms f ON f.form_id=fs.id AND f.formdir='soap' "
            f"WHERE f.pid={pid} AND f.encounter={enc} ORDER BY fs.id DESC LIMIT 1"
        )
        if not result:
            check("3. SOAP Assessment+Plan", 2, False, "no SOAP form found")
            return
        low = result.lower()
        a_ok = "mumps" in low and "parotitis" in low
        p_ok = "droplet precautions" in low
        check("3. SOAP Assessment+Plan", 2, a_ok and p_ok,
              f"assess={'ok' if a_ok else 'missing'}, plan={'ok' if p_ok else 'missing'}")
    except Exception as e:
        check("3. SOAP Assessment+Plan", 2, False, f"exception: {e}")


def check_4_ros_findings() -> None:
    """ROS form has constitutional (fever) and respiratory (cough) findings."""
    try:
        pid, enc = get_pid(), get_encounter()
        if not pid or not enc:
            check("4. ROS constitutional+respiratory", 2, False, "no patient/encounter")
            return
        result = openemr_sql(
            f"SELECT fr.* FROM form_ros fr "
            f"INNER JOIN forms f ON f.form_id=fr.id AND f.formdir='ros' "
            f"WHERE f.pid={pid} AND f.encounter={enc} ORDER BY fr.id DESC LIMIT 1"
        )
        if not result:
            check("4. ROS constitutional+respiratory", 2, False, "no ROS form found")
            return
        low = result.lower()
        fever_ok = "fever" in low
        cough_ok = "cough" in low
        check("4. ROS constitutional+respiratory", 2, fever_ok and cough_ok,
              f"fever={'found' if fever_ok else 'not found'}, cough={'found' if cough_ok else 'not found'}")
    except Exception as e:
        check("4. ROS constitutional+respiratory", 2, False, f"exception: {e}")


def check_5_physical_exam() -> None:
    """Physical Exam form has ENT findings (parotid gland, Stensen duct)."""
    try:
        pid, enc = get_pid(), get_encounter()
        if not pid or not enc:
            check("5. Physical Exam ENT findings", 2, False, "no patient/encounter")
            return
        # No standard form_physical_exam table — search all encounter forms
        forms_data = openemr_sql(
            f"SELECT formdir, form_id FROM forms "
            f"WHERE pid={pid} AND encounter={enc} "
            f"AND formdir NOT IN ('soap','ros','newpatient','vitals') "
            f"AND deleted=0"
        )
        found = False
        for line in (forms_data or "").split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            formdir, form_id = parts[0], parts[1]
            try:
                data = openemr_sql(f"SELECT * FROM `form_{formdir}` WHERE id={form_id}")
                if data and "parotid" in data.lower():
                    found = True
                    break
            except Exception:
                continue
        check("5. Physical Exam ENT findings", 2, found,
              "parotid/Stensen findings found" if found else "ENT findings not found in any form")
    except Exception as e:
        check("5. Physical Exam ENT findings", 2, False, f"exception: {e}")


def check_6_problem_mumps() -> None:
    """Active medical problem Mumps with ICD-10 B26.9."""
    try:
        pid = get_pid()
        if not pid:
            check("6. Medical problem Mumps B26.9", 1, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT title, diagnosis FROM lists "
            f"WHERE pid={pid} AND type='medical_problem' AND activity=1 "
            f"AND (diagnosis LIKE '%B26.9%' OR LOWER(title) LIKE '%mumps%')"
        )
        check("6. Medical problem Mumps B26.9", 1, bool(result),
              f"found: {result[:80]}" if result else "not found")
    except Exception as e:
        check("6. Medical problem Mumps B26.9", 1, False, f"exception: {e}")


def check_7_problem_sialadenitis() -> None:
    """Active medical problem Sialadenitis with ICD-10 K11.20."""
    try:
        pid = get_pid()
        if not pid:
            check("7. Medical problem Sialadenitis K11.20", 1, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT title, diagnosis FROM lists "
            f"WHERE pid={pid} AND type='medical_problem' AND activity=1 "
            f"AND (diagnosis LIKE '%K11.20%' OR LOWER(title) LIKE '%ialadenitis%')"
        )
        check("7. Medical problem Sialadenitis K11.20", 1, bool(result),
              f"found: {result[:80]}" if result else "not found")
    except Exception as e:
        check("7. Medical problem Sialadenitis K11.20", 1, False, f"exception: {e}")


def check_8_procedure_order() -> None:
    """Procedure order for mumps serology with URGENT priority."""
    try:
        pid = get_pid()
        if not pid:
            check("8. Procedure order mumps serology URGENT", 2, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT po.order_priority, poc.procedure_name "
            f"FROM procedure_order po "
            f"LEFT JOIN procedure_order_code poc "
            f"  ON po.procedure_order_id=poc.procedure_order_id "
            f"WHERE po.patient_id={pid} "
            f"ORDER BY po.procedure_order_id DESC LIMIT 5"
        )
        if not result:
            check("8. Procedure order mumps serology URGENT", 2, False, "no procedure orders")
            return
        low = result.lower()
        has_mumps = "mumps" in low
        has_urgent = "urgent" in low
        check("8. Procedure order mumps serology URGENT", 2, has_mumps and has_urgent,
              f"mumps={'found' if has_mumps else 'not found'}, "
              f"urgent={'found' if has_urgent else 'not found'}")
    except Exception as e:
        check("8. Procedure order mumps serology URGENT", 2, False, f"exception: {e}")


def check_9_billing_codes() -> None:
    """Fee Sheet contains ICD-10 B26.9, K11.20 and CPT 99214."""
    try:
        pid, enc = get_pid(), get_encounter()
        if not pid or not enc:
            check("9. Fee Sheet billing codes", 2, False, "no patient/encounter")
            return
        result = openemr_sql(
            f"SELECT code_type, code FROM billing "
            f"WHERE pid={pid} AND encounter={enc} AND activity=1"
        )
        has_b269 = "B26.9" in (result or "")
        has_k1120 = "K11.20" in (result or "")
        has_99214 = "99214" in (result or "")
        passed = has_b269 and has_k1120 and has_99214
        found = [c for c, ok in [("B26.9", has_b269), ("K11.20", has_k1120), ("99214", has_99214)] if ok]
        missing = [c for c, ok in [("B26.9", has_b269), ("K11.20", has_k1120), ("99214", has_99214)] if not ok]
        detail = f"found={','.join(found) or 'none'}"
        if missing:
            detail += f", missing={','.join(missing)}"
        check("9. Fee Sheet billing codes", 2, passed, detail)
    except Exception as e:
        check("9. Fee Sheet billing codes", 2, False, f"exception: {e}")


def check_10_message() -> None:
    """Internal message to dr_hartmann about suspected mumps."""
    try:
        pid = get_pid()
        if not pid:
            check("10. Message to dr_hartmann", 2, False, "patient not found")
            return
        result = openemr_sql(
            f"SELECT title, assigned_to FROM pnotes "
            f"WHERE pid={pid} AND deleted=0 "
            f"ORDER BY id DESC LIMIT 10"
        )
        if not result:
            check("10. Message to dr_hartmann", 2, False, "no messages found")
            return
        low = result.lower()
        has_subject = "mumps" in low and "urgent" in low
        has_recipient = "hartmann" in low
        check("10. Message to dr_hartmann", 2, has_subject and has_recipient,
              f"subject={'found' if has_subject else 'not found'}, "
              f"recipient={'found' if has_recipient else 'not found'}")
    except Exception as e:
        check("10. Message to dr_hartmann", 2, False, f"exception: {e}")


def check_11_flow_board_status() -> None:
    """Patient Flow Board status includes Droplet Precautions."""
    try:
        pid = get_pid()
        if not pid:
            check("11. Flow Board Droplet Precautions", 1, False, "patient not found")
            return
        # Check if a custom apptstat option was added with Droplet in the title
        status_row = openemr_sql(
            "SELECT option_id, title FROM list_options "
            "WHERE list_id='apptstat' "
            "AND (LOWER(title) LIKE '%droplet%' OR LOWER(title) LIKE '%precaution%')"
        )
        if status_row:
            option_id = status_row.split("\t")[0].strip()
            tracker = openemr_sql(
                f"SELECT pte.status FROM patient_tracker pt "
                f"JOIN patient_tracker_element pte ON pt.id=pte.pt_tracker_id "
                f"WHERE pt.pid={pid} AND pte.status='{option_id}' "
                f"ORDER BY pte.start_datetime DESC LIMIT 1"
            )
            check("11. Flow Board Droplet Precautions", 1, bool(tracker),
                  f"status '{option_id}' {'assigned' if tracker else 'exists but not assigned'}")
        else:
            # Fallback: check tracker room field or any non-standard status
            tracker = openemr_sql(
                f"SELECT pte.status, pte.room FROM patient_tracker pt "
                f"JOIN patient_tracker_element pte ON pt.id=pte.pt_tracker_id "
                f"WHERE pt.pid={pid} ORDER BY pte.start_datetime DESC LIMIT 1"
            )
            if tracker and "droplet" in tracker.lower():
                check("11. Flow Board Droplet Precautions", 1, True,
                      "found in tracker data")
            else:
                check("11. Flow Board Droplet Precautions", 1, False,
                      f"no Droplet status in list_options, tracker={tracker[:60] if tracker else 'empty'}")
    except Exception as e:
        check("11. Flow Board Droplet Precautions", 1, False, f"exception: {e}")


# ── OpnForm checks ────────────────────────────────────────────────────────────

def check_12_opnform_form() -> None:
    """Form exists with correct title, public visibility, simple theme."""
    try:
        result = opnform_sql(
            "SELECT id, title, visibility, theme, color, submit_button_text, "
            "show_progress_bar, presentation_style "
            "FROM forms "
            "WHERE title LIKE '%Massachusetts Mumps%' "
            "   OR title LIKE '%Mumps%Parotitis%Surveillance%' "
            "ORDER BY id DESC LIMIT 1"
        )
        if not result:
            check("12. OpnForm form exists (public, styled)", 2, False, "form not found")
            return
        parts = result.split("|")
        title = parts[1].strip() if len(parts) > 1 else ""
        visibility = parts[2].strip() if len(parts) > 2 else ""
        theme = parts[3].strip() if len(parts) > 3 else ""
        color = parts[4].strip() if len(parts) > 4 else ""
        submit_text = parts[5].strip() if len(parts) > 5 else ""
        progress = parts[6].strip() if len(parts) > 6 else ""

        title_ok = "mumps" in title.lower() and "surveillance" in title.lower()
        vis_ok = visibility.lower() == "public"
        theme_ok = theme.lower() == "simple"
        issues = []
        if not title_ok:
            issues.append("title mismatch")
        if not vis_ok:
            issues.append(f"visibility={visibility}")
        if not theme_ok:
            issues.append(f"theme={theme}")
        if color.upper() != "#2563EB":
            issues.append(f"color={color}")
        if "submit notification" not in submit_text.lower():
            issues.append(f"submit_text={submit_text[:30]}")
        if progress not in ("t", "true", "1"):
            issues.append(f"progress_bar={progress}")

        passed = title_ok and vis_ok and theme_ok
        check("12. OpnForm form exists (public, styled)", 2, passed,
              "all ok" if not issues else ", ".join(issues))
    except Exception as e:
        check("12. OpnForm form exists (public, styled)", 2, False, f"exception: {e}")


def check_13_opnform_fields() -> None:
    """Form has 14+ fields including conditional logic and code block."""
    try:
        result = opnform_sql(
            "SELECT properties FROM forms "
            "WHERE title LIKE '%Massachusetts Mumps%' "
            "   OR title LIKE '%Mumps%Parotitis%Surveillance%' "
            "ORDER BY id DESC LIMIT 1"
        )
        if not result:
            check("13. OpnForm form has 14+ fields", 2, False, "form not found")
            return
        props = json.loads(result)
        field_count = len(props) if isinstance(props, list) else 0
        has_code_block = False
        has_conditional = False
        for f in (props if isinstance(props, list) else []):
            if f.get("type") == "nf-code":
                has_code_block = True
            if f.get("logic"):
                has_conditional = True
        passed = field_count >= 14
        check("13. OpnForm form has 14+ fields", 2, passed,
              f"count={field_count}, code_block={'yes' if has_code_block else 'no'}, "
              f"conditional={'yes' if has_conditional else 'no'}")
    except Exception as e:
        check("13. OpnForm form has 14+ fields", 2, False, f"exception: {e}")


def check_14_opnform_email_notification() -> None:
    """Email notification integration sends to mumps.surveillance@mass.gov."""
    try:
        form_id = opnform_sql(
            "SELECT id FROM forms "
            "WHERE title LIKE '%Massachusetts Mumps%' "
            "   OR title LIKE '%Mumps%Parotitis%Surveillance%' "
            "ORDER BY id DESC LIMIT 1"
        )
        if not form_id:
            check("14. OpnForm email notification", 1, False, "form not found")
            return
        # Check integrations table
        integ = opnform_sql(
            f"SELECT data FROM form_integrations WHERE form_id={form_id}"
        )
        has_email = "mumps.surveillance@mass.gov" in (integ or "").lower()
        if not has_email:
            # Fallback: check forms table notification columns
            row = opnform_sql(f"SELECT * FROM forms WHERE id={form_id}")
            has_email = "mumps.surveillance@mass.gov" in (row or "").lower()
        check("14. OpnForm email notification", 1, has_email,
              "email found" if has_email else "mumps.surveillance@mass.gov not found")
    except Exception as e:
        check("14. OpnForm email notification", 1, False, f"exception: {e}")


# ── OnlyOffice check ──────────────────────────────────────────────────────────

def check_15_onlyoffice_document() -> None:
    """Document 'Communicable Disease Case Report - MUM-2026-0509-FS' exists."""
    try:
        result = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%MUM-2026-0509-FS%' "
            "   OR title LIKE '%Communicable Disease Case Report%' "
            "ORDER BY id DESC LIMIT 5"
        )
        found = bool(result) and "mum-2026-0509-fs" in result.lower()
        check("15. OnlyOffice document exists", 2, found,
              f"found: {result[:100]}" if found else "document not found")
    except Exception as e:
        check("15. OnlyOffice document exists", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_encounter()
    check_2_soap_subjective_objective()
    check_3_soap_assessment_plan()
    check_4_ros_findings()
    check_5_physical_exam()
    check_6_problem_mumps()
    check_7_problem_sialadenitis()
    check_8_procedure_order()
    check_9_billing_codes()
    check_10_message()
    check_11_flow_board_status()
    check_12_opnform_form()
    check_13_opnform_fields()
    check_14_opnform_email_notification()
    check_15_onlyoffice_document()

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
