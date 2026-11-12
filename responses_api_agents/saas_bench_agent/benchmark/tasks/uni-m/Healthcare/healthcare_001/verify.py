#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_001/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-001-I5: Launch diabetes program across OpnForm, OpenEMR, OnlyOffice

Checks: 12 weighted checks (total 21 pts) across opnform, openemr, onlyoffice.
Strategy: docker exec DB queries for all three sites.

Required env vars:
  SERVER_HOSTNAME, OPNFORM_PORT, OPNFORM_CONTAINER,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess
import json

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPNFORM_PORT = os.environ.get("OPNFORM_PORT")
OPNFORM_CONTAINER = os.environ.get("OPNFORM_CONTAINER")
OPENEMR_PORT = os.environ.get("OPENEMR_PORT")
OPENEMR_CONTAINER = os.environ.get("OPENEMR_CONTAINER")
OPENEMR_DB = os.environ.get("OPENEMR_DB_CONTAINER")
ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB = os.environ.get("ONLYOFFICE_DB_CONTAINER")

_REQUIRED = [
    "OPNFORM_PORT", "OPNFORM_CONTAINER",
    "OPENEMR_PORT", "OPENEMR_CONTAINER", "OPENEMR_DB_CONTAINER",
    "ONLYOFFICE_PORT", "ONLYOFFICE_CONTAINER", "ONLYOFFICE_DB_CONTAINER",
]
for _v in _REQUIRED:
    if not os.environ.get(_v):
        print(f"FATAL: {_v} not set", file=sys.stderr)
        sys.exit(1)

# ── Slot values (from expected_output / slot_values) ──────────────────────────
FORM_TITLE = "Diabetes Self-Management Follow-Up Form"
SPREADSHEET_TITLE = "Diabetes Program Multi-Site Tracker June 2026"
PATIENT_1_FNAME, PATIENT_1_LNAME = "Cyrstal", "Labadie"
PATIENT_2_FNAME, PATIENT_2_LNAME = "Julianne", "Mueller"
P1_BPS, P1_BPD, P1_WEIGHT = "144", "89", "80"
P2_BPS, P2_BPD, P2_WEIGHT = "156", "98", "104"
CARE_GOAL_SUBSTR = "HbA1c below 6.8"
CARE_INSTR_SUBSTR = "fasting glucose each morning"
SOAP_ASSESS_SUBSTR = "Type 2 Diabetes Mellitus"
SOAP_PLAN_SUBSTR = "metformin 1000mg BID"

# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Helpers (docker exec) ─────────────────────────────────────────────────────
def docker_exec(container: str, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    r = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


def opnform_psql(sql: str) -> str:
    """Query OpnForm's embedded PostgreSQL (forge/forge)."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "psql", "-U", "forge", "-d", "forge", "-t", "-A", "-c", sql,
        timeout=20,
    )
    return out.strip()


def openemr_sql(sql: str) -> str:
    """Query OpenEMR MariaDB."""
    rc, out, err = docker_exec(
        OPENEMR_DB,
        "mysql", "-u", "openemr", "-popenemr_pass",
        "--default-character-set=utf8mb4",
        "-D", "openemr", "-N", "-B", "-e", sql,
        timeout=15,
    )
    return out.strip()


def onlyoffice_sql(sql: str) -> str:
    """Query OnlyOffice MySQL."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB,
        "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "-D", "onlyoffice", "-N", "-B", "-e", sql,
        timeout=15,
    )
    return out.strip()


def get_patient_pid(fname: str, lname: str) -> str | None:
    row = openemr_sql(
        f"SELECT pid FROM patient_data WHERE fname='{fname}' AND lname='{lname}' LIMIT 1"
    )
    return row if row else None


# ── OpnForm checks ───────────────────────────────────────────────────────────

def check_1_form_exists() -> None:
    """Form 'Diabetes Self-Management Follow-Up Form' exists and is public/published."""
    try:
        row = opnform_psql(
            f"SELECT visibility FROM forms WHERE title = '{FORM_TITLE}' LIMIT 1"
        )
        if not row:
            check("1. OpnForm form exists & published", 2, False, "form not found")
            return
        passed = row.strip().lower() == "public"
        check("1. OpnForm form exists & published", 2, passed, f"visibility={row}")
    except Exception as e:
        check("1. OpnForm form exists & published", 2, False, f"exception: {e}")


def check_2_form_fields() -> None:
    """Form has 6 fields: date, number x2, scale (1-10), multi_select (5 opts), text."""
    try:
        raw = opnform_psql(
            f"SELECT properties FROM forms WHERE title = '{FORM_TITLE}' LIMIT 1"
        )
        if not raw:
            check("2. OpnForm 6 correct field types", 2, False, "form not found")
            return
        fields = json.loads(raw)
        # Filter out layout/non-input fields (nf-* prefix)
        input_fields = [f for f in fields if not f.get("type", "").startswith("nf-")]
        type_list = [f.get("type", "") for f in input_fields]
        num_count = type_list.count("number")
        has_date = "date" in type_list
        has_scale = "scale" in type_list
        has_multi = "multi_select" in type_list
        has_text = "text" in type_list
        has_6 = len(input_fields) >= 6
        types_ok = has_date and num_count >= 2 and has_scale and has_multi and has_text

        # Check multi-select has 5 options
        ms_field = next((f for f in input_fields if f.get("type") == "multi_select"), None)
        ms_opts = 0
        if ms_field:
            sel = ms_field.get("multi_select") or ms_field.get("select") or {}
            opts = sel.get("options", [])
            ms_opts = len(opts)

        passed = has_6 and types_ok and ms_opts >= 5
        check("2. OpnForm 6 correct field types", 2, passed,
              f"fields={len(input_fields)}, types={type_list}, ms_opts={ms_opts}")
    except Exception as e:
        check("2. OpnForm 6 correct field types", 2, False, f"exception: {e}")


def check_3_conditional_logic() -> None:
    """Barriers to Adherence has conditional logic: show when adherence < 5."""
    try:
        raw = opnform_psql(
            f"SELECT properties FROM forms WHERE title = '{FORM_TITLE}' LIMIT 1"
        )
        if not raw:
            check("3. OpnForm conditional logic on Barriers", 2, False, "form not found")
            return
        fields = json.loads(raw)
        barriers = None
        for f in fields:
            name = (f.get("name") or "").lower()
            if "barrier" in name:
                barriers = f
                break
        if not barriers:
            check("3. OpnForm conditional logic on Barriers", 2, False,
                  "no field with 'barrier' in name")
            return
        logic = barriers.get("logic")
        has_logic = bool(logic) and logic != {} and logic != []
        logic_str = json.dumps(logic).lower() if logic else ""
        # Look for evidence of "less than 5" or "show" action referencing the scale field
        refs_threshold = "5" in logic_str
        refs_action = "show" in logic_str or "hide" in logic_str
        passed = has_logic and (refs_threshold or refs_action)
        check("3. OpnForm conditional logic on Barriers", 2, passed,
              f"has_logic={has_logic}, snippet={logic_str[:120]}")
    except Exception as e:
        check("3. OpnForm conditional logic on Barriers", 2, False, f"exception: {e}")


# ── OpenEMR checks ───────────────────────────────────────────────────────────

def check_4_encounter_p1() -> str | None:
    """Encounter exists for Cyrstal Labadie."""
    try:
        pid = get_patient_pid(PATIENT_1_FNAME, PATIENT_1_LNAME)
        if not pid:
            check("4. Encounter for Cyrstal Labadie", 1, False, "patient not found")
            return None
        cnt = openemr_sql(f"SELECT COUNT(*) FROM form_encounter WHERE pid={pid}")
        ok = int(cnt or 0) > 0
        check("4. Encounter for Cyrstal Labadie", 1, ok, f"pid={pid}, encounters={cnt}")
        return pid
    except Exception as e:
        check("4. Encounter for Cyrstal Labadie", 1, False, f"exception: {e}")
        return None


def check_5_vitals_p1(pid: str | None) -> None:
    """Vitals for Cyrstal Labadie: BP 144/89, weight 80 kg."""
    if not pid:
        check("5. Vitals for Cyrstal Labadie", 2, False, "no pid")
        return
    try:
        row = openemr_sql(
            f"SELECT bps, bpd, weight FROM form_vitals "
            f"WHERE pid={pid} AND activity=1 ORDER BY date DESC LIMIT 1"
        )
        if not row:
            check("5. Vitals for Cyrstal Labadie", 2, False, "no vitals found")
            return
        parts = row.split("\t")
        bps = parts[0].strip() if len(parts) > 0 else ""
        bpd = parts[1].strip() if len(parts) > 1 else ""
        wt = parts[2].strip() if len(parts) > 2 else ""
        try:
            wt_ok = abs(float(wt) - float(P1_WEIGHT)) < 1.0
        except ValueError:
            wt_ok = False
        passed = bps == P1_BPS and bpd == P1_BPD and wt_ok
        check("5. Vitals for Cyrstal Labadie", 2, passed,
              f"bps={bps}, bpd={bpd}, wt={wt}")
    except Exception as e:
        check("5. Vitals for Cyrstal Labadie", 2, False, f"exception: {e}")


def check_6_soap_p1(pid: str | None) -> None:
    """SOAP note for Cyrstal Labadie: assessment + plan."""
    if not pid:
        check("6. SOAP note for Cyrstal Labadie", 2, False, "no pid")
        return
    try:
        row = openemr_sql(
            f"SELECT assessment, plan FROM form_soap "
            f"WHERE pid={pid} AND activity=1 ORDER BY date DESC LIMIT 1"
        )
        if not row:
            check("6. SOAP note for Cyrstal Labadie", 2, False, "no SOAP note found")
            return
        parts = row.split("\t")
        assessment = parts[0] if len(parts) > 0 else ""
        plan_text = parts[1] if len(parts) > 1 else ""
        a_ok = SOAP_ASSESS_SUBSTR.lower() in assessment.lower()
        p_ok = SOAP_PLAN_SUBSTR.lower() in plan_text.lower()
        check("6. SOAP note for Cyrstal Labadie", 2, a_ok and p_ok,
              f"assess_match={a_ok}, plan_match={p_ok}")
    except Exception as e:
        check("6. SOAP note for Cyrstal Labadie", 2, False, f"exception: {e}")


def check_7_careplan_p1(pid: str | None) -> None:
    """Care plan for Cyrstal Labadie: goal + instructions."""
    if not pid:
        check("7. Care plan for Cyrstal Labadie", 2, False, "no pid")
        return
    try:
        # Goal could be in description or codetext
        goal_row = openemr_sql(
            f"SELECT CONCAT_WS('|||', COALESCE(description,''), COALESCE(codetext,''), "
            f"COALESCE(reason_description,''), COALESCE(note_related_to,'')) "
            f"FROM form_care_plan WHERE pid={pid} AND activity=1 ORDER BY date DESC LIMIT 5"
        )
        goal_found = CARE_GOAL_SUBSTR.lower() in (goal_row or "").lower()

        # Instructions in form_clinical_instructions or care plan columns
        instr_row = openemr_sql(
            f"SELECT instruction FROM form_clinical_instructions "
            f"WHERE pid={pid} AND activity=1 ORDER BY date DESC LIMIT 5"
        )
        instr_found = CARE_INSTR_SUBSTR.lower() in (instr_row or "").lower()
        if not instr_found:
            instr_found = CARE_INSTR_SUBSTR.lower() in (goal_row or "").lower()

        check("7. Care plan for Cyrstal Labadie", 2, goal_found and instr_found,
              f"goal_found={goal_found}, instr_found={instr_found}")
    except Exception as e:
        check("7. Care plan for Cyrstal Labadie", 2, False, f"exception: {e}")


def check_8_encounter_p2() -> str | None:
    """Encounter exists for Julianne Mueller."""
    try:
        pid = get_patient_pid(PATIENT_2_FNAME, PATIENT_2_LNAME)
        if not pid:
            check("8. Encounter for Julianne Mueller", 1, False, "patient not found")
            return None
        cnt = openemr_sql(f"SELECT COUNT(*) FROM form_encounter WHERE pid={pid}")
        ok = int(cnt or 0) > 0
        check("8. Encounter for Julianne Mueller", 1, ok, f"pid={pid}, encounters={cnt}")
        return pid
    except Exception as e:
        check("8. Encounter for Julianne Mueller", 1, False, f"exception: {e}")
        return None


def check_9_vitals_p2(pid: str | None) -> None:
    """Vitals for Julianne Mueller: BP 156/98, weight 104 kg."""
    if not pid:
        check("9. Vitals for Julianne Mueller", 2, False, "no pid")
        return
    try:
        row = openemr_sql(
            f"SELECT bps, bpd, weight FROM form_vitals "
            f"WHERE pid={pid} AND activity=1 ORDER BY date DESC LIMIT 1"
        )
        if not row:
            check("9. Vitals for Julianne Mueller", 2, False, "no vitals found")
            return
        parts = row.split("\t")
        bps = parts[0].strip() if len(parts) > 0 else ""
        bpd = parts[1].strip() if len(parts) > 1 else ""
        wt = parts[2].strip() if len(parts) > 2 else ""
        try:
            wt_ok = abs(float(wt) - float(P2_WEIGHT)) < 1.0
        except ValueError:
            wt_ok = False
        passed = bps == P2_BPS and bpd == P2_BPD and wt_ok
        check("9. Vitals for Julianne Mueller", 2, passed,
              f"bps={bps}, bpd={bpd}, wt={wt}")
    except Exception as e:
        check("9. Vitals for Julianne Mueller", 2, False, f"exception: {e}")


def check_10_soap_p2(pid: str | None) -> None:
    """SOAP note for Julianne Mueller: assessment + plan."""
    if not pid:
        check("10. SOAP note for Julianne Mueller", 2, False, "no pid")
        return
    try:
        row = openemr_sql(
            f"SELECT assessment, plan FROM form_soap "
            f"WHERE pid={pid} AND activity=1 ORDER BY date DESC LIMIT 1"
        )
        if not row:
            check("10. SOAP note for Julianne Mueller", 2, False, "no SOAP note found")
            return
        parts = row.split("\t")
        assessment = parts[0] if len(parts) > 0 else ""
        plan_text = parts[1] if len(parts) > 1 else ""
        a_ok = SOAP_ASSESS_SUBSTR.lower() in assessment.lower()
        p_ok = SOAP_PLAN_SUBSTR.lower() in plan_text.lower()
        check("10. SOAP note for Julianne Mueller", 2, a_ok and p_ok,
              f"assess_match={a_ok}, plan_match={p_ok}")
    except Exception as e:
        check("10. SOAP note for Julianne Mueller", 2, False, f"exception: {e}")


def check_11_careplan_p2(pid: str | None) -> None:
    """Care plan for Julianne Mueller: goal + instructions."""
    if not pid:
        check("11. Care plan for Julianne Mueller", 2, False, "no pid")
        return
    try:
        goal_row = openemr_sql(
            f"SELECT CONCAT_WS('|||', COALESCE(description,''), COALESCE(codetext,''), "
            f"COALESCE(reason_description,''), COALESCE(note_related_to,'')) "
            f"FROM form_care_plan WHERE pid={pid} AND activity=1 ORDER BY date DESC LIMIT 5"
        )
        goal_found = CARE_GOAL_SUBSTR.lower() in (goal_row or "").lower()

        instr_row = openemr_sql(
            f"SELECT instruction FROM form_clinical_instructions "
            f"WHERE pid={pid} AND activity=1 ORDER BY date DESC LIMIT 5"
        )
        instr_found = CARE_INSTR_SUBSTR.lower() in (instr_row or "").lower()
        if not instr_found:
            instr_found = CARE_INSTR_SUBSTR.lower() in (goal_row or "").lower()

        check("11. Care plan for Julianne Mueller", 2, goal_found and instr_found,
              f"goal_found={goal_found}, instr_found={instr_found}")
    except Exception as e:
        check("11. Care plan for Julianne Mueller", 2, False, f"exception: {e}")


# ── OnlyOffice check ─────────────────────────────────────────────────────────

def check_12_spreadsheet() -> None:
    """Spreadsheet 'Diabetes Program Multi-Site Tracker June 2026' exists in OnlyOffice."""
    try:
        row = onlyoffice_sql(
            "SELECT id, title FROM files_file "
            f"WHERE title LIKE '%{SPREADSHEET_TITLE}%' "
            "AND current_version = 0 LIMIT 1"
        )
        if not row:
            # Retry without current_version filter
            row = onlyoffice_sql(
                "SELECT id, title FROM files_file "
                f"WHERE title LIKE '%{SPREADSHEET_TITLE}%' LIMIT 1"
            )
        if not row:
            # Broader search
            row = onlyoffice_sql(
                "SELECT id, title FROM files_file "
                "WHERE title LIKE '%Diabetes%Tracker%' LIMIT 1"
            )
        check("12. OnlyOffice spreadsheet exists", 1, bool(row),
              f"found={row[:120]}" if row else "not found")
    except Exception as e:
        check("12. OnlyOffice spreadsheet exists", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # OpnForm (3 checks)
    check_1_form_exists()
    check_2_form_fields()
    check_3_conditional_logic()

    # OpenEMR — Patient 1 (4 checks)
    pid1 = check_4_encounter_p1()
    if pid1 is None:
        pid1 = get_patient_pid(PATIENT_1_FNAME, PATIENT_1_LNAME)
    check_5_vitals_p1(pid1)
    check_6_soap_p1(pid1)
    check_7_careplan_p1(pid1)

    # OpenEMR — Patient 2 (4 checks)
    pid2 = check_8_encounter_p2()
    if pid2 is None:
        pid2 = get_patient_pid(PATIENT_2_FNAME, PATIENT_2_LNAME)
    check_9_vitals_p2(pid2)
    check_10_soap_p2(pid2)
    check_11_careplan_p2(pid2)

    # OnlyOffice (1 check)
    check_12_spreadsheet()

    # Summary
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
