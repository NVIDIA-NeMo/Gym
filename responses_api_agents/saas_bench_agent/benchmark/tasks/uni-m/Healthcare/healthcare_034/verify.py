#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_034/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-034-I1: New Patient Orientation Workflow for Maria Gonzalez

Checks: 15 weighted checks across opnform, openemr, onlyoffice.
Strategy: docker exec (DB) for OpnForm and OpenEMR; API + filesystem for OnlyOffice.

Required env vars:
  SERVER_HOSTNAME, OPNFORM_PORT, OPNFORM_CONTAINER,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import json
import subprocess
import requests

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

for _v in [
    "OPNFORM_PORT", "OPNFORM_CONTAINER",
    "OPENEMR_PORT", "OPENEMR_CONTAINER", "OPENEMR_DB_CONTAINER",
    "ONLYOFFICE_PORT", "ONLYOFFICE_CONTAINER", "ONLYOFFICE_DB_CONTAINER",
]:
    if not os.getenv(_v):
        print(f"FATAL: {_v} not set", file=sys.stderr)
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


def opnform_sql(sql: str) -> str:
    _, out, _ = docker_exec(
        OPNFORM_CONTAINER,
        "psql", "-U", "forge", "-d", "forge", "-t", "-A", "-c", sql,
    )
    return out.strip()


def openemr_sql(sql: str) -> str:
    _, out, _ = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "-u", "openemr", "-popenemr_pass", "-D", "openemr",
        "--default-character-set=utf8mb4", "-N", "-e", sql,
    )
    return out.strip()


_patient_pid: str | None = None


def pid() -> str:
    global _patient_pid
    if _patient_pid is None:
        _patient_pid = openemr_sql(
            "SELECT pid FROM patient_data WHERE fname='Maria' AND lname='Gonzalez' LIMIT 1;"
        ) or ""
    return _patient_pid


# ── OpnForm: cached form row ─────────────────────────────────────────────────
_form_data: dict | None = None


def get_form() -> dict:
    global _form_data
    if _form_data is None:
        out = opnform_sql(
            "SELECT row_to_json(f) FROM forms f "
            "WHERE title = 'New Patient Orientation Questionnaire 2026' LIMIT 1;"
        )
        _form_data = json.loads(out) if out else {}
    return _form_data


# ── Check 1: OpnForm form exists ─────────────────────────────────────────────
def check_1_form_exists() -> None:
    """Form titled 'New Patient Orientation Questionnaire 2026' exists in OpnForm."""
    try:
        f = get_form()
        check("1. OpnForm form exists", 1, bool(f),
              f"title={f.get('title', '')}" if f else "not found")
    except Exception as e:
        check("1. OpnForm form exists", 1, False, f"exception: {e}")


# ── Check 2: OpnForm form settings ───────────────────────────────────────────
def check_2_form_settings() -> None:
    """Form settings: visibility public, editable submissions, redirect URL, re-fillable off."""
    try:
        f = get_form()
        if not f:
            check("2. OpnForm form settings", 2, False, "form not found")
            return

        issues = []
        if f.get("visibility") != "public":
            issues.append(f"visibility={f.get('visibility')}")
        if not f.get("editable_submissions"):
            issues.append("editable_submissions off")
        rurl = f.get("redirect_url") or ""
        if "clinic.example.com/orientation/thank-you" not in rurl:
            issues.append(f"redirect_url={rurl!r}")
        if f.get("re_fillable"):
            issues.append("re_fillable should be off")

        check("2. OpnForm form settings", 2, not issues,
              "correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("2. OpnForm form settings", 2, False, f"exception: {e}")


# ── Check 3: OpnForm form fields ─────────────────────────────────────────────
def check_3_form_fields() -> None:
    """Form has expected field types: video embed, page break, multi-select, scale, conditionals."""
    try:
        f = get_form()
        if not f:
            check("3. OpnForm form fields", 2, False, "form not found")
            return

        props = f.get("properties", [])
        if isinstance(props, str):
            props = json.loads(props)
        if not isinstance(props, list):
            check("3. OpnForm form fields", 2, False, f"properties not a list: {type(props)}")
            return

        n = len(props)
        types = set()
        for p in props:
            t = (p.get("type") or "").lower()
            types.add(t)

        has_video = any("video" in t for t in types)
        has_break = any("page" in t or "break" in t for t in types)

        issues = []
        if n < 12:
            issues.append(f"only {n} fields (expect 15+)")
        if not has_video:
            issues.append("no video field")
        if not has_break:
            issues.append("no page break")

        check("3. OpnForm form fields", 2, not issues,
              f"{n} fields, types={sorted(types)}" if not issues else "; ".join(issues))
    except Exception as e:
        check("3. OpnForm form fields", 2, False, f"exception: {e}")


# ── Check 4: Patient demographics ────────────────────────────────────────────
def check_4_demographics() -> None:
    """Patient Maria Gonzalez exists with DOB 1985-07-22, Female, Spanish."""
    try:
        out = openemr_sql(
            "SELECT fname, lname, DOB, sex, language "
            "FROM patient_data WHERE fname='Maria' AND lname='Gonzalez' LIMIT 1;"
        )
        if not out:
            check("4. Patient demographics", 2, False, "patient not found")
            return

        cols = out.split("\t")
        issues = []
        if len(cols) < 5:
            issues.append(f"cols={len(cols)}")
        else:
            if "1985-07-22" not in cols[2]:
                issues.append(f"DOB={cols[2]}")
            if "female" not in cols[3].lower():
                issues.append(f"sex={cols[3]}")
            if "spanish" not in cols[4].lower() and "spa" not in cols[4].lower():
                issues.append(f"lang={cols[4]}")

        check("4. Patient demographics", 2, not issues,
              "correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("4. Patient demographics", 2, False, f"exception: {e}")


# ── Check 5: Patient contact info ────────────────────────────────────────────
def check_5_contact() -> None:
    """Patient address contains Oakwood, phone 413-555-0198, email maria.gonzalez@example.com."""
    try:
        p = pid()
        if not p:
            check("5. Contact info", 1, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT street, postal_code, city, state, phone_home, phone_cell, email "
            f"FROM patient_data WHERE pid={p};"
        )
        lo = out.lower()
        issues = []
        if "oakwood" not in lo:
            issues.append("address")
        if "413-555-0198" not in out and "4135550198" not in out:
            issues.append("phone")
        if "maria.gonzalez@example.com" not in lo:
            issues.append("email")

        check("5. Contact info", 1, not issues,
              "correct" if not issues else f"missing: {issues}")
    except Exception as e:
        check("5. Contact info", 1, False, f"exception: {e}")


# ── Check 6: Primary provider ────────────────────────────────────────────────
def check_6_provider() -> None:
    """Primary provider is Dr. Elizbeth Dickinson."""
    try:
        p = pid()
        if not p:
            check("6. Primary provider", 1, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT u.fname, u.lname FROM users u "
            f"WHERE u.id IN (SELECT providerID FROM patient_data WHERE pid={p}) "
            f"AND u.id > 0;"
        )
        ok = "dickinson" in out.lower()
        check("6. Primary provider", 1, ok,
              out.strip()[:80] if out.strip() else "none assigned")
    except Exception as e:
        check("6. Primary provider", 1, False, f"exception: {e}")


# ── Check 7: Patient history ─────────────────────────────────────────────────
def check_7_history() -> None:
    """Past medical, family, tobacco, and alcohol history recorded."""
    try:
        p = pid()
        if not p:
            check("7. Patient history", 2, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT * FROM history_data WHERE pid={p} ORDER BY id DESC LIMIT 1;"
        )
        lo = out.lower()
        issues = []
        if "asthma" not in lo and "appendectomy" not in lo:
            issues.append("past_medical")
        if "diabetes" not in lo and "hypertension" not in lo and "breast cancer" not in lo:
            issues.append("family")
        if "never" not in lo:
            issues.append("tobacco")
        if "social" not in lo and "1-2" not in lo:
            issues.append("alcohol")

        check("7. Patient history", 2, not issues,
              "all present" if not issues else f"missing: {issues}")
    except Exception as e:
        check("7. Patient history", 2, False, f"exception: {e}")


# ── Check 8: Issue — Essential Hypertension I10 ──────────────────────────────
def check_8_hypertension() -> None:
    """Active medical problem Essential Hypertension with ICD-10 I10."""
    try:
        p = pid()
        if not p:
            check("8. Issue: Hypertension I10", 1, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT title, diagnosis, activity FROM lists "
            f"WHERE pid={p} AND type='medical_problem' AND diagnosis LIKE '%I10%';"
        )
        ok = "I10" in out and bool(out.strip())
        check("8. Issue: Hypertension I10", 1, ok,
              out.strip()[:80] if out.strip() else "not found")
    except Exception as e:
        check("8. Issue: Hypertension I10", 1, False, f"exception: {e}")


# ── Check 9: Issue — Type 2 Diabetes E11.9 ───────────────────────────────────
def check_9_diabetes() -> None:
    """Active medical problem Type 2 Diabetes with ICD-10 E11.9."""
    try:
        p = pid()
        if not p:
            check("9. Issue: Diabetes E11.9", 1, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT title, diagnosis, activity FROM lists "
            f"WHERE pid={p} AND type='medical_problem' AND diagnosis LIKE '%E11.9%';"
        )
        ok = "E11.9" in out and bool(out.strip())
        check("9. Issue: Diabetes E11.9", 1, ok,
              out.strip()[:80] if out.strip() else "not found")
    except Exception as e:
        check("9. Issue: Diabetes E11.9", 1, False, f"exception: {e}")


# ── Check 10: Encounter vitals ───────────────────────────────────────────────
def check_10_vitals() -> None:
    """Vitals: BP 138/88, pulse 76, temp 98.4, height 64, weight 168."""
    try:
        p = pid()
        if not p:
            check("10. Encounter vitals", 2, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT fv.bps, fv.bpd, fv.pulse, fv.temperature, fv.height, fv.weight "
            f"FROM form_vitals fv "
            f"JOIN forms f ON f.form_id = fv.id AND f.formdir = 'vitals' "
            f"WHERE f.pid = {p} ORDER BY fv.id DESC LIMIT 1;"
        )
        if not out.strip():
            check("10. Encounter vitals", 2, False, "no vitals found")
            return

        c = out.strip().split("\t")
        issues = []
        if len(c) >= 6:
            if c[0].strip() != "138":
                issues.append(f"sys={c[0].strip()}")
            if c[1].strip() != "88":
                issues.append(f"dia={c[1].strip()}")
            if c[2].strip() != "76":
                issues.append(f"pulse={c[2].strip()}")
            if "98.4" not in c[3]:
                issues.append(f"temp={c[3].strip()}")
            if c[4].strip() != "64":
                issues.append(f"ht={c[4].strip()}")
            if c[5].strip() != "168":
                issues.append(f"wt={c[5].strip()}")
        else:
            issues.append(f"unexpected cols={len(c)}")

        check("10. Encounter vitals", 2, not issues,
              "BP 138/88, pulse 76, temp 98.4, ht 64, wt 168" if not issues else "; ".join(issues))
    except Exception as e:
        check("10. Encounter vitals", 2, False, f"exception: {e}")


# ── Check 11: Care Plan ──────────────────────────────────────────────────────
def check_11_care_plan() -> None:
    """Care Plan goal: BP below 130/80 and HbA1c below 7.0 within 6 months."""
    try:
        p = pid()
        if not p:
            check("11. Care Plan", 2, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT fcp.description FROM form_care_plan fcp "
            f"JOIN forms f ON f.form_id = fcp.id AND f.formdir = 'care_plan' "
            f"WHERE f.pid = {p} ORDER BY fcp.id DESC LIMIT 5;"
        )
        if not out.strip():
            # Fallback: try querying by pid directly
            out = openemr_sql(
                f"SELECT description FROM form_care_plan WHERE pid = {p} "
                f"ORDER BY id DESC LIMIT 5;"
            )

        lo = out.lower()
        ok = "130/80" in lo or ("hba1c" in lo.replace(" ", "") and "bp" in lo)
        check("11. Care Plan", 2, ok,
              "goal found" if ok else f"not found in: {out[:150]}")
    except Exception as e:
        check("11. Care Plan", 2, False, f"exception: {e}")


# ── Check 12: Clinical Instructions ──────────────────────────────────────────
def check_12_clinical_instructions() -> None:
    """Clinical Instructions contain orientation and follow-up text."""
    try:
        p = pid()
        if not p:
            check("12. Clinical Instructions", 1, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT fci.instruction FROM form_clinical_instructions fci "
            f"JOIN forms f ON f.form_id = fci.id AND f.formdir = 'clinical_instructions' "
            f"WHERE f.pid = {p} ORDER BY fci.id DESC LIMIT 1;"
        )
        if not out.strip():
            out = openemr_sql(
                f"SELECT instruction FROM form_clinical_instructions "
                f"WHERE pid = {p} ORDER BY id DESC LIMIT 1;"
            )

        lo = out.lower()
        ok = ("orientation" in lo or "patient portal" in lo or
              "lab work" in lo or "follow-up" in lo or "follow up" in lo)
        check("12. Clinical Instructions", 1, ok,
              "found" if ok else f"not found in: {out[:150]}")
    except Exception as e:
        check("12. Clinical Instructions", 1, False, f"exception: {e}")


# ── Check 13: Fee Sheet Z00.00 ───────────────────────────────────────────────
def check_13_fee_sheet() -> None:
    """Fee Sheet contains ICD-10 code Z00.00."""
    try:
        p = pid()
        if not p:
            check("13. Fee Sheet Z00.00", 1, False, "no patient")
            return

        out = openemr_sql(
            f"SELECT code FROM billing "
            f"WHERE pid = {p} AND code = 'Z00.00' AND activity = 1;"
        )
        ok = "Z00.00" in out
        check("13. Fee Sheet Z00.00", 1, ok,
              "found" if ok else "not found")
    except Exception as e:
        check("13. Fee Sheet Z00.00", 1, False, f"exception: {e}")


# ── OnlyOffice helpers ────────────────────────────────────────────────────────
_oo_token: str | None = None


def oo_auth() -> str:
    global _oo_token
    if _oo_token is None:
        try:
            r = requests.post(
                f"http://{HOST}:{ONLYOFFICE_PORT}/api/2.0/authentication",
                json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
                timeout=15,
            )
            r.raise_for_status()
            _oo_token = r.json().get("response", {}).get("token", "") or ""
        except Exception:
            _oo_token = ""
    return _oo_token


def oo_find_doc() -> dict | None:
    token = oo_auth()
    if not token:
        return None
    try:
        r = requests.get(
            f"http://{HOST}:{ONLYOFFICE_PORT}/api/2.0/files/@search/Welcome Letter Maria Gonzalez",
            headers={"Authorization": token},
            timeout=15,
        )
        r.raise_for_status()
        for item in r.json().get("response", []):
            t = (item.get("title") or "").lower()
            if "welcome" in t and "gonzalez" in t:
                return item
    except Exception:
        pass
    return None


# ── Check 14: OnlyOffice document exists ──────────────────────────────────────
def check_14_doc_exists() -> None:
    """Document 'Welcome Letter - Maria Gonzalez' exists in OnlyOffice."""
    try:
        doc = oo_find_doc()
        if doc:
            check("14. OnlyOffice doc exists", 1, True,
                  f"title={doc.get('title', '')}")
            return

        # DB fallback: try files_file table, then files table
        for tbl in ("files_file", "files"):
            try:
                _, out, _ = docker_exec(
                    ONLYOFFICE_DB_CONTAINER,
                    "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
                    "-D", "onlyoffice", "--default-character-set=utf8mb4", "-N", "-e",
                    f"SELECT id, title FROM {tbl} WHERE title LIKE '%Welcome%Gonzalez%' LIMIT 1;",
                )
                if out.strip():
                    check("14. OnlyOffice doc exists", 1, True,
                          f"DB: {out.strip()[:80]}")
                    return
            except Exception:
                continue

        check("14. OnlyOffice doc exists", 1, False, "not found via API or DB")
    except Exception as e:
        check("14. OnlyOffice doc exists", 1, False, f"exception: {e}")


# ── Check 15: OnlyOffice document content ─────────────────────────────────────
def check_15_doc_content() -> None:
    """Document contains key content: clinic name, provider, health goal, next steps."""
    try:
        # Find docx files on the container filesystem and extract text
        _, find_out, _ = docker_exec(
            ONLYOFFICE_CONTAINER, "bash", "-c",
            "find /var/www/onlyoffice/Data -name '*.docx' 2>/dev/null | head -30",
            timeout=15,
        )

        for fpath in find_out.strip().split("\n"):
            if not fpath.strip():
                continue
            _, xml, _ = docker_exec(
                ONLYOFFICE_CONTAINER, "bash", "-c",
                f"unzip -p '{fpath}' word/document.xml 2>/dev/null || echo ''",
                timeout=15,
            )
            if not xml or "gonzalez" not in xml.lower():
                continue

            lo = xml.lower()
            found, missing = [], []
            for substr, lbl in [
                ("springfield community health clinic", "clinic_name"),
                ("dickinson", "provider"),
                ("130/80", "bp_goal"),
                ("portal registration", "next_step"),
            ]:
                (found if substr in lo else missing).append(lbl)

            check("15. OnlyOffice doc content", 2, len(missing) == 0,
                  f"found: {found}" if not missing else f"missing: {missing}, found: {found}")
            return

        check("15. OnlyOffice doc content", 2, False,
              "no matching docx found on filesystem")
    except Exception as e:
        check("15. OnlyOffice doc content", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_form_exists()
    check_2_form_settings()
    check_3_form_fields()
    check_4_demographics()
    check_5_contact()
    check_6_provider()
    check_7_history()
    check_8_hypertension()
    check_9_diabetes()
    check_10_vitals()
    check_11_care_plan()
    check_12_clinical_instructions()
    check_13_fee_sheet()
    check_14_doc_exists()
    check_15_doc_content()

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
