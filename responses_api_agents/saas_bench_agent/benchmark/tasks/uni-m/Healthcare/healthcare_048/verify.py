# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_048/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-048-I1: Immunization History Audit & Catch-Up Schedule

Checks: 9 weighted checks across openemr, opnform, onlyoffice.
Strategy: docker exec (MariaDB for OpenEMR, Postgres for OpnForm, MySQL for OnlyOffice)

Required env vars:
  SERVER_HOSTNAME,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  OPNFORM_PORT, OPNFORM_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import json
import os
import subprocess
import sys

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OPENEMR_PORT = os.getenv("OPENEMR_PORT")
OPENEMR_CONTAINER = os.getenv("OPENEMR_CONTAINER")
OPENEMR_DB_CONTAINER = os.getenv("OPENEMR_DB_CONTAINER")

OPNFORM_PORT = os.getenv("OPNFORM_PORT")
OPNFORM_CONTAINER = os.getenv("OPNFORM_CONTAINER")

ONLYOFFICE_PORT = os.getenv("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.getenv("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.getenv("ONLYOFFICE_DB_CONTAINER")

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
for var, val in _required.items():
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
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
    """Run a MariaDB query against OpenEMR and return stdout."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "openemr", "-popenemr_pass", "-D", "openemr",
        "-N", "-e", query,
    )
    return out.strip()


def opnform_sql(query: str) -> str:
    """Run a Postgres query against OpnForm (embedded in app container)."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "psql", "-U", "forge", "-d", "forge", "-t", "-A", "-c", query,
    )
    return out.strip()


def onlyoffice_sql(query: str) -> str:
    """Run a MySQL query against OnlyOffice."""
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "-D", "onlyoffice", "-N", "-e", query,
    )
    return out.strip()


# ── OpenEMR appointment checks ───────────────────────────────────────────────

# Patient / provider / category slot values
PATIENTS = [
    {
        "name": "Cyrstal Labadie",
        "date": "2026-05-12",
        "time": "09:30:00",
        "provider_lname": "Pouros",
        "comment_substr": "Immunization catch-up visit",
        "label": "Cyrstal Labadie",
    },
    {
        "name": "Numbers Mohr",
        "date": "2026-05-13",
        "time": "10:00:00",
        "provider_lname": "Dickinson",
        "comment_substr": "Catch-up immunization appointment",
        "label": "Numbers Mohr",
    },
    {
        "name": "Adrianne Simonis",
        "date": "2026-05-14",
        "time": "14:00:00",
        "provider_lname": "Reinger",
        "comment_substr": "Catch-up vaccination visit",
        "label": "Adrianne Simonis",
    },
]


def _check_appointment(idx: int, p: dict) -> None:
    """Verify an OpenEMR calendar appointment for a given patient."""
    label = f"{idx}. Appointment for {p['label']}"
    try:
        # Find patient ID
        fname, lname = p["name"].split(" ", 1)
        pid_raw = openemr_sql(
            f"SELECT id FROM patient_data WHERE fname='{fname}' AND lname='{lname}' LIMIT 1"
        )
        if not pid_raw:
            check(label, 2, False, f"patient '{p['name']}' not found in patient_data")
            return
        pid = pid_raw.strip().split("\n")[0].strip()

        # Find provider ID
        prov_raw = openemr_sql(
            f"SELECT id FROM users WHERE lname='{p['provider_lname']}' LIMIT 1"
        )
        if not prov_raw:
            check(label, 2, False, f"provider '{p['provider_lname']}' not found")
            return
        prov_id = prov_raw.strip().split("\n")[0].strip()

        # Find "Office Visit" category ID
        cat_raw = openemr_sql(
            "SELECT pc_catid FROM openemr_postcalendar_categories WHERE pc_catname='Office Visit' LIMIT 1"
        )
        cat_id = cat_raw.strip().split("\n")[0].strip() if cat_raw else ""

        # Query appointment
        appt_raw = openemr_sql(
            f"SELECT pc_eventDate, pc_startTime, pc_aid, pc_catid, pc_hometext "
            f"FROM openemr_postcalendar_events "
            f"WHERE pc_pid='{pid}' AND pc_eventDate='{p['date']}' "
            f"ORDER BY pc_eid DESC LIMIT 1"
        )
        if not appt_raw:
            check(label, 2, False, f"no appointment on {p['date']} for pid={pid}")
            return

        # Parse tab-separated row
        parts = appt_raw.split("\t")
        if len(parts) < 5:
            # Try newline-split first row
            parts = appt_raw.split("\n")[0].split("\t")

        appt_date = parts[0].strip() if len(parts) > 0 else ""
        appt_time = parts[1].strip() if len(parts) > 1 else ""
        appt_aid = parts[2].strip() if len(parts) > 2 else ""
        appt_catid = parts[3].strip() if len(parts) > 3 else ""
        appt_comment = parts[4].strip() if len(parts) > 4 else ""

        issues = []
        if appt_date != p["date"]:
            issues.append(f"date={appt_date}")
        if not appt_time.startswith(p["time"][:5]):
            issues.append(f"time={appt_time} expected {p['time'][:5]}")
        if str(appt_aid) != str(prov_id):
            issues.append(f"provider_id={appt_aid} expected {prov_id}")
        if cat_id and str(appt_catid) != str(cat_id):
            issues.append(f"catid={appt_catid} expected {cat_id}")
        if p["comment_substr"].lower() not in appt_comment.lower():
            issues.append(f"comment missing substring '{p['comment_substr']}'")

        check(label, 2, not issues, ", ".join(issues) if issues else "all fields match")
    except Exception as e:
        check(label, 2, False, f"exception: {e}")


def check_1_appointment_cyrstal():
    """Appointment for Cyrstal Labadie on 2026-05-12."""
    _check_appointment(1, PATIENTS[0])


def check_2_appointment_numbers():
    """Appointment for Numbers Mohr on 2026-05-13."""
    _check_appointment(2, PATIENTS[1])


def check_3_appointment_adrianne():
    """Appointment for Adrianne Simonis on 2026-05-14."""
    _check_appointment(3, PATIENTS[2])


# ── OpnForm checks ───────────────────────────────────────────────────────────

FORM_TITLE = "Immunization Catch-Up Consent Form 2026"


def _get_form_row() -> dict | None:
    """Fetch the OpnForm form row as a dict."""
    raw = opnform_sql(
        f"SELECT row_to_json(f) FROM forms f WHERE title = '{FORM_TITLE}' LIMIT 1"
    )
    if not raw:
        return None
    try:
        return json.loads(raw.split("\n")[0])
    except (json.JSONDecodeError, IndexError):
        return None


def check_4_form_exists():
    """OpnForm form with correct title exists."""
    try:
        row = _get_form_row()
        check("4. OpnForm form exists", 1, row is not None,
              f"title='{FORM_TITLE}'" if row else "form not found")
    except Exception as e:
        check("4. OpnForm form exists", 1, False, f"exception: {e}")


def check_5_form_settings():
    """OpnForm form settings: color, theme, size, progress bar, submit text, visibility."""
    try:
        row = _get_form_row()
        if not row:
            check("5. OpnForm form settings", 2, False, "form not found")
            return

        issues = []
        if row.get("color", "").upper() != "#2563EB":
            issues.append(f"color={row.get('color')}")
        if row.get("theme") != "default":
            issues.append(f"theme={row.get('theme')}")
        if row.get("size") != "lg":
            issues.append(f"size={row.get('size')}")
        if not row.get("show_progress_bar"):
            issues.append("progress bar not enabled")
        if (row.get("submit_button_text") or "").strip() != "Submit Consent Form":
            issues.append(f"submit_text='{row.get('submit_button_text')}'")
        if row.get("visibility") != "public":
            issues.append(f"visibility={row.get('visibility')}")

        check("5. OpnForm form settings", 2, not issues,
              ", ".join(issues) if issues else "all settings correct")
    except Exception as e:
        check("5. OpnForm form settings", 2, False, f"exception: {e}")


def check_6_form_core_fields():
    """OpnForm form has required core fields with correct types."""
    try:
        row = _get_form_row()
        if not row:
            check("6. OpnForm core fields", 3, False, "form not found")
            return

        props = row.get("properties") or []
        if isinstance(props, str):
            props = json.loads(props)

        # Build a lookup: lowered name -> property dict
        fields_by_name: dict[str, dict] = {}
        for prop in props:
            name = (prop.get("name") or prop.get("label") or "").lower().strip()
            if name:
                fields_by_name[name] = prop

        issues = []

        # Check for key fields by scanning names
        def find_field(keywords: list[str]) -> dict | None:
            for name, prop in fields_by_name.items():
                if all(k.lower() in name for k in keywords):
                    return prop
            return None

        # Patient Name (text)
        f = find_field(["patient", "name"])
        if not f:
            issues.append("Patient Name field missing")

        # Date of Birth (date)
        f = find_field(["date", "birth"])
        if not f:
            f = find_field(["dob"])
        if not f:
            issues.append("Date of Birth field missing")

        # Parent/Guardian Name (text)
        f = find_field(["parent"])
        if not f:
            f = find_field(["guardian"])
        if not f:
            issues.append("Parent/Guardian Name field missing")

        # Vaccines Due (multi-select)
        f = find_field(["vaccine"])
        if not f:
            issues.append("Vaccines Due field missing")

        # Consent Decision (select)
        f = find_field(["consent"])
        if not f:
            issues.append("Consent Decision field missing")

        # Egg Allergy checkbox
        f = find_field(["egg"])
        if not f:
            issues.append("Egg Allergy checkbox missing")

        # Adverse Reaction checkbox
        f = find_field(["adverse"])
        if not f:
            f = find_field(["reaction"])
        if not f:
            issues.append("Adverse Reaction checkbox missing")

        check("6. OpnForm core fields", 3, not issues,
              ", ".join(issues) if issues else f"all 7 key fields found in {len(props)} properties")
    except Exception as e:
        check("6. OpnForm core fields", 3, False, f"exception: {e}")


def check_7_form_conditional_and_signature():
    """OpnForm form has conditional fields, page break, and signature."""
    try:
        row = _get_form_row()
        if not row:
            check("7. OpnForm conditional/signature", 2, False, "form not found")
            return

        props = row.get("properties") or []
        if isinstance(props, str):
            props = json.loads(props)

        has_page_break = False
        has_signature = False
        has_conditional = False

        for prop in props:
            ptype = (prop.get("type") or "").lower()

            # Page break
            if "page" in ptype and "break" in ptype:
                has_page_break = True
            if ptype in ("nf-page-break",):
                has_page_break = True

            # Signature
            if "signature" in ptype:
                has_signature = True
            if "signature" in (prop.get("name") or "").lower():
                has_signature = True

            # Conditional logic — look for logic property or conditions
            logic = prop.get("logic") or prop.get("conditionalLogic") or prop.get("conditions")
            if logic:
                has_conditional = True
            # Also check hidden field with conditions
            if prop.get("hidden") and (prop.get("conditions") or prop.get("logic")):
                has_conditional = True

        issues = []
        if not has_page_break:
            issues.append("no page break found")
        if not has_signature:
            issues.append("no signature field found")
        if not has_conditional:
            issues.append("no conditional logic found")

        check("7. OpnForm conditional/signature", 2, not issues,
              ", ".join(issues) if issues else "page break + signature + conditional logic present")
    except Exception as e:
        check("7. OpnForm conditional/signature", 2, False, f"exception: {e}")


def check_8_form_vaccines_options():
    """OpnForm Vaccines Due multi-select has correct options (MMR, Tdap, IPV, Varicella, Hepatitis B)."""
    try:
        row = _get_form_row()
        if not row:
            check("8. OpnForm vaccine options", 2, False, "form not found")
            return

        props = row.get("properties") or []
        if isinstance(props, str):
            props = json.loads(props)

        expected_vaccines = {"mmr", "tdap", "ipv", "varicella", "hepatitis b"}

        vaccine_field = None
        for prop in props:
            name = (prop.get("name") or prop.get("label") or "").lower()
            if "vaccine" in name and "due" in name:
                vaccine_field = prop
                break
        if not vaccine_field:
            # Fallback: find any multi-select with vaccine-related options
            for prop in props:
                ptype = (prop.get("type") or "").lower()
                if "multi" in ptype or "select" in ptype or "checkbox" in ptype:
                    opts = prop.get("options") or prop.get("choices") or []
                    if isinstance(opts, list):
                        opt_names = {(o.get("name") or o.get("value") or str(o)).lower() for o in opts} if opts and isinstance(opts[0], dict) else {str(o).lower() for o in opts}
                        if "mmr" in opt_names or "tdap" in opt_names:
                            vaccine_field = prop
                            break

        if not vaccine_field:
            check("8. OpnForm vaccine options", 2, False, "Vaccines Due field not found")
            return

        # Extract option values
        raw_opts = vaccine_field.get("options") or vaccine_field.get("choices") or []
        if isinstance(raw_opts, list) and raw_opts:
            if isinstance(raw_opts[0], dict):
                found = {(o.get("name") or o.get("value") or "").lower() for o in raw_opts}
            else:
                found = {str(o).lower() for o in raw_opts}
        else:
            found = set()

        missing = expected_vaccines - found
        check("8. OpnForm vaccine options", 2, not missing,
              f"missing options: {missing}" if missing else f"all 5 vaccine options present in {len(raw_opts)} options")
    except Exception as e:
        check("8. OpnForm vaccine options", 2, False, f"exception: {e}")


# ── OnlyOffice checks ────────────────────────────────────────────────────────

SPREADSHEET_TITLE = "Immunization_Audit_CatchUp_2026Q2"


def check_9_spreadsheet_exists():
    """OnlyOffice spreadsheet with correct title exists."""
    try:
        # Search for the file by title in files_file table.
        # Category 5 = spreadsheet. Also try without extension and with .xlsx.
        raw = onlyoffice_sql(
            f"SELECT id, title, category FROM files_file "
            f"WHERE (title = '{SPREADSHEET_TITLE}' "
            f"OR title = '{SPREADSHEET_TITLE}.xlsx' "
            f"OR title LIKE '{SPREADSHEET_TITLE}%') "
            f"AND tenant_id = 1 "
            f"ORDER BY create_on DESC LIMIT 1"
        )
        if not raw:
            check("9. OnlyOffice spreadsheet exists", 1, False, "file not found in files_file")
            return

        parts = raw.split("\t")
        file_id = parts[0].strip() if len(parts) > 0 else ""
        title = parts[1].strip() if len(parts) > 1 else ""
        category = parts[2].strip() if len(parts) > 2 else ""

        title_ok = SPREADSHEET_TITLE.lower() in title.lower()
        check("9. OnlyOffice spreadsheet exists", 1, title_ok,
              f"found id={file_id} title='{title}' category={category}")
    except Exception as e:
        check("9. OnlyOffice spreadsheet exists", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_appointment_cyrstal()
    check_2_appointment_numbers()
    check_3_appointment_adrianne()
    check_4_form_exists()
    check_5_form_settings()
    check_6_form_core_fields()
    check_7_form_conditional_and_signature()
    check_8_form_vaccines_options()
    check_9_spreadsheet_exists()

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
