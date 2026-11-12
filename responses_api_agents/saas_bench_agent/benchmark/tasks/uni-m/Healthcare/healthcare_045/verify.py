# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Healthcare/healthcare_045/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Healthcare-045-I3: Q4 2026 Patient Access & Safety Experience Survey Deployment

Checks: 12 weighted checks across opnform, openemr, onlyoffice.
Strategy: docker exec (DB) for OpnForm and OpenEMR; REST API for OnlyOffice.

Required env vars:
  SERVER_HOSTNAME, OPNFORM_PORT, OPNFORM_CONTAINER,
  OPENEMR_PORT, OPENEMR_CONTAINER, OPENEMR_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess
import json
import io
import zipfile
import xml.etree.ElementTree as ET

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

_required = [
    "OPNFORM_PORT", "OPNFORM_CONTAINER",
    "OPENEMR_PORT", "OPENEMR_CONTAINER", "OPENEMR_DB_CONTAINER",
    "ONLYOFFICE_PORT", "ONLYOFFICE_CONTAINER", "ONLYOFFICE_DB_CONTAINER",
]
for _v in _required:
    if not os.environ.get(_v):
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


def opnform_sql(query: str) -> str:
    """Query OpnForm PostgreSQL (embedded in app container)."""
    rc, out, err = docker_exec(
        OPNFORM_CONTAINER,
        "psql", "-U", "forge", "-d", "forge", "-t", "-A", "-c", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"psql error: {err.strip()}")
    return out.strip()


def openemr_sql(query: str) -> str:
    """Query OpenEMR MariaDB."""
    rc, out, err = docker_exec(
        OPENEMR_DB_CONTAINER,
        "mysql", "-u", "openemr", "-popenemr_pass", "openemr",
        "--default-character-set=utf8mb4", "-N", "-B", "-e", query,
        timeout=15,
    )
    if rc != 0:
        raise RuntimeError(f"mysql error: {err.strip()}")
    return out.strip()


# ── OpnForm: cached form lookup ───────────────────────────────────────────────
FORM_TITLE = "Q4 2026 Patient Access & Safety Experience Survey"
_form_cache: dict | None = None


def _get_form() -> dict:
    global _form_cache
    if _form_cache is not None:
        return _form_cache
    row = opnform_sql(
        f"SELECT row_to_json(f) FROM forms f WHERE f.title = '{FORM_TITLE}' LIMIT 1;"
    )
    if not row:
        raise RuntimeError("Form not found")
    _form_cache = json.loads(row)
    return _form_cache


def _get_form_props() -> list[dict]:
    form = _get_form()
    props = form.get("properties", [])
    if isinstance(props, str):
        props = json.loads(props)
    return props


# ── Check 1 ───────────────────────────────────────────────────────────────────
def check_1_form_exists() -> None:
    """Form exists with correct title."""
    try:
        form = _get_form()
        ok = form.get("title") == FORM_TITLE
        check("1. OpnForm: form exists", 1, ok, f"title={form.get('title')!r}")
    except Exception as e:
        check("1. OpnForm: form exists", 1, False, f"exception: {e}")


# ── Check 2 ───────────────────────────────────────────────────────────────────
def check_2_form_settings() -> None:
    """Form settings: theme minimal, color #10B981, size md, border small, confetti, re-fillable, submitted text, no indexing."""
    try:
        form = _get_form()
        issues = []

        if form.get("theme") != "minimal":
            issues.append(f"theme={form.get('theme')!r}")
        if form.get("color") != "#10B981":
            issues.append(f"color={form.get('color')!r}")
        if form.get("size") != "md":
            issues.append(f"size={form.get('size')!r}")
        if form.get("border_radius") != "small":
            issues.append(f"border_radius={form.get('border_radius')!r}")
        if not form.get("confetti_on_submission"):
            issues.append("confetti off")
        if not form.get("re_fillable"):
            issues.append("re_fillable off")

        expected_ty = "Thank you for helping us enhance your care journey!"
        submitted = str(form.get("submitted_text", ""))
        if expected_ty not in submitted:
            issues.append("submitted_text mismatch")

        # Indexing disabled: check multiple possible column names
        no_index = form.get("no_index")
        if no_index is None:
            can_index = form.get("can_be_indexed")
            if can_index is True:
                issues.append("indexing not disabled")
        elif not no_index:
            issues.append("no_index=False")

        check("2. OpnForm: form settings", 2, not issues,
              "all correct" if not issues else "; ".join(issues))
    except Exception as e:
        check("2. OpnForm: form settings", 2, False, f"exception: {e}")


# ── Check 3 ───────────────────────────────────────────────────────────────────
def check_3_field_count_and_types() -> None:
    """Form has ≥14 fields with key types present."""
    try:
        props = _get_form_props()
        types_found = {p.get("type", "") for p in props}
        count = len(props)

        # Expected types (names may vary slightly across OpnForm versions)
        needed = {"date", "text", "rating", "scale", "select", "matrix",
                  "slider", "checkbox", "phone_number", "email", "signature"}
        # Also accept alternate names
        alt = {"phone_number": "phone", "nf-page-break": "page_break"}
        missing = []
        for t in needed:
            if t not in types_found and alt.get(t, "") not in types_found:
                missing.append(t)

        ok = count >= 14 and len(missing) <= 2
        detail = f"count={count}, types={sorted(types_found)}"
        if missing:
            detail += f", missing={missing}"
        check("3. OpnForm: field count & types", 2, ok, detail)
    except Exception as e:
        check("3. OpnForm: field count & types", 2, False, f"exception: {e}")


# ── Check 4 ───────────────────────────────────────────────────────────────────
def check_4_matrix_field() -> None:
    """Matrix field has 4 service-area rows and 4 rating columns."""
    try:
        props = _get_form_props()
        matrix = next((p for p in props if p.get("type") == "matrix"), None)
        if not matrix:
            check("4. OpnForm: matrix field", 2, False, "no matrix field found")
            return

        rows = matrix.get("rows", [])
        columns = matrix.get("columns", [])
        row_names = [r.get("name", r) if isinstance(r, dict) else str(r) for r in rows]
        col_names = [c.get("name", c) if isinstance(c, dict) else str(c) for c in columns]
        rn_lower = [r.lower() for r in row_names]
        cn_lower = [c.lower() for c in col_names]

        exp_rows = ["wayfinding", "medical records", "specialist referral", "patient safety"]
        exp_cols = ["poor", "fair", "good", "excellent"]

        rows_ok = all(any(er in r for r in rn_lower) for er in exp_rows)
        cols_ok = all(any(ec in c for c in cn_lower) for ec in exp_cols)

        check("4. OpnForm: matrix field", 2, rows_ok and cols_ok,
              f"rows={row_names}, cols={col_names}")
    except Exception as e:
        check("4. OpnForm: matrix field", 2, False, f"exception: {e}")


# ── Check 5 ───────────────────────────────────────────────────────────────────
def check_5_conditional_fields() -> None:
    """Conditional fields: Problem Description, Contact Email, Contact Phone."""
    try:
        props = _get_form_props()
        cond_count = 0
        for p in props:
            name = (p.get("name") or "").lower()
            logic = p.get("logic") or p.get("conditionalLogic") or {}
            has_cond = bool(logic) and logic != {} and logic != {"conditions": None, "actions": []}

            if "problem" in name and "description" in name and has_cond:
                cond_count += 1
            elif "contact" in name and "email" in name and has_cond:
                cond_count += 1
            elif "contact" in name and "phone" in name and has_cond:
                cond_count += 1

        ok = cond_count >= 3
        check("5. OpnForm: conditional fields", 2, ok,
              f"found {cond_count}/3 conditional fields")
    except Exception as e:
        check("5. OpnForm: conditional fields", 2, False, f"exception: {e}")


# ── Check 6 ───────────────────────────────────────────────────────────────────
def check_6_form_public() -> None:
    """Form visibility is public."""
    try:
        form = _get_form()
        vis = form.get("visibility", "")
        check("6. OpnForm: public visibility", 1, vis == "public",
              f"visibility={vis!r}")
    except Exception as e:
        check("6. OpnForm: public visibility", 1, False, f"exception: {e}")


# ── Check 7 ───────────────────────────────────────────────────────────────────
def check_7_email_notification() -> None:
    """Email notification integration to patient.access@clinic.local."""
    try:
        form = _get_form()
        form_id = form.get("id")
        target = "patient.access@clinic.local"
        found = False

        # Method 1: check notification_emails on form
        notif = str(form.get("notification_emails", ""))
        if target in notif:
            found = True

        # Method 2: check entire form JSON for the target email
        if not found and target in json.dumps(form):
            found = True

        # Method 3: check form_integrations table
        if not found:
            try:
                rows = opnform_sql(
                    f"SELECT data FROM form_integrations WHERE form_id = {form_id};"
                )
                if target in rows:
                    found = True
            except Exception:
                pass

        # Method 4: check integrations column or notification_settings
        if not found:
            try:
                rows = opnform_sql(
                    f"SELECT integration_data FROM form_integrations_events "
                    f"WHERE form_id = {form_id};"
                )
                if target in rows:
                    found = True
            except Exception:
                pass

        check("7. OpnForm: email notification", 2, found,
              f"target={target}, found={found}")
    except Exception as e:
        check("7. OpnForm: email notification", 2, False, f"exception: {e}")


# ── Check 8: OpenEMR ─────────────────────────────────────────────────────────
def check_8_office_note() -> None:
    """OpenEMR office note with survey launch text."""
    try:
        result = openemr_sql(
            "SELECT body FROM onotes WHERE body LIKE "
            "'%Q4 2026 Patient Access%Safety Experience Survey Program%' LIMIT 1;"
        )
        expected = "Q4 2026 Patient Access & Safety Experience Survey Program officially launched today"
        ok = expected.lower() in result.lower() if result else False
        check("8. OpenEMR: office note", 2, ok,
              f"found={bool(result)}, len={len(result) if result else 0}")
    except Exception as e:
        check("8. OpenEMR: office note", 2, False, f"exception: {e}")


# ── OnlyOffice helpers ────────────────────────────────────────────────────────
SPREADSHEET_TITLE = "Q4 2026 Patient Access & Safety Experience Survey Analysis"
_oo_session_cache: tuple[requests.Session, str] | None = None


def _oo_session() -> tuple[requests.Session, str]:
    global _oo_session_cache
    if _oo_session_cache is not None:
        return _oo_session_cache
    base = f"http://{HOST}:{ONLYOFFICE_PORT}"
    sess = requests.Session()
    resp = sess.post(
        f"{base}/api/2.0/authentication",
        json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
        timeout=15,
    )
    resp.raise_for_status()
    token = resp.json().get("response", {}).get("token", "")
    if not token:
        raise RuntimeError(f"No auth token: {resp.text[:200]}")
    sess.headers["Authorization"] = token
    _oo_session_cache = (sess, base)
    return sess, base


def _oo_find_file(sess: requests.Session, base: str) -> dict | None:
    """Search OnlyOffice for the spreadsheet."""
    resp = sess.get(
        f"{base}/api/2.0/files/@search/{requests.utils.quote(SPREADSHEET_TITLE)}",
        timeout=15,
    )
    resp.raise_for_status()
    entries = resp.json().get("response", [])
    for e in entries:
        t = e.get("title", "")
        if SPREADSHEET_TITLE in t:
            return e
    return entries[0] if entries else None


def _oo_download(sess: requests.Session, base: str, file_id: int) -> bytes:
    """Download a file from OnlyOffice as raw bytes."""
    url = f"{base}/Products/Files/HttpHandlers/filehandler.ashx?action=download&fileid={file_id}"
    resp = sess.get(url, timeout=30, allow_redirects=True)
    if resp.status_code == 200 and len(resp.content) > 100:
        return resp.content
    # Fallback: API download endpoint
    url2 = f"{base}/api/2.0/files/file/{file_id}/download"
    resp2 = sess.get(url2, timeout=30, allow_redirects=True)
    resp2.raise_for_status()
    return resp2.content


def _xlsx_sheet_names(data: bytes) -> list[str]:
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read("xl/workbook.xml"))
    return [s.get("name", "") for s in root.findall(".//s:sheets/s:sheet", ns)]


def _xlsx_shared_strings(data: bytes) -> list[str]:
    ns_t = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        try:
            raw = z.read("xl/sharedStrings.xml")
        except KeyError:
            return []
    root = ET.fromstring(raw)
    strings = []
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    for si in root.findall(".//s:si", ns):
        parts = [t.text or "" for t in si.iter(ns_t)]
        strings.append("".join(parts))
    return strings


# ── Check 9 ───────────────────────────────────────────────────────────────────
def check_9_spreadsheet_exists() -> None:
    """OnlyOffice spreadsheet with correct title exists."""
    try:
        sess, base = _oo_session()
        entry = _oo_find_file(sess, base)
        ok = entry is not None
        check("9. OnlyOffice: spreadsheet exists", 1, ok,
              entry.get("title", "?") if entry else "not found")
    except Exception as e:
        check("9. OnlyOffice: spreadsheet exists", 1, False, f"exception: {e}")


# ── Check 10 ──────────────────────────────────────────────────────────────────
def check_10_sheet_names() -> None:
    """Spreadsheet has 3 sheets: Raw Responses, Quality Metrics, Safety Action Plan."""
    try:
        sess, base = _oo_session()
        entry = _oo_find_file(sess, base)
        if not entry:
            check("10. OnlyOffice: sheet names", 2, False, "spreadsheet not found")
            return
        data = _oo_download(sess, base, entry["id"])
        sheets = _xlsx_sheet_names(data)
        expected = ["Raw Responses", "Quality Metrics", "Safety Action Plan"]
        ok = all(any(e.lower() == s.lower() for s in sheets) for e in expected)
        check("10. OnlyOffice: sheet names", 2, ok, f"sheets={sheets}")
    except Exception as e:
        check("10. OnlyOffice: sheet names", 2, False, f"exception: {e}")


# ── Check 11 ──────────────────────────────────────────────────────────────────
def check_11_raw_response_data() -> None:
    """Sheet 1 has 5 response rows with expected provider names."""
    try:
        sess, base = _oo_session()
        entry = _oo_find_file(sess, base)
        if not entry:
            check("11. OnlyOffice: raw response data", 2, False, "spreadsheet not found")
            return
        data = _oo_download(sess, base, entry["id"])
        strings = [s.lower() for s in _xlsx_shared_strings(data)]
        providers = ["dr. pouros", "dr. dickinson", "dr. kuhic", "dr. reinger", "dr. hartmann"]
        found = [p for p in providers if any(p in s for s in strings)]
        ok = len(found) >= 4
        check("11. OnlyOffice: raw response data", 2, ok,
              f"providers {len(found)}/5: {found}")
    except Exception as e:
        check("11. OnlyOffice: raw response data", 2, False, f"exception: {e}")


# ── Check 12 ──────────────────────────────────────────────────────────────────
def check_12_action_plan() -> None:
    """Sheet 3 has 4 service-area rows with improvement actions."""
    try:
        sess, base = _oo_session()
        entry = _oo_find_file(sess, base)
        if not entry:
            check("12. OnlyOffice: action plan", 2, False, "spreadsheet not found")
            return
        data = _oo_download(sess, base, entry["id"])
        strings = [s.lower() for s in _xlsx_shared_strings(data)]
        areas = ["wayfinding", "medical records", "specialist referral", "patient safety"]
        actions = ["signage", "portal", "referral coordinator", "medication reconciliation"]
        areas_found = [a for a in areas if any(a in s for s in strings)]
        actions_found = [a for a in actions if any(a in s for s in strings)]
        ok = len(areas_found) >= 4 and len(actions_found) >= 3
        check("12. OnlyOffice: action plan", 2, ok,
              f"areas={len(areas_found)}/4, actions={len(actions_found)}/4")
    except Exception as e:
        check("12. OnlyOffice: action plan", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_form_exists()
    check_2_form_settings()
    check_3_field_count_and_types()
    check_4_matrix_field()
    check_5_conditional_fields()
    check_6_form_public()
    check_7_email_notification()
    check_8_office_note()
    check_9_spreadsheet_exists()
    check_10_sheet_names()
    check_11_raw_response_data()
    check_12_action_plan()

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
