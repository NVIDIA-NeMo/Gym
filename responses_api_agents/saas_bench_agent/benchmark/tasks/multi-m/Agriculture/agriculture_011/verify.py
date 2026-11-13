#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/tasks/multi-m/Agriculture/agriculture_011/verify.py
# Original license: CC-BY-4.0; Copyright: SaaS-Bench Authors; see bundled licenses.
# NVIDIA modifications: Register Drupal Unicode NOCASE_UTF8 and route PHP failures to stderr; grading SQL unchanged.
# Adaptation: register the shipped Drupal collation for database reads; task checks unchanged.
"""Verify actual FarmOS image evidence and the Grocy handoff for agriculture_011."""

import base64
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request


FARMOS_CONTAINER = os.getenv("FARMOS_CONTAINER")
GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")
FARMOS_DB = "/opt/drupal/web/sites/default/files/.ht.sqlite"
GROCY_DB_CANDIDATES = [
    "/config/data/grocy.db",
    "/config/data/data/grocy.db",
    "/var/www/data/grocy.db",
]
EXPECTED_HASHES = {
    "e6e265e1e1840f67623744ebaf60499c8f2505c777f817cb12ceed49541610d8",
    "9324bf60f73bdfb92ae0792896ca6bf933db9da0b1984670ae22cc088c5f760a",
}
EMERGENCY_NAME = "AG011 - Corn Aphid Emergency Assessment"
INPUT_NAME = "AG011 - Corn Aphid Treatment"
FOLLOWUP_NAME = "AG011 - Corn Aphid Follow-up"
MAINTENANCE_NAME = "AG011 - Sprayer Decontamination"
TREATMENT = "Pyrethrin (OMRI-listed)"

for _name, _value in [
    ("FARMOS_CONTAINER", FARMOS_CONTAINER),
    ("GROCY_CONTAINER", GROCY_CONTAINER),
]:
    if not _value:
        print(f"FATAL: {_name} not set", file=sys.stderr)
        sys.exit(1)


_checks: list[tuple[str, int, bool, str]] = []
_grocy_db_path = ""
_corn_id = 0
_sprayer_id = 0
_emergency: dict = {}
_input_log: dict = {}
_followup: dict = {}
_maintenance: dict = {}
_attached_images: list[bytes] = []
_emergency_ok = False
_visual_ok = False
_treatment_chain_ok = False
_maintenance_ok = False


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{suffix}", file=sys.stderr)


def docker_exec(container: str, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    result = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def farmos_query(sql: str) -> list[dict]:
    php = (
        "require '/opt/drupal/vendor/autoload.php';"
        "$db=new PDO('sqlite:" + FARMOS_DB + "');"
        "$db->sqliteCreateCollation('NOCASE_UTF8',['Drupal\\Component\\Utility\\Unicode','strcasecmp']);"
        "$db->setAttribute(PDO::ATTR_ERRMODE,PDO::ERRMODE_EXCEPTION);"
        "$s=$db->query($argv[1]);"
        "echo json_encode($s->fetchAll(PDO::FETCH_ASSOC));"
    )
    rc, out, err = docker_exec(FARMOS_CONTAINER, "php", "-d", "display_errors=stderr", "-r", php, "--", sql)
    if rc != 0:
        raise RuntimeError(f"FarmOS query failed: {err.strip()[:300]}")
    return json.loads(out) if out.strip() else []


def _find_grocy_db() -> str:
    global _grocy_db_path
    if _grocy_db_path:
        return _grocy_db_path
    for candidate in GROCY_DB_CANDIDATES:
        rc, _, _ = docker_exec(GROCY_CONTAINER, "test", "-f", candidate, timeout=5)
        if rc == 0:
            _grocy_db_path = candidate
            return candidate
    return GROCY_DB_CANDIDATES[0]


def grocy_query(sql: str) -> list[dict]:
    php = (
        '$db=new PDO("sqlite:' + _find_grocy_db() + '");'
        '$db->setAttribute(PDO::ATTR_ERRMODE,PDO::ERRMODE_EXCEPTION);'
        '$s=$db->query($argv[1]);'
        'echo json_encode($s->fetchAll(PDO::FETCH_ASSOC));'
    )
    rc, out, err = docker_exec(GROCY_CONTAINER, "php", "-r", php, "--", sql)
    if rc != 0:
        raise RuntimeError(f"Grocy query failed: {err.strip()[:300]}")
    return json.loads(out) if out.strip() else []


def strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()


def exact_log(name: str, log_type: str) -> dict:
    safe = name.replace("'", "''")
    rows = farmos_query(
        "SELECT id,name,type,timestamp,notes__value FROM log_field_data "
        f"WHERE name='{safe}' AND type='{log_type}' ORDER BY id"
    )
    return rows[0] if len(rows) == 1 else {}


def exact_asset(name: str, asset_type: str) -> int:
    safe = name.replace("'", "''")
    rows = farmos_query(
        "SELECT id FROM asset_field_data "
        f"WHERE name='{safe}' AND type='{asset_type}' ORDER BY id"
    )
    return int(rows[0]["id"]) if len(rows) == 1 else 0


def linked_asset_ids(log_id: int) -> list[int]:
    rows = farmos_query(
        "SELECT asset_target_id FROM log__asset "
        f"WHERE entity_id={log_id} AND deleted=0 ORDER BY delta"
    )
    return [int(row["asset_target_id"]) for row in rows]


def local_date(timestamp: object) -> dt.date | None:
    try:
        return dt.datetime.fromtimestamp(int(timestamp)).date()
    except (TypeError, ValueError, OSError):
        return None


def read_farmos_uri(uri: str) -> bytes:
    roots = {
        "private://": "/opt/drupal/private/",
        "public://": "/opt/drupal/web/sites/default/files/",
    }
    path = ""
    for scheme, root in roots.items():
        if uri.startswith(scheme):
            relative = uri[len(scheme):]
            if relative and ".." not in relative.split("/"):
                path = root + relative.lstrip("/")
            break
    if not path:
        return b""
    rc, encoded, _ = docker_exec(FARMOS_CONTAINER, "base64", path)
    if rc != 0 or not encoded.strip():
        return b""
    try:
        return base64.b64decode(re.sub(r"\s+", "", encoded), validate=True)
    except ValueError:
        return b""


def emergency_attachments(log_id: int) -> list[dict]:
    return farmos_query(
        "SELECT fm.fid,fm.filename,fm.uri,fm.filemime FROM log__image li "
        "JOIN file_managed fm ON fm.fid=li.image_target_id "
        f"WHERE li.entity_id={log_id} AND li.deleted=0 "
        "UNION SELECT fm.fid,fm.filename,fm.uri,fm.filemime FROM log__file lf "
        "JOIN file_managed fm ON fm.fid=lf.file_target_id "
        f"WHERE lf.entity_id={log_id} AND lf.deleted=0 ORDER BY fid"
    )


def vision_high(notes: str) -> tuple[bool, str]:
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
            },
        }
        for image in _attached_images
    ]
    content.append({
        "type": "text",
        "text": (
            "These are the actual two files attached to the target FarmOS emergency log. "
            "Do they jointly meet the High rubric: the overview shows the corn field but "
            "cannot establish insect density at distance, while the close-up shows a dense "
            "aphid mass at the tassel or leaf-sheath base with visible shed skins? Also "
            "require the notes to distinguish the two views accurately and conclude exactly "
            f"Severity: High.\nLOG NOTES: {notes}\nAnswer only YES or NO."
        ),
    })
    payload = {
        "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 32,
    }
    try:
        request = urllib.request.Request(
            os.getenv("MINDRA_BASE_URL", "https://api.mindracode.com/v1") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {os.getenv('MINDRA_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read())
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return False, "vision judge response missing choices"
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return False, "vision judge response missing message"
        content_value = message.get("content")
        if not isinstance(content_value, str) or not content_value.strip():
            return False, "vision judge response missing message content"
        answer = content_value.strip().upper()
        return answer.startswith("YES"), answer[:120]
    except Exception as exc:
        return False, f"vision judge error: {exc}"


def load_context() -> None:
    global _corn_id, _sprayer_id, _emergency, _input_log, _followup, _maintenance
    _corn_id = exact_asset("2023 Sweet Corn Planting 1", "plant")
    _sprayer_id = exact_asset("Tractor-Mounted Boom Sprayer", "equipment")
    _emergency = exact_log(EMERGENCY_NAME, "observation")
    _input_log = exact_log(INPUT_NAME, "input")
    _followup = exact_log(FOLLOWUP_NAME, "observation")
    _maintenance = exact_log(MAINTENANCE_NAME, "maintenance")


def check_1_exact_emergency_and_actual_files() -> None:
    global _emergency_ok, _attached_images
    problems = []
    if not _corn_id:
        problems.append("exact corn asset missing")
    if not _emergency:
        problems.append("expected exactly one emergency observation")
    if _emergency and linked_asset_ids(int(_emergency["id"])) != [_corn_id]:
        problems.append("emergency log is not linked only to the corn asset")
    if _emergency and local_date(_emergency.get("timestamp")) != dt.date.today():
        problems.append("emergency log is not dated today")
    rows = emergency_attachments(int(_emergency["id"])) if _emergency else []
    if len(rows) != 2:
        problems.append(f"expected exactly two attached files, found {len(rows)}")
    _attached_images = [read_farmos_uri(row.get("uri") or "") for row in rows]
    hashes = {hashlib.sha256(data).hexdigest() for data in _attached_images if data}
    if hashes != EXPECTED_HASHES:
        problems.append(f"actual attachment hashes do not match both supplied images: {sorted(hashes)}")
    _emergency_ok = not problems
    check(
        "1. exact emergency log carries both actual image bytes",
        3,
        _emergency_ok,
        f"log_id={_emergency.get('id')} hashes={sorted(hashes)}"
        if _emergency_ok else "; ".join(problems),
    )


def check_2_visual_severity_and_notes() -> None:
    global _visual_ok
    if not _emergency_ok:
        check("2. attached evidence supports recorded High severity", 4, False,
              "gated: emergency evidence invalid")
        return
    notes = strip_html(_emergency.get("notes__value") or "")
    if not re.search(r"(?:^|\s)Severity:\s*High\s*$", notes, re.IGNORECASE):
        check("2. attached evidence supports recorded High severity", 4, False,
              "notes lack exact Severity: High")
        return
    _visual_ok, detail = vision_high(notes)
    check("2. attached evidence supports recorded High severity", 4, _visual_ok, detail)


def check_3_treatment_and_followup_chain() -> None:
    global _treatment_chain_ok
    if not _visual_ok:
        check("3. rubric-matched treatment and follow-up", 2, False,
              "gated: visual classification invalid")
        return
    problems = []
    if not _input_log:
        problems.append("expected exactly one treatment Input log")
    if not _followup:
        problems.append("expected exactly one follow-up Observation log")
    for label, log in [("input", _input_log), ("follow-up", _followup)]:
        if log and linked_asset_ids(int(log["id"])) != [_corn_id]:
            problems.append(f"{label} log is not linked only to corn")
    emergency_date = local_date(_emergency.get("timestamp"))
    if _input_log and local_date(_input_log.get("timestamp")) != emergency_date:
        problems.append("input date differs from emergency date")
    if (_followup and emergency_date
            and local_date(_followup.get("timestamp")) != emergency_date + dt.timedelta(days=7)):
        problems.append("follow-up date is not exactly 7 days later")
    input_text = strip_html(_input_log.get("notes__value") or "").casefold()
    required = ["pyrethrin", "omri-2023-py-001", "li shifu",
                "tractor-mounted boom sprayer"]
    if not all(value in input_text for value in required):
        problems.append("input notes miss treatment, certification, operator, or equipment")
    if not re.search(r"\b200\s*m(?:illi)?l(?:ilit(?:er|re)s?)?\s*/\s*acre\b", input_text):
        problems.append("input notes miss exact 200 mL/acre rate")
    follow_text = strip_html(_followup.get("notes__value") or "").casefold()
    if not (re.search(r"\b70\s*%", follow_text)
            and "monitor" in follow_text and "7 more days" in follow_text):
        problems.append("follow-up notes miss reduction and monitoring decision")
    _treatment_chain_ok = not problems
    check("3. rubric-matched treatment and follow-up", 2, _treatment_chain_ok,
          f"input_id={_input_log.get('id')}" if _treatment_chain_ok else "; ".join(problems))


def check_4_sprayer_decontamination() -> None:
    global _maintenance_ok
    if not _treatment_chain_ok:
        check("4. sprayer decontamination closes FarmOS chain", 1, False,
              "gated: treatment/follow-up incomplete")
        return
    problems = []
    if not _sprayer_id:
        problems.append("exact sprayer asset missing")
    if not _maintenance:
        problems.append("expected exactly one maintenance log")
    if _maintenance and linked_asset_ids(int(_maintenance["id"])) != [_sprayer_id]:
        problems.append("maintenance log is not linked only to sprayer")
    if _maintenance and local_date(_maintenance.get("timestamp")) != dt.date.today():
        problems.append("maintenance log is not dated today")
    text = strip_html(_maintenance.get("notes__value") or "").casefold()
    if not ("water" in text and "rinse" in text and "cross-contamination" in text):
        problems.append("maintenance notes miss water rinse/cross-contamination")
    _maintenance_ok = not problems
    check("4. sprayer decontamination closes FarmOS chain", 1, _maintenance_ok,
          f"log_id={_maintenance.get('id')}" if _maintenance_ok else "; ".join(problems))


def check_5_grocy_generated_id_handoff() -> None:
    if not _maintenance_ok:
        check("5. Grocy supply references generated FarmOS input ID", 10, False,
              "gated: FarmOS chain incomplete")
        return
    product_rows = grocy_query(
        "SELECT id,name FROM products WHERE name='Pyrethrin (OMRI-listed)' ORDER BY id"
    )
    marked = grocy_query(
        "SELECT sl.id,sl.product_id,sl.amount,COALESCE(sl.note,'') AS note,"
        "COALESCE(p.name,'') AS product_name FROM shopping_list sl "
        "LEFT JOIN products p ON p.id=sl.product_id "
        "WHERE sl.note LIKE 'AG011%';"
    )
    expected_note = f"AG011 | FarmOS input #{int(_input_log['id'])} | {TREATMENT}"
    passed = (
        len(product_rows) == 1
        and len(marked) == 1
        and int(marked[0].get("product_id") or 0) == int(product_rows[0]["id"])
        and abs(float(marked[0].get("amount") or 0) - 1.0) < 1e-6
        and marked[0].get("note") == expected_note
        and marked[0].get("product_name") == TREATMENT
    )
    check("5. Grocy supply references generated FarmOS input ID", 10, passed,
          f"products={len(product_rows)} marked_rows={len(marked)} expected='{expected_note}'")


def main() -> None:
    try:
        load_context()
    except Exception as exc:
        print(f"WARNING: context load failed: {exc}", file=sys.stderr)
    check_1_exact_emergency_and_actual_files()
    check_2_visual_severity_and_notes()
    check_3_treatment_and_followup_chain()
    check_4_sprayer_decontamination()
    check_5_grocy_generated_id_handoff()
    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    print(f"SCORE: {earned / total if total else 0:.3f}  PASS: {all_pass}  "
          f"({earned}/{total})", file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
