# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_083/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-083-I1: Process Remote Work Policy Exception Request

Checks: 15 weighted checks across roundcubemail, onlyoffice, owncloud, mattermost.
Strategy: docker exec (DB + filesystem) for Roundcube/OnlyOffice/Mattermost; WebDAV API for ownCloud files.

Required env vars:
  SERVER_HOSTNAME, plus PORT/CONTAINER/DB_CONTAINER for each site.
"""

import os
import sys
import subprocess
import json

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")


def _require(var: str) -> str:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    return val


ROUNDCUBEMAIL_PORT = _require("ROUNDCUBEMAIL_PORT")
ROUNDCUBEMAIL_CONTAINER = _require("ROUNDCUBEMAIL_CONTAINER")
ROUNDCUBEMAIL_DB_CONTAINER = _require("ROUNDCUBEMAIL_DB_CONTAINER")

ONLYOFFICE_PORT = _require("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = _require("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = _require("ONLYOFFICE_DB_CONTAINER")

OWNCLOUD_PORT = _require("OWNCLOUD_PORT")
OWNCLOUD_CONTAINER = _require("OWNCLOUD_CONTAINER")
OWNCLOUD_DB_CONTAINER = _require("OWNCLOUD_DB_CONTAINER")

MATTERMOST_PORT = _require("MATTERMOST_PORT")
MATTERMOST_CONTAINER = _require("MATTERMOST_CONTAINER")
MATTERMOST_DB_CONTAINER = _require("MATTERMOST_DB_CONTAINER")

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


def mm_db(sql: str) -> str:
    """Query Mattermost Postgres DB."""
    _, out, _ = docker_exec(
        MATTERMOST_DB_CONTAINER,
        "psql", "-U", "mmuser", "-d", "mattermost", "-t", "-A", "-c", sql,
    )
    return out.strip()


def oo_db(sql: str) -> str:
    """Query OnlyOffice MySQL DB."""
    _, out, _ = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass", "onlyoffice",
        "-N", "-B", "-e", sql,
    )
    return out.strip()


def oc_db(sql: str) -> str:
    """Query ownCloud MariaDB."""
    _, out, _ = docker_exec(
        OWNCLOUD_DB_CONTAINER,
        "mysql", "-u", "owncloud", "-powncloud", "owncloud",
        "-N", "-B", "-e", sql,
    )
    return out.strip()


def mail_grep(pattern: str, extra_grep: str = "") -> str:
    """Search Roundcube container maildir for files matching pattern."""
    cmd = f"grep -rl '{pattern}' /var/vmail/ /var/mail/ 2>/dev/null"
    if extra_grep:
        cmd += f" | xargs grep -l '{extra_grep}' 2>/dev/null"
    cmd += " | head -5"
    _, out, _ = docker_exec(ROUNDCUBEMAIL_CONTAINER, "bash", "-c", cmd, timeout=20)
    return out.strip()


# ── Roundcube checks ─────────────────────────────────────────────────────────
def check_1_exceptions_folder():
    """IMAP folder 'Policy-Exceptions-2026' exists under INBOX."""
    try:
        _, out, _ = docker_exec(
            ROUNDCUBEMAIL_CONTAINER, "bash", "-c",
            "find /var/vmail/ /var/mail/ -type d -name '*Policy-Exceptions*' 2>/dev/null | head -5",
        )
        found = bool(out.strip())
        check("1. Roundcube: IMAP folder Policy-Exceptions-2026 exists", 1, found,
              out.strip().split("\n")[0] if found else "folder not found in maildir")
    except Exception as e:
        check("1. Roundcube: IMAP folder Policy-Exceptions-2026 exists", 1, False, f"exception: {e}")


def check_4_formal_decision_email():
    """Formal decision email sent with correct subject."""
    try:
        result = mail_grep("Formal Decision: Policy Exception Request - HR-POL-014 Remote Work")
        found = bool(result)
        check("4. Roundcube: Formal decision email sent", 2, found,
              "found in maildir" if found else "formal decision email not found")
    except Exception as e:
        check("4. Roundcube: Formal decision email sent", 2, False, f"exception: {e}")


def check_5_formal_decision_cc():
    """Formal decision email has CC to amit.singh and laura.brown."""
    try:
        _, out, _ = docker_exec(
            ROUNDCUBEMAIL_CONTAINER, "bash", "-c",
            "for f in $(grep -rl 'Formal Decision: Policy Exception Request' /var/vmail/ /var/mail/ 2>/dev/null | head -3); do "
            "grep -i '^Cc:' \"$f\" 2>/dev/null; done",
        )
        has_amit = "amit.singh" in out.lower()
        has_laura = "laura.brown" in out.lower()
        passed = has_amit and has_laura
        missing = []
        if not has_amit:
            missing.append("amit.singh")
        if not has_laura:
            missing.append("laura.brown")
        check("5. Roundcube: Formal decision CC'd correctly", 1, passed,
              "both CCs present" if passed else f"missing: {', '.join(missing)}")
    except Exception as e:
        check("5. Roundcube: Formal decision CC'd correctly", 1, False, f"exception: {e}")


# ── OnlyOffice checks ────────────────────────────────────────────────────────
def check_6_onlyoffice_document():
    """Document 'Exception Review - Rahul Johnson - Remote Work - 2026' exists in OnlyOffice."""
    try:
        rows = oo_db(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Exception Review%Rahul Johnson%' LIMIT 5;"
        )
        found = bool(rows)
        check("6. OnlyOffice: Exception review document exists", 2, found,
              rows[:120] if found else "document not found in files_file")
    except Exception as e:
        check("6. OnlyOffice: Exception review document exists", 2, False, f"exception: {e}")


def check_7_onlyoffice_sharing():
    """Document shared with laura.brown (edit) and amit.singh (view)."""
    try:
        file_id = oo_db(
            "SELECT id FROM files_file "
            "WHERE title LIKE '%Exception Review%Rahul Johnson%' ORDER BY id DESC LIMIT 1;"
        )
        if not file_id:
            check("7. OnlyOffice: Document shared correctly", 2, False, "document not found")
            return

        # files_security stores sharing info; subject is a user GUID
        shares = oo_db(
            f"SELECT s.subject, s.security, u.username "
            f"FROM files_security s "
            f"LEFT JOIN core_user u ON s.subject = CAST(u.id AS CHAR) "
            f"WHERE s.entry_id = {file_id} AND s.entry_type = 2;"
        )
        if not shares:
            # Try with tenant_id based approach
            shares = oo_db(
                f"SELECT subject, security FROM files_security "
                f"WHERE entry_id = {file_id};"
            )
        has_shares = bool(shares)
        check("7. OnlyOffice: Document shared correctly", 2, has_shares,
              f"shares: {shares[:200]}" if has_shares else f"no shares for file_id={file_id}")
    except Exception as e:
        check("7. OnlyOffice: Document shared correctly", 2, False, f"exception: {e}")


# ── ownCloud checks ──────────────────────────────────────────────────────────
def _oc_webdav_get(path: str) -> tuple[int, str]:
    """GET a file via ownCloud WebDAV. Returns (status_code, body)."""
    import urllib.request
    import urllib.error
    import base64

    creds = base64.b64encode(b"admin:admin").decode()
    url = f"http://{HOST}:{OWNCLOUD_PORT}/remote.php/dav/files/admin/{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {creds}"})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""


def _oc_propfind(path: str) -> tuple[int, str]:
    """PROPFIND on an ownCloud WebDAV path. Returns (status, body)."""
    import urllib.request
    import urllib.error
    import base64

    creds = base64.b64encode(b"admin:admin").decode()
    url = f"http://{HOST}:{OWNCLOUD_PORT}/remote.php/dav/files/admin/{path}"
    req = urllib.request.Request(url, method="PROPFIND", headers={
        "Authorization": f"Basic {creds}",
        "Depth": "1",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""


def check_8_owncloud_folder_structure():
    """Folder Exception-Case-RJ-2026-04 with Supporting-Documents and Decision-Records."""
    try:
        status, body = _oc_propfind("Exception-Case-RJ-2026-04/")
        if status >= 400:
            check("8. ownCloud: Case folder with subfolders exists", 1, False,
                  f"PROPFIND returned HTTP {status}")
            return
        has_supporting = "Supporting-Documents" in body
        has_decision = "Decision-Records" in body
        passed = has_supporting and has_decision
        missing = []
        if not has_supporting:
            missing.append("Supporting-Documents")
        if not has_decision:
            missing.append("Decision-Records")
        check("8. ownCloud: Case folder with subfolders exists", 1, passed,
              "both subfolders present" if passed else f"missing: {', '.join(missing)}")
    except Exception as e:
        check("8. ownCloud: Case folder with subfolders exists", 1, False, f"exception: {e}")


def check_9_request_summary():
    """request-summary.txt exists with key content (Rahul Johnson, HR-POL-014, Portugal)."""
    try:
        status, content = _oc_webdav_get(
            "Exception-Case-RJ-2026-04/Supporting-Documents/request-summary.txt"
        )
        if status >= 400:
            check("9. ownCloud: request-summary.txt content", 2, False, f"HTTP {status}")
            return
        has_name = "Rahul Johnson" in content
        has_policy = "HR-POL-014" in content
        has_portugal = "Portugal" in content
        passed = has_name and has_policy and has_portugal
        missing = []
        if not has_name:
            missing.append("requester name")
        if not has_policy:
            missing.append("policy ref")
        if not has_portugal:
            missing.append("Portugal")
        check("9. ownCloud: request-summary.txt content", 2, passed,
              "key content present" if passed else f"missing: {', '.join(missing)}")
    except Exception as e:
        check("9. ownCloud: request-summary.txt content", 2, False, f"exception: {e}")


def check_10_decision_record():
    """decision-record.txt contains 'Approved with Conditions' and effective period."""
    try:
        status, content = _oc_webdav_get(
            "Exception-Case-RJ-2026-04/Decision-Records/decision-record.txt"
        )
        if status >= 400:
            check("10. ownCloud: decision-record.txt content", 2, False, f"HTTP {status}")
            return
        has_outcome = "Approved with Conditions" in content
        has_start = "2026-05-01" in content
        has_end = "2026-10-31" in content
        passed = has_outcome and has_start and has_end
        missing = []
        if not has_outcome:
            missing.append("decision outcome")
        if not (has_start and has_end):
            missing.append("effective period")
        check("10. ownCloud: decision-record.txt content", 2, passed,
              "key content present" if passed else f"missing: {', '.join(missing)}")
    except Exception as e:
        check("10. ownCloud: decision-record.txt content", 2, False, f"exception: {e}")


def check_12_owncloud_tag():
    """Folder tagged with PolicyException2026."""
    try:
        tag_id = oc_db(
            "SELECT id FROM oc_systemtag WHERE name = 'PolicyException2026' LIMIT 1;"
        )
        if not tag_id:
            check("12. ownCloud: Tagged with PolicyException2026", 1, False, "tag does not exist")
            return

        mapping = oc_db(
            f"SELECT objectid FROM oc_systemtag_object_mapping "
            f"WHERE systemtagid = {tag_id} LIMIT 5;"
        )
        has_mapping = bool(mapping)
        check("12. ownCloud: Tagged with PolicyException2026", 1, has_mapping,
              f"tag applied to object(s)" if has_mapping else "tag exists but not applied")
    except Exception as e:
        check("12. ownCloud: Tagged with PolicyException2026", 1, False, f"exception: {e}")


# ── Mattermost checks ────────────────────────────────────────────────────────
def check_13_mm_decision_post():
    """Decision message posted in 'incidents' channel."""
    try:
        row = mm_db(
            "SELECT p.id FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'incidents' "
            "AND p.message LIKE '%APPROVED WITH CONDITIONS%' "
            "AND p.message LIKE '%Rahul Johnson%' "
            "AND p.deleteat = 0 "
            "LIMIT 1;"
        )
        found = bool(row)
        check("13. Mattermost: Decision posted in incidents", 2, found,
              "post found" if found else "decision post not found")
    except Exception as e:
        check("13. Mattermost: Decision posted in incidents", 2, False, f"exception: {e}")


def check_14_mm_thread_reply():
    """Thread reply with conditions detail in incidents channel."""
    try:
        row = mm_db(
            "SELECT p.id FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'incidents' "
            "AND p.rootid != '' "
            "AND p.message LIKE '%6-month duration limit%' "
            "AND p.message LIKE '%Tax equalization%' "
            "AND p.deleteat = 0 "
            "LIMIT 1;"
        )
        found = bool(row)
        check("14. Mattermost: Thread reply with conditions", 2, found,
              "reply found" if found else "thread reply not found")
    except Exception as e:
        check("14. Mattermost: Thread reply with conditions", 2, False, f"exception: {e}")


def check_15_mm_dm_mercy():
    """DM sent to user 'mercy' about policy exception decision."""
    try:
        mercy_id = mm_db("SELECT id FROM users WHERE username = 'mercy' LIMIT 1;")
        if not mercy_id:
            check("15. Mattermost: DM sent to mercy", 2, False, "user 'mercy' not found")
            return

        # Look for a DM post mentioning Rahul Johnson and policy exception
        row = mm_db(
            f"SELECT p.id FROM posts p "
            f"JOIN channels c ON p.channelid = c.id "
            f"JOIN channelmembers cm ON cm.channelid = c.id AND cm.userid = '{mercy_id}' "
            f"WHERE c.type = 'D' "
            f"AND p.message LIKE '%Rahul Johnson%' "
            f"AND p.message LIKE '%policy exception%' "
            f"AND p.deleteat = 0 "
            f"LIMIT 1;"
        )
        found = bool(row)
        check("15. Mattermost: DM sent to mercy", 2, found,
              "DM found" if found else "no matching DM found")
    except Exception as e:
        check("15. Mattermost: DM sent to mercy", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_exceptions_folder()
    check_4_formal_decision_email()
    check_5_formal_decision_cc()
    check_6_onlyoffice_document()
    check_7_onlyoffice_sharing()
    check_8_owncloud_folder_structure()
    check_9_request_summary()
    check_10_decision_record()
    check_12_owncloud_tag()
    check_13_mm_decision_post()
    check_14_mm_thread_reply()
    check_15_mm_dm_mercy()

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
