# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_030/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-030-I2: Cross-Team Status Report across Mattermost, OnlyOffice, ownCloud, Roundcube

Checks: 14 weighted checks across 4 sites.
Strategy: DB (Mattermost, Roundcube prefs), API (OnlyOffice, ownCloud), maildir (Roundcube email)

Required env vars:
  SERVER_HOSTNAME,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER,
  OWNCLOUD_PORT, OWNCLOUD_CONTAINER, OWNCLOUD_DB_CONTAINER,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
import urllib.parse

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

REQUIRED_VARS = [
    "MATTERMOST_PORT", "MATTERMOST_CONTAINER", "MATTERMOST_DB_CONTAINER",
    "ONLYOFFICE_PORT", "ONLYOFFICE_CONTAINER", "ONLYOFFICE_DB_CONTAINER",
    "OWNCLOUD_PORT", "OWNCLOUD_CONTAINER", "OWNCLOUD_DB_CONTAINER",
    "ROUNDCUBEMAIL_PORT", "ROUNDCUBEMAIL_CONTAINER", "ROUNDCUBEMAIL_DB_CONTAINER",
]

_env = {}
for var in REQUIRED_VARS:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    _env[var] = val

MM_DB = _env["MATTERMOST_DB_CONTAINER"]
MM_PORT = _env["MATTERMOST_PORT"]
MM_CONTAINER = _env["MATTERMOST_CONTAINER"]
OO_PORT = _env["ONLYOFFICE_PORT"]
OO_CONTAINER = _env["ONLYOFFICE_CONTAINER"]
OO_DB = _env["ONLYOFFICE_DB_CONTAINER"]
OC_PORT = _env["OWNCLOUD_PORT"]
OC_CONTAINER = _env["OWNCLOUD_CONTAINER"]
OC_DB = _env["OWNCLOUD_DB_CONTAINER"]
RC_PORT = _env["ROUNDCUBEMAIL_PORT"]
RC_CONTAINER = _env["ROUNDCUBEMAIL_CONTAINER"]
RC_DB = _env["ROUNDCUBEMAIL_DB_CONTAINER"]

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


def mm_db_query(sql: str) -> str:
    """Query Mattermost Postgres DB."""
    rc, out, err = docker_exec(
        MM_DB, "psql", "-U", "mmuser", "-d", "mattermost", "-t", "-A", "-c", sql
    )
    return out.strip()


def oc_db_query(sql: str) -> str:
    """Query ownCloud MariaDB."""
    rc, out, err = docker_exec(
        OC_DB, "mysql", "-u", "owncloud", "-powncloud", "owncloud",
        "--default-character-set=utf8mb4", "-N", "-B", "-e", sql
    )
    return out.strip()


def rc_db_query(sql: str) -> str:
    """Query Roundcube MariaDB."""
    rc, out, err = docker_exec(
        RC_DB, "mysql", "-u", "roundcube", "-proundcube123", "roundcubemail",
        "--default-character-set=utf8mb4", "-N", "-B", "-e", sql
    )
    return out.strip()


def http_request(url: str, method: str = "GET", data: bytes | None = None,
                 headers: dict | None = None, timeout: int = 15) -> tuple[int, str, dict]:
    """Make an HTTP request. Returns (status_code, body, response_headers)."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body, dict(resp.headers)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, body, dict(e.headers) if e.headers else {}
    except Exception as e:
        return 0, str(e), {}


def oo_api_get_token() -> str | None:
    """Authenticate to OnlyOffice and return auth token."""
    url = f"http://{HOST}:{OO_PORT}/api/2.0/authentication"
    payload = json.dumps({"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"}).encode()
    status, body, _ = http_request(url, "POST", payload, {"Content-Type": "application/json"})
    if status == 200 or status == 201:
        resp = json.loads(body)
        return resp.get("response", {}).get("token")
    return None


def oo_api(endpoint: str, token: str) -> tuple[int, dict]:
    """Make an authenticated OnlyOffice API call."""
    url = f"http://{HOST}:{OO_PORT}/api/2.0/{endpoint}"
    status, body, _ = http_request(url, "GET", headers={"Authorization": token})
    try:
        return status, json.loads(body)
    except Exception:
        return status, {}


# ── Individual checks ─────────────────────────────────────────────────────────

def check_3_oo_document_exists() -> None:
    """Verify OnlyOffice document 'Cross-Team Bi-Weekly Status Report - W26-W27 2026' exists in Common Documents."""
    try:
        token = oo_api_get_token()
        if not token:
            check("3. OO document exists", 2, False, "auth failed")
            return
        # List common documents
        status, data = oo_api("files/@common", token)
        files = data.get("response", {}).get("files", [])
        title = "Cross-Team Bi-Weekly Status Report - W26-W27 2026"
        found = any(f.get("title", "") == title for f in files)
        check("3. OO document exists", 2, found,
              f"not found among {len(files)} files" if not found else "")
    except Exception as e:
        check("3. OO document exists", 2, False, f"exception: {e}")


def _oo_get_doc_id(token: str) -> int | None:
    """Find the document ID for the status report."""
    status, data = oo_api("files/@common", token)
    files = data.get("response", {}).get("files", [])
    title = "Cross-Team Bi-Weekly Status Report - W26-W27 2026"
    for f in files:
        if f.get("title", "") == title:
            return f.get("id")
    return None


def check_4_oo_shared_junchen() -> None:
    """Verify document shared with jun.chen for viewing."""
    try:
        token = oo_api_get_token()
        if not token:
            check("4. OO shared with jun.chen (view)", 2, False, "auth failed")
            return
        doc_id = _oo_get_doc_id(token)
        if not doc_id:
            check("4. OO shared with jun.chen (view)", 2, False, "document not found")
            return
        status, data = oo_api(f"files/file/{doc_id}/share", token)
        shares = data.get("response", [])
        found = False
        for s in shares:
            user = s.get("sharedTo", {})
            uname = user.get("userName", "") or user.get("id", "")
            access = s.get("access", -1)
            # access 2 = read-only in OnlyOffice
            if "jun.chen" in str(uname) and access in (1, 2):
                found = True
                break
        check("4. OO shared with jun.chen (view)", 2, found,
              f"shares={len(shares)}" if not found else "")
    except Exception as e:
        check("4. OO shared with jun.chen (view)", 2, False, f"exception: {e}")


def check_5_oo_shared_amitsingh() -> None:
    """Verify document shared with amit.singh for editing."""
    try:
        token = oo_api_get_token()
        if not token:
            check("5. OO shared with amit.singh (edit)", 2, False, "auth failed")
            return
        doc_id = _oo_get_doc_id(token)
        if not doc_id:
            check("5. OO shared with amit.singh (edit)", 2, False, "document not found")
            return
        status, data = oo_api(f"files/file/{doc_id}/share", token)
        shares = data.get("response", [])
        found = False
        for s in shares:
            user = s.get("sharedTo", {})
            uname = user.get("userName", "") or user.get("id", "")
            access = s.get("access", -1)
            # access 1 = read-write in OnlyOffice
            if "amit.singh" in str(uname) and access == 1:
                found = True
                break
        check("5. OO shared with amit.singh (edit)", 2, found,
              f"shares={len(shares)}" if not found else "")
    except Exception as e:
        check("5. OO shared with amit.singh (edit)", 2, False, f"exception: {e}")


def check_6_oc_folder_structure() -> None:
    """Verify ownCloud folder structure: Leadership_BiWeekly_Reports/2026-P13-P14 exists."""
    try:
        url = f"http://{HOST}:{OC_PORT}/remote.php/dav/files/admin/Leadership_BiWeekly_Reports/2026-P13-P14/"
        import base64
        auth = base64.b64encode(b"admin:admin").decode()
        status, body, _ = http_request(url, "PROPFIND", headers={
            "Authorization": f"Basic {auth}",
            "Depth": "0",
        })
        passed = status in (200, 207)
        check("6. OC folder structure exists", 1, passed,
              f"HTTP {status}" if not passed else "")
    except Exception as e:
        check("6. OC folder structure exists", 1, False, f"exception: {e}")


def check_7_oc_exec_summary_content() -> None:
    """Verify exec_summary.txt exists with correct content."""
    try:
        url = f"http://{HOST}:{OC_PORT}/remote.php/dav/files/admin/Leadership_BiWeekly_Reports/2026-P13-P14/exec_summary.txt"
        import base64
        auth = base64.b64encode(b"admin:admin").decode()
        status, body, _ = http_request(url, "GET", headers={
            "Authorization": f"Basic {auth}",
        })
        if status != 200:
            check("7. OC exec_summary.txt content", 2, False, f"HTTP {status}")
            return
        expected_fragment = "Executive Summary - Bi-Weekly Period June 22"
        passed = expected_fragment in body
        check("7. OC exec_summary.txt content", 2, passed,
              f"content mismatch, got {body[:80]}..." if not passed else "")
    except Exception as e:
        check("7. OC exec_summary.txt content", 2, False, f"exception: {e}")


def check_10_oc_tag() -> None:
    """Verify tag 'biweekly-leadership' applied to Leadership_BiWeekly_Reports."""
    try:
        # Use ownCloud DB to check tags
        sql = (
            "SELECT t.name FROM oc_systemtag t "
            "JOIN oc_systemtag_object_mapping m ON t.id = m.systemtagid "
            "JOIN oc_filecache f ON m.objectid = f.fileid "
            "WHERE f.name = 'Leadership_BiWeekly_Reports' "
            "AND t.name = 'biweekly-leadership'"
        )
        out = oc_db_query(sql)
        passed = "biweekly-leadership" in out
        check("10. OC tag biweekly-leadership applied", 1, passed,
              f"got: {out!r}" if not passed else "")
    except Exception as e:
        check("10. OC tag biweekly-leadership applied", 1, False, f"exception: {e}")


def check_11_rc_email_subject_in_sent() -> None:
    """Verify email with correct subject exists in sent folder."""
    try:
        expected_subject = "Bi-Weekly Cross-Team Status Report - June 22 to July 3, 2026"
        # Search in Dovecot maildir for the sent email
        rc_code, out, err = docker_exec(
            RC_CONTAINER,
            "grep", "-rl", f"Subject: {expected_subject}",
            "/var/mail/", timeout=20,
        )
        # Also check in /var/vmail/ if /var/mail/ doesn't have it
        if not out.strip():
            rc_code, out, err = docker_exec(
                RC_CONTAINER,
                "find", "/", "-path", "*/Sent*", "-name", "*.eml",
                timeout=20,
            )
            if not out.strip():
                rc_code, out, err = docker_exec(
                    RC_CONTAINER,
                    "bash", "-c",
                    f"find /var -type f 2>/dev/null | head -500 | xargs grep -l 'Subject: {expected_subject}' 2>/dev/null || true",
                    timeout=30,
                )
        passed = bool(out.strip())
        check("11. RC email with correct subject in sent", 2, passed,
              f"no matching email found" if not passed else "")
    except Exception as e:
        check("11. RC email with correct subject in sent", 2, False, f"exception: {e}")


def check_12_rc_email_recipients() -> None:
    """Verify email was sent to correct To recipients."""
    try:
        expected_to = ["jun.chen@onlyoffice.local", "amit.singh@onlyoffice.local", "laura.brown@onlyoffice.local"]
        # Find the email file and check To header
        rc_code, out, err = docker_exec(
            RC_CONTAINER,
            "bash", "-c",
            "grep -rl 'Bi-Weekly Cross-Team Status Report' /var/mail/ 2>/dev/null || "
            "grep -rl 'Bi-Weekly Cross-Team Status Report' /var/vmail/ 2>/dev/null || "
            "grep -rl 'Bi-Weekly Cross-Team Status Report' /home/ 2>/dev/null || true",
            timeout=20,
        )
        if not out.strip():
            check("12. RC email recipients correct", 1, False, "email file not found")
            return
        email_file = out.strip().splitlines()[0]
        rc_code, content, err = docker_exec(RC_CONTAINER, "cat", email_file, timeout=10)
        found_all = all(addr in content for addr in expected_to)
        missing = [a for a in expected_to if a not in content]
        check("12. RC email recipients correct", 1, found_all,
              f"missing: {missing}" if missing else "")
    except Exception as e:
        check("12. RC email recipients correct", 1, False, f"exception: {e}")


def check_13_rc_mdn_requested() -> None:
    """Verify read receipt (MDN) was requested on the email."""
    try:
        rc_code, out, err = docker_exec(
            RC_CONTAINER,
            "bash", "-c",
            "grep -rl 'Bi-Weekly Cross-Team Status Report' /var/mail/ 2>/dev/null || "
            "grep -rl 'Bi-Weekly Cross-Team Status Report' /var/vmail/ 2>/dev/null || true",
            timeout=20,
        )
        if not out.strip():
            check("13. RC read receipt (MDN) requested", 1, False, "email file not found")
            return
        email_file = out.strip().splitlines()[0]
        rc_code, content, err = docker_exec(RC_CONTAINER, "cat", email_file, timeout=10)
        # MDN is indicated by Disposition-Notification-To header
        passed = "Disposition-Notification-To" in content
        check("13. RC read receipt (MDN) requested", 1, passed,
              "no Disposition-Notification-To header" if not passed else "")
    except Exception as e:
        check("13. RC read receipt (MDN) requested", 1, False, f"exception: {e}")


def check_14_rc_draft_interval() -> None:
    """Verify auto-save draft interval set to 5 minutes."""
    try:
        sql = (
            "SELECT preferences FROM users "
            "WHERE username = 'james.whitfield@mail.local'"
        )
        out = rc_db_query(sql)
        if not out:
            check("14. RC draft interval set to 5 min", 1, False, "user prefs not found")
            return
        # Roundcube stores prefs as serialized PHP. Check for draft_autosave value.
        # The value for 5 minutes is typically 300 (seconds)
        passed = ("draft_autosave" in out and ("300" in out or "5min" in out or '"5"' in out))
        check("14. RC draft interval set to 5 min", 1, passed,
              f"prefs snippet: ...{out[max(0,out.find('draft_autosave')-10):out.find('draft_autosave')+50]}..." if "draft_autosave" in out else "draft_autosave not in prefs")
    except Exception as e:
        check("14. RC draft interval set to 5 min", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_3_oo_document_exists()
    check_4_oo_shared_junchen()
    check_5_oo_shared_amitsingh()
    check_6_oc_folder_structure()
    check_7_oc_exec_summary_content()
    check_10_oc_tag()
    check_11_rc_email_subject_in_sent()
    check_12_rc_email_recipients()
    check_13_rc_mdn_requested()
    check_14_rc_draft_interval()

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
