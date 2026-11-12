# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_052/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-052-I3: Investigate and Resolve Data Export Client Complaint from Helix Analytics

Checks: 10 weighted checks across roundcubemail, mattermost, owncloud, onlyoffice.
Strategy: docker exec (DB queries) + docker exec (maildir) + API (ownCloud WebDAV, OnlyOffice)

Required env vars:
  SERVER_HOSTNAME,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  OWNCLOUD_PORT, OWNCLOUD_CONTAINER, OWNCLOUD_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess
import json
import requests
import re

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

def _require(var: str) -> str:
    val = os.getenv(var, "")
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    return val

ROUNDCUBEMAIL_PORT = _require("ROUNDCUBEMAIL_PORT")
ROUNDCUBEMAIL_CONTAINER = _require("ROUNDCUBEMAIL_CONTAINER")
ROUNDCUBEMAIL_DB_CONTAINER = _require("ROUNDCUBEMAIL_DB_CONTAINER")

MATTERMOST_PORT = _require("MATTERMOST_PORT")
MATTERMOST_CONTAINER = _require("MATTERMOST_CONTAINER")
MATTERMOST_DB_CONTAINER = _require("MATTERMOST_DB_CONTAINER")

OWNCLOUD_PORT = _require("OWNCLOUD_PORT")
OWNCLOUD_CONTAINER = _require("OWNCLOUD_CONTAINER")
OWNCLOUD_DB_CONTAINER = _require("OWNCLOUD_DB_CONTAINER")

ONLYOFFICE_PORT = _require("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = _require("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = _require("ONLYOFFICE_DB_CONTAINER")


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


def mysql_query(container: str, db: str, user: str, password: str, sql: str) -> str:
    """Run a MySQL/MariaDB query and return stdout."""
    rc, out, err = docker_exec(
        container,
        "mysql", f"-u{user}", f"-p{password}", db,
        "--default-character-set=utf8mb4",
        "-N", "-B", "-e", sql,
        timeout=15,
    )
    return out.strip()


def psql_query(container: str, db: str, user: str, password: str, sql: str) -> str:
    """Run a PostgreSQL query and return stdout."""
    rc, out, err = docker_exec(
        container,
        "psql", "-U", user, "-d", db, "-t", "-A", "-c", sql,
        timeout=15,
    )
    return out.strip()


def mattermost_api(endpoint: str, method: str = "GET", data: dict = None) -> requests.Response:
    """Call Mattermost REST API with admin auth."""
    base = f"http://{HOST}:{MATTERMOST_PORT}/api/v4"
    # Login
    r = requests.post(f"{base}/users/login",
                      json={"login_id": "admin", "password": "SeedAdmin1pass"}, timeout=10)
    r.raise_for_status()
    token = r.headers.get("Token", "")
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{base}{endpoint}"
    if method == "GET":
        return requests.get(url, headers=headers, timeout=10)
    elif method == "POST":
        return requests.post(url, headers=headers, json=data, timeout=10)
    return requests.get(url, headers=headers, timeout=10)


# ── Roundcube checks ──────────────────────────────────────────────────────────

def check_1_mail_folder_exists() -> None:
    """Mail folder 'Priority Client Issues' exists under INBOX for james.whitfield."""
    try:
        # Check IMAP subscriptions / maildir for the folder
        rc, out, err = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "find", "/var/mail/", "-type", "d", "-name", ".INBOX.Priority Client Issues",
            timeout=15,
        )
        # Also try without INBOX prefix and alternative naming
        rc2, out2, err2 = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "find", "/var/mail/", "-type", "d", "-name", ".Priority Client Issues",
            timeout=15,
        )
        found = bool(out.strip()) or bool(out2.strip())
        if not found:
            # Try broader search
            rc3, out3, err3 = docker_exec(
                ROUNDCUBEMAIL_CONTAINER,
                "bash", "-c", "find /var/mail/ -type d 2>/dev/null | grep -i 'priority'",
                timeout=15,
            )
            found = bool(out3.strip())
            detail = f"found={out3.strip()}" if found else "folder not found in maildir"
        else:
            detail = "folder exists"
        check("1. Mail folder 'Priority Client Issues' exists", 1, found, detail)
    except Exception as e:
        check("1. Mail folder 'Priority Client Issues' exists", 1, False, f"exception: {e}")


# ── Mattermost checks ────────────────────────────────────────────────────────

def check_5_mm_private_channel() -> None:
    """Private channel 'helix-analytics-export-investigation' exists with correct purpose."""
    try:
        # Query DB for the channel
        sql = (
            "SELECT c.type, c.purpose FROM channels c "
            "JOIN teams t ON c.teamid = t.id "
            "WHERE c.name = 'helix-analytics-export-investigation' "
            "AND t.displayname = 'Engineering Hub' LIMIT 1;"
        )
        out = psql_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", "mmuser_password", sql)
        if not out:
            check("5. MM private channel exists with purpose", 2, False, "channel not found")
            return
        parts = out.split("|")
        ch_type = parts[0].strip() if len(parts) > 0 else ""
        purpose = parts[1].strip() if len(parts) > 1 else ""

        is_private = ch_type == "P"
        expected_purpose = "Private coordination channel for investigating the Helix Analytics Inc. data export pipeline failure and tracking resolution"
        purpose_ok = expected_purpose.lower() in purpose.lower() or purpose.lower() in expected_purpose.lower()

        passed = is_private and purpose_ok
        issues = []
        if not is_private:
            issues.append(f"type={ch_type}, expected P")
        if not purpose_ok:
            issues.append(f"purpose mismatch: got '{purpose[:60]}...'")
        check("5. MM private channel exists with purpose", 2, passed,
              "channel OK" if passed else "; ".join(issues))
    except Exception as e:
        check("5. MM private channel exists with purpose", 2, False, f"exception: {e}")


def check_6_mm_complaint_brief() -> None:
    """Complaint brief message posted in investigation channel."""
    try:
        sql = (
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'helix-analytics-export-investigation' "
            "AND p.message LIKE '%Complaint Brief%' "
            "AND p.deleteat = 0 LIMIT 1;"
        )
        out = psql_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", "mmuser_password", sql)
        has_brief = "Complaint Brief" in out
        has_helix = "Helix Analytics" in out
        has_critical = "Critical" in out
        passed = has_brief and has_helix and has_critical
        detail = "brief posted" if passed else f"missing content: brief={has_brief}, helix={has_helix}, critical={has_critical}"
        check("6. Complaint brief posted in channel", 2, passed, detail)
    except Exception as e:
        check("6. Complaint brief posted in channel", 2, False, f"exception: {e}")


def check_7_mm_thread_replies() -> None:
    """Thread replies tagging delana and stephany exist."""
    try:
        # Find the complaint brief post and check for thread replies
        sql = (
            "SELECT p.id FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'helix-analytics-export-investigation' "
            "AND p.message LIKE '%Complaint Brief%' "
            "AND p.deleteat = 0 LIMIT 1;"
        )
        root_id = psql_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", "mmuser_password", sql).strip()

        if not root_id:
            check("7. Thread replies tagging delana & stephany", 2, False, "root post not found")
            return

        # Check for replies in thread
        sql_replies = (
            f"SELECT message FROM posts "
            f"WHERE rootid = '{root_id}' AND deleteat = 0 "
            f"ORDER BY createat;"
        )
        out = psql_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", "mmuser_password", sql_replies)
        has_delana = "@delana" in out.lower() or "delana" in out.lower()
        has_stephany = "@stephany" in out.lower() or "stephany" in out.lower()
        passed = has_delana and has_stephany
        detail = f"delana={has_delana}, stephany={has_stephany}"
        check("7. Thread replies tagging delana & stephany", 2, passed, detail)
    except Exception as e:
        check("7. Thread replies tagging delana & stephany", 2, False, f"exception: {e}")


# ── ownCloud checks ──────────────────────────────────────────────────────────

def check_8_oc_folder_structure() -> None:
    """Folder structure exists: main folder + Technical-Evidence + Client-Communications."""
    try:
        base_url = f"http://{HOST}:{OWNCLOUD_PORT}"
        auth = ("admin", "admin")

        # Check main folder
        r = requests.request("PROPFIND", f"{base_url}/remote.php/dav/files/admin/Helix-Analytics-Export-Investigation-2026-04/",
                             auth=auth, headers={"Depth": "1"}, timeout=10)
        folder_exists = r.status_code in (207, 200)

        has_tech = "Technical-Evidence" in r.text if folder_exists else False
        has_client = "Client-Communications" in r.text if folder_exists else False

        passed = folder_exists and has_tech and has_client
        detail = f"folder={folder_exists}, tech_evidence={has_tech}, client_comms={has_client}"
        check("8. ownCloud folder structure", 1, passed, detail)
    except Exception as e:
        check("8. ownCloud folder structure", 1, False, f"exception: {e}")


def check_10_oc_investigation_log() -> None:
    """export-investigation-log.txt exists with correct content."""
    try:
        base_url = f"http://{HOST}:{OWNCLOUD_PORT}"
        auth = ("admin", "admin")
        r = requests.get(
            f"{base_url}/remote.php/dav/files/admin/Helix-Analytics-Export-Investigation-2026-04/export-investigation-log.txt",
            auth=auth, timeout=10)
        if r.status_code != 200:
            check("10. Investigation log content", 2, False, f"HTTP {r.status_code}")
            return
        content = r.text
        has_client = "Helix Analytics Inc." in content
        has_contact = "carlos.mendez@mail.local" in content
        has_severity = "Critical" in content
        has_assessment = "executive dashboard" in content.lower()

        passed = has_client and has_contact and has_severity and has_assessment
        issues = []
        if not has_client:
            issues.append("missing client name")
        if not has_contact:
            issues.append("missing contact email")
        if not has_severity:
            issues.append("missing severity")
        if not has_assessment:
            issues.append("missing assessment")
        check("10. Investigation log content", 2, passed,
              "content OK" if passed else "; ".join(issues))
    except Exception as e:
        check("10. Investigation log content", 2, False, f"exception: {e}")


def check_12_oc_public_share_upload_only() -> None:
    """Technical-Evidence has public share with upload-only (file drop) permissions."""
    try:
        base_url = f"http://{HOST}:{OWNCLOUD_PORT}"
        auth = ("admin", "admin")
        r = requests.get(
            f"{base_url}/ocs/v2.php/apps/files_sharing/api/v1/shares",
            auth=auth,
            params={"format": "json",
                    "path": "/Helix-Analytics-Export-Investigation-2026-04/Technical-Evidence",
                    "reshares": "true"},
            timeout=10)
        data = r.json()
        shares = data.get("ocs", {}).get("data", [])
        # Public link share type = 3; upload-only (file drop) = permissions 4
        public_upload = False
        for s in shares:
            share_type = s.get("share_type", -1)
            perms = s.get("permissions", 0)
            if share_type == 3 and perms == 4:
                public_upload = True
                break
        check("12. Technical-Evidence public share (upload-only)", 2, public_upload,
              "public file-drop OK" if public_upload else f"shares: {[(s.get('share_type'), s.get('permissions')) for s in shares]}")
    except Exception as e:
        check("12. Technical-Evidence public share (upload-only)", 2, False, f"exception: {e}")


# ── OnlyOffice checks ────────────────────────────────────────────────────────

def check_13_oo_document_exists() -> None:
    """Document 'Helix Analytics Export Failure - Resolution Report' exists in Common Documents."""
    try:
        base_url = f"http://{HOST}:{ONLYOFFICE_PORT}"
        # Authenticate
        session = requests.Session()
        auth_r = session.post(f"{base_url}/api/2.0/authentication",
                              json={"userName": "admin@onlyoffice.local",
                                    "password": "NewAdmin123!"},
                              timeout=15)
        auth_data = auth_r.json()
        token = auth_data.get("response", {}).get("token", "")
        headers = {"Authorization": token}

        # Get Common Documents folder (id=2 typically)
        r = session.get(f"{base_url}/api/2.0/files/@common", headers=headers, timeout=15)
        files_data = r.json()
        files = files_data.get("response", {}).get("files", [])
        found = any("Helix Analytics Export Failure" in f.get("title", "") and
                     "Resolution Report" in f.get("title", "")
                     for f in files)
        check("13. OnlyOffice document exists in Common Documents", 1, found,
              "document found" if found else f"searched {len(files)} files, not found")
    except Exception as e:
        check("13. OnlyOffice document exists in Common Documents", 1, False, f"exception: {e}")


def check_14_oo_document_shared() -> None:
    """Document shared with laura.brown (view) and jun.chen (edit)."""
    try:
        base_url = f"http://{HOST}:{ONLYOFFICE_PORT}"
        session = requests.Session()
        auth_r = session.post(f"{base_url}/api/2.0/authentication",
                              json={"userName": "admin@onlyoffice.local",
                                    "password": "NewAdmin123!"},
                              timeout=15)
        auth_data = auth_r.json()
        token = auth_data.get("response", {}).get("token", "")
        headers = {"Authorization": token}

        # Find the document
        r = session.get(f"{base_url}/api/2.0/files/@common", headers=headers, timeout=15)
        files_data = r.json()
        files = files_data.get("response", {}).get("files", [])
        doc_id = None
        for f in files:
            if "Helix Analytics Export Failure" in f.get("title", ""):
                doc_id = f.get("id")
                break

        if doc_id is None:
            check("14. Document shared with laura.brown & jun.chen", 2, False, "document not found")
            return

        # Get sharing info
        share_r = session.get(f"{base_url}/api/2.0/files/file/{doc_id}/share",
                              headers=headers, timeout=15)
        share_data = share_r.json()
        shares = share_data.get("response", [])

        laura_view = False
        jun_edit = False
        for s in shares:
            shared_to = s.get("sharedTo", {})
            user_name = shared_to.get("userName", "") or shared_to.get("email", "")
            display_name = shared_to.get("displayName", "")
            access = s.get("access", -1)
            # access: 1=full, 2=read, 3=deny  (OnlyOffice DocSpace may differ)
            if "laura.brown" in user_name or "laura.brown" in display_name or "Laura Brown" in display_name:
                laura_view = access == 2  # read-only
            if "jun.chen" in user_name or "jun.chen" in display_name or "Jun Chen" in display_name:
                jun_edit = access == 1  # full access / edit

        passed = laura_view and jun_edit
        detail = f"laura_view={laura_view}, jun_edit={jun_edit}"
        check("14. Document shared with laura.brown & jun.chen", 2, passed, detail)
    except Exception as e:
        check("14. Document shared with laura.brown & jun.chen", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_mail_folder_exists()
    check_5_mm_private_channel()
    check_6_mm_complaint_brief()
    check_7_mm_thread_replies()
    check_8_oc_folder_structure()
    check_10_oc_investigation_log()
    check_12_oc_public_share_upload_only()
    check_13_oo_document_exists()
    check_14_oo_document_shared()

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
