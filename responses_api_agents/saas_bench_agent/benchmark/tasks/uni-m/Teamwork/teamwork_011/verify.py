# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_011/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-011-I2: Prepare and Distribute Annual Strategy Board Meeting Package

Checks: 14 weighted checks across onlyoffice, owncloud, mattermost, roundcubemail.
Strategy: API (OnlyOffice), DB + WebDAV (ownCloud), DB (Mattermost), DB + maildir (Roundcube)

Required env vars:
  SERVER_HOSTNAME,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER,
  OWNCLOUD_PORT, OWNCLOUD_CONTAINER, OWNCLOUD_DB_CONTAINER,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER
"""

import os
import sys
import subprocess
import json

try:
    import requests
except ImportError:
    print("FATAL: requests library not available", file=sys.stderr)
    sys.exit(1)

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")


def _require(var: str) -> str:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    return val


ONLYOFFICE_PORT = _require("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = _require("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = _require("ONLYOFFICE_DB_CONTAINER")

OWNCLOUD_PORT = _require("OWNCLOUD_PORT")
OWNCLOUD_CONTAINER = _require("OWNCLOUD_CONTAINER")
OWNCLOUD_DB_CONTAINER = _require("OWNCLOUD_DB_CONTAINER")

MATTERMOST_PORT = _require("MATTERMOST_PORT")
MATTERMOST_CONTAINER = _require("MATTERMOST_CONTAINER")
MATTERMOST_DB_CONTAINER = _require("MATTERMOST_DB_CONTAINER")

ROUNDCUBEMAIL_PORT = _require("ROUNDCUBEMAIL_PORT")
ROUNDCUBEMAIL_CONTAINER = _require("ROUNDCUBEMAIL_CONTAINER")
ROUNDCUBEMAIL_DB_CONTAINER = _require("ROUNDCUBEMAIL_DB_CONTAINER")


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


def owncloud_db(sql: str) -> str:
    rc, out, err = docker_exec(
        OWNCLOUD_DB_CONTAINER,
        "mysql", "-u", "owncloud", "-powncloud", "-D", "owncloud",
        "--default-character-set=utf8mb4", "-N", "-B", "-e", sql,
        timeout=15,
    )
    return out.strip()


def mattermost_db(sql: str) -> str:
    rc, out, err = docker_exec(
        MATTERMOST_DB_CONTAINER,
        "psql", "-U", "mmuser", "-d", "mattermost", "-t", "-A", "-c", sql,
        timeout=15,
    )
    return out.strip()


def roundcube_db(sql: str) -> str:
    rc, out, err = docker_exec(
        ROUNDCUBEMAIL_DB_CONTAINER,
        "mysql", "-u", "roundcube", "-proundcube123", "-D", "roundcubemail",
        "--default-character-set=utf8mb4", "-N", "-B", "-e", sql,
        timeout=15,
    )
    return out.strip()


# ── Shared state for OnlyOffice checks ────────────────────────────────────────
_oo_token: str = ""
_oo_base: str = ""
_oo_pres_id: int | None = None


def _oo_auth() -> bool:
    """Authenticate to OnlyOffice and cache token. Returns True on success."""
    global _oo_token, _oo_base
    if _oo_token:
        return True
    _oo_base = f"http://{HOST}:{ONLYOFFICE_PORT}"
    try:
        resp = requests.post(
            f"{_oo_base}/api/2.0/authentication",
            json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
            timeout=15,
        )
        _oo_token = resp.json().get("response", {}).get("token", "")
        return bool(_oo_token)
    except Exception:
        return False


# ── OnlyOffice checks ────────────────────────────────────────────────────────
def check_1_presentation_exists() -> None:
    """Presentation 'Annual_Strategy_Board_Session_2026' exists in Common Documents."""
    global _oo_pres_id
    try:
        if not _oo_auth():
            check("1. OnlyOffice presentation exists", 1, False, "auth failed")
            return
        headers = {"Authorization": _oo_token}
        resp = requests.get(f"{_oo_base}/api/2.0/files/@common", headers=headers, timeout=15)
        data = resp.json().get("response", {})
        files = data.get("files", [])

        pres = [f for f in files if "Annual_Strategy_Board_Session_2026" in f.get("title", "")]
        if pres:
            _oo_pres_id = pres[0]["id"]
        found = bool(pres)
        check("1. OnlyOffice presentation exists", 1, found,
              f"found={found}, titles={[f.get('title') for f in files[:5]]}")
    except Exception as e:
        check("1. OnlyOffice presentation exists", 1, False, f"exception: {e}")


def check_2_shared_laura_edit() -> None:
    """Presentation shared with laura.brown for editing."""
    try:
        if not _oo_token or _oo_pres_id is None:
            check("2. Shared with laura.brown (edit)", 2, False, "no token or presentation not found")
            return
        headers = {"Authorization": _oo_token}
        resp = requests.get(
            f"{_oo_base}/api/2.0/files/file/{_oo_pres_id}/share",
            headers=headers, timeout=15,
        )
        shares = resp.json().get("response", [])

        # access: 1 = ReadWrite (edit), 2 = Read
        laura_edit = any(
            s.get("sharedTo", {}).get("userName", "") == "laura.brown"
            and s.get("access", -1) == 1
            for s in shares
        )
        check("2. Shared with laura.brown (edit)", 2, laura_edit,
              f"shares={[(s.get('sharedTo', {}).get('userName', '?'), s.get('access')) for s in shares]}")
    except Exception as e:
        check("2. Shared with laura.brown (edit)", 2, False, f"exception: {e}")


def check_3_shared_jun_view() -> None:
    """Presentation shared with jun.chen for viewing."""
    try:
        if not _oo_token or _oo_pres_id is None:
            check("3. Shared with jun.chen (view)", 2, False, "no token or presentation not found")
            return
        headers = {"Authorization": _oo_token}
        resp = requests.get(
            f"{_oo_base}/api/2.0/files/file/{_oo_pres_id}/share",
            headers=headers, timeout=15,
        )
        shares = resp.json().get("response", [])

        # access: 2 = Read (view)
        jun_view = any(
            s.get("sharedTo", {}).get("userName", "") == "jun.chen"
            and s.get("access", -1) == 2
            for s in shares
        )
        check("3. Shared with jun.chen (view)", 2, jun_view,
              f"shares={[(s.get('sharedTo', {}).get('userName', '?'), s.get('access')) for s in shares]}")
    except Exception as e:
        check("3. Shared with jun.chen (view)", 2, False, f"exception: {e}")


# ── ownCloud checks ──────────────────────────────────────────────────────────
def check_4_owncloud_folder_structure() -> None:
    """Folder 'Annual_Strategy_Session_2026' with subfolders exists."""
    try:
        main_id = owncloud_db(
            "SELECT fc.fileid FROM oc_filecache fc "
            "JOIN oc_storages s ON fc.storage = s.numeric_id "
            "WHERE s.id = 'home::admin' "
            "AND fc.path = 'files/Annual_Strategy_Session_2026';"
        )
        sub_fin = owncloud_db(
            "SELECT fc.fileid FROM oc_filecache fc "
            "JOIN oc_storages s ON fc.storage = s.numeric_id "
            "WHERE s.id = 'home::admin' "
            "AND fc.path = 'files/Annual_Strategy_Session_2026/Financial_Close';"
        )
        sub_gov = owncloud_db(
            "SELECT fc.fileid FROM oc_filecache fc "
            "JOIN oc_storages s ON fc.storage = s.numeric_id "
            "WHERE s.id = 'home::admin' "
            "AND fc.path = 'files/Annual_Strategy_Session_2026/Governance_Materials';"
        )
        passed = bool(main_id) and bool(sub_fin) and bool(sub_gov)
        check("4. ownCloud folder structure", 1, passed,
              f"main={bool(main_id)}, Financial_Close={bool(sub_fin)}, Governance_Materials={bool(sub_gov)}")
    except Exception as e:
        check("4. ownCloud folder structure", 1, False, f"exception: {e}")


def check_5_pdf_in_folder() -> None:
    """PDF file exists in Annual_Strategy_Session_2026."""
    try:
        pdf = owncloud_db(
            "SELECT fc.name FROM oc_filecache fc "
            "JOIN oc_storages s ON fc.storage = s.numeric_id "
            "WHERE s.id = 'home::admin' "
            "AND fc.path LIKE 'files/Annual\\_Strategy\\_Session\\_2026/%.pdf' "
            "AND fc.path NOT LIKE 'files/Annual\\_Strategy\\_Session\\_2026/%/%.pdf';"
        )
        passed = bool(pdf)
        check("5. PDF in Annual_Strategy_Session_2026", 1, passed, f"name={pdf or 'none'}")
    except Exception as e:
        check("5. PDF in Annual_Strategy_Session_2026", 1, False, f"exception: {e}")


def check_6_resolution_log_content() -> None:
    """resolution_log.txt in Governance_Materials with correct content."""
    try:
        # Check existence in DB
        exists = owncloud_db(
            "SELECT fc.fileid FROM oc_filecache fc "
            "JOIN oc_storages s ON fc.storage = s.numeric_id "
            "WHERE s.id = 'home::admin' "
            "AND fc.path = 'files/Annual_Strategy_Session_2026/Governance_Materials/resolution_log.txt';"
        )
        if not exists:
            check("6. resolution_log.txt content", 2, False, "file not found in DB")
            return

        # Read via WebDAV
        resp = requests.get(
            f"http://{HOST}:{OWNCLOUD_PORT}/remote.php/dav/files/admin/"
            "Annual_Strategy_Session_2026/Governance_Materials/resolution_log.txt",
            auth=("admin", "admin"),
            timeout=15,
        )
        if resp.status_code != 200:
            check("6. resolution_log.txt content", 2, False, f"WebDAV GET status={resp.status_code}")
            return

        expected = (
            "Annual Strategy Session 2026 Resolution Log\n"
            "Resolution A: Adoption of FY2027 Operating Plan - PENDING\n"
            "Resolution B: Renewal of Audit Committee Charter - PENDING\n"
            "Resolution C: Ratification of Director Compensation - PENDING\n"
            "Quorum Verification: TBD\n"
            "Minutes Certified By: Office of the Board Secretary"
        )
        content = resp.text.rstrip("\n")
        passed = content == expected
        detail = "exact match" if passed else f"len={len(content)} vs expected={len(expected)}"
        check("6. resolution_log.txt content", 2, passed, detail)
    except Exception as e:
        check("6. resolution_log.txt content", 2, False, f"exception: {e}")


def check_7_tag_restricted() -> None:
    """Folder tagged 'Restricted'."""
    try:
        folder_id = owncloud_db(
            "SELECT fc.fileid FROM oc_filecache fc "
            "JOIN oc_storages s ON fc.storage = s.numeric_id "
            "WHERE s.id = 'home::admin' "
            "AND fc.path = 'files/Annual_Strategy_Session_2026';"
        )
        if not folder_id:
            check("7. Tag 'Restricted' on folder", 1, False, "folder not found")
            return

        tag = owncloud_db(
            "SELECT st.name FROM oc_systemtag st "
            "JOIN oc_systemtag_object_mapping stom ON st.id = stom.systemtagid "
            f"WHERE stom.objectid = '{folder_id}' AND stom.objecttype = 'files';"
        )
        passed = "Restricted" in tag
        check("7. Tag 'Restricted' on folder", 1, passed, f"tags={tag or 'none'}")
    except Exception as e:
        check("7. Tag 'Restricted' on folder", 1, False, f"exception: {e}")


def check_8_shared_admin_readonly() -> None:
    """Folder shared with group 'admin' read-only."""
    try:
        folder_id = owncloud_db(
            "SELECT fc.fileid FROM oc_filecache fc "
            "JOIN oc_storages s ON fc.storage = s.numeric_id "
            "WHERE s.id = 'home::admin' "
            "AND fc.path = 'files/Annual_Strategy_Session_2026';"
        )
        if not folder_id:
            check("8. Shared with group admin (read-only)", 1, False, "folder not found")
            return

        # share_type=1 is group share; permissions=1 is read-only
        share = owncloud_db(
            "SELECT share_type, share_with, permissions FROM oc_share "
            f"WHERE file_source = {folder_id} AND share_type = 1 AND share_with = 'admin';"
        )
        if not share:
            check("8. Shared with group admin (read-only)", 1, False, "no group share found")
            return

        parts = share.split("\t")
        permissions = int(parts[2]) if len(parts) >= 3 else -1
        passed = permissions == 1
        check("8. Shared with group admin (read-only)", 1, passed,
              f"permissions={permissions} (expected 1=read-only)")
    except Exception as e:
        check("8. Shared with group admin (read-only)", 1, False, f"exception: {e}")


# ── Mattermost checks ────────────────────────────────────────────────────────
def check_9_pinned_message_roadmap() -> None:
    """Pinned message in Roadmap channel containing expected announcement text."""
    try:
        channel_id = mattermost_db(
            "SELECT c.id FROM channels c "
            "JOIN teams t ON c.teamid = t.id "
            "WHERE c.displayname = 'Roadmap' AND t.displayname = 'Product & Design';"
        )
        if not channel_id:
            check("9. Pinned message in Roadmap", 2, False, "channel not found")
            return

        pinned = mattermost_db(
            "SELECT message FROM posts "
            f"WHERE channelid = '{channel_id}' "
            "AND ispinned = true AND deleteat = 0 "
            "AND message LIKE '%Annual Strategy Session 2026 materials have been finalized%';"
        )
        passed = bool(pinned) and "Annual Strategy Session 2026" in pinned
        check("9. Pinned message in Roadmap", 2, passed,
              f"found={'yes' if pinned else 'no'}")
    except Exception as e:
        check("9. Pinned message in Roadmap", 2, False, f"exception: {e}")


def check_10_dm_katheleen() -> None:
    """DM to katheleen with expected legal review request text."""
    try:
        dm = mattermost_db(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.type = 'D' AND p.deleteat = 0 "
            "AND p.message LIKE '%Hello Katheleen%kindly perform a legal review%';"
        )
        passed = bool(dm) and "Annual Strategy Session 2026" in dm
        check("10. DM to katheleen", 2, passed,
              f"found={'yes' if dm else 'no'}")
    except Exception as e:
        check("10. DM to katheleen", 2, False, f"exception: {e}")


# ── Roundcube checks ─────────────────────────────────────────────────────────
def check_11_roundcube_identity() -> None:
    """Identity 'Marcus Torres - Corporate Secretariat' with correct email/org/signature."""
    try:
        result = roundcube_db(
            "SELECT name, email, organization, signature FROM identities "
            "WHERE name = 'Marcus Torres - Corporate Secretariat' AND del = 0;"
        )
        if not result:
            check("11. Roundcube identity", 2, False, "identity not found")
            return

        parts = result.split("\t")
        email = parts[1] if len(parts) > 1 else ""
        org = parts[2] if len(parts) > 2 else ""
        sig = parts[3] if len(parts) > 3 else ""

        email_ok = email == "marcus.torres@mail.local"
        org_ok = org == "Office of the Corporate Secretariat"
        sig_ok = "Marcus Torres" in sig and "Acting Board Secretary" in sig

        passed = email_ok and org_ok and sig_ok
        check("11. Roundcube identity", 2, passed,
              f"email_ok={email_ok}, org_ok={org_ok}, sig_ok={sig_ok}")
    except Exception as e:
        check("11. Roundcube identity", 2, False, f"exception: {e}")


def check_12_email_sent_subject_recipients() -> None:
    """Email sent with correct subject and recipients (To + CC)."""
    try:
        rc, out, err = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "bash", "-c",
            "grep -rl 'Subject: Annual Strategy Session 2026 - Governance Binder and Resolutions' /var/mail/ 2>/dev/null | head -5",
            timeout=15,
        )
        if rc != 0 or not out.strip():
            check("12. Email sent with correct subject/recipients", 2, False,
                  "email not found in maildir")
            return

        email_file = out.strip().split("\n")[0]
        rc2, content, _ = docker_exec(
            ROUNDCUBEMAIL_CONTAINER, "cat", email_file, timeout=15,
        )

        has_sarah = "sarah.obrien@mail.local" in content
        has_ben = "ben.kowalski@mail.local" in content
        has_emma = "emma.larsson@mail.local" in content
        has_cc = "rachel.goldberg@mail.local" in content

        passed = has_sarah and has_ben and has_emma and has_cc
        check("12. Email sent with correct subject/recipients", 2, passed,
              f"sarah={has_sarah}, ben={has_ben}, emma={has_emma}, cc_rachel={has_cc}")
    except Exception as e:
        check("12. Email sent with correct subject/recipients", 2, False, f"exception: {e}")


def check_13_email_priority_high() -> None:
    """Sent email has high priority header."""
    try:
        rc, out, err = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "bash", "-c",
            "grep -rl 'Subject: Annual Strategy Session 2026 - Governance Binder and Resolutions' /var/mail/ 2>/dev/null | head -1",
            timeout=15,
        )
        if rc != 0 or not out.strip():
            check("13. Email priority high", 1, False, "email not found")
            return

        email_file = out.strip()
        rc2, content, _ = docker_exec(
            ROUNDCUBEMAIL_CONTAINER, "cat", email_file, timeout=15,
        )

        # High priority: X-Priority: 1 or Importance: High
        has_xpriority = "X-Priority: 1" in content
        has_importance = "importance: high" in content.lower()
        passed = has_xpriority or has_importance
        check("13. Email priority high", 1, passed,
              f"X-Priority-1={has_xpriority}, Importance-high={has_importance}")
    except Exception as e:
        check("13. Email priority high", 1, False, f"exception: {e}")


def check_14_email_flagged_sent() -> None:
    """Sent email is flagged in the Sent folder."""
    try:
        # In Dovecot maildir, flagged emails have 'F' in the flags portion (:2,*F*)
        # Search Sent folder for our subject among flagged files
        rc, out, err = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "bash", "-c",
            r"find /var/mail/ -path '*ent*' -name '*:2,*F*' "
            r"-exec grep -l 'Subject: Annual Strategy Session 2026 - Governance Binder' {} \; 2>/dev/null",
            timeout=15,
        )
        if out.strip():
            check("14. Sent email flagged", 1, True, "flagged email found in Sent")
            return

        # Fallback: check if the email exists in Sent at all (for diagnostics)
        rc2, out2, _ = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "bash", "-c",
            "find /var/mail/ -path '*ent*' "
            "-exec grep -l 'Subject: Annual Strategy Session 2026 - Governance Binder' {} \\; 2>/dev/null",
            timeout=15,
        )
        if out2.strip():
            # Email exists but not flagged
            # Check if the filename has flags
            files = out2.strip().split("\n")
            check("14. Sent email flagged", 1, False,
                  f"email in Sent but not flagged; files={files[:3]}")
        else:
            check("14. Sent email flagged", 1, False, "email not found in Sent folder")
    except Exception as e:
        check("14. Sent email flagged", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_presentation_exists()
    check_2_shared_laura_edit()
    check_3_shared_jun_view()
    check_4_owncloud_folder_structure()
    check_5_pdf_in_folder()
    check_6_resolution_log_content()
    check_7_tag_restricted()
    check_8_shared_admin_readonly()
    check_9_pinned_message_roadmap()
    check_10_dm_katheleen()
    check_11_roundcube_identity()
    check_12_email_sent_subject_recipients()
    check_13_email_priority_high()
    check_14_email_flagged_sent()

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
