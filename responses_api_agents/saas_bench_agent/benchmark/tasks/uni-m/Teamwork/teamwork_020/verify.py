# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_020/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-020-I2: Roll Out Updated Healthcare Telehealth Fee Policy

Checks: 15 weighted checks across owncloud, onlyoffice, mattermost, roundcubemail.
Strategy: docker exec DB queries (ownCloud MariaDB, Mattermost Postgres, Roundcube MariaDB),
          REST API (ownCloud WebDAV, OnlyOffice API), docker exec doveadm (Roundcube email).

Required env vars:
  SERVER_HOSTNAME,
  OWNCLOUD_PORT, OWNCLOUD_CONTAINER, OWNCLOUD_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER.
"""

import os
import sys
import subprocess
import requests


# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"FATAL: {name} not set", file=sys.stderr)
        sys.exit(1)
    return val


OC_PORT = _require("OWNCLOUD_PORT")
OC_CONTAINER = _require("OWNCLOUD_CONTAINER")
OC_DB = _require("OWNCLOUD_DB_CONTAINER")

OO_PORT = _require("ONLYOFFICE_PORT")
OO_CONTAINER = _require("ONLYOFFICE_CONTAINER")
OO_DB = _require("ONLYOFFICE_DB_CONTAINER")

MM_PORT = _require("MATTERMOST_PORT")
MM_CONTAINER = _require("MATTERMOST_CONTAINER")
MM_DB = _require("MATTERMOST_DB_CONTAINER")

RC_PORT = _require("ROUNDCUBEMAIL_PORT")
RC_CONTAINER = _require("ROUNDCUBEMAIL_CONTAINER")
RC_DB = _require("ROUNDCUBEMAIL_DB_CONTAINER")


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


def oc_sql(sql: str) -> str:
    _, out, _ = docker_exec(
        OC_DB, "mysql", "-u", "owncloud", "-powncloud",
        "--default-character-set=utf8mb4", "owncloud", "-N", "-e", sql,
    )
    return out.strip()


def mm_sql(sql: str) -> str:
    _, out, _ = docker_exec(
        MM_DB, "psql", "-U", "mmuser", "-d", "mattermost", "-t", "-A", "-c", sql,
    )
    return out.strip()


def rc_sql(sql: str) -> str:
    _, out, _ = docker_exec(
        RC_DB, "mysql", "-u", "roundcube", "-proundcube123",
        "--default-character-set=utf8mb4", "roundcubemail", "-N", "-e", sql,
    )
    return out.strip()


# ── ownCloud checks ──────────────────────────────────────────────────────────

def check_1_archived_file() -> None:
    """Renamed policy file exists as *_ARCHIVED.docx in oc_filecache."""
    try:
        out = oc_sql(
            "SELECT path FROM oc_filecache "
            "WHERE path LIKE '%Telehealth_GP_Phone_Service_Fee_Alignment_Determination_2021_ARCHIVED.docx'"
        )
        found = "ARCHIVED.docx" in out
        check("1. Archived file exists", 1, found,
              f"path={out!r}" if found else "not found in oc_filecache")
    except Exception as e:
        check("1. Archived file exists", 1, False, f"exception: {e}")


def check_2_archived_tagged() -> None:
    """Archived file has system tag 'archived'."""
    try:
        fileid = oc_sql(
            "SELECT fileid FROM oc_filecache "
            "WHERE path LIKE '%Telehealth_GP_Phone_Service_Fee_Alignment_Determination_2021_ARCHIVED.docx' "
            "LIMIT 1"
        )
        if not fileid:
            check("2. Archived file tagged", 1, False, "archived file not found")
            return
        out = oc_sql(
            f"SELECT t.name FROM oc_systemtag t "
            f"JOIN oc_systemtag_object_mapping m ON t.id = m.systemtagid "
            f"WHERE m.objectid = '{fileid}' AND m.objecttype = 'files' "
            f"AND LOWER(t.name) = 'archived'"
        )
        check("2. Archived file tagged", 1, "archived" in out.lower(),
              f"tags={out!r}")
    except Exception as e:
        check("2. Archived file tagged", 1, False, f"exception: {e}")


def check_3_new_policy_file() -> None:
    """New policy file exists in oc_filecache under doc/healthcare."""
    try:
        out = oc_sql(
            "SELECT path FROM oc_filecache "
            "WHERE path LIKE '%doc/healthcare/Telehealth_Fee_Alignment_Policy_v2.txt'"
        )
        found = "Telehealth_Fee_Alignment_Policy_v2.txt" in out
        check("3. New policy file exists", 1, found,
              f"path={out!r}" if found else "not found")
    except Exception as e:
        check("3. New policy file exists", 1, False, f"exception: {e}")


def check_4_new_policy_content() -> None:
    """New policy file content is correct (via WebDAV)."""
    try:
        url = (f"http://{HOST}:{OC_PORT}/remote.php/dav/files/admin/"
               "doc/healthcare/Telehealth_Fee_Alignment_Policy_v2.txt")
        r = requests.get(url, auth=("admin", "admin"), timeout=15)
        if r.status_code != 200:
            check("4. New policy content", 2, False, f"WebDAV HTTP {r.status_code}")
            return
        c = r.text
        ok = all(phrase in c for phrase in [
            "Tier A", "Tier B", "Tier C",
            "15% premium", "30 days", "Schedule B",
        ])
        check("4. New policy content", 2, ok, f"len={len(c)}")
    except Exception as e:
        check("4. New policy content", 2, False, f"exception: {e}")



# ── OnlyOffice checks ────────────────────────────────────────────────────────

def _oo_session() -> requests.Session:
    """Authenticate to OnlyOffice and return a session with auth header."""
    s = requests.Session()
    r = s.post(
        f"http://{HOST}:{OO_PORT}/api/2.0/authentication",
        json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
        timeout=15,
    )
    r.raise_for_status()
    token = r.json()["response"]["token"]
    s.headers["Authorization"] = token
    return s


def check_7_oo_doc_exists() -> None:
    """Change summary document exists in OnlyOffice Common Documents."""
    try:
        s = _oo_session()
        r = s.get(f"http://{HOST}:{OO_PORT}/api/2.0/files/@common", timeout=15)
        data = r.json().get("response", {})
        files = data.get("files", [])
        folders = data.get("folders", [])

        target = "Telehealth Fee Alignment Policy v2 - Change Summary"
        found = any(f.get("title", "").startswith(target) for f in files)

        if not found:
            for folder in folders:
                fid = folder.get("id")
                r2 = s.get(f"http://{HOST}:{OO_PORT}/api/2.0/files/{fid}", timeout=15)
                sub = r2.json().get("response", {}).get("files", [])
                if any(f.get("title", "").startswith(target) for f in sub):
                    found = True
                    break

        check("7. OnlyOffice doc exists", 1, found,
              f"files_in_common={len(files)}")
    except Exception as e:
        check("7. OnlyOffice doc exists", 1, False, f"exception: {e}")


def check_8_oo_shared_laura() -> None:
    """Document shared with laura.brown for editing in OnlyOffice."""
    try:
        s = _oo_session()
        r = s.get(f"http://{HOST}:{OO_PORT}/api/2.0/files/@common", timeout=15)
        files = r.json().get("response", {}).get("files", [])

        target = "Telehealth Fee Alignment Policy v2 - Change Summary"
        doc_id = None
        for f in files:
            if f.get("title", "").startswith(target):
                doc_id = f.get("id")
                break

        if doc_id is None:
            check("8. OnlyOffice shared with laura.brown", 2, False, "doc not found")
            return

        r = s.get(f"http://{HOST}:{OO_PORT}/api/2.0/files/file/{doc_id}/share",
                  timeout=15)
        shares = r.json().get("response", [])
        laura_found = False
        for sh in shares:
            shared_to = sh.get("sharedTo", {})
            uname = str(shared_to.get("userName", "")).lower()
            email = str(shared_to.get("email", "")).lower()
            display = str(shared_to.get("displayName", "")).lower()
            if "laura.brown" in uname or "laura.brown" in email or "laura" in display:
                laura_found = True
                break

        check("8. OnlyOffice shared with laura.brown", 2, laura_found,
              f"shares_count={len(shares)}")
    except Exception as e:
        check("8. OnlyOffice shared with laura.brown", 2, False, f"exception: {e}")


# ── Mattermost checks ────────────────────────────────────────────────────────

def check_9_mm_announcement() -> None:
    """Policy announcement posted in Analytics channel (Marketing & Growth team)."""
    try:
        out = mm_sql(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "JOIN teams t ON c.teamid = t.id "
            "WHERE t.displayname = 'Marketing & Growth' "
            "AND c.displayname = 'Analytics' "
            "AND p.deleteat = 0 "
            "AND p.message LIKE '%Telehealth Fee Alignment Policy v2%' "
            "AND p.message LIKE '%Policy Update%'"
        )
        found = "Telehealth Fee Alignment Policy v2" in out
        check("9. MM announcement in Analytics", 1, found,
              f"msg_len={len(out)}" if found else "not found")
    except Exception as e:
        check("9. MM announcement in Analytics", 1, False, f"exception: {e}")


def check_10_mm_pinned() -> None:
    """Policy announcement is pinned in Analytics."""
    try:
        out = mm_sql(
            "SELECT p.ispinned FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "JOIN teams t ON c.teamid = t.id "
            "WHERE t.displayname = 'Marketing & Growth' "
            "AND c.displayname = 'Analytics' "
            "AND p.deleteat = 0 "
            "AND p.message LIKE '%Telehealth Fee Alignment Policy v2%' "
            "AND p.message LIKE '%Policy Update%'"
        )
        pinned = out.strip() == "t"
        check("10. MM announcement pinned", 1, pinned, f"ispinned={out.strip()!r}")
    except Exception as e:
        check("10. MM announcement pinned", 1, False, f"exception: {e}")


def check_11_mm_hr_note() -> None:
    """HR internal note posted in Brand Design channel (root post, not reply)."""
    try:
        out = mm_sql(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "JOIN teams t ON c.teamid = t.id "
            "WHERE t.displayname = 'Marketing & Growth' "
            "AND c.displayname = 'Brand Design' "
            "AND p.deleteat = 0 "
            "AND (p.rootid = '' OR p.rootid IS NULL) "
            "AND p.message LIKE '%Telehealth Fee Alignment Policy v2 rollout%'"
        )
        found = "rollout" in out.lower()
        check("11. MM HR note in Brand Design", 1, found,
              f"msg_len={len(out)}" if found else "not found")
    except Exception as e:
        check("11. MM HR note in Brand Design", 1, False, f"exception: {e}")


def check_12_mm_thread_reply() -> None:
    """Thread reply under HR note in Brand Design with follow-up content."""
    try:
        root_id = mm_sql(
            "SELECT p.id FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "JOIN teams t ON c.teamid = t.id "
            "WHERE t.displayname = 'Marketing & Growth' "
            "AND c.displayname = 'Brand Design' "
            "AND p.deleteat = 0 "
            "AND (p.rootid = '' OR p.rootid IS NULL) "
            "AND p.message LIKE '%Telehealth Fee Alignment Policy v2 rollout%' "
            "LIMIT 1"
        ).strip()
        if not root_id:
            check("12. MM thread reply in Brand Design", 2, False, "root post not found")
            return
        out = mm_sql(
            f"SELECT p.message FROM posts p "
            f"WHERE p.rootid = '{root_id}' AND p.deleteat = 0 "
            f"AND p.message LIKE '%acknowledgment%'"
        )
        found = "acknowledgment" in out.lower() or "100%" in out
        check("12. MM thread reply in Brand Design", 2, found,
              f"reply_len={len(out)}" if found else "reply not found")
    except Exception as e:
        check("12. MM thread reply in Brand Design", 2, False, f"exception: {e}")


# ── Roundcube checks ─────────────────────────────────────────────────────────

def check_13_rc_identity_org() -> None:
    """Default identity organization is 'Acme Health Services — HR Communications'."""
    try:
        out = rc_sql(
            "SELECT organization FROM identities "
            "WHERE email = 'james.whitfield@mail.local' AND standard = 1"
        )
        expected = "Acme Health Services"
        found = expected in out
        check("13. RC identity organization", 1, found, f"org={out!r}")
    except Exception as e:
        check("13. RC identity organization", 1, False, f"exception: {e}")


def check_14_rc_email_sent() -> None:
    """Email with correct subject, To, CC was sent (via doveadm or maildir grep)."""
    try:
        # Try doveadm first
        rc, out, _ = docker_exec(
            RC_CONTAINER, "doveadm", "fetch", "-u", "james.whitfield@mail.local",
            "hdr", "mailbox", "Sent", "subject", "Action Required",
            timeout=20,
        )
        if rc == 0 and out.strip():
            has_subject = "Telehealth Fee Alignment Policy v2" in out
            has_to = "all-staff@acmehealth.local" in out
            has_cc = "hr.director@acmehealth.local" in out
            ok = has_subject and has_to and has_cc
            check("14. RC email sent", 2, ok,
                  f"subject={has_subject}, to={has_to}, cc={has_cc}")
            return

        # Fallback: grep maildir
        rc2, out2, _ = docker_exec(
            RC_CONTAINER, "bash", "-c",
            "grep -rl 'Action Required' /var/mail/ /var/vmail/ 2>/dev/null || echo ''",
            timeout=20,
        )
        if out2.strip():
            first = out2.strip().split("\n")[0]
            _, content, _ = docker_exec(RC_CONTAINER, "head", "-80", first, timeout=10)
            has_subject = "Telehealth Fee Alignment Policy v2" in content
            has_to = "all-staff@acmehealth.local" in content
            has_cc = "hr.director@acmehealth.local" in content
            ok = has_subject and has_to and has_cc
            check("14. RC email sent", 2, ok,
                  f"subject={has_subject}, to={has_to}, cc={has_cc}")
        else:
            check("14. RC email sent", 2, False, "email not found via doveadm or maildir")
    except Exception as e:
        check("14. RC email sent", 2, False, f"exception: {e}")


def check_15_rc_sent_flagged() -> None:
    """Sent email is flagged (\\Flagged) in IMAP."""
    try:
        # Try doveadm flags
        rc, out, _ = docker_exec(
            RC_CONTAINER, "doveadm", "fetch", "-u", "james.whitfield@mail.local",
            "flags", "mailbox", "Sent", "subject", "Action Required",
            timeout=15,
        )
        if rc == 0 and out.strip():
            flagged = "\\Flagged" in out or "\\flagged" in out
            check("15. RC sent email flagged", 1, flagged, f"flags={out.strip()!r}")
            return

        # Fallback: check maildir filename for F flag
        rc2, out2, _ = docker_exec(
            RC_CONTAINER, "bash", "-c",
            "find /var/mail/ /var/vmail/ -path '*Sent*' -type f 2>/dev/null "
            "| xargs grep -l 'Action Required' 2>/dev/null || echo ''",
            timeout=20,
        )
        if out2.strip():
            fname = out2.strip().split("\n")[0]
            flagged = ":2," in fname and "F" in fname.split(":2,")[-1]
            check("15. RC sent email flagged", 1, flagged, f"file={fname!r}")
        else:
            check("15. RC sent email flagged", 1, False, "sent email not found")
    except Exception as e:
        check("15. RC sent email flagged", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_archived_file()
    check_2_archived_tagged()
    check_3_new_policy_file()
    check_4_new_policy_content()
    check_7_oo_doc_exists()
    check_8_oo_shared_laura()
    check_9_mm_announcement()
    check_10_mm_pinned()
    check_11_mm_hr_note()
    check_12_mm_thread_reply()
    check_13_rc_identity_org()
    check_14_rc_email_sent()
    check_15_rc_sent_flagged()

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
