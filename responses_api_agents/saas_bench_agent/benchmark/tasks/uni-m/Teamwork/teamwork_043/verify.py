# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_043/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-043-I3: Archive expiring PPT files and notify stakeholders

Checks: 13 weighted checks across owncloud, onlyoffice, mattermost, roundcubemail.
Strategy: docker exec DB for owncloud/mattermost/roundcubemail,
          REST API for onlyoffice, docker exec maildir for roundcubemail email.

Required env vars:
  SERVER_HOSTNAME,
  OWNCLOUD_PORT, OWNCLOUD_CONTAINER, OWNCLOUD_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER
"""

import os
import sys
import subprocess
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")


def _require(var: str) -> str:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    return val


OWNCLOUD_PORT = _require("OWNCLOUD_PORT")
OWNCLOUD_CONTAINER = _require("OWNCLOUD_CONTAINER")
OWNCLOUD_DB_CONTAINER = _require("OWNCLOUD_DB_CONTAINER")
ONLYOFFICE_PORT = _require("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = _require("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = _require("ONLYOFFICE_DB_CONTAINER")
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


def oc_db(sql: str) -> str:
    """Query ownCloud MariaDB."""
    _, out, _ = docker_exec(
        OWNCLOUD_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "owncloud", "-powncloud", "owncloud", "-N", "-e", sql,
    )
    return out.strip()


def mm_db(sql: str) -> str:
    """Query Mattermost PostgreSQL."""
    _, out, _ = docker_exec(
        MATTERMOST_DB_CONTAINER,
        "psql", "-U", "mmuser", "-d", "mattermost", "-t", "-A", "-c", sql,
    )
    return out.strip()


def rc_db(sql: str) -> str:
    """Query Roundcube MariaDB."""
    _, out, _ = docker_exec(
        ROUNDCUBEMAIL_DB_CONTAINER,
        "mysql", "--default-character-set=utf8mb4",
        "-u", "roundcube", "-proundcube123", "roundcubemail", "-N", "-e", sql,
    )
    return out.strip()


# ── Constants ─────────────────────────────────────────────────────────────────
EXPIRED_FILES = [
    "IDCC2022_FosterinCollaborationDMP_JAC_EXPIRED_2025-09-30.pptx",
    "Ionescu_S1_EXPIRED_2025-09-30.pptx",
    "Li_et_al_ESR1_mutations_paper_updated_SM_EXPIRED_2025-09-30.pptx",
    "Module5-Repositories_presentation_EXPIRED_2025-09-30.pptx",
    "Fathallah_Exeter University_November 2023_slides_EXPIRED_2025-09-30.pptx",
]

REPLACEMENT_FILES = [
    "IDCC2022_FosterinCollaborationDMP_JAC_2025Q4.txt",
    "Ionescu_S1_2025Q4.txt",
    "Li_et_al_ESR1_mutations_paper_updated_SM_2025Q4.txt",
    "Module5-Repositories_presentation_2025Q4.txt",
    "Fathallah_Exeter_University_November_2023_slides_2025Q4.txt",
]


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_retired_folder() -> None:
    """Verify Retired_2025Q3 folder exists inside ppt in ownCloud."""
    try:
        result = oc_db(
            "SELECT COUNT(*) FROM oc_filecache "
            "WHERE path LIKE '%/ppt/Retired\\_2025Q3' "
            "OR path = 'files/ppt/Retired_2025Q3'"
        )
        found = int(result) > 0
        check("1. ownCloud: Retired_2025Q3 folder in ppt", 1, found,
              "found" if found else "not found")
    except Exception as e:
        check("1. ownCloud: Retired_2025Q3 folder in ppt", 1, False, f"exception: {e}")


def check_2_expired_files() -> None:
    """Verify 5 expired/renamed files exist in Retired_2025Q3."""
    try:
        result = oc_db(
            "SELECT name FROM oc_filecache "
            "WHERE path LIKE '%/Retired\\_2025Q3/%' "
            "AND name LIKE '%\\_EXPIRED\\_2025-09-30%'"
        )
        found_names = {n.strip() for n in result.split("\n") if n.strip()} if result else set()
        found_count = sum(1 for ef in EXPIRED_FILES if ef in found_names)
        missing = [ef for ef in EXPIRED_FILES if ef not in found_names]
        detail = f"{found_count}/5 found"
        if missing:
            detail += f"; missing e.g. {missing[0][:50]}"
        check("2. ownCloud: expired files in Retired_2025Q3", 2, found_count == 5, detail)
    except Exception as e:
        check("2. ownCloud: expired files in Retired_2025Q3", 2, False, f"exception: {e}")


def check_3_archived_tags() -> None:
    """Verify expired files tagged 'archived' and not 'pending'."""
    try:
        archived_count = int(oc_db(
            "SELECT COUNT(DISTINCT fc.fileid) FROM oc_filecache fc "
            "JOIN oc_systemtag_object_mapping m ON CAST(fc.fileid AS CHAR) = m.objectid "
            "JOIN oc_systemtag st ON st.id = m.systemtagid "
            "WHERE fc.path LIKE '%/Retired\\_2025Q3/%' "
            "AND fc.name LIKE '%\\_EXPIRED\\_2025-09-30%' "
            "AND m.objecttype = 'files' AND st.name = 'archived'"
        ) or "0")
        pending_count = int(oc_db(
            "SELECT COUNT(DISTINCT fc.fileid) FROM oc_filecache fc "
            "JOIN oc_systemtag_object_mapping m ON CAST(fc.fileid AS CHAR) = m.objectid "
            "JOIN oc_systemtag st ON st.id = m.systemtagid "
            "WHERE fc.path LIKE '%/Retired\\_2025Q3/%' "
            "AND fc.name LIKE '%\\_EXPIRED\\_2025-09-30%' "
            "AND m.objecttype = 'files' AND st.name = 'pending'"
        ) or "0")
        passed = archived_count >= 5 and pending_count == 0
        check("3. ownCloud: expired files tagged archived/not pending", 2, passed,
              f"archived={archived_count}, pending={pending_count}")
    except Exception as e:
        check("3. ownCloud: expired files tagged archived/not pending", 2, False, f"exception: {e}")


def check_4_replacement_files() -> None:
    """Verify 5 replacement .txt files exist in ppt folder (or subfolders)."""
    try:
        result = oc_db(
            "SELECT name FROM oc_filecache "
            "WHERE path LIKE '%/ppt/%' "
            "AND path NOT LIKE '%/Retired\\_2025Q3/%' "
            "AND name LIKE '%\\_2025Q4.txt'"
        )
        found_names = {n.strip() for n in result.split("\n") if n.strip()} if result else set()
        found_count = sum(1 for rf in REPLACEMENT_FILES if rf in found_names)
        check("4. ownCloud: replacement files in ppt", 2, found_count == 5,
              f"{found_count}/5 found")
    except Exception as e:
        check("4. ownCloud: replacement files in ppt", 2, False, f"exception: {e}")


def check_5_approved_tags() -> None:
    """Verify replacement files tagged 'approved'."""
    try:
        approved_count = int(oc_db(
            "SELECT COUNT(DISTINCT fc.fileid) FROM oc_filecache fc "
            "JOIN oc_systemtag_object_mapping m ON CAST(fc.fileid AS CHAR) = m.objectid "
            "JOIN oc_systemtag st ON st.id = m.systemtagid "
            "WHERE fc.path LIKE '%/ppt/%' "
            "AND fc.path NOT LIKE '%/Retired\\_2025Q3/%' "
            "AND fc.name LIKE '%\\_2025Q4.txt' "
            "AND m.objecttype = 'files' AND st.name = 'approved'"
        ) or "0")
        check("5. ownCloud: replacement files tagged approved", 1, approved_count >= 5,
              f"{approved_count}/5 tagged")
    except Exception as e:
        check("5. ownCloud: replacement files tagged approved", 1, False, f"exception: {e}")


def check_6_oo_spreadsheet() -> None:
    """Verify Presentation_Renewal_Register_2025Q3 exists in OnlyOffice Common Docs."""
    try:
        base = f"http://{HOST}:{ONLYOFFICE_PORT}"
        auth = requests.post(
            f"{base}/api/2.0/authentication",
            json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
            timeout=15,
        )
        token = auth.json()["response"]["token"]
        hdrs = {"Authorization": f"Bearer {token}"}

        resp = requests.get(f"{base}/api/2.0/files/@common", headers=hdrs, timeout=15)
        data = resp.json().get("response", {})
        files = data.get("files", []) if isinstance(data, dict) else []
        found = any("Presentation_Renewal_Register_2025Q3" in f.get("title", "") for f in files)
        check("6. OnlyOffice: spreadsheet exists", 1, found,
              "found" if found else "not in Common Documents")
    except Exception as e:
        check("6. OnlyOffice: spreadsheet exists", 1, False, f"exception: {e}")


def _oo_get_file_id() -> tuple[str, str, int | None]:
    """Authenticate to OnlyOffice and find the spreadsheet. Returns (base, token, file_id)."""
    base = f"http://{HOST}:{ONLYOFFICE_PORT}"
    auth = requests.post(
        f"{base}/api/2.0/authentication",
        json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
        timeout=15,
    )
    token = auth.json()["response"]["token"]
    hdrs = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{base}/api/2.0/files/@common", headers=hdrs, timeout=15)
    data = resp.json().get("response", {})
    files = data.get("files", []) if isinstance(data, dict) else []
    for f in files:
        if "Presentation_Renewal_Register_2025Q3" in f.get("title", ""):
            return base, token, f["id"]
    return base, token, None


def check_7_oo_sharing() -> None:
    """Verify spreadsheet shared with Jun Chen (edit) and Laura Brown (view)."""
    try:
        base, token, file_id = _oo_get_file_id()
        if file_id is None:
            check("7. OnlyOffice: spreadsheet sharing", 2, False, "file not found")
            return

        hdrs = {"Authorization": f"Bearer {token}"}
        share_resp = requests.get(
            f"{base}/api/2.0/files/file/{file_id}/share", headers=hdrs, timeout=15,
        )
        shares = share_resp.json().get("response", [])
        jun_edit = False
        laura_view = False
        for s in shares:
            shared_to = s.get("sharedTo", {})
            name = shared_to.get("displayName", "")
            access = s.get("access", -1)
            # OnlyOffice access levels: 0=FullAccess, 1=ReadWrite, 2=Read
            if "Jun Chen" in name and access in (0, 1):
                jun_edit = True
            if "Laura Brown" in name and access == 2:
                laura_view = True
        passed = jun_edit and laura_view
        check("7. OnlyOffice: spreadsheet sharing", 2, passed,
              f"Jun_edit={jun_edit}, Laura_view={laura_view}")
    except Exception as e:
        check("7. OnlyOffice: spreadsheet sharing", 2, False, f"exception: {e}")


def check_8_mm_renewal_notice() -> None:
    """Verify renewal notice posted in UX Research channel."""
    try:
        result = mm_db(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.displayname = 'UX Research' "
            "AND p.deleteat = 0 "
            "AND p.message LIKE '%Presentation Document Renewal Notice%' "
            "LIMIT 1"
        )
        found = bool(result) and "IDCC2022" in result and "Ionescu" in result
        check("8. Mattermost: renewal notice in UX Research", 2, found,
              "found with file refs" if found else "not found or incomplete")
    except Exception as e:
        check("8. Mattermost: renewal notice in UX Research", 2, False, f"exception: {e}")


def check_9_mm_purpose() -> None:
    """Verify UX Research channel purpose updated."""
    try:
        result = mm_db(
            "SELECT purpose FROM channels WHERE displayname = 'UX Research' LIMIT 1"
        )
        expected = "track Q3 2025 presentation renewals"
        passed = expected in (result or "")
        check("9. Mattermost: UX Research purpose updated", 1, passed,
              "matched" if passed else f"got: {(result or '')[:80]}")
    except Exception as e:
        check("9. Mattermost: UX Research purpose updated", 1, False, f"exception: {e}")


def check_10_mm_group_dm() -> None:
    """Verify group DM sent to genesis, ginny, nilda."""
    try:
        result = mm_db(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.type = 'G' AND p.deleteat = 0 "
            "AND p.message LIKE '%presentation files you previously contributed%' "
            "LIMIT 1"
        )
        found = bool(result) and "retired as of 2025-10-01" in (result or "")
        check("10. Mattermost: group DM to genesis/ginny/nilda", 2, found,
              "found" if found else "not found in group channels")
    except Exception as e:
        check("10. Mattermost: group DM to genesis/ginny/nilda", 2, False, f"exception: {e}")


def check_11_rc_archive_pref() -> None:
    """Verify Roundcube archive folder preference set to 'Archive'."""
    try:
        result = rc_db(
            "SELECT preferences FROM users "
            "WHERE username = 'james.whitfield@mail.local'"
        )
        passed = "archive_mbox" in (result or "") and "Archive" in (result or "")
        check("11. Roundcube: archive folder set to Archive", 1, passed,
              "preference found" if passed else "archive_mbox not set or user not found")
    except Exception as e:
        check("11. Roundcube: archive folder set to Archive", 1, False, f"exception: {e}")


def check_12_rc_email_sent() -> None:
    """Verify email sent with correct subject and high priority."""
    try:
        rc, out, _ = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "bash", "-c",
            "grep -rl 'Subject: Q3 2025 Presentation Document Renewal Notification' "
            "/var/mail/ 2>/dev/null || true",
            timeout=15,
        )
        mail_files = [f.strip() for f in out.strip().split("\n") if f.strip()]
        found = len(mail_files) > 0
        high_priority = False
        if mail_files:
            rc2, out2, _ = docker_exec(
                ROUNDCUBEMAIL_CONTAINER,
                "bash", "-c",
                f"grep -iE 'X-Priority: 1|Importance: high' "
                f"'{mail_files[0]}' 2>/dev/null || true",
                timeout=15,
            )
            high_priority = bool(out2.strip())
        passed = found and high_priority
        check("12. Roundcube: email sent with high priority", 2, passed,
              f"email_found={found}, high_priority={high_priority}")
    except Exception as e:
        check("12. Roundcube: email sent with high priority", 2, False, f"exception: {e}")


def check_13_rc_email_archived() -> None:
    """Verify sent email archived to Archive folder."""
    try:
        rc, out, _ = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "bash", "-c",
            "grep -rl 'Subject: Q3 2025 Presentation Document Renewal Notification' "
            "/var/mail/ 2>/dev/null | grep -i archive || true",
            timeout=15,
        )
        found = bool(out.strip())
        check("13. Roundcube: email archived", 1, found,
              "found in Archive" if found else "not in Archive folder")
    except Exception as e:
        check("13. Roundcube: email archived", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_retired_folder()
    check_2_expired_files()
    check_3_archived_tags()
    check_4_replacement_files()
    check_5_approved_tags()
    check_6_oo_spreadsheet()
    check_7_oo_sharing()
    check_8_mm_renewal_notice()
    check_9_mm_purpose()
    check_10_mm_group_dm()
    check_11_rc_archive_pref()
    check_12_rc_email_sent()
    check_13_rc_email_archived()

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
