# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_032/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-032-I2: Healthcare Document Archive & Notification

Checks: 13 weighted checks across owncloud, onlyoffice, mattermost, roundcubemail.
Strategy: docker exec (DB queries) + API (ownCloud WebDAV for content) + docker exec (mail)

Required env vars:
  SERVER_HOSTNAME,
  OWNCLOUD_PORT, OWNCLOUD_CONTAINER, OWNCLOUD_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER
"""

import json
import os
import re
import subprocess
import sys

import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

REQUIRED_VARS = {
    "OWNCLOUD_PORT": None, "OWNCLOUD_CONTAINER": None, "OWNCLOUD_DB_CONTAINER": None,
    "ONLYOFFICE_PORT": None, "ONLYOFFICE_CONTAINER": None, "ONLYOFFICE_DB_CONTAINER": None,
    "MATTERMOST_PORT": None, "MATTERMOST_CONTAINER": None, "MATTERMOST_DB_CONTAINER": None,
    "ROUNDCUBEMAIL_PORT": None, "ROUNDCUBEMAIL_CONTAINER": None, "ROUNDCUBEMAIL_DB_CONTAINER": None,
}
for var in REQUIRED_VARS:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    REQUIRED_VARS[var] = val

OC_PORT = REQUIRED_VARS["OWNCLOUD_PORT"]
OC_CONTAINER = REQUIRED_VARS["OWNCLOUD_CONTAINER"]
OC_DB = REQUIRED_VARS["OWNCLOUD_DB_CONTAINER"]
OO_PORT = REQUIRED_VARS["ONLYOFFICE_PORT"]
OO_CONTAINER = REQUIRED_VARS["ONLYOFFICE_CONTAINER"]
OO_DB = REQUIRED_VARS["ONLYOFFICE_DB_CONTAINER"]
MM_PORT = REQUIRED_VARS["MATTERMOST_PORT"]
MM_CONTAINER = REQUIRED_VARS["MATTERMOST_CONTAINER"]
MM_DB = REQUIRED_VARS["MATTERMOST_DB_CONTAINER"]
RC_PORT = REQUIRED_VARS["ROUNDCUBEMAIL_PORT"]
RC_CONTAINER = REQUIRED_VARS["ROUNDCUBEMAIL_CONTAINER"]
RC_DB = REQUIRED_VARS["ROUNDCUBEMAIL_DB_CONTAINER"]

# ── Slot values / expected data ───────────────────────────────────────────────
ARCHIVED_FILES = [
    "ARCHIVED_Kima_w_Medical_Center_Nursing_Position_Description.docx",
    "ARCHIVED_Inflammation_Protein_Results_Mann_Whitney_U_Test_Ratios_Sensitivity_Specificity.docx",
    "ARCHIVED_MassHealth_Medicaid_CHIP_Section_1115_Demonstration_Waiver.docx",
]
ARCHIVE_FOLDER = "Obsolete_Healthcare_2026H1"
ACTIVE_DOCS_CONTENT_FRAGMENTS = [
    "Current Active Healthcare Documents:",
    "nursing_position_v3.docx",
    "inflammation_protein_final_2026.docx",
    "masshealth_1115_renewal_2026.docx",
]
MM_ARCHIVE_NOTICE_FRAGMENT = "Healthcare Archive Notice"
MM_HEADER_FRAGMENT = "Healthcare document archive audit complete 2026-04-20"
MM_DM_FRAGMENT = "revoke any external sharing links"
EMAIL_SUBJECT = "Formal Notice: Healthcare Document Supersession Effective 2026-04-20"
IDENTITY_EMAIL = "admin.healthcare@mail.local"
IDENTITY_NAME = "Admin Manager - Healthcare Records"
IDENTITY_ORG = "Corporate Records Office"
ARCHIVE_NOTICES_FOLDER = "Healthcare_Archive_Notices"

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


def oc_db_query(sql: str) -> str:
    """Query ownCloud MariaDB and return stdout."""
    _, out, _ = docker_exec(
        OC_DB, "mysql", "-u", "owncloud", "-powncloud",
        "--default-character-set=utf8mb4", "-N", "-B",
        "owncloud", "-e", sql,
        timeout=15,
    )
    return out.strip()


def oo_db_query(sql: str) -> str:
    """Query OnlyOffice MySQL and return stdout."""
    _, out, _ = docker_exec(
        OO_DB, "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "--default-character-set=utf8mb4", "-N", "-B",
        "onlyoffice", "-e", sql,
        timeout=15,
    )
    return out.strip()


def mm_db_query(sql: str) -> str:
    """Query Mattermost PostgreSQL and return stdout."""
    _, out, _ = docker_exec(
        MM_DB, "psql", "-U", "mmuser", "-d", "mattermost",
        "-t", "-A", "-c", sql,
        timeout=15,
    )
    return out.strip()


def rc_db_query(sql: str) -> str:
    """Query Roundcube MariaDB and return stdout."""
    _, out, _ = docker_exec(
        RC_DB, "mysql", "-u", "roundcube", "-proundcube123",
        "--default-character-set=utf8mb4", "-N", "-B",
        "roundcubemail", "-e", sql,
        timeout=15,
    )
    return out.strip()


# ── ownCloud checks ──────────────────────────────────────────────────────────

def check_1_archived_files_in_folder() -> None:
    """3 archived files with ARCHIVED_ prefix exist inside Obsolete_Healthcare_2026H1."""
    try:
        sql = (
            "SELECT name FROM oc_filecache "
            f"WHERE path LIKE '%{ARCHIVE_FOLDER}%' "
            "AND name LIKE 'ARCHIVED_%';"
        )
        out = oc_db_query(sql)
        found_names = [line.strip() for line in out.splitlines() if line.strip()]
        missing = [f for f in ARCHIVED_FILES if f not in found_names]
        check("1. Archived files in Obsolete_Healthcare_2026H1", 2,
              not missing,
              f"found {len(found_names)}/3" if missing else "all 3 found")
    except Exception as e:
        check("1. Archived files in Obsolete_Healthcare_2026H1", 2, False, f"exception: {e}")


def check_2_files_tagged_obsolete() -> None:
    """Each archived file is tagged 'obsolete'."""
    try:
        # Find the systemtag id for 'obsolete'
        tag_sql = "SELECT id FROM oc_systemtag WHERE name='obsolete';"
        tag_id = oc_db_query(tag_sql).strip()
        if not tag_id:
            check("2. Files tagged 'obsolete'", 1, False, "tag 'obsolete' not found")
            return

        # Count tagged file objects
        count_sql = (
            "SELECT COUNT(*) FROM oc_systemtag_object_mapping "
            f"WHERE systemtagid={tag_id} AND objecttype='files';"
        )
        count = int(oc_db_query(count_sql) or "0")
        check("2. Files tagged 'obsolete'", 1, count >= 3,
              f"{count} files tagged")
    except Exception as e:
        check("2. Files tagged 'obsolete'", 1, False, f"exception: {e}")


def check_3_active_docs_txt() -> None:
    """ACTIVE_HEALTHCARE_DOCS.txt exists in doc/healthcare with correct content."""
    try:
        url = f"http://{HOST}:{OC_PORT}/remote.php/dav/files/admin/doc/healthcare/ACTIVE_HEALTHCARE_DOCS.txt"
        r = requests.get(url, auth=("admin", "admin"), timeout=10)
        if r.status_code != 200:
            check("3. ACTIVE_HEALTHCARE_DOCS.txt content", 2, False,
                  f"HTTP {r.status_code}")
            return
        content = r.text
        missing = [frag for frag in ACTIVE_DOCS_CONTENT_FRAGMENTS if frag not in content]
        check("3. ACTIVE_HEALTHCARE_DOCS.txt content", 2, not missing,
              "all fragments found" if not missing else f"missing: {missing}")
    except Exception as e:
        check("3. ACTIVE_HEALTHCARE_DOCS.txt content", 2, False, f"exception: {e}")



# ── OnlyOffice checks ────────────────────────────────────────────────────────

def check_5_spreadsheet_exists() -> None:
    """Spreadsheet 'Healthcare_Archive_Register_2026H1' exists in Common Documents."""
    try:
        sql = (
            "SELECT id, title FROM files_file "
            "WHERE title LIKE '%Healthcare_Archive_Register_2026H1%';"
        )
        out = oo_db_query(sql)
        if not out.strip():
            # Try alternate table name
            sql2 = (
                "SELECT id, title FROM files_folder f "
                "JOIN files_file ff ON ff.folder_id = f.id "
                "WHERE ff.title LIKE '%Healthcare_Archive_Register%';"
            )
            out2 = oo_db_query(sql2)
            if not out2.strip():
                check("5. Spreadsheet in OnlyOffice", 2, False, "not found in DB")
                return
            out = out2
        check("5. Spreadsheet in OnlyOffice", 2, True, "found")
    except Exception as e:
        check("5. Spreadsheet in OnlyOffice", 2, False, f"exception: {e}")


def check_6_spreadsheet_shared() -> None:
    """Spreadsheet shared with amit.singh for editing."""
    try:
        # Find amit.singh user id, then check if they have a security entry for any file
        sql = (
            "SELECT fs.entry_id, fs.security FROM files_security fs "
            "JOIN core_user cu ON fs.subject = cu.id "
            "WHERE cu.username = 'amit.singh' "
            "OR cu.email LIKE '%amit%';"
        )
        out = oo_db_query(sql)
        if not out.strip():
            check("6. Spreadsheet shared with amit.singh", 1,
                  False, "no share entry found for amit.singh")
            return
        check("6. Spreadsheet shared with amit.singh", 1, True,
              f"share found: {out.strip().splitlines()[0]}")
    except Exception as e:
        check("6. Spreadsheet shared with amit.singh", 1, False, f"exception: {e}")


# ── Mattermost checks ────────────────────────────────────────────────────────

def check_7_archive_notice_posted() -> None:
    """Archive notice posted in 'bug-triage' channel."""
    try:
        sql = (
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'bug-triage' "
            "AND p.deleteat = 0 "
            f"AND p.message LIKE '%{MM_ARCHIVE_NOTICE_FRAGMENT}%' "
            "LIMIT 1;"
        )
        out = mm_db_query(sql)
        found = MM_ARCHIVE_NOTICE_FRAGMENT in out
        check("7. Archive notice in bug-triage", 2, found,
              "message found" if found else "not found")
    except Exception as e:
        check("7. Archive notice in bug-triage", 2, False, f"exception: {e}")


def check_8_channel_header_updated() -> None:
    """bug-triage channel header contains archive audit info."""
    try:
        sql = (
            "SELECT header FROM channels WHERE name = 'bug-triage';"
        )
        out = mm_db_query(sql)
        found = MM_HEADER_FRAGMENT in out
        check("8. bug-triage channel header updated", 2, found,
              f"header contains expected text" if found else f"got: {out[:120]}")
    except Exception as e:
        check("8. bug-triage channel header updated", 2, False, f"exception: {e}")


def check_9_dm_to_admin() -> None:
    """DM sent to 'admin' requesting permissions cleanup."""
    try:
        sql = (
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.type = 'D' "
            "AND p.deleteat = 0 "
            f"AND p.message LIKE '%{MM_DM_FRAGMENT}%' "
            "LIMIT 1;"
        )
        out = mm_db_query(sql)
        found = MM_DM_FRAGMENT in out
        check("9. DM to admin re: sharing links", 2, found,
              "DM found" if found else "not found")
    except Exception as e:
        check("9. DM to admin re: sharing links", 2, False, f"exception: {e}")


# ── Roundcube checks ─────────────────────────────────────────────────────────

def check_10_identity_created() -> None:
    """Identity with correct display name, email, org exists."""
    try:
        sql = (
            "SELECT name, email, organization FROM identities "
            f"WHERE email = '{IDENTITY_EMAIL}';"
        )
        out = rc_db_query(sql)
        if not out.strip():
            check("10. Roundcube identity created", 2, False,
                  f"no identity with email {IDENTITY_EMAIL}")
            return
        # MariaDB -N -B output: tab-separated
        parts = out.strip().split("\t")
        name_ok = IDENTITY_NAME in (parts[0] if len(parts) > 0 else "")
        org_ok = IDENTITY_ORG in (parts[2] if len(parts) > 2 else "")
        check("10. Roundcube identity created", 2, name_ok and org_ok,
              f"name={'ok' if name_ok else parts[0] if parts else 'N/A'}, "
              f"org={'ok' if org_ok else parts[2] if len(parts)>2 else 'N/A'}")
    except Exception as e:
        check("10. Roundcube identity created", 2, False, f"exception: {e}")


def check_11_email_sent() -> None:
    """Email with correct subject sent to department heads."""
    try:
        rc, out, err = docker_exec(
            RC_CONTAINER,
            "grep", "-rl", EMAIL_SUBJECT, "/var/mail/",
            timeout=15,
        )
        if rc != 0 or not out.strip():
            # Try alternative mail paths
            rc2, out2, _ = docker_exec(
                RC_CONTAINER,
                "find", "/var/mail", "-type", "f", "-name", "*",
                "-exec", "grep", "-l", EMAIL_SUBJECT, "{}", "+",
                timeout=15,
            )
            if rc2 != 0 or not out2.strip():
                check("11. Email sent with correct subject", 2, False,
                      "email not found in /var/mail/")
                return
            out = out2
        files = [f for f in out.strip().splitlines() if f.strip()]
        check("11. Email sent with correct subject", 2, len(files) > 0,
              f"found in {len(files)} mailbox(es)")
    except Exception as e:
        check("11. Email sent with correct subject", 2, False, f"exception: {e}")


def check_12_archive_notices_folder() -> None:
    """Mail folder 'Healthcare_Archive_Notices' created."""
    try:
        # Check Dovecot maildir for the folder
        rc, out, _ = docker_exec(
            RC_CONTAINER,
            "find", "/var/mail", "-type", "d",
            "-name", f"*{ARCHIVE_NOTICES_FOLDER}*",
            timeout=15,
        )
        if rc == 0 and out.strip():
            check("12. Healthcare_Archive_Notices folder", 1, True,
                  f"found: {out.strip().splitlines()[0]}")
            return

        # Try alternate: check via doveadm
        rc2, out2, _ = docker_exec(
            RC_CONTAINER,
            "bash", "-c",
            f"find /var/mail -type d | grep -i 'archive'",
            timeout=15,
        )
        found = ARCHIVE_NOTICES_FOLDER.lower() in out2.lower() if out2 else False
        check("12. Healthcare_Archive_Notices folder", 1, found,
              f"found" if found else "folder not found")
    except Exception as e:
        check("12. Healthcare_Archive_Notices folder", 1, False, f"exception: {e}")


def check_13_email_moved_to_folder() -> None:
    """Sent email moved to Healthcare_Archive_Notices folder."""
    try:
        # Look for the email inside the Healthcare_Archive_Notices maildir
        rc, out, _ = docker_exec(
            RC_CONTAINER,
            "bash", "-c",
            f"find /var/mail -path '*{ARCHIVE_NOTICES_FOLDER}*' -type f "
            f"-exec grep -l '{EMAIL_SUBJECT}' {{}} +",
            timeout=15,
        )
        if rc == 0 and out.strip():
            check("13. Email moved to Healthcare_Archive_Notices", 1, True,
                  "email found in folder")
            return

        # Try case-insensitive search
        rc2, out2, _ = docker_exec(
            RC_CONTAINER,
            "bash", "-c",
            f"find /var/mail -path '*rchive*otice*' -type f "
            f"-exec grep -l 'Supersession' {{}} +",
            timeout=15,
        )
        found = bool(out2 and out2.strip())
        check("13. Email moved to Healthcare_Archive_Notices", 1, found,
              "email found in folder" if found else "email not found in archive folder")
    except Exception as e:
        check("13. Email moved to Healthcare_Archive_Notices", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_archived_files_in_folder()
    check_2_files_tagged_obsolete()
    check_3_active_docs_txt()
    check_5_spreadsheet_exists()
    check_6_spreadsheet_shared()
    check_7_archive_notice_posted()
    check_8_channel_header_updated()
    check_9_dm_to_admin()
    check_10_identity_created()
    check_11_email_sent()
    check_12_archive_notices_folder()
    check_13_email_moved_to_folder()

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
