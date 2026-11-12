# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_014/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-014-I3: Client onboarding package across OnlyOffice, ownCloud, Mattermost, Roundcube.

Checks: 14 weighted checks across 4 sites.
Strategy: DB queries (OnlyOffice MySQL, Mattermost Postgres, Roundcube MariaDB),
          REST/WebDAV API (ownCloud), docker exec (Roundcube mail).

Required env vars:
  SERVER_HOSTNAME,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER,
  OWNCLOUD_PORT, OWNCLOUD_CONTAINER, OWNCLOUD_DB_CONTAINER,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER.
"""

import os
import sys
import json
import subprocess
import requests
import xml.etree.ElementTree as ET

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.environ.get("ONLYOFFICE_DB_CONTAINER")

OWNCLOUD_PORT = os.environ.get("OWNCLOUD_PORT")
OWNCLOUD_CONTAINER = os.environ.get("OWNCLOUD_CONTAINER")
OWNCLOUD_DB_CONTAINER = os.environ.get("OWNCLOUD_DB_CONTAINER")

MATTERMOST_PORT = os.environ.get("MATTERMOST_PORT")
MATTERMOST_CONTAINER = os.environ.get("MATTERMOST_CONTAINER")
MATTERMOST_DB_CONTAINER = os.environ.get("MATTERMOST_DB_CONTAINER")

ROUNDCUBEMAIL_PORT = os.environ.get("ROUNDCUBEMAIL_PORT")
ROUNDCUBEMAIL_CONTAINER = os.environ.get("ROUNDCUBEMAIL_CONTAINER")
ROUNDCUBEMAIL_DB_CONTAINER = os.environ.get("ROUNDCUBEMAIL_DB_CONTAINER")

_required = {
    "ONLYOFFICE_PORT": ONLYOFFICE_PORT,
    "ONLYOFFICE_CONTAINER": ONLYOFFICE_CONTAINER,
    "ONLYOFFICE_DB_CONTAINER": ONLYOFFICE_DB_CONTAINER,
    "OWNCLOUD_PORT": OWNCLOUD_PORT,
    "OWNCLOUD_CONTAINER": OWNCLOUD_CONTAINER,
    "OWNCLOUD_DB_CONTAINER": OWNCLOUD_DB_CONTAINER,
    "MATTERMOST_PORT": MATTERMOST_PORT,
    "MATTERMOST_CONTAINER": MATTERMOST_CONTAINER,
    "MATTERMOST_DB_CONTAINER": MATTERMOST_DB_CONTAINER,
    "ROUNDCUBEMAIL_PORT": ROUNDCUBEMAIL_PORT,
    "ROUNDCUBEMAIL_CONTAINER": ROUNDCUBEMAIL_CONTAINER,
    "ROUNDCUBEMAIL_DB_CONTAINER": ROUNDCUBEMAIL_DB_CONTAINER,
}
for var_name, var_val in _required.items():
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
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


def mysql_query(container: str, db: str, user: str, password: str, sql: str) -> str:
    rc, out, err = docker_exec(
        container, "mysql", f"-u{user}", f"-p{password}",
        "--default-character-set=utf8mb4", "-N", "-B", "-e", sql, db,
        timeout=15,
    )
    return out.strip()


def pg_query(container: str, db: str, user: str, sql: str) -> str:
    rc, out, err = docker_exec(
        container, "psql", "-U", user, "-d", db, "-t", "-A", "-c", sql,
        timeout=15,
    )
    return out.strip()


# ── OnlyOffice checks ────────────────────────────────────────────────────────
def check_1_welcome_letter_exists():
    """Check that 'Meridian Biotech - Welcome Letter' exists in OnlyOffice files."""
    try:
        sql = "SELECT id, title FROM files_file WHERE title = 'Meridian Biotech - Welcome Letter';"
        out = mysql_query(ONLYOFFICE_DB_CONTAINER, "onlyoffice", "onlyoffice_user", "onlyoffice_pass", sql)
        found = "Meridian Biotech - Welcome Letter" in out
        check("1. OnlyOffice: Welcome Letter document exists", 1, found,
              f"found={found}, query_result={out[:200]}")
    except Exception as e:
        check("1. OnlyOffice: Welcome Letter document exists", 1, False, f"exception: {e}")


def check_2_service_agreement_exists():
    """Check that 'Meridian Biotech - Service Agreement' exists in OnlyOffice files."""
    try:
        sql = "SELECT id, title FROM files_file WHERE title = 'Meridian Biotech - Service Agreement';"
        out = mysql_query(ONLYOFFICE_DB_CONTAINER, "onlyoffice", "onlyoffice_user", "onlyoffice_pass", sql)
        found = "Meridian Biotech - Service Agreement" in out
        check("2. OnlyOffice: Service Agreement document exists", 1, found,
              f"found={found}, query_result={out[:200]}")
    except Exception as e:
        check("2. OnlyOffice: Service Agreement document exists", 1, False, f"exception: {e}")


def check_3_service_agreement_favorite():
    """Check that Service Agreement is marked as favorite."""
    try:
        # files_tag / files_tag_link or a flag column -- try checking tag/favorite status
        # OnlyOffice uses files_tag with tag_name='favorite' linked via files_tag_link
        sql = (
            "SELECT f.title FROM files_file f "
            "INNER JOIN files_tag_link tl ON tl.entry_id = f.id AND tl.entry_type = 1 "
            "INNER JOIN files_tag t ON t.id = tl.tag_id AND t.name = 'favorite' "
            "WHERE f.title = 'Meridian Biotech - Service Agreement';"
        )
        out = mysql_query(ONLYOFFICE_DB_CONTAINER, "onlyoffice", "onlyoffice_user", "onlyoffice_pass", sql)
        found = "Meridian Biotech - Service Agreement" in out
        if not found:
            # Fallback: try flag column approach
            sql2 = (
                "SELECT f.title FROM files_file f "
                "INNER JOIN files_tag_link tl ON tl.entry_id = CAST(f.id AS CHAR) "
                "INNER JOIN files_tag t ON t.id = tl.tag_id "
                "WHERE f.title = 'Meridian Biotech - Service Agreement' AND t.name = 'favorite';"
            )
            out2 = mysql_query(ONLYOFFICE_DB_CONTAINER, "onlyoffice", "onlyoffice_user", "onlyoffice_pass", sql2)
            found = "Meridian Biotech - Service Agreement" in out2
        check("3. OnlyOffice: Service Agreement marked as favorite", 2, found,
              f"found={found}")
    except Exception as e:
        check("3. OnlyOffice: Service Agreement marked as favorite", 2, False, f"exception: {e}")


# ── ownCloud checks (WebDAV + OCS API) ───────────────────────────────────────
def _oc_base():
    return f"http://{HOST}:{OWNCLOUD_PORT}"


def check_4_owncloud_folder_structure():
    """Check Meridian-Biotech-Onboarding folder with Client-Messages and Signed-Agreements subfolders."""
    try:
        base = _oc_base()
        auth = ("admin", "admin")
        # PROPFIND on the main folder
        url = f"{base}/remote.php/dav/files/admin/Meridian-Biotech-Onboarding/"
        r = requests.request("PROPFIND", url, auth=auth, headers={"Depth": "1"}, timeout=15)
        body = r.text
        has_main = r.status_code in (207, 200)
        has_client_messages = "Client-Messages" in body
        has_signed_agreements = "Signed-Agreements" in body
        all_ok = has_main and has_client_messages and has_signed_agreements
        check("4. ownCloud: Folder structure (main + 2 subfolders)", 2, all_ok,
              f"main={has_main}, Client-Messages={has_client_messages}, Signed-Agreements={has_signed_agreements}")
    except Exception as e:
        check("4. ownCloud: Folder structure (main + 2 subfolders)", 2, False, f"exception: {e}")


def check_5_onboarding_plan_txt():
    """Check meridian-onboarding-plan.txt exists with correct content."""
    try:
        base = _oc_base()
        auth = ("admin", "admin")
        url = f"{base}/remote.php/dav/files/admin/Meridian-Biotech-Onboarding/meridian-onboarding-plan.txt"
        r = requests.get(url, auth=auth, timeout=15)
        if r.status_code != 200:
            check("5. ownCloud: meridian-onboarding-plan.txt content", 2, False,
                  f"HTTP {r.status_code}")
            return
        content = r.text
        expected_lines = [
            "Meridian Biotech Onboarding Plan:",
            "Kickoff workshop scheduled",
            "Welcome letter distributed",
            "Service agreement executed",
            "Lab systems integration scoped",
            "Validation protocol draft circulated",
            "Production cutover date confirmed",
            "60-day post-launch review booked",
        ]
        missing = [ln for ln in expected_lines if ln not in content]
        check("5. ownCloud: meridian-onboarding-plan.txt content", 2, len(missing) == 0,
              f"missing_lines={missing}" if missing else "all lines present")
    except Exception as e:
        check("5. ownCloud: meridian-onboarding-plan.txt content", 2, False, f"exception: {e}")


def check_6_public_share_client_messages():
    """Check Client-Messages has a public share link with password."""
    try:
        base = _oc_base()
        auth = ("admin", "admin")
        url = f"{base}/ocs/v2.php/apps/files_sharing/api/v1/shares?format=json"
        r = requests.get(url, auth=auth, headers={"OCS-APIREQUEST": "true"}, timeout=15)
        data = r.json()
        shares = data.get("ocs", {}).get("data", [])
        # Find public share on Client-Messages
        found_public = False
        has_password = False
        for s in shares:
            path = s.get("path", "")
            share_type = s.get("share_type")
            if "Client-Messages" in path and share_type == 3:  # 3 = public link
                found_public = True
                # password presence: share_with is set or password-protected fields
                if s.get("share_with") or s.get("share_with_displayname"):
                    has_password = True
                break
        passed = found_public and has_password
        check("6. ownCloud: Client-Messages public share with password", 2, passed,
              f"public_link={found_public}, password_set={has_password}")
    except Exception as e:
        check("6. ownCloud: Client-Messages public share with password", 2, False, f"exception: {e}")



# ── Mattermost checks (DB) ───────────────────────────────────────────────────
def check_8_mm_channel_exists():
    """Check client-meridian-biotech channel exists in Product & Design with correct purpose."""
    try:
        sql = (
            "SELECT c.name, c.purpose, c.type FROM channels c "
            "INNER JOIN teams t ON t.id = c.teamid "
            "WHERE c.name = 'client-meridian-biotech' "
            "AND t.displayname = 'Product & Design';"
        )
        out = pg_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", sql)
        if not out:
            check("8. Mattermost: Channel exists in Product & Design", 2, False, "channel not found")
            return
        parts = out.split("|")
        channel_name = parts[0].strip() if len(parts) > 0 else ""
        purpose = parts[1].strip() if len(parts) > 1 else ""
        ch_type = parts[2].strip() if len(parts) > 2 else ""
        expected_purpose = "Dedicated workspace for coordinating the Meridian Biotech onboarding program, product alignment, and ongoing partnership updates."
        name_ok = channel_name == "client-meridian-biotech"
        purpose_ok = expected_purpose in purpose
        type_ok = ch_type == "O"  # O = open/public
        passed = name_ok and purpose_ok and type_ok
        check("8. Mattermost: Channel exists in Product & Design", 2, passed,
              f"name={name_ok}, purpose={purpose_ok}, public={type_ok}")
    except Exception as e:
        check("8. Mattermost: Channel exists in Product & Design", 2, False, f"exception: {e}")


def check_9_mm_kickoff_message():
    """Check kickoff message posted in channel."""
    try:
        sql = (
            "SELECT p.id, p.message FROM posts p "
            "INNER JOIN channels c ON c.id = p.channelid "
            "WHERE c.name = 'client-meridian-biotech' "
            "AND p.message LIKE '%Kicking off the Meridian Biotech client channel%' "
            "AND p.deleteat = 0 "
            "LIMIT 1;"
        )
        out = pg_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", sql)
        found = "Kicking off the Meridian Biotech" in out
        check("9. Mattermost: Kickoff message posted", 1, found,
              f"found={found}")
    except Exception as e:
        check("9. Mattermost: Kickoff message posted", 1, False, f"exception: {e}")


def check_10_mm_thread_reply():
    """Check threaded reply with action items exists."""
    try:
        # Find the kickoff post id first
        sql_root = (
            "SELECT p.id FROM posts p "
            "INNER JOIN channels c ON c.id = p.channelid "
            "WHERE c.name = 'client-meridian-biotech' "
            "AND p.message LIKE '%Kicking off the Meridian Biotech client channel%' "
            "AND p.deleteat = 0 "
            "LIMIT 1;"
        )
        root_id = pg_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", sql_root).strip()
        if not root_id:
            check("10. Mattermost: Thread reply with action items", 2, False, "root post not found")
            return
        sql_reply = (
            f"SELECT message FROM posts "
            f"WHERE rootid = '{root_id}' "
            f"AND message LIKE '%Action items%' "
            f"AND deleteat = 0 LIMIT 1;"
        )
        out = pg_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", sql_reply)
        has_reply = "Action items" in out
        has_key_items = "countersign the service agreement" in out and "kickoff workshop" in out
        passed = has_reply and has_key_items
        check("10. Mattermost: Thread reply with action items", 2, passed,
              f"reply_found={has_reply}, key_items={has_key_items}")
    except Exception as e:
        check("10. Mattermost: Thread reply with action items", 2, False, f"exception: {e}")


def check_11_mm_handshake_reaction():
    """Check handshake reaction on kickoff message."""
    try:
        sql = (
            "SELECT r.emojiname FROM reactions r "
            "INNER JOIN posts p ON p.id = r.postid "
            "INNER JOIN channels c ON c.id = p.channelid "
            "WHERE c.name = 'client-meridian-biotech' "
            "AND p.message LIKE '%Kicking off the Meridian Biotech client channel%' "
            "AND r.emojiname = 'handshake' "
            "AND p.deleteat = 0 LIMIT 1;"
        )
        out = pg_query(MATTERMOST_DB_CONTAINER, "mattermost", "mmuser", sql)
        found = "handshake" in out
        check("11. Mattermost: Handshake reaction on kickoff message", 1, found,
              f"found={found}")
    except Exception as e:
        check("11. Mattermost: Handshake reaction on kickoff message", 1, False, f"exception: {e}")


# ── Roundcube checks (DB + mail) ─────────────────────────────────────────────
def check_12_roundcube_contacts():
    """Check 5 contacts exist in Personal Address Book."""
    try:
        expected_emails = [
            "eleanor.chadwick@meridianbiotech.com",
            "rajiv.venkatraman@meridianbiotech.com",
            "astrid.lindqvist@meridianbiotech.com",
            "omar.elsayed@meridianbiotech.com",
            "nora.whitfield@meridianbiotech.com",
        ]
        sql = "SELECT email FROM contacts WHERE email LIKE '%meridianbiotech.com%';"
        out = mysql_query(ROUNDCUBEMAIL_DB_CONTAINER, "roundcubemail", "roundcube", "roundcube123", sql)
        found_count = 0
        for em in expected_emails:
            if em in out.lower():
                found_count += 1
        passed = found_count >= 5
        check("12. Roundcube: 5 Meridian contacts in address book", 2, passed,
              f"found {found_count}/5 contacts")
    except Exception as e:
        check("12. Roundcube: 5 Meridian contacts in address book", 2, False, f"exception: {e}")


def check_13_roundcube_contact_group():
    """Check contact group 'Meridian Biotech Contacts' exists."""
    try:
        sql = "SELECT name FROM contactgroups WHERE name = 'Meridian Biotech Contacts';"
        out = mysql_query(ROUNDCUBEMAIL_DB_CONTAINER, "roundcubemail", "roundcube", "roundcube123", sql)
        found = "Meridian Biotech Contacts" in out
        check("13. Roundcube: Contact group 'Meridian Biotech Contacts' exists", 1, found,
              f"found={found}")
    except Exception as e:
        check("13. Roundcube: Contact group 'Meridian Biotech Contacts' exists", 1, False, f"exception: {e}")


def check_14_roundcube_email_sent():
    """Check email with correct subject was sent (search in mail files)."""
    try:
        expected_subject = "Welcome to Meridian Biotech Onboarding - Your Partnership Package Inside"
        # Search for the email in the mail directory
        rc, out, err = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "grep", "-rl", f"Subject: {expected_subject}", "/var/mail/",
            timeout=15,
        )
        found_by_grep = bool(out.strip())
        if not found_by_grep:
            # Fallback: check postfix mail log
            rc2, out2, err2 = docker_exec(
                ROUNDCUBEMAIL_CONTAINER,
                "bash", "-c",
                f"find /var/mail/ -type f | xargs grep -l 'Partnership Package' 2>/dev/null || true",
                timeout=15,
            )
            found_by_grep = bool(out2.strip())
        check("14. Roundcube: Onboarding email sent with correct subject", 2, found_by_grep,
              f"email_found={found_by_grep}")
    except Exception as e:
        check("14. Roundcube: Onboarding email sent with correct subject", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_welcome_letter_exists()
    check_2_service_agreement_exists()
    check_3_service_agreement_favorite()
    check_4_owncloud_folder_structure()
    check_5_onboarding_plan_txt()
    check_6_public_share_client_messages()
    check_8_mm_channel_exists()
    check_9_mm_kickoff_message()
    check_10_mm_thread_reply()
    check_11_mm_handshake_reaction()
    check_12_roundcube_contacts()
    check_13_roundcube_contact_group()
    check_14_roundcube_email_sent()

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
