# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_074/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-074-I2: Globex Q2 2026 QBR Package Preparation and Distribution

Checks: 14 weighted checks across onlyoffice, owncloud, mattermost, roundcubemail.
Strategy: docker exec (DB) primary; WebDAV for ownCloud file content; maildir grep for email.

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
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"FATAL: {name} not set", file=sys.stderr)
        sys.exit(1)
    return val


ONLYOFFICE_PORT = require_env("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = require_env("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = require_env("ONLYOFFICE_DB_CONTAINER")

OWNCLOUD_PORT = require_env("OWNCLOUD_PORT")
OWNCLOUD_CONTAINER = require_env("OWNCLOUD_CONTAINER")
OWNCLOUD_DB_CONTAINER = require_env("OWNCLOUD_DB_CONTAINER")

MATTERMOST_PORT = require_env("MATTERMOST_PORT")
MATTERMOST_CONTAINER = require_env("MATTERMOST_CONTAINER")
MATTERMOST_DB_CONTAINER = require_env("MATTERMOST_DB_CONTAINER")

ROUNDCUBEMAIL_PORT = require_env("ROUNDCUBEMAIL_PORT")
ROUNDCUBEMAIL_CONTAINER = require_env("ROUNDCUBEMAIL_CONTAINER")
ROUNDCUBEMAIL_DB_CONTAINER = require_env("ROUNDCUBEMAIL_DB_CONTAINER")

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


def docker_exec_env(container: str, env: dict, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    cmd = ["docker", "exec"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(container)
    cmd.extend(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def onlyoffice_db(sql: str) -> str:
    rc, out, err = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "onlyoffice", "-N", "-e", sql,
    )
    return out.strip()


def owncloud_db(sql: str) -> str:
    rc, out, err = docker_exec(
        OWNCLOUD_DB_CONTAINER,
        "mysql", "-u", "owncloud", "-powncloud",
        "owncloud", "-N", "-e", sql,
    )
    return out.strip()


def mattermost_db(sql: str) -> str:
    rc, out, err = docker_exec_env(
        MATTERMOST_DB_CONTAINER,
        {"PGPASSWORD": "mmuser_password"},
        "psql", "-U", "mmuser", "-d", "mattermost", "-t", "-A", "-c", sql,
    )
    return out.strip()


def roundcube_db(sql: str) -> str:
    rc, out, err = docker_exec(
        ROUNDCUBEMAIL_DB_CONTAINER,
        "mysql", "-u", "roundcube", "-proundcube123",
        "roundcubemail", "-N", "-e", sql,
    )
    return out.strip()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_1_onlyoffice_spreadsheet() -> None:
    """Spreadsheet 'Globex_Industries_Q2_2026_QBR_Data' exists in My Documents."""
    try:
        result = onlyoffice_db(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE 'Globex\\_Industries\\_Q2\\_2026\\_QBR\\_Data%'"
        )
        passed = "Globex_Industries_Q2_2026_QBR_Data" in result
        check("1. OnlyOffice spreadsheet exists", 2, passed,
              "found" if passed else "not found in files_file")
    except Exception as e:
        check("1. OnlyOffice spreadsheet exists", 2, False, f"exception: {e}")


def check_2_onlyoffice_presentation() -> None:
    """Presentation 'Globex_Industries_Q2_2026_QBR_Presentation' exists in Common Documents."""
    try:
        result = onlyoffice_db(
            "SELECT id, title FROM files_file "
            "WHERE title LIKE 'Globex\\_Industries\\_Q2\\_2026\\_QBR\\_Presentation%'"
        )
        passed = "Globex_Industries_Q2_2026_QBR_Presentation" in result
        check("2. OnlyOffice presentation exists", 2, passed,
              "found" if passed else "not found in files_file")
    except Exception as e:
        check("2. OnlyOffice presentation exists", 2, False, f"exception: {e}")


def check_3_onlyoffice_shares() -> None:
    """Presentation shared with jun.chen (editing) and amit.singh (viewing)."""
    try:
        # Get file id
        fid = onlyoffice_db(
            "SELECT id FROM files_file "
            "WHERE title LIKE 'Globex\\_Industries\\_Q2\\_2026\\_QBR\\_Presentation%' "
            "LIMIT 1"
        ).strip().split("\n")[0].strip() if onlyoffice_db(
            "SELECT id FROM files_file "
            "WHERE title LIKE 'Globex\\_Industries\\_Q2\\_2026\\_QBR\\_Presentation%' "
            "LIMIT 1"
        ).strip() else ""

        if not fid:
            check("3. OnlyOffice presentation shares", 2, False, "presentation not found")
            return

        shares = onlyoffice_db(
            f"SELECT subject, security FROM files_security WHERE entry_id = {fid}"
        )
        has_jun = "jun.chen" in shares.lower()
        has_amit = "amit.singh" in shares.lower()
        passed = has_jun and has_amit
        detail = (f"jun.chen={'found' if has_jun else 'missing'}, "
                  f"amit.singh={'found' if has_amit else 'missing'}")
        check("3. OnlyOffice presentation shares", 2, passed, detail)
    except Exception as e:
        check("3. OnlyOffice presentation shares", 2, False, f"exception: {e}")


def check_4_owncloud_folders() -> None:
    """Folder Globex_Industries_QBR_Q2_2026 with MetricsData and SlideDeck subfolders."""
    try:
        result = owncloud_db(
            "SELECT path FROM oc_filecache "
            "WHERE path LIKE 'files/Globex\\_Industries\\_QBR\\_Q2\\_2026%' "
            "AND mimetype = (SELECT id FROM oc_mimetypes WHERE mimetype='httpd/unix-directory')"
        )
        has_main = "files/Globex_Industries_QBR_Q2_2026" in result
        has_metrics = "MetricsData" in result
        has_slide = "SlideDeck" in result
        passed = has_main and has_metrics and has_slide
        detail = (f"main={'ok' if has_main else 'missing'}, "
                  f"MetricsData={'ok' if has_metrics else 'missing'}, "
                  f"SlideDeck={'ok' if has_slide else 'missing'}")
        check("4. ownCloud folder structure", 2, passed, detail)
    except Exception as e:
        check("4. ownCloud folder structure", 2, False, f"exception: {e}")


def check_5_owncloud_pdf() -> None:
    """PDF file exists in SlideDeck subfolder."""
    try:
        result = owncloud_db(
            "SELECT name FROM oc_filecache "
            "WHERE path LIKE 'files/Globex\\_Industries\\_QBR\\_Q2\\_2026/SlideDeck/%' "
            "AND name LIKE '%.pdf'"
        )
        passed = bool(result) and ".pdf" in result.lower()
        check("5. ownCloud PDF in SlideDeck", 1, passed,
              result if passed else "no PDF found in SlideDeck")
    except Exception as e:
        check("5. ownCloud PDF in SlideDeck", 1, False, f"exception: {e}")


def check_6_owncloud_metric_sources() -> None:
    """metric_sources.txt in MetricsData with correct data-source content."""
    try:
        exists = owncloud_db(
            "SELECT name FROM oc_filecache "
            "WHERE path = 'files/Globex_Industries_QBR_Q2_2026/MetricsData/metric_sources.txt'"
        )
        if not exists:
            check("6. ownCloud metric_sources.txt", 2, False, "file not found in filecache")
            return

        url = (f"http://{HOST}:{OWNCLOUD_PORT}/remote.php/dav/files/admin/"
               "Globex_Industries_QBR_Q2_2026/MetricsData/metric_sources.txt")
        resp = requests.get(url, auth=("admin", "admin"), timeout=15)
        content = resp.text

        has_hubspot = "HubSpot" in content
        has_intercom = "Intercom" in content
        has_sla = "SLA" in content
        has_nps = "AskNicely" in content

        passed = has_hubspot and has_intercom and has_sla and has_nps
        detail = (f"HubSpot={'ok' if has_hubspot else 'missing'}, "
                  f"Intercom={'ok' if has_intercom else 'missing'}, "
                  f"SLA={'ok' if has_sla else 'missing'}, "
                  f"AskNicely={'ok' if has_nps else 'missing'}")
        check("6. ownCloud metric_sources.txt", 2, passed, detail)
    except Exception as e:
        check("6. ownCloud metric_sources.txt", 2, False, f"exception: {e}")


def check_7_owncloud_shares() -> None:
    """admin group share rw on main folder, admin group share ro on SlideDeck."""
    try:
        # share_type=1 means group share
        # permissions: 1=read, 15=read+update+create+delete, 31=all
        shares = owncloud_db(
            "SELECT file_target, share_with, permissions FROM oc_share "
            "WHERE share_with = 'admin' AND share_type = 1 "
            "AND (file_target LIKE '%Globex\\_Industries\\_QBR\\_Q2\\_2026%' "
            "     OR file_target LIKE '%SlideDeck%')"
        )

        has_main_rw = False
        has_slide_ro = False

        for line in shares.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                target = parts[0]
                try:
                    perms = int(parts[2].strip())
                except ValueError:
                    continue
                if "SlideDeck" in target and perms == 1:
                    has_slide_ro = True
                elif ("Globex_Industries_QBR_Q2_2026" in target
                      and "SlideDeck" not in target
                      and perms >= 15):
                    has_main_rw = True

        passed = has_main_rw and has_slide_ro
        detail = (f"main_rw={'ok' if has_main_rw else 'missing'}, "
                  f"slidedeck_ro={'ok' if has_slide_ro else 'missing'}")
        if not shares:
            detail = "no matching group shares found"
        check("7. ownCloud shares", 2, passed, detail)
    except Exception as e:
        check("7. ownCloud shares", 2, False, f"exception: {e}")


def check_8_mattermost_channel() -> None:
    """Private channel globex-qbr-q2-review with correct header and purpose."""
    try:
        result = mattermost_db(
            "SELECT name, type, header, purpose FROM channels "
            "WHERE name = 'globex-qbr-q2-review'"
        )
        if not result:
            check("8. Mattermost channel exists", 2, False, "channel not found")
            return

        parts = result.split("|")
        is_private = len(parts) >= 2 and parts[1].strip() == "P"
        has_header = "Internal review channel for Globex Industries Q2 2026 QBR deliverables" in result
        has_purpose = "Coordinate internal review" in result

        passed = is_private and has_header and has_purpose
        detail = (f"private={'ok' if is_private else 'no'}, "
                  f"header={'ok' if has_header else 'wrong'}, "
                  f"purpose={'ok' if has_purpose else 'wrong'}")
        check("8. Mattermost channel exists", 2, passed, detail)
    except Exception as e:
        check("8. Mattermost channel exists", 2, False, f"exception: {e}")


def check_9_mattermost_review_message() -> None:
    """Review request message posted in the channel with private link and deadline."""
    try:
        result = mattermost_db(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'globex-qbr-q2-review' "
            "AND p.message LIKE '%please review the Globex Industries Q2 2026 QBR%' "
            "AND p.rootid = '' "
            "ORDER BY p.createat ASC LIMIT 1"
        )
        has_review = "please review" in result.lower()
        has_deadline = "2026-07-22" in result
        passed = has_review and has_deadline
        check("9. Mattermost review message", 2, passed,
              "found" if passed else f"review={'ok' if has_review else 'missing'}, deadline={'ok' if has_deadline else 'missing'}")
    except Exception as e:
        check("9. Mattermost review message", 2, False, f"exception: {e}")


def check_10_mattermost_thread_reply() -> None:
    """Thread reply tagging christene asking for SLA verification."""
    try:
        result = mattermost_db(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'globex-qbr-q2-review' "
            "AND p.rootid != '' "
            "AND p.message LIKE '%christene%'"
        )
        has_mention = "christene" in result.lower()
        has_sla_ref = "SLA" in result or "sla" in result.lower()
        passed = has_mention and has_sla_ref
        detail = (f"mention={'ok' if has_mention else 'missing'}, "
                  f"sla_ref={'ok' if has_sla_ref else 'missing'}")
        check("10. Mattermost thread reply @christene", 2, passed, detail)
    except Exception as e:
        check("10. Mattermost thread reply @christene", 2, False, f"exception: {e}")


def check_11_mattermost_reaction() -> None:
    """Thumbsup reaction on the original review request message."""
    try:
        result = mattermost_db(
            "SELECT r.emojiname FROM reactions r "
            "JOIN posts p ON r.postid = p.id "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'globex-qbr-q2-review' "
            "AND p.rootid = '' "
            "AND p.message LIKE '%please review%' "
            "AND r.emojiname = 'thumbsup'"
        )
        passed = "thumbsup" in result
        check("11. Mattermost thumbsup reaction", 1, passed,
              "found" if passed else "no thumbsup reaction on review message")
    except Exception as e:
        check("11. Mattermost thumbsup reaction", 1, False, f"exception: {e}")


def check_12_roundcube_identity() -> None:
    """Identity 'Marcus Torres - Sales Operations' with correct email, org, signature."""
    try:
        result = roundcube_db(
            "SELECT name, email, organization, signature FROM identities "
            "WHERE name = 'Marcus Torres - Sales Operations'"
        )
        if not result:
            check("12. Roundcube identity", 2, False, "identity not found")
            return

        has_email = "marcus.torres@mail.local" in result
        has_org = "Sales Operations - TechCorp" in result
        has_sig = "Sales Operations Manager" in result

        passed = has_email and has_org and has_sig
        detail = (f"email={'ok' if has_email else 'wrong'}, "
                  f"org={'ok' if has_org else 'wrong'}, "
                  f"sig={'ok' if has_sig else 'wrong'}")
        check("12. Roundcube identity", 2, passed, detail)
    except Exception as e:
        check("12. Roundcube identity", 2, False, f"exception: {e}")


def check_13_roundcube_email_sent() -> None:
    """Email sent to linda.park@globexindustries.com with correct subject and CC."""
    try:
        rc, out, err = docker_exec(
            ROUNDCUBEMAIL_CONTAINER,
            "grep", "-rl",
            "Subject: Globex Industries Q2 2026 Quarterly Business Review - Proposed Meeting Date",
            "/var/mail/",
            timeout=30,
        )
        has_subject = rc == 0 and out.strip() != ""

        has_to = False
        has_cc = False
        if has_subject:
            mail_file = out.strip().split("\n")[0]
            rc2, content, _ = docker_exec(
                ROUNDCUBEMAIL_CONTAINER, "cat", mail_file, timeout=15
            )
            has_to = "linda.park@globexindustries.com" in content
            has_cc = "emma.larsson@mail.local" in content

        passed = has_subject and has_to and has_cc
        detail = (f"subject={'ok' if has_subject else 'missing'}, "
                  f"to={'ok' if has_to else 'missing'}, "
                  f"cc={'ok' if has_cc else 'missing'}")
        check("13. Roundcube email sent", 2, passed, detail)
    except Exception as e:
        check("13. Roundcube email sent", 2, False, f"exception: {e}")


def check_14_roundcube_contact() -> None:
    """Contact Linda Park with email linda.park@globexindustries.com in personal address book."""
    try:
        result = roundcube_db(
            "SELECT name, email, firstname, surname FROM contacts "
            "WHERE firstname = 'Linda' AND surname = 'Park'"
        )
        if not result:
            check("14. Roundcube contact Linda Park", 1, False, "contact not found")
            return

        has_email = "linda.park@globexindustries.com" in result
        passed = has_email
        check("14. Roundcube contact Linda Park", 1, passed,
              "found with correct email" if passed else f"email mismatch: {result[:80]}")
    except Exception as e:
        check("14. Roundcube contact Linda Park", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_onlyoffice_spreadsheet()
    check_2_onlyoffice_presentation()
    check_3_onlyoffice_shares()
    check_4_owncloud_folders()
    check_5_owncloud_pdf()
    check_6_owncloud_metric_sources()
    check_7_owncloud_shares()
    check_8_mattermost_channel()
    check_9_mattermost_review_message()
    check_10_mattermost_thread_reply()
    check_11_mattermost_reaction()
    check_12_roundcube_identity()
    check_13_roundcube_email_sent()
    check_14_roundcube_contact()

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
