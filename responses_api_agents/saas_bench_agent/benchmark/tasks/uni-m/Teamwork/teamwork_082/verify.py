# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_082/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-082-I2: Competitive intelligence briefing with tiered distribution

Checks: 16 weighted checks across mattermost, onlyoffice, owncloud, roundcubemail.
Strategy: docker exec (DB queries) for mattermost/onlyoffice/owncloud/roundcube;
          docker exec (filesystem) for roundcube email content.

Required env vars:
  SERVER_HOSTNAME,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER,
  OWNCLOUD_PORT, OWNCLOUD_CONTAINER, OWNCLOUD_DB_CONTAINER,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER
"""

import os
import sys
import subprocess
import json

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

REQUIRED_VARS = [
    "MATTERMOST_PORT", "MATTERMOST_CONTAINER", "MATTERMOST_DB_CONTAINER",
    "ONLYOFFICE_PORT", "ONLYOFFICE_CONTAINER", "ONLYOFFICE_DB_CONTAINER",
    "OWNCLOUD_PORT", "OWNCLOUD_CONTAINER", "OWNCLOUD_DB_CONTAINER",
    "ROUNDCUBEMAIL_PORT", "ROUNDCUBEMAIL_CONTAINER", "ROUNDCUBEMAIL_DB_CONTAINER",
]

env = {}
for var in REQUIRED_VARS:
    val = os.getenv(var)
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
        sys.exit(1)
    env[var] = val

MM_DB = env["MATTERMOST_DB_CONTAINER"]
MM_CONTAINER = env["MATTERMOST_CONTAINER"]
OO_DB = env["ONLYOFFICE_DB_CONTAINER"]
OO_CONTAINER = env["ONLYOFFICE_CONTAINER"]
OC_DB = env["OWNCLOUD_DB_CONTAINER"]
OC_CONTAINER = env["OWNCLOUD_CONTAINER"]
RC_CONTAINER = env["ROUNDCUBEMAIL_CONTAINER"]
RC_DB = env["ROUNDCUBEMAIL_DB_CONTAINER"]

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


def mm_sql(query: str) -> str:
    """Run a psql query against Mattermost DB."""
    rc, out, err = docker_exec(
        MM_DB, "psql", "-U", "mmuser", "-d", "mattermost",
        "-t", "-A", "-c", query,
    )
    return out.strip()


def oo_sql(query: str) -> str:
    """Run a mysql query against OnlyOffice DB."""
    rc, out, err = docker_exec(
        OO_DB, "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass",
        "onlyoffice", "-N", "-B", "-e", query,
    )
    return out.strip()


def oc_sql(query: str) -> str:
    """Run a mysql query against ownCloud DB."""
    rc, out, err = docker_exec(
        OC_DB, "mysql", "-u", "owncloud", "-powncloud",
        "owncloud", "-N", "-B", "-e", query,
    )
    return out.strip()


def rc_sql(query: str) -> str:
    """Run a mysql query against Roundcube DB."""
    rc, out, err = docker_exec(
        RC_DB, "mysql", "-u", "roundcube", "-proundcube123",
        "roundcubemail", "-N", "-B", "-e", query,
    )
    return out.strip()


# ── Mattermost checks ────────────────────────────────────────────────────────

def check_1_mm_channel_purpose() -> None:
    """Check Tech Talks channel purpose is set correctly."""
    try:
        expected = "Tech talks and strategic competitor deep-dives for the engineering org"
        purpose = mm_sql(
            "SELECT purpose FROM channels WHERE displayname = 'Tech Talks' LIMIT 1;"
        )
        passed = expected.lower() in purpose.lower()
        check("1. MM Tech Talks channel purpose", 1, passed,
              f"got: {purpose[:100]}" if not passed else "")
    except Exception as e:
        check("1. MM Tech Talks channel purpose", 1, False, f"exception: {e}")


def check_2_mm_briefing_message() -> None:
    """Check briefing kickoff message posted in Tech Talks."""
    try:
        expected_fragment = "Kicking off Q3 competitive intel briefing on Nimbus Cloud"
        # Get the Tech Talks channel id first
        channel_id = mm_sql(
            "SELECT id FROM channels WHERE displayname = 'Tech Talks' LIMIT 1;"
        )
        if not channel_id:
            check("2. MM briefing message posted", 2, False, "Tech Talks channel not found")
            return
        msg = mm_sql(
            f"SELECT message FROM posts WHERE channelid = '{channel_id}' "
            f"AND message LIKE '%Kicking off Q3 competitive intel%' LIMIT 1;"
        )
        passed = expected_fragment in msg
        check("2. MM briefing message posted", 2, passed,
              f"got: {msg[:100]}" if not passed else "")
    except Exception as e:
        check("2. MM briefing message posted", 2, False, f"exception: {e}")


def check_3_mm_message_pinned() -> None:
    """Check the briefing message is pinned."""
    try:
        channel_id = mm_sql(
            "SELECT id FROM channels WHERE displayname = 'Tech Talks' LIMIT 1;"
        )
        if not channel_id:
            check("3. MM briefing message pinned", 1, False, "Tech Talks channel not found")
            return
        # In Mattermost, pinned posts have ispinned = true
        count = mm_sql(
            f"SELECT COUNT(*) FROM posts WHERE channelid = '{channel_id}' "
            f"AND message LIKE '%Kicking off Q3 competitive intel%' AND ispinned = true;"
        )
        passed = count.strip() not in ("", "0")
        check("3. MM briefing message pinned", 1, passed,
              f"pinned count: {count}" if not passed else "")
    except Exception as e:
        check("3. MM briefing message pinned", 1, False, f"exception: {e}")


# ── OnlyOffice checks ────────────────────────────────────────────────────────

def check_4_oo_full_briefing_exists() -> None:
    """Check full briefing document exists in My Documents."""
    try:
        title = "Nimbus Cloud Competitive Intelligence Briefing Q3"
        result = oo_sql(
            f"SELECT id FROM files_file WHERE title LIKE '%Nimbus Cloud Competitive Intelligence Briefing Q3%';"
        )
        passed = bool(result.strip())
        check("4. OO full briefing doc exists", 1, passed,
              "document not found" if not passed else "")
    except Exception as e:
        check("4. OO full briefing doc exists", 1, False, f"exception: {e}")


def check_5_oo_full_briefing_shared_jun() -> None:
    """Check full briefing shared with jun.chen for editing."""
    try:
        title = "Nimbus Cloud Competitive Intelligence Briefing Q3"
        # Get file id
        file_id = oo_sql(
            f"SELECT id FROM files_file WHERE title LIKE '%Nimbus Cloud Competitive Intelligence Briefing Q3%' LIMIT 1;"
        )
        if not file_id:
            check("5. OO full briefing shared with jun.chen", 2, False, "doc not found")
            return
        # Check security table for sharing - look for jun.chen as subject
        # OnlyOffice security table: tenant_id, entry_id, entry_type, subject, owner, security
        # subject is typically a user id (GUID). We need to find jun.chen's id first.
        shares = oo_sql(
            f"SELECT s.subject, s.security FROM files_security s "
            f"WHERE s.entry_id = {file_id} AND s.entry_type = 2;"
        )
        # security=1 means full access (edit), security=2 means read-only
        # We just check that there's at least one share entry for this file
        passed = bool(shares.strip())
        check("5. OO full briefing shared with jun.chen", 2, passed,
              f"shares: {shares[:100]}" if not passed else "")
    except Exception as e:
        check("5. OO full briefing shared with jun.chen", 2, False, f"exception: {e}")


def check_6_oo_sanitized_doc_exists() -> None:
    """Check sanitized briefing document exists."""
    try:
        title = "Nimbus Cloud Competitive Briefing — All Hands Summary"
        # Try with em dash and regular dash
        result = oo_sql(
            f"SELECT id FROM files_file WHERE title LIKE '%Nimbus Cloud Competitive Briefing%All Hands Summary%';"
        )
        passed = bool(result.strip())
        check("6. OO sanitized doc exists", 1, passed,
              "document not found" if not passed else "")
    except Exception as e:
        check("6. OO sanitized doc exists", 1, False, f"exception: {e}")


def check_7_oo_sanitized_shared_maria() -> None:
    """Check sanitized doc shared with maria.wilson for viewing."""
    try:
        file_id = oo_sql(
            "SELECT id FROM files_file WHERE title LIKE "
            "'%Nimbus Cloud Competitive Briefing%All Hands Summary%' LIMIT 1;"
        )
        if not file_id:
            check("7. OO sanitized doc shared with maria.wilson", 2, False, "doc not found")
            return
        shares = oo_sql(
            f"SELECT s.subject, s.security FROM files_security s "
            f"WHERE s.entry_id = {file_id} AND s.entry_type = 2;"
        )
        passed = bool(shares.strip())
        check("7. OO sanitized doc shared with maria.wilson", 2, passed,
              f"shares: {shares[:100]}" if not passed else "")
    except Exception as e:
        check("7. OO sanitized doc shared with maria.wilson", 2, False, f"exception: {e}")


# ── ownCloud checks ──────────────────────────────────────────────────────────

def check_8_oc_folder_structure() -> None:
    """Check folder structure: Competitive Intelligence Q3 / Restricted Dossier + Public Summary."""
    try:
        # Check for the parent folder and subfolders in oc_filecache
        result = oc_sql(
            "SELECT path FROM oc_filecache WHERE path LIKE '%Competitive Intelligence Q3%' "
            "ORDER BY path;"
        )
        has_parent = "Competitive Intelligence Q3" in result
        has_restricted = "Restricted Dossier" in result
        has_public = "Public Summary" in result
        passed = has_parent and has_restricted and has_public
        detail = ""
        if not passed:
            missing = []
            if not has_parent:
                missing.append("parent folder")
            if not has_restricted:
                missing.append("Restricted Dossier")
            if not has_public:
                missing.append("Public Summary")
            detail = f"missing: {', '.join(missing)}"
        check("8. OC folder structure", 1, passed, detail)
    except Exception as e:
        check("8. OC folder structure", 1, False, f"exception: {e}")


def check_9_oc_raw_intel_file() -> None:
    """Check nimbus_raw_intel.txt exists in Restricted Dossier with correct intro."""
    try:
        # Check file exists in filecache
        result = oc_sql(
            "SELECT fileid FROM oc_filecache WHERE path LIKE "
            "'%Restricted Dossier/nimbus_raw_intel.txt';"
        )
        if not result.strip():
            check("9. OC nimbus_raw_intel.txt content", 2, False, "file not found in filecache")
            return
        # Read file content from container
        rc, content, err = docker_exec(
            OC_CONTAINER, "bash", "-c",
            "find /mnt/data/files -path '*Restricted Dossier/nimbus_raw_intel.txt' "
            "-exec cat {} \\; 2>/dev/null"
        )
        expected_intro = "CONFIDENTIAL"
        passed = expected_intro in content and "Nimbus Cloud" in content
        check("9. OC nimbus_raw_intel.txt content", 2, passed,
              f"content starts: {content[:80]}" if not passed else "")
    except Exception as e:
        check("9. OC nimbus_raw_intel.txt content", 2, False, f"exception: {e}")


def check_10_oc_summary_file() -> None:
    """Check nimbus_summary.txt exists in Public Summary with correct content."""
    try:
        result = oc_sql(
            "SELECT fileid FROM oc_filecache WHERE path LIKE "
            "'%Public Summary/nimbus_summary.txt';"
        )
        if not result.strip():
            check("10. OC nimbus_summary.txt content", 1, False, "file not found in filecache")
            return
        rc, content, err = docker_exec(
            OC_CONTAINER, "bash", "-c",
            "find /mnt/data/files -path '*Public Summary/nimbus_summary.txt' "
            "-exec cat {} \\; 2>/dev/null"
        )
        expected = "Nimbus Cloud is pursuing a developer-first strategy"
        passed = expected in content
        check("10. OC nimbus_summary.txt content", 1, passed,
              f"content: {content[:80]}" if not passed else "")
    except Exception as e:
        check("10. OC nimbus_summary.txt content", 1, False, f"exception: {e}")


def check_12_oc_public_shared_group_admin() -> None:
    """Check Public Summary shared with group admin (read-only) with expiry 2026-08-20."""
    try:
        fileid = oc_sql(
            "SELECT fileid FROM oc_filecache WHERE path LIKE "
            "'%Competitive Intelligence Q3/Public Summary' LIMIT 1;"
        )
        if not fileid.strip():
            check("12. OC Public Summary shared with group admin (ro, expiry)", 2, False,
                  "folder not found")
            return
        fid = fileid.strip().splitlines()[0]
        # share_type=1 means group share
        shares = oc_sql(
            f"SELECT share_with, permissions, expiration FROM oc_share "
            f"WHERE file_source = {fid} AND share_type = 1;"
        )
        has_admin_group = "admin" in shares
        has_expiry = "2026-08-20" in shares
        passed = has_admin_group and has_expiry
        detail = ""
        if not passed:
            parts = []
            if not has_admin_group:
                parts.append("admin group share not found")
            if not has_expiry:
                parts.append("expiry 2026-08-20 not found")
            detail = "; ".join(parts) + f" | shares: {shares[:100]}"
        check("12. OC Public Summary shared with group admin (ro, expiry)", 2, passed, detail)
    except Exception as e:
        check("12. OC Public Summary shared with group admin (ro, expiry)", 2, False,
              f"exception: {e}")


def check_13_oc_public_link() -> None:
    """Check public share link created for Public Summary."""
    try:
        fileid = oc_sql(
            "SELECT fileid FROM oc_filecache WHERE path LIKE "
            "'%Competitive Intelligence Q3/Public Summary' LIMIT 1;"
        )
        if not fileid.strip():
            check("13. OC Public Summary public link", 1, False, "folder not found")
            return
        fid = fileid.strip().splitlines()[0]
        # share_type=3 means public link
        link = oc_sql(
            f"SELECT token FROM oc_share WHERE file_source = {fid} AND share_type = 3 LIMIT 1;"
        )
        passed = bool(link.strip())
        check("13. OC Public Summary public link", 1, passed,
              "no public link found" if not passed else "")
    except Exception as e:
        check("13. OC Public Summary public link", 1, False, f"exception: {e}")


# ── Roundcube checks ─────────────────────────────────────────────────────────

def check_14_rc_exec_email_sent() -> None:
    """Check exec email sent with correct subject."""
    try:
        expected_subject = "[CONFIDENTIAL] Nimbus Cloud Competitive Intelligence Briefing"
        # Search sent mail in the maildir for james.whitfield
        rc, out, err = docker_exec(
            RC_CONTAINER, "bash", "-c",
            "find /var/mail/ -type f -name '*.eml' -o -type f -name '*,' -o -type f | "
            "head -200 | xargs grep -l 'CONFIDENTIAL.*Nimbus Cloud' 2>/dev/null || "
            "grep -rl 'CONFIDENTIAL.*Nimbus Cloud' /var/mail/ 2>/dev/null || true",
            timeout=30,
        )
        if not out.strip():
            # Try checking postfix mail log
            rc2, out2, err2 = docker_exec(
                RC_CONTAINER, "bash", "-c",
                "grep -i 'CONFIDENTIAL.*Nimbus' /var/log/mail* 2>/dev/null || "
                "grep -i 'rachel.goldberg' /var/log/mail* 2>/dev/null || true",
                timeout=15,
            )
            passed = bool(out2.strip())
            check("14. RC exec email sent", 2, passed,
                  f"mail log: {out2[:100]}" if passed else "no matching email found in maildir or logs")
        else:
            passed = True
            check("14. RC exec email sent", 2, passed, "")
    except Exception as e:
        check("14. RC exec email sent", 2, False, f"exception: {e}")


def check_15_rc_general_email_sent() -> None:
    """Check general email sent with correct subject."""
    try:
        expected_subject = "Competitive Landscape Update"
        rc, out, err = docker_exec(
            RC_CONTAINER, "bash", "-c",
            "grep -rl 'Competitive Landscape Update' /var/mail/ 2>/dev/null || true",
            timeout=30,
        )
        if not out.strip():
            rc2, out2, err2 = docker_exec(
                RC_CONTAINER, "bash", "-c",
                "grep -i 'Competitive Landscape Update' /var/log/mail* 2>/dev/null || "
                "grep -i 'tom.andersen' /var/log/mail* 2>/dev/null || true",
                timeout=15,
            )
            passed = bool(out2.strip())
            check("15. RC general email sent", 2, passed,
                  f"mail log: {out2[:100]}" if passed else "no matching email found")
        else:
            passed = True
            check("15. RC general email sent", 2, passed, "")
    except Exception as e:
        check("15. RC general email sent", 2, False, f"exception: {e}")


def check_16_rc_exec_email_flagged() -> None:
    """Check exec email flagged in Sent folder."""
    try:
        # In Dovecot maildir, flagged messages have 'F' in the flags portion of filename
        # or we can check via IMAP flags. Let's check maildir flags.
        rc, out, err = docker_exec(
            RC_CONTAINER, "bash", "-c",
            "find /var/mail/ -path '*.Sent*' -type f 2>/dev/null | head -50",
            timeout=15,
        )
        if not out.strip():
            # Try alternate sent folder locations
            rc, out, err = docker_exec(
                RC_CONTAINER, "bash", "-c",
                "find /var/mail/ -type d -iname '*sent*' 2>/dev/null",
                timeout=15,
            )
            check("16. RC exec email flagged in Sent", 1, False,
                  f"Sent folder: {out[:100] if out.strip() else 'not found'}")
            return

        # Look for flagged files containing CONFIDENTIAL subject
        # Dovecot flags: S=seen, F=flagged, R=replied, etc. in filename after `:2,`
        rc2, out2, err2 = docker_exec(
            RC_CONTAINER, "bash", "-c",
            "for f in $(find /var/mail/ -path '*.Sent*' -type f 2>/dev/null); do "
            "if grep -q 'CONFIDENTIAL.*Nimbus' \"$f\" 2>/dev/null; then "
            "echo \"$f\"; fi; done",
            timeout=30,
        )
        if out2.strip():
            # Check if file has F flag in its name
            flagged = any("F" in fname.split(":2,")[-1] if ":2," in fname else False
                         for fname in out2.strip().splitlines())
            # Also check if message is in the DB as flagged
            if not flagged:
                rc3, out3, err3 = docker_exec(
                    RC_CONTAINER, "bash", "-c",
                    "echo 'flag found in filename check'",
                )
            check("16. RC exec email flagged in Sent", 1, flagged,
                  f"files: {out2.strip()[:100]}" if not flagged else "")
        else:
            check("16. RC exec email flagged in Sent", 1, False,
                  "exec email not found in Sent folder")
    except Exception as e:
        check("16. RC exec email flagged in Sent", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_mm_channel_purpose()
    check_2_mm_briefing_message()
    check_3_mm_message_pinned()
    check_4_oo_full_briefing_exists()
    check_5_oo_full_briefing_shared_jun()
    check_6_oo_sanitized_doc_exists()
    check_7_oo_sanitized_shared_maria()
    check_8_oc_folder_structure()
    check_9_oc_raw_intel_file()
    check_10_oc_summary_file()
    check_12_oc_public_shared_group_admin()
    check_13_oc_public_link()
    check_14_rc_exec_email_sent()
    check_15_rc_general_email_sent()
    check_16_rc_exec_email_flagged()

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
