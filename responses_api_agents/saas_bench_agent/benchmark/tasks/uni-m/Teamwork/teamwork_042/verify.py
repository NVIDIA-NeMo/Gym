# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_042/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-042-I2: Onboard Freelance Contractor Diego Martinez

Checks: 15 weighted checks across owncloud, onlyoffice, mattermost, roundcubemail.
Strategy: docker exec DB (ownCloud MariaDB, Mattermost Postgres, Roundcube MariaDB),
          REST API (OnlyOffice DocSpace, ownCloud WebDAV for file content).

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
import json
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

OC_PORT = os.environ.get("OWNCLOUD_PORT")
OC_CONTAINER = os.environ.get("OWNCLOUD_CONTAINER")
OC_DB_CONTAINER = os.environ.get("OWNCLOUD_DB_CONTAINER")

OO_PORT = os.environ.get("ONLYOFFICE_PORT")
OO_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
OO_DB_CONTAINER = os.environ.get("ONLYOFFICE_DB_CONTAINER")

MM_PORT = os.environ.get("MATTERMOST_PORT")
MM_CONTAINER = os.environ.get("MATTERMOST_CONTAINER")
MM_DB_CONTAINER = os.environ.get("MATTERMOST_DB_CONTAINER")

RC_PORT = os.environ.get("ROUNDCUBEMAIL_PORT")
RC_CONTAINER = os.environ.get("ROUNDCUBEMAIL_CONTAINER")
RC_DB_CONTAINER = os.environ.get("ROUNDCUBEMAIL_DB_CONTAINER")

_required = {
    "OWNCLOUD_PORT": OC_PORT, "OWNCLOUD_CONTAINER": OC_CONTAINER,
    "OWNCLOUD_DB_CONTAINER": OC_DB_CONTAINER,
    "ONLYOFFICE_PORT": OO_PORT, "ONLYOFFICE_CONTAINER": OO_CONTAINER,
    "ONLYOFFICE_DB_CONTAINER": OO_DB_CONTAINER,
    "MATTERMOST_PORT": MM_PORT, "MATTERMOST_CONTAINER": MM_CONTAINER,
    "MATTERMOST_DB_CONTAINER": MM_DB_CONTAINER,
    "ROUNDCUBEMAIL_PORT": RC_PORT, "ROUNDCUBEMAIL_CONTAINER": RC_CONTAINER,
    "ROUNDCUBEMAIL_DB_CONTAINER": RC_DB_CONTAINER,
}
for var, val in _required.items():
    if not val:
        print(f"FATAL: {var} not set", file=sys.stderr)
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


def oc_db(sql: str) -> str:
    """Query ownCloud MariaDB."""
    _, out, _ = docker_exec(
        OC_DB_CONTAINER,
        "mysql", "-u", "owncloud", "-powncloud", "-D", "owncloud",
        "--default-character-set=utf8mb4", "-N", "-e", sql,
    )
    return out.strip()


def mm_db(sql: str) -> str:
    """Query Mattermost Postgres."""
    r = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=mmuser_password",
         MM_DB_CONTAINER, "psql", "-U", "mmuser", "-d", "mattermost",
         "-t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=15,
    )
    return r.stdout.strip()


def rc_db(sql: str) -> str:
    """Query Roundcube MariaDB."""
    _, out, _ = docker_exec(
        RC_DB_CONTAINER,
        "mysql", "-u", "roundcube", "-proundcube123", "-D", "roundcubemail",
        "--default-character-set=utf8mb4", "-N", "-e", sql,
    )
    return out.strip()


def oo_api_auth() -> tuple[str, dict]:
    """Authenticate to OnlyOffice DocSpace, return (base_url, headers)."""
    base = f"http://{HOST}:{OO_PORT}"
    resp = requests.post(
        f"{base}/api/2.0/authentication",
        json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
        timeout=15,
    )
    data = resp.json()
    token = data.get("response", {}).get("token", "")
    return base, {"Authorization": f"Bearer {token}"}


# ── ownCloud Checks ──────────────────────────────────────────────────────────

def check_1_oc_user() -> None:
    """User diego.martinez exists with email and in group admin."""
    try:
        user_row = oc_db("SELECT uid FROM oc_users WHERE uid = 'diego.martinez';")
        user_exists = "diego.martinez" in user_row

        # Email stored in oc_accounts or oc_preferences
        email_row = oc_db(
            "SELECT email FROM oc_accounts WHERE uid = 'diego.martinez';"
        )
        if not email_row:
            email_row = oc_db(
                "SELECT configvalue FROM oc_preferences "
                "WHERE userid='diego.martinez' AND appid='settings' AND configkey='email';"
            )
        email_ok = "diego.martinez@contractors.local" in (email_row or "")

        group_row = oc_db(
            "SELECT gid FROM oc_group_user "
            "WHERE uid='diego.martinez' AND gid='admin';"
        )
        group_ok = "admin" in (group_row or "")

        passed = user_exists and email_ok and group_ok
        detail = (f"user={'found' if user_exists else 'missing'}, "
                  f"email={'ok' if email_ok else 'missing'}, "
                  f"group={'ok' if group_ok else 'missing'}")
        check("1. ownCloud user diego.martinez", 2, passed, detail)
    except Exception as e:
        check("1. ownCloud user diego.martinez", 2, False, f"exception: {e}")


def check_2_oc_quota() -> None:
    """User storage quota set to 10 GB."""
    try:
        quota = oc_db(
            "SELECT configvalue FROM oc_preferences "
            "WHERE userid='diego.martinez' AND appid='files' AND configkey='quota';"
        )
        passed = False
        if quota:
            q = quota.strip().lower()
            if "10 gb" in q or "10gb" in q:
                passed = True
            try:
                if int(q) == 10737418240:  # 10 * 1024^3
                    passed = True
            except ValueError:
                pass
        check("2. ownCloud user quota 10 GB", 1, passed, f"quota={quota!r}")
    except Exception as e:
        check("2. ownCloud user quota 10 GB", 1, False, f"exception: {e}")


def check_3_oc_folders() -> None:
    """Diego_Martinez_Workspace with Deliverables and Briefs subfolders."""
    try:
        rows = oc_db(
            "SELECT f.path FROM oc_filecache f "
            "WHERE f.path IN ("
            "  'files/Diego_Martinez_Workspace',"
            "  'files/Diego_Martinez_Workspace/Deliverables',"
            "  'files/Diego_Martinez_Workspace/Briefs'"
            ");"
        )
        found = set(r.strip() for r in rows.split('\n') if r.strip()) if rows else set()
        ws = "files/Diego_Martinez_Workspace" in found
        dl = "files/Diego_Martinez_Workspace/Deliverables" in found
        br = "files/Diego_Martinez_Workspace/Briefs" in found

        passed = ws and dl and br
        detail = (f"workspace={'ok' if ws else 'missing'}, "
                  f"deliverables={'ok' if dl else 'missing'}, "
                  f"briefs={'ok' if br else 'missing'}")
        check("3. ownCloud folder structure", 2, passed, detail)
    except Exception as e:
        check("3. ownCloud folder structure", 2, False, f"exception: {e}")


def check_4_oc_brief_content() -> None:
    """Mobile_App_Brief.txt has correct project brief content."""
    try:
        expected = (
            "Project: Mobile App MVP. Goal: Ship a cross-platform iOS/Android MVP "
            "within 12 weeks using React Native. Primary deliverables include "
            "architecture document, authentication module, core feature screens, "
            "and production-ready builds for both stores. Review cadence: bi-weekly "
            "on Tuesdays."
        )
        resp = requests.get(
            f"http://{HOST}:{OC_PORT}/remote.php/dav/files/admin/"
            "Diego_Martinez_Workspace/Briefs/Mobile_App_Brief.txt",
            auth=("admin", "admin"),
            timeout=15,
        )
        if resp.status_code == 200:
            content = resp.text.strip()
            passed = expected in content
            detail = "content matches" if passed else f"got {content[:100]!r}"
        else:
            passed = False
            detail = f"HTTP {resp.status_code}"
        check("4. ownCloud Mobile_App_Brief.txt content", 2, passed, detail)
    except Exception as e:
        check("4. ownCloud Mobile_App_Brief.txt content", 2, False, f"exception: {e}")


def check_5_oc_public_link() -> None:
    """Public link for Briefs folder: read-only, expiry 2026-09-30."""
    try:
        rows = oc_db(
            "SELECT s.permissions, s.expiration "
            "FROM oc_share s "
            "JOIN oc_filecache f ON s.file_source = f.fileid "
            "WHERE s.share_type = 3 "
            "AND f.path LIKE '%Diego_Martinez_Workspace/Briefs';"
        )
        if rows:
            parts = rows.split('\t')
            perms = int(parts[0].strip()) if parts else -1
            expiry = parts[1].strip() if len(parts) > 1 else ""
            read_only = perms == 1 or (perms & 1 and not (perms & 2))
            expiry_ok = "2026-09-30" in expiry
            passed = read_only and expiry_ok
            detail = f"permissions={perms}, expiry={expiry!r}"
        else:
            passed = False
            detail = "no public link found for Briefs"
        check("5. ownCloud public link for Briefs", 2, passed, detail)
    except Exception as e:
        check("5. ownCloud public link for Briefs", 2, False, f"exception: {e}")



# ── OnlyOffice Checks ────────────────────────────────────────────────────────

def check_7_oo_document() -> dict | None:
    """Engagement_Letter_Diego_Martinez exists in My Documents."""
    try:
        base, headers = oo_api_auth()
        resp = requests.get(f"{base}/api/2.0/files/@my", headers=headers, timeout=15)
        data = resp.json()
        files = data.get("response", {}).get("files", [])
        if isinstance(data.get("response"), list):
            files = [f for f in data["response"] if f.get("fileType") is not None or "title" in f]

        found = None
        for f in files:
            title = f.get("title", "")
            if "Engagement_Letter_Diego_Martinez" in title:
                found = f
                break

        passed = found is not None
        detail = f"title={found['title']!r}" if found else "not found in My Documents"
        check("7. OnlyOffice document exists", 2, passed, detail)
        return found
    except Exception as e:
        check("7. OnlyOffice document exists", 2, False, f"exception: {e}")
        return None


def check_8_oo_shared(doc_id=None) -> None:
    """Document shared with jun.chen for editing."""
    try:
        if doc_id is None:
            check("8. OnlyOffice shared with jun.chen", 2, False, "no doc_id from check 7")
            return
        base, headers = oo_api_auth()
        resp = requests.get(
            f"{base}/api/2.0/files/file/{doc_id}/share",
            headers=headers, timeout=15,
        )
        data = resp.json()
        shares = data.get("response", [])

        found = False
        for s in shares:
            user = s.get("sharedTo", {})
            uname = (user.get("userName", "") or user.get("email", "")).lower()
            display = (user.get("displayName", "") or "").lower()
            access = s.get("access", -1)
            if "jun.chen" in uname or "jun.chen" in display or "jun" in uname:
                # access 1 = editing/collaborator, 2 = read-only
                if access in (0, 1):
                    found = True
                    break

        check("8. OnlyOffice shared with jun.chen", 2, found,
              "share found" if found else "jun.chen not in share list")
    except Exception as e:
        check("8. OnlyOffice shared with jun.chen", 2, False, f"exception: {e}")


def check_9_oo_favorite(doc_id=None) -> None:
    """Document marked as favorite."""
    try:
        if doc_id is None:
            check("9. OnlyOffice favorite", 1, False, "no doc_id from check 7")
            return
        base, headers = oo_api_auth()
        resp = requests.get(f"{base}/api/2.0/files/@favorites", headers=headers, timeout=15)
        data = resp.json()

        # Handle different response structures
        response = data.get("response", {})
        if isinstance(response, dict):
            files = response.get("files", [])
        elif isinstance(response, list):
            files = response
        else:
            files = []

        found = any(
            "Engagement_Letter_Diego_Martinez" in (f.get("title", "") if isinstance(f, dict) else "")
            for f in files
        )
        check("9. OnlyOffice favorite", 1, found,
              "in favorites" if found else "not in favorites")
    except Exception as e:
        check("9. OnlyOffice favorite", 1, False, f"exception: {e}")


# ── Mattermost Checks ────────────────────────────────────────────────────────

def check_10_mm_karrie_member() -> None:
    """karrie is a member of Brand Design channel."""
    try:
        row = mm_db(
            "SELECT u.username FROM channelmembers cm "
            "JOIN channels c ON cm.channelid = c.id "
            "JOIN users u ON cm.userid = u.id "
            "WHERE c.displayname = 'Brand Design' AND u.username = 'karrie';"
        )
        passed = "karrie" in (row or "")
        check("10. Mattermost karrie in Brand Design", 1, passed,
              "member" if passed else "not found")
    except Exception as e:
        check("10. Mattermost karrie in Brand Design", 1, False, f"exception: {e}")


def check_11_mm_intro_message() -> None:
    """Intro message posted in Brand Design channel."""
    try:
        row = mm_db(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.displayname = 'Brand Design' "
            "AND p.message LIKE '%please welcome Diego Martinez%' "
            "AND p.deleteat = 0 LIMIT 1;"
        )
        passed = "please welcome Diego Martinez" in (row or "")
        check("11. Mattermost intro message", 2, passed,
              "found" if passed else "not found")
    except Exception as e:
        check("11. Mattermost intro message", 2, False, f"exception: {e}")


def check_12_mm_dm_karrie() -> None:
    """DM to karrie with workspace access details."""
    try:
        row = mm_db(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.type = 'D' "
            "AND p.message LIKE '%Welcome aboard%' "
            "AND p.message LIKE '%Diego_Martinez_Workspace%' "
            "AND p.deleteat = 0 LIMIT 1;"
        )
        passed = bool(row and row.strip())
        check("12. Mattermost DM to karrie", 2, passed,
              "found" if passed else "DM not found")
    except Exception as e:
        check("12. Mattermost DM to karrie", 2, False, f"exception: {e}")


def check_13_mm_channel_header() -> None:
    """Brand Design channel header updated with contractor onboarding info."""
    try:
        row = mm_db(
            "SELECT header FROM channels WHERE displayname = 'Brand Design';"
        )
        expected = (
            "Brand assets, style guides, and design feedback. "
            "Contractor onboarding active through 2026-09-30"
        )
        passed = expected.lower() in (row or "").lower()
        check("13. Mattermost channel header", 1, passed,
              "matches" if passed else f"got {(row or '')[:100]!r}")
    except Exception as e:
        check("13. Mattermost channel header", 1, False, f"exception: {e}")


# ── Roundcube Checks ─────────────────────────────────────────────────────────

def check_14_rc_identity() -> None:
    """Identity with Sarah O'Brien — Co-Founder, correct email, org, signature."""
    try:
        rows = rc_db(
            "SELECT name, email, organization, signature "
            "FROM identities "
            "WHERE email = 'sarah.obrien@mail.local';"
        )
        if rows:
            parts = rows.split('\t')
            name = parts[0] if len(parts) > 0 else ""
            org = parts[2] if len(parts) > 2 else ""
            sig = parts[3] if len(parts) > 3 else ""

            name_ok = "Sarah" in name and "Co-Founder" in name
            org_ok = "Acme Ventures" in org
            sig_ok = "Sarah" in sig and "Co-Founder" in sig

            passed = name_ok and org_ok and sig_ok
            detail = (f"name={'ok' if name_ok else repr(name)}, "
                      f"org={'ok' if org_ok else repr(org)}, "
                      f"sig={'ok' if sig_ok else 'wrong/missing'}")
        else:
            passed = False
            detail = "identity not found for sarah.obrien@mail.local"
        check("14. Roundcube identity", 2, passed, detail)
    except Exception as e:
        check("14. Roundcube identity", 2, False, f"exception: {e}")


def check_15_rc_email_sent() -> None:
    """Email sent to diego.martinez@contractors.local with onboarding subject."""
    try:
        expected_subject = "Welcome to Acme Ventures"
        # Search maildir in roundcube container
        rc_code, out, _ = docker_exec(
            RC_CONTAINER,
            "bash", "-c",
            f"grep -rl 'Subject:.*{expected_subject}' /var/mail/ 2>/dev/null || "
            f"grep -rl 'Subject:.*{expected_subject}' /var/vmail/ 2>/dev/null || "
            f"grep -rl 'Subject:.*{expected_subject}' /home/vmail/ 2>/dev/null || true",
            timeout=20,
        )
        if out and out.strip():
            mail_file = out.strip().split('\n')[0].strip()
            _, content, _ = docker_exec(RC_CONTAINER, "cat", mail_file, timeout=15)
            has_recipient = "diego.martinez@contractors.local" in content
            passed = has_recipient
            detail = "email found with correct recipient" if passed else "subject found, wrong recipient"
        else:
            # Fallback: check mail log
            _, log_out, _ = docker_exec(
                RC_CONTAINER,
                "bash", "-c",
                "grep -i 'diego.martinez' /var/log/mail* 2>/dev/null || "
                "grep -i 'diego.martinez' /var/log/syslog 2>/dev/null || true",
                timeout=15,
            )
            if "diego.martinez" in (log_out or ""):
                passed = True
                detail = "found in mail logs"
            else:
                passed = False
                detail = "email not found in maildir or logs"
        check("15. Roundcube email sent", 2, passed, detail)
    except Exception as e:
        check("15. Roundcube email sent", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ownCloud (6 checks)
    check_1_oc_user()
    check_2_oc_quota()
    check_3_oc_folders()
    check_4_oc_brief_content()
    check_5_oc_public_link()

    # OnlyOffice (3 checks)
    doc = check_7_oo_document()
    doc_id = doc.get("id") if doc else None
    check_8_oo_shared(doc_id)
    check_9_oo_favorite(doc_id)

    # Mattermost (4 checks)
    check_10_mm_karrie_member()
    check_11_mm_intro_message()
    check_12_mm_dm_karrie()
    check_13_mm_channel_header()

    # Roundcube (2 checks)
    check_14_rc_identity()
    check_15_rc_email_sent()

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
