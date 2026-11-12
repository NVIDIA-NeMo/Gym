# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/uni-m/Teamwork/teamwork_009/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for Teamwork-009-I5: Coordinate Media Response to Data Privacy Inquiry

Checks: 13 weighted checks across roundcubemail, mattermost, onlyoffice.
Strategy: docker exec (maildir) for roundcubemail, docker exec (DB) for mattermost,
          docker exec (DB) + API for onlyoffice.

Required env vars:
  SERVER_HOSTNAME,
  ROUNDCUBEMAIL_PORT, ROUNDCUBEMAIL_CONTAINER, ROUNDCUBEMAIL_DB_CONTAINER,
  MATTERMOST_PORT, MATTERMOST_CONTAINER, MATTERMOST_DB_CONTAINER,
  ONLYOFFICE_PORT, ONLYOFFICE_CONTAINER, ONLYOFFICE_DB_CONTAINER
"""

import os
import sys
import subprocess
import json
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

ROUNDCUBEMAIL_PORT = os.environ.get("ROUNDCUBEMAIL_PORT")
ROUNDCUBEMAIL_CONTAINER = os.environ.get("ROUNDCUBEMAIL_CONTAINER")
ROUNDCUBEMAIL_DB_CONTAINER = os.environ.get("ROUNDCUBEMAIL_DB_CONTAINER")

MATTERMOST_PORT = os.environ.get("MATTERMOST_PORT")
MATTERMOST_CONTAINER = os.environ.get("MATTERMOST_CONTAINER")
MATTERMOST_DB_CONTAINER = os.environ.get("MATTERMOST_DB_CONTAINER")

ONLYOFFICE_PORT = os.environ.get("ONLYOFFICE_PORT")
ONLYOFFICE_CONTAINER = os.environ.get("ONLYOFFICE_CONTAINER")
ONLYOFFICE_DB_CONTAINER = os.environ.get("ONLYOFFICE_DB_CONTAINER")

_REQUIRED = [
    ("ROUNDCUBEMAIL_PORT", ROUNDCUBEMAIL_PORT),
    ("ROUNDCUBEMAIL_CONTAINER", ROUNDCUBEMAIL_CONTAINER),
    ("ROUNDCUBEMAIL_DB_CONTAINER", ROUNDCUBEMAIL_DB_CONTAINER),
    ("MATTERMOST_PORT", MATTERMOST_PORT),
    ("MATTERMOST_CONTAINER", MATTERMOST_CONTAINER),
    ("MATTERMOST_DB_CONTAINER", MATTERMOST_DB_CONTAINER),
    ("ONLYOFFICE_PORT", ONLYOFFICE_PORT),
    ("ONLYOFFICE_CONTAINER", ONLYOFFICE_CONTAINER),
    ("ONLYOFFICE_DB_CONTAINER", ONLYOFFICE_DB_CONTAINER),
]
for _name, _val in _REQUIRED:
    if not _val:
        print(f"FATAL: {_name} not set", file=sys.stderr)
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


def mm_db_query(sql: str) -> str:
    """Query Mattermost PostgreSQL DB."""
    rc, stdout, stderr = docker_exec(
        MATTERMOST_DB_CONTAINER,
        "psql", "-U", "mmuser", "-d", "mattermost", "-t", "-A", "-c", sql,
    )
    return stdout.strip()


def oo_db_query(sql: str) -> str:
    """Query OnlyOffice MySQL DB."""
    rc, stdout, stderr = docker_exec(
        ONLYOFFICE_DB_CONTAINER,
        "mysql", "-u", "onlyoffice_user", "-ponlyoffice_pass", "-D", "onlyoffice",
        "-N", "-e", sql,
    )
    return stdout.strip()


def mail_grep(pattern: str, path: str = "/var/mail/", extra_flags: str = "") -> list[str]:
    """Grep for pattern in the maildir inside the Roundcube container. Returns matching file paths."""
    cmd = f"grep -rl {extra_flags} '{pattern}' {path} 2>/dev/null || true"
    rc, stdout, stderr = docker_exec(ROUNDCUBEMAIL_CONTAINER, "bash", "-c", cmd, timeout=20)
    return [f for f in stdout.strip().split("\n") if f]


def mail_cat(filepath: str) -> str:
    """Read a mail file from the Roundcube container."""
    rc, stdout, stderr = docker_exec(
        ROUNDCUBEMAIL_CONTAINER, "bash", "-c", f"cat '{filepath}' 2>/dev/null", timeout=15
    )
    return stdout


# ── Mattermost checks ────────────────────────────────────────────────────────

def check_6_mm_channel() -> None:
    """Private channel 'media-response-privacy-security' in Marketing & Growth with correct purpose."""
    try:
        result = mm_db_query(
            "SELECT c.type, c.purpose "
            "FROM channels c JOIN teams t ON c.teamid = t.id "
            "WHERE c.name = 'media-response-privacy-security' "
            "AND t.displayname = 'Marketing & Growth';"
        )

        if not result:
            check("6. MM private channel with correct purpose", 2, False, "channel not found")
            return

        parts = result.split("|")
        ch_type = parts[0].strip() if len(parts) > 0 else ""
        ch_purpose = parts[1].strip() if len(parts) > 1 else ""

        is_private = ch_type == "P"
        expected_purpose = "Coordinate response to Robert Singh media inquiry regarding customer data privacy and security posture."
        has_purpose = expected_purpose.lower() in ch_purpose.lower()

        passed = is_private and has_purpose
        details = []
        if not is_private:
            details.append(f"type={ch_type}, expected P")
        if not has_purpose:
            details.append(f"purpose mismatch: '{ch_purpose[:100]}'")
        check("6. MM private channel with correct purpose", 2, passed,
              "; ".join(details) if details else "OK")
    except Exception as e:
        check("6. MM private channel with correct purpose", 2, False, f"exception: {e}")


def check_7_mm_inquiry_brief() -> None:
    """Inquiry brief message posted in channel."""
    try:
        result = mm_db_query(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'media-response-privacy-security' "
            "AND p.message LIKE '%Journalist: Robert Singh%' "
            "AND p.deleteat = 0;"
        )

        has_brief = bool(result)
        has_content = False
        if has_brief:
            has_content = (
                "data protection controls" in result
                and "incident response readiness" in result
                and "executive availability" in result
            )

        passed = has_brief and has_content
        check("7. Inquiry brief message posted", 2, passed,
              "found with correct content" if passed else
              "not found" if not has_brief else "missing key content")
    except Exception as e:
        check("7. Inquiry brief message posted", 2, False, f"exception: {e}")


def check_8_mm_thread_reply() -> None:
    """Thread reply mentioning @tonda with review request text."""
    try:
        result = mm_db_query(
            "SELECT p.message FROM posts p "
            "JOIN channels c ON p.channelid = c.id "
            "WHERE c.name = 'media-response-privacy-security' "
            "AND p.message LIKE '%tonda%' "
            "AND p.rootid != '' "
            "AND p.deleteat = 0;"
        )

        has_reply = bool(result)
        has_content = False
        if has_reply:
            has_content = (
                "legal vetting" in result
                and "executive sign-off" in result
                and "24 hours" in result
            )

        passed = has_reply and has_content
        check("8. Thread reply with @tonda review request", 2, passed,
              "found with correct content" if passed else
              "not found" if not has_reply else "missing key content")
    except Exception as e:
        check("8. Thread reply with @tonda review request", 2, False, f"exception: {e}")


def check_9_mm_message_saved() -> None:
    """Inquiry brief message is saved/flagged in Mattermost."""
    try:
        result = mm_db_query(
            "SELECT pr.name FROM preferences pr "
            "WHERE pr.category = 'flagged_post' "
            "AND pr.name IN ("
            "  SELECT p.id FROM posts p "
            "  JOIN channels c ON p.channelid = c.id "
            "  WHERE c.name = 'media-response-privacy-security' "
            "  AND p.message LIKE '%Journalist: Robert Singh%' "
            "  AND p.deleteat = 0"
            ");"
        )

        passed = bool(result)
        check("9. Inquiry brief message saved/flagged", 1, passed,
              "flagged" if passed else "not flagged in preferences")
    except Exception as e:
        check("9. Inquiry brief message saved/flagged", 1, False, f"exception: {e}")


# ── OnlyOffice checks ────────────────────────────────────────────────────────

def _oo_auth_session() -> requests.Session | None:
    """Authenticate to OnlyOffice and return a session with token."""
    base_url = f"http://{HOST}:{ONLYOFFICE_PORT}"
    session = requests.Session()
    try:
        resp = session.post(
            f"{base_url}/api/2.0/authentication",
            json={"userName": "admin@onlyoffice.local", "password": "NewAdmin123!"},
            timeout=15,
        )
        if resp.status_code not in (200, 201):
            return None
        data = resp.json()
        token = data.get("response", {}).get("token", "")
        if not token:
            return None
        session.headers.update({"Authorization": f"Bearer {token}"})
        session.base_url = base_url  # type: ignore[attr-defined]
        return session
    except Exception:
        return None


def check_10_oo_document_exists() -> None:
    """Document 'Media Statement - Data Privacy and Security Inquiry' exists in OnlyOffice."""
    try:
        result = oo_db_query(
            "SELECT id, title FROM files "
            "WHERE title LIKE '%Media Statement%Data Privacy%Security Inquiry%' "
            "LIMIT 5;"
        )

        passed = bool(result) and "Media Statement" in result
        check("10. OnlyOffice document exists", 1, passed,
              f"found: {result[:200]}" if passed else "document not found in DB")
    except Exception as e:
        check("10. OnlyOffice document exists", 1, False, f"exception: {e}")


def check_11_oo_shared_with_user() -> None:
    """Document shared with amit.singh for editing."""
    try:
        # Use API to check sharing as DB join logic for OnlyOffice is complex
        session = _oo_auth_session()
        if not session:
            # Fallback to DB
            file_id = oo_db_query(
                "SELECT id FROM files "
                "WHERE title LIKE '%Media Statement%Data Privacy%Security Inquiry%' "
                "LIMIT 1;"
            ).strip()
            if not file_id:
                check("11. Document shared with amit.singh (edit)", 2, False, "document not found")
                return
            share_result = oo_db_query(
                f"SELECT s.security, s.subject FROM security s "
                f"WHERE s.entry_id = '{file_id}' AND s.entry_type = 2;"
            )
            check("11. Document shared with amit.singh (edit)", 2, bool(share_result),
                  f"shares: {share_result[:200]}" if share_result else "no shares found")
            return

        base_url = session.base_url  # type: ignore[attr-defined]
        # List My Documents
        resp = session.get(f"{base_url}/api/2.0/files/@my", timeout=15)
        if resp.status_code != 200:
            check("11. Document shared with amit.singh (edit)", 2, False,
                  f"API list failed: {resp.status_code}")
            return

        files_list = resp.json().get("response", {}).get("files", [])
        doc_id = None
        for f in files_list:
            title = f.get("title", "")
            if "Media Statement" in title and "Data Privacy" in title:
                doc_id = f.get("id")
                break

        if not doc_id:
            check("11. Document shared with amit.singh (edit)", 2, False,
                  "document not found via API")
            return

        # Check shares on the document
        share_resp = session.get(
            f"{base_url}/api/2.0/files/file/{doc_id}/share", timeout=15
        )
        if share_resp.status_code != 200:
            check("11. Document shared with amit.singh (edit)", 2, False,
                  f"share API failed: {share_resp.status_code}")
            return

        shares = share_resp.json().get("response", [])
        amit_shared = False
        amit_can_edit = False
        for s in shares:
            shared_to = s.get("sharedTo", {})
            username = shared_to.get("userName", "") or shared_to.get("displayName", "")
            if "amit" in username.lower() and "singh" in username.lower():
                amit_shared = True
                # access: 1 = read+write, 2 = read
                access = s.get("access", -1)
                if access == 1:
                    amit_can_edit = True

        passed = amit_shared and amit_can_edit
        details = []
        if not amit_shared:
            details.append("amit.singh not in share list")
        elif not amit_can_edit:
            details.append("amit.singh shared but not with edit access")
        check("11. Document shared with amit.singh (edit)", 2, passed,
              "; ".join(details) if details else "OK")
    except Exception as e:
        check("11. Document shared with amit.singh (edit)", 2, False, f"exception: {e}")


def check_12_oo_document_content() -> None:
    """Document contains Background, Key Messages, and Approved Quote sections."""
    try:
        session = _oo_auth_session()
        if not session:
            check("12. Document contains key content sections", 2, False, "auth failed")
            return

        base_url = session.base_url  # type: ignore[attr-defined]
        resp = session.get(f"{base_url}/api/2.0/files/@my", timeout=15)
        files_list = resp.json().get("response", {}).get("files", [])

        doc_id = None
        content_url = None
        for f in files_list:
            title = f.get("title", "")
            if "Media Statement" in title and "Data Privacy" in title:
                doc_id = f.get("id")
                content_url = f.get("viewUrl", "") or f.get("webUrl", "")
                break

        if not doc_id:
            check("12. Document contains key content sections", 2, False,
                  "document not found via API")
            return

        # Download file content and inspect
        dl_resp = session.get(
            f"{base_url}/api/2.0/files/file/{doc_id}/download", timeout=30,
            allow_redirects=True,
        )

        if dl_resp.status_code != 200:
            # Fallback: try to find and inspect the file on the filesystem
            rc, stdout, _ = docker_exec(
                ONLYOFFICE_CONTAINER, "bash", "-c",
                "find /var/www/onlyoffice/Data -name '*Media*Statement*' -o -name '*.docx' 2>/dev/null | head -20",
                timeout=15,
            )
            check("12. Document contains key content sections", 2, False,
                  f"download failed ({dl_resp.status_code}); fs files: {stdout.strip()[:200]}")
            return

        # The response is likely a docx file (ZIP). Write to temp and inspect XML.
        import tempfile
        import zipfile
        import io

        content_data = dl_resp.content
        found_sections = {"background": False, "key_messages": False, "approved_quote": False}

        try:
            zf = zipfile.ZipFile(io.BytesIO(content_data))
            for name in zf.namelist():
                if "document.xml" in name.lower() or "word/document" in name.lower():
                    xml_content = zf.read(name).decode("utf-8", errors="replace")
                    if "Stellar Tech reporter Robert Singh" in xml_content:
                        found_sections["background"] = True
                    if "foundational to our business" in xml_content:
                        found_sections["key_messages"] = True
                    if "Customer trust is earned" in xml_content:
                        found_sections["approved_quote"] = True
        except zipfile.BadZipFile:
            # Maybe it's plain text or HTML
            text = content_data.decode("utf-8", errors="replace")
            if "Stellar Tech reporter Robert Singh" in text:
                found_sections["background"] = True
            if "foundational to our business" in text:
                found_sections["key_messages"] = True
            if "Customer trust is earned" in text:
                found_sections["approved_quote"] = True

        all_found = all(found_sections.values())
        missing = [k for k, v in found_sections.items() if not v]
        check("12. Document contains key content sections", 2, all_found,
              "all sections found" if all_found else f"missing: {missing}")
    except Exception as e:
        check("12. Document contains key content sections", 2, False, f"exception: {e}")


def check_13_oo_track_changes() -> None:
    """Track changes is enabled on the document."""
    try:
        session = _oo_auth_session()
        if not session:
            check("13. Track changes enabled", 1, False, "auth failed")
            return

        base_url = session.base_url  # type: ignore[attr-defined]
        resp = session.get(f"{base_url}/api/2.0/files/@my", timeout=15)
        files_list = resp.json().get("response", {}).get("files", [])

        doc_id = None
        for f in files_list:
            title = f.get("title", "")
            if "Media Statement" in title and "Data Privacy" in title:
                doc_id = f.get("id")
                break

        if not doc_id:
            check("13. Track changes enabled", 1, False, "document not found")
            return

        # Download file and check for trackRevisions in settings XML
        dl_resp = session.get(
            f"{base_url}/api/2.0/files/file/{doc_id}/download", timeout=30,
            allow_redirects=True,
        )

        if dl_resp.status_code != 200:
            check("13. Track changes enabled", 1, False,
                  f"download failed: {dl_resp.status_code}")
            return

        import zipfile
        import io

        track_changes_on = False
        try:
            zf = zipfile.ZipFile(io.BytesIO(dl_resp.content))
            for name in zf.namelist():
                if "settings.xml" in name.lower():
                    xml_content = zf.read(name).decode("utf-8", errors="replace")
                    # <w:trackRevisions/> or <w:trackRevisions w:val="true"/>
                    if "trackRevisions" in xml_content:
                        # Check it's not explicitly set to false
                        if 'val="false"' not in xml_content and "val='false'" not in xml_content:
                            track_changes_on = True
        except zipfile.BadZipFile:
            pass

        check("13. Track changes enabled", 1, track_changes_on,
              "trackRevisions found in settings" if track_changes_on else "trackRevisions not found")
    except Exception as e:
        check("13. Track changes enabled", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_6_mm_channel()
    check_7_mm_inquiry_brief()
    check_8_mm_thread_reply()
    check_9_mm_message_saved()
    check_10_oo_document_exists()
    check_11_oo_shared_with_user()
    check_12_oo_document_content()
    check_13_oo_track_changes()

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
