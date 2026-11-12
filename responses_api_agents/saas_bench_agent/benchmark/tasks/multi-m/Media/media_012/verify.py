# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_012/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_012: Cross-media comparative analysis — The Valley of Fear vs A Game of Shadows

Checks: 11 weighted checks across watcharr, booklore, siyuan.
Strategy: host-side SQLite via docker cp (watcharr), docker exec MariaDB (booklore), REST API (siyuan), llm_judge (content quality).

Required env vars:
  SERVER_HOSTNAME, WATCHARR_PORT, WATCHARR_CONTAINER,
  BOOKLORE_PORT, BOOKLORE_CONTAINER,
  SIYUAN_PORT, SIYUAN_CONTAINER
"""

import os
import sys
import subprocess
import json
import re

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

WATCHARR_PORT = os.getenv("WATCHARR_PORT")
WATCHARR_CONTAINER = os.getenv("WATCHARR_CONTAINER")
BOOKLORE_PORT = os.getenv("BOOKLORE_PORT")
BOOKLORE_CONTAINER = os.getenv("BOOKLORE_CONTAINER")
SIYUAN_PORT = os.getenv("SIYUAN_PORT")
SIYUAN_CONTAINER = os.getenv("SIYUAN_CONTAINER")

_missing = []
for var in ["WATCHARR_PORT", "WATCHARR_CONTAINER",
            "BOOKLORE_PORT", "BOOKLORE_CONTAINER",
            "SIYUAN_PORT", "SIYUAN_CONTAINER"]:
    if not os.getenv(var):
        _missing.append(var)
if _missing:
    print(f"FATAL: {', '.join(_missing)} not set", file=sys.stderr)
    sys.exit(1)

BOOKLORE_DB_CONTAINER = os.getenv("BOOKLORE_DB_CONTAINER") or BOOKLORE_CONTAINER

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


# ── Watcharr helpers (SQLite via docker exec) ────────────────────────────────
_watcharr_db_path: str | None = None


def _find_watcharr_db() -> str:
    global _watcharr_db_path
    if _watcharr_db_path is not None:
        return _watcharr_db_path
    for candidate in ["/data/watcharr.db", "/data/database.sqlite", "/app/watcharr.db"]:
        rc, _, _ = docker_exec(WATCHARR_CONTAINER, "test", "-f", candidate, timeout=5)
        if rc == 0:
            _watcharr_db_path = candidate
            return _watcharr_db_path
    rc, stdout, _ = docker_exec(
        WATCHARR_CONTAINER, "find", "/", "-maxdepth", "4",
        "-name", "*.db", "-o", "-name", "*.sqlite",
        timeout=10,
    )
    for line in stdout.strip().splitlines():
        line = line.strip()
        if "watcharr" in line.lower() or "database" in line.lower():
            _watcharr_db_path = line
            return _watcharr_db_path
    _watcharr_db_path = "/data/watcharr.db"
    return _watcharr_db_path


def watcharr_sql(sql: str, timeout: int = 15) -> tuple[int, str, str]:
    """Query Watcharr's SQLite DB.

    The mw-watcharr image ships no sqlite3 CLI and the container has no
    outbound network to install one, so docker cp the DB (plus WAL sidecars)
    to a host temp dir and query it with Python's sqlite3 module. Output
    mimics `sqlite3 -separator "\t"`.
    """
    import shutil
    import sqlite3
    import tempfile

    db = _find_watcharr_db()
    tmpdir = tempfile.mkdtemp(prefix="watcharr_verify_")
    try:
        local_db = os.path.join(tmpdir, os.path.basename(db))
        r = subprocess.run(["docker", "cp", f"{WATCHARR_CONTAINER}:{db}", local_db],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return r.returncode, "", r.stderr
        for suffix in ("-wal", "-shm"):
            subprocess.run(["docker", "cp", f"{WATCHARR_CONTAINER}:{db}{suffix}", tmpdir],
                           capture_output=True, text=True, timeout=timeout)
        con = sqlite3.connect(local_db)
        try:
            rows = con.execute(sql).fetchall()
        finally:
            con.close()
        out = "\n".join("\t".join("" if v is None else str(v) for v in row) for row in rows)
        return 0, (out + "\n") if out else "", ""
    except Exception as e:
        return 1, "", str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Booklore helpers (MariaDB via docker exec) ──────────────────────────────
def mariadb_query(query: str, timeout: int = 15) -> str:
    r = subprocess.run(
        [
            "docker", "exec", BOOKLORE_DB_CONTAINER,
            "mariadb", "-u", "booklore",
            "-pChangeMe_BookLoreApp_2025!",
            "--default-character-set=utf8mb4",
            "-D", "booklore",
            "-N", "-B", "-e", query,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        r2 = subprocess.run(
            [
                "docker", "exec", BOOKLORE_DB_CONTAINER,
                "mysql", "-u", "booklore",
                "-pChangeMe_BookLoreApp_2025!",
                "--default-character-set=utf8mb4",
                "-D", "booklore",
                "-N", "-B", "-e", query,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        return r2.stdout.strip()
    return r.stdout.strip()


# ── SiYuan helpers (REST API) ───────────────────────────────────────────────
_siyuan_token: str | None = None


def _get_siyuan_token() -> str:
    global _siyuan_token
    if _siyuan_token is not None:
        return _siyuan_token
    rc, out, _ = docker_exec(
        SIYUAN_CONTAINER, "cat", "/siyuan/workspace/conf/conf.json", timeout=10,
    )
    if rc == 0 and out.strip():
        try:
            conf = json.loads(out)
            _siyuan_token = conf.get("api", {}).get("token", "")
        except (json.JSONDecodeError, AttributeError):
            _siyuan_token = ""
    else:
        _siyuan_token = ""
    return _siyuan_token


def siyuan_api(endpoint: str, payload: dict, timeout: int = 15) -> dict:
    url = f"http://{HOST}:{SIYUAN_PORT}{endpoint}"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = _get_siyuan_token()
    if token:
        headers["Authorization"] = f"Token {token}"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"SiYuan API error: {body.get('msg', 'unknown')}")
        return body.get("data", {})
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"SiYuan request failed: {e}")


def siyuan_sql(stmt: str) -> list:
    data = siyuan_api("/api/query/sql", {"stmt": stmt})
    if isinstance(data, list):
        return data
    return []


def siyuan_export_md(doc_id: str) -> str:
    """Export a document's markdown in TRUE document order.

    NOTE: the blocks table's `sort` column is grouped by block type (headings,
    paragraphs, lists), NOT document order — section extraction from
    `ORDER BY sort` yields headings first and prose last, i.e. empty sections.
    exportMdContent returns the document in real reading order.
    """
    data = siyuan_api("/api/export/exportMdContent", {"id": doc_id})
    if isinstance(data, dict):
        return data.get("content", "") or ""
    return ""


def md_sections(md: str) -> list[tuple[int, str, list[str]]]:
    """Split markdown into (heading_level, heading_text, body_lines) sections."""
    sections: list[tuple[int, str, list[str]]] = []
    current: list | None = None
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            if current is not None:
                sections.append((current[0], current[1], current[2]))
            current = [len(m.group(1)), m.group(2).strip(), []]
        elif current is not None:
            current[2].append(line)
    if current is not None:
        sections.append((current[0], current[1], current[2]))
    return sections


# ── LLM judge helper ────────────────────────────────────────────────────────
def llm_judge(content: str, condition: str, timeout: int = 30) -> tuple[bool, str]:
    api_base = os.getenv("MINDRA_BASE_URL", "https://api.mindracode.com/v1")
    api_key = os.getenv("MINDRA_API_KEY", "")
    if not api_key:
        return False, "MINDRA_API_KEY not set"
    prompt = (
        f"Does the following content satisfy this condition?\n"
        f"Condition: {condition}\n\n"
        f"Content:\n{content}\n\n"
        f"Answer only YES or NO."
    )
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 512},
            timeout=timeout,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        return answer.startswith("YES"), answer
    except Exception as e:
        return False, f"llm_judge error: {e}"


# ── Shared state ─────────────────────────────────────────────────────────────
_film_data: dict | None = None
_book_id: str | None = None
_book_notes_text: str = ""
_siyuan_doc_id: str | None = None
_siyuan_full_content: str = ""
_siyuan_headings: list[str] = []


# ── Individual checks ────────────────────────────────────────────────────────

def check_1_watcharr_film_exists() -> None:
    global _film_data
    try:
        rc, out, err = watcharr_sql(
            "SELECT c.id, c.title, w.status, w.rating, w.thoughts "
            "FROM contents c "
            "JOIN watcheds w ON w.content_id = c.id "
            "JOIN users u ON w.user_id = u.id "
            "WHERE c.type = 'movie' AND u.username = 'admin' "
            "AND (c.title LIKE '%Game of Shadows%' OR c.title LIKE '%game of shadows%') "
            "AND w.deleted_at IS NULL "
            "ORDER BY w.updated_at DESC LIMIT 1;"
        )
        if rc != 0:
            check("1. Watcharr: film found", 1, False, f"sqlite error: {err.strip()[:200]}")
            return
        lines = [l for l in out.strip().splitlines() if l.strip()]
        if not lines:
            rc2, all_titles, _ = watcharr_sql(
                "SELECT c.title FROM contents c JOIN watcheds w ON w.content_id = c.id LIMIT 10;"
            )
            check("1. Watcharr: film found", 1, False,
                  f"film not found; watched titles: {all_titles.strip()[:200]}")
            return
        parts = lines[0].split("\t")
        _film_data = {
            "content_id": parts[0] if len(parts) > 0 else "",
            "title": parts[1] if len(parts) > 1 else "",
            "status": parts[2] if len(parts) > 2 else "",
            "rating": parts[3] if len(parts) > 3 else "",
            "thoughts": parts[4] if len(parts) > 4 else "",
        }
        check("1. Watcharr: film found", 1, True,
              f"title='{_film_data['title']}'")
    except Exception as e:
        check("1. Watcharr: film found", 1, False, f"exception: {e}")


def check_2_watcharr_status_rating() -> None:
    if not _film_data:
        check("2. Watcharr: status Watched + rating 8/10", 1, False,
              "skipped: film not found")
        return
    try:
        status = _film_data["status"]
        rating_str = _film_data["rating"]
        status_ok = status.upper() == "FINISHED"
        try:
            rating_val = float(rating_str)
        except (ValueError, TypeError):
            rating_val = -1
        rating_ok = abs(rating_val - 8.0) < 0.5
        passed = status_ok and rating_ok
        detail = f"status='{status}', rating={rating_str}"
        if not status_ok:
            detail += " (expected FINISHED)"
        if not rating_ok:
            detail += " (expected 8.0)"
        check("2. Watcharr: status Watched + rating 8/10", 1, passed, detail)
    except Exception as e:
        check("2. Watcharr: status Watched + rating 8/10", 1, False, f"exception: {e}")


def check_3_watcharr_review_quality() -> None:
    if not _film_data:
        check("3. Watcharr: review quality (Moriarty/adaptation focus)", 2, False,
              "skipped: film not found")
        return
    try:
        thoughts = _film_data.get("thoughts", "")
        if not thoughts or len(thoughts) < 50:
            check("3. Watcharr: review quality (Moriarty/adaptation focus)", 2, False,
                  f"review too short: {len(thoughts)} chars (need ≥50)")
            return
        passed, answer = llm_judge(
            thoughts,
            "The review focuses on Moriarty's characterisation, the adaptation treatment "
            "of Moriarty, or Guy Ritchie's narrative/directorial choices — "
            "NOT a generic plot summary or generic praise like 'great film, exciting action'."
        )
        check("3. Watcharr: review quality (Moriarty/adaptation focus)", 2, passed,
              f"llm_judge={answer}, review_len={len(thoughts)}")
    except Exception as e:
        check("3. Watcharr: review quality (Moriarty/adaptation focus)", 2, False,
              f"exception: {e}")


def check_4_booklore_book_exists() -> None:
    global _book_id
    try:
        result = mariadb_query(
            "SELECT bm.book_id FROM book_metadata bm "
            "WHERE bm.title LIKE '%Valley of Fear%' LIMIT 1"
        )
        if result:
            _book_id = result.strip().split("\n")[0].strip()
            author_q = (
                f"SELECT a.name FROM author a "
                f"JOIN book_metadata_author_mapping m ON a.id = m.author_id "
                f"WHERE m.book_id = {_book_id}"
            )
            author = mariadb_query(author_q)
            status_q = (
                f"SELECT COALESCE(ubp.read_status, b.read_status) "
                f"FROM book b "
                f"LEFT JOIN user_book_progress ubp ON b.id = ubp.book_id "
                f"WHERE b.id = {_book_id} LIMIT 1"
            )
            status = mariadb_query(status_q)
            status_ok = status.strip().upper() in ("READ", "COMPLETED", "FINISHED")
            detail = f"book_id={_book_id}, author={author}, status={status}"
            if not status_ok:
                detail += " (expected Read status)"
            check("4. Booklore: Valley of Fear exists (status Read)", 1, status_ok, detail)
        else:
            all_q = "SELECT bm.book_id, bm.title FROM book_metadata bm LIMIT 10"
            all_titles = mariadb_query(all_q)
            check("4. Booklore: Valley of Fear exists (status Read)", 1, False,
                  f"book not found; available: {all_titles[:200]}")
    except Exception as e:
        check("4. Booklore: Valley of Fear exists (status Read)", 1, False,
              f"exception: {e}")


def check_5_booklore_notes_count() -> None:
    global _book_notes_text
    if not _book_id:
        check("5. Booklore: ≥5 reading notes", 2, False, "skipped: book not found")
        return
    try:
        notes_v2 = mariadb_query(
            f"SELECT note_content FROM book_notes_v2 WHERE book_id = {_book_id}"
        )
        notes_v1 = mariadb_query(
            f"SELECT content FROM book_notes WHERE book_id = {_book_id}"
        )
        all_notes_raw = notes_v2 or notes_v1
        if not all_notes_raw:
            check("5. Booklore: ≥5 reading notes", 2, False, "no notes found (v1 or v2)")
            return

        note_lines = [l.strip() for l in all_notes_raw.strip().split("\n") if l.strip()]
        _book_notes_text = "\n".join(note_lines)

        count_v2 = mariadb_query(
            f"SELECT COUNT(*) FROM book_notes_v2 WHERE book_id = {_book_id}"
        )
        count_v1 = mariadb_query(
            f"SELECT COUNT(*) FROM book_notes WHERE book_id = {_book_id}"
        )
        try:
            note_count = max(int(count_v2 or "0"), int(count_v1 or "0"))
        except ValueError:
            note_count = len(note_lines)

        if note_count < 5 and len(note_lines) >= 5:
            note_count = len(note_lines)

        passed = note_count >= 5
        check("5. Booklore: ≥5 reading notes", 2, passed,
              f"count={note_count}")
    except Exception as e:
        check("5. Booklore: ≥5 reading notes", 2, False, f"exception: {e}")


def check_6_booklore_notes_dimensions() -> None:
    if not _book_notes_text:
        check("6. Booklore: notes cover ≥3 distinct dimensions", 2, False,
              "skipped: no notes text")
        return
    try:
        passed, answer = llm_judge(
            _book_notes_text,
            "The reading notes list at least 5 specific differences between the novel "
            "'The Valley of Fear' by Arthur Conan Doyle and the film "
            "'Sherlock Holmes: A Game of Shadows'. The differences must span at least "
            "3 of the following 5 dimensions: "
            "(1) Moriarty's characterisation (presence, representation method, screen/page time), "
            "(2) Mystery/detective structure (plot architecture, reveal mechanism), "
            "(3) Historical period and setting (Victorian authenticity vs Edwardian/steampunk style), "
            "(4) Visual/linguistic style (Doyle's prose restraint vs Ritchie's kinetic choreography), "
            "(5) Supporting character functions (Lestrade, Watson's role, secondary figures)."
        )
        check("6. Booklore: notes cover ≥3 distinct dimensions", 2, passed,
              f"llm_judge={answer}")
    except Exception as e:
        check("6. Booklore: notes cover ≥3 distinct dimensions", 2, False,
              f"exception: {e}")


def check_7_siyuan_doc_exists() -> None:
    global _siyuan_doc_id
    try:
        rows = siyuan_sql(
            "SELECT id, content FROM blocks "
            "WHERE type = 'd' AND content LIKE '%EP-44%'"
        )
        if rows:
            _siyuan_doc_id = rows[0].get("id", "")
            title = rows[0].get("content", "")
            check("7. SiYuan: EP-44 doc exists", 1, True,
                  f"doc_id={_siyuan_doc_id}, title='{title}'")
            return

        rows_alt = siyuan_sql(
            "SELECT id, content FROM blocks "
            "WHERE type = 'd' AND content LIKE '%恐惧谷%'"
        )
        if rows_alt:
            _siyuan_doc_id = rows_alt[0].get("id", "")
            title = rows_alt[0].get("content", "")
            check("7. SiYuan: EP-44 doc exists", 1, True,
                  f"doc_id={_siyuan_doc_id}, title='{title}' (alt match)")
            return

        all_docs = siyuan_sql(
            "SELECT id, content FROM blocks WHERE type = 'd' LIMIT 10"
        )
        titles = [r.get("content", "") for r in (all_docs or [])]
        check("7. SiYuan: EP-44 doc exists", 1, False,
              f"no EP-44 doc found; available docs: {titles[:5]}")
    except Exception as e:
        check("7. SiYuan: EP-44 doc exists", 1, False, f"exception: {e}")


def check_8_siyuan_four_sections() -> None:
    global _siyuan_full_content, _siyuan_headings
    if not _siyuan_doc_id:
        check("8. SiYuan: all 4 required sections present", 2, False,
              "skipped: doc not found")
        return
    try:
        md = siyuan_export_md(_siyuan_doc_id)
        sections = md_sections(md)
        _siyuan_full_content = md
        _siyuan_headings = [title for _lvl, title, _lines in sections]

        section_patterns = {
            "Introduction": r"(导言|introduction|intro)",
            "Five Core Differences": r"(五处核心差异|five\s+core\s+differ|核心差异|差异)",
            "Creator Intent": r"(创作者意图|creator\s+intent|意图解读)",
            "Conclusion": r"(结论|conclusion)",
        }
        found_sections = []
        missing_sections = []
        headings_lower = "\n".join(_siyuan_headings).lower()
        content_lower = _siyuan_full_content.lower()

        for name, pattern in section_patterns.items():
            if re.search(pattern, headings_lower, re.IGNORECASE) or \
               re.search(pattern, content_lower, re.IGNORECASE):
                found_sections.append(name)
            else:
                missing_sections.append(name)

        passed = len(missing_sections) == 0
        detail = f"found={found_sections}"
        if missing_sections:
            detail += f", missing={missing_sections}"
        detail += f", headings={_siyuan_headings[:8]}"
        check("8. SiYuan: all 4 required sections present", 2, passed, detail)
    except Exception as e:
        check("8. SiYuan: all 4 required sections present", 2, False, f"exception: {e}")


def check_9_siyuan_introduction_length() -> None:
    if not _siyuan_doc_id:
        check("9. SiYuan: Introduction ≥120 chars", 1, False, "skipped: doc not found")
        return
    try:
        md = siyuan_export_md(_siyuan_doc_id)
        intro_text = ""
        for _lvl, title, lines in md_sections(md):
            if re.search(r"(导言|introduction|intro)", title, re.IGNORECASE):
                intro_text = " ".join(l.strip() for l in lines if l.strip())
                break

        intro_len = len(intro_text.strip())
        passed = intro_len >= 120
        check("9. SiYuan: Introduction ≥120 chars", 1, passed,
              f"intro_length={intro_len}")
    except Exception as e:
        check("9. SiYuan: Introduction ≥120 chars", 1, False, f"exception: {e}")


def check_10_siyuan_content_consistency() -> None:
    if not _siyuan_full_content:
        check("10. SiYuan: content consistent with Booklore notes", 2, False,
              "skipped: SiYuan doc empty or not found")
        return
    try:
        combined = f"=== SiYuan Document Content ===\n{_siyuan_full_content[:3000]}"
        if _book_notes_text:
            combined += f"\n\n=== Booklore Reading Notes ===\n{_book_notes_text[:2000]}"

        passed, answer = llm_judge(
            combined,
            "The SiYuan document's 'Five Core Differences' section presents content that is "
            "semantically consistent with the reading notes recorded in Booklore. "
            "The differences discussed in both sources should overlap substantially — "
            "they should cover the same or related comparison points between "
            "'The Valley of Fear' (novel) and 'Sherlock Holmes: A Game of Shadows' (film)."
        )
        check("10. SiYuan: content consistent with Booklore notes", 2, passed,
              f"llm_judge={answer}")
    except Exception as e:
        check("10. SiYuan: content consistent with Booklore notes", 2, False,
              f"exception: {e}")


def check_11_siyuan_conclusion_position() -> None:
    if not _siyuan_doc_id:
        check("11. SiYuan: conclusion takes explicit position", 2, False,
              "skipped: doc not found")
        return
    try:
        md = siyuan_export_md(_siyuan_doc_id)
        conclusion_text = ""
        for _lvl, title, lines in md_sections(md):
            if re.search(r"(结论|conclusion)", title, re.IGNORECASE):
                conclusion_text = " ".join(l.strip() for l in lines if l.strip())
                break

        if not conclusion_text.strip():
            check("11. SiYuan: conclusion takes explicit position", 2, False,
                  "conclusion section empty or not found")
            return

        passed, answer = llm_judge(
            conclusion_text,
            "The conclusion explicitly recommends either the novel ('The Valley of Fear') "
            "or the film ('Sherlock Holmes: A Game of Shadows') with a specific reason. "
            "A vague answer like 'both have merit' or 'each has its strengths' without "
            "a clear preference is NOT acceptable — the author must take a definitive position."
        )
        check("11. SiYuan: conclusion takes explicit position", 2, passed,
              f"llm_judge={answer}, text_len={len(conclusion_text.strip())}")
    except Exception as e:
        check("11. SiYuan: conclusion takes explicit position", 2, False,
              f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_watcharr_film_exists()
    check_2_watcharr_status_rating()
    check_3_watcharr_review_quality()
    check_4_booklore_book_exists()
    check_5_booklore_notes_count()
    check_6_booklore_notes_dimensions()
    check_7_siyuan_doc_exists()
    check_8_siyuan_four_sections()
    check_9_siyuan_introduction_length()
    check_10_siyuan_content_consistency()
    check_11_siyuan_conclusion_position()

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
