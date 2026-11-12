# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_024/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_024: Spider-Man poster → Watcharr review + Booklore graphic novel + SiYuan comparison episode

Checks: 14 weighted checks (24 total points) across watcharr, booklore, siyuan.
Strategy: host-side SQLite via docker cp (watcharr), docker exec MariaDB (booklore), REST API (siyuan),
          llm_judge + llm_judge_vision for content quality and cross-modal consistency.

Required env vars:
  SERVER_HOSTNAME, WATCHARR_PORT, WATCHARR_CONTAINER,
  BOOKLORE_PORT, BOOKLORE_CONTAINER,
  SIYUAN_PORT, SIYUAN_CONTAINER
"""

import base64
import json
import os
import re
import subprocess
import sys

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import requests

# -- Config (from env) ---------------------------------------------------------
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

WATCHARR_PORT = os.getenv("WATCHARR_PORT")
WATCHARR_CONTAINER = os.getenv("WATCHARR_CONTAINER")
BOOKLORE_PORT = os.getenv("BOOKLORE_PORT")
BOOKLORE_CONTAINER = os.getenv("BOOKLORE_CONTAINER")
SIYUAN_PORT = os.getenv("SIYUAN_PORT")
SIYUAN_CONTAINER = os.getenv("SIYUAN_CONTAINER")

_missing = []
for _var in ["WATCHARR_PORT", "WATCHARR_CONTAINER",
             "BOOKLORE_PORT", "BOOKLORE_CONTAINER",
             "SIYUAN_PORT", "SIYUAN_CONTAINER"]:
    if not os.getenv(_var):
        _missing.append(_var)
if _missing:
    print(f"FATAL: {', '.join(_missing)} not set", file=sys.stderr)
    sys.exit(1)

BOOKLORE_DB_CONTAINER = os.getenv("BOOKLORE_DB_CONTAINER") or BOOKLORE_CONTAINER

_INPUTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs"
)

INPUT_FILES: list[str] = [
    os.path.join(_INPUTS_DIR, "watcharr_poster_001.jpg"),
]

# -- Result accumulator --------------------------------------------------------
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# -- Helpers -------------------------------------------------------------------
def docker_exec(container: str, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    r = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


# -- Watcharr helpers (SQLite via docker exec) ---------------------------------
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


# -- Booklore helpers (MariaDB via docker exec) --------------------------------
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


# -- SiYuan helpers (REST API) -------------------------------------------------
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


# -- LLM judge helpers ---------------------------------------------------------
def llm_judge(content: str, condition: str, timeout: int = 30) -> tuple[bool, str]:
    api_base = os.getenv("MINDRA_BASE_URL", "https://api.mindracode.com/v1")
    api_key = os.getenv("MINDRA_API_KEY", "")
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


def llm_judge_vision(
    image_path: str,
    recorded_value: str,
    condition: str,
    timeout: int = 45,
) -> tuple[bool, str]:
    api_base = os.getenv("MINDRA_BASE_URL", "https://api.mindracode.com/v1")
    api_key = os.getenv("MINDRA_API_KEY", "")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")

    if not os.path.isfile(image_path):
        return False, f"image not found: {image_path}"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = (
        f"You are given an image and a value that an AI agent extracted from it.\n"
        f"Recorded value: «{recorded_value}»\n"
        f"Condition: {condition}\n\n"
        f"Does the recorded value accurately match the information visible in the image, "
        f"satisfying the condition above?\n"
        f"Answer only YES or NO."
    )
    msg_content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
                  "messages": [{"role": "user", "content": msg_content}],
                  "max_tokens": 512},
            timeout=timeout,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        return answer.startswith("YES"), answer
    except Exception as e:
        return False, f"llm_judge_vision error: {e}"


# -- Shared state --------------------------------------------------------------
_input_files_ok: bool = False
_watcharr_row: dict = {}
_book_id: str | None = None
_book_notes: list[str] = []
_doc_id: str = ""
_doc_blocks: list[dict] = []


# -- Individual checks ---------------------------------------------------------
def check_0_input_files_exist() -> None:
    global _input_files_ok
    missing = [p for p in INPUT_FILES if not os.path.isfile(p)]
    if missing:
        check("0. input_files_exist", 1, False, "missing: " + ", ".join(missing))
    else:
        _input_files_ok = True
        check("0. input_files_exist", 1, True)


def check_1_watcharr_film_exists() -> None:
    global _watcharr_row
    try:
        rc, out, err = watcharr_sql(
            "SELECT c.id, c.title, w.status, w.rating, w.thoughts "
            "FROM contents c "
            "JOIN watcheds w ON w.content_id = c.id "
            "JOIN users u ON w.user_id = u.id "
            "WHERE c.type = 'movie' AND u.username = 'admin' "
            "AND w.deleted_at IS NULL "
            "AND LOWER(c.title) LIKE '%spider-man%no way home%' "
            "ORDER BY w.updated_at DESC LIMIT 1;"
        )
        if rc != 0:
            check("1. watcharr_film_exists", 2, False, f"sqlite error: {err.strip()[:200]}")
            return
        lines = [l for l in out.strip().splitlines() if l.strip()]
        if not lines:
            rc2, fallback, _ = watcharr_sql(
                "SELECT c.id, c.title, w.status, w.rating, w.thoughts "
                "FROM contents c "
                "JOIN watcheds w ON w.content_id = c.id "
                "JOIN users u ON w.user_id = u.id "
                "WHERE c.type = 'movie' AND u.username = 'admin' "
                "AND w.deleted_at IS NULL AND LOWER(c.title) LIKE '%spider%' "
                "ORDER BY w.updated_at DESC LIMIT 1;"
            )
            fb_lines = [l for l in fallback.strip().splitlines() if l.strip()]
            if rc2 == 0 and fb_lines:
                lines = fb_lines
            else:
                rc3, all_titles, _ = watcharr_sql(
                    "SELECT c.title FROM contents c JOIN watcheds w ON w.content_id = c.id "
                    "JOIN users u ON w.user_id = u.id "
                    "WHERE u.username = 'admin' AND w.deleted_at IS NULL LIMIT 10;"
                )
                check("1. watcharr_film_exists", 2, False,
                      f"no Spider-Man: No Way Home entry found for admin; "
                      f"watched titles: {all_titles.strip()[:200]}")
                return
        parts = lines[0].split("\t")
        _watcharr_row = {
            "content_id": parts[0] if len(parts) > 0 else "",
            "title": parts[1] if len(parts) > 1 else "",
            "status": parts[2] if len(parts) > 2 else "",
            "rating": parts[3] if len(parts) > 3 else "",
            "thoughts": parts[4] if len(parts) > 4 else "",
        }
        check("1. watcharr_film_exists", 2, True, f"found: {_watcharr_row['title']}")
    except Exception as e:
        check("1. watcharr_film_exists", 2, False, f"exception: {e}")


def check_2_watcharr_status_watched() -> None:
    if not _watcharr_row:
        check("2. watcharr_status_watched", 1, False, "no watched row available")
        return
    try:
        status = _watcharr_row["status"]
        passed = status.upper() == "FINISHED"
        detail = "" if passed else f"status is '{status}', expected 'FINISHED'"
        check("2. watcharr_status_watched", 1, passed, detail)
    except Exception as e:
        check("2. watcharr_status_watched", 1, False, f"exception: {e}")


def check_3_watcharr_rating_7() -> None:
    if not _watcharr_row:
        check("3. watcharr_rating_7", 1, False, "no watched row available")
        return
    try:
        raw = _watcharr_row["rating"]
        rating = float(raw)
        passed = abs(rating - 7.0) < 0.5
        detail = "" if passed else f"rating is {rating}, expected ~7.0"
        check("3. watcharr_rating_7", 1, passed, detail)
    except Exception as e:
        check("3. watcharr_rating_7", 1, False, f"exception: {e}")


def check_4_watcharr_review_length() -> None:
    if not _watcharr_row:
        check("4. watcharr_review_length", 1, False, "no watched row available")
        return
    try:
        thoughts = _watcharr_row.get("thoughts", "")
        length = len(thoughts.strip())
        passed = length >= 80
        detail = "" if passed else f"review length {length} chars, need >=80"
        check("4. watcharr_review_length", 1, passed, detail)
    except Exception as e:
        check("4. watcharr_review_length", 1, False, f"exception: {e}")


def check_5_watcharr_review_content() -> None:
    if not _watcharr_row:
        check("5. watcharr_review_content", 2, False, "no watched row available")
        return
    try:
        thoughts = _watcharr_row.get("thoughts", "")
        if len(thoughts) < 20:
            check("5. watcharr_review_content", 2, False,
                  f"review too short ({len(thoughts)} chars) to evaluate")
            return
        condition = (
            "The review discusses how the film balances fan-service (nostalgia, "
            "callbacks, returning characters, crowd-pleasing moments) with narrative "
            "stakes (story tension, character development, emotional weight, plot "
            "consequences). It is NOT a generic plot summary or simple rating."
        )
        passed, raw = llm_judge(thoughts, condition)
        check("5. watcharr_review_content", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("5. watcharr_review_content", 2, False, f"exception: {e}")


def check_6_cross_modal_poster_title() -> None:
    if not _input_files_ok:
        check("6. cross_modal_poster_title", 2, False, "skipped: input file missing")
        return
    try:
        title = _watcharr_row.get("title", "") if _watcharr_row else ""
        if not title:
            check("6. cross_modal_poster_title", 2, False,
                  "skipped: no film title recorded in Watcharr")
            return
        condition = (
            "The movie poster shown in the image is for the film whose title matches "
            f"or is clearly the same film as '{title}'. The title or key visual elements "
            "on the poster correspond to this film."
        )
        passed, raw = llm_judge_vision(INPUT_FILES[0], title, condition)
        check("6. cross_modal_poster_title", 2, passed, f"llm_judge_vision: {raw}")
    except Exception as e:
        check("6. cross_modal_poster_title", 2, False, f"exception: {e}")


def check_7_booklore_book_exists() -> None:
    global _book_id
    try:
        for pattern in [
            "Spider-Man: Blue",
            "Spider-Man%Blue",
            "spider-man%blue",
            "Spiderman Blue",
        ]:
            result = mariadb_query(
                f"SELECT bm.book_id FROM book_metadata bm "
                f"WHERE bm.title LIKE '%{pattern}%' LIMIT 1"
            )
            if result and result.strip():
                _book_id = result.strip().splitlines()[0].strip()
                title_q = mariadb_query(
                    f"SELECT bm.title FROM book_metadata bm WHERE bm.book_id = {_book_id}"
                )
                author_q = mariadb_query(
                    f"SELECT a.name FROM author a "
                    f"JOIN book_metadata_author_mapping m ON a.id = m.author_id "
                    f"WHERE m.book_id = {_book_id}"
                )
                check("7. booklore_book_exists", 2, True,
                      f"book_id={_book_id}, title='{title_q}', author='{author_q}'")
                return
        all_titles = mariadb_query(
            "SELECT bm.book_id, bm.title FROM book_metadata bm ORDER BY bm.book_id DESC LIMIT 10"
        )
        check("7. booklore_book_exists", 2, False,
              f"'Spider-Man: Blue' not found; recent: {all_titles[:200]}")
    except Exception as e:
        check("7. booklore_book_exists", 2, False, f"exception: {e}")


def check_8_booklore_read_status() -> None:
    if not _book_id:
        check("8. booklore_read_status", 1, False, "book not found")
        return
    try:
        result = mariadb_query(
            f"SELECT ubp.read_status FROM user_book_progress ubp "
            f"WHERE ubp.book_id = {_book_id} LIMIT 1"
        )
        if not result:
            result = mariadb_query(
                f"SELECT s.name FROM shelf s "
                f"JOIN book_shelf_mapping bsm ON s.id = bsm.shelf_id "
                f"WHERE bsm.book_id = {_book_id}"
            )
            if result and "read" in result.lower():
                check("8. booklore_read_status", 1, True,
                      f"on shelf: {result}")
                return
            check("8. booklore_read_status", 1, False,
                  f"no progress record for book_id={_book_id}")
            return
        status = result.strip().upper()
        passed = status == "READ"
        check("8. booklore_read_status", 1, passed,
              "" if passed else f"status='{status}', expected 'READ'")
    except Exception as e:
        check("8. booklore_read_status", 1, False, f"exception: {e}")


def check_9_booklore_notes_count() -> None:
    global _book_notes
    if not _book_id:
        check("9. booklore_notes_count", 2, False, "book not found")
        return
    try:
        notes_list: list[str] = []
        for table, col in [("book_notes_v2", "note_content"), ("book_notes", "content"),
                           ("annotations", "note")]:
            count_str = mariadb_query(
                f"SELECT COUNT(*) FROM {table} WHERE book_id = {_book_id}"
            )
            if count_str and int(count_str) > 0:
                result = mariadb_query(
                    f"SELECT {col} FROM {table} WHERE book_id = {_book_id}"
                )
                if result:
                    notes_list = [l.strip() for l in result.strip().splitlines() if l.strip()]
                break
        _book_notes = notes_list
        note_count = 0
        for table in ["book_notes_v2", "book_notes", "annotations"]:
            cs = mariadb_query(
                f"SELECT COUNT(*) FROM {table} WHERE book_id = {_book_id}"
            )
            if cs and int(cs) > 0:
                note_count = int(cs)
                break
        if note_count == 0:
            note_count = len(notes_list)
        passed = note_count >= 2
        check("9. booklore_notes_count", 2, passed,
              f"count={note_count}" if passed else f"found {note_count} notes, need >=2")
    except Exception as e:
        check("9. booklore_notes_count", 2, False, f"exception: {e}")


def check_10_booklore_notes_content() -> None:
    if not _book_notes:
        check("10. booklore_notes_content", 2, False, "no notes available")
        return
    try:
        combined = "\n---\n".join(f"Note {i+1}: {n}" for i, n in enumerate(_book_notes))
        condition = (
            "The notes discuss character relationship differences between the graphic novel "
            "'Spider-Man: Blue' (by Jeph Loeb) and the Spider-Man films. The notes should "
            "focus on how character relationships (e.g., Peter Parker and Gwen Stacy, "
            "Peter and Mary Jane, or other key relationships) differ between the comic "
            "and the film adaptation. Generic notes without relationship-specific analysis "
            "do not count."
        )
        passed, raw = llm_judge(combined, condition)
        check("10. booklore_notes_content", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("10. booklore_notes_content", 2, False, f"exception: {e}")


def check_11_siyuan_doc_exists() -> None:
    global _doc_id, _doc_blocks
    try:
        for pattern in [
            "EP-Adapt%Spider-Man Mythos",
            "EP-Adapt%Spider%Mythos",
            "EP-Adapt",
            "Spider-Man Mythos",
            "Spider-Man%Mythos",
        ]:
            rows = siyuan_sql(
                f"SELECT id, content FROM blocks "
                f"WHERE type = 'd' AND content LIKE '%{pattern}%' LIMIT 5"
            )
            if rows:
                for r in rows:
                    content = r.get("content", "")
                    if ("ep-adapt" in content.lower() or
                            "spider" in content.lower() and "mythos" in content.lower()):
                        _doc_id = r["id"]
                        _doc_blocks = siyuan_sql(
                            f"SELECT id, type, subtype, content, markdown "
                            f"FROM blocks WHERE root_id='{_doc_id}' "
                            f"AND type != 'd' ORDER BY sort"
                        )
                        check("11. siyuan_doc_exists", 2, True,
                              f"doc='{content[:80]}'")
                        return
        all_docs = siyuan_sql(
            "SELECT content FROM blocks WHERE type = 'd' ORDER BY updated DESC LIMIT 10"
        )
        titles = [r.get("content", "") for r in (all_docs or [])]
        check("11. siyuan_doc_exists", 2, False,
              f"no EP-Adapt / Spider-Man Mythos doc found; available: {titles[:5]}")
    except Exception as e:
        check("11. siyuan_doc_exists", 2, False, f"exception: {e}")


def check_12_siyuan_structure() -> None:
    if not _doc_id:
        check("12. siyuan_structure", 3, False, "doc not found")
        return
    try:
        full_md = siyuan_export_md(_doc_id)
        sections = md_sections(full_md)
        headings = [title for _lvl, title, _lines in sections]

        # Intro: body of the first intro-like heading's section; fallback to the
        # leading paragraphs before the first heading (no-heading docs).
        has_intro = False
        intro_text = ""
        lead_lines: list[str] = []
        for line in full_md.splitlines():
            if re.match(r"^#{1,6}\s", line):
                break
            if line.strip():
                lead_lines.append(line.strip())
        intro_text = " ".join(lead_lines)
        for _lvl, title, lines in sections:
            tl = title.lower()
            if any(kw in tl for kw in ["intro", "introduction", "引言", "开场"]):
                has_intro = True
                body = " ".join(l.strip() for l in lines if l.strip())
                if len(body) > len(intro_text):
                    intro_text = body
                break

        if not has_intro and len(intro_text.strip()) >= 120:
            has_intro = True

        intro_long_enough = len(intro_text.strip()) >= 120

        has_differences = False
        has_conclusion = False
        for h in headings:
            hl = h.lower()
            if any(kw in hl for kw in ["3 key", "key difference", "difference",
                                        "三个", "关键差异", "对比"]):
                has_differences = True
            if any(kw in hl for kw in ["conclusion", "recommend", "结论", "推荐"]):
                has_conclusion = True

        if not has_differences:
            numbered_items = re.findall(r'^\s*\d+[\.\)]\s+.{10,}', full_md, re.MULTILINE)
            if len(numbered_items) >= 3:
                has_differences = True

        issues = []
        if not has_intro:
            issues.append("no intro section detected")
        elif not intro_long_enough:
            issues.append(f"intro only {len(intro_text.strip())} chars, need >=120")
        if not has_differences:
            issues.append("no '3 Key Differences' section or numbered list with >=3 items")
        if not has_conclusion:
            issues.append("no conclusion/recommendation section")

        passed = has_intro and intro_long_enough and has_differences and has_conclusion
        if not passed and len(full_md) > 200:
            condition = (
                "This document has a structured comparison of Spider-Man film and source material. "
                "It contains: (1) an introductory section of at least 120 characters, "
                "(2) a section listing at least 3 key differences between the film and "
                "the graphic novel/comic, numbered or bulleted, and (3) a conclusion that "
                "recommends one version (film or comic) to the audience."
            )
            llm_passed, raw = llm_judge(full_md[:4000], condition)
            if llm_passed:
                passed = True
                issues.append(f"llm_judge override: {raw}")

        detail = f"headings={headings[:8]}"
        if issues:
            detail += "; " + "; ".join(issues)
        check("12. siyuan_structure", 3, passed, detail)
    except Exception as e:
        check("12. siyuan_structure", 3, False, f"exception: {e}")


def check_13_siyuan_links() -> None:
    if not _doc_id:
        check("13. siyuan_links", 2, False, "doc not found")
        return
    try:
        all_md = "\n".join(
            b.get("markdown", "") or b.get("content", "")
            for b in _doc_blocks
        )

        has_watcharr_link = bool(re.search(
            r'https?://[^\s)]*watcharr[^\s)]*|watcharr.*(?:http|link|url)',
            all_md, re.IGNORECASE,
        ))
        if not has_watcharr_link:
            has_watcharr_link = bool(re.search(
                r'http[s]?://[^\s)]+:\d+[^\s)]*',
                all_md,
            )) and "watcharr" in all_md.lower()

        has_booklore_link = bool(re.search(
            r'https?://[^\s)]*booklore[^\s)]*|booklore.*(?:http|link|url)',
            all_md, re.IGNORECASE,
        ))
        if not has_booklore_link:
            has_booklore_link = bool(re.search(
                r'http[s]?://[^\s)]+:\d+[^\s)]*',
                all_md,
            )) and "booklore" in all_md.lower()

        if not has_watcharr_link:
            has_watcharr_link = bool(re.search(
                r'http[s]?://[^\s)]+/(?:movie|film|content|watched)[^\s)]*',
                all_md, re.IGNORECASE,
            ))
        if not has_booklore_link:
            has_booklore_link = bool(re.search(
                r'http[s]?://[^\s)]+/(?:book|library|reading)[^\s)]*',
                all_md, re.IGNORECASE,
            ))

        url_count = len(re.findall(r'https?://[^\s)]+', all_md))

        refs = siyuan_sql(
            f"SELECT def_block_root_id, content, markdown FROM refs "
            f"WHERE root_id = '{_doc_id}' LIMIT 20"
        )
        has_siyuan_refs = len(refs) >= 2 if refs else False

        if has_watcharr_link and has_booklore_link:
            check("13. siyuan_links", 2, True,
                  "links to both Watcharr and Booklore entries found")
        elif url_count >= 2:
            check("13. siyuan_links", 2, True,
                  f"found {url_count} URLs (assumed Watcharr + Booklore)")
        elif has_siyuan_refs:
            check("13. siyuan_links", 2, True,
                  f"found {len(refs)} SiYuan block refs (bidirectional links)")
        elif url_count == 1:
            which = "Watcharr" if has_watcharr_link else (
                "Booklore" if has_booklore_link else "unknown app"
            )
            check("13. siyuan_links", 2, False,
                  f"only 1 URL found (likely {which}); need links to both apps")
        else:
            check("13. siyuan_links", 2, False,
                  "no hyperlinks to Watcharr or Booklore entries found")
    except Exception as e:
        check("13. siyuan_links", 2, False, f"exception: {e}")


# -- Main ----------------------------------------------------------------------
def main() -> None:
    check_0_input_files_exist()
    check_1_watcharr_film_exists()
    check_2_watcharr_status_watched()
    check_3_watcharr_rating_7()
    check_4_watcharr_review_length()
    check_5_watcharr_review_content()
    check_6_cross_modal_poster_title()
    check_7_booklore_book_exists()
    check_8_booklore_read_status()
    check_9_booklore_notes_count()
    check_10_booklore_notes_content()
    check_11_siyuan_doc_exists()
    check_12_siyuan_structure()
    check_13_siyuan_links()

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
