# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_065/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_065: Vigilante film (from poster) + Atrocious Judges book → SiYuan comparison script

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
    os.path.join(_INPUTS_DIR, "watcharr_poster_002.jpg"),
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
    paragraphs, lists), NOT document order — position-sensitive extraction from
    `ORDER BY sort` yields headings first and prose last, i.e. wrong positions.
    exportMdContent returns the document in real reading order.
    """
    data = siyuan_api("/api/export/exportMdContent", {"id": doc_id})
    if isinstance(data, dict):
        return data.get("content", "") or ""
    return ""


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
_ep47_doc_id: str = ""
_ep47_blocks: list[dict] = []


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
            "JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
            "WHERE c.type = 'movie' "
            "AND LOWER(c.title) LIKE '%the batman%' "
            "AND w.deleted_at IS NULL "
            "ORDER BY w.updated_at DESC LIMIT 20;"
        )
        if rc != 0:
            check("1. watcharr_film_exists", 2, False, f"sqlite error: {err.strip()[:200]}")
            return
        lines = [l for l in out.strip().splitlines() if l.strip()]
        if not lines:
            check("1. watcharr_film_exists", 2, False, "no watched movies found")
            return
        for line in lines:
            parts = line.split("\t")
            title = parts[1] if len(parts) > 1 else ""
            rating_str = parts[3] if len(parts) > 3 else ""
            try:
                rating_val = float(rating_str)
            except (ValueError, TypeError):
                rating_val = -1
            if abs(rating_val - 7.5) < 0.01:
                _watcharr_row = {
                    "content_id": parts[0] if len(parts) > 0 else "",
                    "title": title,
                    "status": parts[2] if len(parts) > 2 else "",
                    "rating": rating_str,
                    "thoughts": parts[4] if len(parts) > 4 else "",
                }
                check("1. watcharr_film_exists", 2, True,
                      f"found: {_watcharr_row['title']}")
                return
        all_titles = "; ".join(
            f"{l.split(chr(9))[1] if len(l.split(chr(9))) > 1 else '?'} "
            f"(r={l.split(chr(9))[3] if len(l.split(chr(9))) > 3 else '?'})"
            for l in lines[:5]
        )
        check("1. watcharr_film_exists", 2, False,
              f"no movie with rating 7.5 found; recent: {all_titles}")
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


def check_3_watcharr_rating_7_5() -> None:
    if not _watcharr_row:
        check("3. watcharr_rating_7.5", 1, False, "no watched row available")
        return
    try:
        raw = _watcharr_row["rating"]
        rating = float(raw)
        passed = abs(rating - 7.5) < 0.5
        detail = "" if passed else f"rating is {rating}, expected 7.5"
        check("3. watcharr_rating_7.5", 1, passed, detail)
    except Exception as e:
        check("3. watcharr_rating_7.5", 1, False, f"exception: {e}")


def check_4_watcharr_review_exists() -> None:
    if not _watcharr_row:
        check("4. watcharr_review_exists", 1, False, "no watched row available")
        return
    try:
        thoughts = _watcharr_row.get("thoughts", "")
        passed = len(thoughts.strip()) > 0
        detail = "" if passed else "review (thoughts) is empty"
        check("4. watcharr_review_exists", 1, passed, detail)
    except Exception as e:
        check("4. watcharr_review_exists", 1, False, f"exception: {e}")


def check_5_watcharr_review_cinematography() -> None:
    if not _watcharr_row:
        check("5. watcharr_review_cinematography", 2, False, "no watched row available")
        return
    try:
        thoughts = _watcharr_row.get("thoughts", "")
        if len(thoughts) < 20:
            check("5. watcharr_review_cinematography", 2, False,
                  f"review too short ({len(thoughts)} chars) for cinematography analysis")
            return
        condition = (
            "The review analyzes the film's cinematography, specifically discussing "
            "the use of shadow and/or focus (depth of field, selective focus, lighting "
            "contrast, chiaroscuro, dark tones, visual framing). It is NOT a generic "
            "plot summary or simple praise."
        )
        passed, raw = llm_judge(thoughts, condition)
        check("5. watcharr_review_cinematography", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("5. watcharr_review_cinematography", 2, False, f"exception: {e}")


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
            "Atrocious Judges",
            "atrocious judges",
            "Judges Infamous",
            "judges infamous",
            "Tools of Tyrants",
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
                check("7. booklore_book_exists", 1, True,
                      f"book_id={_book_id}, title='{title_q}'")
                return
        all_titles = mariadb_query(
            "SELECT bm.book_id, bm.title FROM book_metadata bm LIMIT 10"
        )
        check("7. booklore_book_exists", 1, False,
              f"'Atrocious Judges' not found; available: {all_titles[:200]}")
    except Exception as e:
        check("7. booklore_book_exists", 1, False, f"exception: {e}")


def check_8_booklore_read_status() -> None:
    if not _book_id:
        check("8. booklore_read_status", 2, False, "book not found")
        return
    try:
        result = mariadb_query(
            f"SELECT ubp.read_status FROM user_book_progress ubp "
            f"WHERE ubp.book_id = {_book_id} LIMIT 1"
        )
        if not result:
            check("8. booklore_read_status", 2, False,
                  f"no progress record for book_id={_book_id}")
            return
        status = result.strip().upper()
        passed = status == "READ"
        check("8. booklore_read_status", 2, passed,
              "" if passed else f"status='{status}', expected 'READ'")
    except Exception as e:
        check("8. booklore_read_status", 2, False, f"exception: {e}")


def check_9_booklore_notes_count() -> None:
    global _book_notes
    if not _book_id:
        check("9. booklore_notes_count", 2, False, "book not found")
        return
    try:
        notes_raw = ""
        for table, col in [("book_notes_v2", "note_content"), ("book_notes", "content")]:
            result = mariadb_query(
                f"SELECT {col} FROM {table} WHERE book_id = {_book_id}"
            )
            if result:
                notes_raw = result
                break
        if not notes_raw:
            check("9. booklore_notes_count", 2, False, "no notes found")
            return
        _book_notes = [l.strip() for l in notes_raw.strip().splitlines() if l.strip()]
        count_str = ""
        for table in ["book_notes_v2", "book_notes"]:
            count_str = mariadb_query(
                f"SELECT COUNT(*) FROM {table} WHERE book_id = {_book_id}"
            )
            if count_str and int(count_str) > 0:
                break
        note_count = int(count_str) if count_str else len(_book_notes)
        passed = note_count >= 3
        check("9. booklore_notes_count", 2, passed,
              f"count={note_count}" if passed else f"found {note_count} notes, need >=3")
    except Exception as e:
        check("9. booklore_notes_count", 2, False, f"exception: {e}")


def check_10_booklore_notes_corrupted_justice() -> None:
    if not _book_notes:
        check("10. booklore_notes_corrupted_justice", 2, False, "no notes available")
        return
    try:
        combined = "\n---\n".join(f"Note {i+1}: {n}" for i, n in enumerate(_book_notes))
        condition = (
            "The notes discuss historical examples of corrupted justice systems, "
            "drawing from the book 'Atrocious Judges: Lives of Judges Infamous as "
            "Tools of Tyrants and Instruments of Oppression'. The notes should contain "
            "at least 3 chapter-level observations about historical judges who abused "
            "their power, acted as tools of tyranny, or participated in unjust legal "
            "proceedings. Generic notes without historical specifics do not count."
        )
        passed, raw = llm_judge(combined, condition)
        check("10. booklore_notes_corrupted_justice", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("10. booklore_notes_corrupted_justice", 2, False, f"exception: {e}")


def check_11_siyuan_ep47_doc_exists() -> None:
    global _ep47_doc_id, _ep47_blocks
    try:
        for pattern in [
            "EP-47%Shadows of Justice",
            "EP-47%shadows of justice",
            "EP-47",
            "Shadows of Justice",
        ]:
            rows = siyuan_sql(
                f"SELECT id, content FROM blocks "
                f"WHERE type = 'd' AND content LIKE '%{pattern}%' LIMIT 5"
            )
            if rows:
                for r in rows:
                    content = r.get("content", "")
                    if "ep-47" in content.lower() or "shadows of justice" in content.lower():
                        _ep47_doc_id = r["id"]
                        _ep47_blocks = siyuan_sql(
                            f"SELECT id, type, subtype, content, markdown "
                            f"FROM blocks WHERE root_id='{_ep47_doc_id}' "
                            f"AND type != 'd' ORDER BY sort"
                        )
                        check("11. siyuan_ep47_doc_exists", 2, True,
                              f"doc='{content[:80]}'")
                        return
        all_docs = siyuan_sql(
            "SELECT content FROM blocks WHERE type = 'd' LIMIT 10"
        )
        titles = [r.get("content", "") for r in (all_docs or [])]
        check("11. siyuan_ep47_doc_exists", 2, False,
              f"no EP-47 / Shadows of Justice doc found; available: {titles[:5]}")
    except Exception as e:
        check("11. siyuan_ep47_doc_exists", 2, False, f"exception: {e}")


def check_12_siyuan_ep47_structure() -> None:
    if not _ep47_doc_id:
        check("12. siyuan_ep47_structure", 3, False, "EP-47 doc not found")
        return
    try:
        headings = [b.get("content", "") for b in _ep47_blocks if b.get("type") == "h"]
        full_content = "\n".join(
            b.get("markdown", "") or b.get("content", "")
            for b in _ep47_blocks
        )
        has_intro = False
        for h in headings:
            hl = h.lower()
            if any(kw in hl for kw in ["intro", "introduction",
                                        "引言", "开场", "导言"]):
                has_intro = True
                break
        if not has_intro:
            # Intro = text before the first heading. Blocks `ORDER BY sort` is
            # grouped by block type, not document order, so use the exported
            # markdown (true reading order) for this position-sensitive check.
            intro_lines: list[str] = []
            for line in siyuan_export_md(_ep47_doc_id).splitlines():
                if re.match(r"^#{1,6}\s+", line):
                    break
                intro_lines.append(line)
            intro_text = " ".join(intro_lines)
            if len(intro_text.strip()) > 80:
                has_intro = True

        has_comparison = False
        for h in headings:
            hl = h.lower()
            if any(kw in hl for kw in ["comparison", "thematic differ",
                                        "对比", "差异",
                                        "比较", "vigilante",
                                        "judge"]):
                has_comparison = True
                break

        condition = (
            "This document contains a structured comparison section that lists at least "
            "4 thematic differences between a film's vigilante character/justice theme and "
            "historical judges from the book 'Atrocious Judges'. The differences should be "
            "clearly enumerated (numbered, bulleted, or under sub-headings) and each one "
            "should identify a distinct thematic point of comparison."
        )
        passed_llm, raw = llm_judge(full_content[:4000], condition)

        issues = []
        if not has_intro:
            issues.append("no intro section detected")
        if not has_comparison and not passed_llm:
            issues.append("no comparison section with >=4 thematic differences")

        passed = has_intro and (has_comparison or passed_llm)
        detail = f"headings={headings[:8]}"
        if passed_llm:
            detail += "; llm confirmed >=4 differences"
        if issues:
            detail += "; issues: " + "; ".join(issues)
        check("12. siyuan_ep47_structure", 3, passed, detail)
    except Exception as e:
        check("12. siyuan_ep47_structure", 3, False, f"exception: {e}")


def check_13_siyuan_ep47_links() -> None:
    if not _ep47_doc_id:
        check("13. siyuan_ep47_links", 2, False, "EP-47 doc not found")
        return
    try:
        all_md = "\n".join(
            b.get("markdown", "") or b.get("content", "")
            for b in _ep47_blocks
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

        if has_watcharr_link and has_booklore_link:
            check("13. siyuan_ep47_links", 2, True,
                  "links to both Watcharr and Booklore entries found")
        elif url_count >= 2:
            check("13. siyuan_ep47_links", 2, True,
                  f"found {url_count} URLs (assumed Watcharr + Booklore)")
        elif url_count == 1:
            which = "Watcharr" if has_watcharr_link else (
                "Booklore" if has_booklore_link else "unknown app"
            )
            check("13. siyuan_ep47_links", 2, False,
                  f"only 1 URL found (likely {which}); need links to both apps")
        else:
            check("13. siyuan_ep47_links", 2, False,
                  "no hyperlinks to Watcharr or Booklore entries found")
    except Exception as e:
        check("13. siyuan_ep47_links", 2, False, f"exception: {e}")


# -- Main ----------------------------------------------------------------------
def main() -> None:
    check_0_input_files_exist()
    check_1_watcharr_film_exists()
    check_2_watcharr_status_watched()
    check_3_watcharr_rating_7_5()
    check_4_watcharr_review_exists()
    check_5_watcharr_review_cinematography()
    check_6_cross_modal_poster_title()
    check_7_booklore_book_exists()
    check_8_booklore_read_status()
    check_9_booklore_notes_count()
    check_10_booklore_notes_corrupted_justice()
    check_11_siyuan_ep47_doc_exists()
    check_12_siyuan_ep47_structure()
    check_13_siyuan_ep47_links()

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
