# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_068/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_068: Log multiverse film from poster in Watcharr + create EP-45 script in SiYuan.

Checks: 12 weighted checks across watcharr, siyuan.
Strategy: host-side SQLite via docker cp (watcharr) + REST API (siyuan)

Required env vars:
  SERVER_HOSTNAME, WATCHARR_PORT, WATCHARR_CONTAINER, SIYUAN_PORT, SIYUAN_CONTAINER
"""

import os
import sys
import subprocess
import json
import re
import base64
import shutil
import sqlite3
import tempfile

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

WATCHARR_PORT = os.getenv("WATCHARR_PORT")
WATCHARR_CONTAINER = os.getenv("WATCHARR_CONTAINER")
SIYUAN_PORT = os.getenv("SIYUAN_PORT")
SIYUAN_CONTAINER = os.getenv("SIYUAN_CONTAINER")

for var_name, var_val in [
    ("WATCHARR_PORT", WATCHARR_PORT),
    ("WATCHARR_CONTAINER", WATCHARR_CONTAINER),
    ("SIYUAN_PORT", SIYUAN_PORT),
    ("SIYUAN_CONTAINER", SIYUAN_CONTAINER),
]:
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
        sys.exit(1)

SIYUAN_BASE = f"http://{HOST}:{SIYUAN_PORT}"
WATCHARR_DB = "/data/watcharr.db"

_INPUTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs"
)

INPUT_FILES = [
    os.path.join(_INPUTS_DIR, "watcharr_poster_001.jpg"),
]

EXPECTED_FILM_TITLE = "Spider-Man: No Way Home"
EXPECTED_RATING = 7.5
EXPECTED_EP_TITLE = "EP-45: The Multiverse Narrative"
EXPECTED_NOTEBOOK = "Podcast Scripts"
EXPECTED_HEADINGS = [
    "Episode Intro",
    "Core Arguments",
    "Representative Scene Analysis",
    "Closing Recommendation",
]

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


def watcharr_sql(query: str) -> str:
    """Run a query against the Watcharr SQLite DB from the host.

    The mw-watcharr image ships no sqlite3 CLI and has no network access to
    install one, so we `docker cp` the database (plus WAL/SHM sidecars) to a
    host temp dir and query it with Python's sqlite3 module. Returns a string
    in the same format as `sqlite3 -json`: a JSON array of objects keyed by
    column name ("[]" for an empty result).
    """
    tmpdir = tempfile.mkdtemp(prefix="watcharr_db_")
    try:
        local_db = os.path.join(tmpdir, "watcharr.db")
        for suffix in ("", "-wal", "-shm"):
            r = subprocess.run(
                ["docker", "cp", f"{WATCHARR_CONTAINER}:{WATCHARR_DB}{suffix}",
                 f"{local_db}{suffix}"],
                capture_output=True, text=True, timeout=30,
            )
            if suffix == "" and r.returncode != 0:
                raise RuntimeError(
                    f"docker cp {WATCHARR_CONTAINER}:{WATCHARR_DB} failed: "
                    f"{(r.stderr or r.stdout).strip()}"
                )
        conn = sqlite3.connect(local_db)
        try:
            cur = conn.execute(query)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        except sqlite3.Error as e:
            raise RuntimeError(f"sqlite3 query failed: {e}") from e
        finally:
            conn.close()
        return json.dumps([dict(zip(cols, row)) for row in rows])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def siyuan_login() -> str:
    auth_code_rc, auth_code_out, _ = docker_exec(
        SIYUAN_CONTAINER,
        "sh", "-c",
        "cat /siyuan/workspace/conf/conf.json 2>/dev/null || cat /root/.siyuan/conf.json 2>/dev/null",
    )
    auth_code = ""
    if auth_code_rc == 0 and auth_code_out.strip():
        try:
            conf = json.loads(auth_code_out.strip())
            auth_code = conf.get("accessAuthCode", "")
        except json.JSONDecodeError:
            pass

    if not auth_code:
        return ""

    resp = requests.post(
        f"{SIYUAN_BASE}/api/system/loginAuth",
        json={"authCode": auth_code},
        timeout=10,
    )
    cookies = resp.cookies
    if "siyuan" in cookies:
        return cookies["siyuan"]

    set_cookie = resp.headers.get("Set-Cookie", "")
    m = re.search(r"siyuan=([^;]+)", set_cookie)
    if m:
        return m.group(1)
    return ""


_siyuan_session = None


def siyuan_api(endpoint: str, payload: dict | None = None) -> dict:
    global _siyuan_session
    if _siyuan_session is None:
        _siyuan_session = siyuan_login()

    cookies = {"siyuan": _siyuan_session} if _siyuan_session else {}
    resp = requests.post(
        f"{SIYUAN_BASE}{endpoint}",
        json=payload or {},
        cookies=cookies,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def siyuan_sql(stmt: str) -> list[dict]:
    result = siyuan_api("/api/query/sql", {"stmt": stmt})
    return result.get("data") or []


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
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            },
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

    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp",
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
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": 512,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        return answer.startswith("YES"), answer
    except Exception as e:
        return False, f"llm_judge_vision error: {e}"


# ── Individual checks ─────────────────────────────────────────────────────────
def check_0_input_files_exist() -> None:
    missing = [p for p in INPUT_FILES if not os.path.isfile(p)]
    if missing:
        check("0. input_files_exist", 1, False, "missing: " + ", ".join(missing))
    else:
        check("0. input_files_exist", 1, True)


def check_1_watcharr_film_exists() -> None:
    try:
        rows_json = watcharr_sql(
            f"SELECT id, title, tmdb_id FROM contents "
            f"WHERE title LIKE '%Spider-Man%No Way Home%' AND type='movie';"
        )
        rows = json.loads(rows_json) if rows_json else []
        if rows:
            check("1. watcharr_film_exists", 1, True, f"content_id={rows[0]['id']}")
        else:
            rows_json2 = watcharr_sql(
                "SELECT id, title FROM contents WHERE title LIKE '%Spider%Man%' AND type='movie';"
            )
            rows2 = json.loads(rows_json2) if rows_json2 else []
            if rows2:
                check("1. watcharr_film_exists", 1, True, f"found as '{rows2[0]['title']}'")
            else:
                check("1. watcharr_film_exists", 1, False, "Spider-Man: No Way Home not in contents")
    except Exception as e:
        check("1. watcharr_film_exists", 1, False, f"exception: {e}")


def _get_watched_row() -> dict | None:
    rows_json = watcharr_sql(
        "SELECT w.id, w.status, w.rating, w.thoughts, c.title "
        "FROM watcheds w JOIN contents c ON w.content_id = c.id "
        "JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
        "WHERE (c.title LIKE '%Spider-Man%No Way Home%' OR c.title LIKE '%Spider%Man%Way%Home%') "
        "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC;"
    )
    rows = json.loads(rows_json) if rows_json else []
    return rows[0] if rows else None


def check_2_watcharr_watched_status() -> None:
    try:
        row = _get_watched_row()
        if not row:
            check("2. watcharr_watched_status", 2, False, "no watched entry for the film")
            return
        status = row.get("status", "")
        passed = status == "FINISHED"
        detail = f"status={status}" if not passed else ""
        check("2. watcharr_watched_status", 2, passed, detail)
    except Exception as e:
        check("2. watcharr_watched_status", 2, False, f"exception: {e}")


def check_3_watcharr_rating() -> None:
    try:
        row = _get_watched_row()
        if not row:
            check("3. watcharr_rating", 2, False, "no watched entry")
            return
        rating = row.get("rating")
        if rating is None:
            check("3. watcharr_rating", 2, False, "rating is NULL")
            return
        rating_f = float(rating)
        passed = abs(rating_f - EXPECTED_RATING) < 0.01
        detail = "" if passed else f"expected {EXPECTED_RATING}, got {rating_f}"
        check("3. watcharr_rating", 2, passed, detail)
    except Exception as e:
        check("3. watcharr_rating", 2, False, f"exception: {e}")


def check_4_watcharr_review_length() -> None:
    try:
        row = _get_watched_row()
        if not row:
            check("4. watcharr_review_length", 1, False, "no watched entry")
            return
        thoughts = (row.get("thoughts") or "").strip()
        if not thoughts:
            check("4. watcharr_review_length", 1, False, "thoughts field is empty")
            return
        word_count = len(thoughts.split())
        passed = 40 <= word_count <= 120
        detail = f"{word_count} words" if not passed else f"{word_count} words"
        check("4. watcharr_review_length", 1, passed, detail)
    except Exception as e:
        check("4. watcharr_review_length", 1, False, f"exception: {e}")


def check_5_cross_modal_watcharr_review() -> None:
    if not os.path.isfile(INPUT_FILES[0]):
        check("5. cross_modal_watcharr_review", 2, False, "skipped: input file missing")
        return
    try:
        row = _get_watched_row()
        if not row:
            check("5. cross_modal_watcharr_review", 2, False, "no watched entry")
            return
        title = row.get("title", "")
        thoughts = (row.get("thoughts") or "").strip()
        recorded = f"Title: {title}. Review: {thoughts}"
        passed, resp = llm_judge_vision(
            INPUT_FILES[0],
            recorded,
            "The film title recorded by the agent matches the film shown in the movie poster, "
            "and the review text is relevant to this specific film.",
        )
        check("5. cross_modal_watcharr_review", 2, passed, resp if not passed else "")
    except Exception as e:
        check("5. cross_modal_watcharr_review", 2, False, f"exception: {e}")


def check_6_watcharr_review_quality() -> None:
    try:
        row = _get_watched_row()
        if not row:
            check("6. watcharr_review_quality", 2, False, "no watched entry")
            return
        thoughts = (row.get("thoughts") or "").strip()
        if not thoughts:
            check("6. watcharr_review_quality", 2, False, "review is empty")
            return
        passed, resp = llm_judge(
            thoughts,
            "The review is an analytical review of a film (specifically about visual effects and/or "
            "narrative pacing). It should NOT be generic emotional praise like 'amazing movie' or "
            "'heartwarming'. It must discuss specific filmmaking aspects such as CGI, VFX, "
            "cinematography, pacing, story structure, or multiverse narrative techniques.",
        )
        check("6. watcharr_review_quality", 2, passed, resp if not passed else "")
    except Exception as e:
        check("6. watcharr_review_quality", 2, False, f"exception: {e}")


def check_7_siyuan_podcast_notebook() -> None:
    try:
        result = siyuan_api("/api/notebook/lsNotebooks")
        notebooks = result.get("data", {}).get("notebooks", [])
        found = any(
            "podcast" in nb.get("name", "").lower() and "script" in nb.get("name", "").lower()
            for nb in notebooks
        )
        if not found:
            found = any(
                nb.get("name", "").strip() == EXPECTED_NOTEBOOK
                for nb in notebooks
            )
        if not found:
            names = [nb.get("name", "") for nb in notebooks]
            check("7. siyuan_podcast_notebook", 1, False, f"not found among: {names[:10]}")
        else:
            check("7. siyuan_podcast_notebook", 1, True)
    except Exception as e:
        check("7. siyuan_podcast_notebook", 1, False, f"exception: {e}")


def check_8_siyuan_ep45_document() -> None:
    try:
        rows = siyuan_sql(
            "SELECT id, content, hpath, box FROM blocks WHERE type='d' "
            "AND content LIKE '%EP-45%Multiverse%Narrative%'"
        )
        if not rows:
            rows = siyuan_sql(
                "SELECT id, content, hpath, box FROM blocks WHERE type='d' "
                "AND content LIKE '%EP-45%'"
            )
        if rows:
            hpath = rows[0].get("hpath", "")
            in_podcast = "podcast" in hpath.lower() or "script" in hpath.lower()
            detail = f"hpath={hpath}"
            if not in_podcast:
                detail += " (WARNING: not under Podcast Scripts notebook)"
            check("8. siyuan_ep45_document", 2, True, detail)
        else:
            check("8. siyuan_ep45_document", 2, False, "EP-45 document not found")
    except Exception as e:
        check("8. siyuan_ep45_document", 2, False, f"exception: {e}")


def check_9_siyuan_ep45_headings() -> None:
    try:
        ep_rows = siyuan_sql(
            "SELECT id FROM blocks WHERE type='d' "
            "AND content LIKE '%EP-45%Multiverse%Narrative%'"
        )
        if not ep_rows:
            ep_rows = siyuan_sql(
                "SELECT id FROM blocks WHERE type='d' AND content LIKE '%EP-45%'"
            )
        if not ep_rows:
            check("9. siyuan_ep45_headings", 2, False, "EP-45 document not found")
            return

        doc_id = ep_rows[0]["id"]
        heading_rows = siyuan_sql(
            f"SELECT content FROM blocks WHERE type='h' AND root_id='{doc_id}'"
        )
        found_headings = [r["content"] for r in heading_rows]

        missing = []
        for expected_h in EXPECTED_HEADINGS:
            matched = any(
                expected_h.lower() in h.lower()
                for h in found_headings
            )
            if not matched:
                missing.append(expected_h)

        if missing:
            check(
                "9. siyuan_ep45_headings", 2, False,
                f"missing: {missing}; found: {found_headings}"
            )
        else:
            check("9. siyuan_ep45_headings", 2, True)
    except Exception as e:
        check("9. siyuan_ep45_headings", 2, False, f"exception: {e}")


def check_10_siyuan_director_profile_doc() -> None:
    try:
        rows = siyuan_sql(
            "SELECT id, content, hpath FROM blocks WHERE type='d' "
            "AND content LIKE '%Director Profile%'"
        )
        if rows:
            check("10. siyuan_director_profile_doc", 1, True, f"found: {rows[0]['content']}")
        else:
            check("10. siyuan_director_profile_doc", 1, False, "no 'Director Profile - *' document")
    except Exception as e:
        check("10. siyuan_director_profile_doc", 1, False, f"exception: {e}")


def check_11_siyuan_bidirectional_link() -> None:
    try:
        ep_rows = siyuan_sql(
            "SELECT id FROM blocks WHERE type='d' "
            "AND content LIKE '%EP-45%Multiverse%Narrative%'"
        )
        if not ep_rows:
            ep_rows = siyuan_sql(
                "SELECT id FROM blocks WHERE type='d' AND content LIKE '%EP-45%'"
            )
        if not ep_rows:
            check("11. siyuan_bidirectional_link", 3, False, "EP-45 document not found")
            return

        dir_rows = siyuan_sql(
            "SELECT id, content FROM blocks WHERE type='d' "
            "AND content LIKE '%Director Profile%'"
        )
        if not dir_rows:
            check("11. siyuan_bidirectional_link", 3, False, "Director Profile document not found")
            return

        ep_doc_id = ep_rows[0]["id"]
        dir_doc_id = dir_rows[0]["id"]

        refs = siyuan_sql(
            f"SELECT id, block_id, def_block_id, def_block_root_id, root_id "
            f"FROM refs WHERE root_id='{ep_doc_id}' AND def_block_root_id='{dir_doc_id}'"
        )

        intro_headings = siyuan_sql(
            f"SELECT id FROM blocks WHERE type='h' AND root_id='{ep_doc_id}' "
            f"AND content LIKE '%Intro%'"
        )

        if refs:
            if intro_headings:
                intro_id = intro_headings[0]["id"]
                intro_refs = siyuan_sql(
                    f"SELECT id FROM refs WHERE root_id='{ep_doc_id}' "
                    f"AND def_block_root_id='{dir_doc_id}'"
                )
                if intro_refs:
                    check("11. siyuan_bidirectional_link", 3, True, "link from EP-45 to Director Profile found")
                else:
                    check("11. siyuan_bidirectional_link", 3, True, "link exists in EP-45 doc (not strictly in Intro section)")
            else:
                check("11. siyuan_bidirectional_link", 3, True, "link found (no Intro heading to verify placement)")
        else:
            all_refs = siyuan_sql(
                f"SELECT def_block_root_id, root_id FROM refs "
                f"WHERE root_id='{ep_doc_id}' OR def_block_root_id='{ep_doc_id}'"
            )
            check(
                "11. siyuan_bidirectional_link", 3, False,
                f"no ref from EP-45 to Director Profile; ep_doc={ep_doc_id}, dir_doc={dir_doc_id}; "
                f"all refs from/to EP-45: {len(all_refs)}"
            )
    except Exception as e:
        check("11. siyuan_bidirectional_link", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()
    check_1_watcharr_film_exists()
    check_2_watcharr_watched_status()
    check_3_watcharr_rating()
    check_4_watcharr_review_length()
    check_5_cross_modal_watcharr_review()
    check_6_watcharr_review_quality()
    check_7_siyuan_podcast_notebook()
    check_8_siyuan_ep45_document()
    check_9_siyuan_ep45_headings()
    check_10_siyuan_director_profile_doc()
    check_11_siyuan_bidirectional_link()

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
