# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_060/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_060: Identify Spider-Man: No Way Home from poster, log in Watcharr, create SiYuan episode notes.

Checks: 11 weighted checks across watcharr, siyuan.
Strategy: watcharr via docker exec SQLite; siyuan via REST API /api/query/sql;
          llm_judge for review quality; llm_judge_vision for cross-modal consistency.

Required env vars:
  SERVER_HOSTNAME, WATCHARR_PORT, WATCHARR_CONTAINER, SIYUAN_PORT, SIYUAN_CONTAINER.
"""

import base64
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error

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

WATCHARR_DB = "/data/watcharr.db"
SIYUAN_API = f"http://{HOST}:{SIYUAN_PORT}"
SIYUAN_TOKEN = ""

_INPUTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs"
)

INPUT_FILES = [
    os.path.join(_INPUTS_DIR, "watcharr_poster_001.jpg"),
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


def get_siyuan_token() -> str:
    global SIYUAN_TOKEN
    if SIYUAN_TOKEN:
        return SIYUAN_TOKEN
    rc, stdout, stderr = docker_exec(
        SIYUAN_CONTAINER, "cat", "/siyuan/workspace/conf/conf.json", timeout=10,
    )
    if rc == 0 and stdout.strip():
        conf = json.loads(stdout)
        SIYUAN_TOKEN = conf.get("api", {}).get("token", "")
    return SIYUAN_TOKEN


_WATCHARR_DB_CACHE: str | None = None


def _get_watcharr_db() -> str:
    global _WATCHARR_DB_CACHE
    if _WATCHARR_DB_CACHE and os.path.exists(_WATCHARR_DB_CACHE):
        return _WATCHARR_DB_CACHE
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp.name
    tmp.close()
    r = subprocess.run(
        ["docker", "cp", f"{WATCHARR_CONTAINER}:{WATCHARR_DB}", tmp_path],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"docker cp failed (watcharr db): {r.stderr.strip()}")
    _WATCHARR_DB_CACHE = tmp_path
    return tmp_path


def watcharr_sql(query: str) -> str:
    import sqlite3 as _sqlite3
    db_path = _get_watcharr_db()
    conn = _sqlite3.connect(db_path)
    cur = conn.execute(query)
    rows = cur.fetchall()
    conn.close()
    return "\n".join("|".join("" if c is None else str(c) for c in row) for row in rows)


def siyuan_sql(stmt: str) -> list[dict]:
    payload = json.dumps({"stmt": stmt}).encode()
    headers = {"Content-Type": "application/json"}
    token = get_siyuan_token()
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(
        f"{SIYUAN_API}/api/query/sql",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"SiYuan API HTTP {e.code}: {e.read().decode()[:200]}")
    if body.get("code") != 0:
        raise RuntimeError(f"SiYuan API error: {body.get('msg', body)}")
    return body.get("data") or []


def llm_judge(content: str, condition: str, timeout: int = 30) -> tuple[bool, str]:
    api_base = os.getenv("MINDRA_BASE_URL", "https://api.mindracode.com/v1")
    api_key = os.getenv("MINDRA_API_KEY", "")
    prompt = (
        f"Does the following content satisfy this condition?\n"
        f"Condition: {condition}\n\n"
        f"Content:\n{content}\n\n"
        f"Answer only YES or NO."
    )
    body = json.dumps({
        "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }).encode()
    try:
        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        answer = data["choices"][0]["message"]["content"].strip().upper()
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
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "webp": "image/webp"}.get(ext, "image/jpeg")

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
    body = json.dumps({
        "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 512,
    }).encode()
    try:
        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        answer = data["choices"][0]["message"]["content"].strip().upper()
        return answer.startswith("YES"), answer
    except Exception as e:
        return False, f"llm_judge_vision error: {e}"


# ── Cached state ──────────────────────────────────────────────────────────────
_watcharr_row: dict = {}
_ep_doc_id: str = ""
_director_doc_id: str = ""
_input_files_ok: bool = False


# ── Individual checks ─────────────────────────────────────────────────────────
def check_0_input_files_exist():
    global _input_files_ok
    try:
        missing = [p for p in INPUT_FILES if not os.path.isfile(p)]
        if missing:
            check("0. input_files_exist", 1, False, "missing: " + ", ".join(missing))
        else:
            _input_files_ok = True
            check("0. input_files_exist", 1, True)
    except Exception as e:
        check("0. input_files_exist", 1, False, f"exception: {e}")


def check_1_watcharr_spiderman_exists():
    global _watcharr_row
    try:
        rows = watcharr_sql(
            "SELECT w.status, w.rating, w.thoughts, c.title "
            "FROM watcheds w JOIN contents c ON w.content_id = c.id "
            "JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
            "WHERE LOWER(c.title) LIKE '%spider-man%no way home%' "
            "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC LIMIT 1;"
        )
        if not rows:
            rows = watcharr_sql(
                "SELECT w.status, w.rating, w.thoughts, c.title "
                "FROM watcheds w JOIN contents c ON w.content_id = c.id "
                "JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
                "WHERE LOWER(c.title) LIKE '%spider%man%' AND LOWER(c.title) LIKE '%no way%' "
                "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC LIMIT 1;"
            )
        if not rows:
            rows = watcharr_sql(
                "SELECT w.status, w.rating, w.thoughts, c.title "
                "FROM watcheds w JOIN contents c ON w.content_id = c.id "
                "JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
                "WHERE LOWER(c.title) LIKE '%spider%man%' "
                "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC LIMIT 1;"
            )
        if not rows:
            check("1. watcharr_spiderman_exists", 2, False, "no watched entry for Spider-Man: No Way Home")
            return
        parts = rows.split("\n")[0].split("|", 3)
        _watcharr_row = {
            "status": parts[0] if len(parts) > 0 else "",
            "rating": parts[1] if len(parts) > 1 else "",
            "thoughts": parts[2] if len(parts) > 2 else "",
            "title": parts[3] if len(parts) > 3 else "",
        }
        check("1. watcharr_spiderman_exists", 2, True, f"found: {_watcharr_row['title']}")
    except Exception as e:
        check("1. watcharr_spiderman_exists", 2, False, f"exception: {e}")


def check_2_watcharr_status_watched():
    try:
        if not _watcharr_row:
            check("2. watcharr_status_watched", 2, False, "no watched row available")
            return
        status = _watcharr_row["status"]
        passed = status in ("FINISHED", "WATCHED", "COMPLETED")
        detail = "" if passed else f"status is '{status}', expected 'FINISHED'"
        check("2. watcharr_status_watched", 2, passed, detail)
    except Exception as e:
        check("2. watcharr_status_watched", 2, False, f"exception: {e}")


def check_3_watcharr_rating_7():
    try:
        if not _watcharr_row:
            check("3. watcharr_rating_7", 1, False, "no watched row available")
            return
        raw = _watcharr_row["rating"]
        rating = float(raw)
        passed = abs(rating - 7.0) < 0.5
        detail = "" if passed else f"rating is {rating}, expected 7.0"
        check("3. watcharr_rating_7", 1, passed, detail)
    except Exception as e:
        check("3. watcharr_rating_7", 1, False, f"exception: {e}")


def check_4_watcharr_review_length():
    try:
        if not _watcharr_row:
            check("4. watcharr_review_length", 1, False, "no watched row available")
            return
        thoughts = _watcharr_row["thoughts"]
        word_count = len(thoughts.split())
        passed = 40 <= word_count <= 150
        detail = "" if passed else f"review is {word_count} words, expected 50-100 (tolerance 40-150)"
        check("4. watcharr_review_length", 1, passed, detail)
    except Exception as e:
        check("4. watcharr_review_length", 1, False, f"exception: {e}")


def check_5_watcharr_review_analytical():
    try:
        if not _watcharr_row:
            check("5. watcharr_review_analytical", 2, False, "no watched row available")
            return
        thoughts = _watcharr_row["thoughts"]
        if len(thoughts) < 20:
            check("5. watcharr_review_analytical", 2, False, "review too short for analysis")
            return
        condition = (
            "The review contains substantive analysis of the film's cinematography or "
            "narrative structure — not just generic praise, a plot summary, or a one-liner."
        )
        passed, raw = llm_judge(thoughts, condition)
        check("5. watcharr_review_analytical", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("5. watcharr_review_analytical", 2, False, f"exception: {e}")


def check_6_cross_modal_poster_matches():
    try:
        if not _input_files_ok:
            check("6. cross_modal_poster_matches", 2, False, "skipped: input file missing")
            return
        if not _watcharr_row:
            check("6. cross_modal_poster_matches", 2, False, "no watched row to compare against poster")
            return
        title = _watcharr_row["title"]
        thoughts = _watcharr_row["thoughts"]
        if not thoughts.strip():
            check("6. cross_modal_poster_matches", 2, False,
                  "no review recorded to compare against poster")
            return
        recorded = f"Title: {title}. Review: {thoughts[:200]}"
        condition = (
            "The movie poster shown is for Spider-Man: No Way Home, and the recorded "
            "title and review correctly identify this film based on the visual elements "
            "in the poster."
        )
        passed, raw = llm_judge_vision(INPUT_FILES[0], recorded, condition)
        check("6. cross_modal_poster_matches", 2, passed, f"llm_judge_vision: {raw}")
    except Exception as e:
        check("6. cross_modal_poster_matches", 2, False, f"exception: {e}")


def check_7_siyuan_ep_doc_exists():
    global _ep_doc_id
    try:
        rows = siyuan_sql(
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND content LIKE '%EP-Quick%' LIMIT 5;"
        )
        matched = None
        for r in rows:
            c = (r.get("content") or "").lower()
            if "ep-quick" in c and "modern mythmaking" in c:
                matched = r
                break
        if not matched:
            for r in rows:
                c = (r.get("content") or "").lower()
                if "ep-quick" in c:
                    matched = r
                    break
        if matched:
            _ep_doc_id = matched["id"]
            check("7. siyuan_ep_doc_exists", 2, True, f"doc: {matched.get('content', '')[:80]}")
        else:
            check("7. siyuan_ep_doc_exists", 2, False, "no document matching 'EP-Quick: Modern Mythmaking'")
    except Exception as e:
        check("7. siyuan_ep_doc_exists", 2, False, f"exception: {e}")


def check_8_siyuan_ep_intro_length():
    try:
        if not _ep_doc_id:
            check("8. siyuan_ep_intro_length", 2, False, "EP-Quick doc not found")
            return
        rows = siyuan_sql(
            f"SELECT content, markdown FROM blocks WHERE root_id = '{_ep_doc_id}' "
            f"AND type IN ('p', 'h', 'i') ORDER BY sort;"
        )
        all_text = " ".join(
            (r.get("content") or "") for r in rows
        )
        char_count = len(all_text.strip())
        passed = char_count >= 100
        detail = "" if passed else f"intro content is {char_count} chars, need >= 100"
        check("8. siyuan_ep_intro_length", 2, passed, detail)
    except Exception as e:
        check("8. siyuan_ep_intro_length", 2, False, f"exception: {e}")


def check_9_siyuan_director_doc_exists():
    global _director_doc_id
    try:
        rows = siyuan_sql(
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND content LIKE '%Jon Watts%' LIMIT 5;"
        )
        matched = None
        for r in rows:
            c = (r.get("content") or "").lower()
            if "jon watts" in c:
                matched = r
                break
        if not matched:
            rows = siyuan_sql(
                "SELECT id, content FROM blocks WHERE type = 'd' "
                "AND content LIKE '%Director Profile%' LIMIT 5;"
            )
            for r in rows:
                c = (r.get("content") or "").lower()
                if "director profile" in c:
                    matched = r
                    break
        if matched:
            _director_doc_id = matched["id"]
            check("9. siyuan_director_doc_exists", 2, True, f"doc: {matched.get('content', '')[:80]}")
        else:
            check("9. siyuan_director_doc_exists", 2, False,
                  "no 'Director Profile - Jon Watts' document found")
    except Exception as e:
        check("9. siyuan_director_doc_exists", 2, False, f"exception: {e}")


def check_10_siyuan_bidirectional_link():
    try:
        if not _ep_doc_id or not _director_doc_id:
            missing = []
            if not _ep_doc_id:
                missing.append("EP-Quick doc")
            if not _director_doc_id:
                missing.append("director doc")
            check("10. siyuan_bidirectional_link", 2, False, f"missing: {', '.join(missing)}")
            return
        forward = siyuan_sql(
            f"SELECT id FROM refs WHERE "
            f"root_id = '{_ep_doc_id}' AND def_block_root_id = '{_director_doc_id}' "
            f"LIMIT 1;"
        )
        backward = siyuan_sql(
            f"SELECT id FROM refs WHERE "
            f"root_id = '{_director_doc_id}' AND def_block_root_id = '{_ep_doc_id}' "
            f"LIMIT 1;"
        )
        has_forward = len(forward) > 0
        has_backward = len(backward) > 0
        if has_forward and has_backward:
            check("10. siyuan_bidirectional_link", 2, True)
        elif has_forward or has_backward:
            direction = "EP->Director" if has_forward else "Director->EP"
            check("10. siyuan_bidirectional_link", 2, True,
                  f"unidirectional ref {direction} found; SiYuan auto-generates backlinks")
        else:
            check("10. siyuan_bidirectional_link", 2, False,
                  "no refs between EP-Quick and director doc")
    except Exception as e:
        check("10. siyuan_bidirectional_link", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()
    check_1_watcharr_spiderman_exists()
    check_2_watcharr_status_watched()
    check_3_watcharr_rating_7()
    check_4_watcharr_review_length()
    check_5_watcharr_review_analytical()
    check_6_cross_modal_poster_matches()
    check_7_siyuan_ep_doc_exists()
    check_8_siyuan_ep_intro_length()
    check_9_siyuan_director_doc_exists()
    check_10_siyuan_bidirectional_link()

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
