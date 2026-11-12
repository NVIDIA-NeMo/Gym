# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_067/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_067: Identify film from poster, track in Watcharr, create podcast script in SiYuan

Checks: 11 weighted checks across watcharr, siyuan.
Strategy: docker exec SQLite for watcharr, SiYuan REST API for siyuan

Required env vars:
  SERVER_HOSTNAME, WATCHARR_PORT, WATCHARR_CONTAINER, SIYUAN_PORT, SIYUAN_CONTAINER
"""

import os
import sys
import re
import json
import subprocess
import requests
import base64

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

_INPUTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs"
)

INPUT_FILES: list[str] = [
    os.path.join(_INPUTS_DIR, "watcharr_poster_066.jpg"),
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


_WATCHARR_DB_CACHE: str | None = None
_WATCHARR_DB_PATH = "/data/watcharr.db"


def _get_watcharr_db() -> str:
    global _WATCHARR_DB_CACHE
    if _WATCHARR_DB_CACHE and os.path.exists(_WATCHARR_DB_CACHE):
        return _WATCHARR_DB_CACHE
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp.name
    tmp.close()
    r = subprocess.run(
        ["docker", "cp", f"{WATCHARR_CONTAINER}:{_WATCHARR_DB_PATH}", tmp_path],
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


def get_siyuan_token() -> str:
    rc, stdout, stderr = docker_exec(
        SIYUAN_CONTAINER, "cat", "/siyuan/workspace/conf/conf.json"
    )
    if rc != 0:
        return ""
    try:
        conf = json.loads(stdout)
        return conf.get("api", {}).get("token", "")
    except (json.JSONDecodeError, KeyError):
        return ""


def siyuan_sql(token: str, stmt: str) -> list[dict]:
    resp = requests.post(
        f"{SIYUAN_BASE}/api/query/sql",
        headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
        json={"stmt": stmt},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"SiYuan SQL error: {data.get('msg')}")
    return data.get("data") or []


def siyuan_export_md(token: str, doc_id: str) -> str:
    resp = requests.post(
        f"{SIYUAN_BASE}/api/export/exportMdContent",
        headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
        json={"id": doc_id},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"SiYuan export error: {data.get('msg')}")
    return data.get("data", {}).get("content", "")


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
            json={"model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"), "messages": [{"role": "user", "content": prompt}],
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

    ext = os.path.splitext(image_path)[1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")

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
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
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


def check_1_watcharr_batman_content() -> int | None:
    """Nightmare Alley exists in watcharr contents table."""
    try:
        row = watcharr_sql(
            "SELECT id, title FROM contents WHERE title = 'Nightmare Alley' AND type = 'movie' LIMIT 1"
        )
        if row:
            content_id = row.split("|")[0]
            check("1. watcharr_batman_content", 1, True, f"content_id={content_id}")
            return int(content_id)
        row = watcharr_sql(
            "SELECT id, title FROM contents WHERE title LIKE '%Nightmare%' AND type = 'movie'"
        )
        check("1. watcharr_batman_content", 1, False,
              f"'Nightmare Alley' not found; similar: {row[:200]}")
        return None
    except Exception as e:
        check("1. watcharr_batman_content", 1, False, f"exception: {e}")
        return None


def check_2_watcharr_watched_status(content_id: int | None) -> None:
    """Nightmare Alley is marked as watched (FINISHED) in watcharr."""
    if content_id is None:
        check("2. watcharr_watched_status", 2, False, "skipped: content not found")
        return
    try:
        status = watcharr_sql(
            f"SELECT w.status FROM watcheds w "
            f"JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
            f"WHERE w.content_id = {content_id} AND w.deleted_at IS NULL "
            f"ORDER BY w.updated_at DESC LIMIT 1"
        )
        if status.upper() == "FINISHED":
            check("2. watcharr_watched_status", 2, True)
        else:
            check("2. watcharr_watched_status", 2, False,
                  f"expected FINISHED, got '{status}'")
    except Exception as e:
        check("2. watcharr_watched_status", 2, False, f"exception: {e}")


def check_3_watcharr_rating(content_id: int | None) -> None:
    """Nightmare Alley has a rating of 8.5 in watcharr."""
    if content_id is None:
        check("3. watcharr_rating", 2, False, "skipped: content not found")
        return
    try:
        raw = watcharr_sql(
            f"SELECT w.rating FROM watcheds w "
            f"JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
            f"WHERE w.content_id = {content_id} AND w.deleted_at IS NULL "
            f"ORDER BY w.updated_at DESC LIMIT 1"
        )
        if not raw:
            check("3. watcharr_rating", 2, False, "no watched entry found")
            return
        rating = float(raw)
        if abs(rating - 8.5) < 0.01:
            check("3. watcharr_rating", 2, True, f"rating={rating}")
        else:
            check("3. watcharr_rating", 2, False, f"expected 8.5, got {rating}")
    except Exception as e:
        check("3. watcharr_rating", 2, False, f"exception: {e}")


def check_4_watcharr_review_length(content_id: int | None) -> str:
    """Nightmare Alley has a review (thoughts) of 50-100 words."""
    if content_id is None:
        check("4. watcharr_review_length", 2, False, "skipped: content not found")
        return ""
    try:
        thoughts = watcharr_sql(
            f"SELECT w.thoughts FROM watcheds w "
            f"JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
            f"WHERE w.content_id = {content_id} AND w.deleted_at IS NULL "
            f"ORDER BY w.updated_at DESC LIMIT 1"
        )
        if not thoughts:
            check("4. watcharr_review_length", 2, False, "no review text found")
            return ""
        word_count = len(thoughts.split())
        if 50 <= word_count <= 100:
            check("4. watcharr_review_length", 2, True,
                  f"{word_count} words (50-100 range)")
        elif 40 <= word_count <= 120:
            check("4. watcharr_review_length", 2, True,
                  f"{word_count} words (within tolerance)")
        else:
            check("4. watcharr_review_length", 2, False,
                  f"{word_count} words, expected 50-100")
        return thoughts
    except Exception as e:
        check("4. watcharr_review_length", 2, False, f"exception: {e}")
        return ""


def check_5_watcharr_review_cinematography(review: str) -> None:
    """Review discusses dark, noir-inspired cinematography."""
    if not review:
        check("5. watcharr_review_cinematography", 2, False,
              "skipped: no review text")
        return
    try:
        passed, answer = llm_judge(
            review,
            "The review discusses the cinematography of a film, focusing on dark, "
            "noir-inspired visual elements. It should mention concepts like "
            "lighting, shadows, color palette, visual style, or similar "
            "cinematographic qualities."
        )
        check("5. watcharr_review_cinematography", 2, passed,
              f"llm_judge: {answer}")
    except Exception as e:
        check("5. watcharr_review_cinematography", 2, False, f"exception: {e}")


def check_6_cross_modal_poster_title(content_id: int | None) -> None:
    """Cross-modal: poster image matches the film the agent logged in Watcharr."""
    if not os.path.isfile(INPUT_FILES[0]):
        check("6. cross_modal_poster_title", 2, False,
              "skipped: input file missing")
        return
    if content_id is None:
        check("6. cross_modal_poster_title", 2, False,
              "skipped: content not found")
        return
    try:
        title = watcharr_sql(
            f"SELECT c.title FROM watcheds w JOIN contents c ON w.content_id = c.id "
            f"JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
            f"WHERE w.content_id = {content_id} "
            f"AND w.deleted_at IS NULL ORDER BY w.updated_at DESC LIMIT 1"
        )
        if not title:
            check("6. cross_modal_poster_title", 2, False,
                  "no watched entry by the logged-in user to compare against poster")
            return
        passed, answer = llm_judge_vision(
            INPUT_FILES[0],
            title,
            "The movie poster shown is for the film 'Nightmare Alley'. The title "
            "or visual elements (characters, logo, style) in the poster "
            "correspond to the 2021 film 'Nightmare Alley' directed by "
            "Guillermo del Toro.",
        )
        check("6. cross_modal_poster_title", 2, passed,
              f"llm_judge_vision: {answer}")
    except Exception as e:
        check("6. cross_modal_poster_title", 2, False, f"exception: {e}")


def check_7_siyuan_ep45_exists(token: str) -> str | None:
    """SiYuan document 'EP-46: The Noir Carnival' exists."""
    try:
        rows = siyuan_sql(
            token,
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND content LIKE '%EP-46%Noir Carnival%'"
        )
        if not rows:
            rows = siyuan_sql(
                token,
                "SELECT id, content FROM blocks WHERE type = 'd' "
                "AND content LIKE '%EP-46%'"
            )
        if rows:
            exact = [r for r in rows
                     if "EP-46" in r.get("content", "")
                     and "Noir Carnival" in r.get("content", "")]
            if exact:
                check("7. siyuan_ep45_exists", 2, True,
                      f"title='{exact[0]['content']}'")
                return exact[0]["id"]
            check("7. siyuan_ep45_exists", 2, True,
                  f"title='{rows[0]['content']}' (partial match)")
            return rows[0]["id"]
        check("7. siyuan_ep45_exists", 2, False,
              "no document matching 'EP-46: The Noir Carnival' found")
        return None
    except Exception as e:
        check("7. siyuan_ep45_exists", 2, False, f"exception: {e}")
        return None


def check_8_siyuan_intro_length(token: str, doc_id: str | None) -> None:
    """EP-46 document has an introduction of >= 100 characters."""
    if doc_id is None:
        check("8. siyuan_intro_length", 2, False, "skipped: EP-46 doc not found")
        return
    try:
        md_content = siyuan_export_md(token, doc_id)
        lines = md_content.strip().split("\n")
        body_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped:
                body_lines.append(stripped)
        body_text = " ".join(body_lines)
        length = len(body_text)
        if length >= 100:
            check("8. siyuan_intro_length", 2, True, f"{length} chars (>=100)")
        else:
            check("8. siyuan_intro_length", 2, False,
                  f"{length} chars, need >=100")
    except Exception as e:
        check("8. siyuan_intro_length", 2, False, f"exception: {e}")


def check_9_siyuan_director_doc(token: str) -> str | None:
    """SiYuan document 'Director - Guillermo del Toro' exists."""
    try:
        rows = siyuan_sql(
            token,
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND content LIKE '%Director%del Toro%'"
        )
        if not rows:
            rows = siyuan_sql(
                token,
                "SELECT id, content FROM blocks WHERE type = 'd' "
                "AND content LIKE '%Director%Toro%'"
            )
        if rows:
            check("9. siyuan_director_doc", 1, True,
                  f"title='{rows[0]['content']}'")
            return rows[0]["id"]
        check("9. siyuan_director_doc", 1, False,
              "no document matching 'Director - Guillermo del Toro' found")
        return None
    except Exception as e:
        check("9. siyuan_director_doc", 1, False, f"exception: {e}")
        return None


def check_10_siyuan_bidir_link(token: str, ep45_id: str | None, director_id: str | None) -> None:
    """EP-46 contains a bidirectional link to the Director - Guillermo del Toro document."""
    if ep45_id is None:
        check("10. siyuan_bidir_link", 3, False,
              "skipped: EP-46 doc not found")
        return
    try:
        md_content = siyuan_export_md(token, ep45_id)

        has_link = bool(re.search(
            r'\(\(.*\)\)|Director.*del\s*Toro|del\s*Toro',
            md_content, re.IGNORECASE
        ))

        if director_id:
            ref_rows = siyuan_sql(
                token,
                f"SELECT id, content FROM blocks WHERE root_id = '{ep45_id}' "
                f"AND (markdown LIKE '%{director_id}%' "
                f"OR markdown LIKE '%Director%del Toro%' "
                f"OR markdown LIKE '%Director - Guillermo del Toro%')"
            )
            if ref_rows:
                check("10. siyuan_bidir_link", 3, True,
                      "block-level link to director doc found")
                return

        ref_rows = siyuan_sql(
            token,
            f"SELECT def_block_id, content FROM refs WHERE root_id = '{ep45_id}'"
        )
        if ref_rows:
            for row in ref_rows:
                if director_id and row.get("def_block_id") == director_id:
                    check("10. siyuan_bidir_link", 3, True,
                          "ref table link to director doc found")
                    return
            if any("Director" in r.get("content", "") or "del Toro" in r.get("content", "")
                   for r in ref_rows):
                check("10. siyuan_bidir_link", 3, True,
                      "ref to director-related block found")
                return

        if has_link:
            check("10. siyuan_bidir_link", 3, True,
                  "markdown contains reference to Director - Guillermo del Toro")
            return

        check("10. siyuan_bidir_link", 3, False,
              "no bidirectional link to Director - Guillermo del Toro found in EP-46")
    except Exception as e:
        check("10. siyuan_bidir_link", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()

    # Watcharr checks
    content_id = check_1_watcharr_batman_content()
    check_2_watcharr_watched_status(content_id)
    check_3_watcharr_rating(content_id)
    review = check_4_watcharr_review_length(content_id)
    check_5_watcharr_review_cinematography(review)
    check_6_cross_modal_poster_title(content_id)

    # SiYuan checks
    token = get_siyuan_token()
    if not token:
        print("FATAL: could not read SiYuan API token from container", file=sys.stderr)
        sys.exit(1)

    ep45_id = check_7_siyuan_ep45_exists(token)
    check_8_siyuan_intro_length(token, ep45_id)
    director_id = check_9_siyuan_director_doc(token)
    check_10_siyuan_bidir_link(token, ep45_id, director_id)

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
