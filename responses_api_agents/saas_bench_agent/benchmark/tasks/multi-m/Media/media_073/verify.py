# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_073/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_073: Encanto poster → Watcharr watchlist + SiYuan outline doc.

Checks: 10 weighted checks (19 total points) across watcharr, siyuan.
Strategy: watcharr via docker exec SQLite; siyuan via REST API; llm_judge + llm_judge_vision.

Required env vars:
  SERVER_HOSTNAME, WATCHARR_PORT, WATCHARR_CONTAINER, SIYUAN_PORT, SIYUAN_CONTAINER.
"""

import base64
import json
import os
import sys
import subprocess
import urllib.request
import urllib.error

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

WATCHARR_PORT = os.getenv("WATCHARR_PORT")
WATCHARR_CONTAINER = os.getenv("WATCHARR_CONTAINER")
SIYUAN_PORT = os.getenv("SIYUAN_PORT")
SIYUAN_CONTAINER = os.getenv("SIYUAN_CONTAINER")

for _var_name, _var_val in [
    ("WATCHARR_PORT", WATCHARR_PORT),
    ("WATCHARR_CONTAINER", WATCHARR_CONTAINER),
    ("SIYUAN_PORT", SIYUAN_PORT),
    ("SIYUAN_CONTAINER", SIYUAN_CONTAINER),
]:
    if not _var_val:
        print(f"FATAL: {_var_name} not set", file=sys.stderr)
        sys.exit(1)

WATCHARR_DB = "/data/watcharr.db"
SIYUAN_API = f"http://{HOST}:{SIYUAN_PORT}"
SIYUAN_TOKEN = ""

_INPUTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs"
)

INPUT_FILES: list[str] = [
    os.path.join(_INPUTS_DIR, "watcharr_poster_004.jpg"),
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
    rc, stdout, _ = docker_exec(
        SIYUAN_CONTAINER, "cat", "/siyuan/workspace/conf/conf.json", timeout=10,
    )
    if rc == 0 and stdout.strip():
        try:
            conf = json.loads(stdout)
            SIYUAN_TOKEN = conf.get("api", {}).get("token", "")
        except json.JSONDecodeError:
            pass
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
    body = json.dumps({
        "model": os.getenv("MINDRA_MODEL", "gemini-3.0-flash-preview"),
        "messages": [{"role": "user", "content": msg_content}],
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
_animation_doc_root_id: str = ""
_genre_doc_root_id: str = ""
_input_files_ok: bool = False


# ── Individual checks ─────────────────────────────────────────────────────────
def check_0_input_files_exist() -> None:
    global _input_files_ok
    missing = [p for p in INPUT_FILES if not os.path.isfile(p)]
    if missing:
        check("0. input_files_exist", 1, False, "missing: " + ", ".join(missing))
    else:
        _input_files_ok = True
        check("0. input_files_exist", 1, True)


def check_1_watcharr_encanto_exists() -> None:
    global _watcharr_row
    try:
        rows = watcharr_sql(
            "SELECT w.status, w.rating, w.thoughts, c.title "
            "FROM watcheds w JOIN contents c ON w.content_id = c.id "
            "JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
            "WHERE LOWER(c.title) LIKE '%encanto%' "
            "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC LIMIT 1;"
        )
        if not rows:
            check("1. watcharr_encanto_exists", 2, False,
                  "no watched entry for Encanto found")
            return
        parts = rows.split("|", 3)
        _watcharr_row = {
            "status": parts[0] if len(parts) > 0 else "",
            "rating": parts[1] if len(parts) > 1 else "",
            "thoughts": parts[2] if len(parts) > 2 else "",
            "title": parts[3] if len(parts) > 3 else "",
        }
        check("1. watcharr_encanto_exists", 2, True,
              f"found: {_watcharr_row['title']}")
    except Exception as e:
        check("1. watcharr_encanto_exists", 2, False, f"exception: {e}")


def check_2_watcharr_status_planned() -> None:
    try:
        if not _watcharr_row:
            check("2. watcharr_status_planned", 2, False, "no watched row available")
            return
        status = _watcharr_row["status"]
        passed = status == "PLANNED"
        detail = "" if passed else f"status is '{status}', expected 'PLANNED' (Want to Watch)"
        check("2. watcharr_status_planned", 2, passed, detail)
    except Exception as e:
        check("2. watcharr_status_planned", 2, False, f"exception: {e}")


def check_3_cross_modal_poster_title() -> None:
    try:
        if not _input_files_ok:
            check("3. cross_modal_poster_title", 2, False, "skipped: input file missing")
            return
        if not _watcharr_row:
            check("3. cross_modal_poster_title", 2, False,
                  "no watched row to compare against poster")
            return
        title = _watcharr_row["title"]
        condition = (
            "The movie poster shown is for the animated film 'Encanto' (2021, Disney). "
            "The title visible on the poster matches the recorded value."
        )
        passed, raw = llm_judge_vision(INPUT_FILES[0], title, condition)
        check("3. cross_modal_poster_title", 2, passed, f"llm_judge_vision: {raw}")
    except Exception as e:
        check("3. cross_modal_poster_title", 2, False, f"exception: {e}")


def check_4_siyuan_animation_doc_exists() -> None:
    global _animation_doc_root_id
    try:
        rows = siyuan_sql(
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND (content LIKE '%Animation and Family%' "
            "OR content LIKE '%animation and family%' "
            "OR content LIKE '%Animation%Family Dynamics%') "
            "LIMIT 10;"
        )
        matched = None
        for r in rows:
            c = (r.get("content") or "").lower()
            if "animation" in c and "family" in c:
                matched = r
                break
        if matched:
            _animation_doc_root_id = matched["id"]
            check("4. siyuan_animation_doc_exists", 2, True,
                  f"doc: {matched.get('content', '')[:80]}")
        else:
            check("4. siyuan_animation_doc_exists", 2, False,
                  "no document matching 'Idea: Animation and Family Dynamics' found")
    except Exception as e:
        check("4. siyuan_animation_doc_exists", 2, False, f"exception: {e}")


def check_5_siyuan_doc_two_paragraphs() -> None:
    try:
        if not _animation_doc_root_id:
            check("5. siyuan_doc_two_paragraphs", 2, False, "animation doc not found")
            return
        rows = siyuan_sql(
            f"SELECT content, type FROM blocks "
            f"WHERE root_id = '{_animation_doc_root_id}' AND type = 'p' "
            f"ORDER BY sort;"
        )
        substantial = [r for r in rows
                       if len((r.get("content") or "").strip()) >= 50]
        count = len(substantial)
        passed = count >= 2
        detail = f"{count} substantial paragraphs found" if passed else \
                 f"only {count} substantial paragraphs (need >=2)"
        check("5. siyuan_doc_two_paragraphs", 2, passed, detail)
    except Exception as e:
        check("5. siyuan_doc_two_paragraphs", 2, False, f"exception: {e}")


def check_6_siyuan_outline_content_quality() -> None:
    try:
        if not _animation_doc_root_id:
            check("6. siyuan_outline_content_quality", 2, False, "animation doc not found")
            return
        rows = siyuan_sql(
            f"SELECT content FROM blocks "
            f"WHERE root_id = '{_animation_doc_root_id}' AND type IN ('p', 'h') "
            f"ORDER BY sort;"
        )
        full_text = "\n".join(r.get("content", "") for r in rows)
        if len(full_text.strip()) < 30:
            check("6. siyuan_outline_content_quality", 2, False,
                  f"document text too short ({len(full_text.strip())} chars)")
            return
        condition = (
            "The text is a 2-paragraph outline for an episode idea about animation and "
            "family dynamics. It discusses the visual style of an animated film (color, "
            "art direction, or animation techniques) AND explores themes related to family "
            "(family relationships, dynamics, or values). Both topics must be present."
        )
        passed, raw = llm_judge(full_text, condition)
        check("6. siyuan_outline_content_quality", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("6. siyuan_outline_content_quality", 2, False, f"exception: {e}")


def check_7_siyuan_genre_animation_doc() -> None:
    global _genre_doc_root_id
    try:
        rows = siyuan_sql(
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND (content LIKE '%Genre: Animation%' "
            "OR content LIKE '%genre: animation%' "
            "OR content LIKE '%Genre%Animation%') "
            "LIMIT 10;"
        )
        matched = None
        for r in rows:
            c = (r.get("content") or "").lower()
            if "genre" in c and "animation" in c:
                matched = r
                break
        if matched:
            _genre_doc_root_id = matched["id"]
            check("7. siyuan_genre_animation_doc", 1, True,
                  f"doc: {matched.get('content', '')[:80]}")
        else:
            check("7. siyuan_genre_animation_doc", 1, False,
                  "no 'Genre: Animation' document found")
    except Exception as e:
        check("7. siyuan_genre_animation_doc", 1, False, f"exception: {e}")


def check_8_siyuan_bidirectional_link() -> None:
    try:
        if not _animation_doc_root_id or not _genre_doc_root_id:
            missing = []
            if not _animation_doc_root_id:
                missing.append("animation doc")
            if not _genre_doc_root_id:
                missing.append("Genre: Animation doc")
            check("8. siyuan_bidirectional_link", 3, False,
                  f"missing: {', '.join(missing)}")
            return
        forward = siyuan_sql(
            f"SELECT id FROM refs WHERE "
            f"root_id = '{_animation_doc_root_id}' AND def_block_root_id = '{_genre_doc_root_id}' "
            f"LIMIT 1;"
        )
        backward = siyuan_sql(
            f"SELECT id FROM refs WHERE "
            f"root_id = '{_genre_doc_root_id}' AND def_block_root_id = '{_animation_doc_root_id}' "
            f"LIMIT 1;"
        )
        has_forward = len(forward) > 0
        has_backward = len(backward) > 0
        if has_forward and has_backward:
            check("8. siyuan_bidirectional_link", 3, True, "refs exist in both directions")
        else:
            anim_blocks = siyuan_sql(
                f"SELECT markdown FROM blocks WHERE root_id='{_animation_doc_root_id}' "
                f"AND type IN ('p','h','l','i') ORDER BY sort;"
            )
            genre_blocks = siyuan_sql(
                f"SELECT markdown FROM blocks WHERE root_id='{_genre_doc_root_id}' "
                f"AND type IN ('p','h','l','i') ORDER BY sort;"
            )
            anim_md = "\n".join(b.get("markdown", "") for b in anim_blocks)
            genre_md = "\n".join(b.get("markdown", "") for b in genre_blocks)
            has_fwd_inline = _genre_doc_root_id in anim_md
            has_bwd_inline = _animation_doc_root_id in genre_md
            if has_fwd_inline and has_bwd_inline:
                check("8. siyuan_bidirectional_link", 3, True,
                      "block IDs found in markdown in both directions")
            elif has_fwd_inline or has_bwd_inline:
                direction = "animation->genre" if has_fwd_inline else "genre->animation"
                check("8. siyuan_bidirectional_link", 3, False,
                      f"only {direction} inline ref found; missing other direction")
            else:
                check("8. siyuan_bidirectional_link", 3, False,
                      "no refs found between animation doc and Genre: Animation doc")
    except Exception as e:
        check("8. siyuan_bidirectional_link", 3, False, f"exception: {e}")


def check_9_cross_modal_outline_poster() -> None:
    try:
        if not _input_files_ok:
            check("9. cross_modal_outline_poster", 2, False, "skipped: input file missing")
            return
        if not _animation_doc_root_id:
            check("9. cross_modal_outline_poster", 2, False, "animation doc not found")
            return
        rows = siyuan_sql(
            f"SELECT content FROM blocks "
            f"WHERE root_id = '{_animation_doc_root_id}' AND type = 'p' "
            f"ORDER BY sort;"
        )
        full_text = "\n".join(r.get("content", "") for r in rows)
        if len(full_text.strip()) < 30:
            check("9. cross_modal_outline_poster", 2, False,
                  f"outline text too short ({len(full_text.strip())} chars)")
            return
        condition = (
            "The outline text accurately describes visual elements consistent with the "
            "animated film poster shown in the image (e.g., color palette, character "
            "depictions, magical/fantastical setting, Colombian cultural elements). "
            "The outline's discussion of visual style is grounded in what is actually "
            "visible in the poster."
        )
        passed, raw = llm_judge_vision(INPUT_FILES[0], full_text[:500], condition)
        check("9. cross_modal_outline_poster", 2, passed, f"llm_judge_vision: {raw}")
    except Exception as e:
        check("9. cross_modal_outline_poster", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()
    check_1_watcharr_encanto_exists()
    check_2_watcharr_status_planned()
    check_3_cross_modal_poster_title()
    check_4_siyuan_animation_doc_exists()
    check_5_siyuan_doc_two_paragraphs()
    check_6_siyuan_outline_content_quality()
    check_7_siyuan_genre_animation_doc()
    check_8_siyuan_bidirectional_link()
    check_9_cross_modal_outline_poster()

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
