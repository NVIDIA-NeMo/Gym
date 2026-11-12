# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_070/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_070: Spider-Man: No Way Home poster → Watcharr watch log
→ SiYuan EP-60 superhero narrative / multiverse script.

Checks: 14 weighted checks (25 total points) across watcharr, siyuan.
Strategy: watcharr via docker exec SQLite; siyuan via REST API; llm_judge + llm_judge_vision.

Required env vars:
  SERVER_HOSTNAME, WATCHARR_PORT, WATCHARR_CONTAINER, SIYUAN_PORT, SIYUAN_CONTAINER.
"""

import base64
import json
import os
import re
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


def siyuan_export_md(doc_id: str) -> str:
    """Export a document's markdown in TRUE document order.

    NOTE: the blocks table's `sort` column is grouped by block type (headings,
    paragraphs, lists), NOT document order — section extraction from
    `ORDER BY sort` yields headings first and prose last, i.e. empty sections.
    exportMdContent returns the document in real reading order.
    """
    payload = json.dumps({"id": doc_id}).encode()
    headers = {"Content-Type": "application/json"}
    token = get_siyuan_token()
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(
        f"{SIYUAN_API}/api/export/exportMdContent",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"SiYuan export HTTP {e.code}: {e.read().decode()[:200]}")
    if body.get("code") != 0:
        raise RuntimeError(f"SiYuan export error: {body.get('msg', body)}")
    return body.get("data", {}).get("content", "")


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


def siyuan_api_call(endpoint: str, payload: dict = None) -> dict:
    data = json.dumps(payload or {}).encode()
    headers = {"Content-Type": "application/json"}
    token = get_siyuan_token()
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(
        f"{SIYUAN_API}{endpoint}",
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


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
_ep60_root_id: str = ""
_director_root_id: str = ""
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


def check_1_watcharr_spiderman_exists() -> None:
    global _watcharr_row
    try:
        rows = watcharr_sql(
            "SELECT w.status, w.rating, w.thoughts, c.title "
            "FROM watcheds w JOIN contents c ON w.content_id = c.id "
            "JOIN users u ON w.user_id = u.id AND u.username = 'admin' "
            "WHERE (LOWER(c.title) LIKE '%spider-man%no way home%' "
            "OR LOWER(c.title) LIKE '%spider man%no way home%' "
            "OR LOWER(c.title) LIKE '%spiderman%no way home%') "
            "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC LIMIT 1;"
        )
        if not rows:
            check("1. watcharr_spiderman_exists", 2, False,
                  "no watched entry for 'Spider-Man: No Way Home' found")
            return
        parts = rows.split("|", 3)
        _watcharr_row = {
            "status": parts[0] if len(parts) > 0 else "",
            "rating": parts[1] if len(parts) > 1 else "",
            "thoughts": parts[2] if len(parts) > 2 else "",
            "title": parts[3] if len(parts) > 3 else "",
        }
        check("1. watcharr_spiderman_exists", 2, True,
              f"found: {_watcharr_row['title']}")
    except Exception as e:
        check("1. watcharr_spiderman_exists", 2, False, f"exception: {e}")


def check_2_watcharr_status_finished() -> None:
    try:
        if not _watcharr_row:
            check("2. watcharr_status_finished", 2, False, "no watched row available")
            return
        status = _watcharr_row["status"]
        passed = status == "FINISHED"
        detail = "" if passed else f"status is '{status}', expected 'FINISHED'"
        check("2. watcharr_status_finished", 2, passed, detail)
    except Exception as e:
        check("2. watcharr_status_finished", 2, False, f"exception: {e}")


def check_3_watcharr_rating_8() -> None:
    try:
        if not _watcharr_row:
            check("3. watcharr_rating_8", 1, False, "no watched row available")
            return
        raw = _watcharr_row["rating"]
        rating = float(raw)
        passed = abs(rating - 8.0) < 0.01
        detail = "" if passed else f"rating is {rating}, expected 8.0"
        check("3. watcharr_rating_8", 1, passed, detail)
    except Exception as e:
        check("3. watcharr_rating_8", 1, False, f"exception: {e}")


def check_4_watcharr_review_length() -> None:
    try:
        if not _watcharr_row:
            check("4. watcharr_review_length", 1, False, "no watched row available")
            return
        thoughts = _watcharr_row["thoughts"]
        if not thoughts:
            check("4. watcharr_review_length", 1, False, "review is empty")
            return
        words = len(thoughts.split())
        passed = 45 <= words <= 120
        detail = f"word_count={words}" if passed else f"word_count={words}, expected 50-100"
        check("4. watcharr_review_length", 1, passed, detail)
    except Exception as e:
        check("4. watcharr_review_length", 1, False, f"exception: {e}")


def check_5_watcharr_review_multiverse_analysis() -> None:
    try:
        if not _watcharr_row:
            check("5. watcharr_review_multiverse_analysis", 2, False,
                  "no watched row available")
            return
        thoughts = _watcharr_row["thoughts"]
        if len(thoughts) < 10:
            check("5. watcharr_review_multiverse_analysis", 2, False,
                  "review too short for analysis")
            return
        condition = (
            "The review is written in English and focuses on the narrative structure "
            "of the film, specifically analyzing the multiverse concept (e.g., how "
            "multiple Spider-Man versions converge, multiversal storytelling mechanics, "
            "narrative stakes of reality-breaking). It does NOT consist of generic "
            "emotional praise or plot summary. It must contain substantive analytical "
            "commentary on at least one narrative or structural technique."
        )
        passed, raw = llm_judge(thoughts, condition)
        check("5. watcharr_review_multiverse_analysis", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("5. watcharr_review_multiverse_analysis", 2, False, f"exception: {e}")


def check_6_cross_modal_poster_title() -> None:
    try:
        if not _input_files_ok:
            check("6. cross_modal_poster_title", 2, False,
                  "skipped: input file missing")
            return
        if not _watcharr_row:
            check("6. cross_modal_poster_title", 2, False,
                  "no watched row to compare against poster")
            return
        title = _watcharr_row["title"]
        condition = (
            "The movie poster shown is for the film 'Spider-Man: No Way Home' "
            "(2021, directed by Jon Watts, starring Tom Holland). The title visible "
            "on the poster matches the recorded value."
        )
        passed, raw = llm_judge_vision(INPUT_FILES[0], title, condition)
        check("6. cross_modal_poster_title", 2, passed, f"llm_judge_vision: {raw}")
    except Exception as e:
        check("6. cross_modal_poster_title", 2, False, f"exception: {e}")


def check_7_siyuan_podcast_scripts_notebook() -> None:
    try:
        result = siyuan_api_call("/api/notebook/lsNotebooks", {})
        notebooks = result.get("data", {}).get("notebooks", [])
        found = False
        for nb in notebooks:
            name = (nb.get("name") or "").lower()
            if "podcast" in name and "script" in name:
                found = True
                break
        if found:
            check("7. siyuan_podcast_scripts_notebook", 1, True)
        else:
            nb_names = [nb.get("name", "") for nb in notebooks[:10]]
            check("7. siyuan_podcast_scripts_notebook", 1, False,
                  f"no 'Podcast Scripts' notebook found; notebooks: {nb_names}")
    except Exception as e:
        check("7. siyuan_podcast_scripts_notebook", 1, False, f"exception: {e}")


def check_8_siyuan_ep60_doc_exists() -> None:
    global _ep60_root_id
    try:
        rows = siyuan_sql(
            "SELECT id, content, box FROM blocks WHERE type = 'd' "
            "AND (content LIKE '%EP-60%' OR content LIKE '%ep-60%' "
            "OR content LIKE '%EP60%') "
            "LIMIT 10;"
        )
        matched = None
        for r in rows:
            content_lower = (r.get("content") or "").lower()
            if "ep-60" in content_lower or "ep60" in content_lower:
                matched = r
                break
        if not matched:
            check("8. siyuan_ep60_doc_exists", 2, False,
                  "no document matching EP-60 found")
            return
        _ep60_root_id = matched["id"]
        check("8. siyuan_ep60_doc_exists", 2, True,
              f"doc='{matched.get('content', '')[:80]}'")
    except Exception as e:
        check("8. siyuan_ep60_doc_exists", 2, False, f"exception: {e}")


def check_9_siyuan_ep60_four_sections() -> None:
    try:
        if not _ep60_root_id:
            check("9. siyuan_ep60_four_sections", 3, False, "EP-60 doc not found")
            return
        rows = siyuan_sql(
            f"SELECT content FROM blocks WHERE root_id = '{_ep60_root_id}' "
            f"AND type = 'h' ORDER BY sort;"
        )
        headings = [r.get("content", "") for r in rows]
        required_keywords = [
            ("episode intro", "intro"),
            ("core argument", "core arguments", "analytical dimensions"),
            ("scene analysis", "representative scene"),
            ("closing", "recommendation", "closing recommendation"),
        ]
        found_sections = []
        for keywords in required_keywords:
            for h in headings:
                h_lower = h.lower()
                if any(kw in h_lower for kw in keywords):
                    found_sections.append(h)
                    break
        count = len(found_sections)
        passed = count >= 4
        detail = "" if passed else f"found {count}/4 sections; headings: {headings[:8]}"
        check("9. siyuan_ep60_four_sections", 3, passed, detail)
    except Exception as e:
        check("9. siyuan_ep60_four_sections", 3, False, f"exception: {e}")


def check_10_siyuan_ep60_content_thresholds() -> None:
    try:
        if not _ep60_root_id:
            check("10. siyuan_ep60_content_thresholds", 2, False,
                  "EP-60 doc not found")
            return
        # NOTE: blocks.`sort` is grouped by block type, not document order, so
        # section bodies must be taken from the exported markdown instead.
        md = siyuan_export_md(_ep60_root_id)
        if not md.strip():
            check("10. siyuan_ep60_content_thresholds", 2, False,
                  "no content in EP-60 doc")
            return
        section_keys = {
            "episode intro": "intro", "intro": "intro",
            "core argument": "core", "core arguments": "core",
            "analytical dimensions": "core",
            "scene analysis": "scene", "representative scene": "scene",
            "closing": "closing", "recommendation": "closing",
            "closing recommendation": "closing",
        }
        sections: dict[str, str] = {}
        for _lvl, title, lines in md_sections(md):
            title_lower = title.lower()
            for kw, label in section_keys.items():
                if kw in title_lower:
                    sections[label] = (
                        sections.get(label, "") + "\n".join(lines) + "\n"
                    )
                    break

        issues = []
        intro_len = len(sections.get("intro", ""))
        if intro_len < 100:
            issues.append(f"intro: {intro_len} chars (need >=100)")

        core_text = sections.get("core", "")
        core_lines = [l.strip() for l in core_text.strip().splitlines() if l.strip()]
        if len(core_lines) < 3:
            issues.append(
                f"core_arguments: {len(core_lines)} items (need >=3 analytical dimensions)")

        scene_text = sections.get("scene", "")
        scene_lines = [l.strip() for l in scene_text.strip().splitlines() if l.strip()]
        if len(scene_lines) < 2:
            issues.append(
                f"scene_analysis: {len(scene_lines)} items (need >=2 scenes)")

        closing_text = sections.get("closing", "")
        if len(closing_text.strip()) < 10:
            issues.append(f"closing: too short ({len(closing_text.strip())} chars)")

        if not issues:
            check("10. siyuan_ep60_content_thresholds", 2, True)
        else:
            check("10. siyuan_ep60_content_thresholds", 2, False,
                  "; ".join(issues))
    except Exception as e:
        check("10. siyuan_ep60_content_thresholds", 2, False, f"exception: {e}")


def check_11_siyuan_ep60_llm_quality() -> None:
    try:
        if not _ep60_root_id:
            check("11. siyuan_ep60_llm_quality", 2, False, "EP-60 doc not found")
            return
        rows = siyuan_sql(
            f"SELECT content FROM blocks "
            f"WHERE root_id = '{_ep60_root_id}' AND type IN ('h', 'p', 'l', 'i') "
            f"ORDER BY sort ASC;"
        )
        full_text = "\n".join(r.get("content", "") for r in rows)
        if len(full_text) < 50:
            check("11. siyuan_ep60_llm_quality", 2, False,
                  f"document too short ({len(full_text)} chars)")
            return
        condition = (
            "The document is titled with 'EP-60' and concerns superhero narratives "
            "and the multiverse concept. It contains four distinct sections: "
            "'Episode Intro' (>=100 chars), 'Core Arguments' (>=3 analytical dimensions), "
            "'Representative Scene Analysis' (>=2 scenes), and 'Closing Recommendation'. "
            "The content is substantive and analytical, not generic filler."
        )
        passed, raw = llm_judge(full_text[:3000], condition)
        check("11. siyuan_ep60_llm_quality", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("11. siyuan_ep60_llm_quality", 2, False, f"exception: {e}")


def check_12_siyuan_director_doc_exists() -> None:
    global _director_root_id
    try:
        rows = siyuan_sql(
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND (content LIKE '%Jon Watts%' OR content LIKE '%Director Filmography%' "
            "OR content LIKE '%jon watts%') "
            "LIMIT 10;"
        )
        matched = None
        for r in rows:
            c = (r.get("content") or "").lower()
            if "jon watts" in c or "director filmography" in c:
                matched = r
                break
        if matched:
            _director_root_id = matched["id"]
            check("12. siyuan_director_doc_exists", 1, True,
                  f"doc: {matched.get('content', '')[:80]}")
        else:
            check("12. siyuan_director_doc_exists", 1, False,
                  "no 'Director Filmography — Jon Watts' document found")
    except Exception as e:
        check("12. siyuan_director_doc_exists", 1, False, f"exception: {e}")


def check_13_siyuan_bidirectional_link() -> None:
    try:
        if not _ep60_root_id or not _director_root_id:
            missing = []
            if not _ep60_root_id:
                missing.append("EP-60 doc")
            if not _director_root_id:
                missing.append("director doc")
            check("13. siyuan_bidirectional_link", 3, False,
                  f"missing: {', '.join(missing)}")
            return

        forward = siyuan_sql(
            f"SELECT id FROM refs WHERE "
            f"root_id = '{_ep60_root_id}' AND def_block_root_id = '{_director_root_id}' "
            f"LIMIT 1;"
        )
        backward = siyuan_sql(
            f"SELECT id FROM refs WHERE "
            f"root_id = '{_director_root_id}' AND def_block_root_id = '{_ep60_root_id}' "
            f"LIMIT 1;"
        )
        has_forward = len(forward) > 0
        has_backward = len(backward) > 0

        if has_forward and has_backward:
            check("13. siyuan_bidirectional_link", 3, True,
                  "refs exist in both directions")
        elif has_forward or has_backward:
            direction = "EP-60->Director" if has_forward else "Director->EP-60"
            check("13. siyuan_bidirectional_link", 3, True,
                  f"ref {direction} found; SiYuan auto-generates the backlink view")
        else:
            ep60_blocks = siyuan_sql(
                f"SELECT markdown FROM blocks WHERE root_id='{_ep60_root_id}' "
                f"AND type IN ('p','h','l','i') ORDER BY sort;"
            )
            dir_blocks = siyuan_sql(
                f"SELECT markdown FROM blocks WHERE root_id='{_director_root_id}' "
                f"AND type IN ('p','h','l','i') ORDER BY sort;"
            )
            ep60_md = "\n".join(b.get("markdown", "") for b in ep60_blocks)
            dir_md = "\n".join(b.get("markdown", "") for b in dir_blocks)
            has_fwd_inline = _director_root_id in ep60_md
            has_bwd_inline = _ep60_root_id in dir_md

            if not has_fwd_inline:
                ep60_lower = ep60_md.lower()
                has_fwd_inline = (
                    "jon watts" in ep60_lower
                    or "director filmography" in ep60_lower
                )
            if not has_bwd_inline:
                dir_lower = dir_md.lower()
                has_bwd_inline = (
                    "ep-60" in dir_lower
                    or "superhero narrative" in dir_lower
                )

            if has_fwd_inline and has_bwd_inline:
                check("13. siyuan_bidirectional_link", 3, True,
                      "text references found in both directions (fallback)")
            elif has_fwd_inline or has_bwd_inline:
                direction = "EP-60->Director" if has_fwd_inline else "Director->EP-60"
                check("13. siyuan_bidirectional_link", 3, False,
                      f"only {direction} inline ref found; missing other direction")
            else:
                check("13. siyuan_bidirectional_link", 3, False,
                      "no refs found between EP-60 and director doc in either direction")
    except Exception as e:
        check("13. siyuan_bidirectional_link", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()
    check_1_watcharr_spiderman_exists()
    check_2_watcharr_status_finished()
    check_3_watcharr_rating_8()
    check_4_watcharr_review_length()
    check_5_watcharr_review_multiverse_analysis()
    check_6_cross_modal_poster_title()
    check_7_siyuan_podcast_scripts_notebook()
    check_8_siyuan_ep60_doc_exists()
    check_9_siyuan_ep60_four_sections()
    check_10_siyuan_ep60_content_thresholds()
    check_11_siyuan_ep60_llm_quality()
    check_12_siyuan_director_doc_exists()
    check_13_siyuan_bidirectional_link()

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
