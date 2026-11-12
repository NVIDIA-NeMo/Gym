# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_035/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_035: Encanto poster → Watcharr rating/review → SiYuan EP-49 script.

Checks: 12 weighted checks (22 total points) across watcharr, siyuan.
Strategy: watcharr via docker exec SQLite; siyuan via REST API SQL; llm_judge + llm_judge_vision.

Required env vars:
  SERVER_HOSTNAME, WATCHARR_PORT, WATCHARR_CONTAINER, SIYUAN_PORT, SIYUAN_CONTAINER.
"""

import base64
import json
import os
import re
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
_ep49_root_id: str = ""
_animation_studios_root_id: str = ""
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
            "JOIN users u ON w.user_id = u.id "
            "WHERE LOWER(c.title) LIKE '%encanto%' AND u.username = 'admin' "
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


def check_2_watcharr_rating_9() -> None:
    try:
        if not _watcharr_row:
            check("2. watcharr_rating_9", 1, False, "no watched row available")
            return
        raw = _watcharr_row["rating"]
        rating = float(raw)
        passed = abs(rating - 9.0) < 0.5
        detail = "" if passed else f"rating is {rating}, expected 9.0"
        check("2. watcharr_rating_9", 1, passed, detail)
    except Exception as e:
        check("2. watcharr_rating_9", 1, False, f"exception: {e}")


def check_3_watcharr_review_length() -> None:
    try:
        if not _watcharr_row:
            check("3. watcharr_review_length", 1, False, "no watched row available")
            return
        thoughts = _watcharr_row["thoughts"]
        if not thoughts:
            check("3. watcharr_review_length", 1, False, "review is empty")
            return
        char_len = len(thoughts)
        passed = char_len >= 60
        detail = f"length={char_len} chars" if passed else f"length={char_len} chars, need >=60"
        check("3. watcharr_review_length", 1, passed, detail)
    except Exception as e:
        check("3. watcharr_review_length", 1, False, f"exception: {e}")


def check_4_watcharr_review_color_magic() -> None:
    try:
        if not _watcharr_row:
            check("4. watcharr_review_color_magic", 2, False, "no watched row available")
            return
        thoughts = _watcharr_row["thoughts"]
        if len(thoughts) < 10:
            check("4. watcharr_review_color_magic", 2, False, "review too short for analysis")
            return
        condition = (
            "The review discusses the animated film Encanto's use of color "
            "and/or magical realism. It should mention aspects like the vibrant "
            "color palette, visual magic elements, or the film's distinctive "
            "artistic style — not just generic praise."
        )
        passed, raw = llm_judge(thoughts, condition)
        check("4. watcharr_review_color_magic", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("4. watcharr_review_color_magic", 2, False, f"exception: {e}")


def check_5_cross_modal_poster_title() -> None:
    try:
        if not _input_files_ok:
            check("5. cross_modal_poster_title", 2, False, "skipped: input file missing")
            return
        if not _watcharr_row or not _watcharr_row.get("title"):
            check("5. cross_modal_poster_title", 2, False,
                  "no Encanto entry in watcharr to validate")
            return
        title = _watcharr_row["title"]
        condition = (
            "The movie poster shown is for the Disney animated film 'Encanto' (2021). "
            "The title visible on the poster or the characters/imagery match the recorded value."
        )
        passed, raw = llm_judge_vision(INPUT_FILES[0], title, condition)
        check("5. cross_modal_poster_title", 2, passed, f"llm_judge_vision: {raw}")
    except Exception as e:
        check("5. cross_modal_poster_title", 2, False, f"exception: {e}")


def check_6_siyuan_ep49_doc_exists() -> None:
    global _ep49_root_id
    try:
        rows = siyuan_sql(
            "SELECT id, content, box FROM blocks WHERE type = 'd' "
            "AND (content LIKE '%EP-49%' OR content LIKE '%ep-49%' "
            "OR content LIKE '%EP49%' OR content LIKE '%Colors of Magic%' "
            "OR content LIKE '%colors of magic%') "
            "LIMIT 10;"
        )
        matched = None
        for r in rows:
            content_lower = (r.get("content") or "").lower()
            if "ep-49" in content_lower or "ep49" in content_lower or "colors of magic" in content_lower:
                matched = r
                break
        if not matched:
            check("6. siyuan_ep49_doc_exists", 2, False,
                  "no document matching EP-49 or 'Colors of Magic' found")
            return
        _ep49_root_id = matched["id"]
        check("6. siyuan_ep49_doc_exists", 2, True,
              f"doc='{matched.get('content', '')[:80]}'")
    except Exception as e:
        check("6. siyuan_ep49_doc_exists", 2, False, f"exception: {e}")


def check_7_siyuan_ep49_three_sections() -> None:
    try:
        if not _ep49_root_id:
            check("7. siyuan_ep49_three_sections", 3, False, "EP-49 doc not found")
            return
        rows = siyuan_sql(
            f"SELECT content FROM blocks WHERE root_id = '{_ep49_root_id}' "
            f"AND type = 'h' ORDER BY sort;"
        )
        headings = [r.get("content", "") for r in rows]
        required_keywords = [
            ("introduction", "引言", "intro"),
            ("visual analysis", "视觉分析", "visual"),
            ("conclusion", "结论", "结语", "总结"),
        ]
        found_sections = []
        for keywords in required_keywords:
            for h in headings:
                h_lower = h.lower()
                if any(kw in h_lower for kw in keywords):
                    found_sections.append(h)
                    break
        count = len(found_sections)
        passed = count >= 3
        detail = "" if passed else f"found {count}/3 sections; headings: {headings[:8]}"
        check("7. siyuan_ep49_three_sections", 3, passed, detail)
    except Exception as e:
        check("7. siyuan_ep49_three_sections", 3, False, f"exception: {e}")


def check_8_siyuan_ep49_content_thresholds() -> None:
    try:
        if not _ep49_root_id:
            check("8. siyuan_ep49_content_thresholds", 2, False, "EP-49 doc not found")
            return
        md = siyuan_export_md(_ep49_root_id)
        section_keys = {
            "introduction": "intro", "引言": "intro", "intro": "intro",
            "visual analysis": "visual", "视觉分析": "visual", "visual": "visual",
            "conclusion": "conclusion", "结论": "conclusion", "结语": "conclusion", "总结": "conclusion",
        }
        sections: dict[str, str] = {}
        for _lvl, title, lines in md_sections(md):
            title_lower = title.lower()
            for kw, label in section_keys.items():
                if kw in title_lower:
                    if label not in sections:
                        sections[label] = ""
                    sections[label] += "\n".join(lines) + "\n"
                    break

        issues = []
        intro_len = len(sections.get("intro", "").strip())
        if intro_len < 50:
            issues.append(f"intro: {intro_len} chars (need >=50)")
        visual_len = len(sections.get("visual", "").strip())
        if visual_len < 50:
            issues.append(f"visual_analysis: {visual_len} chars (need >=50)")
        conclusion_len = len(sections.get("conclusion", "").strip())
        if conclusion_len < 20:
            issues.append(f"conclusion: {conclusion_len} chars (need >=20)")

        if not issues:
            check("8. siyuan_ep49_content_thresholds", 2, True)
        else:
            check("8. siyuan_ep49_content_thresholds", 2, False, "; ".join(issues))
    except Exception as e:
        check("8. siyuan_ep49_content_thresholds", 2, False, f"exception: {e}")


def check_9_cross_modal_visual_analysis() -> None:
    try:
        if not _input_files_ok:
            check("9. cross_modal_visual_analysis", 2, False, "skipped: input file missing")
            return
        if not _ep49_root_id:
            check("9. cross_modal_visual_analysis", 2, False, "EP-49 doc not found")
            return
        md = siyuan_export_md(_ep49_root_id)
        visual_text = ""
        for _lvl, title, lines in md_sections(md):
            tl = title.lower()
            if "visual analysis" in tl or "visual" in tl or "视觉分析" in tl:
                visual_text = "\n".join(lines) + "\n"
                break

        if len(visual_text.strip()) < 20:
            check("9. cross_modal_visual_analysis", 2, False,
                  f"visual analysis section too short ({len(visual_text.strip())} chars)")
            return

        condition = (
            "The text in the 'Visual Analysis' section explicitly references the color "
            "palette or visual characteristics that can be observed in the Encanto movie "
            "poster. It should mention specific colors, visual motifs, or artistic elements "
            "consistent with the poster's imagery."
        )
        passed, raw = llm_judge_vision(INPUT_FILES[0], visual_text.strip()[:500], condition)
        check("9. cross_modal_visual_analysis", 2, passed, f"llm_judge_vision: {raw}")
    except Exception as e:
        check("9. cross_modal_visual_analysis", 2, False, f"exception: {e}")


def check_10_siyuan_animation_studios_doc() -> None:
    global _animation_studios_root_id
    try:
        rows = siyuan_sql(
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND (content LIKE '%Animation Studios%' OR content LIKE '%animation studios%' "
            "OR content LIKE '%动画工作室%') "
            "LIMIT 10;"
        )
        matched = None
        for r in rows:
            c = (r.get("content") or "").lower()
            if "animation studios" in c or "动画工作室" in c:
                matched = r
                break
        if matched:
            _animation_studios_root_id = matched["id"]
            check("10. siyuan_animation_studios_doc", 1, True,
                  f"doc: {matched.get('content', '')[:80]}")
        else:
            check("10. siyuan_animation_studios_doc", 1, False,
                  "no 'Animation Studios' document found")
    except Exception as e:
        check("10. siyuan_animation_studios_doc", 1, False, f"exception: {e}")


def check_11_siyuan_bidirectional_link() -> None:
    try:
        if not _ep49_root_id or not _animation_studios_root_id:
            missing = []
            if not _ep49_root_id:
                missing.append("EP-49 doc")
            if not _animation_studios_root_id:
                missing.append("Animation Studios doc")
            check("11. siyuan_bidirectional_link", 3, False,
                  f"missing: {', '.join(missing)}")
            return
        forward = siyuan_sql(
            f"SELECT id FROM refs WHERE "
            f"root_id = '{_ep49_root_id}' AND def_block_root_id = '{_animation_studios_root_id}' "
            f"LIMIT 1;"
        )
        backward = siyuan_sql(
            f"SELECT id FROM refs WHERE "
            f"root_id = '{_animation_studios_root_id}' AND def_block_root_id = '{_ep49_root_id}' "
            f"LIMIT 1;"
        )
        has_forward = len(forward) > 0
        has_backward = len(backward) > 0
        if has_forward and has_backward:
            check("11. siyuan_bidirectional_link", 3, True, "refs exist in both directions")
        elif has_forward:
            check("11. siyuan_bidirectional_link", 3, False,
                  "only forward link (EP-49 → Animation Studios); missing reverse link")
        elif has_backward:
            check("11. siyuan_bidirectional_link", 3, False,
                  "only reverse link (Animation Studios → EP-49); missing forward link")
        else:
            ep49_blocks = siyuan_sql(
                f"SELECT markdown FROM blocks WHERE root_id='{_ep49_root_id}' "
                f"AND type IN ('p','h','l','i') ORDER BY sort;"
            )
            anim_blocks = siyuan_sql(
                f"SELECT markdown FROM blocks WHERE root_id='{_animation_studios_root_id}' "
                f"AND type IN ('p','h','l','i') ORDER BY sort;"
            )
            ep49_md = "\n".join(b.get("markdown", "") for b in ep49_blocks)
            anim_md = "\n".join(b.get("markdown", "") for b in anim_blocks)
            has_fwd_inline = _animation_studios_root_id in ep49_md
            has_bwd_inline = _ep49_root_id in anim_md
            if has_fwd_inline and has_bwd_inline:
                check("11. siyuan_bidirectional_link", 3, True,
                      "block IDs found in markdown in both directions")
            elif has_fwd_inline or has_bwd_inline:
                direction = "EP-49→AnimStudios" if has_fwd_inline else "AnimStudios→EP-49"
                check("11. siyuan_bidirectional_link", 3, False,
                      f"only {direction} inline ref found; missing other direction")
            else:
                check("11. siyuan_bidirectional_link", 3, False,
                      "no refs found between EP-49 and Animation Studios in either direction")
    except Exception as e:
        check("11. siyuan_bidirectional_link", 3, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()
    check_1_watcharr_encanto_exists()
    check_2_watcharr_rating_9()
    check_3_watcharr_review_length()
    check_4_watcharr_review_color_magic()
    check_5_cross_modal_poster_title()
    check_6_siyuan_ep49_doc_exists()
    check_7_siyuan_ep49_three_sections()
    check_8_siyuan_ep49_content_thresholds()
    check_9_cross_modal_visual_analysis()
    check_10_siyuan_animation_studios_doc()
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
