# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_027/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_027: Horror poster → Watcharr 'Want to Watch' + SiYuan planning doc.

Checks: 9 weighted checks (14 total points) across watcharr, siyuan.
Strategy: watcharr via docker exec SQLite; siyuan via REST API; llm_judge_vision for cross-modal.

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
    os.path.join(_INPUTS_DIR, "watcharr_poster_007.jpg"),
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
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"SiYuan API HTTP {e.code}: {e.read().decode()[:200]}")


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
    try:
        import requests as req_lib
        resp = req_lib.post(
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


def check_1_scream_content_exists() -> None:
    try:
        rows = watcharr_sql(
            "SELECT id, title FROM contents WHERE LOWER(title) LIKE '%scream%';"
        )
        if rows:
            check("1. scream_content_exists", 1, True, f"found: {rows.split(chr(10))[0]}")
        else:
            check("1. scream_content_exists", 1, False, "no content with title containing 'Scream'")
    except Exception as e:
        check("1. scream_content_exists", 1, False, f"exception: {e}")


def check_2_scream_status_planned() -> None:
    try:
        rows = watcharr_sql(
            "SELECT w.status FROM watcheds w "
            "JOIN contents c ON w.content_id = c.id "
            "JOIN users u ON w.user_id = u.id "
            "WHERE LOWER(c.title) LIKE '%scream%' AND u.username = 'admin' "
            "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC;"
        )
        if not rows:
            check("2. scream_status_planned", 2, False, "no watched entry for Scream")
            return
        status = rows.split("\n")[0].strip()
        passed = status == "PLANNED"
        detail = f"status='{status}'" if not passed else ""
        check("2. scream_status_planned", 2, passed, detail)
    except Exception as e:
        check("2. scream_status_planned", 2, False, f"exception: {e}")


def check_3_scream_no_rating() -> None:
    try:
        rows = watcharr_sql(
            "SELECT w.rating, w.thoughts FROM watcheds w "
            "JOIN contents c ON w.content_id = c.id "
            "JOIN users u ON w.user_id = u.id "
            "WHERE LOWER(c.title) LIKE '%scream%' AND u.username = 'admin' "
            "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC;"
        )
        if not rows:
            check("3. scream_no_rating", 2, False, "no watched entry for Scream")
            return
        parts = rows.split("\n")[0].split("|")
        rating = parts[0].strip() if len(parts) > 0 else ""
        thoughts = parts[1].strip() if len(parts) > 1 else ""
        no_rating = (rating == "" or rating == "0")
        no_review = (thoughts == "")
        passed = no_rating and no_review
        detail = ""
        if not no_rating:
            detail += f"rating={rating}"
        if not no_review:
            detail += f" thoughts='{thoughts[:50]}'"
        check("3. scream_no_rating", 2, passed, detail.strip())
    except Exception as e:
        check("3. scream_no_rating", 2, False, f"exception: {e}")


def check_4_cross_modal_poster_title() -> None:
    image_path = INPUT_FILES[0]
    if not os.path.isfile(image_path):
        check("4. cross_modal_poster_title", 2, False, "skipped: input file missing")
        return
    try:
        title_row = watcharr_sql(
            "SELECT c.title FROM watcheds w "
            "JOIN contents c ON w.content_id = c.id "
            "JOIN users u ON w.user_id = u.id "
            "WHERE LOWER(c.title) LIKE '%scream%' AND u.username = 'admin' "
            "AND w.deleted_at IS NULL ORDER BY w.updated_at DESC;"
        )
        if not title_row:
            check("4. cross_modal_poster_title", 2, False, "no Scream entry in watcharr to validate")
            return
        recorded_title = title_row.split("\n")[0].strip()
        passed, raw = llm_judge_vision(
            image_path,
            recorded_title,
            "The film title visible in the movie poster matches the recorded value.",
        )
        check("4. cross_modal_poster_title", 2, passed, raw if not passed else "")
    except Exception as e:
        check("4. cross_modal_poster_title", 2, False, f"exception: {e}")


def check_5_siyuan_doc_exists() -> None:
    try:
        rows = siyuan_sql(
            "SELECT id, content FROM blocks "
            "WHERE type = 'd' AND content LIKE '%Upcoming%90s Horror Revival%' LIMIT 5;"
        )
        if rows:
            check("5. siyuan_doc_exists", 1, True)
        else:
            check("5. siyuan_doc_exists", 1, False,
                  "no document with title containing 'Upcoming: 90s Horror Revival'")
    except Exception as e:
        check("5. siyuan_doc_exists", 1, False, f"exception: {e}")


def check_6_siyuan_doc_in_episode_planning() -> None:
    try:
        notebooks_resp = siyuan_api_call("/api/notebook/lsNotebooks")
        notebooks = notebooks_resp.get("data", {}).get("notebooks", [])
        ep_box_ids = [nb["id"] for nb in notebooks
                      if "episode planning" in nb.get("name", "").lower()]
        if not ep_box_ids:
            check("6. siyuan_doc_in_episode_planning", 2, False,
                  "no notebook named 'Episode Planning' found")
            return

        found = False
        for box_id in ep_box_ids:
            rows = siyuan_sql(
                f"SELECT id FROM blocks WHERE type = 'd' "
                f"AND box = '{box_id}' "
                f"AND content LIKE '%Upcoming%90s Horror Revival%' LIMIT 1;"
            )
            if rows:
                found = True
                break

        check("6. siyuan_doc_in_episode_planning", 2, found,
              "" if found else "doc not found in 'Episode Planning' notebook")
    except Exception as e:
        check("6. siyuan_doc_in_episode_planning", 2, False, f"exception: {e}")


def check_7_siyuan_doc_mentions_scream() -> None:
    try:
        rows = siyuan_sql(
            "SELECT id FROM blocks WHERE type = 'd' "
            "AND content LIKE '%Upcoming%90s Horror Revival%' LIMIT 1;"
        )
        if not rows:
            check("7. siyuan_doc_mentions_scream", 2, False, "planning doc not found")
            return

        doc_id = rows[0].get("id", "")
        child_rows = siyuan_sql(
            f"SELECT content FROM blocks WHERE root_id = '{doc_id}' AND type != 'd';"
        )
        all_content = " ".join(r.get("content", "") for r in child_rows)

        passed = "scream" in all_content.lower()
        check("7. siyuan_doc_mentions_scream", 2, passed,
              "" if passed else f"'Scream' not found in doc content (len={len(all_content)})")
    except Exception as e:
        check("7. siyuan_doc_mentions_scream", 2, False, f"exception: {e}")


def check_8_siyuan_doc_length() -> None:
    try:
        rows = siyuan_sql(
            "SELECT id FROM blocks WHERE type = 'd' "
            "AND content LIKE '%Upcoming%90s Horror Revival%' LIMIT 1;"
        )
        if not rows:
            check("8. siyuan_doc_length_gte50", 1, False, "planning doc not found")
            return

        doc_id = rows[0].get("id", "")
        child_rows = siyuan_sql(
            f"SELECT content FROM blocks WHERE root_id = '{doc_id}' AND type != 'd';"
        )
        all_content = " ".join(r.get("content", "") for r in child_rows)
        length = len(all_content.strip())

        passed = length >= 50
        check("8. siyuan_doc_length_gte50", 1, passed,
              "" if passed else f"content length={length}, need >=50")
    except Exception as e:
        check("8. siyuan_doc_length_gte50", 1, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()
    check_1_scream_content_exists()
    check_2_scream_status_planned()
    check_3_scream_no_rating()
    check_4_cross_modal_poster_title()
    check_5_siyuan_doc_exists()
    check_6_siyuan_doc_in_episode_planning()
    check_7_siyuan_doc_mentions_scream()
    check_8_siyuan_doc_length()

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
