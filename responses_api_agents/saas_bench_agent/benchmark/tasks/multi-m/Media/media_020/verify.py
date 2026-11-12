# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_020/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_020: Paris urban texture photo curation (PhotoPrism) + podcast episode (MediaCMS)

Checks: 12 weighted checks (22 total points) across photoprism and mediacms.
Strategy: docker exec into the single app containers — the DBs run inside them
(MariaDB in the PhotoPrism container, PostgreSQL in the MediaCMS container) —
plus PhotoPrism REST API + llm_judge + llm_judge_vision.

Required env vars:
  SERVER_HOSTNAME, PHOTOPRISM_PORT, PHOTOPRISM_CONTAINER,
  MEDIACMS_PORT, MEDIACMS_CONTAINER

Optional env vars:
  PHOTOPRISM_DB_CONTAINER, MEDIACMS_DB_CONTAINER — override the container used
  for DB queries (default: the app containers themselves).
"""

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

PHOTOPRISM_PORT = os.getenv("PHOTOPRISM_PORT")
PHOTOPRISM_CONTAINER = os.getenv("PHOTOPRISM_CONTAINER")
MEDIACMS_PORT = os.getenv("MEDIACMS_PORT")
MEDIACMS_CONTAINER = os.getenv("MEDIACMS_CONTAINER")

for _var_name, _var_val in [
    ("PHOTOPRISM_PORT", PHOTOPRISM_PORT),
    ("PHOTOPRISM_CONTAINER", PHOTOPRISM_CONTAINER),
    ("MEDIACMS_PORT", MEDIACMS_PORT),
    ("MEDIACMS_CONTAINER", MEDIACMS_CONTAINER),
]:
    if not _var_val:
        print(f"FATAL: {_var_name} not set", file=sys.stderr)
        sys.exit(1)

PHOTOPRISM_DB: str = ""
MEDIACMS_DB: str = ""

_INPUTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs"
)

INPUT_FILES: list[str] = [
    os.path.join(_INPUTS_DIR, "photoprism_photo_003.jpg"),
]

PROHIBITED_KEYWORDS = {
    "paris", "city", "urban", "photo", "image", "photograph", "texture",
    "france", "europe", "canon", "camera",
}

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


def mariadb_query(sql: str, timeout: int = 15) -> tuple[int, str, str]:
    for client in ("mariadb", "mysql"):
        rc, stdout, stderr = docker_exec(
            PHOTOPRISM_DB,
            client, "--user=photoprism", "--password=insecure",
            "--default-character-set=utf8mb4", "-N", "-B", "-e", sql, "photoprism",
            timeout=timeout,
        )
        if rc == 0 or "not found" not in stderr.lower():
            return rc, stdout, stderr
    return rc, stdout, stderr


def psql_query(sql: str, timeout: int = 15) -> tuple[int, str, str]:
    return docker_exec(
        MEDIACMS_DB,
        "psql", "-U", "mediacms", "-d", "mediacms",
        "-t", "-A", "-c", sql,
        timeout=timeout,
    )


_pp_session: str | None = None


def photoprism_api(endpoint: str, params: dict | None = None) -> dict | list | None:
    global _pp_session
    if _pp_session is None:
        try:
            data = json.dumps({"username": "admin", "password": "photoprism123"}).encode()
            req = urllib.request.Request(
                f"http://{HOST}:{PHOTOPRISM_PORT}/api/v1/session",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())
            _pp_session = body.get("id", "")
        except Exception:
            _pp_session = ""

    url = f"http://{HOST}:{PHOTOPRISM_PORT}{endpoint}"
    if params:
        qs = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
        url += "?" + qs
    try:
        req = urllib.request.Request(url, headers={"X-Session-ID": _pp_session})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


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
    if not api_key:
        return False, "MINDRA_API_KEY not set"

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


# ── Shared state ──────────────────────────────────────────────────────────────
_target_photo_uid: str = ""
_target_photo_id: int = 0


# ── Individual checks ─────────────────────────────────────────────────────────

def check_0_input_files_exist() -> None:
    missing = [p for p in INPUT_FILES if not os.path.isfile(p)]
    if missing:
        check("0. input_files_exist", 1, False, "missing: " + ", ".join(missing))
    else:
        check("0. input_files_exist", 1, True)


def check_1_photo_favorited() -> None:
    """The Paris/Canon EOS 90D photo is marked as favourite in PhotoPrism."""
    global _target_photo_uid, _target_photo_id
    try:
        rc, out, err = mariadb_query(
            "SELECT p.id, p.photo_uid, p.photo_favorite, c.camera_model, pl.place_city "
            "FROM photos p "
            "JOIN cameras c ON p.camera_id = c.id "
            "JOIN places pl ON p.place_id = pl.id "
            "WHERE p.deleted_at IS NULL "
            "AND (c.camera_model LIKE '%90D%' OR c.camera_model LIKE '%EOS 90D%') "
            "AND (pl.place_city LIKE '%Paris%' OR pl.place_label LIKE '%Paris%' "
            "OR pl.place_city LIKE '%Charenton%')"
        )
        if rc != 0:
            check("1. photo_favorited", 2, False, f"db error: {err.strip()}")
            return
        rows = [line.split("\t") for line in out.strip().splitlines() if line.strip()]
        if not rows:
            rc2, out2, _ = mariadb_query(
                "SELECT p.id, p.photo_uid, p.photo_favorite "
                "FROM photos p WHERE p.deleted_at IS NULL AND p.photo_favorite = 1"
            )
            fav_rows = [l.split("\t") for l in out2.strip().splitlines() if l.strip()] if rc2 == 0 else []
            check("1. photo_favorited", 2, False,
                  f"no Paris+Canon EOS 90D photo found; {len(fav_rows)} total favourites")
            return
        row = rows[0]
        _target_photo_id = int(row[0])
        _target_photo_uid = row[1]
        is_fav = row[2] == "1"
        if is_fav:
            check("1. photo_favorited", 2, True)
        else:
            check("1. photo_favorited", 2, False,
                  f"photo uid={_target_photo_uid} found but not favourited (favorite={row[2]})")
    except Exception as e:
        check("1. photo_favorited", 2, False, f"exception: {e}")


def check_2_keywords_count() -> None:
    """The target photo has 3-5 visual keywords (excluding generic terms)."""
    if not _target_photo_id:
        check("2. keywords_count_3_to_5", 2, False, "target photo not identified")
        return
    try:
        rc, out, _ = mariadb_query(
            f"SELECT k.keyword FROM keywords k "
            f"JOIN photos_keywords pk ON k.id = pk.keyword_id "
            f"WHERE pk.photo_id = {_target_photo_id}"
        )
        if rc != 0:
            check("2. keywords_count_3_to_5", 2, False, "db error querying keywords")
            return
        kws = [line.strip() for line in out.strip().splitlines() if line.strip()]
        valid_kws = [kw for kw in kws if kw.lower() not in PROHIBITED_KEYWORDS]
        if 3 <= len(valid_kws) <= 5:
            check("2. keywords_count_3_to_5", 2, True,
                  f"found {len(valid_kws)} valid keywords: {valid_kws}")
        else:
            check("2. keywords_count_3_to_5", 2, False,
                  f"expected 3-5 valid keywords, found {len(valid_kws)}: {valid_kws} "
                  f"(all kws: {kws})")
    except Exception as e:
        check("2. keywords_count_3_to_5", 2, False, f"exception: {e}")


def check_3_keywords_visually_specific() -> None:
    """Keywords are visually specific, describing observable content (llm_judge)."""
    if not _target_photo_id:
        check("3. keywords_visually_specific", 2, False, "target photo not identified")
        return
    try:
        rc, out, _ = mariadb_query(
            f"SELECT k.keyword FROM keywords k "
            f"JOIN photos_keywords pk ON k.id = pk.keyword_id "
            f"WHERE pk.photo_id = {_target_photo_id}"
        )
        kws = [line.strip() for line in out.strip().splitlines() if line.strip()] if rc == 0 else []
        valid_kws = [kw for kw in kws if kw.lower() not in PROHIBITED_KEYWORDS]
        if not valid_kws:
            check("3. keywords_visually_specific", 2, False, "no valid keywords found")
            return
        kw_str = ", ".join(valid_kws)
        passed, reason = llm_judge(
            kw_str,
            "These keywords are visually specific and describe concrete visual "
            "elements observable in a photograph (e.g. lighting conditions, "
            "architectural features, weather, textures, materials, human activity, "
            "surface patterns). They are NOT generic placeholders like 'city', 'urban', "
            "'photo', 'texture'."
        )
        check("3. keywords_visually_specific", 2, passed,
              f"keywords: {kw_str}; judge: {reason}")
    except Exception as e:
        check("3. keywords_visually_specific", 2, False, f"exception: {e}")


def check_4_cross_modal_keywords() -> None:
    """Keywords match observable content in the actual input image (llm_judge_vision)."""
    if not _target_photo_id:
        check("4. cross_modal_keywords_match_image", 2, False, "target photo not identified")
        return
    if not os.path.isfile(INPUT_FILES[0]):
        check("4. cross_modal_keywords_match_image", 2, False, "skipped: input file missing")
        return
    try:
        rc, out, _ = mariadb_query(
            f"SELECT k.keyword FROM keywords k "
            f"JOIN photos_keywords pk ON k.id = pk.keyword_id "
            f"WHERE pk.photo_id = {_target_photo_id}"
        )
        kws = [line.strip() for line in out.strip().splitlines() if line.strip()] if rc == 0 else []
        valid_kws = [kw for kw in kws if kw.lower() not in PROHIBITED_KEYWORDS]
        if not valid_kws:
            check("4. cross_modal_keywords_match_image", 2, False, "no valid keywords to check")
            return
        kw_str = ", ".join(valid_kws)
        passed, reason = llm_judge_vision(
            INPUT_FILES[0],
            kw_str,
            "The visual keywords accurately describe specific observable elements "
            "in this photograph (textures, materials, architectural details, lighting, etc.)."
        )
        check("4. cross_modal_keywords_match_image", 2, passed,
              f"keywords: {kw_str}; judge: {reason}")
    except Exception as e:
        check("4. cross_modal_keywords_match_image", 2, False, f"exception: {e}")


def check_5_description_camera_model() -> None:
    """Photo description mentions Canon EOS 90D (matching EXIF)."""
    if not _target_photo_uid:
        check("5. description_mentions_camera", 2, False, "target photo not identified")
        return
    try:
        desc = _get_photo_description(_target_photo_uid)
        desc_lower = desc.lower()
        has_camera = ("canon" in desc_lower and "90d" in desc_lower) or "eos 90d" in desc_lower
        if has_camera:
            check("5. description_mentions_camera", 2, True)
        else:
            check("5. description_mentions_camera", 2, False,
                  f"'Canon EOS 90D' not found in description (len={len(desc)}): "
                  f"{desc[:120]}...")
    except Exception as e:
        check("5. description_mentions_camera", 2, False, f"exception: {e}")


def _get_photo_description(uid: str) -> str:
    data = photoprism_api(f"/api/v1/photos/{uid}")
    if data is not None:
        desc = (data.get("Description", "") or "").strip()
        if not desc:
            desc = (data.get("Caption", "") or "").strip()
        if not desc:
            details = data.get("Details", {})
            if isinstance(details, dict):
                desc = (details.get("Caption", "") or details.get("Notes", "") or "").strip()
        return desc
    rc, out, _ = mariadb_query(
        f"SELECT p.photo_caption FROM photos p WHERE p.photo_uid = '{uid}'"
    )
    caption = out.strip() if rc == 0 else ""
    if not caption:
        rc2, out2, _ = mariadb_query(
            f"SELECT d.notes FROM details d "
            f"JOIN photos p ON d.photo_id = p.id "
            f"WHERE p.photo_uid = '{uid}'"
        )
        caption = out2.strip() if rc2 == 0 else ""
    return caption


def check_6_camera_capability_factual() -> None:
    """Camera capability sentence is factually accurate for Canon EOS 90D (llm_judge)."""
    if not _target_photo_uid:
        check("6. camera_capability_factual", 2, False, "target photo not identified")
        return
    try:
        desc = _get_photo_description(_target_photo_uid)
        if len(desc) < 25:
            check("6. camera_capability_factual", 2, False,
                  f"description too short ({len(desc)} chars)")
            return
        passed, reason = llm_judge(
            desc,
            "The text contains a sentence about the imaging characteristics of "
            "the Canon EOS 90D camera (e.g. sensor format, dynamic range, autofocus, "
            "resolution, APS-C sensor). The stated specifications are factually "
            "consistent with the real-world Canon EOS 90D — no incorrect sensor format "
            "or obviously wrong specification claims."
        )
        check("6. camera_capability_factual", 2, passed, f"judge: {reason}")
    except Exception as e:
        check("6. camera_capability_factual", 2, False, f"exception: {e}")


def check_7_description_visual_commentary() -> None:
    """Description has >=25 char visual commentary on light and composition."""
    if not _target_photo_uid:
        check("7. description_visual_commentary", 2, False, "target photo not identified")
        return
    try:
        desc = _get_photo_description(_target_photo_uid)
        if len(desc) < 25:
            check("7. description_visual_commentary", 2, False,
                  f"description too short ({len(desc)} chars, need >=25)")
            return
        passed, reason = llm_judge(
            desc,
            "This description contains a visual commentary that describes specific "
            "light and compositional elements of a photograph (e.g. directional light, "
            "leading lines, shadow contrast, golden hour, depth of field, perspective). "
            "It is NOT generic praise like 'beautiful photo' or just a location name."
        )
        check("7. description_visual_commentary", 2, passed, f"judge: {reason}")
    except Exception as e:
        check("7. description_visual_commentary", 2, False, f"exception: {e}")


def check_8_mediacms_entry_exists_published() -> None:
    """MediaCMS entry 'EP-55: Textures of Paris' exists with state=public."""
    try:
        rc, out, err = psql_query(
            "SELECT id, state, title FROM files_media "
            "WHERE title = 'EP-55: Textures of Paris' LIMIT 1"
        )
        if rc != 0:
            check("8. mediacms_entry_published", 2, False, f"db error: {err.strip()}")
            return
        row = out.strip()
        if not row:
            rc2, out2, _ = psql_query(
                "SELECT id, title, state FROM files_media "
                "WHERE title LIKE '%EP-55%' OR title LIKE '%Textures%Paris%' LIMIT 5"
            )
            fuzzy = out2.strip() if rc2 == 0 else ""
            detail = "entry 'EP-55: Textures of Paris' not found"
            if fuzzy:
                detail += f"; similar: {fuzzy}"
            check("8. mediacms_entry_published", 2, False, detail)
            return
        parts = row.split("|")
        state = parts[1].strip() if len(parts) > 1 else ""
        if state == "public":
            check("8. mediacms_entry_published", 2, True)
        else:
            check("8. mediacms_entry_published", 2, False,
                  f"state='{state}', expected 'public'")
    except Exception as e:
        check("8. mediacms_entry_published", 2, False, f"exception: {e}")


def check_9_mediacms_category_podcast() -> None:
    """MediaCMS entry has 'Art' category."""
    try:
        rc, out, err = psql_query(
            "SELECT c.title FROM files_category c "
            "JOIN files_media_category mc ON c.id = mc.category_id "
            "JOIN files_media m ON mc.media_id = m.id "
            "WHERE m.title = 'EP-55: Textures of Paris'"
        )
        if rc != 0:
            check("9. mediacms_category_art", 1, False, f"db error: {err.strip()}")
            return
        categories = [line.strip().lower() for line in out.strip().splitlines() if line.strip()]
        if any("art" in c for c in categories):
            check("9. mediacms_category_art", 1, True)
        else:
            check("9. mediacms_category_art", 1, False,
                  f"'Art' not found; categories: {categories}")
    except Exception as e:
        check("9. mediacms_category_art", 1, False, f"exception: {e}")


def check_10_mediacms_tags() -> None:
    """MediaCMS entry has tags 'Paris', 'urban', 'cinematic'."""
    required_tags = {"paris", "urban", "cinematic"}
    try:
        rc, out, err = psql_query(
            "SELECT t.title FROM files_tag t "
            "JOIN files_media_tags mt ON t.id = mt.tag_id "
            "JOIN files_media m ON mt.media_id = m.id "
            "WHERE m.title = 'EP-55: Textures of Paris'"
        )
        if rc != 0:
            check("10. mediacms_tags_exact", 2, False, f"db error: {err.strip()}")
            return
        actual_tags = {line.strip().lower() for line in out.strip().splitlines() if line.strip()}
        missing = required_tags - actual_tags
        if missing:
            check("10. mediacms_tags_exact", 2, False,
                  f"missing tags: {missing}; found: {actual_tags}")
        else:
            check("10. mediacms_tags_exact", 2, True, f"found: {actual_tags}")
    except Exception as e:
        check("10. mediacms_tags_exact", 2, False, f"exception: {e}")


def check_11_mediacms_cover_image() -> None:
    """MediaCMS entry has an uploaded cover/poster image."""
    try:
        rc, out, err = psql_query(
            "SELECT uploaded_poster, poster FROM files_media "
            "WHERE title = 'EP-55: Textures of Paris' LIMIT 1"
        )
        if rc != 0:
            check("11. mediacms_cover_image", 2, False, f"db error: {err.strip()}")
            return
        row = out.strip()
        if not row:
            check("11. mediacms_cover_image", 2, False, "entry not found")
            return
        parts = row.split("|")
        uploaded_poster = (parts[0] if len(parts) > 0 else "").strip()
        poster = (parts[1] if len(parts) > 1 else "").strip()
        if uploaded_poster or poster:
            check("11. mediacms_cover_image", 2, True,
                  f"uploaded_poster={uploaded_poster!r}, poster={poster!r}")
        else:
            check("11. mediacms_cover_image", 2, False,
                  "no cover image (uploaded_poster and poster both empty)")
    except Exception as e:
        check("11. mediacms_cover_image", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    global PHOTOPRISM_DB, MEDIACMS_DB
    # The harness starts each app as a single container with the DB running
    # inside it (MariaDB in PhotoPrism, PostgreSQL in MediaCMS), so DB queries
    # are executed in the app containers themselves. The <APP>_DB_CONTAINER
    # env vars remain as an explicit override, e.g. for a real sidecar setup.
    PHOTOPRISM_DB = os.getenv("PHOTOPRISM_DB_CONTAINER") or PHOTOPRISM_CONTAINER
    MEDIACMS_DB = os.getenv("MEDIACMS_DB_CONTAINER") or MEDIACMS_CONTAINER
    print(f"[INFO] PhotoPrism DB container: {PHOTOPRISM_DB}", file=sys.stderr)
    print(f"[INFO] MediaCMS DB container: {MEDIACMS_DB}", file=sys.stderr)

    check_0_input_files_exist()
    check_1_photo_favorited()
    check_2_keywords_count()
    check_3_keywords_visually_specific()
    check_4_cross_modal_keywords()
    check_5_description_camera_model()
    check_6_camera_capability_factual()
    check_7_description_visual_commentary()
    check_8_mediacms_entry_exists_published()
    check_9_mediacms_category_podcast()
    check_10_mediacms_tags()
    check_11_mediacms_cover_image()

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
