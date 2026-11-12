# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_063/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_063: Add ballads book to Booklore, create SiYuan research note with link.

Checks: 8 weighted checks across booklore, siyuan.
Strategy: docker exec MariaDB for booklore; REST API for siyuan.

Required env vars:
  SERVER_HOSTNAME, BOOKLORE_PORT, BOOKLORE_CONTAINER, SIYUAN_PORT, SIYUAN_CONTAINER
"""

import os
import sys
import json
import subprocess
import re
import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")

BOOKLORE_PORT = os.getenv("BOOKLORE_PORT")
BOOKLORE_CONTAINER = os.getenv("BOOKLORE_CONTAINER")
SIYUAN_PORT = os.getenv("SIYUAN_PORT")
SIYUAN_CONTAINER = os.getenv("SIYUAN_CONTAINER")

for var_name, var_val in [
    ("BOOKLORE_PORT", BOOKLORE_PORT),
    ("BOOKLORE_CONTAINER", BOOKLORE_CONTAINER),
    ("SIYUAN_PORT", SIYUAN_PORT),
    ("SIYUAN_CONTAINER", SIYUAN_CONTAINER),
]:
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
        sys.exit(1)

BOOKLORE_DB_CONTAINER = os.getenv("BOOKLORE_DB_CONTAINER") or BOOKLORE_CONTAINER
BOOKLORE_BASE = f"http://{HOST}:{BOOKLORE_PORT}"
SIYUAN_BASE = f"http://{HOST}:{SIYUAN_PORT}"


def _get_siyuan_token() -> str:
    try:
        r = subprocess.run(
            ["docker", "exec", SIYUAN_CONTAINER, "cat",
             "/siyuan/workspace/conf/conf.json"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            conf = json.loads(r.stdout)
            return conf.get("api", {}).get("token", "")
    except Exception:
        pass
    return ""


SIYUAN_TOKEN = _get_siyuan_token()

# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def _siyuan_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if SIYUAN_TOKEN:
        h["Authorization"] = f"Token {SIYUAN_TOKEN}"
    return h


def siyuan_sql(stmt: str) -> list:
    try:
        resp = requests.post(
            f"{SIYUAN_BASE}/api/query/sql",
            headers=_siyuan_headers(),
            json={"stmt": stmt},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code", -1) != 0:
            print(f"  siyuan_sql non-zero code: {body.get('msg')}", file=sys.stderr)
            return []
        data = body.get("data", [])
        return data if data else []
    except Exception as e:
        print(f"  siyuan_sql error: {e}", file=sys.stderr)
        return []


def siyuan_api(endpoint: str, payload: dict) -> dict:
    try:
        resp = requests.post(
            f"{SIYUAN_BASE}{endpoint}",
            headers=_siyuan_headers(),
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  siyuan_api error: {e}", file=sys.stderr)
        return {}


# ── Individual checks ─────────────────────────────────────────────────────────

_book_id: str | None = None


def check_1_book_exists() -> None:
    global _book_id
    try:
        q = (
            "SELECT bm.book_id, bm.title FROM book_metadata bm "
            "WHERE bm.title LIKE '%Old Ballads%' AND bm.title LIKE '%Volume 4%' "
            "LIMIT 1"
        )
        result = mariadb_query(q)
        if result:
            parts = result.split("\t", 1)
            _book_id = parts[0]
            title = parts[1] if len(parts) > 1 else ""
            check("1. Book 'A Book of Old Ballads — Volume 4' exists", 1, True,
                  f"book_id={_book_id}, title={title}")
        else:
            q_broad = (
                "SELECT bm.book_id, bm.title FROM book_metadata bm "
                "WHERE bm.title LIKE '%Ballad%' LIMIT 5"
            )
            broad = mariadb_query(q_broad)
            check("1. Book 'A Book of Old Ballads — Volume 4' exists", 1, False,
                  f"not found; ballad titles: {broad[:200]}")
    except Exception as e:
        check("1. Book 'A Book of Old Ballads — Volume 4' exists", 1, False,
              f"exception: {e}")


def check_2_read_status() -> None:
    if not _book_id:
        check("2. Book is on admin's 'Want to Read' shelf", 2, False,
              "skipped: book not found")
        return
    try:
        q_shelf = (
            f"SELECT s.name FROM shelf s "
            f"JOIN book_shelf_mapping bsm ON s.id = bsm.shelf_id "
            f"JOIN users u ON u.id = s.user_id "
            f"WHERE bsm.book_id = {_book_id} AND u.username = 'admin' "
            f"AND LOWER(TRIM(s.name)) = 'want to read'"
        )
        shelves = [line.strip() for line in mariadb_query(q_shelf).splitlines() if line.strip()]
        check("2. Book is on admin's 'Want to Read' shelf", 2, bool(shelves),
              f"shelves={shelves}" if shelves
              else "book is not on admin's exact 'Want to Read' shelf")
    except Exception as e:
        check("2. Book is on admin's 'Want to Read' shelf", 2, False,
              f"exception: {e}")


def check_3_reading_note_oral_traditions() -> None:
    if not _book_id:
        check("3. Book has note mentioning 'Oral Traditions'", 2, False,
              "skipped: book not found")
        return
    try:
        q_v1 = (
            f"SELECT content FROM book_notes "
            f"WHERE book_id = {_book_id} "
            f"ORDER BY updated_at DESC LIMIT 5"
        )
        q_v2 = (
            f"SELECT note_content FROM book_notes_v2 "
            f"WHERE book_id = {_book_id} "
            f"ORDER BY created_at DESC LIMIT 5"
        )
        notes_v1 = mariadb_query(q_v1)
        notes_v2 = mariadb_query(q_v2)

        all_notes = (notes_v1 + " " + notes_v2).lower()
        has_oral = "oral tradition" in all_notes or "oral traditions" in all_notes

        if has_oral:
            check("3. Book has note mentioning 'Oral Traditions'", 2, True,
                  "found 'Oral Traditions' in note content")
        else:
            snippet_v1 = notes_v1[:150] if notes_v1 else "(none)"
            snippet_v2 = notes_v2[:150] if notes_v2 else "(none)"
            check("3. Book has note mentioning 'Oral Traditions'", 2, False,
                  f"'Oral Traditions' not found; v1={snippet_v1}; v2={snippet_v2}")
    except Exception as e:
        check("3. Book has note mentioning 'Oral Traditions'", 2, False,
              f"exception: {e}")


_notebook_id: str | None = None


def check_4_siyuan_podcast_scripts_notebook() -> None:
    global _notebook_id
    try:
        resp = siyuan_api("/api/notebook/lsNotebooks", {})
        notebooks = resp.get("data", {}).get("notebooks", [])
        for nb in notebooks:
            name = nb.get("name", "")
            if name.strip().lower() == "podcast scripts":
                _notebook_id = nb.get("id", "")
                check("4. SiYuan 'Podcast Scripts' notebook exists", 2, True,
                      f"notebook_id={_notebook_id}, name='{name}'")
                return

        names = [nb.get("name", "") for nb in notebooks]
        check("4. SiYuan 'Podcast Scripts' notebook exists", 2, False,
              f"not found; notebooks: {names}")
    except Exception as e:
        check("4. SiYuan 'Podcast Scripts' notebook exists", 2, False,
              f"exception: {e}")


_doc_id: str | None = None


def check_5_siyuan_doc_exists() -> None:
    global _doc_id
    try:
        notebook_filter = f" AND box = '{_notebook_id}'" if _notebook_id else ""
        rows = siyuan_sql(
            "SELECT id, content, box FROM blocks "
            "WHERE type = 'd' "
            "AND LOWER(TRIM(content)) = 'ep-research: oral traditions'"
            + notebook_filter
        )

        if rows:
            doc = rows[0]
            _doc_id = doc.get("id", "")
            title = doc.get("content", "")
            check("5. SiYuan doc 'EP-Research: Oral Traditions' exists", 2, True,
                  f"doc_id={_doc_id}, title='{title}'")
        else:
            all_docs = siyuan_sql(
                "SELECT content FROM blocks WHERE type = 'd' "
                "AND (content LIKE '%EP%' OR content LIKE '%Oral%' OR content LIKE '%Research%') "
                "LIMIT 5"
            )
            titles = [r.get("content", "") for r in (all_docs or [])]
            check("5. SiYuan doc 'EP-Research: Oral Traditions' exists", 2, False,
                  f"not found; related docs: {titles}")
    except Exception as e:
        check("5. SiYuan doc 'EP-Research: Oral Traditions' exists", 2, False,
              f"exception: {e}")


def check_6_doc_in_podcast_notebook() -> None:
    if not _doc_id:
        check("6. Doc is under 'Podcast Scripts' notebook", 1, False,
              "skipped: doc not found")
        return
    if not _notebook_id:
        check("6. Doc is under 'Podcast Scripts' notebook", 1, False,
              "skipped: notebook not found")
        return
    try:
        rows = siyuan_sql(
            f"SELECT box FROM blocks WHERE id = '{_doc_id}' AND type = 'd'"
        )
        if rows:
            box = rows[0].get("box", "")
            passed = box == _notebook_id
            check("6. Doc is under 'Podcast Scripts' notebook", 1, passed,
                  f"doc box={box}, expected notebook={_notebook_id}")
        else:
            check("6. Doc is under 'Podcast Scripts' notebook", 1, False,
                  "could not retrieve doc box")
    except Exception as e:
        check("6. Doc is under 'Podcast Scripts' notebook", 1, False,
              f"exception: {e}")


def check_7_siyuan_doc_mentions_book() -> None:
    if not _doc_id:
        check("7. SiYuan doc mentions the ballads book", 1, False,
              "skipped: doc not found")
        return
    try:
        blocks = siyuan_sql(
            f"SELECT content, markdown FROM blocks "
            f"WHERE root_id = '{_doc_id}' AND type != 'd'"
        )
        full_content = " ".join(
            (b.get("content", "") + " " + b.get("markdown", ""))
            for b in (blocks or [])
        ).lower()

        has_ballad = "ballad" in full_content or "old ballads" in full_content
        check("7. SiYuan doc mentions the ballads book", 1, has_ballad,
              "found 'ballad' reference" if has_ballad
              else f"'ballad' not in doc content (len={len(full_content)})")
    except Exception as e:
        check("7. SiYuan doc mentions the ballads book", 1, False,
              f"exception: {e}")


def check_8_siyuan_hyperlink_to_booklore() -> None:
    if not _doc_id:
        check("8. SiYuan doc has hyperlink to Booklore entry", 3, False,
              "skipped: doc not found")
        return
    try:
        blocks = siyuan_sql(
            f"SELECT markdown FROM blocks "
            f"WHERE root_id = '{_doc_id}' AND type != 'd'"
        )
        full_md = " ".join(b.get("markdown", "") for b in (blocks or []))

        booklore_port = BOOKLORE_PORT
        all_urls = set(re.findall(r'https?://[^\s\)\">\'\]]+', full_md))
        all_urls.update(url for _, url in re.findall(r'\[([^\]]*)\]\((https?://[^)]+)\)', full_md))
        target_pattern = re.compile(
            rf"/book/{re.escape(str(_book_id))}(?:[/?#]|$)"
        )
        target_urls = [
            url for url in all_urls
            if ("booklore" in url.lower() or f":{booklore_port}" in url)
            and target_pattern.search(url)
        ]

        if target_urls:
            check("8. SiYuan doc has hyperlink to Booklore entry", 3, True,
                  f"target link: {target_urls[0]}")
        else:
            check("8. SiYuan doc has hyperlink to Booklore entry", 3, False,
                  f"no Booklore link points to book_id={_book_id}; URLs: {list(all_urls)[:5]}")
    except Exception as e:
        check("8. SiYuan doc has hyperlink to Booklore entry", 3, False,
              f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_1_book_exists()
    check_2_read_status()
    check_3_reading_note_oral_traditions()
    check_4_siyuan_podcast_scripts_notebook()
    check_5_siyuan_doc_exists()
    check_6_doc_in_podcast_notebook()
    check_7_siyuan_doc_mentions_book()
    check_8_siyuan_hyperlink_to_booklore()

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
