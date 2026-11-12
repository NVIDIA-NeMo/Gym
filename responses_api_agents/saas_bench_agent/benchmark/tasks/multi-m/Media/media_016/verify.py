# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_016/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_016: PDF paper → Booklore book entry + SiYuan paper digest

Checks: 11 weighted checks (20pt total) across booklore and siyuan.
Strategy: docker exec MariaDB (booklore), SiYuan REST API
          (/api/query/sql, /api/notebook/lsNotebooks);
          llm_judge for note/contribution content quality.

Required env vars:
  SERVER_HOSTNAME, BOOKLORE_PORT, BOOKLORE_CONTAINER,
  SIYUAN_PORT, SIYUAN_CONTAINER
"""

import os
import sys
import subprocess
import json
import re

try:
    import requests
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "requests"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import requests

# ── Config (from env) ─────────────────────────────────────────────────────────
HOST = os.getenv("SERVER_HOSTNAME", "localhost")
BOOKLORE_PORT = os.getenv("BOOKLORE_PORT")
BOOKLORE_CONTAINER = os.getenv("BOOKLORE_CONTAINER")
SIYUAN_PORT = os.getenv("SIYUAN_PORT")
SIYUAN_CONTAINER = os.getenv("SIYUAN_CONTAINER")

_missing = []
for _var in ["BOOKLORE_PORT", "BOOKLORE_CONTAINER", "SIYUAN_PORT", "SIYUAN_CONTAINER"]:
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
    os.path.join(_INPUTS_DIR, "siyuan_paper_001.pdf"),
]

# Facts about the input paper (siyuan_paper_001.pdf)
PAPER_TITLE = "Collaborative Knowledge Creation and Management in Information Retrieval"
PAPER_AUTHOR_SURNAMES = ["odumuyiwa", "david"]
SIYUAN_DOC_TITLE = f"Paper Digest: {PAPER_TITLE}"

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


# ── Booklore helpers (MariaDB via docker exec) ────────────────────────────────
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


# ── SiYuan helpers (REST API) ─────────────────────────────────────────────────
_siyuan_auth_cached = None


def _get_siyuan_auth() -> str:
    global _siyuan_auth_cached
    if _siyuan_auth_cached is not None:
        return _siyuan_auth_cached
    try:
        rc, stdout, _ = docker_exec(
            SIYUAN_CONTAINER, "cat", "/siyuan/workspace/conf/conf.json",
            timeout=10,
        )
        if rc == 0 and stdout.strip():
            conf = json.loads(stdout)
            token = conf.get("api", {}).get("token", "")
            if token:
                _siyuan_auth_cached = token
                return _siyuan_auth_cached
    except Exception:
        pass
    try:
        rc, stdout, _ = docker_exec(
            SIYUAN_CONTAINER, "sh", "-c",
            "cat /proc/1/cmdline | tr '\\0' '\\n'",
        )
        for line in stdout.splitlines():
            if "accessAuthCode=" in line:
                _siyuan_auth_cached = line.split("=", 1)[1].strip()
                return _siyuan_auth_cached
    except Exception:
        pass
    _siyuan_auth_cached = ""
    return _siyuan_auth_cached


def siyuan_api(endpoint: str, payload: dict, timeout: int = 15) -> dict:
    url = f"http://{HOST}:{SIYUAN_PORT}{endpoint}"
    headers = {"Content-Type": "application/json"}
    auth = _get_siyuan_auth()
    if auth:
        headers["Authorization"] = f"Token {auth}"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        return resp.json()
    except Exception as e:
        return {"code": -1, "msg": str(e), "data": None}


def siyuan_sql(stmt: str) -> list:
    result = siyuan_api("/api/query/sql", {"stmt": stmt})
    data = result.get("data")
    if isinstance(data, list):
        return data
    return []


def siyuan_export_md(doc_id: str) -> str:
    """Export a document's markdown in TRUE document order.

    NOTE: the blocks table's `sort` column is grouped by block type (headings,
    paragraphs, lists), NOT document order — section extraction from
    `ORDER BY sort` yields headings first and prose last, i.e. empty sections.
    exportMdContent returns the document in real reading order.
    """
    result = siyuan_api("/api/export/exportMdContent", {"id": doc_id})
    data = result.get("data")
    if isinstance(data, dict):
        return data.get("content", "") or ""
    return ""


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


# ── LLM judge helper ──────────────────────────────────────────────────────────
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


# ── Cached lookups (booklore) ─────────────────────────────────────────────────
_book_id: str | None = None
_book_notes: list[str] = []


def _find_book_id() -> str | None:
    global _book_id
    if _book_id:
        return _book_id
    for pattern in [
        "Collaborative Knowledge Creation%Information Retrieval",
        "Collaborative Knowledge Creation",
        "collaborative knowledge creation",
    ]:
        result = mariadb_query(
            f"SELECT bm.book_id FROM book_metadata bm "
            f"WHERE bm.title LIKE '%{pattern}%' LIMIT 1"
        )
        if result and result.strip():
            _book_id = result.strip().splitlines()[0].strip()
            return _book_id
    return None


# ── Cached lookups (siyuan) ───────────────────────────────────────────────────
_notebook_id = None
_notebook_id_searched = False


def _find_notebook_id() -> str | None:
    global _notebook_id, _notebook_id_searched
    if _notebook_id_searched:
        return _notebook_id
    _notebook_id_searched = True
    result = siyuan_api("/api/notebook/lsNotebooks", {})
    notebooks = result.get("data", {}).get("notebooks", [])
    for nb in notebooks:
        if nb.get("name", "").strip().lower() == "podcast scripts":
            _notebook_id = nb["id"]
            return _notebook_id
    return None


_doc_id = None
_doc_id_searched = False


def _find_doc() -> str | None:
    global _doc_id, _doc_id_searched
    if _doc_id_searched:
        return _doc_id
    _doc_id_searched = True
    nb_id = _find_notebook_id()
    if not nb_id:
        return None
    escaped_title = SIYUAN_DOC_TITLE.replace("'", "''")
    rows = siyuan_sql(
        "SELECT id, content, box FROM blocks "
        f"WHERE type = 'd' AND LOWER(TRIM(content)) = LOWER('{escaped_title}') "
        f"AND box = '{nb_id}' LIMIT 1"
    )
    if rows:
        _doc_id = rows[0]["id"]
        return _doc_id
    return None


_doc_blocks_cache = None


def _get_doc_blocks() -> list[dict]:
    global _doc_blocks_cache
    if _doc_blocks_cache is not None:
        return _doc_blocks_cache
    doc_id = _find_doc()
    if not doc_id:
        _doc_blocks_cache = []
        return _doc_blocks_cache
    rows = siyuan_sql(
        f"SELECT id, parent_id, type, subtype, content, markdown FROM blocks "
        f"WHERE root_id = '{doc_id}' AND type != 'd' ORDER BY sort"
    )
    _doc_blocks_cache = rows
    return _doc_blocks_cache


def _get_full_doc_text() -> str:
    doc_id = _find_doc()
    return siyuan_export_md(doc_id) if doc_id else ""


def _get_intro_text() -> str:
    """Introduction = text under an introduction-like heading, else the leading
    paragraph lines before the first heading or list (no-heading fallback).

    Uses exportMdContent: the blocks table's `sort` column is grouped by block
    type, not document order (see siyuan_export_md).
    """
    doc_id = _find_doc()
    if not doc_id:
        return ""
    md = siyuan_export_md(doc_id)
    if not md.strip():
        return ""
    intro_kw = ["intro", "引言", "overview", "summary", "abstract", "background", "摘要"]
    in_section = False
    parts: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            h = m.group(2).strip().lower()
            if any(k in h for k in intro_kw):
                in_section = True
                parts = []
                continue
            elif in_section:
                break
        if in_section and line.strip():
            parts.append(line.strip())
    if parts:
        return "\n".join(parts)
    lead: list[str] = []
    for line in md.splitlines():
        if re.match(r"^#{1,6}\s", line) or re.match(r"^\s*(?:[-*+]|\d{1,2}[.)])\s", line):
            break
        if line.strip():
            lead.append(line.strip())
    return "\n".join(lead)


def _count_numbered_items(doc_id: str) -> int:
    full_md = siyuan_export_md(doc_id)
    return len(re.findall(r'^\s*\d{1,2}[\.\)、]\s+\S', full_md, re.MULTILINE))


# ── Individual checks ─────────────────────────────────────────────────────────
def check_0_input_files_exist() -> None:
    missing = [p for p in INPUT_FILES if not os.path.isfile(p)]
    if missing:
        check("0. input_files_exist", 1, False, "missing: " + ", ".join(missing))
    else:
        check("0. input_files_exist", 1, True)


def check_1_booklore_book_exists() -> None:
    try:
        book_id = _find_book_id()
        if book_id:
            title_q = mariadb_query(
                f"SELECT bm.title FROM book_metadata bm WHERE bm.book_id = {book_id}"
            )
            check("1. booklore_book_exists", 2, True,
                  f"book_id={book_id}, title='{title_q}'")
        else:
            all_titles = mariadb_query(
                "SELECT bm.book_id, bm.title FROM book_metadata bm "
                "ORDER BY bm.book_id DESC LIMIT 10"
            )
            check("1. booklore_book_exists", 2, False,
                  f"no book titled like the paper found; recent: {all_titles[:200]}")
    except Exception as e:
        check("1. booklore_book_exists", 2, False, f"exception: {e}")


def check_2_booklore_author_correct() -> None:
    book_id = _find_book_id()
    if not book_id:
        check("2. booklore_author_correct", 2, False, "book not found")
        return
    try:
        authors = mariadb_query(
            f"SELECT a.name FROM author a "
            f"JOIN book_metadata_author_mapping m ON a.id = m.author_id "
            f"WHERE m.book_id = {book_id}"
        )
        authors_l = authors.lower()
        passed = all(s in authors_l for s in PAPER_AUTHOR_SURNAMES)
        check("2. booklore_author_correct", 2, passed,
              "" if passed else
              f"authors found: '{authors}', expected Victor Odumuyiwa / Amos David")
    except Exception as e:
        check("2. booklore_author_correct", 2, False, f"exception: {e}")


def check_3_booklore_research_shelf() -> None:
    book_id = _find_book_id()
    if not book_id:
        check("3. booklore_research_shelf", 2, False, "book not found")
        return
    try:
        shelves = mariadb_query(
            f"SELECT s.name FROM shelf s "
            f"JOIN book_shelf_mapping bsm ON s.id = bsm.shelf_id "
            f"JOIN users u ON u.id = s.user_id "
            f"WHERE bsm.book_id = {book_id} AND u.username = 'admin'"
        )
        names = [n.strip() for n in shelves.splitlines() if n.strip()]
        passed = any(n.lower() == "research" for n in names)
        check("3. booklore_research_shelf", 2, passed,
              "" if passed else
              f"admin book shelves: {names}, expected a shelf named 'Research'")
    except Exception as e:
        check("3. booklore_research_shelf", 2, False, f"exception: {e}")


def check_4_booklore_notes_count() -> None:
    global _book_notes
    book_id = _find_book_id()
    if not book_id:
        check("4. booklore_notes_count", 2, False, "book not found")
        return
    try:
        notes_list: list[str] = []
        note_count = 0
        for table, col in [("book_notes_v2", "note_content"), ("book_notes", "content"),
                           ("annotations", "note")]:
            count_str = mariadb_query(
                f"SELECT COUNT(*) FROM {table} WHERE book_id = {book_id}"
            )
            if count_str and int(count_str) > 0:
                note_count = int(count_str)
                result = mariadb_query(
                    f"SELECT {col} FROM {table} WHERE book_id = {book_id}"
                )
                if result:
                    notes_list = [l.strip() for l in result.strip().splitlines()
                                  if l.strip()]
                break
        _book_notes = notes_list
        if note_count == 0:
            note_count = len(notes_list)
        passed = note_count >= 2
        check("4. booklore_notes_count", 2, passed,
              f"count={note_count}" if passed else
              f"found {note_count} notes, need >=2")
    except Exception as e:
        check("4. booklore_notes_count", 2, False, f"exception: {e}")


def check_5_booklore_notes_quality() -> None:
    if not _book_notes:
        check("5. booklore_notes_quality", 2, False, "no notes available")
        return
    try:
        combined = "\n---\n".join(f"Note {i+1}: {n}" for i, n in enumerate(_book_notes))
        condition = (
            "The notes are reading notes about the research paper 'Collaborative "
            "Knowledge Creation and Management in Information Retrieval'. They discuss "
            "specific aspects of the paper, such as collaborative information retrieval "
            "(CIR), collaborative information behaviour, knowledge creation through "
            "Nonaka's knowledge conversion processes, the MECOCIR prototype and its "
            "functional architecture, or annotation/knowledge organization features. "
            "Different notes address different sections or contributions of the paper. "
            "Generic remarks (e.g. 'interesting paper', 'must read') do not count."
        )
        passed, raw = llm_judge(combined, condition)
        check("5. booklore_notes_quality", 2, passed, f"llm_judge: {raw}")
    except Exception as e:
        check("5. booklore_notes_quality", 2, False, f"exception: {e}")


def check_6_siyuan_doc_exists() -> None:
    try:
        doc_id = _find_doc()
        passed = doc_id is not None and _find_notebook_id() is not None
        detail = "" if passed else (
            f"no exact '{SIYUAN_DOC_TITLE}' document in 'Podcast Scripts' notebook"
        )
        check("6. siyuan_doc_exists", 2, passed, detail)
    except Exception as e:
        check("6. siyuan_doc_exists", 2, False, f"exception: {e}")


def check_7_siyuan_intro_length() -> None:
    if not _find_doc():
        check("7. siyuan_intro_length", 1, False, "doc not found")
        return
    try:
        text = _get_intro_text()
        n = len(text.strip())
        passed = n >= 100
        check("7. siyuan_intro_length", 1, passed,
              "" if passed else f"intro is {n} chars, need >=100")
    except Exception as e:
        check("7. siyuan_intro_length", 1, False, f"exception: {e}")


def check_8_siyuan_numbered_contributions() -> None:
    doc_id = _find_doc()
    if not doc_id:
        check("8. siyuan_numbered_contributions", 2, False, "doc not found")
        return
    try:
        n = _count_numbered_items(doc_id)
        passed = n >= 3
        check("8. siyuan_numbered_contributions", 2, passed,
              "" if passed else f"found {n} numbered items, need >=3")
    except Exception as e:
        check("8. siyuan_numbered_contributions", 2, False, f"exception: {e}")


def check_9_siyuan_contributions_relevance() -> None:
    if not _find_doc():
        check("9. siyuan_contributions_relevance", 2, False, "doc not found")
        return
    try:
        full_text = _get_full_doc_text()
        if not full_text:
            check("9. siyuan_contributions_relevance", 2, False,
                  "document empty or not found")
            return
        condition = (
            "The document lists core contributions of the research paper "
            "'Collaborative Knowledge Creation and Management in Information "
            "Retrieval'. The listed contributions are specific to this paper — for "
            "example: explaining collaborative information retrieval (CIR) and how it "
            "culminates in knowledge creation, how created knowledge is organized and "
            "structured, the functional architecture of the MECOCIR prototype and its "
            "features for collaborative knowledge exploitation, or the use of Nonaka's "
            "knowledge conversion/transformation processes in CIR. The contributions "
            "must reflect the paper's actual content on information retrieval and "
            "knowledge creation, not generic statements that could apply to any paper."
        )
        passed, raw = llm_judge(full_text[:4000], condition)
        check("9. siyuan_contributions_relevance", 2, passed,
              "" if passed else f"llm_judge: {raw}")
    except Exception as e:
        check("9. siyuan_contributions_relevance", 2, False, f"exception: {e}")


def check_10_siyuan_booklore_url() -> None:
    if not _find_doc():
        check("10. siyuan_booklore_url", 2, False, "doc not found")
        return
    try:
        full_md = _get_full_doc_text()
        urls = set(re.findall(r'https?://[^\s)\]>"\']+', full_md))
        urls.update(url for _, url in re.findall(r'\[([^\]]*)\]\((https?://[^)]+)\)', full_md))
        book_id = _find_book_id()
        target_pattern = re.compile(
            rf"/book/{re.escape(str(book_id))}(?:[/?#]|$)"
        )
        matching = [
            url for url in urls
            if ("booklore" in url.lower() or f":{BOOKLORE_PORT}" in url)
            and target_pattern.search(url)
        ] if book_id else []
        check("10. siyuan_booklore_url", 2, bool(matching),
              f"target link: {matching[0]}" if matching else
              f"found {len(urls)} URLs, none points to Booklore book_id={book_id}")
    except Exception as e:
        check("10. siyuan_booklore_url", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()
    check_1_booklore_book_exists()
    check_2_booklore_author_correct()
    check_3_booklore_research_shelf()
    check_4_booklore_notes_count()
    check_5_booklore_notes_quality()
    check_6_siyuan_doc_exists()
    check_7_siyuan_intro_length()
    check_8_siyuan_numbered_contributions()
    check_9_siyuan_contributions_relevance()
    check_10_siyuan_booklore_url()

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
