# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Media/media_066/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""
Verifier for media_066: Research summary of collaborative IR PDF in SiYuan

Checks: 11 weighted checks across siyuan.
Strategy: SiYuan REST API (SQL query + markdown export)

Required env vars:
  SERVER_HOSTNAME, SIYUAN_PORT, SIYUAN_CONTAINER
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

SIYUAN_PORT = os.getenv("SIYUAN_PORT")
SIYUAN_CONTAINER = os.getenv("SIYUAN_CONTAINER")

for var_name, var_val in [("SIYUAN_PORT", SIYUAN_PORT), ("SIYUAN_CONTAINER", SIYUAN_CONTAINER)]:
    if not var_val:
        print(f"FATAL: {var_name} not set", file=sys.stderr)
        sys.exit(1)

SIYUAN_BASE = f"http://{HOST}:{SIYUAN_PORT}"

_INPUTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "inputs"
)

INPUT_FILES: list[str] = [
    os.path.join(_INPUTS_DIR, "siyuan_paper_001.pdf"),
]

# ── Result accumulator ────────────────────────────────────────────────────────
_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_siyuan_token() -> str:
    """Read the API token from SiYuan's conf.json inside the container."""
    r = subprocess.run(
        ["docker", "exec", SIYUAN_CONTAINER, "cat", "/siyuan/workspace/conf/conf.json"],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        return ""
    try:
        conf = json.loads(r.stdout)
        return conf.get("api", {}).get("token", "")
    except (json.JSONDecodeError, KeyError):
        return ""


def siyuan_sql(token: str, stmt: str) -> list[dict]:
    """Execute a SQL query against SiYuan's API."""
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
    """Export a document's full markdown content."""
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
    """Returns (passed, raw_response)."""
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
    pdf_path: str,
    recorded_value: str,
    condition: str,
    timeout: int = 60,
) -> tuple[bool, str]:
    """Send a PDF (as base64) + recorded value to a vision LLM for cross-modal check."""
    api_base = os.getenv("MINDRA_BASE_URL", "https://api.mindracode.com/v1")
    api_key = os.getenv("MINDRA_API_KEY", "")

    if not os.path.isfile(pdf_path):
        return False, f"file not found: {pdf_path}"

    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = (
        f"You are given a PDF academic paper and a summary that an AI agent wrote based on it.\n"
        f"Agent's summary:\n«{recorded_value}»\n\n"
        f"Condition: {condition}\n\n"
        f"Does the agent's summary accurately reflect the content of the PDF, satisfying the condition?\n"
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
                        {"type": "image_url",
                         "image_url": {"url": f"data:application/pdf;base64,{b64}"}},
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
    """All multimodal input files referenced in the task must exist on disk."""
    missing = [p for p in INPUT_FILES if not os.path.isfile(p)]
    if missing:
        check("0. input_files_exist", 1, False, "missing: " + ", ".join(missing))
    else:
        check("0. input_files_exist", 1, True)


def check_1_document_exists(token: str) -> str | None:
    """SiYuan document titled 'Research: Collaborative IR' exists."""
    try:
        rows = siyuan_sql(
            token,
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND content LIKE '%Research%Collaborative IR%'"
        )
        if not rows:
            rows = siyuan_sql(
                token,
                "SELECT id, content FROM blocks WHERE type = 'd' "
                "AND content LIKE '%Collaborative IR%'"
            )
        found = [r for r in rows if "Collaborative IR" in r.get("content", "")]
        if found:
            exact = [r for r in found if r.get("content", "").strip() == "Research: Collaborative IR"]
            if exact:
                check("1. document_exists", 2, True, f"title='{exact[0]['content']}'")
                return exact[0]["id"]
            else:
                check("1. document_exists", 2, True,
                      f"title='{found[0]['content']}' (close match)")
                return found[0]["id"]
        else:
            check("1. document_exists", 2, False, "no document matching 'Collaborative IR' found")
            return None
    except Exception as e:
        check("1. document_exists", 2, False, f"exception: {e}")
        return None


def check_2_exact_title(token: str) -> None:
    """Document title is exactly 'Research: Collaborative IR'."""
    try:
        rows = siyuan_sql(
            token,
            "SELECT id, content FROM blocks WHERE type = 'd' "
            "AND content = 'Research: Collaborative IR'"
        )
        if rows:
            check("2. exact_title", 1, True)
        else:
            rows = siyuan_sql(
                token,
                "SELECT id, content FROM blocks WHERE type = 'd' "
                "AND content LIKE '%Collaborative IR%'"
            )
            titles = [r.get("content", "") for r in rows]
            check("2. exact_title", 1, False,
                  f"expected 'Research: Collaborative IR', found: {titles[:3]}")
    except Exception as e:
        check("2. exact_title", 1, False, f"exception: {e}")


def check_3_core_argument_section(token: str, doc_id: str | None, md_content: str) -> str:
    """'Core Argument' section exists as a heading."""
    try:
        if doc_id:
            rows = siyuan_sql(
                token,
                f"SELECT id, content, subtype FROM blocks WHERE root_id = '{doc_id}' "
                f"AND type = 'h' AND content LIKE '%Core Argument%'"
            )
            if rows:
                check("3. core_argument_heading", 1, True)
                return rows[0]["id"]

        pattern = re.compile(r'#+\s+.*Core Argument', re.IGNORECASE)
        if pattern.search(md_content):
            check("3. core_argument_heading", 1, True)
        else:
            check("3. core_argument_heading", 1, False, "heading 'Core Argument' not found")
    except Exception as e:
        check("3. core_argument_heading", 1, False, f"exception: {e}")
    return ""


def check_4_methodology_section(token: str, doc_id: str | None, md_content: str) -> str:
    """'Methodology' section exists as a heading."""
    try:
        if doc_id:
            rows = siyuan_sql(
                token,
                f"SELECT id, content, subtype FROM blocks WHERE root_id = '{doc_id}' "
                f"AND type = 'h' AND content LIKE '%Methodology%'"
            )
            if rows:
                check("4. methodology_heading", 1, True)
                return rows[0]["id"]

        pattern = re.compile(r'#+\s+.*Methodology', re.IGNORECASE)
        if pattern.search(md_content):
            check("4. methodology_heading", 1, True)
        else:
            check("4. methodology_heading", 1, False, "heading 'Methodology' not found")
    except Exception as e:
        check("4. methodology_heading", 1, False, f"exception: {e}")
    return ""


def check_5_podcast_relevance_section(token: str, doc_id: str | None, md_content: str) -> str:
    """'Podcast Relevance' section exists as a heading."""
    try:
        if doc_id:
            rows = siyuan_sql(
                token,
                f"SELECT id, content, subtype FROM blocks WHERE root_id = '{doc_id}' "
                f"AND type = 'h' AND content LIKE '%Podcast Relevance%'"
            )
            if rows:
                check("5. podcast_relevance_heading", 1, True)
                return rows[0]["id"]

        pattern = re.compile(r'#+\s+.*Podcast Relevance', re.IGNORECASE)
        if pattern.search(md_content):
            check("5. podcast_relevance_heading", 1, True)
        else:
            check("5. podcast_relevance_heading", 1, False, "heading 'Podcast Relevance' not found")
    except Exception as e:
        check("5. podcast_relevance_heading", 1, False, f"exception: {e}")
    return ""


def _extract_section_text(md_content: str, section_name: str) -> str:
    """Extract text under a markdown heading matching section_name until the next heading."""
    pattern = re.compile(
        rf'^(#{{1,6}})\s+.*{re.escape(section_name)}.*$',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(md_content)
    if not match:
        return ""
    heading_level = len(match.group(1))
    start = match.end()
    next_heading = re.compile(
        rf'^#{{{1},{heading_level}}}\s+',
        re.MULTILINE,
    )
    next_match = next_heading.search(md_content, start)
    if next_match:
        section = md_content[start:next_match.start()]
    else:
        section = md_content[start:]
    return section.strip()


def check_6_core_argument_length(md_content: str) -> str:
    """'Core Argument' section has >= 100 characters of content."""
    try:
        text = _extract_section_text(md_content, "Core Argument")
        length = len(text)
        if length >= 100:
            check("6. core_argument_length", 2, True, f"{length} chars (>=100)")
        else:
            check("6. core_argument_length", 2, False,
                  f"{length} chars, need >=100")
        return text
    except Exception as e:
        check("6. core_argument_length", 2, False, f"exception: {e}")
        return ""


def check_7_methodology_length(md_content: str) -> str:
    """'Methodology' section has >= 100 characters of content."""
    try:
        text = _extract_section_text(md_content, "Methodology")
        length = len(text)
        if length >= 100:
            check("7. methodology_length", 2, True, f"{length} chars (>=100)")
        else:
            check("7. methodology_length", 2, False,
                  f"{length} chars, need >=100")
        return text
    except Exception as e:
        check("7. methodology_length", 2, False, f"exception: {e}")
        return ""


def check_8_podcast_relevance_length(md_content: str) -> str:
    """'Podcast Relevance' section has >= 50 characters of content."""
    try:
        text = _extract_section_text(md_content, "Podcast Relevance")
        length = len(text)
        if length >= 50:
            check("8. podcast_relevance_length", 2, True, f"{length} chars (>=50)")
        else:
            check("8. podcast_relevance_length", 2, False,
                  f"{length} chars, need >=50")
        return text
    except Exception as e:
        check("8. podcast_relevance_length", 2, False, f"exception: {e}")
        return ""


def check_9_content_accuracy(md_content: str) -> None:
    """Summary content accurately reflects PDF about collaborative knowledge creation."""
    try:
        full_body = _extract_section_text(md_content, "Core Argument")
        full_body += "\n" + _extract_section_text(md_content, "Methodology")
        full_body += "\n" + _extract_section_text(md_content, "Podcast Relevance")

        if len(full_body.strip()) < 50:
            check("9. content_accuracy_llm", 3, False,
                  "insufficient content to judge accuracy")
            return

        passed, answer = llm_judge(
            full_body,
            "The summary discusses collaborative knowledge creation and/or "
            "collaborative information retrieval. It contains a core argument "
            "about collaboration in IR, a methodology section, and discusses "
            "relevance to podcasting or media production. The content is "
            "substantive and reflects themes of an academic paper on "
            "collaborative knowledge creation in information retrieval.",
        )
        check("9. content_accuracy_llm", 3, passed, f"llm_judge: {answer}")
    except Exception as e:
        check("9. content_accuracy_llm", 3, False, f"exception: {e}")


def check_10_cross_modal_pdf(md_content: str) -> None:
    """Cross-modal: summary content matches the actual PDF paper content."""
    try:
        full_body = _extract_section_text(md_content, "Core Argument")
        full_body += "\n" + _extract_section_text(md_content, "Methodology")

        if len(full_body.strip()) < 50:
            check("10. cross_modal_pdf_consistency", 2, False,
                  "skipped: insufficient content to compare")
            return

        if not os.path.isfile(INPUT_FILES[0]):
            check("10. cross_modal_pdf_consistency", 2, False,
                  "skipped: input file missing")
            return

        passed, answer = llm_judge_vision(
            INPUT_FILES[0],
            full_body,
            "The summary accurately reflects the main thesis and methodology "
            "described in the PDF paper about collaborative knowledge creation "
            "in information retrieval systems.",
        )
        check("10. cross_modal_pdf_consistency", 2, passed,
              f"llm_judge_vision: {answer}")
    except Exception as e:
        check("10. cross_modal_pdf_consistency", 2, False, f"exception: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    check_0_input_files_exist()

    token = get_siyuan_token()
    if not token:
        print("FATAL: could not read SiYuan API token from container", file=sys.stderr)
        sys.exit(1)

    doc_id = check_1_document_exists(token)
    check_2_exact_title(token)

    md_content = ""
    if doc_id:
        try:
            md_content = siyuan_export_md(token, doc_id)
        except Exception as e:
            print(f"WARNING: could not export markdown: {e}", file=sys.stderr)

    check_3_core_argument_section(token, doc_id, md_content)
    check_4_methodology_section(token, doc_id, md_content)
    check_5_podcast_relevance_section(token, doc_id, md_content)
    check_6_core_argument_length(md_content)
    check_7_methodology_length(md_content)
    check_8_podcast_relevance_length(md_content)
    check_9_content_accuracy(md_content)
    check_10_cross_modal_pdf(md_content)

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
