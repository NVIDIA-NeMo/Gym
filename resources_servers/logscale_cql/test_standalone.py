#!/usr/bin/env python3
"""
Standalone test of all 3 scoring components against the live LogScale container.

No nemo_gym or ray required -- just requests + pandas.

Usage:
    python test_standalone.py                    # all 3 scores (needs LLM API key)
    python test_standalone.py --no-judge         # validity + execution only (no LLM)
    python test_standalone.py --url http://host:8080
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests as _requests

from container_engine import LogScaleContainerEngine

# ---------------------------------------------------------------------------
# Judge prompts (same as app.py)
# ---------------------------------------------------------------------------

SEMANTIC_JUDGE_PROMPT = """\
You are evaluating whether a CQL query captures the user's intent.

Question: {question}
Generated CQL: {query}

Score the SEMANTIC correctness on a 1-5 scale:
- 5 = Query perfectly captures the user's intent (correct fields, filters, aggregations, ordering)
- 4 = Minor deviation that would still give useful results
- 3 = Partially correct — misses an important aspect of the question
- 2 = Mostly wrong — fundamentally misunderstands the question
- 1 = Completely wrong or unrelated

Respond EXACTLY in this format:
SEMANTIC_SCORE: <1-5>
SEMANTIC_REASONING: <one sentence>"""

EXECUTION_JUDGE_PROMPT = """\
You are evaluating whether a CQL query produced correct results.

Question: {question}
Generated CQL: {query}
Query Results:
{results}

Score the EXECUTION correctness on a 1-5 scale:
- 5 = Results are exactly what you'd expect for this question
- 4 = Results are mostly correct with minor issues (e.g. slightly off sort order)
- 3 = Results are partially correct but have notable issues
- 2 = Results are mostly wrong or empty when they shouldn't be
- 1 = Complete failure — error, empty when data should exist, or nonsensical output

Respond EXACTLY in this format:
EXECUTION_SCORE: <1-5>
EXECUTION_REASONING: <one sentence>"""


# ---------------------------------------------------------------------------
# LLM client (self-contained, reads from ~/.env or env vars)
# ---------------------------------------------------------------------------

def _load_env():
    """Load API key from ~/.env if python-dotenv isn't available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        load_dotenv(Path.home() / ".env")
    except ImportError:
        env_file = Path.home() / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _make_judge_fn(api_key: str, base_url: str, model: str):
    """Return judge_fn(prompt) -> str that calls the LLM."""
    def judge_fn(prompt: str) -> str:
        resp = _requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a CQL query evaluation judge."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    return judge_fn


def _parse_semantic(raw: str) -> dict:
    sm = re.search(r"SEMANTIC_SCORE:\s*(\d)", raw)
    rm = re.search(r"SEMANTIC_REASONING:\s*(.+)", raw)
    return {
        "score": int(sm.group(1)) if sm else None,
        "reasoning": rm.group(1).strip() if rm else raw.strip()[:100],
    }


def _parse_execution(raw: str) -> dict:
    sm = re.search(r"EXECUTION_SCORE:\s*(\d)", raw)
    rm = re.search(r"EXECUTION_REASONING:\s*(.+)", raw)
    return {
        "score": int(sm.group(1)) if sm else None,
        "reasoning": rm.group(1).strip() if rm else raw.strip()[:100],
    }


def compute_reward(validity: int, semantic, execution) -> float:
    sem = (semantic / 5.0) if semantic is not None else 0.0
    exe = (execution / 5.0) if execution is not None else 0.0
    return (float(validity) + sem + exe) / 3.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test LogScale Gym scoring components")
    parser.add_argument("--url", default="http://localhost:8080", help="LogScale URL")
    parser.add_argument("--repository", default=None)
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (validity + execution only)")
    args = parser.parse_args()

    _load_env()

    api_key = os.getenv("INFERENCE_API_KEY", "")
    base_url = os.getenv("NVIDIA_API_URL", "https://inference-api.nvidia.com").rstrip("/")
    model = os.getenv("LLM_MODEL", "aws/anthropic/bedrock-claude-opus-4-6")

    use_judge = not args.no_judge and bool(api_key)
    if not args.no_judge and not api_key:
        print("[WARN] No API key found (INFERENCE_API_KEY). Running without LLM judge.")
        print("       Set INFERENCE_API_KEY in ~/.env or run with --no-judge.\n")

    judge_fn = _make_judge_fn(api_key, base_url, model) if use_judge else None

    print("=" * 60)
    print("  LogScale Gym – Standalone Scoring Test")
    print("=" * 60)
    print(f"  LogScale:  {args.url}")
    print(f"  LLM Judge: {'ON (' + model + ')' if use_judge else 'OFF'}")

    # ── Connect ───────────────────────────────────────────────────
    print(f"\n[..] Connecting to LogScale...")
    engine = LogScaleContainerEngine(base_url=args.url, repository=args.repository)
    print(f"[OK] Repository: {engine.repository}")

    # ── Score 1: Validity ─────────────────────────────────────────
    print("\n── Score 1: Validity (LogScale validateQuery) ──────────")

    valid_queries = [
        'count()',
        'ProcessRollup2 | count()',
        '#event_simpleName=ProcessRollup2 | groupBy(ImageFileName, function=count())',
        'groupBy(ComputerName, function=count()) | sort(_count, order=desc) | head(5)',
        'hashRewrite(field="@ingesttimestamp", salt="salt1")',
    ]
    invalid_queries = [
        '| invalid_function(',
        'groupBy(( broken syntax',
        'count(field=,)',
    ]

    for q in valid_queries:
        r = engine.validate_query(q)
        ok = r["is_valid"]
        print(f"  [{'PASS' if ok else 'FAIL'}] valid=1  {q[:60]}")

    for q in invalid_queries:
        r = engine.validate_query(q)
        ok = not r["is_valid"]
        print(f"  [{'PASS' if ok else 'FAIL'}] valid=0  {q[:60]}")

    # ── Full 3-component scoring ──────────────────────────────────
    print("\n── Full Rubric (validity + semantic + execution) ───────")

    test_cases = [
        {"question": "How many events are there in total?", "cql": "count()"},
        {"question": "Show top 5 computers by event count", "cql": "groupBy(ComputerName, function=count()) | sort(_count, order=desc) | head(5)"},
        {"question": "Count ProcessRollup2 events", "cql": "#event_simpleName=ProcessRollup2 | count()"},
        {"question": "Show top 10 remote IPs for network connections", "cql": "NetworkConnectIP4 | groupBy(RemoteAddressIP4, function=count()) | sort(_count, order=desc) | head(10)"},
        {"question": "Count all events", "cql": "| broken syntax("},
    ]

    for tc in test_cases:
        question = tc["question"]
        cql = tc["cql"]

        # Score 1: Validity
        v = engine.validate_query(cql)
        validity = 1 if v["is_valid"] else 0

        # Execute
        e = engine.execute(cql)
        exec_ok = e.get("success", False)
        results_str = e["preview"] if exec_ok else f"ERROR: {e.get('error', '')}"

        # Score 2: Semantic
        semantic = None
        sem_reason = ""
        if judge_fn and validity:
            try:
                raw = judge_fn(SEMANTIC_JUDGE_PROMPT.format(question=question, query=cql))
                parsed = _parse_semantic(raw)
                semantic = parsed["score"]
                sem_reason = parsed["reasoning"]
            except Exception as ex:
                sem_reason = f"Error: {ex}"

        # Score 3: Execution
        execution = None
        exe_reason = ""
        if judge_fn and exec_ok:
            try:
                raw = judge_fn(EXECUTION_JUDGE_PROMPT.format(question=question, query=cql, results=results_str))
                parsed = _parse_execution(raw)
                execution = parsed["score"]
                exe_reason = parsed["reasoning"]
            except Exception as ex:
                exe_reason = f"Error: {ex}"
        elif not judge_fn:
            execution = 5 if exec_ok else 1
            exe_reason = "success" if exec_ok else "failed"

        reward = compute_reward(validity, semantic, execution)

        icon = "+" if reward >= 0.6 else "~" if reward >= 0.3 else "-"
        print(f"\n  [{icon}] reward={reward:.3f}  Q: {question}")
        print(f"      CQL: {cql[:65]}")
        print(f"      validity={validity}  semantic={semantic}  execution={execution}")
        if sem_reason:
            print(f"      semantic:  {sem_reason[:80]}")
        if exe_reason:
            print(f"      execution: {exe_reason[:80]}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  LLM Judge: {'ON' if use_judge else 'OFF'}")
    print(f"  Repository: {engine.repository}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
