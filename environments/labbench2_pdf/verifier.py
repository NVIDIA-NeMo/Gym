#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local Harbor verifier for LitQA3, FigQA2, and TableQA2."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any


SEMANTIC_SYSTEM = (
    "Grade a submitted answer against the reference answer for a biomedical "
    "literature question. Accept semantically equivalent wording and reasonable "
    "numerical or unit equivalence, but reject answers that change the scientific "
    "fact. Return only JSON with fields result (correct, incorrect, or unsure) and rationale."
)

EXACT_SYSTEM = (
    "Grade a short answer from a scientific figure or table question. Interpret the "
    "reference and submitted answers as numbers when possible, ignoring harmless "
    "formatting and accepting scientific notation. Numeric values are correct when "
    "their absolute or relative difference is below 1e-6. For non-numeric answers, "
    "require the same entity or value, allowing only harmless formatting differences. "
    "Return only JSON with fields result (correct, incorrect, or unsure) and rationale."
)

NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - malformed task artifacts must grade as empty
        return {}
    return value if isinstance(value, dict) else {}


def normalize(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)?\s*", "", text)
    text = text.strip(" \t\r\n\"'`.,;:")
    return re.sub(r"\s+", " ", text)


def normalized_token_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9.+/-]+", " ", normalize(value)).strip()


def single_number(value: str) -> float | None:
    matches = NUMBER.findall(value.replace(",", ""))
    if len(matches) != 1:
        return None
    try:
        result = float(matches[0])
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def deterministic_match(ideal: str, answer: str) -> tuple[bool, str]:
    if normalize(ideal) == normalize(answer):
        return True, "Exact normalized match."
    if normalized_token_text(ideal) == normalized_token_text(answer):
        return True, "Equivalent after harmless punctuation normalization."
    expected_number = single_number(ideal)
    submitted_number = single_number(answer)
    if expected_number is not None and submitted_number is not None:
        difference = abs(expected_number - submitted_number)
        scale = max(abs(expected_number), abs(submitted_number), 1.0)
        if difference < 1e-6 or difference / scale < 1e-6:
            return True, "Numerically equivalent within 1e-6 tolerance."
    return False, "No deterministic exact or numeric match."


def extract_final_answer(trajectory: dict[str, Any]) -> str:
    for step in reversed(trajectory.get("steps") or []):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        message = str(step.get("message") or "").strip()
        if message and not step.get("tool_calls"):
            return message
    return ""


def chat_completions_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            item["text"] for item in value if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(value or "")


def exact_fallback(ideal: str, answer: str, error: str | None = None) -> dict[str, Any]:
    matched, rationale = deterministic_match(ideal, answer)
    result: dict[str, Any] = {
        "result": "correct" if matched else "unavailable",
        "rationale": rationale if matched else "Judge unavailable and exact fallback did not match.",
        "score": 1.0 if matched else 0.0,
        "judge_available": False,
        "fallback_exact": matched,
    }
    if error:
        result["judge_error"] = error
    return result


def call_judge(tag: str, question: str, ideal: str, answer: str) -> dict[str, Any]:
    matched, rationale = deterministic_match(ideal, answer)
    if matched:
        return {
            "result": "correct",
            "rationale": rationale,
            "score": 1.0,
            "judge_available": False,
            "fallback_exact": True,
        }

    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    model = os.environ.get("JUDGE_MODEL") or ""
    if not api_key or not base_url or not model:
        return exact_fallback(ideal, answer, "judge configuration is incomplete")

    system = EXACT_SYSTEM if tag.startswith(("figqa2", "tableqa2")) else SEMANTIC_SYSTEM
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": int(os.environ.get("JUDGE_MAX_TOKENS", "2048")),
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "reference_answer": ideal,
                        "submitted_answer": answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - configured judge URL
            response_payload = json.loads(response.read().decode("utf-8"))
        content = message_text(response_payload["choices"][0]["message"]["content"])
        match = re.search(r"\{.*\}", content, re.S)
        parsed = json.loads(match.group(0) if match else content)
        outcome = str(parsed.get("result") or parsed.get("judge_result") or "unsure").casefold()
        if outcome not in {"correct", "incorrect", "unsure"}:
            raise ValueError(f"invalid judge result: {outcome!r}")
        return {
            "result": outcome,
            "rationale": str(parsed.get("rationale") or ""),
            "score": 1.0 if outcome == "correct" else 0.0,
            "judge_available": True,
            "judge_model": model,
            "fallback_exact": False,
        }
    except Exception as exc:  # noqa: BLE001 - deterministic fallback preserves the trial
        return exact_fallback(ideal, answer, f"{type(exc).__name__}: {exc}")


def main() -> None:
    app_dir = Path(os.environ.get("HARBOR_APP_DIR", "/app"))
    tests_dir = Path(os.environ.get("HARBOR_TESTS_DIR", "/tests"))
    logs_dir = Path(os.environ.get("HARBOR_LOGS_DIR", "/logs/verifier"))
    agent_logs_dir = Path(os.environ.get("HARBOR_AGENT_LOGS_DIR", "/logs/agent"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    gold = read_json(tests_dir / "gold_metadata.json")

    answer_path = app_dir / "answer.txt"
    answer = answer_path.read_text(encoding="utf-8", errors="replace").strip() if answer_path.is_file() else ""
    answer_source = "file"
    if not answer:
        answer = extract_final_answer(read_json(agent_logs_dir / "trajectory.json"))
        answer_source = "trajectory_fallback" if answer else "missing"

    tag = str(gold.get("tag") or "")
    question = str(gold.get("question") or "")
    ideal = str(gold.get("ideal") or "")
    if not answer:
        evaluation = {
            "result": "incorrect",
            "rationale": "No gradable answer was submitted.",
            "score": 0.0,
            "judge_available": False,
            "fallback_exact": False,
        }
        execution_status = "task_failure"
        error_type = "missing_answer"
    else:
        evaluation = call_judge(tag, question, ideal, answer)
        if evaluation.get("judge_available") or evaluation.get("fallback_exact"):
            execution_status = "completed"
            error_type = None
        else:
            execution_status = "infrastructure_error"
            error_type = "judge_unavailable"

    score = float(evaluation.get("score") or 0.0)
    result = {
        "schema_version": "labbench2_pdf_reward.v1",
        "execution_status": execution_status,
        "error_type": error_type,
        "reward": score,
        "score": score,
        "judge_result": evaluation.get("result"),
        "judge_rationale": evaluation.get("rationale"),
        "judge_available": evaluation.get("judge_available", False),
        "judge_model": evaluation.get("judge_model"),
        "judge_error": evaluation.get("judge_error"),
        "fallback_exact": evaluation.get("fallback_exact", False),
        "answer": answer,
        "answer_source": answer_source,
        "question": question,
        "ideal": ideal,
        "tag": tag,
        "item_id": gold.get("item_id"),
    }
    (logs_dir / "answer.txt").write_text(answer + ("\n" if answer else ""), encoding="utf-8")
    (logs_dir / "reward.txt").write_text(f"{score}\n", encoding="utf-8")
    (logs_dir / "reward.json").write_text(
        json.dumps({"reward": score}, indent=2) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "details.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Reward: {score}")
    print(f"Execution status: {execution_status}")


if __name__ == "__main__":
    main()
