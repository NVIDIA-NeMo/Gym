# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = [
#   "litellm",
# ]
# ///

# General LLM Judge for evaluating agent traces against a rubric.
# This script is task-agnostic and reads the rubric from /tests/rubric.txt

import json
import os
import re
from pathlib import Path

import litellm


def parse_rubric_levels(rubric_text: str) -> dict[str, dict[str, int]]:
    """Parse the rubric into {criterion_<N>: {"A": pts, "B": pts, "C": pts}}.

    Supports the current rubric format (single `Levels: A=X B=Y C=0` header per
    criterion) and the legacy format (per-line `[A] (X points): ...`).
    """
    out: dict[str, dict[str, int]] = {}
    parts = re.split(r"^Criterion\s+(\d+)\s*:", rubric_text, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        n = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        levels: dict[str, int] = {}
        # Current format: single "Levels: A=X B=Y C=0" header
        m = re.search(
            r"Levels:\s*((?:[A-Z]=\d+\s*)+)",
            body,
        )
        if m:
            for lm in re.finditer(r"([A-Z])=(\d+)", m.group(1)):
                levels[lm.group(1).upper()] = int(lm.group(2))
        # Legacy fallback: per-line "[A] (N points)"
        if not levels:
            for lm in re.finditer(r"\[([A-Z])\]\s*\(\s*(\d+)\s*points?\s*\)", body):
                levels[lm.group(1).upper()] = int(lm.group(2))
        if levels:
            out[f"criterion_{n}"] = levels
    return out


def main():
    # Read rubric from tests directory
    rubric_path = Path("/tests/rubric.txt")
    if rubric_path.exists():
        rubric = rubric_path.read_text()
    else:
        print("ERROR: rubric.txt not found in /tests/")
        reward_path = Path("/logs/verifier/reward.json")
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(json.dumps({"score": 0}, indent=2))
        return

    # Read agent outputs (copied to /logs/verifier/ by test.sh)
    trace_path = Path("/logs/verifier/trace.md")
    answer_path = Path("/logs/verifier/answer.txt")

    trace_content = ""
    answer_content = ""

    if trace_path.exists():
        trace_content = trace_path.read_text()
    else:
        print("Warning: trace.md not found")

    if answer_path.exists():
        answer_content = answer_path.read_text()
    else:
        print("Warning: answer.txt not found")

    # If no output files exist, score is 0
    if not trace_content and not answer_content:
        print("No output files found. Score: 0")
        reward_path = Path("/logs/verifier/reward.json")
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(json.dumps({"score": 0}, indent=2))
        return

    prompt = f"""You are an expert evaluator for a data analysis task.

Evaluate the agent's work using the following rubric:

{rubric}

Here is the agent's analysis trace:

<trace>
{trace_content if trace_content else "[No trace file provided]"}
</trace>

Here is the agent's final answer:

<answer>
{answer_content if answer_content else "[No answer file provided]"}
</answer>

For each criterion in the rubric, choose ONE level: A, B, or C — based purely on which level description best describes the agent's work. Do not output numerical points; the score for each level is computed automatically from the rubric.

You MUST respond with a JSON object in exactly this format:
{{
  "criteria": {{
    "criterion_1": {{"level": "A", "reason": "<one-sentence explanation>"}},
    "criterion_2": {{"level": "B", "reason": "<one-sentence explanation>"}},
    ...
  }},
  "overall_reasoning": "<short summary>"
}}

Each "level" value must be exactly the single character "A", "B", or "C". Only output the JSON object, nothing else."""

    # Provider-aware judge call: Gemini via its OpenAI-compatible endpoint,
    # otherwise the Anthropic SDK. Selected by the MODEL_NAME prefix.
    judge_response = litellm.completion(
        model=os.getenv("JUDGE_MODEL"),
        api_base=os.getenv("JUDGE_MODEL_API_BASE"),
        api_key=os.getenv("JUDGE_MODEL_API_KEY"),
        messages=[{"role": "user", "content": prompt}],
        timeout=30,
    )
    response_text = judge_response.choices[0].message.content or ""

    print(f"Raw response (first 1000 chars): {response_text[:1000]}...")

    # Parse JSON from response
    try:
        # Try to find JSON object in response
        # Look for opening brace and find matching closing brace
        start_idx = response_text.find("{")
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
        else:
            result = json.loads(response_text)

        criteria = result.get("criteria", {})
        reasoning = result.get("overall_reasoning", result.get("reasoning", "No reasoning provided"))

        # The LLM produces only level letters (A / B / C). Map each letter to
        # its rubric-defined point value programmatically. This eliminates
        # judge arithmetic noise entirely.
        try:
            criterion_levels = parse_rubric_levels(rubric)  # {criterion_n: {"A": pts, "B": pts, "C": pts}}
        except Exception as parse_err:  # noqa: BLE001
            print(f"NOTE: failed to parse rubric levels: {parse_err}")
            criterion_levels = {}

        for k, c in list(criteria.items()):
            if not isinstance(c, dict):
                continue
            allowed = criterion_levels.get(k) or {}
            level = (c.get("level") or "").strip().upper()
            if level in allowed:
                c["score"] = allowed[level]
            elif "score" in c:
                # Legacy fallback: LLM gave a numeric score; snap to nearest allowed
                try:
                    stated = int(c.get("score", 0))
                except (TypeError, ValueError):
                    stated = 0
                if allowed:
                    target = min(allowed.values(), key=lambda v: abs(v - stated))
                    c["score"] = target
            else:
                c["score"] = 0  # missing level + missing score → no credit

        # Total = sum of programmatically-derived per-criterion scores.
        if criteria:
            criterion_sum = 0
            for c in criteria.values():
                if not isinstance(c, dict):
                    continue
                try:
                    criterion_sum += int(c.get("score", 0))
                except (TypeError, ValueError):
                    pass
            total_score = criterion_sum
        else:
            total_score = int(result.get("total_score", result.get("score", 0)))

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Response was: {response_text}")

        # Try to extract total score from text
        score_match = re.search(r'"total_score"\s*:\s*(\d+)', response_text)
        if not score_match:
            score_match = re.search(r'"score"\s*:\s*(\d+)', response_text)

        if score_match:
            total_score = int(score_match.group(1))
        else:
            total_score = 0

        criteria = {}
        reasoning = f"Failed to parse full response: {e!s}"

    # Clamp score to valid range
    total_score = max(0, min(100, total_score))

    print(f"Total Score: {total_score}/100")
    print(f"Criteria: {json.dumps(criteria, indent=2)}")
    print(f"Reasoning: {reasoning}")

    # Write reward.json with exactly ONE key (Harbor requirement)
    reward_path = Path("/logs/verifier/reward.json")
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps({"score": total_score / 100}, indent=2))

    # Write detailed evaluation to separate file (not parsed by Harbor)
    evaluation_data = {"total_score": total_score, "criteria": criteria, "reasoning": reasoning}
    evaluation_path = Path("/logs/verifier/evaluation.json")
    evaluation_path.write_text(json.dumps(evaluation_data, indent=2))


if __name__ == "__main__":
    main()
