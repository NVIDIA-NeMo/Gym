# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GDPVal rubric scoring via LLM judge.

Separated from the task strategy so it can be tested / reused
independently.  Provides three scoring modes:

- ``score_with_rubric`` — text-based (sends extracted text to any LLM)
- ``score_with_rubric_visual`` — multimodal (sends PDF renders to Gemini)
- ``score_with_rubric_structured`` — structured scoring with tagged output
  format, multi-trial averaging, and formatting retries
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Structured scoring constants (structured format)
# ---------------------------------------------------------------------------
FINAL_SCORE_TAG = "FINAL_SCORE"
MAX_POSSIBLE_SCORE_TAG = "MAX_POSSIBLE_SCORE"

STRUCTURED_JUDGE_PROMPT = (
    "Given a task description, reference files, an evaluation rubric, and submission file(s) for the task-- "
    "score the submission file(s) according to the rubric. Make sure the final overall score doesn't exceed "
    "the maximum score possible according to the points possible for each criterion and the sum of those "
    "points. For each criterion, give an explanation for the number of points you awarded. Then, list your "
    "awarded points in the format: 'CRITERION_NUMBER[criterion_number]: GRADE[numeric_grade] out of "
    "MAX_POSSIBLE_POINTS[max_possible_points]'. Lastly, give your final overall score in the format: "
    f"'{FINAL_SCORE_TAG}[final_score] out of {MAX_POSSIBLE_SCORE_TAG}[max_possible_score]' "
    "Each value must be surrounded by the appropriate tag with square brackets [] around each number as "
    "described above. Double check that there are no math errors in any of your score calculations.\n"
)

_FINAL_SCORE_RE = re.compile(rf"{FINAL_SCORE_TAG}\[\s*([+-]?\d+(?:\.\d+)?)\s*\]")
_MAX_SCORE_RE = re.compile(rf"{MAX_POSSIBLE_SCORE_TAG}\[\s*([+-]?\d+(?:\.\d+)?)\s*\]")


def parse_structured_score(response_text: str) -> tuple[float | None, float | None]:
    """Extract ``FINAL_SCORE[x]`` and ``MAX_POSSIBLE_SCORE[y]`` from judge response.

    Returns ``(score, max_possible_score)`` or ``(None, None)`` if not found.
    """
    score_match = _FINAL_SCORE_RE.search(response_text)
    max_match = _MAX_SCORE_RE.search(response_text)
    score = float(score_match.group(1)) if score_match else None
    max_score = float(max_match.group(1)) if max_match else None
    return score, max_score


def _render_template(template_path: str, **kwargs) -> str:
    from jinja2 import Environment

    path = Path(template_path)
    if not path.is_file():
        raise FileNotFoundError(f"Template not found at '{template_path}'.")
    template_source = path.read_text()
    return Environment().from_string(template_source).render(**kwargs)


def _score_from_truncated_json(text: str) -> float:
    """Extract a score from truncated judge JSON by averaging parsed criterion scores."""
    scores = [float(m) for m in re.findall(r'"score"\s*:\s*([\d.]+)', text)]
    if not scores:
        return 0.0
    return max(0.0, min(1.0, sum(scores) / len(scores)))


async def score_with_rubric(
    deliverable_text: str,
    rubric_json: Any,
    rubric_pretty: str,
    task_prompt: str,
    judge_prompt_template: str,
    model_base_url: str,
    model_name: str,
    api_key: str = "dummy",
) -> tuple[float, dict | None]:
    """Score a deliverable against a rubric using an LLM judge.

    Returns ``(score, judge_response)`` where *score* is a float in [0, 1]
    and *judge_response* is the parsed JSON dict from the judge (or ``None``
    on failure).
    """
    from openai import AsyncOpenAI

    rubric_str = rubric_pretty if rubric_pretty else json.dumps(rubric_json, indent=2)

    judge_prompt = _render_template(
        judge_prompt_template,
        task_prompt=task_prompt,
        rubric=rubric_str,
        deliverable_text=deliverable_text,
    )

    client = AsyncOpenAI(base_url=model_base_url, api_key=api_key)

    max_retries = 5
    base_delay = 2.0

    try:
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. You must respond with valid JSON only."},
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=8192,
                )
                break
            except Exception as retry_err:
                err_str = str(retry_err)
                is_retryable = "429" in err_str or "503" in err_str or "504" in err_str or "rate" in err_str.lower()
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + asyncio.get_event_loop().time() % 1
                    print(f"Rubric judge rate-limited (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...", flush=True)
                    await asyncio.sleep(delay)
                else:
                    raise

        content = response.choices[0].message.content
        if content is None:
            print(
                f"Rubric judge returned no text content. "
                f"Finish reason: {response.choices[0].finish_reason}. "
                f"Tool calls: {response.choices[0].message.tool_calls}",
                flush=True,
            )
            return 0.0, None

        response_text = content.strip()
        print(
            f"Rubric judge response length: {len(response_text)} chars, "
            f"finish_reason: {response.choices[0].finish_reason}",
            flush=True,
        )

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            score = _score_from_truncated_json(response_text)
            print(f"Rubric JSON was truncated, computed partial score: {score}", flush=True)
            return score, None

        print(f"Rubric judge parsed keys: {list(result.keys())}", flush=True)
        if "criteria_scores" in result:
            scores = [c.get("score", 0) for c in result["criteria_scores"] if isinstance(c, dict)]
            print(f"Criteria scores: {scores}", flush=True)
            print(f"Criteria count: {len(scores)}, mean: {sum(scores) / len(scores) if scores else 0}", flush=True)

        score = None
        for key in ["overall_score", "total_score", "score", "average_score", "final_score"]:
            if key in result:
                score = float(result[key])
                print(f"Found score under key '{key}': {score}", flush=True)
                break

        if score is None and "criteria_scores" in result:
            scores = [float(c.get("score", 0)) for c in result["criteria_scores"] if isinstance(c, dict)]
            if scores:
                score = sum(scores) / len(scores)
                print(f"No overall_score key found, computed mean of criteria: {score}", flush=True)

        if score is None:
            print(f"Could not extract score. Full result: {json.dumps(result)[:1000]}", flush=True)
            score = 0.0

        print(f"Rubric final score: {score}", flush=True)
        return max(0.0, min(1.0, score)), result

    except Exception as e:
        import traceback

        print(f"Rubric scoring failed: {e}", flush=True)
        traceback.print_exc()
        return 0.0, None


async def score_with_rubric_visual(
    deliverable_content_blocks: list[dict],
    rubric_json: Any,
    rubric_pretty: str,
    task_prompt: str,
    judge_prompt_template: str,
    model_base_url: str,
    model_name: str,
    api_key: str = "dummy",
) -> tuple[float, dict | None]:
    """Score deliverables visually using a multimodal judge (e.g., Gemini 3 Pro).

    Instead of extracted text, sends PDF renders and images as base64 content
    blocks so the judge can verify formatting, tables, charts, and structure.

    *deliverable_content_blocks* is a list of OpenAI-compatible content blocks
    (text and image_url) produced by ``file_reader.convert_deliverables_to_content_blocks()``.

    Returns ``(score, judge_response)`` — same contract as ``score_with_rubric``.
    """
    from openai import AsyncOpenAI

    rubric_str = rubric_pretty if rubric_pretty else json.dumps(rubric_json, indent=2)

    judge_text = _render_template(
        judge_prompt_template,
        task_prompt=task_prompt,
        rubric=rubric_str,
        deliverable_text="[Deliverable files are attached below as PDFs/images.]",
    )

    # Build multimodal content: prompt text + file content blocks
    content: list[dict] = [{"type": "text", "text": judge_text}]
    content.extend(deliverable_content_blocks)

    client = AsyncOpenAI(base_url=model_base_url, api_key=api_key)

    max_retries = 5
    base_delay = 2.0

    try:
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. You must respond with valid JSON only."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.1,
                    max_tokens=8192,
                )
                break
            except Exception as retry_err:
                err_str = str(retry_err)
                is_retryable = "429" in err_str or "503" in err_str or "504" in err_str or "rate" in err_str.lower()
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + asyncio.get_event_loop().time() % 1
                    print(f"Visual judge rate-limited (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...", flush=True)
                    await asyncio.sleep(delay)
                else:
                    raise

        response_text = (response.choices[0].message.content or "").strip()
        print(
            f"Visual judge response length: {len(response_text)} chars, "
            f"finish_reason: {response.choices[0].finish_reason}, "
            f"content_blocks_sent: {len(content)}",
            flush=True,
        )

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            score = _score_from_truncated_json(response_text)
            print(f"Visual judge JSON was truncated, computed partial score: {score}", flush=True)
            return score, None

        print(f"Visual judge parsed keys: {list(result.keys())}", flush=True)
        if "criteria_scores" in result:
            scores = [c.get("score", 0) for c in result["criteria_scores"] if isinstance(c, dict)]
            print(f"Criteria scores: {scores}", flush=True)

        score = None
        for key in ["overall_score", "total_score", "score", "average_score", "final_score"]:
            if key in result:
                score = float(result[key])
                print(f"Found score under key '{key}': {score}", flush=True)
                break

        if score is None and "criteria_scores" in result:
            scores = [float(c.get("score", 0)) for c in result["criteria_scores"] if isinstance(c, dict)]
            if scores:
                score = sum(scores) / len(scores)
                print(f"No overall_score key found, computed mean of criteria: {score}", flush=True)

        if score is None:
            print(f"Could not extract score. Full result: {json.dumps(result)[:1000]}", flush=True)
            score = 0.0

        print(f"Visual judge final score: {score}", flush=True)
        return max(0.0, min(1.0, score)), result

    except Exception as e:
        import traceback

        print(f"Visual rubric scoring failed: {e}", flush=True)
        traceback.print_exc()
        return 0.0, None


# ---------------------------------------------------------------------------
# Structured rubric scoring (structured format)
# ---------------------------------------------------------------------------


async def score_with_rubric_structured(
    deliverable_text: str,
    rubric_json: Any,
    rubric_pretty: str,
    task_prompt: str,
    model_base_url: str,
    model_name: str,
    api_key: str = "dummy",
    num_trials: int = 2,
    formatting_retries: int = 3,
    deliverable_content_blocks: list[dict] | None = None,
) -> tuple[float, dict | None]:
    """Score a deliverable using structured tagged output format.

    Uses ``FINAL_SCORE[x] out of MAX_POSSIBLE_SCORE[y]`` tags for reliable
    parsing.  Runs *num_trials* scoring rounds (each with up to
    *formatting_retries* retries on parse failure) and averages the results.

    Returns ``(normalized_score, metadata)`` where *normalized_score* is in
    [0, 1] and *metadata* contains per-trial scores and percentages.
    """
    from openai import AsyncOpenAI

    # Compute max possible score from rubric
    if isinstance(rubric_json, str):
        rubric_json = json.loads(rubric_json) if rubric_json else []
    if isinstance(rubric_json, list):
        max_possible = sum(item.get("score", 0) for item in rubric_json)
    elif isinstance(rubric_json, dict) and "criteria" in rubric_json:
        max_possible = sum(c.get("score", 0) for c in rubric_json["criteria"])
    else:
        max_possible = 0

    rubric_str = rubric_pretty if rubric_pretty else json.dumps(rubric_json, indent=2)
    if max_possible > 0:
        rubric_str += f"\nTotal possible score: {max_possible}\n"

    # Build message content
    content: list[dict] = []
    task_text = (
        STRUCTURED_JUDGE_PROMPT
        + f"<TASK_DESCRIPTION_START>\n{task_prompt}\n<TASK_DESCRIPTION_END>\n\n"
    )

    if deliverable_content_blocks:
        content.append({"type": "text", "text": task_text + "<SUBMISSION_START>\n"})
        content.extend(deliverable_content_blocks)
        content.append({"type": "text", "text": "\n<SUBMISSION_END>\n\n"})
    else:
        content.append({
            "type": "text",
            "text": task_text + f"<SUBMISSION_START>\n{deliverable_text}\n<SUBMISSION_END>\n\n",
        })

    content.append({"type": "text", "text": f"<RUBRIC_START>\n{rubric_str}\n<RUBRIC_END>\n\n"})

    messages = [{"role": "user", "content": content}]

    client = AsyncOpenAI(base_url=model_base_url, api_key=api_key)

    scores: list[float] = []
    max_scores: list[float] = []
    percentages: list[float] = []
    trial_responses: list[str] = []

    for trial in range(num_trials):
        trial_num = trial + 1
        parsed_ok = False

        for retry in range(formatting_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=65535,
                )
                resp_text = (response.choices[0].message.content or "").strip()
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(m in err_str for m in ("429", "503", "504", "rate", "timeout"))
                if is_retryable and retry < formatting_retries - 1:
                    delay = 5.0 * (2 ** retry)
                    print(
                        f"[structured-rubric] trial {trial_num} retry {retry + 1}: {e}, "
                        f"retrying in {delay:.0f}s",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            score, parsed_max = parse_structured_score(resp_text)

            if score is not None and parsed_max is not None:
                # Validate max matches computed max (if we have one)
                if max_possible > 0 and abs(parsed_max - max_possible) > 0.01:
                    print(
                        f"[structured-rubric] trial {trial_num} retry {retry + 1}: "
                        f"max_possible mismatch (parsed={parsed_max}, expected={max_possible})",
                        flush=True,
                    )
                    continue

                scores.append(score)
                max_scores.append(parsed_max)
                percentages.append((score / parsed_max) * 100 if parsed_max > 0 else 0)
                trial_responses.append(resp_text)
                parsed_ok = True
                print(
                    f"[structured-rubric] trial {trial_num}: score={score}/{parsed_max} "
                    f"({percentages[-1]:.1f}%)",
                    flush=True,
                )
                break
            else:
                print(
                    f"[structured-rubric] trial {trial_num} retry {retry + 1}/{formatting_retries}: "
                    f"failed to parse FINAL_SCORE/MAX_POSSIBLE_SCORE tags",
                    flush=True,
                )

        if not parsed_ok:
            print(f"[structured-rubric] trial {trial_num}: all retries exhausted, skipping trial", flush=True)

    if not scores:
        print("[structured-rubric] no valid scores from any trial", flush=True)
        return 0.0, {"error": "no_valid_scores", "num_trials": num_trials}

    avg_score = sum(scores) / len(scores)
    avg_pct = sum(percentages) / len(percentages)
    effective_max = max_scores[0] if max_scores else max_possible

    # Normalize to [0, 1]
    normalized = avg_score / effective_max if effective_max > 0 else 0.0
    normalized = max(0.0, min(1.0, normalized))

    metadata = {
        "scoring_method": "structured_rubric",
        "scores": scores,
        "max_possible_scores": max_scores,
        "score_percentages": percentages,
        "average_score": avg_score,
        "overall_score_percentage": avg_pct,
        "max_possible_score": effective_max,
        "num_trials_completed": len(scores),
        "num_trials_requested": num_trials,
    }

    print(
        f"[structured-rubric] final: avg={avg_score:.1f}/{effective_max} ({avg_pct:.1f}%), "
        f"normalized={normalized:.3f}, trials={len(scores)}/{num_trials}",
        flush=True,
    )
    return normalized, metadata
