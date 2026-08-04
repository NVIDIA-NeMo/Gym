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
import os
import random
import re
from pathlib import Path
from typing import Any, Optional

from nemo_gym.judge import JudgeError
from resources_servers.gdpval.judge_panel import ResolvedJudge, merge_create_kwargs, sample_judge


# ---------------------------------------------------------------------------
# Structured scoring constants (structured format)
# ---------------------------------------------------------------------------
FINAL_SCORE_TAG = "FINAL_SCORE"
MAX_POSSIBLE_SCORE_TAG = "MAX_POSSIBLE_SCORE"
# The OpenAI SDK otherwise applies a 600-second timeout plus automatic retries.
# Image-dense local-judge requests can legitimately run longer, so allow one
# bounded 60-minute attempt and disable SDK-level multiplication of that bound.
STRUCTURED_JUDGE_REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get("GDPVAL_STRUCTURED_JUDGE_REQUEST_TIMEOUT_SECONDS", "3600")
)

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
_CRITERION_GRADE_RE = re.compile(
    r"CRITERION_NUMBER\[\s*(\d+)\s*\]\s*:\s*"
    r"GRADE\[\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\]\s*"
    r"out\s+of\s+MAX_POSSIBLE_POINTS\[\s*"
    r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\]",
    re.IGNORECASE,
)


def parse_structured_score(response_text: str) -> tuple[float | None, float | None]:
    """Extract ``FINAL_SCORE[x]`` and ``MAX_POSSIBLE_SCORE[y]`` from judge response.

    Returns ``(score, max_possible_score)`` or ``(None, None)`` if not found.
    """
    score_match = _FINAL_SCORE_RE.search(response_text)
    max_match = _MAX_SCORE_RE.search(response_text)
    score = float(score_match.group(1)) if score_match else None
    max_score = float(max_match.group(1)) if max_match else None
    return score, max_score


def parse_structured_criterion_grades(response_text: str, rubric_json: Any) -> dict[str, Any]:
    """Extract and annotate per-criterion grades from structured judge text.

    The requested format is one-based, but a consistently zero-based response
    is also mapped. Raw points are retained; ``binary_grade`` is 1 only for
    full credit. Completeness diagnostics let the strict persistence path
    reject omissions, duplicates, or inconsistent totals before caching.
    """
    if isinstance(rubric_json, str):
        try:
            rubric_json = json.loads(rubric_json) if rubric_json else []
        except json.JSONDecodeError:
            rubric_json = []
    if isinstance(rubric_json, list):
        rubric_items = rubric_json
    elif isinstance(rubric_json, dict) and isinstance(rubric_json.get("criteria"), list):
        rubric_items = rubric_json["criteria"]
    else:
        rubric_items = []

    raw_grades = [
        {
            "criterion_number": int(match.group(1)),
            "awarded_points": float(match.group(2)),
            "max_possible_points": float(match.group(3)),
        }
        for match in _CRITERION_GRADE_RE.finditer(response_text)
    ]
    numbers = [grade["criterion_number"] for grade in raw_grades]
    unique_numbers = set(numbers)
    expected_count = len(rubric_items)

    if numbers and expected_count and all(1 <= number <= expected_count for number in numbers):
        numbering = "one_based"
        index_for = lambda number: number - 1
        expected_numbers = set(range(1, expected_count + 1))
    elif numbers and expected_count and all(0 <= number < expected_count for number in numbers):
        numbering = "zero_based"
        index_for = lambda number: number
        expected_numbers = set(range(expected_count))
    else:
        numbering = "unmapped"
        index_for = lambda number: None
        expected_numbers = set()

    criteria: list[dict[str, Any]] = []
    for grade in raw_grades:
        rubric_index = index_for(grade["criterion_number"])
        rubric_item = (
            rubric_items[rubric_index]
            if isinstance(rubric_index, int)
            and 0 <= rubric_index < expected_count
            and isinstance(rubric_items[rubric_index], dict)
            else {}
        )
        max_points = grade["max_possible_points"]
        criteria.append(
            {
                **grade,
                "binary_grade": int(grade["awarded_points"] >= max_points - 1e-9) if max_points > 0 else None,
                "rubric_index": rubric_index,
                "rubric_item_id": rubric_item.get("rubric_item_id"),
                "criterion": rubric_item.get("criterion"),
                "rubric_weight": rubric_item.get("score", rubric_item.get("weight")),
            }
        )

    duplicate_numbers = sorted(number for number in unique_numbers if numbers.count(number) > 1)
    missing_criterion_numbers = sorted(expected_numbers - unique_numbers)
    unexpected_criterion_numbers = sorted(unique_numbers - expected_numbers)
    complete = (
        expected_count > 0 and numbering != "unmapped" and not duplicate_numbers and unique_numbers == expected_numbers
    )
    return {
        "criteria": criteria,
        "parsed_criteria_count": len(criteria),
        "expected_criteria_count": expected_count,
        "criterion_numbering": numbering,
        "duplicate_criterion_numbers": duplicate_numbers,
        "missing_criterion_numbers": missing_criterion_numbers,
        "unexpected_criterion_numbers": unexpected_criterion_numbers,
        "complete": complete,
        "awarded_points_sum": sum(criterion["awarded_points"] for criterion in criteria),
        "max_possible_points_sum": sum(criterion["max_possible_points"] for criterion in criteria),
    }


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
    judges: list[ResolvedJudge],
    rng: Optional[random.Random] = None,
    include_raw_responses: bool = False,
) -> tuple[float, dict | None]:
    """Score a deliverable against a rubric using an LLM judge.

    Returns ``(score, judge_response)`` where *score* is a float in [0, 1]
    and *judge_response* is the parsed JSON dict from the judge (or ``None``
    on failure).

    One member of *judges* is sampled for this scoring call (see
    ``judge_panel.sample_judge``); its ``create_overrides`` (reasoning settings,
    ``max_tokens``, etc.) are merged into ``client.chat.completions.create``.
    Pass *rng* (a seeded ``random.Random``) for reproducible selection.
    """
    from openai import AsyncOpenAI

    judge = sample_judge(judges, rng or random.Random())

    rubric_str = rubric_pretty if rubric_pretty else json.dumps(rubric_json, indent=2)

    judge_prompt = _render_template(
        judge_prompt_template,
        task_prompt=task_prompt,
        rubric=rubric_str,
        deliverable_text=deliverable_text,
    )

    client = AsyncOpenAI(base_url=judge.base_url, api_key=judge.api_key)

    max_retries = 5
    base_delay = 2.0

    try:
        response = None
        for attempt in range(max_retries + 1):
            try:
                create_kwargs: dict = merge_create_kwargs(
                    {
                        "model": judge.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert evaluator. You must respond with valid JSON only.",
                            },
                            {"role": "user", "content": judge_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 65535,
                    },
                    judge.create_overrides,
                )
                response = await client.chat.completions.create(**create_kwargs)
                break
            except Exception as retry_err:
                err_str = str(retry_err)
                is_retryable = "429" in err_str or "503" in err_str or "504" in err_str or "rate" in err_str.lower()
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2**attempt) + asyncio.get_event_loop().time() % 1
                    print(
                        f"Rubric judge rate-limited (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...",
                        flush=True,
                    )
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
        raw_response_text = response_text
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

        print(f"Rubric final score: {score} (judge: {judge.name})", flush=True)
        if isinstance(result, dict):
            result["judge_name"] = judge.name
            if include_raw_responses:
                result["raw_responses"] = [raw_response_text]
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
    judges: list[ResolvedJudge],
    rng: Optional[random.Random] = None,
    include_raw_responses: bool = False,
) -> tuple[float, dict | None]:
    """Score deliverables visually using a multimodal judge (e.g., Gemini 3 Pro).

    Instead of extracted text, sends PDF renders and images as base64 content
    blocks so the judge can verify formatting, tables, charts, and structure.

    *deliverable_content_blocks* is a list of OpenAI-compatible content blocks
    (text and image_url) produced by ``file_reader.convert_deliverables_to_content_blocks()``.

    One member of *judges* is sampled for this scoring call; its
    ``create_overrides`` are merged into ``client.chat.completions.create``.
    Pass *rng* (a seeded ``random.Random``) for reproducible selection.

    Returns ``(score, judge_response)`` — same contract as ``score_with_rubric``.
    """
    from openai import AsyncOpenAI

    judge = sample_judge(judges, rng or random.Random())

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

    client = AsyncOpenAI(base_url=judge.base_url, api_key=judge.api_key)

    max_retries = 5
    base_delay = 2.0

    try:
        response = None
        for attempt in range(max_retries + 1):
            try:
                create_kwargs: dict = merge_create_kwargs(
                    {
                        "model": judge.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert evaluator. You must respond with valid JSON only.",
                            },
                            {"role": "user", "content": content},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 65535,
                    },
                    judge.create_overrides,
                )
                response = await client.chat.completions.create(**create_kwargs)
                break
            except Exception as retry_err:
                err_str = str(retry_err)
                is_retryable = "429" in err_str or "503" in err_str or "504" in err_str or "rate" in err_str.lower()
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2**attempt) + asyncio.get_event_loop().time() % 1
                    print(
                        f"Visual judge rate-limited (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        response_text = (response.choices[0].message.content or "").strip()
        raw_response_text = response_text
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

        print(f"Visual judge final score: {score} (judge: {judge.name})", flush=True)
        if isinstance(result, dict):
            result["judge_name"] = judge.name
            if include_raw_responses:
                result["raw_responses"] = [raw_response_text]
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
    judges: list[ResolvedJudge],
    rng: Optional[random.Random] = None,
    num_trials: int = 2,
    formatting_retries: int = 3,
    deliverable_content_blocks: list[dict] | None = None,
    include_raw_responses: bool = False,
    request_trace_id: str | None = None,
) -> tuple[float, dict | None]:
    """Score a deliverable using structured tagged output format.

    Uses ``FINAL_SCORE[x] out of MAX_POSSIBLE_SCORE[y]`` tags for reliable
    parsing.  Runs *num_trials* scoring rounds (each with up to
    *formatting_retries* retries on parse failure) and averages the results.
    A judge is sampled from *judges* per trial ("sample between the judges"),
    so the averaged score pools the panel; pass *rng* for reproducibility.

    Returns ``(normalized_score, metadata)`` where *normalized_score* is in
    [0, 1] and *metadata* contains per-trial scores and percentages.
    """
    from openai import APITimeoutError, AsyncOpenAI

    rng = rng or random.Random()
    # One AsyncOpenAI client per distinct upstream (base_url, api_key), reused
    # across trials that sample the same judge.
    client_cache: dict[tuple[str, str], Any] = {}

    def _client_for(judge: ResolvedJudge) -> Any:
        key = (judge.base_url, judge.api_key)
        if key not in client_cache:
            client_cache[key] = AsyncOpenAI(
                base_url=judge.base_url,
                api_key=judge.api_key,
                timeout=STRUCTURED_JUDGE_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
        return client_cache[key]

    # Compute max possible score from rubric. Different upstream formats name
    # the per-criterion point field differently — accept either ``score`` or
    # ``weight`` so multiple datasets can mix in the same training run without
    # a per-source pre-pass.
    def _criterion_points(item: Any) -> float:
        if not isinstance(item, dict):
            return 0
        for key in ("score", "weight"):
            v = item.get(key)
            if isinstance(v, (int, float)):
                return v
        return 0

    if isinstance(rubric_json, str):
        rubric_json = json.loads(rubric_json) if rubric_json else []
    if isinstance(rubric_json, list):
        rubric_items = rubric_json
    elif isinstance(rubric_json, dict) and isinstance(rubric_json.get("criteria"), list):
        rubric_items = rubric_json["criteria"]
    else:
        rubric_items = []
    max_possible = sum(_criterion_points(item) for item in rubric_items)
    criterion_maxima = [float(_criterion_points(item)) for item in rubric_items]
    criterion_count = len(rubric_items)

    def _format_number(value: float) -> str:
        return str(int(value)) if value.is_integer() else str(value)

    criterion_maxima_summary = ", ".join(
        f"{index}:{_format_number(points)}" for index, points in enumerate(criterion_maxima, start=1)
    )
    structured_requirements = (
        f"The rubric has exactly {criterion_count} criteria, numbered 1 through "
        f"{criterion_count}. Emit exactly one CRITERION_NUMBER line for every criterion "
        "in that range, with no omissions or duplicates. The required "
        f"MAX_POSSIBLE_POINTS values are [{criterion_maxima_summary}], whose sum is "
        f"{_format_number(float(max_possible))}. The FINAL_SCORE must equal the sum of "
        "all GRADE values, and MAX_POSSIBLE_SCORE must equal that stated sum.\n"
    )

    rubric_str = rubric_pretty if rubric_pretty else json.dumps(rubric_json, indent=2)
    if max_possible > 0:
        rubric_str += f"\nTotal possible score: {max_possible}\n{structured_requirements}"

    # Build message content
    content: list[dict] = []
    task_text = STRUCTURED_JUDGE_PROMPT + f"<TASK_DESCRIPTION_START>\n{task_prompt}\n<TASK_DESCRIPTION_END>\n\n"

    if deliverable_content_blocks:
        content.append({"type": "text", "text": task_text + "<SUBMISSION_START>\n"})
        content.extend(deliverable_content_blocks)
        content.append({"type": "text", "text": "\n<SUBMISSION_END>\n\n"})
    else:
        content.append(
            {
                "type": "text",
                "text": task_text + f"<SUBMISSION_START>\n{deliverable_text}\n<SUBMISSION_END>\n\n",
            }
        )

    content.append({"type": "text", "text": f"<RUBRIC_START>\n{rubric_str}\n<RUBRIC_END>\n\n"})

    messages = [{"role": "user", "content": content}]

    scores: list[float] = []
    max_scores: list[float] = []
    percentages: list[float] = []
    trial_responses: list[str] = []
    trial_judges: list[str] = []
    trial_criterion_grades: list[dict[str, Any]] = []

    for trial in range(num_trials):
        trial_num = trial + 1
        parsed_ok = False
        judge = sample_judge(judges, rng)
        client = _client_for(judge)
        create_kwargs = merge_create_kwargs(
            {"model": judge.model, "messages": messages, "temperature": 0.3, "max_tokens": 65535},
            judge.create_overrides,
        )
        if request_trace_id:
            create_kwargs["user"] = f"{request_trace_id}/trial_{trial_num}"

        retry_feedback: str | None = None
        for retry in range(formatting_retries):
            if retry_feedback is not None:
                # Do not replay the potentially long invalid response. Compact
                # diagnostics give the judge an actionable correction while
                # keeping retry context bounded.
                create_kwargs["messages"] = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was rejected for structured-output "
                            "consistency. Produce a fresh complete evaluation and obey these "
                            f"requirements exactly:\n{structured_requirements}"
                            f"Previous-response diagnostics: {retry_feedback}"
                        ),
                    }
                ]
            try:
                response = await client.chat.completions.create(**create_kwargs)
                resp_text = (response.choices[0].message.content or "").strip()
            except Exception as e:
                err_str = str(e).lower()
                # A full request-budget timeout is retried by the persisted
                # rollout/resume path, not inside the same verify call.
                is_retryable = not isinstance(e, APITimeoutError) and any(
                    marker in err_str for marker in ("429", "503", "504", "rate", "timeout")
                )
                if is_retryable and retry < formatting_retries - 1:
                    delay = 5.0 * (2**retry)
                    print(
                        f"[structured-rubric] trial {trial_num} retry {retry + 1}: {e}, retrying in {delay:.0f}s",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            score, parsed_max = parse_structured_score(resp_text)

            if score is not None and parsed_max is not None:
                # Validate max matches computed max (if we have one)
                if max_possible > 0 and abs(parsed_max - max_possible) > 0.01:
                    retry_feedback = (
                        f"MAX_POSSIBLE_SCORE was {_format_number(float(parsed_max))}, but it "
                        f"must be {_format_number(float(max_possible))}. "
                    )
                    print(
                        f"[structured-rubric] trial {trial_num} retry {retry + 1}: "
                        f"max_possible mismatch (parsed={parsed_max}, expected={max_possible})",
                        flush=True,
                    )
                    continue

                criterion_result = None
                if include_raw_responses:
                    criterion_result = parse_structured_criterion_grades(resp_text, rubric_json)
                    criterion_result.update(
                        {
                            "trial_number": trial_num,
                            "judge": judge.name,
                            "grade_sum_matches_final_score": (
                                abs(criterion_result["awarded_points_sum"] - score) <= 0.01
                            ),
                            "max_sum_matches_final_max": (
                                abs(criterion_result["max_possible_points_sum"] - parsed_max) <= 0.01
                            ),
                        }
                    )
                    if not (
                        criterion_result.get("complete") is True
                        and criterion_result["grade_sum_matches_final_score"]
                        and criterion_result["max_sum_matches_final_max"]
                    ):
                        retry_feedback = (
                            f"Parsed {criterion_result['parsed_criteria_count']} of "
                            f"{criterion_result['expected_criteria_count']} criteria; "
                            f"numbering={criterion_result['criterion_numbering']}; "
                            f"missing criterion numbers={criterion_result['missing_criterion_numbers']}; "
                            f"unexpected criterion numbers={criterion_result['unexpected_criterion_numbers']}; "
                            f"duplicate criterion numbers={criterion_result['duplicate_criterion_numbers']}; "
                            "the parsed GRADE sum must equal FINAL_SCORE and the parsed "
                            "MAX_POSSIBLE_POINTS sum must equal MAX_POSSIBLE_SCORE. "
                        )
                        print(
                            f"[structured-rubric] trial {trial_num} retry {retry + 1}/{formatting_retries}: "
                            "criterion payload is incomplete or inconsistent with final score",
                            flush=True,
                        )
                        continue

                scores.append(score)
                max_scores.append(parsed_max)
                percentages.append((score / parsed_max) * 100 if parsed_max > 0 else 0)
                trial_responses.append(resp_text)
                trial_judges.append(judge.name)
                if criterion_result is not None:
                    trial_criterion_grades.append(criterion_result)
                parsed_ok = True
                print(
                    f"[structured-rubric] trial {trial_num}: score={score}/{parsed_max} ({percentages[-1]:.1f}%)",
                    flush=True,
                )
                break
            else:
                retry_feedback = "The FINAL_SCORE and/or MAX_POSSIBLE_SCORE tags were missing or malformed. "
                print(
                    f"[structured-rubric] trial {trial_num} retry {retry + 1}/{formatting_retries}: "
                    f"failed to parse FINAL_SCORE/MAX_POSSIBLE_SCORE tags",
                    flush=True,
                )

        if not parsed_ok:
            print(f"[structured-rubric] trial {trial_num}: all retries exhausted, skipping trial", flush=True)
            if include_raw_responses:
                # Once one persisted trial is missing, later trials cannot make
                # this attempt valid; avoid spending calls on a result that must
                # be retried as a whole.
                print(
                    f"[structured-rubric] trial {trial_num}: aborting remaining trials because "
                    "a complete persisted trial set is now impossible",
                    flush=True,
                )
                break

    if include_raw_responses and len(scores) != num_trials:
        raise JudgeError(
            f"structured rubric produced an incomplete or inconsistent trial set: {len(scores)}/{num_trials} completed"
        )

    if not scores:
        print("[structured-rubric] no valid scores from any trial", flush=True)
        no_valid_metadata: dict = {"error": "no_valid_scores", "num_trials": num_trials}
        if include_raw_responses:
            no_valid_metadata["raw_responses"] = trial_responses
            no_valid_metadata["criterion_grades"] = trial_criterion_grades
        return 0.0, no_valid_metadata

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
        "trial_judges": trial_judges,
    }
    if include_raw_responses:
        metadata["raw_responses"] = trial_responses
        metadata["criterion_grades"] = trial_criterion_grades

    print(
        f"[structured-rubric] final: avg={avg_score:.1f}/{effective_max} ({avg_pct:.1f}%), "
        f"normalized={normalized:.3f}, trials={len(scores)}/{num_trials}",
        flush=True,
    )
    return normalized, metadata
