"""stem_judge resources server.

LLM-as-judge rewards for STEM RL. Built from Grigor Nalbandyan's ``stem_v11``
server: the same judge prompt and verdict parser, with the inherited
``equivalence_llm_judge`` machinery (A/B swap, full-generation rescue,
per-record extraction regexes) removed, and three additions:

1. Length guards. A rollout that hit the policy's length cap, or whose final
   answer is too long for the judge, scores 0.
2. A per-row answer contract, ``answer_format``. ``boxed`` rows must put the
   answer in a ``\\boxed{}``. ``free`` rows (proofs, explanations) are judged
   on the whole final answer.
3. Rubric rows. A row with a ``rubric`` list is graded criterion by criterion;
   the reward is the share of points earned, or pass/fail at a threshold.

The policy and judge model servers are assumed to run with a reasoning parser,
so the assistant message holds only the final answer and no ``<think>`` tags.

Grading one rollout. Every early return scores 0 without calling the judge and
records why in ``grading_status``:

1. ``truncated``: the policy hit its length cap.
2. ``empty_final_answer``: no assistant text.
3. ``no_boxed`` / ``box_too_long``: boxed rows without a non-empty box, or
   with a box over ``max_box_chars``.
4. ``answer_too_long``: the judge would see the whole final answer and it is
   over ``max_answer_chars``.

Then one judge call. Answer rows end with ``Judgement: 'yes'|'no'`` (reward 1
or 0). Rubric rows end with a JSON object marking every criterion; the server
sums the points itself. A judge output that cannot be parsed scores 0
(``judge_unparsed``).
"""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import nullcontext
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The pre-reorganization import paths: the Gym checkouts the STEM RL runs use
# have only these, and newer Gym still serves them as aliases.
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json


# Same as stem_v11: strips the fixed instruction preamble that ends with the
# format line "Answer: {your final answer}", leaving only the problem.
PROBLEM_EXTRACT_REGEX = r"Answer:\s*\{your final answer\}\s*(.*)"

# stem_v11's verdict patterns, unchanged. The primary pattern is the format the
# prompt asks for; the fallback is consulted only when the primary finds nothing.
VERDICT_PATTERN = r"\*{0,2}\s*Judgement[\s*]*:[\s*]*['\"]?(yes|no)['\"]?"
VERDICT_FALLBACK_PATTERN = (
    r"\*{0,2}\s*(?:Final\s+)?(?:Judgement|Judgment|Answer|Verdict|Conclusion)"
    r"[\s*]*[:=][\s*]*"
    r"\**\s*\$?\s*(?:\\boxed\s*\{)?\s*(?:\\text\s*\{)?\s*\{?\s*['\"]?(yes|no)\b"
)


class StemJudgeConfig(BaseResourcesServerConfig):
    """Configuration for the stem_judge server.

    Template paths resolve against this server's directory, which is the cwd
    Gym runs the entrypoint with.
    """

    name: str = "stem_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # Concurrency limit for judge requests. None disables limiting.
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_system_message: Optional[str] = None
    # Placeholders: {question}, {expected_answer}, {generated_answer}.
    answer_prompt_template_fpath: str = "prompt_templates/stem_v11.txt"
    # Placeholders: {question}, {expected_answer}, {rubric}, {generated_answer}.
    rubric_prompt_template_fpath: str = "prompt_templates/stem_rubric.txt"

    # The question shown to the judge is the last user message. An explicit
    # question_extract_regex wins over extract_problem_from_prompt.
    question_extract_regex: Optional[str] = None
    extract_problem_from_prompt: bool = False

    # --- Answer contract. A row's own answer_format wins. ---
    default_answer_format: Literal["boxed", "free"] = "boxed"
    # What the judge sees on boxed rows: "final_answer" gates on the box but shows
    # the judge the whole final answer (the setup the offline study measured);
    # "box" shows only the box's content.
    boxed_judge_input: Literal["final_answer", "box"] = "final_answer"
    # Off by default: some chemistry references run to 4K characters, and correct
    # boxes to 5K, so a fixed 2K cap zeroes correct answers.
    max_box_chars: Optional[int] = Field(default=None, gt=0)
    # Applies whenever the judge sees the whole final answer.
    max_answer_chars: Optional[int] = Field(default=16000, gt=0)

    # --- Rubric rows ---
    # "fraction": reward = earned / total positive points, clipped to [0, 1].
    # "threshold": reward = 1 if that fraction >= rubric_pass_fraction, else 0.
    rubric_reward: Literal["fraction", "threshold"] = "fraction"
    rubric_pass_fraction: float = Field(default=1.0, gt=0.0, le=1.0)

    # --- Disk logging ---
    # When set, every verify() appends one JSONL record (the policy's final answer,
    # what the judge saw, the judge's output, the reward) to a file in this
    # directory, suffixed with the pid. Must be a mounted, writable path.
    generation_log_dir: Optional[str] = None
    generation_log_filename: str = "stem_judge_generations.jsonl"


class RubricCriterion(BaseModel):
    id: str = Field(min_length=1)
    # Negative points mark a penalty: the criterion describes an error.
    points: float
    criterion: str = Field(min_length=1)

    @model_validator(mode="after")
    def _points_nonzero(self) -> RubricCriterion:
        if self.points == 0:
            raise ValueError(f"rubric criterion {self.id!r} has 0 points")
        return self


class StemJudgeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    answer_format: Optional[Literal["boxed", "free"]] = None
    rubric: Optional[list[RubricCriterion]] = None

    @model_validator(mode="after")
    def _rubric_is_gradable(self) -> StemJudgeRunRequest:
        if self.rubric is None:
            return self
        if not self.rubric:
            raise ValueError("rubric is an empty list; leave it out for an answer row")
        ids = [c.id for c in self.rubric]
        if len(set(ids)) != len(ids):
            raise ValueError(f"rubric criterion ids are not unique: {ids}")
        if not any(c.points > 0 for c in self.rubric):
            raise ValueError("rubric has no criterion with positive points")
        return self


class StemJudgeVerifyRequest(StemJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    # Answer rows: "yes" or "no"; None when no verdict could be parsed.
    verdict_label: Optional[str] = None
    # Rubric rows: criterion id -> met; None when the output could not be parsed.
    criteria_met: Optional[dict[str, bool]] = None


class StemJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    answer_format: str
    # "judged", "judge_unparsed", or why the judge was not called.
    grading_status: str
    # Rubric rows only: earned / total positive points, before rubric_reward.
    rubric_score: Optional[float] = None
    judge_evaluations: list[JudgeEvaluation]


def _assistant_text(response: NeMoGymResponse) -> str:
    """Text of the last assistant message; content blocks are joined with newlines."""
    for item in reversed(response.output):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            content = getattr(item, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(t for t in (getattr(c, "text", None) for c in content) if isinstance(t, str))
            return ""
    return ""


def _is_length_truncated(response: NeMoGymResponse) -> bool:
    """True if the policy hit its length cap.

    Gym's vLLM model server maps ``finish_reason == "length"`` to
    ``incomplete_details.reason == "max_output_tokens"``.
    """
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) == "max_output_tokens"


def final_answer(response: NeMoGymResponse) -> tuple[Optional[str], str]:
    """The policy's final answer, or None and the reason there is none.

    The reasoning parser has already moved the reasoning out of the message, so
    the assistant text is the final answer.
    """
    if _is_length_truncated(response):
        return None, "truncated"
    text = _assistant_text(response).strip()
    if not text:
        return None, "empty_final_answer"
    return text, "ok"


def last_boxed(text: str) -> Optional[str]:
    r"""Content of the last ``\boxed{...}``, brace-balanced; None if absent, unclosed or blank.

    Same rule as math_with_judge's ``_extract_last_boxed_answer``: only the last
    ``\boxed{`` counts, and a blank box is no answer.
    """
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    content_start = start + len("\\boxed{")
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                content = text[content_start:index]
                return content if content.strip() else None
    return None


def parse_verdict(text: str) -> Optional[str]:
    """The judge's last "yes"/"no", or None."""
    for pattern in (VERDICT_PATTERN, VERDICT_FALLBACK_PATTERN):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower()
    return None


def _brace_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every balanced {...} span, in order of start."""
    spans, stack = [], []
    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            spans.append((stack.pop(), index))
    return sorted(spans)


def parse_criteria(text: str, ids: list[str]) -> Optional[dict[str, bool]]:
    """The judge's marks for every criterion id, from the last JSON object that covers them all.

    A value must be a JSON boolean or the string "true"/"false". Returns None when no
    object marks every id.
    """
    for start, end in reversed(_brace_spans(text)):
        try:
            obj = json.loads(text[start : end + 1])
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        marks: dict[str, bool] = {}
        for cid in ids:
            value = obj.get(cid)
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            if not isinstance(value, bool):
                break
            marks[cid] = value
        else:
            return marks
    return None


LEADING_MINUS = re.compile(r"^(\s*)-\s+(?=\S)")
LIST_ITEM = re.compile(r"^\s*[-*+]\s", re.MULTILINE)


def render_reference(expected: str) -> str:
    """The reference as the judge prompt shows it.

    The templates put the reference on a line of its own, so "- (π^2/9) m^2" reads as a
    markdown list item and the judge drops the minus sign. Close the gap after a leading
    minus unless the reference is a list, i.e. has two or more item lines.
    """
    if LEADING_MINUS.match(expected) and len(LIST_ITEM.findall(expected)) == 1:
        return LEADING_MINUS.sub(r"\1-", expected, count=1)
    return expected


def render_rubric(rubric: list[RubricCriterion]) -> str:
    lines = []
    for c in rubric:
        points = f"{c.points:g}"
        label = f"{points} points" if c.points > 0 else f"penalty, {points} points"
        lines.append(f"[{c.id}] ({label}) {c.criterion}")
    return "\n".join(lines)


def rubric_fraction(rubric: list[RubricCriterion], met: dict[str, bool]) -> float:
    earned = sum(c.points for c in rubric if met[c.id])
    total = sum(c.points for c in rubric if c.points > 0)
    return min(max(earned / total, 0.0), 1.0)


def _extract_question_text(params: NeMoGymResponseCreateParamsNonStreaming, regex: Optional[str]) -> str:
    """The last user message; with ``regex``, its last match's first non-empty group."""
    last_text: Optional[str] = None
    for m in params.input or []:
        if getattr(m, "role", None) == "user":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                last_text = c
    text = (last_text or "").strip()
    if not text or not regex:
        return text
    matches = list(re.finditer(regex, text, flags=re.MULTILINE | re.DOTALL))
    if not matches:
        return text
    m = matches[-1]
    for group in m.groups():
        if isinstance(group, str) and group.strip():
            return group.strip()
    return m.group(0).strip()


def _extract_step(req: BaseVerifyRequest) -> Optional[str]:
    """The training step the trainer put in ``responses_create_params.metadata["step"]``."""
    metadata = getattr(getattr(req, "responses_create_params", None), "metadata", None)
    if isinstance(metadata, dict) and metadata.get("step") is not None:
        return str(metadata["step"])
    return None


class StemJudgeResourcesServer(SimpleResourcesServer):
    config: StemJudgeConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = nullcontext()

        with open(self.config.answer_prompt_template_fpath) as f:
            self._answer_template = f.read().strip()
        with open(self.config.rubric_prompt_template_fpath) as f:
            self._rubric_template = f.read().strip()

        # The pid suffix keeps server replicas from sharing one file; appends within
        # one process are serialized by the event loop (no await between open and close).
        self._gen_log_path: Optional[str] = None
        if self.config.generation_log_dir:
            try:
                os.makedirs(self.config.generation_log_dir, exist_ok=True)
                base, ext = os.path.splitext(self.config.generation_log_filename)
                self._gen_log_path = os.path.join(
                    self.config.generation_log_dir, f"{base}.{os.getpid()}{ext or '.jsonl'}"
                )
            except Exception as e:
                print(f"DEBUG: stem_judge generation logging disabled: {type(e).__name__} {e}", flush=True)

    def _question_regex(self) -> Optional[str]:
        if self.config.question_extract_regex:
            return self.config.question_extract_regex
        return PROBLEM_EXTRACT_REGEX if self.config.extract_problem_from_prompt else None

    async def verify(self, body: StemJudgeVerifyRequest) -> StemJudgeVerifyResponse:
        cfg = self.config
        expected = str(body.expected_answer or (body.metadata or {}).get("expected_answer") or "")
        answer_format = body.answer_format or cfg.default_answer_format

        def zero(status: str, judged: Optional[str] = None) -> StemJudgeVerifyResponse:
            return self._finish(body, expected, answer_format, status, 0.0, judged, [])

        final, status = final_answer(body.response)
        if final is None:
            return zero(status)

        solution = final
        if answer_format == "boxed":
            box = last_boxed(final)
            if box is None:
                return zero("no_boxed")
            if cfg.max_box_chars is not None and len(box) > cfg.max_box_chars:
                return zero("box_too_long")
            if cfg.boxed_judge_input == "box":
                solution = box
        judges_whole_answer = answer_format == "free" or cfg.boxed_judge_input == "final_answer"
        if judges_whole_answer and cfg.max_answer_chars is not None and len(final) > cfg.max_answer_chars:
            return zero("answer_too_long")

        question = _extract_question_text(body.responses_create_params, self._question_regex())
        if body.rubric:
            return await self._grade_rubric(body, expected, answer_format, question, solution)
        return await self._grade_answer(body, expected, answer_format, question, solution)

    async def _grade_answer(
        self, body: StemJudgeVerifyRequest, expected: str, answer_format: str, question: str, solution: str
    ) -> StemJudgeVerifyResponse:
        prompt = self._answer_template.format(
            question=question, expected_answer=render_reference(expected), generated_answer=solution
        )
        evaluation, text = await self._call_judge(prompt)
        evaluation.verdict_label = parse_verdict(text) if text is not None else None
        if evaluation.verdict_label is None:
            return self._finish(body, expected, answer_format, "judge_unparsed", 0.0, solution, [evaluation])
        reward = 1.0 if evaluation.verdict_label == "yes" else 0.0
        return self._finish(body, expected, answer_format, "judged", reward, solution, [evaluation])

    async def _grade_rubric(
        self, body: StemJudgeVerifyRequest, expected: str, answer_format: str, question: str, solution: str
    ) -> StemJudgeVerifyResponse:
        rubric = body.rubric
        prompt = self._rubric_template.format(
            question=question,
            expected_answer=render_reference(expected) or "(none)",
            rubric=render_rubric(rubric),
            generated_answer=solution,
        )
        evaluation, text = await self._call_judge(prompt)
        evaluation.criteria_met = parse_criteria(text, [c.id for c in rubric]) if text is not None else None
        if evaluation.criteria_met is None:
            return self._finish(body, expected, answer_format, "judge_unparsed", 0.0, solution, [evaluation])
        score = rubric_fraction(rubric, evaluation.criteria_met)
        if self.config.rubric_reward == "fraction":
            reward = score
        else:
            reward = 1.0 if score >= self.config.rubric_pass_fraction else 0.0
        return self._finish(
            body, expected, answer_format, "judged", reward, solution, [evaluation], rubric_score=score
        )

    async def _call_judge(self, prompt: str) -> tuple[JudgeEvaluation, Optional[str]]:
        """One judge call. Returns the evaluation and the text of the judge's last message, if any."""
        cfg = self.config
        params = cfg.judge_responses_create_params.model_copy(deep=True)
        msgs: list[NeMoGymEasyInputMessage] = []
        if cfg.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=cfg.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        params.input = msgs

        async with self._judge_endpoint_max_concurrency:
            try:
                response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name, url_path="/v1/responses", json=params
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            except Exception as e:
                print(f"DEBUG: stem_judge: judge model server HTTP POST error: {type(e).__name__} {e}", flush=True)
                raise

        evaluation = JudgeEvaluation(responses_create_params=params, response=judge_response)
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                return evaluation, None
            text = getattr(last_output.content[-1], "text", "")
        except Exception:
            return evaluation, None
        return evaluation, text

    def _finish(
        self,
        body: StemJudgeVerifyRequest,
        expected: str,
        answer_format: str,
        status: str,
        reward: float,
        judged: Optional[str],
        evaluations: list[JudgeEvaluation],
        rubric_score: Optional[float] = None,
    ) -> StemJudgeVerifyResponse:
        self._log(body, expected, answer_format, status, reward, judged, evaluations)
        payload = body.model_dump()
        for key in ("expected_answer", "answer_format"):
            payload.pop(key, None)
        return StemJudgeVerifyResponse(
            **payload,
            reward=reward,
            expected_answer=expected,
            answer_format=answer_format,
            grading_status=status,
            rubric_score=rubric_score,
            judge_evaluations=evaluations,
        )

    def _log(
        self,
        body: StemJudgeVerifyRequest,
        expected: str,
        answer_format: str,
        status: str,
        reward: float,
        judged: Optional[str],
        evaluations: list[JudgeEvaluation],
    ) -> None:
        """Append one JSONL record. Never raises into the verify path."""
        if not self._gen_log_path:
            return
        try:
            judge_texts = []
            for e in evaluations:
                try:
                    judge_texts.append(getattr(e.response.output[-1].content[-1], "text", "") or "")
                except Exception:
                    judge_texts.append("")
            record = {
                "ts": time.time(),
                "step": _extract_step(body),
                "id": getattr(body, "id", None) or body.uuid,
                "question": _extract_question_text(body.responses_create_params, self._question_regex()),
                "expected_answer": expected,
                "answer_format": answer_format,
                "rubric": [c.model_dump() for c in body.rubric] if body.rubric else None,
                "generation": _assistant_text(body.response),
                "judged_generation": judged,
                "grading_status": status,
                "reward": reward,
                "verdicts": [e.verdict_label for e in evaluations],
                "criteria_met": [e.criteria_met for e in evaluations],
                "judge_generations": judge_texts,
            }
            with open(self._gen_log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"DEBUG: stem_judge generation logging failed: {type(e).__name__} {e}", flush=True)


if __name__ == "__main__":
    StemJudgeResourcesServer.run_webserver()
