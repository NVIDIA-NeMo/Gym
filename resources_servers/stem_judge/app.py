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
"""stem_judge resources server.

LLM-as-judge rewards for STEM RL. The answer-row judge prompt, ``stem_v11``, is the
prompt of Grigor Nalbandyan's ``stem_llm_judge`` server, byte for byte: branch
``gnalbandyan/stem_llm_judge`` at d002fd98a175865e465699bb434ddcf691bffbb6 in this
repository (https://github.com/NVIDIA-NeMo/Gym/tree/d002fd98a175865e465699bb434ddcf691bffbb6/resources_servers/stem_llm_judge,
Apache-2.0). Adapted from that server's ``app.py``: the verdict patterns, the
problem-extraction regex, the question, assistant-text, truncation and step helpers, the
``metadata.expected_answer`` fallback, the judge concurrency limit and the generation-log
record. On top of that server, this one:

1. Length guards. A rollout that hit the policy's length cap scores 0 (optional
   there, always on here), and so does a final answer too long for the judge.
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

An opt-in Math-Verify match on a box-only answer row earns 1 immediately.
Otherwise the server makes one judge call. By default, answer rows end with
``Judgement: 'yes'|'no'`` (reward 1 or 0). Rubric rows end with a JSON object
marking every criterion and the server sums the points. A judge reply that cannot be parsed scores 0
(``judge_unparsed``). A failed judge call is retried by ``call_judge`` when another
attempt can succeed (408, 429, 5xx, timeout, lost connection) and otherwise fails at
once; either way it is reported as a judge failure, which masks the row instead of
scoring it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import threading
import time
from contextlib import nullcontext
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    JUDGE_UNPARSEABLE,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    observe_llm_judge_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError, call_judge, record_parsed_judge_verdict
from resources_servers.stem_judge.answer_options import (
    AnswerRuleList,
    GuidelineScope,
    JsonJudgment,
    VerdictFormat,
    VerdictMode,
    add_answer_rules,
    assistant_text,
    answer_template,
    final_verdict,
    load_answer_rules,
    parse_json_judgment,
)
from resources_servers.stem_judge.task_data import RubricCriterion, TaskData, reference_of


logger = logging.getLogger(__name__)


# Same as stem_v11: strips the fixed instruction preamble that ends with the
# format line "Answer: {your final answer}", leaving only the problem.
PROBLEM_EXTRACT_REGEX = r"Answer:\s*\{your final answer\}\s*(.*)"

# stem_v11's verdict patterns. The primary pattern is the format the prompt asks for;
# the fallback is consulted only when the primary finds nothing. They capture the same
# verdicts as stem_v11's, rewritten so degenerate judge text cannot make them backtrack:
# the optional leading "**" and whitespace (which never changed the captured verdict) are
# dropped, and every run of whitespace or "*" is possessive, since no token after one can
# start with those characters.
VERDICT_PATTERN = r"Judgement[\s*]*+:[\s*]*+['\"]?(yes|no)['\"]?"
VERDICT_FALLBACK_PATTERN = (
    r"(?:Final\s++)?(?:Judgement|Judgment|Answer|Verdict|Conclusion)"
    r"[\s*]*+[:=][\s*]*+"
    r"\**+\s*+\$?\s*+(?:\\boxed\s*+\{)?\s*+(?:\\text\s*+\{)?\s*+\{?\s*+['\"]?(yes|no)\b"
)


class MathVerifyConfig(BaseModel, extra="forbid"):
    units: bool = Field(default=True, description="Strip recognized unit suffixes during LaTeX normalization.")
    precision: int = Field(default=6, ge=0, strict=True, description="Decimal places for comparing floats.")
    bidirectional: bool = Field(default=False, description="Require Math-Verify to match in both directions.")
    timeout_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    max_concurrency: int = Field(default=8, gt=0)


class StemJudgeConfig(BaseResourcesServerConfig):
    """Configuration for the stem_judge server.

    Template paths resolve against this server's directory, which is the cwd
    Gym runs the entrypoint with.
    """

    # verify() needs only the request, so a saved rollout can be graded again.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    name: str = "stem_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # Concurrency limit for judge requests. None disables limiting.
    judge_endpoint_max_concurrency: Optional[int] = Field(default=64, gt=0)
    judge_system_message: Optional[str] = None
    # Placeholders: {question}, {expected_answer}, {generated_answer}.
    answer_prompt_template_fpath: str = "prompt_templates/stem_v11.txt"
    # Placeholders: {question}, {expected_answer}, {rubric}, {generated_answer}.
    rubric_prompt_template_fpath: str = "prompt_templates/stem_rubric.txt"

    # The question shown to the judge is the last user message. An explicit
    # question_extract_regex wins over extract_problem_from_prompt.
    question_extract_regex: Optional[str] = None
    extract_problem_from_prompt: bool = False

    @field_validator("question_extract_regex")
    @classmethod
    def _regex_compiles(cls, regex: Optional[str]) -> Optional[str]:
        if regex:
            try:
                re.compile(regex)
            except re.error as e:
                raise ValueError(f"question_extract_regex does not compile: {e}") from e
        return regex

    guideline_scope: GuidelineScope = "stem"
    extra_answer_rules: AnswerRuleList | str = Field(default_factory=list)
    extra_answer_rules_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verdict_mode: VerdictMode = "binary"
    verdict_format: VerdictFormat = "line"
    box_parser: Literal["exact", "last_command"] = "exact"
    math_verify: MathVerifyConfig | None = None

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


class StemJudgeRunRequest(TaskData, BaseRunRequest):
    """The row contract is ``TaskData`` (task_data.py), shared with dataset tooling."""


class StemJudgeVerifyRequest(StemJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    # "Judgement: yes" or "Judgement: no" on answer rows (the form Gym's rollout recovery checks
    # against the reward); None when the reply could not be parsed, on any row. A parsed
    # rubric grade leaves the field out: its verdict is criteria_met, and Gym's recovery reads a
    # None verdict_label as a reply without a verdict.
    verdict_label: Optional[str] = None
    judgment: JsonJudgment | None = Field(default=None, exclude_if=lambda value: value is None)
    # Rubric rows: criterion id -> met; None when the output could not be parsed.
    criteria_met: Optional[dict[str, bool]] = Field(default=None, exclude_if=lambda value: value is None)

    @model_serializer(mode="wrap")
    def _omit_rubric_verdict_label(self, handler):
        data = handler(self)
        if self.criteria_met is not None and self.verdict_label is None:
            data.pop("verdict_label", None)
        return data


class StemJudgeVerifyResponse(BaseVerifyResponse):
    """Echoes the request's extra row fields (``id``, dataset labels) so saved rollouts keep them."""

    model_config = ConfigDict(extra="allow")

    expected_answer: str
    answer_format: str
    # "judged", "judge_unparsed", or why the judge was not called.
    grading_status: str
    # Rubric rows only: earned / total positive points, before rubric_reward.
    rubric_score: Optional[float] = None
    judge_evaluations: list[JudgeEvaluation]


RESPONSE_FIELDS = ("reward", "expected_answer", "answer_format", "grading_status", "rubric_score", "judge_evaluations")
FAILURE_FIELDS = (
    "mask_sample",
    "failure_kind",
    "failure_reason",
    "failure_outcome_action",
    "failure_outcome_attribution",
)


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
    text = (assistant_text(response) or "").strip()
    if not text:
        return None, "empty_final_answer"
    return text, "ok"


def last_boxed(text: str, *, last_command: bool = False) -> Optional[str]:
    r"""Content of the last ``\boxed{...}``, brace-balanced; None if absent, unclosed or blank.

    As in math_with_judge's ``_extract_last_boxed_answer``, only the last ``\boxed{``
    counts in exact mode. With last_command, a final bare or spaced command
    supersedes earlier boxes. A blank box is no answer. Escaped braces (``\{``, ``\}``) are literal, so
    ``\boxed{f=\left\{ ... \right.}`` closes at its last brace.
    """
    if last_command:
        # Select the final unescaped command first. An invalid final argument
        # belongs to that command and cannot expose an earlier boxed answer.
        opening = None
        for match in re.finditer(r"\\\\|\\boxed(?![A-Za-z])", text):
            if match[0].endswith("boxed"):
                opening = match.end()
        if opening is None:
            return None
        while opening < len(text) and text[opening].isspace():
            opening += 1
        if opening == len(text) or text[opening] != "{":
            return None
        content_start = opening + 1
    else:
        start = text.rfind("\\boxed{")
        if start < 0:
            return None
        content_start = start + len("\\boxed{")
    depth = 1
    index = content_start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                content = text[content_start:index]
                return content if content.strip() else None
        index += 1
    return None


def parse_verdict(text: str) -> Optional[str]:
    """The judge's last "yes"/"no" after its own last ``</think>``, if any; or None."""
    text = text.rsplit("</think>", 1)[-1]
    for pattern in (VERDICT_PATTERN, VERDICT_FALLBACK_PATTERN):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower()
    return None


MAX_CRITERIA_CANDIDATES = 64
# Where a JSON object with at least one key starts: "{", optional whitespace, then a quoted key.
JSON_OBJECT_START = re.compile(r'\{\s*"')
_JSON_DECODER = json.JSONDecoder()


def parse_criteria(text: str, ids: list[str]) -> Optional[dict[str, bool]]:
    """The judge's marks for every criterion id, from the last JSON object that covers them all.

    Only the text after the judge's own last ``</think>`` counts. Objects are decoded with a JSON
    decoder from each place one starts, so braces inside keys and strings (a criterion id ``r}``)
    do not matter, and only the last ``MAX_CRITERIA_CANDIDATES`` starts are tried, so degenerate
    nesting stays cheap and LaTeX braces do not crowd out the object. A value must be a JSON
    boolean or the string "true"/"false". Returns None when no object marks every id.
    """
    text = text.rsplit("</think>", 1)[-1]
    starts = [match.start() for match in JSON_OBJECT_START.finditer(text)]
    for start in reversed(starts[-MAX_CRITERIA_CANDIDATES:]):
        try:
            obj, _ = _JSON_DECODER.raw_decode(text, start)
        except (ValueError, RecursionError):
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
LIST_ITEM = re.compile(r"^[ \t]*[-*+]\s", re.MULTILINE)


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
    """The last user message; with ``regex``, its last match's first non-empty group.

    A string ``input`` is the user message itself; a content list (e.g. a row with a
    figure) contributes its text blocks, joined with newlines.
    """
    last_text: Optional[str] = params.input if isinstance(params.input, str) else None
    for m in [] if isinstance(params.input, str) else params.input or []:
        if getattr(m, "role", None) == "user":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                last_text = c
            elif isinstance(c, list):
                texts = [getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None) for b in c]
                texts = [t for t in texts if isinstance(t, str)]
                if texts:
                    last_text = "\n".join(texts)
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
    # Verification has no session state to export,
    # and a checkpoint can re-run a verification after a crash instead of waiting for it.
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"

    config: StemJudgeConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = nullcontext()

        with open(self.config.answer_prompt_template_fpath, encoding="utf-8") as f:
            template = answer_template(
                f.read().strip(),
                scope=self.config.guideline_scope,
                verdict_mode=self.config.verdict_mode,
                verdict_format=self.config.verdict_format,
            )
        rules = load_answer_rules(self.config.extra_answer_rules, self.config.extra_answer_rules_sha256)
        self._answer_template = add_answer_rules(template, rules)
        with open(self.config.rubric_prompt_template_fpath, encoding="utf-8") as f:
            self._rubric_template = f.read().strip()
        # A custom template with an unknown placeholder or a stray brace fails here, not on every judged row.
        self._answer_template.format(question="", expected_answer="", generated_answer="")
        self._rubric_template.format(question="", expected_answer="", rubric="", generated_answer="")

        self._math_library = None
        if (math_config := self.config.math_verify) is not None and not rules:
            from resources_servers.stem_judge.boxed_math import BoxedMathVerifier

            self._math_library = BoxedMathVerifier(math_config, self.config.name)

        # The host and pid suffix keeps server worker processes from sharing one file. Writes run
        # in a worker thread so a slow filesystem does not stall the event loop; the lock keeps the
        # lines of one process whole. A log file that cannot be created fails startup.
        self._gen_log_path: Optional[str] = None
        self._gen_log_lock = threading.Lock()
        self._gen_log_failed = False
        if self.config.generation_log_dir:
            os.makedirs(self.config.generation_log_dir, exist_ok=True)
            base, ext = os.path.splitext(self.config.generation_log_filename)
            suffix = f"{socket.gethostname()}.{os.getpid()}{ext or '.jsonl'}"
            self._gen_log_path = os.path.join(self.config.generation_log_dir, f"{base}.{suffix}")
            open(self._gen_log_path, "a", encoding="utf-8").close()

    def _question_regex(self) -> Optional[str]:
        if self.config.question_extract_regex:
            return self.config.question_extract_regex
        return PROBLEM_EXTRACT_REGEX if self.config.extract_problem_from_prompt else None

    async def verify(self, body: StemJudgeVerifyRequest) -> StemJudgeVerifyResponse:
        cfg = self.config
        expected = reference_of(body)
        answer_format = body.answer_format or cfg.default_answer_format

        async def zero(status: str, judged: Optional[str] = None) -> StemJudgeVerifyResponse:
            return await self._finish(body, expected, answer_format, status, 0.0, judged, [])

        final, status = final_answer(body.response)
        if final is None:
            return await zero(status)

        solution = final
        if answer_format == "boxed":
            box = last_boxed(final, last_command=cfg.box_parser == "last_command")
            if box is None:
                return await zero("no_boxed")
            if cfg.max_box_chars is not None and len(box) > cfg.max_box_chars:
                return await zero("box_too_long")
            if cfg.boxed_judge_input == "box":
                solution = box
        judges_whole_answer = answer_format == "free" or cfg.boxed_judge_input == "final_answer"
        if judges_whole_answer and cfg.max_answer_chars is not None and len(final) > cfg.max_answer_chars:
            return await zero("answer_too_long")

        if self._math_library and not body.rubric and not judges_whole_answer:
            try:
                matches = await self._math_library.matches(expected, solution)
            except OSError as exc:
                logger.warning("Math-Verify unavailable, grading with the STEM judge: %s", exc)
            else:
                if matches:
                    return await self._finish(body, expected, answer_format, "math_verified", 1.0, solution, [])

        question = _extract_question_text(body.responses_create_params, self._question_regex())
        grade = self._grade_rubric if body.rubric else self._grade_answer
        return await observe_llm_judge_grading(lambda: grade(body, expected, answer_format, question, solution))

    async def _grade_answer(
        self, body: StemJudgeVerifyRequest, expected: str, answer_format: str, question: str, solution: str
    ) -> StemJudgeVerifyResponse:
        prompt = self._answer_template.format(
            question=question, expected_answer=render_reference(expected), generated_answer=solution
        )
        json_answer = self.config.verdict_format == "json"
        evaluation, text = await self._call_judge(prompt, json_answer=json_answer)
        if json_answer:
            evaluation.judgment = parse_json_judgment(text, verdict_mode=self.config.verdict_mode)
            verdict = evaluation.judgment.label if evaluation.judgment else None
        else:
            parser = final_verdict if self.config.verdict_mode == "yes_no_unresolved" else parse_verdict
            verdict = parser(text) if text is not None else None
        evaluation.verdict_label = f"Judgement: {verdict}" if verdict is not None else None
        if evaluation.verdict_label is None:
            return await self._finish(body, expected, answer_format, "judge_unparsed", 0.0, solution, [evaluation])
        if verdict == "unresolved":
            return await self._finish(body, expected, answer_format, "judge_unresolved", 0.0, solution, [evaluation])
        record_parsed_judge_verdict()
        reward = 1.0 if verdict == "yes" else 0.0
        return await self._finish(body, expected, answer_format, "judged", reward, solution, [evaluation])

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
            return await self._finish(body, expected, answer_format, "judge_unparsed", 0.0, solution, [evaluation])
        record_parsed_judge_verdict()
        score = rubric_fraction(rubric, evaluation.criteria_met)
        if self.config.rubric_reward == "fraction":
            reward = score
        else:
            reward = 1.0 if score >= self.config.rubric_pass_fraction - 1e-9 else 0.0
        return await self._finish(
            body, expected, answer_format, "judged", reward, solution, [evaluation], rubric_score=score
        )

    async def _call_judge(self, prompt: str, *, json_answer: bool = False) -> tuple[JudgeEvaluation, Optional[str]]:
        """One judge call. Returns the evaluation and the text of the judge's last message, if any."""
        cfg = self.config
        params = cfg.judge_responses_create_params.model_copy(deep=True)
        if json_answer:
            text = params.text or {}
            if (text.get("format") or {}).get("type") not in ("json_schema", "json_object"):
                params.text = {**text, "format": {"type": "json_object"}}
        msgs: list[NeMoGymEasyInputMessage] = []
        if cfg.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=cfg.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=prompt))
        params.input = msgs

        async with self._judge_endpoint_max_concurrency:
            judge_response = await call_judge(
                self.server_client,
                server_name=cfg.judge_model_server.name,
                url_path="/v1/responses",
                json=params,
                response_model=NeMoGymResponse,
            )

        if judge_response.status == "failed" or judge_response.error is not None:
            # The judge's own failure, not a verdict on the answer: mask the row (judge_failsafe).
            raise JudgeError(f"judge response failed: {judge_response.error or judge_response.status}")
        evaluation = JudgeEvaluation(responses_create_params=params, response=judge_response)
        if judge_response.status == "incomplete":
            # A cut-off reply has no final judgement; a verdict-like line in it could be quoted text.
            return evaluation, None
        return evaluation, assistant_text(judge_response, judge_reply=True)

    async def _finish(
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
        if self._gen_log_path:
            line = self._log_line(body, expected, answer_format, status, reward, judged, evaluations)
            await asyncio.to_thread(self._append_log_line, line)
        payload = body.model_dump()
        # Fields this server sets; a row that carries them (e.g. one rebuilt from a rollout) must not override them.
        for key in (
            *RESPONSE_FIELDS,
            *FAILURE_FIELDS,
            "verdict",
            "correctness",
            "evaluation_completed",
            "library_reward",
        ):
            payload.pop(key, None)
        if status == "math_verified":
            payload["library_reward"] = reward
        if self.config.verdict_mode == "yes_no_unresolved" and not body.rubric:
            unknown = status in ("judge_unparsed", "judge_unresolved")
            payload.update(
                verdict="unresolved" if unknown else ("yes" if reward else "no"),
                evaluation_completed=not unknown,
                correctness=None if unknown else bool(reward),
                mask_sample=unknown,
            )
            if unknown:
                payload["failure_kind"] = (
                    "stem_judge:unresolved" if status == "judge_unresolved" else JUDGE_UNPARSEABLE
                )
                payload["failure_reason"] = (
                    "Judge abstained" if status == "judge_unresolved" else "Missing or incomplete final verdict"
                )
        return StemJudgeVerifyResponse(
            **payload,
            reward=reward,
            expected_answer=expected,
            answer_format=answer_format,
            grading_status=status,
            rubric_score=rubric_score,
            judge_evaluations=evaluations,
        )

    def _log_line(
        self,
        body: StemJudgeVerifyRequest,
        expected: str,
        answer_format: str,
        status: str,
        reward: float,
        judged: Optional[str],
        evaluations: list[JudgeEvaluation],
    ) -> str:
        """One JSONL record for the generation log."""
        judge_texts = [assistant_text(e.response, judge_reply=True) or "" for e in evaluations]
        record = {
            "ts": time.time(),
            "step": _extract_step(body),
            "id": getattr(body, "id", None) or body.uuid,
            "question": _extract_question_text(body.responses_create_params, self._question_regex()),
            "expected_answer": expected,
            "answer_format": answer_format,
            "rubric": [c.model_dump() for c in body.rubric] if body.rubric else None,
            "generation": assistant_text(body.response, allow_partial=True) or "",
            "judged_generation": judged,
            "grading_status": status,
            "reward": reward,
            "verdicts": [e.verdict_label for e in evaluations],
            "criteria_met": [e.criteria_met for e in evaluations],
            "judge_generations": judge_texts,
        }
        if judgments := [e.judgment.model_dump() for e in evaluations if e.judgment is not None]:
            record["judgments"] = judgments
        return json.dumps(record, ensure_ascii=False, default=str) + "\n"

    def _append_log_line(self, line: str) -> None:
        """Append one line to the generation log; a failure is logged once and never fails verify()."""
        try:
            with self._gen_log_lock, open(self._gen_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:  # noqa: BLE001 - the log is best effort and must not fail verify()
            if not self._gen_log_failed:
                self._gen_log_failed = True
                logger.warning("stem_judge generation log write failed (further failures are not logged): %s", e)


if __name__ == "__main__":
    StemJudgeResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = StemJudgeResourcesServer.run_webserver()  # noqa: F401
