"""
LLM-as-judge resources server.

Compares a model's generated answer to an expected answer using an LLM judge.
The judge prompt is fully configurable via server config.
"""

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
from collections import Counter
from contextlib import nullcontext
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import MEAN_PREFIX
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


# Name of the aggregate metric reporting unparseable judge verdicts. Promoted into
# key_metrics by get_key_metrics, which otherwise keeps only `mean/*` entries.
JUDGEMENT_PARSING_ISSUE_RATE = "judgement_parsing_issue_rate"


def _count_verdict_occurrences(text: str, equal_label: str, not_equal_label: str) -> tuple[int, int]:
    """Count non-overlapping occurrences of each verdict label in ``text``.

    Scans left to right and, where the two labels overlap, credits the longer one.
    A plain ``text.count()`` would report lc_judge's "INCORRECT" as containing a
    "CORRECT" too, flagging every ordinary not-equal verdict as a conflict.

    Diagnostics only -- the verdict itself is decided by the ranking in
    ``_generate_judge_evaluation``, which this function does not influence.
    """
    counts = {equal_label: 0, not_equal_label: 0}
    i = 0
    # Jump between matches with str.find rather than walking characters: judge messages
    # run to tens of thousands of characters, and a per-character Python loop costs
    # milliseconds of blocking CPU per call on an event loop shared by every rollout.
    while i < len(text):
        eq_at = text.find(equal_label, i) if equal_label else -1
        neq_at = text.find(not_equal_label, i) if not_equal_label else -1
        if eq_at < 0 and neq_at < 0:
            break
        if eq_at == neq_at:
            # Same start: one label is a prefix of the other, so the longer one is meant.
            label = equal_label if len(equal_label) > len(not_equal_label) else not_equal_label
            pos = eq_at
        elif neq_at < 0 or (eq_at >= 0 and eq_at < neq_at):
            label, pos = equal_label, eq_at
        else:
            label, pos = not_equal_label, neq_at
        counts[label] += 1
        i = pos + len(label)
    return counts[equal_label], counts[not_equal_label]


class LLMJudgeResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the LLM judge server.

    - judge_model_server: target model server to use as the judge.
    - judge_responses_create_params: base create params; input will be set per request.
    - judge_system_message: optional custom system message for the judge.
    - judge_prompt_template: optional custom prompt template. Supported placeholders:
        {question}, {expected_answer}, {generated_answer}
    - judge_equal_label / judge_not_equal_label: labels the judge must output. The
        verdict is the label whose *last* occurrence ends latest in the judge's final
        message, so a label named mid-reasoning does not override the closing verdict.
    """

    # Default logical name for this resources server
    name: str = "equivalence_llm_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Concurrency limit for judge endpoint requests. Set to None to disable limiting.
    judge_endpoint_max_concurrency: Optional[int] = 64

    judge_system_message: Optional[str] = None
    judge_prompt_template_fpath: str = "prompt_templates/equivalence_llm_judge.txt"
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"
    # Optional regex to extract the question from the last user message.
    # If provided and a match is found, the first non-empty capture group is used;
    # otherwise the full match is used.
    question_extract_regex: Optional[str] = None
    # Optional regex to extract the generated response from the last assistant message.
    # The last match is used. If capture groups exist, the first non-empty group is
    # returned; otherwise, the entire last match is used.
    response_extract_regex: Optional[str] = None
    msg_extraction_failure: str = "[NO VALID ANSWER EXTRACTED]"

    # Swap check: Run second judge pass with swapped expected/generated to detect positional bias
    check_twice_swap: bool = False
    # Reward to assign if the second (swap) pass fails. Defaults to 0.0; can be set to -1.0.
    reward_if_swap_fails: float = 0.0

    # ========================================================================
    # Per-Record Regex Features (OpenQA support)
    # ========================================================================
    # These features enable mixed datasets with different answer formats.
    # They only activate when template_metadata.output_regex is present.
    # Safe to enable by default - falls back to response_extract_regex when
    # no per-record regex is present.

    # [NEW] Enable per-record regex override from template_metadata.output_regex
    use_per_record_regex: bool = True

    # --- The following features ONLY work when use_per_record_regex=True ---

    # [NEW] If set, skip regex extraction when expected_answer length exceeds this threshold.
    # When skipped, the full generation is used instead of extracting with regex.
    # Only applies when per-record regex is present. Set to None to disable.
    extraction_length_threshold: Optional[int] = 120

    # [NEW] If true, when first pass fails, retry with full generation (no regex) for partial credit.
    # Helps recover from regex extraction failures. Only activates when per-record regex exists.
    check_full_generation_on_fail: bool = True

    # [NEW] Reward when full generation check succeeds after first pass fails.
    # Default is 0.5 (partial credit).
    reward_if_full_generation_succeeds: float = 0.5


class LLMJudgeRunRequest(BaseRunRequest):
    """Run/verify request payload.

    Compatible with MCQA-like datasets. Only `expected_answer` is required for
    grading, but `options` and `metadata` are accepted for compatibility.
    """

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    options: Optional[list[dict[str, str]]] = None
    metadata: Optional[dict[str, Any]] = None


class LLMJudgeVerifyRequest(LLMJudgeRunRequest, BaseVerifyRequest):
    pass


# Marks a rollout whose verdict is absent because the judge service failed,
# as distinct from a judge that ran and returned no parseable verdict.
JUDGE_ERROR_LABEL = "JUDGE_ERROR"


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # None when the judge could not be reached or returned an unusable payload.
    # The rollout is still recorded so the failure is visible in the artifacts
    # rather than taking down the run that produced it.
    response: Optional[NeMoGymResponse] = None
    # Extracted verdict token from judge output, e.g., "[[A=B]]" or "[[A!=B]]",
    # or JUDGE_ERROR_LABEL when the judge itself failed.
    verdict_label: Optional[str] = None
    # Ways this judge response was malformed; empty when the verdict parsed cleanly.
    # See LLMJudgeResourcesServer._flag_judgement_parsing_issue for the vocabulary.
    # Persisted into the rollout JSONL so bad rows can be sliced out after a run:
    #   jq '.judge_evaluations[].judgement_parsing_issues[]?' rollouts.jsonl | sort | uniq -c
    judgement_parsing_issues: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _drop_empty_judgement_parsing_issues(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit the issues key entirely when the verdict parsed cleanly.

        Most rows are clean, so serialising an empty list on every one of them would
        add a field to the rollout JSONL that never says anything. Absent therefore
        means "parsed cleanly" -- and equally means "written before this field
        existed", so read a missing key as no-signal rather than as a clean run.
        """
        data = handler(self)
        if not data.get("judgement_parsing_issues"):
            data.pop("judgement_parsing_issues", None)
        return data


class LLMJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    judge_evaluations: list[JudgeEvaluation]


def _extract_last_assistant_text(
    body: BaseVerifyRequest, extract_regex: Optional[str], extraction_failure_message: str = ""
) -> str:
    """Extract the last assistant message text from the response.

    - If the assistant message has multiple text blocks, they are joined with newlines.
    - If ``extract_regex`` is provided, the last regex match is used; if capture
      groups exist, the first non-empty group is returned, otherwise the full match.
    - Returns ``extraction_failure_message`` when no assistant text is available.
    """
    # Return only the last assistant message's text content.
    for o in reversed(body.response.output):
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                # Some providers split a single assistant message into multiple text blocks.
                # Join all text blocks to reconstruct the full message text.
                texts: list[str] = []
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
                text = "\n".join(texts).strip()
                if not text:
                    return extraction_failure_message
                if extract_regex:
                    try:
                        matches = list(re.finditer(extract_regex, text, flags=re.MULTILINE | re.DOTALL))
                    except re.error:
                        matches = []
                    if matches:
                        m = matches[-1]
                        groups = m.groups()
                        if groups:
                            for idx in range(1, len(groups) + 1):
                                gv = m.group(idx)
                                if isinstance(gv, str) and gv.strip() != "":
                                    return gv.strip()
                        return m.group(0).strip()
                return text
            elif isinstance(content, str):
                text = content.strip()
                if not text:
                    return extraction_failure_message
                if extract_regex:
                    try:
                        matches = list(re.finditer(extract_regex, text, flags=re.MULTILINE | re.DOTALL))
                    except re.error:
                        matches = []
                    if matches:
                        m = matches[-1]
                        groups = m.groups()
                        if groups:
                            for idx in range(1, len(groups) + 1):
                                gv = m.group(idx)
                                if isinstance(gv, str) and gv.strip() != "":
                                    return gv.strip()
                        return m.group(0).strip()
                return text
            break
    return extraction_failure_message


def _extract_expected_answer(req: LLMJudgeRunRequest) -> Optional[str]:
    if req.expected_answer:
        return str(req.expected_answer)
    md = req.metadata or {}
    exp = md.get("expected_answer")
    return str(exp) if exp is not None else None


def _extract_question_text(
    params: NeMoGymResponseCreateParamsNonStreaming,
    question_extract_regex: Optional[str],
) -> str:
    """Extract the question text from the last user message in ``params``.

    - Returns the raw last user message text by default.
    - If ``question_extract_regex`` is provided, the last regex match is used; if
      capture groups exist, the first non-empty group is returned, otherwise the
      full match.
    - Returns an empty string if no user text is available.
    """
    # Return only the last user message's text content.
    last_text: Optional[str] = None
    for m in params.input or []:
        if getattr(m, "role", None) == "user":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                last_text = c
            elif isinstance(c, list):
                # Multimodal user turns (e.g. vision rows) carry a content list;
                # join the text blocks so the judge still sees the question.
                texts: list[str] = []
                for block in c:
                    t = getattr(block, "text", None)
                    if t is None and isinstance(block, dict):
                        t = block.get("text")
                    if isinstance(t, str):
                        texts.append(t)
                if texts:
                    last_text = "\n".join(texts)
    text = (last_text or "").strip()
    if not text:
        return text
    # Optionally apply a regex to extract a portion of the question text.
    if question_extract_regex:
        try:
            matches = list(re.finditer(question_extract_regex, text, flags=re.MULTILINE | re.DOTALL))
        except re.error:
            matches = []
        if matches:
            m = matches[-1]  # Use the last match
            # Prefer first non-empty capturing group, else the entire match.
            groups = m.groups()
            if groups:
                for idx in range(1, len(groups) + 1):
                    gv = m.group(idx)
                    if isinstance(gv, str) and gv.strip() != "":
                        return gv.strip()
            return m.group(0).strip()
    return text


class LLMJudgeResourcesServer(SimpleResourcesServer):
    """Judge-only verifier using an LLM to compare answers."""

    config: LLMJudgeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = nullcontext()

        with open(self.config.judge_prompt_template_fpath, "r") as f:
            self._judge_prompt_template = f.read().strip()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    @staticmethod
    def _flag_judgement_parsing_issue(kind: str, eval_record: JudgeEvaluation) -> None:
        """Record on the row that the judge's verdict did not parse cleanly.

        ``kind`` is one of:
          - ``unparseable_judge_output``: the judge response carried no assistant text.
          - ``truncated_judge_output``: the judge hit its token limit, which usually
            severs the trailing verdict line.
          - ``no_verdict``: neither label appears, so the row scores 0 by default
            rather than by judgement.
          - ``conflicting_verdicts``: both labels appear; the ranking picked the last.
          - ``repeated_verdict``: one label appears more than once.

        A row can carry several of these at once (a truncated response usually also
        has no verdict), so they accumulate in a list rather than overwriting.

        Nothing is logged: these servers run at high concurrency and a broken judge
        makes *every* row bad. The flags ride the rollout JSONL instead, where a run
        can be swept in one pass:
          jq '.judge_evaluations[].judgement_parsing_issues[]?' rollouts.jsonl | sort | uniq -c
        """
        if kind not in eval_record.judgement_parsing_issues:
            eval_record.judgement_parsing_issues.append(kind)

    def _should_skip_for_length(self, body: LLMJudgeVerifyRequest, expected: str) -> bool:
        """Check if length threshold should skip second evaluation (rescue or swap).

        When length exceeds threshold AND per-record regex is present, second eval is redundant:
        - Already using full generation (no regex benefit from rescue)
        - Swap unreliable for long text

        Only applies when there's an actual per-record regex that was skipped due to length.
        """
        if not self.config.use_per_record_regex:
            return False
        if self.config.extraction_length_threshold is None:
            return False
        if len(expected) <= self.config.extraction_length_threshold:
            return False

        # Only skip if there's a per-record regex that would have been skipped
        if hasattr(body, "template_metadata") and isinstance(body.template_metadata, dict):
            if body.template_metadata.get("output_regex"):
                return True  # Per-record regex exists and was skipped due to length

        return False  # No per-record regex, length threshold doesn't apply

    def _get_extraction_regex(self, body: LLMJudgeVerifyRequest, expected: str) -> Optional[str]:
        """Determine which regex to use for extraction, considering per-record overrides and length threshold.

        Returns:
            - str: regex pattern to extract answer (default or per-record override)
            - None: use full generation (when length threshold exceeded)
        """
        extract_regex = self.config.response_extract_regex

        if self.config.use_per_record_regex:
            # Check for per-record regex override
            if hasattr(body, "template_metadata") and isinstance(body.template_metadata, dict):
                regex_override = body.template_metadata.get("output_regex")
                if regex_override:
                    extract_regex = regex_override

                    # Skip per-record regex for long expected answers (return None → full generation)
                    if self.config.extraction_length_threshold is not None:
                        if len(expected) > self.config.extraction_length_threshold:
                            extract_regex = None

        return extract_regex

    def _make_response(
        self, body: LLMJudgeVerifyRequest, expected: str, reward: float, evaluations: list
    ) -> LLMJudgeVerifyResponse:
        """Create verification response with reward and evaluations."""
        payload = body.model_dump()
        payload.pop("expected_answer", None)
        return LLMJudgeVerifyResponse(
            **payload, reward=reward, expected_answer=expected, judge_evaluations=evaluations
        )

    async def _handle_first_pass_failed(
        self,
        body: LLMJudgeVerifyRequest,
        expected: str,
        question: str,
        first_eval,
    ) -> LLMJudgeVerifyResponse:
        """Handle when first judge evaluation fails (returns not equal).

        Options:
        1. Skip rescue for long answers (already using full generation)
        2. Try rescue with full generation (for short answers with regex)
        3. Return immediate failure
        """
        # Skip rescue for long answers - already using full generation
        if self._should_skip_for_length(body, expected):
            return self._make_response(body, expected, reward=0.0, evaluations=[first_eval])

        # Try rescue if configured (only when per-record regex exists and could have failed)
        if (
            self.config.check_full_generation_on_fail
            and self.config.use_per_record_regex
            and hasattr(body, "template_metadata")
            and isinstance(body.template_metadata, dict)
            and body.template_metadata.get("output_regex")
        ):
            # Retry with full generation (no regex) - rescue from regex extraction failure
            generated_full = _extract_last_assistant_text(
                body, extract_regex=None, extraction_failure_message=self.config.msg_extraction_failure
            )
            second_equal, second_eval = await self._generate_judge_evaluation(
                question=question, expected_answer=expected, generated_answer=generated_full
            )

            reward = self.config.reward_if_full_generation_succeeds if second_equal else 0.0
            return self._make_response(body, expected, reward, [first_eval, second_eval])

        # No rescue - immediate failure
        return self._make_response(body, expected, reward=0.0, evaluations=[first_eval])

    async def _handle_first_pass_succeeded(
        self,
        body: LLMJudgeVerifyRequest,
        expected: str,
        question: str,
        generated: str,
        first_eval,
    ) -> LLMJudgeVerifyResponse:
        """Handle when first judge evaluation succeeds (returns equal).

        Options:
        1. Return immediate success (no swap check)
        2. Skip swap for long answers (unreliable for long text)
        3. Run swap check to detect positional bias
        """
        # No swap check configured
        if not self.config.check_twice_swap:
            return self._make_response(body, expected, reward=1.0, evaluations=[first_eval])

        # Skip swap for long answers
        if self._should_skip_for_length(body, expected):
            return self._make_response(body, expected, reward=1.0, evaluations=[first_eval])

        # Run swap check
        second_equal, second_eval = await self._generate_judge_evaluation(
            question=question, expected_answer=generated, generated_answer=expected
        )
        reward = 1.0 if second_equal else self.config.reward_if_swap_fails
        return self._make_response(body, expected, reward, [first_eval, second_eval])

    async def verify(self, body: LLMJudgeVerifyRequest) -> LLMJudgeVerifyResponse:
        """Verify model response by comparing with expected answer using LLM judge.

        Flow:
        1. Extract question and expected answer
        2. Determine extraction regex (per-record override, length threshold)
        3. Extract answer to judge (could be regex-extracted OR full generation)
        4. Run first judge evaluation on extracted answer
        5. Handle failure → rescue with full generation or immediate fail
        6. Handle success → swap check or immediate success
        """
        # Step 1: Extract question and expected answer
        expected = _extract_expected_answer(body) or ""
        question = _extract_question_text(body.responses_create_params, self.config.question_extract_regex)

        # Step 2: Determine extraction regex (None if long answer triggers threshold)
        extract_regex = self._get_extraction_regex(body, expected)

        # Step 3: Extract answer to judge
        # - If extract_regex is not None → regex-extracted answer
        # - If extract_regex is None (long answer) → full generation
        generated = _extract_last_assistant_text(
            body, extract_regex, extraction_failure_message=self.config.msg_extraction_failure
        )

        # Step 4: Run first judge evaluation
        first_equal, first_eval = await self._generate_judge_evaluation(
            question=question, expected_answer=expected, generated_answer=generated
        )

        # Step 5 & 6: Handle result based on first evaluation
        if not first_equal:
            return await self._handle_first_pass_failed(body, expected, question, first_eval)
        else:
            return await self._handle_first_pass_succeeded(body, expected, question, generated, first_eval)

    async def _generate_judge_evaluation(
        self, *, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        cfg = self.config
        equal_label = cfg.judge_equal_label
        not_equal_label = cfg.judge_not_equal_label

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)
        prompt_template = self._judge_prompt_template
        system_message = cfg.judge_system_message

        user_prompt = prompt_template.format(
            question=question, expected_answer=expected_answer, generated_answer=generated_answer
        )

        msgs: list[NeMoGymEasyInputMessage] = []
        if system_message is not None and system_message != "":
            msgs.append(NeMoGymEasyInputMessage(role="system", content=system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        async with self._judge_endpoint_max_concurrency:
            try:
                judge_response = await call_judge(
                    self.server_client,
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                    response_model=NeMoGymResponse,
                )
            except JudgeError as e:
                print(
                    f"DEBUG: LLMJudgeResourcesServer: judge model server HTTP POST error: {e}",
                    flush=True,
                )
                # Do not re-raise. The judge is a separate service, and a
                # transient failure from it is not a failure of the rollout:
                # propagating here aborts the whole evaluation and discards every
                # rollout already generated, which can be many hours of work.
                # Record the failure on the rollout and score it not-equal, so
                # the run completes and a downstream check can decide whether the
                # judge-error rate makes the score untrustworthy.
                return False, JudgeEvaluation(
                    responses_create_params=responses_create_params,
                    response=None,
                    verdict_label=JUDGE_ERROR_LABEL,
                )

        eval_record = JudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            verdict_label=None,
        )

        # Parse the last output; fall back to not-equal if unexpected.
        try:
            last_output = judge_response.output[-1]
            is_message = getattr(last_output, "type", None) == "message"
            text = getattr(last_output.content[-1], "text", "") if is_message else None
        except Exception:
            text = None

        truncated = judge_response.status == "incomplete"

        if text is None:
            # Truncation that severed the whole message explains the missing text.
            if truncated:
                self._flag_judgement_parsing_issue("truncated_judge_output", eval_record)
            self._flag_judgement_parsing_issue("unparseable_judge_output", eval_record)
            return False, eval_record

        eq_count, neq_count = _count_verdict_occurrences(text, equal_label, not_equal_label)
        if eq_count == 0 and neq_count == 0:
            # Truncation is only flagged when it cost the verdict. The HLE judge prompt
            # puts `Confidence:` *after* `Judgement:`, so a judge cut off on its last
            # line routinely still committed to a verdict; flagging those would count
            # correctly graded rows as problems and inflate the rate.
            if truncated:
                self._flag_judgement_parsing_issue("truncated_judge_output", eval_record)
            self._flag_judgement_parsing_issue("no_verdict", eval_record)
        else:
            if eq_count > 0 and neq_count > 0:
                self._flag_judgement_parsing_issue("conflicting_verdicts", eval_record)
            if eq_count > 1 or neq_count > 1:
                self._flag_judgement_parsing_issue("repeated_verdict", eval_record)

        # Take the *last* occurrence of each label, not the first. Judges routinely
        # name both verdicts while reasoning ("this would be Judgement: no if the
        # units differed...") and only commit at the very end, so the first mention
        # is frequently not the verdict.
        eq_pos = text.rfind(equal_label)
        neq_pos = text.rfind(not_equal_label)
        if eq_pos < 0 and neq_pos < 0:
            eval_record.verdict_label = None
            return False, eval_record

        # Rank by where each match *ends*, not where it starts, because one label can
        # contain the other: lc_judge grades with CORRECT / INCORRECT, and every
        # "INCORRECT" also contains "CORRECT" starting one character later. Comparing
        # starts would read those as equal; comparing ends makes them tie, and the
        # longer (more specific) label then wins. An exact tie between two same-length
        # labels keeps the historical not-equal default.
        eq_end = eq_pos + len(equal_label) if eq_pos >= 0 else -1
        neq_end = neq_pos + len(not_equal_label) if neq_pos >= 0 else -1
        if eq_end > neq_end or (eq_end == neq_end and len(equal_label) > len(not_equal_label)):
            eval_record.verdict_label = equal_label
            return True, eval_record
        eval_record.verdict_label = not_equal_label
        return False, eval_record

    # -------------------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------------------

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Report how much of the run was graded by a verdict that did not parse cleanly.

        Every other signal this server produces is a reward, and a judge that stops
        emitting parseable verdicts produces the same column of zeros as a model that
        got every question wrong. This rate is what tells the two apart, so it is
        emitted even when it is 0.0 -- a missing key would mean "not measured", which
        is exactly the ambiguity the metric exists to remove.

        Not every flagged row scored wrongly: ``no_verdict`` and
        ``unparseable_judge_output`` force a 0, while ``conflicting_verdicts`` and
        ``repeated_verdict`` did produce a verdict that the ranking rule resolved. The
        per-kind rates alongside the headline are what separate the two.

        Denominator is rollouts that actually reached the judge, not all rollouts:
        a run where half the rows errored out before grading should report the rate
        among the rows the judge did see. Each rollout counts once per kind however
        many judge passes it made, so the rates read as "share of datapoints", not
        "share of judge calls" -- swap and rescue configs make two calls per row.
        """
        judged = [r for task in tasks for r in task if r.get("judge_evaluations")]
        if not judged:
            return {}

        kind_counts: Counter[str] = Counter()
        rollouts_with_issues = 0
        for rollout in judged:
            issues = {
                issue
                for evaluation in rollout["judge_evaluations"]
                for issue in ((evaluation or {}).get("judgement_parsing_issues") or [])
            }
            if issues:
                rollouts_with_issues += 1
                kind_counts.update(issues)

        metrics: dict[str, Any] = {JUDGEMENT_PARSING_ISSUE_RATE: rollouts_with_issues / len(judged)}
        # Per-kind rates only when non-zero: a clean run should not carry five zeros,
        # and the headline rate above already says the run was checked.
        for kind, count in sorted(kind_counts.items()):
            metrics[f"{JUDGEMENT_PARSING_ISSUE_RATE}/{kind}"] = count / len(judged)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """The default headline set, plus the parsing-issue rate.

        The default keeps only ``mean/*``, which would leave the rate buried in the
        full metrics JSON -- the one number that says whether the accuracy beside it
        can be trusted at all. Only the headline rate is promoted; the per-kind
        breakdown stays in ``agent_metrics`` for whoever is actually debugging.
        """
        key = {k: v for k, v in agent_metrics.items() if k.startswith(MEAN_PREFIX)}
        if JUDGEMENT_PARSING_ISSUE_RATE in agent_metrics:
            key[JUDGEMENT_PARSING_ISSUE_RATE] = agent_metrics[JUDGEMENT_PARSING_ISSUE_RATE]
        return key


if __name__ == "__main__":
    LLMJudgeResourcesServer.run_webserver()
