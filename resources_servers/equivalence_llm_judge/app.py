"""Compare generated and expected answers using a configurable LLM judge."""

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

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


# Aggregate parsing metric, also included in key_metrics.
JUDGEMENT_PARSING_ISSUE_RATE = "judgement_parsing_issue_rate"


def _count_verdict_occurrences(text: str, equal_label: str, not_equal_label: str) -> tuple[int, int]:
    """Count non-overlapping verdict labels for diagnostics.

    Consume each match so INCORRECT does not also count as CORRECT.
    """
    counts = {equal_label: 0, not_equal_label: 0}
    i = 0
    # Search by match position to avoid scanning each character in Python.
    while i < len(text):
        eq_at = text.find(equal_label, i) if equal_label else -1
        neq_at = text.find(not_equal_label, i) if not_equal_label else -1
        if eq_at < 0 and neq_at < 0:
            break
        if eq_at == neq_at:
            # Prefer the longer label when both start here.
            label = equal_label if len(equal_label) > len(not_equal_label) else not_equal_label
            pos = eq_at
        elif neq_at < 0 or (eq_at >= 0 and eq_at < neq_at):
            label, pos = equal_label, eq_at
        else:
            label, pos = not_equal_label, neq_at
        counts[label] += 1
        i = pos + len(label)
    return counts[equal_label], counts[not_equal_label]


def _parse_judge_verdict(
    text: Optional[str], equal_label: str, not_equal_label: str, truncated: bool = False
) -> tuple[Optional[str], list[str]]:
    """Return the last verdict and any parsing issues."""
    if text is None:
        issues = ["unparseable_judge_output"]
        if truncated:
            issues.insert(0, "truncated_judge_output")
        return None, issues

    eq_count, neq_count = _count_verdict_occurrences(text, equal_label, not_equal_label)
    issues = []
    if eq_count == 0 and neq_count == 0:
        if truncated:
            issues.append("truncated_judge_output")
        issues.append("no_verdict")
    else:
        if eq_count > 0 and neq_count > 0:
            issues.append("conflicting_verdicts")
        if eq_count > 1 or neq_count > 1:
            issues.append("repeated_verdict")

    eq_pos = text.rfind(equal_label)
    neq_pos = text.rfind(not_equal_label)
    if eq_pos < 0 and neq_pos < 0:
        return None, issues

    # Compare match ends, then lengths, so INCORRECT wins over its CORRECT suffix.
    eq_end = eq_pos + len(equal_label) if eq_pos >= 0 else -1
    neq_end = neq_pos + len(not_equal_label) if neq_pos >= 0 else -1
    if eq_end > neq_end or (eq_end == neq_end and len(equal_label) > len(not_equal_label)):
        return equal_label, issues
    return not_equal_label, issues


class LLMJudgeResourcesServerConfig(BaseResourcesServerConfig):
    """Configure the judge, answer extraction, and optional second pass.

    Prompt placeholders: {question}, {expected_answer}, {generated_answer}.
    The verdict uses the last label by match end, with longer labels winning ties.
    """

    name: str = "equivalence_llm_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # None disables the concurrency limit.
    judge_endpoint_max_concurrency: Optional[int] = 64

    judge_system_message: Optional[str] = None
    judge_prompt_template_fpath: str = "prompt_templates/equivalence_llm_judge.txt"
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"
    # Last regex match in the user text; use the first non-empty group or full match.
    question_extract_regex: Optional[str] = None
    # Last regex match in the assistant text; use the first non-empty group or full match.
    response_extract_regex: Optional[str] = None
    msg_extraction_failure: str = "[NO VALID ANSWER EXTRACTED]"

    # Check positional bias by swapping expected and generated answers.
    check_twice_swap: bool = False
    # Reward when the swap check fails.
    reward_if_swap_fails: float = 0.0

    # Override response_extract_regex with template_metadata.output_regex when present.
    use_per_record_regex: bool = True

    # Bypass per-record regex extraction above this answer length; None disables the limit.
    extraction_length_threshold: Optional[int] = 120

    # Retry with the full response after a failed per-record regex evaluation.
    check_full_generation_on_fail: bool = True

    # Reward when the full-response retry succeeds.
    reward_if_full_generation_succeeds: float = 0.5


class LLMJudgeRunRequest(BaseRunRequest):
    """Run/verify payload with an expected answer and optional dataset metadata."""

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    options: Optional[list[dict[str, str]]] = None
    metadata: Optional[dict[str, Any]] = None


class LLMJudgeVerifyRequest(LLMJudgeRunRequest, BaseVerifyRequest):
    pass


# Verdict marker for judge-service failures.
JUDGE_ERROR_LABEL = "JUDGE_ERROR"


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # None if the judge request failed.
    response: Optional[NeMoGymResponse] = None
    # Parsed label, JUDGE_ERROR_LABEL on service failure, or None if no verdict.
    verdict_label: Optional[str] = None
    # Per-evaluation parsing issues; empty when none were detected.
    judgement_parsing_issues: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _drop_empty_judgement_parsing_issues(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit empty parsing issues; older rollouts may also lack this field."""
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
    """Join text blocks from the last assistant message.

    Apply the last regex match, using its first non-empty group or the full match.
    Return extraction_failure_message when no assistant text is available.
    """
    for o in reversed(body.response.output):
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                # Providers may split a message into multiple text blocks.
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
    """Extract user text and optionally apply the last regex match.

    Use the first non-empty capture group or the full match. Return "" if no text.
    """
    last_text: Optional[str] = None
    for m in params.input or []:
        if getattr(m, "role", None) == "user":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                last_text = c
            elif isinstance(c, list):
                # Extract text blocks from multimodal input.
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
    if question_extract_regex:
        try:
            matches = list(re.finditer(question_extract_regex, text, flags=re.MULTILINE | re.DOTALL))
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

    def _should_skip_for_length(self, body: LLMJudgeVerifyRequest, expected: str) -> bool:
        """Skip a second pass when a per-record regex was bypassed for answer length."""
        if not self.config.use_per_record_regex:
            return False
        if self.config.extraction_length_threshold is None:
            return False
        if len(expected) <= self.config.extraction_length_threshold:
            return False

        if hasattr(body, "template_metadata") and isinstance(body.template_metadata, dict):
            if body.template_metadata.get("output_regex"):
                return True

        return False

    def _get_extraction_regex(self, body: LLMJudgeVerifyRequest, expected: str) -> Optional[str]:
        """Select the response regex, applying per-record and answer-length overrides."""
        extract_regex = self.config.response_extract_regex

        if self.config.use_per_record_regex:
            if hasattr(body, "template_metadata") and isinstance(body.template_metadata, dict):
                regex_override = body.template_metadata.get("output_regex")
                if regex_override:
                    extract_regex = regex_override

                    if self.config.extraction_length_threshold is not None:
                        if len(expected) > self.config.extraction_length_threshold:
                            extract_regex = None

        return extract_regex

    def _make_response(
        self, body: LLMJudgeVerifyRequest, expected: str, reward: float, evaluations: list
    ) -> LLMJudgeVerifyResponse:
        """Create a verification response with reward and judge evaluations."""
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
        """Optionally retry a failed regex-based evaluation using the full response."""
        if self._should_skip_for_length(body, expected):
            return self._make_response(body, expected, reward=0.0, evaluations=[first_eval])

        if (
            self.config.check_full_generation_on_fail
            and self.config.use_per_record_regex
            and hasattr(body, "template_metadata")
            and isinstance(body.template_metadata, dict)
            and body.template_metadata.get("output_regex")
        ):
            generated_full = _extract_last_assistant_text(
                body, extract_regex=None, extraction_failure_message=self.config.msg_extraction_failure
            )
            second_equal, second_eval = await self._generate_judge_evaluation(
                question=question, expected_answer=expected, generated_answer=generated_full
            )

            reward = self.config.reward_if_full_generation_succeeds if second_equal else 0.0
            return self._make_response(body, expected, reward, [first_eval, second_eval])

        return self._make_response(body, expected, reward=0.0, evaluations=[first_eval])

    async def _handle_first_pass_succeeded(
        self,
        body: LLMJudgeVerifyRequest,
        expected: str,
        question: str,
        generated: str,
        first_eval,
    ) -> LLMJudgeVerifyResponse:
        """Optionally confirm a passing verdict with expected and generated answers swapped."""
        if not self.config.check_twice_swap:
            return self._make_response(body, expected, reward=1.0, evaluations=[first_eval])

        if self._should_skip_for_length(body, expected):
            return self._make_response(body, expected, reward=1.0, evaluations=[first_eval])

        second_equal, second_eval = await self._generate_judge_evaluation(
            question=question, expected_answer=generated, generated_answer=expected
        )
        reward = 1.0 if second_equal else self.config.reward_if_swap_fails
        return self._make_response(body, expected, reward, [first_eval, second_eval])

    async def verify(self, body: LLMJudgeVerifyRequest) -> LLMJudgeVerifyResponse:
        """Extract the answer, judge it, and apply an optional rescue or swap pass."""
        expected = _extract_expected_answer(body) or ""
        question = _extract_question_text(body.responses_create_params, self.config.question_extract_regex)

        extract_regex = self._get_extraction_regex(body, expected)

        generated = _extract_last_assistant_text(
            body, extract_regex, extraction_failure_message=self.config.msg_extraction_failure
        )

        first_equal, first_eval = await self._generate_judge_evaluation(
            question=question, expected_answer=expected, generated_answer=generated
        )

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
                # Preserve the rollout and mark the judge-service failure.
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

        try:
            last_output = judge_response.output[-1]
            is_message = getattr(last_output, "type", None) == "message"
            text = getattr(last_output.content[-1], "text", "") if is_message else None
        except Exception:
            text = None

        eval_record.verdict_label, eval_record.judgement_parsing_issues = _parse_judge_verdict(
            text, equal_label, not_equal_label, truncated=judge_response.status == "incomplete"
        )
        return equal_label != not_equal_label and eval_record.verdict_label == equal_label, eval_record

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Aggregate parsing issues, recovering missing diagnostics from saved responses.

        Exclude service failures and records without diagnostics or a saved response.
        Count each rollout once, including runs with multiple judge passes.
        """
        kind_counts: Counter[str] = Counter()
        judged = 0
        rollouts_with_issues = 0
        for task in tasks:
            for rollout in task:
                issues: set[str] = set()
                measured = False
                for evaluation in rollout.get("judge_evaluations") or []:
                    if not evaluation or evaluation.get("verdict_label") == JUDGE_ERROR_LABEL:
                        continue
                    parsing_issues = evaluation.get("judgement_parsing_issues")
                    if parsing_issues is None:
                        response = evaluation.get("response")
                        if not isinstance(response, dict):
                            continue
                        # Preserve the verifier's last-output, last-content-block extraction.
                        try:
                            last_output = response["output"][-1]
                            text = (
                                last_output["content"][-1].get("text", "")
                                if last_output.get("type") == "message"
                                else None
                            )
                        except (KeyError, IndexError, TypeError, AttributeError):
                            text = None
                        _, parsing_issues = _parse_judge_verdict(
                            text,
                            self.config.judge_equal_label,
                            self.config.judge_not_equal_label,
                            truncated=response.get("status") == "incomplete",
                        )
                    measured = True
                    issues.update(parsing_issues)

                if measured:
                    judged += 1
                    rollouts_with_issues += bool(issues)
                    kind_counts.update(issues)

        if not judged:
            return {}
        metrics: dict[str, Any] = {JUDGEMENT_PARSING_ISSUE_RATE: rollouts_with_issues / judged}
        for kind, count in sorted(kind_counts.items()):
            metrics[f"{JUDGEMENT_PARSING_ISSUE_RATE}/{kind}"] = count / judged
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Add the parsing-issue rate to the default mean metrics."""
        key = super().get_key_metrics(agent_metrics)
        if JUDGEMENT_PARSING_ISSUE_RATE in agent_metrics:
            key[JUDGEMENT_PARSING_ISSUE_RATE] = agent_metrics[JUDGEMENT_PARSING_ISSUE_RATE]
        return key


if __name__ == "__main__":
    LLMJudgeResourcesServer.run_webserver()
