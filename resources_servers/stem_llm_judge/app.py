"""stem_llm_judge resources server.

LLM-as-judge for broad-STEM (physics, chemistry, biology) open-answer grading.
The judge is shown the problem, the reference answer and the student's solution,
and answers ``Judgement: 'yes'`` (reward 1.0) or ``Judgement: 'no'`` (reward 0.0).

Prompt: ``prompt_templates/stem_llm_judge.txt``.
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
import json
import os
import re
import time
from contextlib import nullcontext
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

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


# Strips the fixed instruction preamble every training prompt is wrapped in, whose
# last line is the format spec "Answer: {your final answer}"; group 1 is the problem
# that follows it. A prompt without the marker does not match and is used whole.
PROBLEM_EXTRACT_REGEX = r"Answer:\s*\{your final answer\}\s*(.*)"


class LLMJudgeResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "stem_llm_judge"

    judge_model_server: ModelServerRef
    # `input` is set per verify() call; the rest (temperature, max_output_tokens...)
    # comes from the config.
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_system_message: Optional[str] = None
    # Resolved relative to this server's directory (Gym runs the entrypoint there).
    # Placeholders: {question}, {expected_answer}, {generated_answer}.
    judge_prompt_template_fpath: str = "prompt_templates/stem_llm_judge.txt"

    # The question shown to the judge: an explicit regex over the last user message
    # always wins; otherwise PROBLEM_EXTRACT_REGEX is applied unless disabled. A
    # boolean because Hydra CLI overrides mangle that pattern's { } ( ) \.
    question_extract_regex: Optional[str] = None
    extract_problem_from_prompt: bool = True

    # The student solution shown to the judge. A per-row
    # template_metadata.output_regex overrides this; a regex miss keeps the full text.
    response_extract_regex: Optional[str] = None

    # Score 0 without calling the judge when the policy hit its length cap: a
    # generation cut off mid-reasoning has no final answer to grade.
    reward_zero_on_truncation: bool = False

    # Mounted, writable dir for one JSONL record per verify(). None disables it.
    generation_log_dir: Optional[str] = None
    generation_log_filename: str = "stem_llm_judge_generations.jsonl"


class LLMJudgeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class LLMJudgeVerifyRequest(LLMJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    # "yes" / "no", or None when the judge's verdict could not be parsed.
    verdict_label: Optional[str] = None


class LLMJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    judge_evaluations: list[JudgeEvaluation]


def _apply_extract_regex(text: str, extract_regex: Optional[str]) -> str:
    """Last match of ``extract_regex``, first non-empty group else the whole match.

    A miss or an invalid pattern returns ``text`` unchanged rather than an empty string.
    """
    if not extract_regex:
        return text
    try:
        matches = list(re.finditer(extract_regex, text, flags=re.MULTILINE | re.DOTALL))
    except re.error:
        matches = []
    if not matches:
        return text
    match = matches[-1]
    for group in match.groups():
        if isinstance(group, str) and group.strip() != "":
            return group.strip()
    return match.group(0).strip()


def _extract_last_assistant_text(body: BaseVerifyRequest, extract_regex: Optional[str]) -> str:
    """The last assistant message, optionally narrowed by ``extract_regex``."""
    for output in reversed(body.response.output):
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            if isinstance(content, list):
                # Some providers split one assistant message into several text blocks.
                texts = [t for t in (getattr(c, "text", None) for c in content) if isinstance(t, str)]
                text = "\n".join(texts).strip()
            elif isinstance(content, str):
                text = content.strip()
            else:
                break
            return _apply_extract_regex(text, extract_regex) if text else text
    return ""


def _extract_question_text(
    params: NeMoGymResponseCreateParamsNonStreaming,
    question_extract_regex: Optional[str],
) -> str:
    """The last user message, optionally narrowed by ``question_extract_regex``."""
    last_text: Optional[str] = None
    for message in params.input or []:
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", None)
            if isinstance(content, str):
                last_text = content
    text = (last_text or "").strip()
    return _apply_extract_regex(text, question_extract_regex) if text else text


def _is_length_truncated(body: BaseVerifyRequest) -> bool:
    """True if the policy generation was cut off by the length cap.

    The Responses API has no ``finish_reason``; the vLLM model server maps
    ``finish_reason == "length"`` to ``incomplete_details.reason == "max_output_tokens"``.
    """
    response = getattr(body, "response", None)
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) == "max_output_tokens"


def _extract_step(req: BaseVerifyRequest) -> Optional[str]:
    """The training step the trainer injected into ``responses_create_params.metadata``."""
    params = getattr(req, "responses_create_params", None)
    metadata = getattr(params, "metadata", None) if params is not None else None
    if isinstance(metadata, dict):
        step = metadata.get("step")
        return str(step) if step is not None else None
    return None


def _extract_expected_answer(req: LLMJudgeRunRequest) -> Optional[str]:
    if req.expected_answer:
        return str(req.expected_answer)
    expected = (req.metadata or {}).get("expected_answer")
    return str(expected) if expected is not None else None


class LLMJudgeResourcesServer(SimpleResourcesServer):
    """Judge-only verifier using an LLM to grade an answer against a reference."""

    config: LLMJudgeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = nullcontext()

        with open(self.config.judge_prompt_template_fpath, "r") as f:
            self._judge_prompt_template = f.read().strip()

        # The pid suffix keeps concurrent server replicas from sharing one file;
        # appends within a process are serialized by the asyncio event loop.
        self._gen_log_path: Optional[str] = None
        if self.config.generation_log_dir:
            try:
                os.makedirs(self.config.generation_log_dir, exist_ok=True)
                base, ext = os.path.splitext(self.config.generation_log_filename)
                self._gen_log_path = os.path.join(
                    self.config.generation_log_dir, f"{base}.{os.getpid()}{ext or '.jsonl'}"
                )
            except Exception as e:
                # Never let a logging-dir issue crash server spinup.
                print(f"DEBUG: stem_llm_judge generation logging disabled: {type(e).__name__} {e}", flush=True)
                self._gen_log_path = None

    def _question_extract_regex(self) -> Optional[str]:
        if self.config.question_extract_regex:
            return self.config.question_extract_regex
        if self.config.extract_problem_from_prompt:
            return PROBLEM_EXTRACT_REGEX
        return None

    def _response_extract_regex(self, body: LLMJudgeVerifyRequest) -> Optional[str]:
        metadata = getattr(body, "template_metadata", None)
        if isinstance(metadata, dict) and metadata.get("output_regex"):
            return metadata["output_regex"]
        return self.config.response_extract_regex

    @staticmethod
    def _eval_text(eval_record: JudgeEvaluation) -> str:
        try:
            return getattr(eval_record.response.output[-1].content[-1], "text", "") or ""
        except Exception:
            return ""

    def _log_generation(self, body: LLMJudgeVerifyRequest, expected: str, reward: float, evaluations: list) -> None:
        """Append one JSONL record. Never raises into the verify path."""
        if not self._gen_log_path:
            return
        try:
            record = {
                "ts": time.time(),
                "step": _extract_step(body),
                "id": getattr(body, "id", None) or getattr(body, "uuid", None),
                "question": _extract_question_text(body.responses_create_params, self._question_extract_regex()),
                "expected_answer": expected,
                "generation": _extract_last_assistant_text(body, extract_regex=None),
                "judged_generation": _extract_last_assistant_text(body, self._response_extract_regex(body)),
                "reward": reward,
                "verdicts": [e.verdict_label for e in evaluations],
                "judge_generations": [self._eval_text(e) for e in evaluations],
            }
            with open(self._gen_log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"DEBUG: stem_llm_judge generation logging failed: {type(e).__name__} {e}", flush=True)

    def _make_response(
        self, body: LLMJudgeVerifyRequest, expected: str, reward: float, evaluations: list
    ) -> LLMJudgeVerifyResponse:
        self._log_generation(body, expected, reward, evaluations)
        payload = body.model_dump()
        payload.pop("expected_answer", None)
        return LLMJudgeVerifyResponse(
            **payload, reward=reward, expected_answer=expected, judge_evaluations=evaluations
        )

    async def verify(self, body: LLMJudgeVerifyRequest) -> LLMJudgeVerifyResponse:
        expected = _extract_expected_answer(body) or ""
        question = _extract_question_text(body.responses_create_params, self._question_extract_regex())
        generated = _extract_last_assistant_text(body, self._response_extract_regex(body))

        # Neither a blank nor a truncated generation can be correct, so score them 0
        # without spending a judge call.
        if not generated.strip():
            return self._make_response(body, expected, reward=0.0, evaluations=[])
        if self.config.reward_zero_on_truncation and _is_length_truncated(body):
            return self._make_response(body, expected, reward=0.0, evaluations=[])

        equal, evaluation = await self._generate_judge_evaluation(
            question=question, expected_answer=expected, generated_answer=generated
        )
        return self._make_response(body, expected, 1.0 if equal else 0.0, [evaluation])

    def _parse_verdict(self, text: str, eval_record: JudgeEvaluation) -> tuple[bool, JudgeEvaluation]:
        """Read the verdict out of the judge's output; no parse -> not equal.

        Only the region after the judge's own last ``</think>`` counts, and the LAST
        verdict in it wins, so a judge that changes its mind is scored on its final call.
        """
        search_text = text.split("</think>")[-1].strip()

        # The prompt asks for "Judgement: 'yes'|'no'"; tolerate bold and quotes.
        primary_pattern = r"\*{0,2}\s*Judgement[\s*]*:[\s*]*['\"]?(yes|no)['\"]?"
        # Judges often close with Answer/Verdict/Conclusion instead, sometimes boxed.
        # Consulted only when the primary finds nothing, so a real "Judgement:" always
        # wins. Recovers ~20-27% of otherwise-unparsed judgements. The token stays
        # strictly yes/no: an answer restatement like "Answer: 2/3" must not match.
        fallback_pattern = (
            r"\*{0,2}\s*(?:Final\s+)?(?:Judgement|Judgment|Answer|Verdict|Conclusion)"
            r"[\s*]*[:=][\s*]*"
            r"\**\s*\$?\s*(?:\\boxed\s*\{)?\s*(?:\\text\s*\{)?\s*\{?\s*['\"]?(yes|no)\b"
        )
        matches: list[str] = []
        for pattern in (primary_pattern, fallback_pattern):
            try:
                matches = re.findall(pattern, search_text, flags=re.IGNORECASE)
            except re.error:
                matches = []
            if matches:
                break

        if not matches:
            eval_record.verdict_label = None
            return False, eval_record

        verdict = matches[-1].lower()
        eval_record.verdict_label = verdict
        return verdict == "yes", eval_record

    async def _generate_judge_evaluation(
        self, *, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        cfg = self.config

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)
        user_prompt = self._judge_prompt_template.format(
            question=question, expected_answer=expected_answer, generated_answer=generated_answer
        )

        msgs: list[NeMoGymEasyInputMessage] = []
        if cfg.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=cfg.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        async with self._judge_endpoint_max_concurrency:
            try:
                response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            except Exception as e:
                print(
                    f"DEBUG: LLMJudgeResourcesServer: judge model server HTTP POST error: {type(e).__name__} {e}",
                    flush=True,
                )
                raise e

        eval_record = JudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            verdict_label=None,
        )

        # An unexpected output shape scores 0 rather than raising.
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                return False, eval_record
            text = getattr(last_output.content[-1], "text", "")
        except Exception:
            return False, eval_record

        return self._parse_verdict(text, eval_record)


if __name__ == "__main__":
    LLMJudgeResourcesServer.run_webserver()
