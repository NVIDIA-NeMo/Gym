"""
LLM-as-judge resources server.

Compares a model's generated answer to an expected answer using an LLM judge.
The judge prompt is fully configurable via server config.
"""

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
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

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
from nemo_gym.server_utils import (
    get_global_config_dict,
)

from resources_servers.equivalence_llm_judge.equivalence_llm_judge_utils import (
    _get_request_expected_answer_text,
)


class MultistepEquivLLMJudgeResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the LLM judge server.

    - judge_model_server: target model server to use as the judge.
    - judge_responses_create_params: base create params; input will be set per request.
    - judge_system_message: optional custom system message for the judge.
    - judge_prompt_template: optional custom prompt template. Supported placeholders:
        {question}, {expected_answer}, {generated_answer}
    - judge_equal_label / judge_not_equal_label: labels the judge must output.
    """

    # Default logical name for this resources server
    name: str = "multistep_equiv_llm_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_endpoint_max_concurrency: Optional[int] = 128
    # judge_endpoint_max_num_retries: int = 2

    # TODO: deprecated config fields.
    judge_system_message: Optional[str] = None
    judge_prompt_template: Optional[str] = None
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"

    judge_extract_system_template_fpath: Optional[str] = None
    judge_extract_prompt_template_fpath: Optional[str] = None
    judge_extract_quorum_template_fpath: Optional[str] = None
    judge_distill_system_template_fpath: Optional[str] = None
    judge_distill_prompt_template_fpath: Optional[str] = None
    judge_distill_quorum_template_fpath: Optional[str] = None
    judge_compare_system_template_fpath: Optional[str] = None
    judge_compare_prompt_template_fpath: Optional[str] = None
    judge_compare_quorum_template_fpath: Optional[str] = None
    judge_verdict_prompt_template_fpath: Optional[str] = None

    quorum_max_samples: int = 1
    quorum_type: str = "majority"

    # If true, perform a second judge pass swapping expected and generated answers
    # to reduce potential positional bias. Default is True.
    swap: bool = True

    # trials: int = 1

    expected_answer_distill: bool = False
    model_response_parse_reasoning: bool = False

    base_log_dir: Optional[str] = None
    debug: bool = False


class MultistepEquivLLMJudgeRunRequest(BaseRunRequest):
    """Run/verify request payload.

    Compatible with MCQA-like datasets. Only `expected_answer` is required for
    grading, but `options` and `metadata` are accepted for compatibility.
    """

    id: Optional[str] = None
    uuid: Optional[str] = None
    expected_answer: Optional[str] = None
    grading_mode: Optional[str] = None
    options: Optional[list[dict[str, str]]] = None
    metadata: Optional[dict[str, Any]] = None
    rl_metadata: Optional[dict[str, Any]] = None


class MultistepEquivLLMJudgeVerifyRequest(MultistepEquivLLMJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    # Extracted verdict token from judge output, e.g., "[[A=B]]" or "[[A!=B]]".
    verdict_label: Optional[str] = None


class MultistepEquivLLMJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    judge_evaluations: list[JudgeEvaluation]


@dataclass
class _CompareQuorum:
    nil_count: int = 0
    neg_count: int = 0
    pos_count: int = 0

    def majority(self, max_samples: int) -> Optional[bool]:
        min_quorum = (max_samples // 2) + 1
        if self.pos_count >= min_quorum:
            return True
        elif self.neg_count >= min_quorum:
            return False
        else:
            return None


def _get_request_first_user_content_text(req) -> Optional[str]:
    # print(f"DEBUG: _get_request_first_user_content_text: req = {req}", flush=True)
    params = req.responses_create_params
    # TODO(peter)
    last_text: Optional[str] = None
    for m in params.input or []:
        if getattr(m, "role", None) == "user":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                last_text = c
    text = (last_text or "").strip()
    return text


def _get_response_content_text(response, turn: int, role: Optional[str] = None) -> Optional[str]:
    if not response.output:
        return None
    if turn >= 0 and len(response.output) <= turn:
        return None
    if role is not None and response.output[turn].role != role:
        return None
    content = response.output[turn].content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if item.type != "output_text":
                print(
                    f"DEBUG: _get_response_content_text: unexpected content item type = {repr(item.type)}",
                    flush=True,
                )
            text_parts.append(item.text)
        text = "".join(text_parts)
    else:
        raise NotImplementedError
    return text


def _get_response_first_user_content_text(response) -> Optional[str]:
    print(f"DEBUG: _get_response_first_user_content_text: response = {response}", flush=True)
    text = _get_response_content_text(response, turn=0, role="user")
    if text is None:
        text = _get_response_content_text(response, turn=1, role="user")
    return text


def _get_response_last_assistant_content_text(response) -> Optional[str]:
    return _get_response_content_text(response, turn=-1, role="assistant")


def _get_response_last_assistant_raw_response_text(response, parse_reasoning: bool = False) -> Optional[str]:
    text = _get_response_content_text(response, turn=-1, role="assistant")
    # FIXME: hardcoded reasoning end token.
    if parse_reasoning:
        reasoning_text, _, raw_response_text = text.partition("</think>")
    else:
        raw_response_text = text
    if raw_response_text.startswith("\n\n"):
        raw_response_text = raw_response_text[2:]
    elif raw_response_text.startswith("\n"):
        raw_response_text = raw_response_text[1:]
    return raw_response_text


def _extract_tagged_section(haystack: str, tag: str, strip: bool = True) -> Optional[str]:
    needle = haystack
    needle, sep, _ = needle.rpartition(f"</{tag}>")
    if not sep:
        needle = haystack
    _, sep, needle = needle.rpartition(f"<{tag}>")
    if not sep:
        return None
    if strip:
        return needle.strip()
    else:
        return needle


class MultistepEquivLLMJudgeResourcesServer(SimpleResourcesServer):
    """Judge-only verifier using an LLM to compare answers."""

    config: MultistepEquivLLMJudgeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.debug:
            print(f"DEBUG: MultistepEquivLLMJudgeResourcesServer: config = {self.config}", flush=True)

        base_log_dir = self.config.base_log_dir
        if base_log_dir is None:
            global_cfg = get_global_config_dict()
            base_log_dir = global_cfg.get("base_log_dir", None)
        if base_log_dir is not None:
            self._log_path = os.path.join(base_log_dir, f"{self.config.name}-judge_log.jsonl")
            self._log_write = asyncio.Lock()
            print(f"DEBUG: MultistepEquivLLMJudgeResourcesServer: log path = {self._log_path!r}", flush=True)
        else:
            self._log_path = None
            self._log_write = contextlib.nullcontext()
            print(f"DEBUG: MultistepEquivLLMJudgeResourcesServer: missing log path", flush=True)

        max_concurrency = self.config.judge_endpoint_max_concurrency
        if max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = contextlib.nullcontext()

        self._judge_extract_system_template = None
        self._judge_extract_prompt_template = None
        self._judge_extract_quorum_template = None

        self._judge_distill_system_template = None
        self._judge_distill_prompt_template = None
        self._judge_distill_quorum_template = None

        self._judge_compare_system_template = None
        self._judge_compare_prompt_template = None

        self._judge_verdict_prompt_template = None

        assert self.config.quorum_max_samples >= 1
        assert self.config.quorum_type == "majority", f"unsupported quorum_type: {self.config.quorum_type!r}"

        self._default_judge_params_dict = self.config.judge_responses_create_params.model_dump()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: MultistepEquivLLMJudgeVerifyRequest) -> MultistepEquivLLMJudgeVerifyResponse:
        return await self._verify_single_trial(body)

    async def _verify_single_trial(
        self, body: MultistepEquivLLMJudgeVerifyRequest
    ) -> MultistepEquivLLMJudgeVerifyResponse:
        model_params_dict = json.loads(body.responses_create_params.model_dump_json())
        model_response_dict = json.loads(body.response.model_dump_json())
        if (
            "output" in model_response_dict and
            model_response_dict["output"]
        ):
            model_response_dict["output"][0].pop("prompt_token_ids", None)
            model_response_dict["output"][0].pop("generation_token_ids", None)
            model_response_dict["output"][0].pop("generation_log_probs", None)

        # question = _get_response_first_user_content_text(body.response)
        question = _get_request_first_user_content_text(body)
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: question        = {repr(question)}",
                flush=True,
            )

        model_raw_response = _get_response_last_assistant_raw_response_text(
            body.response, parse_reasoning=self.config.model_response_parse_reasoning
        )
        expected_answer = _get_request_expected_answer_text(body) or ""

        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: expected answer = {repr(expected_answer)}",
                flush=True,
            )

        if not model_raw_response:
            payload = body.model_dump()
            payload.pop("expected_answer", None)
            reward = 0.0
            return MultistepEquivLLMJudgeVerifyResponse(
                **payload,
                reward=reward,
                expected_answer=expected_answer,
                judge_evaluations=[],
            )

        model_answer = await self._query_judge_distill_quorum(
            question=question,
            answer=model_raw_response,
            max_samples=self.config.quorum_max_samples,
        )
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: model answer = {repr(model_answer)}",
                flush=True,
            )
        if not model_answer:
            payload = body.model_dump()
            payload.pop("expected_answer", None)
            reward = 0.0
            return MultistepEquivLLMJudgeVerifyResponse(
                **payload,
                reward=reward,
                expected_answer=expected_answer,
                judge_evaluations=[],
            )
        model_distilled_answer = model_answer

        if self.config.expected_answer_distill:
            expected_distilled_answer = await self._query_judge_distill_quorum(
                question=question,
                answer=expected_answer,
                max_samples=self.config.quorum_max_samples,
            )
            if self.config.debug:
                print(
                    f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: expected distilled answer = {repr(expected_distilled_answer)}",
                    flush=True,
                )
            if not expected_distilled_answer:
                expected_distilled_answer = expected_answer
        else:
            expected_distilled_answer = expected_answer

        equivalent = await self._query_judge_compare_quorum(
            question=question,
            expected_answer=expected_distilled_answer,
            model_answer=model_distilled_answer,
            max_samples=self.config.quorum_max_samples,
            swap=self.config.swap,
        )
        if equivalent:
            reward = 1.0
        else:
            reward = 0.0
        if self._log_path is not None:
            # print(f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: log path = {repr(self._log_path)}", flush=True)
            async with self._log_write:
                try:
                    log_file = open(self._log_path, "a")
                except Exception as e:
                    log_file = None
                    print(
                        f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: log except: {type(e).__name__} {e}",
                        flush=True,
                    )
                if log_file is not None:
                    log_item = {
                        "id": body.id,
                        "uuid": body.uuid,
                        "question": question,
                        "expected_answer": expected_answer,
                        "expected_distilled_answer": expected_distilled_answer,
                        "model_answer": model_answer,
                        "model_distilled_answer": model_distilled_answer,
                        "equivalent": equivalent,
                        "reward": reward,
                        "quorum_max_samples": self.config.quorum_max_samples,
                        "quorum_type": self.config.quorum_type,
                        "swap": self.config.swap,
                        "default_judge_responses_create_params": self._default_judge_params_dict,
                        "model_responses_create_params": model_params_dict,
                        "model_response": model_response_dict,
                    }
                    if body.rl_metadata is not None:
                        log_item["rl_metadata"] = body.rl_metadata
                    print(json.dumps(log_item), file=log_file, flush=True)
                    log_file.close()

        payload = body.model_dump()
        payload.pop("expected_answer", None)
        return MultistepEquivLLMJudgeVerifyResponse(
            **payload,
            reward=reward,
            expected_answer=expected_distilled_answer,
            judge_evaluations=[],
        )

    async def _post_judge_response(self, params) -> NeMoGymResponse:
        try:
            async with self._judge_endpoint_max_concurrency:
                response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                )
            response = NeMoGymResponse.model_validate(await response.json())
        except Exception as e:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._post_judge_response: dummy response b/c of POST exception: {type(e).__name__} {e}",
                flush=True,
            )
            response = NeMoGymResponse.model_validate(
                {
                    "output": [
                        {
                            "role": "assistant",
                            "content": "",
                        }
                    ],
                }
            )
        return response

    async def _generate_judge_extract_response(
        self,
        *,
        question: str,
        raw_response: str,
    ):
        cfg = self.config

        if self._judge_extract_system_template is None:
            assert cfg.judge_extract_system_template_fpath is not None
            with open(cfg.judge_extract_system_template_fpath, "r") as file:
                self._judge_extract_system_template = file.read().rstrip()
        if self._judge_extract_prompt_template is None:
            assert cfg.judge_extract_prompt_template_fpath is not None
            with open(cfg.judge_extract_prompt_template_fpath, "r") as file:
                self._judge_extract_prompt_template = file.read().rstrip()

        extract_messages: list[NeMoGymEasyInputMessage] = []
        extract_messages.append(
            NeMoGymEasyInputMessage(
                role="system",
                content=self._judge_extract_system_template,
            )
        )
        extract_messages.append(
            NeMoGymEasyInputMessage(
                role="user",
                content=self._judge_extract_prompt_template.format(
                    question=question,
                    raw_response=raw_response,
                ),
            )
        )

        extract_params = cfg.judge_responses_create_params.model_copy(deep=True)
        extract_params.input = extract_messages
        extract_response = await self._post_judge_response(extract_params)
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_extract_response: {extract_response}",
                flush=True,
            )

        return extract_response

    async def _query_judge_extract_sample(
        self,
        *,
        question: str,
        raw_response: str,
    ) -> Optional[str]:
        extract_response = await self._generate_judge_extract_response(
            question=question,
            raw_response=raw_response,
        )
        extract_text = _get_response_last_assistant_content_text(extract_response) or ""
        answer = _extract_tagged_section(extract_text, "answer")
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._query_judge_extract_sample: answer = {repr(answer)}",
                flush=True,
            )
        return answer

    async def _query_judge_extract_quorum(
        self,
        *,
        question: str,
        raw_response: str,
        max_samples: int,
    ) -> Optional[str]:
        work = []
        for _ in range(max_samples):
            work.append(
                self._query_judge_distill_sample(
                    question=question,
                    raw_response=raw_response,
                )
            )
        results = await asyncio.gather(*work, return_exceptions=True)
        if len(results) == 1:
            if isinstance(results[0], str):
                return results[0]
            else:
                return None
        # TODO: quorum.
        raise NotImplementedError

    async def _generate_judge_distill_response(
        self,
        *,
        question: str,
        answer: str,
    ):
        cfg = self.config

        if self._judge_distill_system_template is None:
            assert cfg.judge_distill_system_template_fpath is not None
            with open(cfg.judge_distill_system_template_fpath, "r") as file:
                self._judge_distill_system_template = file.read().rstrip()
        if self._judge_distill_prompt_template is None:
            assert cfg.judge_distill_prompt_template_fpath is not None
            with open(cfg.judge_distill_prompt_template_fpath, "r") as file:
                self._judge_distill_prompt_template = file.read().rstrip()

        distill_messages: list[NeMoGymEasyInputMessage] = []
        distill_messages.append(
            NeMoGymEasyInputMessage(
                role="system",
                content=self._judge_distill_system_template,
            )
        )
        distill_messages.append(
            NeMoGymEasyInputMessage(
                role="user",
                content=self._judge_distill_prompt_template.format(
                    question=question,
                    answer=answer,
                ),
            )
        )

        distill_params = cfg.judge_responses_create_params.model_copy(deep=True)
        distill_params.input = distill_messages
        distill_response = await self._post_judge_response(distill_params)
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_distill_response: {distill_response}",
                flush=True,
            )

        return distill_response

    async def _query_judge_distill_sample(
        self,
        *,
        question: str,
        answer: str,
    ) -> Optional[str]:
        distill_response = await self._generate_judge_distill_response(
            question=question,
            answer=answer,
        )
        distill_text = _get_response_last_assistant_content_text(distill_response) or ""
        distilled_answer = _extract_tagged_section(distill_text, "distilled_answer")
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._query_judge_distill_sample: model distilled answer = {repr(distilled_answer)}",
                flush=True,
            )
        return distilled_answer

    async def _query_judge_distill_quorum(
        self,
        *,
        question: str,
        answer: str,
        max_samples: int,
    ) -> Optional[str]:
        assert question is not None
        work = []
        for _ in range(max_samples):
            work.append(
                self._query_judge_distill_sample(
                    question=question,
                    answer=answer,
                )
            )
        results = await asyncio.gather(*work, return_exceptions=True)
        if len(results) == 1:
            if isinstance(results[0], str):
                return results[0]
            else:
                return None

        if self._judge_distill_quorum_template is None:
            assert self.config.judge_distill_quorum_template_fpath is not None
            with open(self.config.judge_distill_quorum_template_fpath, "r") as file:
                self._judge_distill_quorum_template = file.read().rstrip()

        quorum_messages: list[NeMoGymEasyInputMessage] = []
        quorum_messages.append(
            NeMoGymEasyInputMessage(
                role="system",
                content=self._judge_distill_quorum_template,
            )
        )
        prompt_parts = []
        prompt_parts.append("<question>")
        prompt_parts.append(question)
        prompt_parts.append("</question>")
        for i, r in enumerate(results):
            if r is None:
                answer_i = ""
            else:
                assert isinstance(r, str)
                answer_i = r
            rank = i + 1
            prompt_parts.append("")
            prompt_parts.append(f"<answer_{rank}>")
            if answer_i:
                prompt_parts.append(answer_i)
            prompt_parts.append(f"</answer_{rank}>")
        prompt = "\n".join(prompt_parts)
        quorum_messages.append(
            NeMoGymEasyInputMessage(
                role="user",
                content=prompt,
            )
        )

        quorum_params = self.config.judge_responses_create_params.model_copy(deep=True)
        quorum_params.input = quorum_messages
        quorum_response = await self._post_judge_response(quorum_params)
        quorum_text = _get_response_last_assistant_content_text(quorum_response) or ""
        final_answer = _extract_tagged_section(quorum_text, "final_answer")

        return final_answer

    async def _generate_judge_compare_response(
        self,
        *,
        question: str,
        answer_0: str,
        answer_1: str,
    ):
        cfg = self.config

        if self._judge_compare_system_template is None:
            assert cfg.judge_compare_system_template_fpath is not None
            with open(cfg.judge_compare_system_template_fpath, "r") as file:
                self._judge_compare_system_template = file.read().rstrip()
        if self._judge_compare_prompt_template is None:
            assert cfg.judge_compare_prompt_template_fpath is not None
            with open(cfg.judge_compare_prompt_template_fpath, "r") as file:
                self._judge_compare_prompt_template = file.read().rstrip()

        compare_messages: list[NeMoGymEasyInputMessage] = []
        compare_messages.append(
            NeMoGymEasyInputMessage(
                role="system",
                content=self._judge_compare_system_template,
            )
        )
        compare_messages.append(
            NeMoGymEasyInputMessage(
                role="user",
                content=self._judge_compare_prompt_template.format(
                    question=question,
                    answer_0=answer_0,
                    answer_1=answer_1,
                ),
            )
        )

        compare_params = cfg.judge_responses_create_params.model_copy(deep=True)
        compare_params.input = compare_messages
        compare_response = await self._post_judge_response(compare_params)
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_compare_response: {compare_response}",
                flush=True,
            )

        return compare_response

    async def _query_judge_compare_sample(
        self,
        *,
        question: str,
        answer_0: str,
        answer_1: str,
    ) -> Optional[bool]:
        compare_response = await self._generate_judge_compare_response(
            question=question,
            answer_0=answer_0,
            answer_1=answer_1,
        )
        compare_text = _get_response_last_assistant_content_text(compare_response) or ""
        compare_answer = _extract_tagged_section(compare_text, "equivalent", strip=True)
        if isinstance(compare_answer, str):
            compare_answer_lo = compare_answer.lower()
            if compare_answer_lo == "true":
                return True
            elif compare_answer_lo == "false":
                return False
            else:
                return None
        else:
            return None

    async def _query_judge_compare_quorum(
        self,
        *,
        question: str,
        expected_answer: str,
        model_answer: str,
        max_samples: int,
        swap: bool = True,
    ) -> Optional[bool]:
        if swap:
            max_samples *= 2
        work = []
        for _ in range(max_samples):
            work.append(
                self._query_judge_compare_sample(
                    question=question,
                    answer_0=expected_answer,
                    answer_1=model_answer,
                )
            )
            if swap:
                work.append(
                    self._query_judge_compare_sample(
                        question=question,
                        answer_0=model_answer,
                        answer_1=expected_answer,
                    )
                )
        results = await asyncio.gather(*work, return_exceptions=True)
        quorum = _CompareQuorum()
        for r in results:
            if r is True:
                quorum.pos_count += 1
            elif r is False:
                quorum.neg_count += 1
            else:
                quorum.nil_count += 1
        return quorum.majority(max_samples)


if __name__ == "__main__":
    MultistepEquivLLMJudgeResourcesServer.run_webserver()
