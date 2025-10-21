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

    judge_endpoint_max_num_retries: int = 2
    judge_endpoint_max_concurrency: Optional[int] = 128

    judge_system_message: Optional[str] = None
    judge_prompt_template: Optional[str] = None

    judge_extract_system_template_fpath: Optional[str] = None
    judge_extract_prompt_template_fpath: Optional[str] = None
    judge_distill_system_template_fpath: Optional[str] = None
    judge_distill_prompt_template_fpath: Optional[str] = None
    judge_verdict_prompt_template_fpath: Optional[str] = None

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

    # If true, perform a second judge pass swapping expected and generated answers
    # to reduce potential positional bias. Default is True.
    # swap: bool = True

    # trials: int = 1

    expected_answer_distill: bool = False
    model_response_parse_reasoning: bool = False

    debug: bool = False


class MultistepEquivLLMJudgeRunRequest(BaseRunRequest):
    """Run/verify request payload.

    Compatible with MCQA-like datasets. Only `expected_answer` is required for
    grading, but `options` and `metadata` are accepted for compatibility.
    """

    uuid: Optional[str] = None
    expected_answer: Optional[str] = None
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


def _get_response_first_user_content_text(response, parse_reasoning: bool = False) -> Optional[str]:
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


def _extract_tagged_section(haystack: str, tag: str) -> Optional[str]:
    needle = haystack
    _, _, needle = needle.partition(f"<{tag}>")
    needle, _, _ = needle.partition(f"</{tag}>")
    return needle.strip()


def _extract_answer_tagged_section(haystack: str) -> Optional[str]:
    return _extract_tagged_section(haystack, "answer")


def _extract_distilled_answer_tagged_section(haystack: str) -> Optional[str]:
    return _extract_tagged_section(haystack, "distilled_answer")


class MultistepEquivLLMJudgeResourcesServer(SimpleResourcesServer):
    """Judge-only verifier using an LLM to compare answers."""

    config: MultistepEquivLLMJudgeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.debug:
            print(f"DEBUG: MultistepEquivLLMJudgeResourcesServer: config = {self.config}", flush=True)
        max_concurrency = self.config.judge_endpoint_max_concurrency
        self._log_write = asyncio.Lock()
        if max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = contextlib.nullcontext()
        self._judge_extract_system_template = None
        self._judge_extract_prompt_template = None
        self._judge_distill_system_template = None
        self._judge_distill_prompt_template = None
        self._judge_verdict_prompt_template = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: MultistepEquivLLMJudgeVerifyRequest) -> MultistepEquivLLMJudgeVerifyResponse:
        return await self._verify_single_trial(body)

    async def _verify_single_trial(
        self, body: MultistepEquivLLMJudgeVerifyRequest
    ) -> MultistepEquivLLMJudgeVerifyResponse:
        model_responses_create_params_dict = body.responses_create_params.model_dump()
        model_response_dict = body.response.model_dump()

        question = _get_response_first_user_content_text(body.response)
        model_raw_response = _get_response_last_assistant_raw_response_text(
            body.response, parse_reasoning=self.config.model_response_parse_reasoning
        )
        expected_answer = _get_request_expected_answer_text(body) or ""

        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: question        = {repr(question)}",
                flush=True,
            )
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

        model_extract_response = await self._generate_judge_extract_response(
            question=question,
            raw_response=model_raw_response,
        )
        model_extract_text = _get_response_last_assistant_content_text(model_extract_response) or ""
        model_answer = _extract_answer_tagged_section(model_extract_text)
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: model answer    = {repr(model_answer)}",
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

        model_distill_response = await self._generate_judge_distill_response(
            question=question,
            answer=model_answer,
        )
        model_distill_text = _get_response_last_assistant_content_text(model_distill_response) or ""
        model_distilled_answer = _extract_distilled_answer_tagged_section(model_distill_text)
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: model distilled answer = {repr(model_distilled_answer)}",
                flush=True,
            )

        if not model_distilled_answer:
            model_distilled_answer = model_answer

        if self.config.expected_answer_distill:
            expected_distill_response = await self._generate_judge_distill_response(
                question=question,
                answer=expected_answer,
            )
            expected_distill_text = _get_response_last_assistant_content_text(expected_distill_response) or ""
            expected_distilled_answer = _extract_distilled_answer_tagged_section(expected_distill_text)
            if self.config.debug:
                print(
                    f"DEBUG: MultistepEquivLLMJudgeResourcesServer.verify: expected distilled answer = {repr(expected_distilled_answer)}",
                    flush=True,
                )

            if not expected_distilled_answer:
                expected_distilled_answer = expected_answer

        else:
            expected_distilled_answer = expected_answer

        # Run judge twice to mitigate positional or presentation bias by swapping orders.
        first_equal, first_eval = await self._generate_judge_evaluation(
            question=question,
            expected_answer=expected_distilled_answer,
            generated_answer=model_distilled_answer,
            model_responses_create_params_dict=model_responses_create_params_dict,
            model_response_dict=model_response_dict,
            rl_metadata=body.rl_metadata,
        )
        if False:
            # if not first_equal:
            reward = 0.0
            payload = body.model_dump()
            # Avoid duplicate field when constructing response
            payload.pop("expected_answer", None)
            return MultistepEquivLLMJudgeVerifyResponse(
                **payload, reward=reward, expected_answer=expected_answer, judge_evaluations=[first_eval]
            )

        # If first pass says equal, optionally confirm with a second pass (swap answers).
        if False:
            # if not self.config.check_twice_swap:
            payload = body.model_dump()
            payload.pop("expected_answer", None)
            return MultistepEquivLLMJudgeVerifyResponse(
                **payload, reward=1.0, expected_answer=expected_answer, judge_evaluations=[first_eval]
            )

        second_equal, second_eval = await self._generate_judge_evaluation(
            question=question,
            expected_answer=model_distilled_answer,
            generated_answer=expected_distilled_answer,
            model_responses_create_params_dict=model_responses_create_params_dict,
            model_response_dict=model_response_dict,
            rl_metadata=body.rl_metadata,
        )
        # If they are both equal, we give a reward of 1.0; otherwise use configured fallback.
        # User has to expect this on the training side to discard the data points if negative.
        # reward = 1.0 if second_equal else self.config.reward_if_swap_fails
        if first_equal and second_equal:
            reward = 1.0
        else:
            reward = 0.0
        payload = body.model_dump()
        payload.pop("expected_answer", None)
        return MultistepEquivLLMJudgeVerifyResponse(
            **payload,
            reward=reward,
            expected_answer=expected_answer,
            judge_evaluations=[first_eval, second_eval],
        )

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
        try:
            async with self._judge_endpoint_max_concurrency:
                extract_response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=extract_params,
                )
        except Exception as e:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_extract_response: dummy response b/c of POST exception: {type(e).__name__} {e}",
                flush=True,
            )
            extract_response = NeMoGymResponse.model_validate(
                {
                    "output": [
                        {
                            # "type": "message",
                            "role": "assistant",
                            # "content": f"<answer>\n{raw_response}\n></answer>",
                            "content": "",
                        }
                    ],
                }
            )
        extract_response = NeMoGymResponse.model_validate(await extract_response.json())
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_extract_response: {extract_response}",
                flush=True,
            )

        return extract_response

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
        try:
            async with self._judge_endpoint_max_concurrency:
                distill_response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=distill_params,
                )
            distill_response = NeMoGymResponse.model_validate(await distill_response.json())
        except Exception as e:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_distill_response: dummy response b/c of POST exception: {type(e).__name__} {e}",
                flush=True,
            )
            distill_response = NeMoGymResponse.model_validate(
                {
                    "output": [
                        {
                            # "type": "message",
                            "role": "assistant",
                            # "content": f"<distilled_answer>\n{answer}\n></distilled_answer>",
                            "content": "",
                        }
                    ],
                }
            )
        if self.config.debug:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_distill_response: {distill_response}",
                flush=True,
            )

        return distill_response

    async def _generate_judge_evaluation(
        self,
        *,
        question: str,
        expected_answer: str,
        generated_answer: str,
        model_responses_create_params_dict: Optional[dict] = None,
        model_response_dict: Optional[dict] = None,
        rl_metadata: Optional[dict] = None,
    ) -> tuple[bool, JudgeEvaluation]:
        global_cfg = get_global_config_dict()
        cfg = self.config

        base_log_dir = cfg.get("base_log_dir", None)
        if base_log_dir is None:
            base_log_dir = global_cfg.get("_x_nemo_rl_base_log_dir", None)

        if base_log_dir is not None:
            # judge_log_path = os.path.join(base_log_dir, f"{cfg.name}-{uuid.uuid4()}-judge_responses_data.jsonl")
            judge_log_path = os.path.join(base_log_dir, f"{cfg.name}-judge_responses_data.jsonl")
        else:
            judge_log_path = None

        equal_label = cfg.judge_equal_label
        not_equal_label = cfg.judge_not_equal_label

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)

        # prompt_template = cfg.judge_prompt_template
        system_message = cfg.judge_system_message

        if self._judge_verdict_prompt_template is None:
            assert cfg.judge_verdict_prompt_template_fpath is not None
            with open(cfg.judge_verdict_prompt_template_fpath, "r") as file:
                self._judge_verdict_prompt_template = file.read().rstrip()
        prompt_template = self._judge_verdict_prompt_template

        user_prompt = prompt_template.format(
            question=question, expected_answer=expected_answer, generated_answer=generated_answer
        )

        msgs: list[NeMoGymEasyInputMessage] = []
        if system_message is not None and system_message != "":
            msgs.append(NeMoGymEasyInputMessage(role="system", content=system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        try:
            async with self._judge_endpoint_max_concurrency:
                response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
            judge_response = NeMoGymResponse.model_validate(await response.json())
        except Exception as e:
            print(
                f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_evaluation: dummy response b/c of POST exception: {type(e).__name__} {e}",
                flush=True,
            )
            judge_response = NeMoGymResponse.model_validate(
                {
                    "output": [
                        {
                            # "type": "message",
                            "role": "assistant",
                            "content": "",
                        }
                    ],
                }
            )
        eval_record = JudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            verdict_label=None,
        )

        # Parse the last output; fall back to not-equal if unexpected.
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                return False, eval_record
            last_content = last_output.content[-1]
            text = getattr(last_content, "text", "")
        except Exception:
            return False, eval_record

        eq_pos = text.find(equal_label)
        neq_pos = text.find(not_equal_label)

        def _compute_verdict(eq_pos, neq_pos):
            if eq_pos < 0 and neq_pos < 0:
                return False, None
            if eq_pos >= 0 and (neq_pos < 0 or eq_pos < neq_pos):
                return True, equal_label
            return False, not_equal_label

        verdict, verdict_label = _compute_verdict(eq_pos, neq_pos)
        eval_record.verdict_label = verdict_label

        if judge_log_path is not None:
            # print(f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_evaluation: log path = {repr(judge_log_path)}", flush=True)
            async with self._log_write:
                try:
                    log_file = open(judge_log_path, "a")
                except Exception as e:
                    log_file = None
                    print(
                        f"DEBUG: MultistepEquivLLMJudgeResourcesServer._generate_judge_evaluation: log except: {type(e).__name__} {e}",
                        flush=True,
                    )
                if log_file is not None:
                    responses_create_params_dict = responses_create_params.model_dump()
                    judge_response_dict = judge_response.model_dump()
                    log_item = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": generated_answer,
                        "judge_responses_create_params": responses_create_params_dict,
                        "judge_response": judge_response_dict,
                        "judge_verdict": verdict,
                        "judge_verdict_label": verdict_label,
                    }
                    if model_responses_create_params_dict is not None:
                        log_item["model_responses_create_params"] = model_responses_create_params_dict
                    if model_response_dict is not None:
                        log_item["model_response"] = model_response_dict
                    if rl_metadata is not None:
                        log_item["rl_metadata"] = rl_metadata
                    print(json.dumps(log_item), file=log_file, flush=True)
                    log_file.close()

        return verdict, eval_record


if __name__ == "__main__":
    MultistepEquivLLMJudgeResourcesServer.run_webserver()
