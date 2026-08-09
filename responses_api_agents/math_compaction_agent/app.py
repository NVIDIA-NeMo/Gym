# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import uuid
from typing import Optional

from fastapi import Request
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentVerifyRequest,
)
from responses_api_models.vllm_model.app import VLLMConverter


class MathCompactionAgentConfig(SimpleAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 8
    min_steps_before_answer: int = 1
    context_budget: int = 73728
    compaction_threshold_tokens: int = 10240
    max_compactions: int = 3
    recent_steps: int = 2
    summary_max_output_tokens: int = 10240
    summary_instruction: str = (
        "Summarize the work so far so another agent can continue solving the problem. "
        "Preserve the original goal, important equations, Python results, failed attempts, "
        "unresolved issues, and the most promising next steps. Do not solve the problem "
        "from scratch and do not call tools."
    )
    resume_template: str = (
        "Continue solving the original problem from the compacted state below.\n\n"
        "<compacted_state>\n{summary}\n</compacted_state>"
    )


class MathCompactionRunRequest(SimpleAgentRunRequest):
    model_config = ConfigDict(extra="allow")


class MathCompactionVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class MathCompactionAgent(SimpleAgent):
    config: MathCompactionAgentConfig
    _policy_model_openai_client: Optional[NeMoGymAsyncOpenAI] = None

    def setup_webserver(self):
        app = super().setup_webserver()
        global_config = get_global_config_dict()
        model_config = get_first_server_config_dict(
            global_config, self.config.model_server.name
        )
        base_urls = model_config["base_url"]
        base_url = base_urls if isinstance(base_urls, str) else base_urls[0]
        self._policy_model_openai_client = NeMoGymAsyncOpenAI(
            base_url=base_url, api_key=model_config["api_key"]
        )
        return app

    async def run(
        self, request: Request, body: MathCompactionRunRequest
    ) -> list[MathCompactionVerifyResponse]:
        cookies = request.cookies
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        resources_cookies = seed_response.cookies
        model_cookies = None

        params = body.responses_create_params.model_copy(deep=True)
        if isinstance(params.input, str):
            params.input = [NeMoGymEasyInputMessage(role="user", content=params.input)]

        system_input = [message for message in params.input if message.role == "system"]
        segment_input = list(params.input)
        segment_outputs = []
        segments: list[dict] = []
        model_response: NeMoGymResponse | None = None
        execution_steps = 0
        compaction_count = 0
        empty_summary_count = 0

        while execution_steps < self.config.max_steps:
            request_params = params.model_copy(
                update={"input": segment_input + segment_outputs}
            )
            if segment_outputs and self._ends_with_atomic_step(segment_outputs):
                prompt_tokens = await self._count_prompt_tokens(request_params)
                should_compact = (
                    self.config.context_budget - prompt_tokens
                    < self.config.compaction_threshold_tokens
                )
                if should_compact and compaction_count < self.config.max_compactions:
                    assert model_response is not None
                    segments.append(
                        self._make_segment(
                            request_params=params.model_copy(update={"input": segment_input}),
                            response=model_response.model_copy(
                                deep=True, update={"output": copy.deepcopy(segment_outputs)}
                            ),
                            segment_type="execution",
                        )
                    )

                    summary_params = self._build_summary_params(
                        params, segment_input + segment_outputs
                    )
                    summary_response, model_cookies = await self._generate(
                        summary_params, body, model_cookies
                    )
                    summary = self._extract_response_text(summary_response)
                    if not summary:
                        empty_summary_count += 1
                        print(
                            "Compaction policy returned no summary text; "
                            "continuing from the retained recent steps",
                            flush=True,
                        )
                    if self._has_trainable_generation(summary_response):
                        segments.append(
                            self._make_segment(
                                request_params=summary_params,
                                response=summary_response,
                                segment_type="summary",
                            )
                        )

                    recent_steps = self._extract_recent_steps(
                        segment_outputs, self.config.recent_steps
                    )
                    resume = NeMoGymEasyInputMessage(
                        role="user",
                        content=self.config.resume_template.format(summary=summary),
                    )
                    segment_input = await self._fit_reconstructed_context(
                        params=params,
                        fixed_input=[*system_input, resume],
                        recent_steps=recent_steps,
                    )
                    segment_outputs = []
                    compaction_count += 1
                    continue

            model_response, model_cookies = await self._generate(
                request_params, body, model_cookies
            )
            execution_steps += 1
            output = model_response.output
            segment_outputs.extend(output)

            if (
                model_response.incomplete_details
                and execution_steps >= self.config.min_steps_before_answer
            ):
                break

            function_calls: list[NeMoGymResponseFunctionToolCall] = [
                item for item in output if item.type == "function_call"
            ]
            output_messages: list[NeMoGymResponseOutputMessage] = [
                item
                for item in output
                if item.type == "message" and item.role == "assistant"
            ]
            if (
                not function_calls
                and output_messages
                and execution_steps >= self.config.min_steps_before_answer
            ):
                break

            for function_call in function_calls:
                try:
                    arguments = json.loads(function_call.arguments)
                except (json.JSONDecodeError, TypeError) as error:
                    segment_outputs.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=function_call.call_id,
                            output=json.dumps(
                                {"error": f"Invalid tool call arguments: {error!r}"}
                            ),
                        )
                    )
                    continue

                tool_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{function_call.name}",
                    json=arguments,
                    cookies=resources_cookies,
                )
                await raise_for_status(tool_response)
                resources_cookies = tool_response.cookies
                segment_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=function_call.call_id,
                        output=(await tool_response.content.read()).decode(),
                    )
                )

        if model_response is None or not segment_outputs:
            raise RuntimeError("Compaction rollout produced no execution output")

        final_response = model_response.model_copy(
            deep=True, update={"output": copy.deepcopy(segment_outputs)}
        )
        segments.append(
            self._make_segment(
                request_params=params.model_copy(update={"input": segment_input}),
                response=final_response,
                segment_type="execution",
            )
        )

        verify_request = SimpleAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": final_response.model_dump()}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=resources_cookies,
        )
        await raise_for_status(verify_response)
        reward = float((await get_response_json(verify_response))["reward"])

        trajectory_id = uuid.uuid4().hex
        results = []
        for segment_index, segment in enumerate(segments):
            entry = body.model_dump()
            entry.update(segment)
            entry.update(
                {
                    "reward": reward,
                    "trajectory_id": trajectory_id,
                    "segment_index": segment_index,
                    "is_final_segment": segment_index == len(segments) - 1,
                    "compaction_count": compaction_count,
                    "empty_summary_count": empty_summary_count,
                    "loss_multiplier": 1.0,
                }
            )
            results.append(MathCompactionVerifyResponse.model_validate(entry))
        return results

    async def _generate(self, params, body, cookies):
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=params.model_dump(exclude_none=True),
            cookies=cookies,
        )
        await raise_for_status(response)
        response_json = await get_response_json(response)
        try:
            parsed = NeMoGymResponse.model_validate(response_json)
        except ValidationError as error:
            raise RuntimeError(
                f"Received an invalid response from model server: {response_json!r}"
            ) from error
        return parsed, response.cookies

    def _build_summary_params(self, params, history):
        summary_prompt = NeMoGymEasyInputMessage(
            role="user", content=self.config.summary_instruction
        )
        return params.model_copy(
            deep=True,
            update={
                "input": [*history, summary_prompt],
                "tools": [],
                "tool_choice": None,
                "max_output_tokens": self.config.summary_max_output_tokens,
            },
        )

    @staticmethod
    def _make_segment(request_params, response, segment_type):
        return {
            "responses_create_params": request_params.model_dump(exclude_none=True),
            "response": response.model_dump(),
            "segment_type": segment_type,
        }

    @staticmethod
    def _ends_with_atomic_step(outputs) -> bool:
        return bool(outputs) and outputs[-1].type != "function_call"

    @staticmethod
    def _extract_recent_steps(outputs, count: int):
        if count <= 0:
            return []
        completed_steps = []
        start = 0
        for index, item in enumerate(outputs):
            if item.type == "function_call_output":
                completed_steps.append(outputs[start : index + 1])
                start = index + 1
        return completed_steps[-count:]

    async def _fit_reconstructed_context(self, params, fixed_input, recent_steps):
        retained_steps = list(recent_steps)
        while True:
            reconstructed_input = [
                *fixed_input,
                *(item for step in retained_steps for item in step),
            ]
            token_count = await self._count_prompt_tokens(
                params.model_copy(deep=True, update={"input": reconstructed_input})
            )
            if token_count <= self.config.context_budget:
                return reconstructed_input
            if not retained_steps:
                raise RuntimeError(
                    "Compacted summary does not fit within the configured context budget"
                )
            retained_steps.pop(0)

    @staticmethod
    def _extract_response_text(response: NeMoGymResponse) -> str:
        message_chunks = []
        reasoning_chunks = []
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    text = getattr(content, "text", None)
                    if text:
                        message_chunks.append(text)
            elif item.type == "reasoning":
                for summary in item.summary:
                    text = getattr(summary, "text", None)
                    if text:
                        reasoning_chunks.append(text)
        chunks = message_chunks or reasoning_chunks
        return "\n".join(chunks).strip()

    @staticmethod
    def _has_trainable_generation(response: NeMoGymResponse) -> bool:
        return any(
            bool(getattr(item, "generation_token_ids", None))
            for item in response.output
        )

    async def _count_prompt_tokens(self, params) -> int:
        if self._policy_model_openai_client is None:
            raise RuntimeError("Policy tokenizer client is not initialized")
        converter = VLLMConverter(return_token_id_information=False)
        chat_params = converter.responses_to_chat_completion_create_params(params)
        body = chat_params.model_dump(exclude_none=True)
        tokenize_body = {
            key: body[key]
            for key in ("model", "messages", "tools", "chat_template_kwargs")
            if key in body
        }
        tokenized = await self._policy_model_openai_client.create_tokenize(
            **tokenize_body
        )
        return len(tokenized["tokens"])


if __name__ == "__main__":
    MathCompactionAgent.run_webserver()
