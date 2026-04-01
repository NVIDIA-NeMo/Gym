# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-turn TIR agent with tool-use loop and summary model support.
#
# Combines simple_agent's inner tool loop with multiturn_proof_agent's
# outer multi-turn management:
#
#   summary_model mode (max_turns=2):
#     Turn 0: tool-use reasoning → verify → get summary_prompt
#     Turn 1: summary generation (enable_thinking=False) → verify → get retry prompt
#     Turn 2: tool-use reasoning (fresh sandbox) → verify → judge → done
#
# Each reasoning turn uses a fresh sandbox session (seed_session on ns_tools).
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status

LOG = logging.getLogger(__name__)


class MathTIRMultiturnAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    tool_server_name: str = "ns_tools"
    max_turns: int = 2
    max_steps: int = 100
    response_processor: str = "summary_model"
    include_all_attempts: bool = True
    max_output_tokens: Optional[int] = None
    summary_max_output_tokens: int = 4096


class MathTIRMultiturnRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MathTIRMultiturnVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MathTIRMultiturnVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    total_turns: int = 0
    all_attempts: Optional[List[Dict[str, Any]]] = None


class MathTIRMultiturnAgent(SimpleResponsesAPIAgent):
    config: MathTIRMultiturnAgentConfig

    # ------------------------------------------------------------------
    #  Inner tool-use loop (adapted from simple_agent)
    # ------------------------------------------------------------------
    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        step = 0
        model_server_cookies = None
        resources_server_cookies = request.cookies

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            new_outputs.extend(output)

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                api_response = await self.server_client.post(
                    server_name=self.config.tool_server_name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

            if self.config.max_steps and step >= self.config.max_steps:
                break

        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        return model_response

    # ------------------------------------------------------------------
    #  Outer multi-turn loop (adapted from multiturn_proof_agent)
    # ------------------------------------------------------------------
    async def run(self, request: Request, body: MathTIRMultiturnRunRequest) -> MathTIRMultiturnVerifyResponse:
        cookies = request.cookies
        all_attempts: List[Dict[str, Any]] = []

        current_input = body.responses_create_params
        if self.config.max_output_tokens is not None:
            if isinstance(current_input, dict):
                current_input["max_output_tokens"] = self.config.max_output_tokens
            else:
                current_input = current_input.model_copy(update={"max_output_tokens": self.config.max_output_tokens})

        turn_index = 0
        use_summary = self.config.response_processor == "summary_model"
        effective_max = self.config.max_turns * 2 - 1 if use_summary else self.config.max_turns
        next_is_summary_prompt = False

        while turn_index < effective_max:
            is_reasoning_turn = not next_is_summary_prompt
            LOG.info("Turn %d: reasoning=%s summary_prompt=%s", turn_index, is_reasoning_turn, next_is_summary_prompt)

            if is_reasoning_turn:
                seed_response = await self.server_client.post(
                    server_name=self.config.tool_server_name,
                    url_path="/seed_session",
                    json=body.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(seed_response)
                cookies = seed_response.cookies

                gen_response = await self.server_client.post(
                    server_name=self.config.name,
                    url_path="/v1/responses",
                    json=current_input,
                    cookies=cookies,
                )
            else:
                summary_input = current_input
                if self.config.summary_max_output_tokens:
                    if isinstance(summary_input, dict):
                        summary_input = {**summary_input, "max_output_tokens": self.config.summary_max_output_tokens}
                    else:
                        summary_input = summary_input.model_copy(
                            update={"max_output_tokens": self.config.summary_max_output_tokens}
                        )
                gen_response = await self.server_client.post(
                    server_name="policy_model_reasoning_off",
                    url_path="/v1/responses",
                    json=summary_input,
                    cookies=cookies,
                )

            await raise_for_status(gen_response)
            cookies = gen_response.cookies
            model_response_json = await gen_response.json()

            is_summary_turn = use_summary and (turn_index % 2 == 1)
            was_truncated = self._check_truncated(model_response_json)

            verify_request_data = body.model_dump()
            verify_request_data["response"] = model_response_json
            verify_request_data["turn_index"] = turn_index
            verify_request_data["was_truncated"] = was_truncated
            verify_request_data["is_summary_turn"] = is_summary_turn

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request_data,
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            cookies = verify_response.cookies
            verify_result = await verify_response.json()

            generation_text = self._extract_generation_text(model_response_json)

            if hasattr(current_input, "model_dump"):
                input_dict = current_input.model_dump()
            else:
                input_dict = current_input

            turn_info = verify_result.get("turn_info", {})
            turn_info["_was_truncated"] = was_truncated
            attempt_record = {
                "turn_index": turn_index,
                "turn_type": "summary" if is_summary_turn else "reasoning",
                "input": input_dict,
                "response": model_response_json,
                "generation": generation_text,
                "reward": verify_result.get("reward", 0.0),
                "is_summary_turn": is_summary_turn,
                "turn_info": turn_info,
            }
            all_attempts.append(attempt_record)

            needs_correction = verify_result.get("needs_correction", False)
            if not needs_correction:
                LOG.info("Turn %d: Done (reward=%s)", turn_index, verify_result.get("reward"))
                break

            correction_prompt = verify_result.get("correction_prompt")
            if not correction_prompt:
                LOG.warning("Turn %d: needs_correction=True but no correction_prompt", turn_index)
                break

            next_is_summary_prompt = verify_result.get("is_summary_prompt", False)

            params = body.responses_create_params
            current_input = {
                "input": [{"role": "user", "content": correction_prompt}],
                "model": getattr(params, "model", None) if hasattr(params, "model") else params.get("model") if isinstance(params, dict) else None,
            }
            keys_to_copy = ["temperature", "top_p"]
            if not next_is_summary_prompt:
                keys_to_copy += ["tools", "tool_choice"]
            for key in keys_to_copy:
                value = getattr(params, key, None) if hasattr(params, key) else params.get(key) if isinstance(params, dict) else None
                if value is not None:
                    current_input[key] = value
            if not next_is_summary_prompt and self.config.max_output_tokens is not None:
                current_input["max_output_tokens"] = self.config.max_output_tokens

            turn_index += 1

        final_response = MathTIRMultiturnVerifyResponse.model_validate(verify_result)
        final_response.total_turns = turn_index + 1
        if self.config.include_all_attempts:
            final_response.all_attempts = all_attempts

        return final_response

    @staticmethod
    def _check_truncated(model_response_json: dict) -> bool:
        inc = model_response_json.get("incomplete_details")
        if inc and inc.get("reason") in ("max_output_tokens",):
            return True
        stop_reason = model_response_json.get("stop_reason")
        if stop_reason in ("length", "max_tokens"):
            return True
        for out in model_response_json.get("output", []):
            if out.get("type") == "message":
                sr = out.get("stop_reason")
                if sr in ("length", "max_tokens"):
                    return True
        return False

    @staticmethod
    def _extract_generation_text(model_response_json: dict) -> str:
        reasoning_parts = []
        content_parts = []
        for output in model_response_json.get("output", []):
            if output.get("type") == "reasoning":
                for s in output.get("summary", []):
                    s_text = s.get("text", "")
                    if s_text:
                        reasoning_parts.append(s_text)
            elif output.get("type") == "message" and output.get("content"):
                for content in output["content"]:
                    if content.get("type") == "output_text":
                        content_parts.append(content.get("text", ""))
        result = ""
        if reasoning_parts:
            result = "<think>" + "\n".join(reasoning_parts) + "</think>"
        result += "".join(content_parts)
        return result


if __name__ == "__main__":
    MathTIRMultiturnAgent.run_webserver()
