# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-turn proof agent with summary model support.
#
# Extends proof_refinement_agent for the multi-turn proof flow:
#   strip_thinking mode:
#     Turn 0: generate proof → verify → reprompt
#     Turn 1: generate refined proof → verify → judge → done
#
#   summary_model mode:
#     Turn 0: generate proof (reasoning) → verify → get summary_prompt
#     Turn 1: generate summary (enable_thinking=False) → verify → get reprompt
#     Turn 2: generate refined proof (reasoning) → verify → judge → done
#
# The agent drives the loop; the resources_server (multiturn_proof_judge)
# provides verification, reprompt building, and judge scoring.

import logging
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict

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
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


class MultiturnProofAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_turns: int = 2
    response_processor: str = "strip_thinking"
    include_all_attempts: bool = True
    max_output_tokens: Optional[int] = None
    summary_max_output_tokens: Optional[int] = None


class MultiturnProofRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MultiturnProofVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MultiturnProofVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    total_turns: int = 0
    all_attempts: Optional[List[Dict[str, Any]]] = None


class MultiturnProofAgent(SimpleResponsesAPIAgent):
    config: MultiturnProofAgentConfig

    def _turn_max_output_tokens(self, is_summary_prompt: bool) -> Optional[int]:
        if is_summary_prompt:
            return self.config.summary_max_output_tokens
        return self.config.max_output_tokens

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        model_response_json = await model_response.json()

        for k, v in model_response.cookies.items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

    async def run(self, request: Request, body: MultiturnProofRunRequest) -> MultiturnProofVerifyResponse:
        cookies = request.cookies
        all_attempts: List[Dict[str, Any]] = []

        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies
        seed_result = await seed_response.json()

        current_input = body.responses_create_params
        initial_prompt = seed_result.get("initial_prompt")
        if initial_prompt:
            params = body.responses_create_params
            current_input = {
                "input": [{"role": "user", "content": initial_prompt}],
                "model": getattr(params, "model", None) if hasattr(params, "model") else params.get("model"),
            }
            for key in ["temperature", "top_p"]:
                value = getattr(params, key, None) if hasattr(params, key) else params.get(key)
                if value is not None:
                    current_input[key] = value
        initial_max_output_tokens = self._turn_max_output_tokens(is_summary_prompt=False)
        if initial_max_output_tokens is not None:
            current_input["max_output_tokens"] = initial_max_output_tokens
        turn_index = 0
        existing_summary = "None"
        use_summary = self.config.response_processor == "summary_model"

        # effective max turns: strip_thinking → max_turns, summary_model → max_turns * 2 - 1
        effective_max = self.config.max_turns * 2 - 1 if use_summary else self.config.max_turns

        next_is_summary_prompt = False
        reasoning_was_truncated = False

        while turn_index < effective_max:
            LOG.info("Turn %d: Generating (summary_prompt=%s)", turn_index, next_is_summary_prompt)

            if next_is_summary_prompt:
                gen_response = await self.server_client.post(
                    server_name="policy_model_reasoning_off",
                    url_path="/v1/responses",
                    json=current_input,
                    cookies=cookies,
                )
            else:
                gen_response = await self.server_client.post(
                    server_name=self.config.name,
                    url_path="/v1/responses",
                    json=current_input,
                    cookies=cookies,
                )
            await raise_for_status(gen_response)
            cookies = gen_response.cookies
            model_response_json = await gen_response.json()

            is_summary_turn = use_summary and (turn_index % 2 == 1)
            was_truncated = self._check_truncated(model_response_json)

            if is_summary_turn:
                was_truncated = was_truncated or reasoning_was_truncated
            else:
                reasoning_was_truncated = was_truncated

            verify_request_data = body.model_dump()
            verify_request_data["response"] = model_response_json
            verify_request_data["turn_index"] = turn_index
            verify_request_data["was_truncated"] = was_truncated
            verify_request_data["is_summary_turn"] = is_summary_turn
            verify_request_data["existing_summary"] = existing_summary

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

            attempt_record = {
                "turn_index": turn_index,
                "input": input_dict,
                "response": model_response_json,
                "generation": generation_text,
                "reward": verify_result.get("reward", 0.0),
                "is_summary_turn": is_summary_turn,
                "turn_info": verify_result.get("turn_info", {}),
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

            # Track summary state
            turn_info = verify_result.get("turn_info", {})
            if is_summary_turn:
                existing_summary = turn_info.get("_existing_summary", existing_summary)

            # Build next input
            next_is_summary_prompt = verify_result.get("is_summary_prompt", False)

            params = body.responses_create_params
            current_input = {
                "input": [{"role": "user", "content": correction_prompt}],
                "model": getattr(params, "model", None) if hasattr(params, "model") else params.get("model"),
            }
            for key in ["temperature", "top_p"]:
                value = getattr(params, key, None) if hasattr(params, key) else params.get(key)
                if value is not None:
                    current_input[key] = value
            next_turn_max_output_tokens = self._turn_max_output_tokens(
                is_summary_prompt=next_is_summary_prompt
            )
            if next_turn_max_output_tokens is not None:
                current_input["max_output_tokens"] = next_turn_max_output_tokens

            turn_index += 1

        final_response = MultiturnProofVerifyResponse.model_validate(verify_result)
        final_response.total_turns = turn_index + 1
        if self.config.include_all_attempts:
            final_response.all_attempts = all_attempts

        return final_response

    @staticmethod
    def _check_truncated(model_response_json: dict) -> bool:
        """Check if the model response was truncated (hit length limit)."""
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
    MultiturnProofAgent.run_webserver()
