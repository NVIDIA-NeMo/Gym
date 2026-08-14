# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses-native, context-compaction-aware Nemotron OSWorld agent.

The legacy ``app.py`` remains the reference chat-completions harness.  This
module uses the pinned Nemotron prompt and parser, but sends every action turn
through the model server's Responses API and records it in
``ContextCompactionSession`` before executing the parsed action.
"""

import asyncio
import base64
import json
import logging
import os
from typing import Any

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import Body
from nemo_gym.context_compaction import (
    ContextCompactedResponse,
    ContextCompactedTransportResponse,
    ContextCompactionSession,
    PreparedContextCompactionCall,
    build_generation_contract,
    build_transport_response,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from nemo_gym.visual_history import ContextMeasurements, VisualHistoryConfig
from responses_api_agents.nemotron_osworld.app import (
    _PYAUTOGUI_PKGS_PREFIX,
    NemotronOSWorldAgent,
    NemotronOSWorldAgentConfig,
    _extract_instruction,
)


logger = logging.getLogger("nemo_gym.osworld.nemotron_cc_agent")

_CC_ROLLOUT_ID_COOKIE = "_nemo_gym_osworld_cc_rollout_id"


class NemotronOSWorldCCAgentConfig(NemotronOSWorldAgentConfig):
    visual_history: VisualHistoryConfig = Field(default_factory=VisualHistoryConfig)


class NemotronOSWorldCCRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    context_compaction_rollout_id: str | None = None
    context_compaction_group_id: str | None = None
    context_compaction_task_id: str | None = None
    context_compaction_rollout_index: int | None = Field(default=None, ge=0)
    context_compaction_attempt_index: int | None = Field(default=None, ge=0)


class NemotronOSWorldCCVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class NemotronOSWorldCCVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: ContextCompactedTransportResponse | ContextCompactedResponse | NeMoGymResponse


def _nemotron_contract() -> tuple[str, str, str, Any]:
    """Load the prompt constants and parser from the pinned OSWorld package."""

    from mm_agents.nvidia.nemotron_agent import (  # noqa: PLC0415
        INSTRUCTION_TEMPLATE,
        SYSTEM_PROMPT_NON_THINKING,
        SYSTEM_PROMPT_THINKING,
        parse_response_to_cot_and_action,
    )

    return (
        SYSTEM_PROMPT_THINKING,
        SYSTEM_PROMPT_NON_THINKING,
        INSTRUCTION_TEMPLATE,
        parse_response_to_cot_and_action,
    )


def _validated_model_body(
    body: NeMoGymResponseCreateParamsNonStreaming,
    *,
    request_input: list[Any],
    required_prefix_token_ids: list[int] | None,
) -> NeMoGymResponseCreateParamsNonStreaming:
    payload = body.model_dump(mode="python", exclude_unset=True)
    payload.update(
        {
            "input": request_input,
            "required_prefix_token_ids": required_prefix_token_ids,
        }
    )
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)


def _merge_usage(
    accumulated: NeMoGymResponseUsage | None,
    current: NeMoGymResponseUsage | None,
) -> NeMoGymResponseUsage | None:
    if current is None:
        return accumulated
    if accumulated is None:
        return current.model_copy(deep=True)
    accumulated.input_tokens += current.input_tokens
    accumulated.output_tokens += current.output_tokens
    accumulated.total_tokens += current.total_tokens
    accumulated.input_tokens_details.cached_tokens = 0
    accumulated.output_tokens_details.reasoning_tokens = 0
    return accumulated


def _extract_model_text(output_items: list[Any]) -> dict[str, str]:
    """Convert Responses output items to the mapping expected by the vendor parser."""

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in output_items:
        payload = item.model_dump(mode="python") if hasattr(item, "model_dump") else item
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "reasoning":
            for part in payload.get("summary") or []:
                if isinstance(part, dict) and part.get("text"):
                    reasoning_parts.append(str(part["text"]))
            continue
        if payload.get("role") != "assistant":
            continue
        content = payload.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]))
    if not any(part.strip() for part in text_parts):
        logger.warning(
            "Nemotron Responses call produced no assistant output text; "
            "the rollout will be marked as FAIL"
        )
    return {
        "content": "\n".join(text_parts),
        "reasoning_content": "\n".join(reasoning_parts),
    }


def _screenshot_observation(image_base64: str, text: str) -> NeMoGymEasyInputMessage:
    return NeMoGymEasyInputMessage(
        role="user",
        content=[
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_base64}",
                "detail": "auto",
            },
            {"type": "input_text", "text": text},
        ],
    )


class NemotronOSWorldCCAgent(NemotronOSWorldAgent):
    config: NemotronOSWorldCCAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> ContextCompactedResponse | NeMoGymResponse:
        body = body.model_copy(
            deep=True,
            update={
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_output_tokens": self.config.max_tokens,
            },
        )
        instruction = _extract_instruction(body.input)
        if not instruction:
            raise RuntimeError("Nemotron OSWorld request has no task instruction")

        thinking_prompt, non_thinking_prompt, instruction_template, parser = _nemotron_contract()
        system_prompt = (thinking_prompt if self.config.thinking else non_thinking_prompt).replace(
            "{password}", self.config.client_password
        )
        agent_input = [NeMoGymEasyInputMessage(role="system", content=system_prompt)]
        instruction_prompt = instruction_template.format(instruction=instruction)

        session_id = request.session[SESSION_ID_KEY]
        rollout_id = request.cookies.get(_CC_ROLLOUT_ID_COOKIE) or str(session_id)
        resources_name = self.config.resources_server.name
        resources_cookies = {
            key: value for key, value in request.cookies.items() if key != _CC_ROLLOUT_ID_COOKIE
        }
        model_cookies = None
        action_history: list[str] = []
        transcript: list[Any] = []
        seed_obs: list[Any] = []
        usage = None
        last_response: NeMoGymResponse | None = None
        context_session: ContextCompactionSession | None = None

        debug_dir = None
        if self.config.debug_trajectory_dir:
            debug_dir = os.path.join(self.config.debug_trajectory_dir, session_id[:8])
            os.makedirs(debug_dir, exist_ok=True)

        async def measure_context(call: PreparedContextCompactionCall) -> ContextMeasurements:
            nonlocal model_cookies
            prompt_token_count = 0
            guards = self.config.visual_history.guards
            if guards.max_total_tokens is not None:
                tokenize_body = _validated_model_body(
                    body,
                    request_input=list(call.request_input),
                    required_prefix_token_ids=(
                        list(call.required_prefix_token_ids)
                        if call.required_prefix_token_ids is not None
                        else None
                    ),
                )
                tokenize_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/tokenize",
                    json=tokenize_body,
                    cookies=model_cookies,
                )
                await raise_for_status(tokenize_response)
                tokenize_payload = await get_response_json(tokenize_response)
                tokens = tokenize_payload.get("tokens")
                if not isinstance(tokens, list) or not all(isinstance(token_id, int) for token_id in tokens):
                    raise RuntimeError("Model tokenize preflight returned invalid tokens")
                prompt_token_count = len(tokens)
                if tokenize_response.cookies:
                    model_cookies = tokenize_response.cookies
            active_images = len(call.prepared_history.view.media_ids)
            return ContextMeasurements(
                prompt_token_count=prompt_token_count,
                active_image_count=active_images,
                vision_token_count=active_images * (guards.projected_vision_tokens_per_image or 0),
            )

        terminal = False
        for step_idx in range(self.config.max_steps):
            shot_response = await self.server_client.post(
                server_name=resources_name,
                url_path="/screenshot",
                cookies=resources_cookies,
            )
            await raise_for_status(shot_response)
            resources_cookies = shot_response.cookies
            shot_payload = await get_response_json(shot_response)
            image_base64 = shot_payload.get("image_base64")
            if not isinstance(image_base64, str) or not image_base64:
                raise RuntimeError("OSWorld screenshot response has no image_base64")

            observation = _screenshot_observation(
                image_base64,
                instruction_prompt + f"You are currently on Step {step_idx + 1}.\n",
            )
            if step_idx == 0:
                seed_obs = [observation]
                transcript.append(observation)
                if self.config.visual_history.enabled:
                    context_session = ContextCompactionSession(
                        config=self.config.visual_history,
                        rollout_id=rollout_id,
                        generation_contract=build_generation_contract(
                            body=body,
                            model_server=self.config.model_server,
                            visual_history=self.config.visual_history,
                        ),
                        initial_context=agent_input,
                        seed_observations=seed_obs,
                    )
            else:
                transcript.append(observation)
                if context_session is not None:
                    context_session.append_observation(
                        [observation],
                        turn_id=step_idx,
                        conditions_action_turn=step_idx + 1,
                    )

            legacy_request_input = [*agent_input, *transcript]
            prepared_call = None
            if context_session is not None:
                prepared_call = await context_session.prepare_model_call(
                    legacy_request_input=legacy_request_input,
                    turn_id=step_idx + 1,
                    measure_context=measure_context,
                )
                request_input = list(prepared_call.request_input)
                required_prefix_token_ids = (
                    list(prepared_call.required_prefix_token_ids)
                    if prepared_call.required_prefix_token_ids is not None
                    else None
                )
            else:
                request_input = legacy_request_input
                required_prefix_token_ids = None

            model_body = _validated_model_body(
                body,
                request_input=request_input,
                required_prefix_token_ids=required_prefix_token_ids,
            )
            model_http_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=model_body,
                cookies=model_cookies,
            )
            await raise_for_status(model_http_response)
            model_payload = await get_response_json(model_http_response)
            model_cookies = model_http_response.cookies
            try:
                last_response = NeMoGymResponse.model_validate(model_payload)
            except ValidationError as exc:
                raise RuntimeError(
                    "Received an invalid response from model server: " + json.dumps(model_payload)
                ) from exc

            if context_session is not None:
                assert prepared_call is not None
                context_session.record_model_response(
                    call=prepared_call,
                    output_items=last_response.output,
                    finish_reason=(
                        last_response.incomplete_details.reason
                        if last_response.incomplete_details is not None
                        else None
                    ),
                )
            usage = _merge_usage(usage, last_response.usage)
            transcript.extend(last_response.output)
            parser_input = _extract_model_text(last_response.output)
            if last_response.incomplete_details is not None:
                # Hitting a policy generation limit is rollout data, not an
                # infrastructure failure. Keep the exact response in the CC
                # trace, then score this rollout as unsuccessful.
                incomplete_reason = last_response.incomplete_details.reason
                logger.warning(
                    "Policy response was incomplete (%s); marking rollout as FAIL",
                    incomplete_reason,
                )
                low_level_instruction = f"Policy response was incomplete: {incomplete_reason}"
                actions = ["FAIL"]
                cot = {"code": "FAIL", "thinking": parser_input.get("reasoning_content", "")}
            else:
                try:
                    low_level_instruction, actions, cot = parser(
                        parser_input,
                        (self.config.screen_width, self.config.screen_height),
                        self.config.coordinate_type,
                        thinking=self.config.thinking,
                    )
                except Exception:
                    # A malformed policy action is rollout data, not an infrastructure
                    # failure. Terminate only this rollout so GRPO can score it as an
                    # unsuccessful sample instead of aborting the entire batch.
                    logger.warning(
                        "Policy response could not be parsed; marking rollout as FAIL",
                        exc_info=True,
                    )
                    low_level_instruction = "Policy response could not be parsed"
                    actions = ["FAIL"]
                    cot = {"code": "FAIL", "thinking": parser_input.get("reasoning_content", "")}
            if (
                not actions
                or str(low_level_instruction).startswith(":")
                or not isinstance(cot, dict)
                or not cot.get("code")
            ):
                logger.warning(
                    "Policy response parsed to an invalid action; marking rollout as FAIL: %s",
                    low_level_instruction,
                )
                actions = ["FAIL"]
                cot = {"code": "FAIL", "thinking": parser_input.get("reasoning_content", "")}

            if step_idx + 1 >= self.config.max_steps and actions[0] not in ("DONE", "FAIL"):
                actions = ["FAIL"]

            if debug_dir is not None:
                with open(os.path.join(debug_dir, f"step_{step_idx:03d}.png"), "wb") as screenshot_file:
                    screenshot_file.write(base64.b64decode(image_base64))
                with open(os.path.join(debug_dir, "trace.jsonl"), "a") as trace_file:
                    trace_file.write(
                        json.dumps(
                            {
                                "step": step_idx,
                                "actions": actions,
                                "message": parser_input["content"][:4000],
                            }
                        )
                        + "\n"
                    )

            for action in actions:
                action_history.append(action)
                if action == "WAIT":
                    await asyncio.sleep(self.config.sleep_after_execution_s)
                    continue
                if action in ("FAIL", "DONE"):
                    terminal = True
                    break
                execute_response = await self.server_client.post(
                    server_name=resources_name,
                    url_path="/execute",
                    json={
                        "command": [
                            "python",
                            "-c",
                            _PYAUTOGUI_PKGS_PREFIX.format(command=action),
                        ],
                        "shell": False,
                    },
                    cookies=resources_cookies,
                )
                resources_cookies = execute_response.cookies
                await asyncio.sleep(self.config.sleep_after_execution_s)
            if terminal:
                break

        self.session_id_to_action_history[session_id] = action_history
        for key, value in (*resources_cookies.items(), *((model_cookies or {}).items())):
            response.set_cookie(key, value)

        if last_response is None:
            raise RuntimeError("Nemotron OSWorld rollout made no model calls")
        last_response.output = transcript[1:]
        last_response.usage = usage
        if context_session is not None:
            context_session.finalize()
            if context_session.authority_mode:
                return context_session.build_response(
                    last_response,
                    output=transcript[1:],
                    agent_input=agent_input,
                    seed_obs=seed_obs,
                )
        return last_response

    async def run(
        self,
        request: Request,
        body: NemotronOSWorldCCRunRequest,
    ) -> NemotronOSWorldCCVerifyResponse:
        """Seed, act, officially verify, then emit the schema-3 CC transport."""

        if self._rollout_semaphore is None:
            self._rollout_semaphore = asyncio.Semaphore(max(1, self.config.max_parallel_rollouts))
        async with self._rollout_semaphore:
            # Unlike the legacy harness, authority mode deliberately propagates
            # failures instead of manufacturing a contractless masked sample.
            return await self._run_cc_rollout(request, body)

    async def _run_cc_rollout(
        self,
        request: Request,
        body: NemotronOSWorldCCRunRequest,
    ) -> NemotronOSWorldCCVerifyResponse:
        cookies = dict(request.cookies)
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies.update(seed_response.cookies)
        if body.context_compaction_rollout_id is not None:
            cookies[_CC_ROLLOUT_ID_COOKIE] = body.context_compaction_rollout_id

        agent_response_http = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(agent_response_http)
        cookies.update(agent_response_http.cookies)
        agent_payload = await get_response_json(agent_response_http)
        compacted_response = None
        if agent_payload.get("context_compaction_contract") is not None:
            compacted_response = ContextCompactedResponse.model_validate(agent_payload)
        elif self.config.visual_history.enabled:
            raise RuntimeError("CC-enabled OSWorld agent returned no context-compaction contract")

        history_response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/action_history",
            cookies=cookies,
        )
        await raise_for_status(history_response)
        cookies.update(history_response.cookies)
        action_history = (await get_response_json(history_response)).get("action_history") or []

        verify_request = NemotronOSWorldCCVerifyRequest.model_validate(
            body.model_dump()
            | {
                "response": agent_payload,
                "action_history": action_history,
            }
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verified = NemotronOSWorldCCVerifyResponse.model_validate(await get_response_json(verify_response))
        if compacted_response is None:
            return verified

        contract = compacted_response.context_compaction_contract
        stamped = compacted_response.model_copy(
            update={
                "context_compaction_contract": contract.model_copy(
                    update={
                        "group_id": body.context_compaction_group_id,
                        "task_id": body.context_compaction_task_id,
                        "rollout_index": body.context_compaction_rollout_index,
                        "attempt_index": body.context_compaction_attempt_index,
                    }
                )
            }
        )
        return verified.model_copy(update={"response": build_transport_response(stamped)})


if __name__ == "__main__":
    NemotronOSWorldCCAgent.run_webserver()
