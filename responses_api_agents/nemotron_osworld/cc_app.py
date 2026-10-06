# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses-native Nemotron OSWorld agent with shared semantic context management."""

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientTimeout
from fastapi import Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import Body
from nemo_gym.context_management import (
    ContextGuardRejected,
    ContextHistoryConfig,
    ContextManagedResponsesClient,
    LogicalCCResult,
)
from nemo_gym.context_management.result import capture_rollout_id
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from responses_api_agents.nemotron_osworld.app import (
    _PYAUTOGUI_PKGS_PREFIX,
    NemotronOSWorldAgent,
    NemotronOSWorldAgentConfig,
    NemotronOSWorldVerifyRequest,
    NemotronOSWorldVerifyResponse,
    _extract_instruction,
)


logger = logging.getLogger("nemo_gym.osworld.nemotron_cc_agent")


class NemotronOSWorldCCAgentConfig(NemotronOSWorldAgentConfig):
    """Context-managed OSWorld policy and long-horizon operation bounds."""

    token_id_capture: bool = True
    context_history: ContextHistoryConfig = Field(default_factory=ContextHistoryConfig)
    rollout_timeout_s: float = Field(default=1200.0, gt=0)
    action_timeout_s: float = Field(default=60.0, gt=0)
    llm_timeout_s: float = Field(default=900.0, gt=0)


class NemotronOSWorldCCRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NemotronOSWorldCCVerifyResponse(NemotronOSWorldVerifyResponse):
    model_config = ConfigDict(extra="allow")
    context_compaction_result: LogicalCCResult


class ContextManagedOSWorldResponse(NeMoGymResponse):
    context_compaction_result: LogicalCCResult


@dataclass
class _EpisodeResult:
    response: NeMoGymResponse
    context_compaction_result: LogicalCCResult
    action_history: list[str]
    resources_cookies: dict[str, str]
    model_cookies: dict[str, str]


def _nemotron_contract() -> tuple[str, str, str, Any]:
    """Load the validated prompt constants and parser from the pinned OSWorld package."""

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


def _cookie_values(cookies: Any) -> dict[str, str]:
    """Normalize aiohttp/Starlette cookie mappings without sharing mutable jars."""

    return {
        str(key): str(getattr(value, "value", value))
        for key, value in (cookies.items() if cookies is not None else ())
    }


def _extract_model_text(output_items: list[Any]) -> dict[str, str]:
    """Convert Responses output items to the mapping expected by the vendor parser."""

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in output_items:
        payload = item.model_dump(mode="python") if hasattr(item, "model_dump") else item
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "reasoning":
            content = payload.get("content")
            if isinstance(content, str):
                reasoning_parts.append(content)
            for part in [*(payload.get("summary") or []), *(content if isinstance(content, list) else [])]:
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
        logger.warning("Nemotron Responses call produced no assistant output text")
    return {
        "content": "\n".join(text_parts),
        "reasoning_content": "\n".join(reasoning_parts),
    }


def _normalize_vendor_parser_input(response: dict[str, str]) -> dict[str, str]:
    """Restore the Action header omitted by some Responses generations.

    The pinned vendor parser requires both ``## Action`` and ``## Code``.
    Nano-v3 sometimes emits the action description as plain final text followed
    by a valid Code section; treating that as FAIL ends an otherwise executable
    OSWorld rollout after one model call.
    """

    content = response["content"]
    if re.search(r"^\s*##\s*Action\b", content, re.MULTILINE):
        return response
    code_match = re.search(r"^\s*##\s*Code\s*:?", content, re.MULTILINE)
    if code_match is None:
        return response
    implicit_action = content[: code_match.start()].strip()
    if not implicit_action:
        return response
    return {
        **response,
        "content": (f"## Action:\n{implicit_action}\n\n{content[code_match.start() :]}"),
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


def _without_raw_images(items: list[dict[str, Any]] | str) -> list[dict[str, Any]] | str:
    """Project response echoes after verification; the receipt owns each media asset once."""

    if isinstance(items, str):
        return items
    projected_items: list[dict[str, Any]] = []
    for item in items:
        projected = dict(item)
        if isinstance(projected.get("content"), list):
            projected["content"] = [
                part
                for part in projected["content"]
                if not (isinstance(part, dict) and part.get("type") in {"input_image", "image", "image_url"})
            ]
        projected_items.append(projected)
    return projected_items


class NemotronOSWorldCCAgent(NemotronOSWorldAgent):
    config: NemotronOSWorldCCAgentConfig

    def _require_owner(self, owner: str | None) -> str:
        if self.config.token_id_capture and (not self._token_id_capture_enabled() or owner is None):
            raise ValueError("Nemotron OSWorld CC requires token capture and an explicit _ng_rollout_id")
        if owner is None:
            owner = f"inference-{uuid.uuid4().hex}_g0"
        capture_rollout_id(owner, 0)
        return owner

    def _model_request(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        system_prompt: str,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        payload = body.model_dump(mode="python")
        payload.update(
            {
                "input": [NeMoGymEasyInputMessage(role="system", content=system_prompt)],
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_output_tokens": self.config.max_tokens,
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            }
        )
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        logical_rollout_id: str,
        session_id: str,
        resources_cookies: Any = None,
    ) -> _EpisodeResult:
        instruction = _extract_instruction(body.input)
        if not instruction:
            raise RuntimeError("Nemotron OSWorld request has no task instruction")

        thinking_prompt, non_thinking_prompt, instruction_template, parser = _nemotron_contract()
        system_prompt = (thinking_prompt if self.config.thinking else non_thinking_prompt).replace(
            "{password}", self.config.client_password
        )
        instruction_prompt = instruction_template.format(instruction=instruction)
        model_request = self._model_request(body, system_prompt=system_prompt)

        resources_name = self.config.resources_server.name
        resource_cookie_values = _cookie_values(resources_cookies)
        action_history: list[str] = []
        usage = None
        client: ContextManagedResponsesClient | None = None
        last_response: NeMoGymResponse | None = None
        outcome = "completed"
        model_call_failed_after_selection = False
        rollout_started_at = time.monotonic()

        debug_dir = None
        if self.config.debug_trajectory_dir:
            debug_dir = os.path.join(self.config.debug_trajectory_dir, session_id[:8])
            os.makedirs(debug_dir, exist_ok=True)

        def time_left() -> float:
            return self.config.rollout_timeout_s - (time.monotonic() - rollout_started_at)

        terminal = False
        for step_idx in range(self.config.max_steps):
            remaining_s = time_left()
            if remaining_s <= 0:
                if last_response is None:
                    raise TimeoutError("OSWorld rollout timed out before its first model response")
                logger.warning("OSWorld rollout timed out at step %d", step_idx + 1)
                action_history.append("FAIL")
                outcome = "execution_failure"
                break

            try:
                shot_response = await self.server_client.post(
                    server_name=resources_name,
                    url_path="/screenshot",
                    cookies=resource_cookie_values,
                    timeout=ClientTimeout(total=remaining_s),
                    _retry=False,
                )
                await raise_for_status(shot_response)
                resource_cookie_values.update(_cookie_values(shot_response.cookies))
                shot_payload = await get_response_json(shot_response)
            except Exception:
                if last_response is None:
                    raise
                logger.warning(
                    "OSWorld screenshot failed at step %d; preserving the selected trace as a failed rollout",
                    step_idx + 1,
                    exc_info=True,
                )
                action_history.append("FAIL")
                outcome = "execution_failure"
                break

            image_base64 = shot_payload.get("image_base64")
            if not isinstance(image_base64, str) or not image_base64:
                if last_response is None:
                    raise RuntimeError("OSWorld screenshot response has no image_base64")
                action_history.append("FAIL")
                outcome = "execution_failure"
                break

            observation = _screenshot_observation(
                image_base64,
                instruction_prompt + f"You are currently on Step {step_idx + 1}.\n",
            )
            if client is None:
                opening_request = NeMoGymResponseCreateParamsNonStreaming.model_validate(
                    model_request.model_dump(mode="python") | {"input": [*model_request.input, observation]}
                )
                client = ContextManagedResponsesClient(
                    server_client=self.server_client,
                    model_server=self.config.model_server,
                    logical_rollout_id=logical_rollout_id,
                    config=self.config.context_history,
                    initial_request=opening_request,
                    token_capture=self._token_id_capture_enabled(),
                )
            else:
                client.append_observation([observation])

            try:
                async with asyncio.timeout(min(self.config.llm_timeout_s, max(time_left(), 0.001))):
                    current_response = await client.create()
            except TimeoutError:
                if last_response is None:
                    raise
                logger.warning(
                    "OSWorld model call timed out at step %d; preserving the last selected trace as a failed rollout",
                    step_idx + 1,
                )
                action_history.append("FAIL")
                outcome = "execution_failure"
                model_call_failed_after_selection = True
                break
            except ContextGuardRejected:
                logger.warning("OSWorld context guard rejected step %d", step_idx + 1)
                # The shared client closes after any failed generation admission.
                # Returning a partial receipt would falsely attest a terminal action.
                raise
            except Exception:
                if last_response is None:
                    raise
                logger.warning(
                    "OSWorld model call failed at step %d; preserving the last selected trace as a failed rollout",
                    step_idx + 1,
                    exc_info=True,
                )
                action_history.append("FAIL")
                outcome = "execution_failure"
                model_call_failed_after_selection = True
                break

            last_response = current_response
            usage = accumulate_response_usage(usage, current_response.usage)
            if current_response.error is not None or current_response.status == "failed":
                action_history.append("FAIL")
                outcome = "execution_failure"
                break
            if current_response.incomplete_details is not None:
                action_history.append("FAIL")
                outcome = (
                    "max_output_tokens"
                    if current_response.incomplete_details.reason == "max_output_tokens"
                    else "execution_failure"
                )
                break

            parser_input = _normalize_vendor_parser_input(_extract_model_text(current_response.output))
            try:
                low_level_instruction, actions, cot = parser(
                    parser_input,
                    (self.config.screen_width, self.config.screen_height),
                    self.config.coordinate_type,
                    thinking=self.config.thinking,
                )
            except Exception:  # noqa: BLE001 - malformed policy output is rollout data
                logger.warning("Policy response could not be parsed; marking it as FAIL", exc_info=True)
                low_level_instruction, actions, cot = (
                    "Policy response could not be parsed",
                    ["FAIL"],
                    {"code": "FAIL"},
                )
            if (
                not actions
                or str(low_level_instruction).startswith(":")
                or not isinstance(cot, dict)
                or not cot.get("code")
            ):
                logger.warning("Policy response parsed to an invalid action: %s", low_level_instruction)
                actions = ["FAIL"]

            if step_idx + 1 >= self.config.max_steps and actions[0] not in ("DONE", "FAIL"):
                actions = ["FAIL"]
                outcome = "max_steps"

            if debug_dir is not None:
                with open(os.path.join(debug_dir, f"step_{step_idx:03d}.png"), "wb") as screenshot_file:
                    screenshot_file.write(base64.b64decode(image_base64))
                with open(os.path.join(debug_dir, "trace.jsonl"), "a", encoding="utf-8") as trace_file:
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
                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(self.config.sleep_after_execution_s),
                            timeout=max(time_left(), 0.001),
                        )
                    except TimeoutError:
                        action_history.append("FAIL")
                        outcome = "execution_failure"
                        terminal = True
                    continue
                if action in ("FAIL", "DONE"):
                    terminal = True
                    break
                try:
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
                        cookies=resource_cookie_values,
                        timeout=ClientTimeout(total=min(self.config.action_timeout_s, max(time_left(), 0.001))),
                        _retry=False,
                    )
                    resource_cookie_values.update(_cookie_values(execute_response.cookies))
                    await asyncio.wait_for(
                        asyncio.sleep(self.config.sleep_after_execution_s),
                        timeout=max(time_left(), 0.001),
                    )
                except Exception:  # noqa: BLE001 - isolate an ambiguous guest action to this rollout
                    logger.warning(
                        "OSWorld action failed; preserving the trace as an execution failure", exc_info=True
                    )
                    action_history.append("FAIL")
                    outcome = "execution_failure"
                    terminal = True
                    break
            if terminal:
                if outcome == "max_steps":
                    break
                if outcome != "execution_failure":
                    outcome = "completed"
                break

        if client is None or last_response is None:
            raise RuntimeError("Nemotron OSWorld rollout made no model calls")
        if model_call_failed_after_selection:
            context_compaction_result = client.finish_after_failed_call(last_response, outcome=outcome)
        else:
            context_compaction_result = client.finish(last_response, outcome=outcome)
        semantic_response = NeMoGymResponse.model_validate(
            last_response.model_dump() | {"output": client.output_items, "usage": usage}
        )
        return _EpisodeResult(
            response=semantic_response,
            context_compaction_result=context_compaction_result,
            action_history=action_history,
            resources_cookies=resource_cookie_values,
            model_cookies=_cookie_values(client.cookies),
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> ContextManagedOSWorldResponse:
        owner = self._require_owner(request.path_params.get("rollout_id"))
        session_id = request.session[SESSION_ID_KEY]
        episode = await self._create_episode(
            body,
            logical_rollout_id=owner,
            session_id=session_id,
            resources_cookies=request.cookies,
        )
        self.session_id_to_action_history[session_id] = episode.action_history
        for key, value in {**episode.resources_cookies, **episode.model_cookies}.items():
            response.set_cookie(key, value)
        return ContextManagedOSWorldResponse.model_validate(
            episode.response.model_dump()
            | {"context_compaction_result": episode.context_compaction_result.model_dump(mode="python")}
        )

    async def run(
        self,
        request: Request,
        body: NemotronOSWorldCCRunRequest,
    ) -> NemotronOSWorldCCVerifyResponse:
        owner = self._require_owner(body.capture_rollout_id)
        if self._rollout_semaphore is None:
            self._rollout_semaphore = asyncio.Semaphore(max(1, self.config.max_parallel_rollouts))
        async with self._rollout_semaphore:
            return await self._run_cc_rollout(request, body, owner)

    async def _run_cc_rollout(
        self,
        request: Request,
        body: NemotronOSWorldCCRunRequest,
        owner: str,
    ) -> NemotronOSWorldCCVerifyResponse:
        cookies = _cookie_values(request.cookies)
        seeded = False
        episode: _EpisodeResult | None = None
        try:
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
                timeout=ClientTimeout(total=self.config.rollout_timeout_s),
                _retry=False,
            )
            await raise_for_status(seed_response)
            cookies.update(_cookie_values(seed_response.cookies))
            seeded = True

            episode = await self._create_episode(
                body.responses_create_params,
                logical_rollout_id=owner,
                session_id=owner,
                resources_cookies=cookies,
            )
            cookies.update(episode.resources_cookies)
            verify_request = NemotronOSWorldVerifyRequest.model_validate(
                body.model_dump()
                | {
                    "response": episode.response.model_dump(mode="python"),
                    "action_history": episode.action_history,
                }
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
                timeout=ClientTimeout(total=self.config.rollout_timeout_s),
                _retry=False,
            )
            await raise_for_status(verify_response)
            verified = await get_response_json(verify_response)
            if (verified.get("response") or {}).get("id") != episode.response.id:
                raise ValueError("OSWorld verifier changed the final selected model response identity")

            transport_response = dict(verified["response"])
            transport_response["output"] = _without_raw_images(transport_response.get("output", []))
            request_echo = dict(verified["responses_create_params"])
            request_echo["input"] = _without_raw_images(request_echo["input"])
            instance_config = dict(verified.get("instance_config") or {})
            if episode.context_compaction_result.outcome == "execution_failure" or verified.get("verify_error"):
                instance_config["mask_sample"] = True
            return NemotronOSWorldCCVerifyResponse.model_validate(
                verified
                | {
                    "response": transport_response,
                    "responses_create_params": request_echo,
                    "instance_config": instance_config,
                    "context_compaction_result": episode.context_compaction_result.model_dump(mode="python"),
                }
            )
        finally:
            if seeded:
                # /verify normally releases in its own finally. /release is idempotent
                # and covers every pre-verification exception without replaying evaluation.
                try:
                    await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/release",
                        cookies={**cookies, **(episode.resources_cookies if episode else {})},
                        timeout=ClientTimeout(total=self.config.action_timeout_s),
                        _retry=False,
                    )
                except Exception:  # noqa: BLE001 - cleanup must not hide the rollout result
                    logger.warning("OSWorld sandbox release failed", exc_info=True)


if __name__ == "__main__":
    NemotronOSWorldCCAgent.run_webserver()
