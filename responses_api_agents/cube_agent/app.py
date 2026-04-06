# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responses API agent for any CUBE resources server (``resources_servers.cube`` contract)."""

from __future__ import annotations

import copy
import json
import logging
from typing import List, Sequence, cast

import aiohttp
from fastapi import HTTPException
from pydantic import ConfigDict, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseInputItem,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
)
from resources_servers.cube.schemas import (
    CubeAgentVerifyRequest,
    CubeAgentVerifyResponse,
    CubeEnvStateEasyInputMessage,
    CubeNeMoGymResponse,
    CubeSeedSessionResponse,
    CubeStepResponse,
)


logger = logging.getLogger(__name__)


def rebuild_input_last_n_env_observations(
    static_prefix: Sequence[NeMoGymResponseInputItem],
    seed_obs: Sequence[NeMoGymResponseInputItem],
    step_model_outputs: list[list[NeMoGymResponseOutputItem]],
    step_observations: list[Sequence[NeMoGymResponseInputItem]],
    n: int,
    placeholder: str,
) -> list[NeMoGymResponseInputItem]:
    """Rebuild chat input: ``static_prefix`` + seed obs + (model → obs)* with at most ``n`` env bundles.

    Env bundles are ``seed_obs`` plus each successful ``/step`` ``obs`` list. Each bundle after seed is
    preceded by the model ``output`` that produced the tool calls for that step.
    """
    if n < 1:
        raise ValueError("render_last_n_obs must be >= 1 when enabled")

    env_blocks: list[Sequence[NeMoGymResponseInputItem]] = [seed_obs] + step_observations
    l_blocks = len(env_blocks)
    out: list[NeMoGymResponseInputItem] = list(static_prefix)

    if l_blocks <= n:
        out.extend(_deepcopy_items(seed_obs))
        for i in range(len(step_observations)):
            out.extend(_deepcopy_items(step_model_outputs[i]))
            out.extend(_deepcopy_items(step_observations[i]))
        return out

    keep_indices = [0] + list(range(l_blocks - n + 1, l_blocks))
    out.extend(_deepcopy_items(env_blocks[0]))

    first_after_seed = keep_indices[1]
    if first_after_seed > 1:
        out.append(
            NeMoGymEasyInputMessage(
                role="user",
                content=placeholder,
                type="message",
            )
        )

    for idx in keep_indices[1:]:
        out.extend(_deepcopy_items(step_model_outputs[idx - 1]))
        out.extend(_deepcopy_items(env_blocks[idx]))
    return out


def _deepcopy_items(items: Sequence[NeMoGymResponseInputItem]) -> list[NeMoGymResponseInputItem]:
    return [copy.deepcopy(x) for x in items]


class CubeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int | None = Field(
        default=None,
        description="Maximum environment steps. If unset, runs until done or model failure.",
    )
    return_transitions: bool = Field(
        default=True,
        description="If True, return list of transitions; else flatten trajectory items.",
    )
    max_total_sequence_length: int | None = Field(
        default=None,
        description="Optional cap on serialized trajectory length to avoid context overflows.",
    )
    done_if_no_tool_calls: bool = Field(
        default=True,
        description="If True, end when the model returns assistant text without tool calls.",
    )
    render_last_n_obs: int | None = Field(
        default=None,
        description="If set (>0), keep at most this many environment observation bundles: "
        "the post-reset observation always stays, plus the newest step observations. "
        "Intermediate bundles and their preceding model outputs are dropped and a short placeholder is inserted. "
        "Similar to cube-harness Genny's render_last_n_obs. When set, collapse_old_env_states is ignored.",
    )
    skipped_observations_placeholder: str = Field(
        default="[Earlier environment observations omitted from context.]",
        description="User message inserted when render_last_n_obs drops intermediate step observations.",
    )
    collapse_old_env_states: bool = Field(
        default=False,
        description="If True, replace prior CubeEnvStateEasyInputMessage turns with a placeholder.",
    )
    old_env_state_message: str = Field(
        default="[Previous environment state - hidden]",
        description="Placeholder used when collapse_old_env_states is True.",
    )


class CubeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    debug_each_step: bool = Field(
        default=False,
        description="When true (e.g. ng_collect_rollouts +debug_each_step=true), log each policy/env loop iteration.",
    )


class CubeAgent(SimpleResponsesAPIAgent):
    config: CubeAgentConfig

    def update_agent_state(
        self,
        agent_state: NeMoGymResponseCreateParamsNonStreaming,
        model_output: list[NeMoGymResponseOutputItem],
        obs: list[NeMoGymEasyInputMessage | NeMoGymFunctionCallOutput],
        successful_transition: bool,
    ) -> NeMoGymResponseCreateParamsNonStreaming:
        prev_messages = agent_state.input
        use_last_n = self.config.render_last_n_obs is not None and self.config.render_last_n_obs > 0
        if successful_transition and self.config.collapse_old_env_states and not use_last_n:
            hidden_message = NeMoGymEasyInputMessage(role="user", content=self.config.old_env_state_message)
            prev_messages = [
                hidden_message if isinstance(m, CubeEnvStateEasyInputMessage) else m for m in prev_messages
            ]

        return agent_state.model_copy(update={"input": prev_messages + model_output + obs})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=5))
    async def _seed_session(self, task_idx: int) -> CubeSeedSessionResponse:
        reset_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={"task_idx": task_idx},
        )
        reset_response.raise_for_status()
        seed_session_response = CubeSeedSessionResponse.model_validate(await reset_response.json())
        if not seed_session_response.obs:
            raise ValueError("No observations in seed session response")
        return seed_session_response

    async def _cube_rollout(self, req: CubeAgentRunRequest) -> tuple[CubeNeMoGymResponse, list[dict]]:
        req = req.model_copy(deep=True)
        body = req.responses_create_params

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        seed_session_response = await self._seed_session(req.task_idx)

        static_prefix = cast(list[NeMoGymResponseInputItem], copy.deepcopy(body.input))
        seed_obs_snapshot = cast(list[NeMoGymResponseInputItem], copy.deepcopy(seed_session_response.obs))
        step_model_outputs: list[list[NeMoGymResponseOutputItem]] = []
        step_observations: list[list[NeMoGymResponseInputItem]] = []
        tail_after_rebuild: list[NeMoGymResponseInputItem] = []
        use_last_n = self.config.render_last_n_obs is not None and self.config.render_last_n_obs > 0

        agent_state = body.model_copy(
            update={
                "input": _deepcopy_items(static_prefix) + _deepcopy_items(seed_obs_snapshot),
                "tools": seed_session_response.tools,
            }
        )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        rollout_failure_detail: str | None = None
        agent_state_history: list[NeMoGymResponseInput] = []
        all_messages: list[NeMoGymResponseOutputItem] = []
        model_server_cookies = None
        debug_step_events: list[dict] = []

        step = 0
        try:
            while True:
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    break
                step += 1
                successful_transition = True

                raw_model_response: aiohttp.ClientResponse | None = None
                policy_body = b""
                try:
                    raw_model_response = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path="/v1/responses",
                        json=agent_state,
                        cookies=model_server_cookies,
                    )
                    policy_body = await raw_model_response.read()
                    if raw_model_response.status >= 400:
                        body_snip = policy_body[:6000].decode(errors="replace")
                        rollout_failure_detail = (
                            f"Policy /v1/responses HTTP {raw_model_response.status}. Body (truncated): {body_snip!r}"
                        )
                        logger.warning("Error calling /v1/responses: %s", rollout_failure_detail)
                        break
                    model_server_cookies = raw_model_response.cookies
                    model_response_json = json.loads(policy_body)
                except json.JSONDecodeError as e:
                    body_snip = policy_body[:2000].decode(errors="replace") if policy_body else ""
                    rollout_failure_detail = (
                        f"Policy /v1/responses returned non-JSON: {e!r}. Body (truncated): {body_snip!r}"
                    )
                    logger.warning("Error calling /v1/responses: %s", rollout_failure_detail)
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    rollout_failure_detail = (
                        f"Policy response failed NeMoGymResponse validation: {e!r}. Raw: {model_response_json!r}"
                    )
                    logger.warning("%s", rollout_failure_detail)
                    break

                model_output = model_response.output
                all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                    o for o in model_output if o.type == "function_call"
                ]
                all_output_messages: List[NeMoGymResponseOutputMessage] = [
                    o for o in model_output if o.type == "message" and o.role == "assistant"
                ]
                done = False

                if all_fn_calls:
                    raw_env_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/step",
                        json={"action": [c.model_dump(mode="json") for c in all_fn_calls], "env_id": env_id},
                    )
                    step_body = await raw_env_response.read()
                    if raw_env_response.status >= 400:
                        body_snip = step_body[:2000].decode(errors="replace")
                        rollout_failure_detail = (
                            f"Resources /step HTTP {raw_env_response.status}. Body (truncated): {body_snip!r}"
                        )
                        logger.warning("%s", rollout_failure_detail)
                        break
                    try:
                        step_payload = json.loads(step_body)
                    except json.JSONDecodeError as e:
                        body_snip = step_body[:2000].decode(errors="replace")
                        rollout_failure_detail = f"Resources /step returned non-JSON: {e!r}. Body: {body_snip!r}"
                        logger.warning("%s", rollout_failure_detail)
                        break
                    try:
                        env_response = CubeStepResponse.model_validate(step_payload)
                    except ValidationError as e:
                        rollout_failure_detail = (
                            f"Resources /step JSON failed validation: {e!r}. Raw: {step_payload!r}"
                        )
                        logger.warning("%s", rollout_failure_detail)
                        break
                    obs = env_response.obs
                    done = env_response.done
                elif all_output_messages:
                    if self.config.done_if_no_tool_calls:
                        done = True
                        obs = []
                    else:
                        obs = [
                            NeMoGymEasyInputMessage(
                                role="user",
                                content="You did not respond with a valid tool call. "
                                "To proceed, please call at least one tool.",
                            )
                        ]
                        successful_transition = False
                else:
                    # Empty output or only non-tool item types (e.g. reasoning): never /step with action=[].
                    if self.config.done_if_no_tool_calls:
                        done = True
                        obs = []
                    else:
                        obs = [
                            NeMoGymEasyInputMessage(
                                role="user",
                                content="Your last reply had no tool calls and no assistant text. "
                                "Call exactly one of the environment tools to act.",
                            )
                        ]
                        successful_transition = False

                from_env_step = len(all_fn_calls) > 0

                if use_last_n:
                    n = cast(int, self.config.render_last_n_obs)
                    if from_env_step:
                        step_model_outputs.append(copy.deepcopy(model_output))
                        step_observations.append(copy.deepcopy(obs))
                        tail_after_rebuild = []
                        trimmed = rebuild_input_last_n_env_observations(
                            static_prefix,
                            seed_obs_snapshot,
                            step_model_outputs,
                            step_observations,
                            n,
                            self.config.skipped_observations_placeholder,
                        )
                        agent_state = agent_state.model_copy(update={"input": trimmed})
                    else:
                        tail_after_rebuild.extend(copy.deepcopy(model_output))
                        tail_after_rebuild.extend(copy.deepcopy(obs))
                        trimmed = rebuild_input_last_n_env_observations(
                            static_prefix,
                            seed_obs_snapshot,
                            step_model_outputs,
                            step_observations,
                            n,
                            self.config.skipped_observations_placeholder,
                        )
                        agent_state = agent_state.model_copy(update={"input": trimmed + tail_after_rebuild})
                else:
                    agent_state = self.update_agent_state(agent_state, model_output, obs, successful_transition)
                if self.config.return_transitions:
                    agent_state_history.append(cast(NeMoGymResponseInput, agent_state.input))
                else:
                    all_messages.extend(model_output)
                    if successful_transition:
                        all_messages.extend(obs)

                if req.debug_each_step:
                    reward_snip: float | None = None
                    done_snip = done
                    if from_env_step:
                        reward_snip = float(env_response.reward)
                        done_snip = env_response.done
                    debug_step_events.append(
                        {
                            "step": step,
                            "tool_calls": len(all_fn_calls),
                            "from_env_step": from_env_step,
                            "reward": reward_snip,
                            "done": done_snip,
                        }
                    )
                    msg = (
                        f"[cube_agent] step={step} tool_calls={len(all_fn_calls)} "
                        f"from_env_step={from_env_step} reward={reward_snip!r} done={done_snip}"
                    )
                    print(msg, flush=True)
                    logger.info("%s", msg)

                if done:
                    break

        finally:
            await self.server_client.post(
                server_name=self.config.resources_server.name, url_path="/close", json={"env_id": env_id}
            )

        if model_response is None:
            raise HTTPException(
                status_code=502,
                detail=rollout_failure_detail
                or "Rollout stopped before a valid policy response (see cube_agent logs).",
            )

        output_overrides = {
            "env_id": env_id,
            "group_id": str(req.task_idx),
            "contains_transitions": self.config.return_transitions,
            "output": agent_state_history if self.config.return_transitions else all_messages,
        }
        output = CubeNeMoGymResponse.model_validate(model_response.model_dump() | output_overrides)
        return output, debug_step_events

    async def responses(self, req: CubeAgentRunRequest) -> CubeNeMoGymResponse:
        out, _ = await self._cube_rollout(req)
        return out

    async def run(self, body: CubeAgentRunRequest) -> CubeAgentVerifyResponse:
        try:
            response, debug_step_events = await self._cube_rollout(body)

            merged: dict = body.model_dump() | {"response": response}
            if body.debug_each_step and debug_step_events:
                merged["debug_step_events"] = debug_step_events
            verify_request = CubeAgentVerifyRequest.model_validate(merged)
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(mode="json"),
            )

            return CubeAgentVerifyResponse.model_validate(await verify_response.json())
        except Exception as e:
            logger.exception("Error in run")
            raise e


if __name__ == "__main__":
    CubeAgent.run_webserver()
