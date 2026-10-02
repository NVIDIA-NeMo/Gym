# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared external-benchmark adapter for AgentDojo-family runtimes."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLMToolFilter
from agentdojo.attacks import load_attack
from agentdojo.task_suite.load_suites import get_suite
from fastapi import Body
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.agentdojo_family.model_bridge import (
    NeMoGymAgentDojoLLM,
    agentdojo_messages_to_response_output,
)


LOG = logging.getLogger(__name__)


class AgentDojoFamilyAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    concurrency: int = Field(default=1, ge=1)
    benchmark_version: str = "v1.2.2"
    attack_model_alias: str = "local"
    defense_model_alias: str | None = None
    model_system_role: Literal["developer", "system"] = "developer"
    metric_prefix: str = "agentdojo"
    # Abandon a rollout that has not finished in this many seconds. None keeps the historical
    # behaviour of waiting forever. See `run` for why waiting forever is not always safe.
    rollout_timeout_seconds: float | None = None
    # Abandoned rollouts leave a thread behind that cannot be cancelled, so a process that has
    # accumulated this many exits, and `gym eval run --resume` restarts from a clean stack.
    max_abandoned_rollouts: int = 2
    default_defense: str | None = None
    system_message: str | None = None
    system_message_name: str | None = None
    tool_output_format: Literal["yaml", "json"] | None = None
    token_id_capture: bool = True


class AgentDojoFamilyRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    suite: str
    user_task_id: str
    injection_task_id: str | None = None
    attack: str | None = None
    defense: str | None = None
    benchmark_version: str | None = None


class AgentDojoFamilyVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    suite: str
    user_task_id: str
    injection_task_id: str | None = None
    attack: str | None = None
    defense: str | None = None
    utility: bool
    security: bool
    attack_success: bool
    reward_utility: float
    reward_security: float
    model_call_count: int
    benchmark_version: str
    adapter_error: str | None = None
    mask_sample: bool = False
    tool_filter_kept_tools: list[str] | None = None


def _empty_response() -> dict[str, Any]:
    return {
        "id": "agentdojo-error",
        "created_at": 0.0,
        "model": "",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _request_options(params: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if "temperature" in params.model_fields_set and params.temperature is not None:
        values["temperature"] = params.temperature
    if "top_p" in params.model_fields_set and params.top_p is not None:
        values["top_p"] = params.top_p
    if "max_output_tokens" in params.model_fields_set and params.max_output_tokens is not None:
        values["max_tokens"] = params.max_output_tokens
    if "parallel_tool_calls" in params.model_fields_set:
        values["parallel_tool_calls"] = params.parallel_tool_calls
    if "reasoning" in params.model_fields_set and params.reasoning is not None and params.reasoning.effort is not None:
        values["reasoning_effort"] = params.reasoning.effort
    return values


def _record_tool_filter_selection(tool_filter: OpenAILLMToolFilter, bridge: NeMoGymAgentDojoLLM) -> None:
    """Record which tools the tool_filter defense kept, so an empty selection is visible in the results.

    The filter asks the model to name the tools a task needs, sending the tool list with
    `tool_choice="none"`, and keeps only tools whose names appear in the reply. Some serving
    configurations drop `tools` from the prompt whenever `tool_choice` is `"none"` (vLLM's
    `--exclude-tools-when-tool-choice-none`); the model then cannot name a real tool, the task runs
    with none, and the collapse would otherwise read as the defense's cost.
    """
    original_query = tool_filter.query

    def recording_query(query, runtime, *args, **kwargs):
        result = original_query(query, runtime, *args, **kwargs)
        bridge.tool_filter_kept_tools = sorted(result[1].functions)
        return result

    tool_filter.query = recording_query


class AgentDojoFamilyAgent(SimpleResponsesAPIAgent):
    config: AgentDojoFamilyAgentConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _abandoned_rollouts: int = PrivateAttr(default=0)

    def model_post_init(self, context: Any) -> None:
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        self._abandoned_rollouts = 0

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("AgentDojo owns the complete rollout; use /run.")

    async def run(self, body: AgentDojoFamilyRunRequest = Body()) -> AgentDojoFamilyVerifyResponse:
        async with self._semaphore:
            if body.defense is None and self.config.default_defense is not None:
                body = body.model_copy(update={"defense": self.config.default_defense})
            benchmark_version = body.benchmark_version or self.config.benchmark_version
            if body.injection_task_id is None and body.attack is not None:
                return self._failure(body, benchmark_version, "attack requires injection_task_id")
            if body.injection_task_id is not None and body.attack is None:
                return self._failure(body, benchmark_version, "injection_task_id requires attack")

            event_loop = asyncio.get_running_loop()
            bridge = NeMoGymAgentDojoLLM(
                pipeline_name=self.config.defense_model_alias or self.config.attack_model_alias,
                event_loop=event_loop,
                server_client=self.server_client,
                model_server_name=self.config.model_server.name,
                model_url_path=self.url_path_for_run("/v1/chat/completions", body),
                request_options=_request_options(body.responses_create_params),
                system_role=self.config.model_system_role,
            )

            try:
                rollout = asyncio.to_thread(
                    self._run_agentdojo,
                    body,
                    bridge,
                    benchmark_version,
                )
                timeout = self.config.rollout_timeout_seconds
                utility, security = await (asyncio.wait_for(rollout, timeout) if timeout else rollout)
            except TimeoutError:
                return self._abandon(body, benchmark_version, bridge)
            except Exception as exc:
                LOG.exception(
                    "AgentDojo rollout failed: suite=%s user_task=%s injection_task=%s",
                    body.suite,
                    body.user_task_id,
                    body.injection_task_id,
                )
                return self._failure(body, benchmark_version, f"{type(exc).__name__}: {exc}", bridge)

            response = (
                bridge.responses[-1].model_copy(deep=True)
                if bridge.responses
                else NeMoGymResponse.model_validate(_empty_response())
            )
            transcript_output = agentdojo_messages_to_response_output(bridge.last_messages)
            if not transcript_output and bridge.responses:
                transcript_output = [item for model_response in bridge.responses for item in model_response.output]
            response.output = transcript_output
            response.usage = bridge.accumulated_usage()
            reward_utility = float(utility)
            reward_security = float(security)
            generated_turns = sum(1 for message in bridge.last_messages if message["role"] == "assistant")
            result_data = body.model_dump()
            result_data["benchmark_version"] = benchmark_version
            return AgentDojoFamilyVerifyResponse(
                **result_data,
                response=response,
                reward=reward_utility * reward_security,
                utility=utility,
                security=security,
                attack_success=not security,
                reward_utility=reward_utility,
                reward_security=reward_security,
                model_call_count=max(len(bridge.responses), generated_turns),
                tool_filter_kept_tools=bridge.tool_filter_kept_tools,
            )

    def _run_agentdojo(
        self,
        body: AgentDojoFamilyRunRequest,
        bridge: NeMoGymAgentDojoLLM,
        benchmark_version: str,
    ) -> tuple[bool, bool]:
        suite = get_suite(benchmark_version, body.suite)
        user_task = suite.get_user_task_by_id(body.user_task_id)
        pipeline_config: dict[str, Any] = {
            "llm": bridge,
            "model_id": None,
            "defense": body.defense,
            "system_message_name": self.config.system_message_name,
            "system_message": self.config.system_message,
            "tool_output_format": self.config.tool_output_format,
        }
        if "suite_name" in PipelineConfig.model_fields:
            pipeline_config["suite_name"] = body.suite
        pipeline = AgentPipeline.from_config(PipelineConfig(**pipeline_config))
        original_query = pipeline.query

        def recording_query(*args, **kwargs):
            result = original_query(*args, **kwargs)
            bridge.last_messages = result[3]
            return result

        pipeline.query = recording_query
        for element in pipeline.elements:
            if isinstance(element, OpenAILLMToolFilter):
                _record_tool_filter_selection(element, bridge)
        if body.injection_task_id is None:
            utility, _ = suite.run_task_with_pipeline(pipeline, user_task, injection_task=None, injections={})
            return utility, True

        injection_task = suite.get_injection_task_by_id(body.injection_task_id)
        pipeline_name = pipeline.name
        pipeline.name = self.config.attack_model_alias
        try:
            attack = load_attack(body.attack or "", suite, pipeline)
        finally:
            pipeline.name = pipeline_name
        injections = attack.attack(user_task, injection_task)
        utility, upstream_attack_success = suite.run_task_with_pipeline(
            pipeline,
            user_task,
            injection_task,
            injections,
        )
        # AgentDojo calls this second boolean `security`, but BaseInjectionTask.security()
        # returns True when the malicious goal *was achieved*. Expose an unambiguous
        # Gym-facing security verdict and retain attack_success as its complement.
        return utility, not upstream_attack_success

    def _abandon(
        self,
        body: AgentDojoFamilyRunRequest,
        benchmark_version: str,
        bridge: NeMoGymAgentDojoLLM,
    ) -> AgentDojoFamilyVerifyResponse:
        """Give up on a rollout that will not finish, and mask it rather than score it.

        A defense can hand the harness a program that does not terminate -- CaMeL interprets
        model-generated Python and imposes no step or time budget of its own -- and because
        rollouts are serialized, one of those would stall the whole run rather than cost a
        single row. Such a rollout can hold a core at full load indefinitely, hundreds of
        interpreter frames deep, with no sign of failure from outside.

        The row is masked, never scored. Nothing ran to completion, so there is no evidence
        the task succeeded, and calling it secure because no injected action was seen would
        credit the defense for a hang.

        `asyncio.wait_for` cannot cancel a CPU-bound thread, so the abandoned rollout keeps
        burning a core until the process ends. That is why the process ends: after
        `max_abandoned_rollouts` it exits, Gym shuts the stack down, and a rerun with `--resume`
        starts a clean stack with this row already recorded, so the hang is not retried forever.
        The exit is deferred so this response reaches the collector first.
        """
        self._abandoned_rollouts += 1
        timeout = self.config.rollout_timeout_seconds
        LOG.error(
            "Abandoned AgentDojo rollout after %ss (%d in this process): suite=%s user_task=%s injection_task=%s",
            timeout,
            self._abandoned_rollouts,
            body.suite,
            body.user_task_id,
            body.injection_task_id,
        )
        if self._abandoned_rollouts >= self.config.max_abandoned_rollouts:
            LOG.error("Exiting so the abandoned rollout's thread stops holding a core; expect a stack restart.")
            asyncio.get_running_loop().call_later(15, os._exit, 3)
        return self._failure(body, benchmark_version, f"RolloutTimeout: exceeded {timeout}s", bridge)

    def _failure(
        self,
        body: AgentDojoFamilyRunRequest,
        benchmark_version: str,
        error: str,
        bridge: NeMoGymAgentDojoLLM | None = None,
    ) -> AgentDojoFamilyVerifyResponse:
        LOG.warning("Returning masked AgentDojo result: %s", error)
        response = (
            bridge.responses[-1].model_copy(deep=True)
            if bridge is not None and bridge.responses
            else NeMoGymResponse.model_validate(_empty_response())
        )
        result_data = body.model_dump()
        result_data["benchmark_version"] = benchmark_version
        return AgentDojoFamilyVerifyResponse(
            **result_data,
            response=response,
            reward=0.0,
            utility=False,
            security=False,
            attack_success=False,
            reward_utility=0.0,
            reward_security=0.0,
            model_call_count=len(bridge.responses) if bridge is not None else 0,
            tool_filter_kept_tools=bridge.tool_filter_kept_tools if bridge is not None else None,
            adapter_error=error,
            mask_sample=True,
            failure_reason=error,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        prefix = self.config.metric_prefix
        rollouts = [rollout for task in tasks for rollout in task]
        scored = [rollout for rollout in rollouts if not rollout.get("mask_sample", False)]
        if not scored:
            return {f"{prefix}/scored_rollout_count": 0, f"{prefix}/masked_rollout_count": len(rollouts)}
        attacked = [rollout for rollout in scored if rollout.get("injection_task_id") is not None]
        benign = [rollout for rollout in scored if rollout.get("injection_task_id") is None]
        metrics: dict[str, Any] = {
            f"{prefix}/scored_rollout_count": len(scored),
            f"{prefix}/masked_rollout_count": len(rollouts) - len(scored),
            f"{prefix}/benign_utility": sum(float(row.get("utility", False)) for row in benign) / len(benign)
            if benign
            else 0.0,
            f"{prefix}/utility_under_attack": sum(float(row.get("utility", False)) for row in attacked) / len(attacked)
            if attacked
            else 0.0,
            f"{prefix}/attack_success_rate": sum(float(row.get("attack_success", False)) for row in attacked)
            / len(attacked)
            if attacked
            else 0.0,
        }
        filtered = [row for row in scored if row.get("tool_filter_kept_tools") is not None]
        if filtered:
            metrics[f"{prefix}/tool_filter_empty_selection_rate"] = sum(
                1.0 for row in filtered if not row["tool_filter_kept_tools"]
            ) / len(filtered)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        prefix = self.config.metric_prefix
        return {
            key: agent_metrics[key]
            for key in (
                f"{prefix}/benign_utility",
                f"{prefix}/utility_under_attack",
                f"{prefix}/attack_success_rate",
                f"{prefix}/tool_filter_empty_selection_rate",
            )
            if key in agent_metrics
        }
