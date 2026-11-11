# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual benchmark policies over the shared OpenCode session implementation."""

import asyncio
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml
from fastapi import HTTPException, Request
from pydantic import Field, model_validator

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    Body,
)
from nemo_gym.core import MediaFile, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models.token_capture import OPENCODE_MAX_STEPS_PROMPT as OPENCODE_1_17_11_MAX_STEPS
from nemo_gym.sandbox import SandboxProvider, SandboxSpec, read_text, upload_text
from nemo_gym.trajectory import AgentEpisode, AgentInvocation, ToolCallObservation
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig, _serialize_replacement_config
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession
from responses_api_agents.opencode_multimodal_agent.media import prepare_image
from responses_api_agents.opencode_multimodal_agent.rewards import (
    EpisodeEvidence,
    OpenCodeScoringConfig,
    final_answer,
    score_opencode_tools,
)


ROOT = Path(__file__).parent
DEFAULT_TOOL_POLICY = yaml.safe_load((ROOT.parent / "opencode_agent/configs/permissions/default.yaml").read_text())[
    "opencode_agent"
]["responses_api_agents"]["opencode_agent"]["opencode_config"]


class OpenCodeMultimodalAgentConfig(OpenCodeScoringConfig, OpenCodeAgentConfig):
    max_turns: int | None = Field(default=8, gt=0)
    final_answer_reserve: int | None = Field(default=2048, gt=0)
    backend_final_answer_reserve: bool = True
    max_source_pixels: int = Field(default=150_000_000, gt=0)
    max_images: int | None = Field(default=33, gt=0)
    command_exec_timeout: int = Field(default=300, gt=0)
    subagents_enabled: bool = True
    system_prompt: str | None = (ROOT / "prompts/visual_prefix.txt").read_text().strip()
    compaction_auto: bool = False
    require_finalized_trace: bool = True

    @model_validator(mode="after")
    def validate_final_answer_policy(self) -> "OpenCodeMultimodalAgentConfig":
        if self.final_answer_reserve is not None:
            if self.final_answer_reserve >= self.context_window:
                raise ValueError("final_answer_reserve must be smaller than context_window")
            if self.max_turns is None:
                raise ValueError("A final-answer reserve requires a finite max_turns")
            if self.opencode_version != "1.17.11":
                raise ValueError("Visual final-answer plugin requires OpenCode 1.17.11")
        return self


class OpenCodeMultimodalResponse(NeMoGymResponse):
    opencode_evidence: EpisodeEvidence


@dataclass
class MultimodalSandboxSession(OpenCodeSandboxSession):
    expected_images: int = 0


class OpenCodeMultimodalAgent(OpenCodeAgent):
    """Add image guards, final-answer controls and reward evidence to main's sessions."""

    config: OpenCodeMultimodalAgentConfig
    session_state_type: ClassVar[type[OpenCodeSandboxSession]] = MultimodalSandboxSession

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> OpenCodeMultimodalResponse:
        return OpenCodeMultimodalResponse.model_validate(await super().responses(request, body))

    def _restrict_owned_sandbox_network(
        self, spec: SandboxSpec, provider: SandboxProvider, body: AgentSeedSessionRequest
    ) -> SandboxSpec:
        if provider.name != "opensandbox":
            raise AgentSeedRejectedError("Visual episodes require an owned OpenSandbox")
        return super()._restrict_owned_sandbox_network(spec, provider, body)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> OpenCodeSandboxSession:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("Visual episodes require an owned OpenSandbox")
        return await super()._seed_agent_session_state(body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        return await super()._close_agent_session_state(state)

    def _agent_continuation_capabilities(self) -> None:
        return None

    def _session_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str, list[MediaFile]]:
        # Sampling remains validated by main; only this adapter's per-call token limit differs.
        metadata = body.metadata or {}
        if any(key in metadata for key in ("network_access", "allowed_tools")):
            raise HTTPException(
                422, "Select network/tool profiles through task-source routing before session creation"
            )
        if body.max_output_tokens is not None and not 0 < body.max_output_tokens <= self.config.context_window:
            raise HTTPException(422, "max_output_tokens must be positive and within context_window")
        validated = body.model_copy(update={"max_output_tokens": None})
        prompt, system, media = super()._session_input(validated)
        if self.config.max_images is not None and len(media) > self.config.max_images:
            raise HTTPException(422, "Initial image count is outside the configured limits")
        try:
            media = [
                prepare_image(
                    image,
                    max_source_pixels=self.config.max_source_pixels,
                )
                for image in media
            ]
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return prompt, system, media

    async def _create_episode(self, *args: Any, **kwargs: Any) -> AgentEpisode:
        raise HTTPException(409, "Visual episodes require an Environment Server and an owned OpenSandbox session")

    async def _prepare_session_launch(
        self, state: OpenCodeSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, payload: dict[str, Any]
    ) -> None:
        config = self._deep_merge(
            copy.deepcopy(DEFAULT_TOOL_POLICY), json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        )
        input_items = body.model_dump()["input"]
        state.expected_images = sum(
            part.get("type") == "input_image"
            for item in (input_items if isinstance(input_items, list) else [])
            for part in (item["content"] if not isinstance(item["content"], str) else [])
        )
        model_id = self.config.sandbox_model_id
        provider_id = config["model"].split("/", 1)[0]
        output_limit = body.max_output_tokens or self.config.max_output_tokens
        config["provider"][provider_id]["models"][model_id]["limit"]["output"] = output_limit
        config["compaction"] = {"auto": self.config.compaction_auto, "prune": False}
        if not self.config.subagents_enabled:
            config.setdefault("permission", {})["task"] = "deny"
        if self.config.max_turns is not None:
            config.setdefault("agent", {}).setdefault("build", {})["steps"] = self.config.max_turns
        if self.config.final_answer_reserve is not None:
            policy = {
                "receipt_path": f"{state.session.session_dir}/visual-final-answer.ready",
                "max_turns": self.config.max_turns,
                "context_window": self.config.context_window,
                "max_output_tokens": output_limit,
                "reserve": self.config.final_answer_reserve,
                "backend_reserve": self.config.backend_final_answer_reserve,
                "stock_max_steps": OPENCODE_1_17_11_MAX_STEPS,
                "prompt": "You have reached the tool-use limit. No further tools are available. "
                "Use the images and information already obtained to answer the original question. "
                "Give your best answer now, with the final answer inside \\boxed{}.",
            }
            plugin = f"{state.session.session_dir}/visual-final-answer.mjs"
            await upload_text(
                state.session.sandbox,
                path=plugin,
                text=f"const policy = {json.dumps(policy)};\n" + (ROOT / "final_answer.mjs").read_text(),
            )
            config.setdefault("plugin", []).append(Path(plugin).as_uri())
        payload["env"]["OPENCODE_CONFIG_CONTENT"] = (
            _serialize_replacement_config(config)
            if self.config.system_prompt_mode == "replace"
            else json.dumps(config)
        )
        # Native Exa search is opt-in in stock OpenCode; permissions still govern use.
        payload["env"]["OPENCODE_ENABLE_EXA"] = (
            "1"
            if self.config.network_access == "inherit" and config.get("permission", {}).get("websearch") == "allow"
            else "0"
        )
        payload["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(output_limit)
        payload["env"]["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = str(self.config.command_exec_timeout * 1000)

    async def _session_response(
        self, state: OpenCodeSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, **kwargs: Any
    ) -> NeMoGymResponse:
        response = await super()._session_response(state, body, **kwargs)
        if self.config.final_answer_reserve is not None and state.session.launch_started:
            async with asyncio.timeout(self.config.session_close_timeout_seconds):
                receipt = await read_text(
                    state.session.sandbox, path=f"{state.session.session_dir}/visual-final-answer.ready"
                )
            if receipt != "ready\n":
                raise HTTPException(502, "OpenCode final-answer policy did not load")
        return response

    def _make_session_response(
        self, state: OpenCodeSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, **kwargs: Any
    ) -> OpenCodeMultimodalResponse:
        response = super()._make_session_response(state, body, **kwargs)
        assert isinstance(state, MultimodalSandboxSession)
        export = json.loads(state.session.artifacts) if state.session.artifacts else {}
        messages = export.get("messages", [])
        turns = sum(message.get("info", {}).get("role") == "assistant" for message in messages)
        verification_error = None
        if self.config.require_finalized_trace:
            assistants = [
                message.get("info", {}) for message in messages if message.get("info", {}).get("role") == "assistant"
            ]
            if any(message.get("info", {}).get("error") for message in messages) or any(
                isinstance(record, AgentInvocation) and record.status == "failed"
                for record in (state.observations.records if state.observations else [])
            ):
                verification_error = "OpenCode recorded a model/runtime error; do not score this rollout"
            elif not assistants or (self.config.max_turns is not None and len(assistants) > self.config.max_turns):
                verification_error = "Missing assistant turns or exceeded max_turns"
            elif any(not (message.get("time") or {}).get("completed") for message in assistants):
                verification_error = "Unfinished OpenCode assistant turn"
        if response.status == "completed":
            images = sum(
                part.get("type") == "file" and part.get("mime", "").startswith("image/")
                for message in messages
                if message.get("info", {}).get("role") == "user"
                for part in message.get("parts", [])
            )
            if images != state.expected_images:
                raise HTTPException(502, "OpenCode dropped initial images")
        tools = None
        if self.config.tool_rewards is not None and verification_error is None:
            call_ids = {
                record.native_tool_call_id: record.tool_call_id
                for record in (state.observations.records if state.observations else [])
                if isinstance(record, ToolCallObservation)
                and record.invocation_id == export.get("session_id")
                and record.native_tool_call_id is not None
            }
            wanted = {item.call_id for item in response.output if item.type == "function_call"}
            records = {}
            for message in messages:
                for part in message.get("parts", []):
                    if part.get("type") != "tool":
                        continue
                    native_id = part["callID"]
                    call_id = call_ids.get(native_id, native_id)
                    record_id = call_id if call_id in wanted else native_id
                    if record_id in records:
                        raise ValueError(f"Ambiguous OpenCode tool record: {call_id}")
                    tool_state = part.get("state") or {}
                    records[record_id] = (
                        part["tool"],
                        tool_state.get("status"),
                        (tool_state.get("metadata") or {}).get("exit"),
                    )
            tools = score_opencode_tools(
                records,
                [item.model_dump() for item in response.output],
                0.0,
                self.config.tool_rewards,
            )
        return OpenCodeMultimodalResponse.model_validate(
            response.model_dump()
            | {
                "opencode_evidence": EpisodeEvidence(
                    verification_error=verification_error,
                    turns_used=turns,
                    finished_naturally=response.status == "completed"
                    and bool(final_answer([item.model_dump() for item in response.output])),
                    tool_reward=tools,
                )
            }
        )


if __name__ == "__main__":
    OpenCodeMultimodalAgent.run_webserver()

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeMultimodalAgent.run_webserver()  # noqa: F401
