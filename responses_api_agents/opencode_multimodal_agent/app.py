# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal inputs and rewards on Gym's stock OpenCode harness."""

import asyncio
import copy
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from typing import Any
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, Field, model_validator

from nemo_gym.base_responses_api_agent import Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import AgentEpisode, ObservationGap, TrajectoryRecord
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.opencode_agent.app import (
    _INTERNAL_OBSERVATIONS_KEY,
    _INTERNAL_TRAJECTORY_KEY,
    OpenCodeAgent,
    OpenCodeAgentConfig,
    OpenCodeAgentRunRequest,
    OpenCodeAgentVerifyResponse,
    _parse_opencode_session,
)
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_agent.runtime import OBSERVABILITY_PATCH
from responses_api_agents.opencode_multimodal_agent.eval_sandbox import EvalSandboxConfig, sandbox_command
from responses_api_agents.opencode_multimodal_agent.media import final_answer, parse_visual_session, stage_inputs
from responses_api_agents.opencode_multimodal_agent.rewards import (
    OpenCodeToolReward,
    OpenCodeToolRewardConfig,
    score_opencode_tools,
)


ROOT = Path(__file__).parent


class OpenCodeImageConfig(BaseModel):
    """Stock attachment policy for both initial images and tool crops."""

    auto_resize: bool = True
    max_width: int = Field(default=8000, gt=0)
    max_height: int = Field(default=8000, gt=0)
    max_base64_bytes: int = Field(default=1024**3, gt=0)


class OpenCodeMultimodalAgentConfig(OpenCodeAgentConfig):
    model_server: ModelServerRef
    model: str
    # Same published runtime as the parent's example config; installed by its installer.
    opencode_version: str = "1.17.11"
    concurrency: int = Field(default=256, gt=0)
    context_window: int = Field(default=49152, gt=0)
    max_output_tokens: int = Field(default=49152, gt=0)
    max_turns: int = Field(default=8, gt=0)
    # None preserves native stock turn behavior (used by the eval entrypoint).
    final_answer_reserve: int | None = Field(default=2048, gt=0)
    # Send the exact-token reserve extension only to backends that implement it.
    backend_final_answer_reserve: bool = True
    strip_reasoning_for_grading: bool = True
    preserve_image_format: bool = False
    image_policy: OpenCodeImageConfig = Field(default_factory=OpenCodeImageConfig)
    # Required in concurrent production recipes. None supports explicitly trusted,
    # unsandboxed local harness tests; a unique cwd does not isolate /tmp.
    sandbox: EvalSandboxConfig | None = None
    max_source_pixels: int = Field(default=150_000_000, gt=0)
    max_images: int = Field(default=33, gt=0)
    command_exec_timeout: int = Field(default=300, gt=0)
    timeout: int = Field(default=1800, gt=0)
    subagents_enabled: bool = True
    # Parent semantics: this is a prefix in the user message, not a system-role override.
    system_prompt: str | None = (ROOT / "prompts/visual_prefix.txt").read_text().strip()
    tool_rewards: OpenCodeToolRewardConfig | None = Field(default_factory=OpenCodeToolRewardConfig)
    resources_server_by_agent: dict[str, str] = Field(
        default_factory=lambda: {
            "string_match_simple_agent": "string_match",
            "math_with_judge_simple_agent": "math_with_judge",
            "mcqa_simple_agent": "mcqa",
        }
    )

    @model_validator(mode="after")
    def validate_episode_config(self) -> "OpenCodeMultimodalAgentConfig":
        if self.sandbox_provider is not None:
            raise ValueError("Use sandbox to configure multimodal episode isolation")
        if self.final_answer_reserve is not None:
            if self.final_answer_reserve >= self.context_window:
                raise ValueError("final_answer_reserve must be smaller than context_window")
            if self.opencode_version != "1.17.11":
                raise ValueError("Visual final-answer plugin requires OpenCode 1.17.11")
        if self.repo_dir:
            raise ValueError("Multimodal episodes require the parent's fresh workspace, not persistent repo_dir")
        if self.extra_args:
            raise ValueError("Multimodal episodes manage file arguments; extra_args must be empty")
        if any("prompt" in agent for agent in self.opencode_config.get("agent", {}).values()):
            raise ValueError("Keep the runtime default prompt; use system_prompt for a user-message prefix")
        return self


class EpisodeEvidence(BaseModel):
    """Validated native evidence passed explicitly across the responses self-call."""

    turns_used: int
    finished_naturally: bool
    tool_reward: OpenCodeToolReward | None


class OpenCodeMultimodalResponse(NeMoGymResponse):
    opencode_evidence: EpisodeEvidence


class OpenCodeMultimodalRunRequest(OpenCodeAgentRunRequest):
    image_tools_base_agent_ref: dict[str, str] | None = None


class OpenCodeMultimodalVerifyResponse(OpenCodeAgentVerifyResponse):
    resolved: bool
    opencode_tool_reward: OpenCodeToolReward | None = None


class OpenCodeMultimodalAgent(OpenCodeAgent):
    """Inherit the stock installer, provider, environment and subprocess runner.

    Production recipes configure a per-episode filesystem sandbox. Token capture
    and native artifact parsing still use the parent's runner.
    """

    config: OpenCodeMultimodalAgentConfig

    # /run and /v1/responses own image staging and the per-episode sandbox.
    # Separate agent-session setup would bypass that lifecycle.
    _seed_agent_session_state = SimpleResponsesAPIAgent._seed_agent_session_state
    _close_agent_session_state = SimpleResponsesAPIAgent._close_agent_session_state

    def _build_opencode_config(self, rollout_id: str | None = None) -> dict[str, Any]:
        config = super()._build_opencode_config(rollout_id)
        # The parent's correlation plugin uses a host-side file URI. The private
        # filesystem has its own explicitly staged plugin (eval.py); do not load
        # the inaccessible parent path or register both correlation hooks.
        if "plugin" in config:
            config["plugin"] = [item for item in config["plugin"] if item != OBSERVABILITY_PATCH.as_uri()]
            if not config["plugin"]:
                config.pop("plugin")
        model = config["provider"]["nemo"]["models"][self.config.model]
        model.update(
            {
                "attachment": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "tool_call": True,
                "temperature": True,
            }
        )
        self._deep_merge(
            config,
            {
                "agent": {
                    "title": {"disable": True},
                    "build": {
                        "steps": self.config.max_turns,
                        "permission": {
                            "*": "allow",
                            "webfetch": "deny",
                            "websearch": "deny",
                            "skill": "deny",
                            "task": "allow" if self.config.subagents_enabled else "deny",
                        },
                    },
                },
                "attachment": {"image": self.config.image_policy.model_dump()},
                "compaction": {"auto": False, "prune": False},
                "share": "disabled",
            },
        )
        return config

    def _write_opencode_config(self, work_dir: Path, rollout_id: str | None = None) -> None:
        super()._write_opencode_config(work_dir, rollout_id)
        if self.config.final_answer_reserve is None:
            return
        # This version-scoped directive is replaced before both lineage and capture.
        from responses_api_models.vllm_model.app_opencode import OPENCODE_1_17_11_MAX_STEPS

        policy = {
            "max_turns": self.config.max_turns,
            "context_window": self.config.context_window,
            "max_output_tokens": self.config.max_output_tokens,
            "reserve": self.config.final_answer_reserve,
            "backend_reserve": self.config.backend_final_answer_reserve,
            "stock_max_steps": OPENCODE_1_17_11_MAX_STEPS,
            "prompt": (
                "You have reached the tool-use limit. No further tools are available. "
                "Use the images and information already obtained to answer the original question. "
                "Give your best answer now, with the final answer inside \\boxed{}."
            ),
        }
        plugin = work_dir / ".visual-final-answer.mjs"
        plugin.write_text(
            f"const policy = {json.dumps(policy)};\n" + Path(__file__).with_name("final_answer.mjs").read_text()
        )
        config_path = work_dir / "opencode.json"
        config = json.loads(config_path.read_text())
        plugin_path = Path("/testbed") / plugin.name if self.config.sandbox is not None else plugin
        config.setdefault("plugin", []).append(plugin_path.as_uri())
        config_path.write_text(json.dumps(config, indent=2))

    def _env(self, data_home: str, rollout_id: str | None = None) -> dict[str, str]:
        env = super()._env(data_home, rollout_id)
        # Supported stock flags, not a separate runtime/provider.
        env["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = str(self.config.command_exec_timeout * 1000)
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(self.config.max_output_tokens)
        return env

    def _input_image_name(self, source: Path, index: int) -> str:
        return source.name

    def _prepare_command(self, command: list[str], project_dir: Path, env: dict[str, str]) -> list[str]:
        if self.config.sandbox is None:
            return super()._prepare_command(command, project_dir, env)
        # The native runner constructs --file arguments from freshly staged inputs.
        # Place them beside cwd inside the sandbox, and hide every other episode.
        mapped = list(command)
        file_start = mapped.index("--file") + 1
        file_end = mapped.index("--", file_start)
        for index in range(file_start, file_end):
            source = Path(mapped[index])
            filename = self._input_image_name(source, index - file_start)
            destination = project_dir / filename
            shutil.copyfile(source, destination)
            mapped[index] = f"/testbed/{filename}"
        mapped[mapped.index("--dir") + 1] = "/testbed"
        executable_path = shutil.which(mapped[0], path=env.get("PATH"))
        if executable_path is None:
            raise FileNotFoundError(f"OpenCode executable not found: {mapped[0]}")
        executable = Path(executable_path).resolve()
        mapped[0] = "/agent-bin/opencode"
        return sandbox_command(self.config.sandbox, project_dir, executable, env, mapped)

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str | None = None,
        collect_observations: bool = True,
    ) -> AgentEpisode:
        if body.temperature is None or body.top_p is None:
            raise ValueError("Explicit temperature and top_p are required for on-policy generation")
        with TemporaryDirectory(prefix="gym-opencode-multimodal-") as temporary:
            directory = Path(temporary)
            text, input_system, images = await asyncio.to_thread(
                stage_inputs,
                body,
                directory,
                max_source_pixels=self.config.max_source_pixels,
                max_images=self.config.max_images,
                preserve_format=self.config.preserve_image_format,
            )
            system_prompt = "\n\n".join(part for part in [self.config.system_prompt, input_system] if part) or None
            native_config = copy.deepcopy(self.config.opencode_config)
            self._deep_merge(
                native_config,
                {
                    "agent": {
                        "build": {
                            "temperature": body.temperature,
                            "top_p": body.top_p,
                        }
                    }
                },
            )
            # Request-local clone: sampling and files cannot leak between concurrent episodes.
            episode_config = self.config.model_copy(
                update={
                    "opencode_config": native_config,
                    "max_output_tokens": body.max_output_tokens or self.config.max_output_tokens,
                    "extra_args": ["--agent", "build", "--file", *map(str, images), "--"],
                }
            )
            runner = self.model_copy(update={"config": episode_config})
            database = directory / "opencode.db"
            trajectory = TrajectoryRecord(task_id="unscoped", rollout_id=rollout_id or "unscoped")
            await runner._run_opencode(
                text,
                system_prompt,
                rollout_id=rollout_id,
                collect_observations=False,
                artifact_path=database,
            )
            output, usage, turns = await asyncio.to_thread(
                parse_visual_session,
                database,
                expected_images=len(images),
                max_turns=self.config.max_turns,
            )
            evidence = None
            if self.config.tool_rewards is not None:
                evidence = await asyncio.to_thread(
                    score_opencode_tools,
                    database,
                    [item.model_dump() for item in output],
                    [],
                    0.0,
                    self.config.tool_rewards,
                )
            observations = await asyncio.to_thread(
                _parse_opencode_session,
                database,
                rollout_id or "multimodal",
                trajectory,
                model_ref=self.config.model_server,
            )
            if self.config.sandbox is None:
                observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))
            naturally = bool(output and output[-1].type == "message")
            if not any(item.type == "message" for item in output):
                output.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(text="", annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            response = OpenCodeMultimodalResponse(
                id=f"resp_{uuid4().hex}",
                created_at=int(time()),
                model=self.config.model,
                object="response",
                output=output,
                tool_choice=body.tool_choice,
                tools=body.tools,
                parallel_tool_calls=body.parallel_tool_calls,
                usage=NeMoGymResponseUsage(
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    input_tokens_details={"cached_tokens": 0},
                    output_tokens_details={"reasoning_tokens": usage.get("reasoning_tokens", 0)},
                    total_tokens=usage["input_tokens"] + usage["output_tokens"],
                ),
                opencode_evidence=EpisodeEvidence(
                    turns_used=turns, finished_naturally=naturally, tool_reward=evidence
                ),
            )
            if collect_observations:
                response = response.model_copy(update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")})
            return AgentEpisode(response=response, observations=observations)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> OpenCodeMultimodalResponse:
        return OpenCodeMultimodalResponse.model_validate(await super().responses(request, body))

    async def run(self, request: Request, body: OpenCodeMultimodalRunRequest) -> OpenCodeMultimodalVerifyResponse:
        async with self.sem:
            resource = self.config.resources_server.name
            if body.image_tools_base_agent_ref is not None:
                resource = self.config.resources_server_by_agent[body.image_tools_base_agent_ref["name"]]
            seed = await self.server_client.post(
                server_name=resource,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=request.cookies,
            )
            await raise_for_status(seed)
            reply = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=seed.cookies,
            )
            await raise_for_status(reply)
            raw = await get_response_json(reply)
            evidence = EpisodeEvidence.model_validate(raw.pop("opencode_evidence"))
            raw_trajectory = raw.pop(_INTERNAL_TRAJECTORY_KEY, None)
            observations = raw.pop(_INTERNAL_OBSERVATIONS_KEY, None)
            rollout_id = self.rollout_id_from_run(body)
            trajectory = (
                scope_opencode_trajectory(TrajectoryRecord.model_validate(raw_trajectory), body, rollout_id)
                if raw_trajectory is not None and rollout_id is not None
                else None
            )
            scored = copy.deepcopy(raw)
            scored["output"] = [
                NeMoGymResponseOutputMessage(
                    id="final",
                    content=[
                        NeMoGymResponseOutputText(
                            text=final_answer(raw["output"], strip_reasoning=self.config.strip_reasoning_for_grading),
                            annotations=[],
                        )
                    ],
                    type="message",
                    role="assistant",
                    status="completed",
                ).model_dump()
            ]
            verification = await self.server_client.post(
                server_name=resource,
                url_path="/verify",
                cookies=reply.cookies,
                json=body.model_dump() | {"response": scored},
            )
            await raise_for_status(verification)
            verified = await get_response_json(verification)
            base_reward = float(verified["reward"])
            reward = base_reward
            tools = evidence.tool_reward
            if self.config.tool_rewards is not None:
                if tools is None:
                    raise ValueError("Missing native tool evidence for reward shaping")
                tools = tools.for_answer(base_reward, self.config.tool_rewards)
                reward += tools.auxiliary_reward
            return OpenCodeMultimodalVerifyResponse.model_validate(
                verified
                | {
                    "response": raw,
                    "responses_create_params": body.responses_create_params,
                    "reward": reward,
                    "resolved": base_reward == 1.0,
                    "opencode_tool_reward": tools,
                    "turns_used": evidence.turns_used,
                    "finished_naturally": evidence.finished_naturally,
                }
                | ({"ng_agent_observations": observations} if observations is not None else {})
                | ({"ng_trajectory": trajectory.model_dump(mode="json")} if trajectory is not None else {})
            )


if __name__ == "__main__":
    OpenCodeMultimodalAgent.run_webserver()
