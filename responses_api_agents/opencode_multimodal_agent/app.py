# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal inputs and rewards on Gym's stock OpenCode harness."""

import asyncio
import copy
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from nemo_gym.agents import Body, SandboxNetworkAccess, resolve_agent_timeout
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseUsage,
)
from nemo_gym.trajectory import AgentEpisode, ObservationGap, TrajectoryRecord
from responses_api_agents.opencode_agent.app import (
    OpenCodeAgent,
    OpenCodeAgentConfig,
    OpenCodeAgentRunRequest,
    OpenCodeAgentVerifyResponse,
)
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_agent.protocol import (
    AGENT_OBSERVATIONS_KEY,
    AGENT_TRAJECTORY_KEY,
    OPENCODE_1_17_11_MAX_STEPS,
)
from responses_api_agents.opencode_agent.runtime import (
    IMAGE_INPUT_MODEL_OPTIONS,
    OBSERVABILITY_PATCH,
    UNRESTRICTED_IMAGE_ATTACHMENT,
)
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession
from responses_api_agents.opencode_multimodal_agent.eval_sandbox import EvalSandboxConfig, sandbox_command
from responses_api_agents.opencode_multimodal_agent.media import final_answer, parse_visual_session, stage_inputs
from responses_api_agents.opencode_multimodal_agent.network_config import control_service_urls, server_origin
from responses_api_agents.opencode_multimodal_agent.network_runner import origin
from responses_api_agents.opencode_multimodal_agent.rewards import (
    OpenCodeToolReward,
    OpenCodeToolRewardConfig,
    score_opencode_tools,
)
from responses_api_agents.opencode_multimodal_agent.sessions import MultimodalSandboxSession, MultimodalSessionMixin


ROOT = Path(__file__).parent
_NETWORK_ALIASES = {
    "all": "inherit",
    "none": "model_only",
}


def normalize_network_access(value: str) -> str:
    return _NETWORK_ALIASES.get(value, value)


class OpenCodeImageConfig(BaseModel):
    """OpenCode attachment limits; defaults preserve initial images and tool crops."""

    auto_resize: bool = True
    max_width: int = Field(default=UNRESTRICTED_IMAGE_ATTACHMENT["image"]["max_width"], gt=0)
    max_height: int = Field(default=UNRESTRICTED_IMAGE_ATTACHMENT["image"]["max_height"], gt=0)
    max_base64_bytes: int = Field(default=UNRESTRICTED_IMAGE_ATTACHMENT["image"]["max_base64_bytes"], gt=0)


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
    preserve_image_format: bool = True
    image_policy: OpenCodeImageConfig = Field(default_factory=OpenCodeImageConfig)
    # Required in concurrent production recipes. None supports explicitly trusted,
    # unsandboxed local harness tests; a unique cwd does not isolate /tmp.
    sandbox: EvalSandboxConfig | None = None
    # Benchmark and training profiles can explicitly restrict the inherited network.
    network_access: SandboxNetworkAccess = "inherit"
    protected_control_urls: list[str] = Field(
        default_factory=list,
        description="Additional control-service origins, including public ingress aliases absent from Gym config.",
    )
    # Optional run allowlist; request metadata may select a subset. None retains native tools.
    allowed_tools: list[str] | None = None
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

    @field_validator("network_access", mode="before")
    @classmethod
    def normalize_network_alias(cls, value: Any) -> Any:
        return normalize_network_access(value) if isinstance(value, str) else value

    @field_validator("protected_control_urls")
    @classmethod
    def validate_control_urls(cls, urls: list[str]) -> list[str]:
        for url in urls:
            origin(url)
        return urls

    @model_validator(mode="after")
    def validate_episode_config(self) -> "OpenCodeMultimodalAgentConfig":
        if self.system_prompt_mode != "append":
            raise ValueError("Use build_system_prompt in the evaluation or combined entrypoint to replace the prompt")
        if self.sandbox_provider is not None and self.sandbox is not None:
            raise ValueError("Choose session sandbox_provider or local sandbox, not both")
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
            raise ValueError("Keep the runtime default prompt; use system_prompt for additional instructions")
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


class OpenCodeMultimodalAgent(MultimodalSessionMixin, OpenCodeAgent):
    """Inherit the stock installer, provider, environment and subprocess runner.

    Production recipes configure a per-episode filesystem sandbox. Token capture
    and native artifact parsing still use the parent's runner.
    """

    config: OpenCodeMultimodalAgentConfig

    def _network_access_for_request(self, body: NeMoGymResponseCreateParamsNonStreaming) -> SandboxNetworkAccess:
        """Select a dataset policy without exceeding the run's configured access."""
        selected = (body.metadata or {}).get("network_access", self.config.network_access)
        selected = normalize_network_access(selected)
        levels = {"model_only": 0, "model_and_tools": 1, "inherit": 2}
        if selected not in levels:
            raise HTTPException(422, "metadata.network_access must be model_only, model_and_tools or inherit")
        if levels[selected] > levels[self.config.network_access]:
            raise HTTPException(422, "Sample network_access exceeds the configured run policy")
        return selected

    def _tools_for_request(self, body: NeMoGymResponseCreateParamsNonStreaming) -> list[str] | None:
        raw = (body.metadata or {}).get("allowed_tools")
        if raw is None:
            selected = self.config.allowed_tools
        else:
            try:
                selected = json.loads(raw)
            except (TypeError, ValueError) as error:
                raise HTTPException(422, "metadata.allowed_tools must be a JSON-encoded list of tool names") from error
        if selected is None and raw is None:
            return None
        supported = {"bash", "edit", "glob", "grep", "read", "task", "todowrite", "write", "websearch", "webfetch"}
        if (
            not isinstance(selected, list)
            or any(not isinstance(tool, str) or tool not in supported for tool in selected)
            or len(set(selected)) != len(selected)
        ):
            raise HTTPException(422, "allowed_tools must contain unique supported tool names")
        # OpenCode uses the edit permission for both editing and writing files.
        if ("edit" in selected) != ("write" in selected):
            raise HTTPException(422, "OpenCode requires edit and write to be selected together")
        if self.config.allowed_tools is not None and not set(selected) <= set(self.config.allowed_tools):
            raise HTTPException(422, "Sample allowed_tools exceeds the configured run allowlist")
        configured = self.config.opencode_config.get("tools", {})
        permissions = self.config.opencode_config.get("permission", {})
        build_permissions = self.config.opencode_config.get("agent", {}).get("build", {}).get("permission", {})
        if any(
            configured.get(tool) is False or permissions.get(tool) == "deny" or build_permissions.get(tool) == "deny"
            for tool in selected
        ):
            raise HTTPException(422, "Sample requests a tool disabled in the run configuration")
        mode = self._network_access_for_request(body)
        if mode != "inherit" and {"webfetch", "websearch"}.intersection(selected):
            raise HTTPException(422, "Sample tools exceed its network policy")
        if "task" in selected and not self.config.subagents_enabled:
            raise HTTPException(422, "Subagents are disabled in the run configuration")
        return list(selected)

    def _apply_tool_allowlist(self, config: dict[str, Any]) -> None:
        if self.config.allowed_tools is None:
            return
        selected = set(self.config.allowed_tools)
        known = selected | {
            "bash",
            "edit",
            "glob",
            "grep",
            "read",
            "task",
            "todowrite",
            "write",
            "websearch",
            "webfetch",
            "skill",
            "codesearch",
            "question",
            "lsp",
            "plan_enter",
            "plan_exit",
            "apply_patch",
        }
        # apply_patch shares the edit permission and replaces edit/write for GPT models.
        if "edit" in selected:
            selected.add("apply_patch")
        config.setdefault("tools", {}).update({name: name in selected for name in sorted(known)})
        top = config.setdefault("permission", {})
        permissions = [top, *(agent.setdefault("permission", {}) for agent in config.get("agent", {}).values())]
        for permission in permissions:
            # An agent's wildcard also matches non-tool permissions such as
            # external_directory. Preserve inherited rules before denying the rest.
            original = {**top, **permission}
            permission.clear()
            permission.update({"*": "deny", **{key: value for key, value in original.items() if key != "*"}})
            for name in sorted(known):
                # Retain command/path restrictions when advertising an enabled tool.
                prior = original.get(name, top.get(name))
                if name not in selected or (prior == "deny" and name not in {"webfetch", "websearch"}):
                    permission[name] = "deny"
                else:
                    permission[name] = prior if isinstance(prior, dict) else "allow"

    def _make_session_response(
        self, state: OpenCodeSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, **kwargs: Any
    ) -> OpenCodeMultimodalResponse:
        response = super()._make_session_response(state, body, **kwargs)
        assert isinstance(state, MultimodalSandboxSession)
        export = json.loads(state.session.artifacts) if state.session.artifacts else {}
        messages = export.get("messages", [])
        turns = sum(message.get("info", {}).get("role") == "assistant" for message in messages)
        if response.status == "completed":
            images = sum(
                part.get("type") == "file" and part.get("mime", "").startswith("image/")
                for message in messages
                if message.get("info", {}).get("role") == "user"
                for part in message.get("parts", [])
            )
            if images != state.expected_images:
                raise HTTPException(502, "OpenCode dropped initial images")
        return OpenCodeMultimodalResponse.model_validate(
            response.model_dump()
            | {
                "opencode_evidence": EpisodeEvidence(
                    turns_used=turns,
                    finished_naturally=response.status == "completed"
                    and bool(final_answer([item.model_dump() for item in response.output])),
                    tool_reward=None,
                )
            }
        )

    def _request_config(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        network_access: SandboxNetworkAccess,
        allowed_tools: list[str] | None,
    ) -> OpenCodeMultimodalAgentConfig:
        """Apply per-request controls to a copy, keeping concurrent episodes independent."""
        native_config = copy.deepcopy(self.config.opencode_config)
        self._deep_merge(native_config, {"agent": {"build": {"temperature": body.temperature, "top_p": body.top_p}}})
        return self.config.model_copy(
            update={
                "network_access": network_access,
                "allowed_tools": allowed_tools,
                "opencode_config": native_config,
                "max_output_tokens": body.max_output_tokens or self.config.max_output_tokens,
            }
        )

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
                **copy.deepcopy(IMAGE_INPUT_MODEL_OPTIONS),
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
        if self.config.network_access != "inherit":
            path = work_dir / "opencode.json"
            config = json.loads(path.read_text())
            config.setdefault("tools", {}).update(webfetch=False, websearch=False)
            permissions = [config.setdefault("permission", {})]
            permissions.extend(agent.setdefault("permission", {}) for agent in config.get("agent", {}).values())
            for permission in permissions:
                permission.pop("*", None)
                permission.update(
                    webfetch="deny",
                    websearch="deny",
                )
            path.write_text(json.dumps(config, indent=2))
        if self.config.allowed_tools is not None:
            path = work_dir / "opencode.json"
            config = json.loads(path.read_text())
            self._apply_tool_allowlist(config)
            path.write_text(json.dumps(config, indent=2))
        if self.config.final_answer_reserve is None:
            return
        # This version-scoped directive is replaced before both lineage and capture.

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
        env.update(self._web_tool_environment())
        if self.config.network_access != "inherit":
            env["OPENCODE_ENABLE_EXA"] = "0"
        if self.config.network_access == "inherit" and self.config.allowed_tools is not None:
            env["OPENCODE_ENABLE_EXA"] = "1" if "websearch" in self.config.allowed_tools else "0"
        # Supported stock flags, not a separate runtime/provider.
        env["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = str(self.config.command_exec_timeout * 1000)
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(self.config.max_output_tokens)
        return env

    def _web_tool_environment(self) -> dict[str, str]:
        return {"OPENCODE_ENABLE_EXA": "0"}

    def _protected_control_urls(self) -> list[str]:
        return sorted(
            set(
                control_service_urls(self.server_client.global_config_dict, model_server=self.config.model_server.name)
            )
            | set(self.config.protected_control_urls)
            | {server_origin(self.config.host, self.config.port)}
        )

    def _input_image_name(self, source: Path, index: int) -> str:
        return source.name

    def _prepare_command(self, command: list[str], project_dir: Path, env: dict[str, str]) -> list[str]:
        if self.config.sandbox is None:
            if self.config.network_access != "inherit":
                raise ValueError("Restricted local networking requires the filesystem sandbox")
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
        launch = sandbox_command(self.config.sandbox, project_dir, executable, env, mapped)
        config = json.loads((project_dir / "opencode.json").read_text())
        return [
            sys.executable,
            str(Path(__file__).with_name("network_runner.py")),
            "--mode",
            self.config.network_access,
            "--model-url",
            config["provider"]["nemo"]["options"]["baseURL"],
            *[arg for url in self._protected_control_urls() for arg in ("--protect-url", url)],
            "--bubblewrap",
            "--",
            *launch,
        ]

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str | None = None,
        collect_observations: bool = True,
    ) -> AgentEpisode:
        network_access = self._network_access_for_request(body)
        allowed_tools = self._tools_for_request(body)
        if self.config.sandbox_provider is not None:
            raise HTTPException(409, "Configured session sandbox requires an Environment Server and agent session")
        if body.temperature is None or body.top_p is None:
            raise ValueError("Explicit temperature and top_p are required for on-policy generation")
        await self._ensure_local_runtime()
        with TemporaryDirectory(prefix="gym-opencode-multimodal-") as temporary:
            directory = Path(temporary)
            try:
                text, input_system, images = await asyncio.to_thread(
                    stage_inputs,
                    body,
                    directory,
                    max_source_pixels=self.config.max_source_pixels,
                    max_images=self.config.max_images,
                    preserve_format=self.config.preserve_image_format,
                )
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            system_prompt = "\n\n".join(part for part in [self.config.system_prompt, input_system] if part) or None
            episode_config = self._request_config(body, network_access=network_access, allowed_tools=allowed_tools)
            episode_config.extra_args = ["--agent", "build", "--file", *map(str, images), "--"]
            runner = self.model_copy(update={"config": episode_config})
            database = directory / "opencode.db"
            trajectory = TrajectoryRecord(task_id="unscoped", rollout_id=rollout_id or "unscoped")
            timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
            await runner._run_opencode(
                text,
                system_prompt,
                rollout_id=rollout_id,
                collect_observations=False,
                artifact_path=database,
                timeout=timeout,
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
                parse_opencode_observations,
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
                metadata={"agent_timeout_sec": str(timeout)},
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
                response = response.model_copy(update={AGENT_TRAJECTORY_KEY: trajectory.model_dump(mode="json")})
            return AgentEpisode(response=response, observations=observations)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> OpenCodeMultimodalResponse:
        return OpenCodeMultimodalResponse.model_validate(await super().responses(request, body))

    async def run(self, request: Request, body: OpenCodeMultimodalRunRequest) -> OpenCodeMultimodalVerifyResponse:
        if self._agent_session_id_from_request(request) is not None:
            raise HTTPException(409, "Agent sessions must run through the Environment Server")
        if self.config.resources_server is None:
            raise HTTPException(422, "The /run endpoint requires resources_server")
        if body.image_tools_base_agent_ref is not None:
            name = body.image_tools_base_agent_ref.get("name")
            if name not in self.config.resources_server_by_agent:
                raise HTTPException(422, "Unknown image_tools_base_agent_ref")
        self._network_access_for_request(body.responses_create_params)
        self._tools_for_request(body.responses_create_params)
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
            raw_trajectory = raw.pop(AGENT_TRAJECTORY_KEY, None)
            observations = raw.pop(AGENT_OBSERVATIONS_KEY, None)
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
