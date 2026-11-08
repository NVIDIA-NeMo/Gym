# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal inputs over the shared OpenCode sandbox/session lifecycle."""

import asyncio
import copy
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

from fastapi import HTTPException

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    SandboxNetworkAccess,
    SandboxPythonRuntime,
)
from nemo_gym.core import NeMoGymFunctionCallOutput, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseOutputItem
from nemo_gym.sandbox import SandboxProvider, SandboxSpec
from nemo_gym.server_runtime import MCPToolAccess
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession
from responses_api_agents.opencode_multimodal_agent.media import stage_inputs


_NETWORK_RUNTIME = SandboxPythonRuntime(
    harness="opencode-network",
    requirements=(),
    import_check="import asyncio, fcntl, socket",
)


@dataclass
class MultimodalSandboxSession(OpenCodeSandboxSession):
    """Track staged inputs alongside the shared sandbox lifecycle."""

    expected_images: int = 0
    network_checked: bool = False


class MultimodalSessionMixin:
    """Keep model/session supervision in OpenCodeAgent; specialize only visual inputs and output."""

    session_state_type: ClassVar[type[OpenCodeSandboxSession]] = MultimodalSandboxSession

    def _sandbox_network_access(self) -> SandboxNetworkAccess:
        return self.config.network_access

    def _restrict_owned_sandbox_network(
        self, spec: SandboxSpec, provider: SandboxProvider, body: AgentSeedSessionRequest
    ) -> SandboxSpec:
        if provider.name == "opensandbox" and self.config.network_access != "inherit":
            # OpenSandbox applies egress restrictions at creation, before seed.
            # Every activation uses private plugin/cache directories, so a binary
            # alone does not prevent npm or ripgrep downloads during activation.
            required = [
                f"sandbox_{asset}_{field}"
                for asset in ("executable", "plugin_dependencies", "ripgrep")
                for field in ("path", "sha256")
            ]
            missing = [name for name in required if not getattr(self.config, name)]
            if missing:
                raise AgentSeedRejectedError(
                    "Restricted OpenSandbox sessions require a pinned offline runtime bundle before creation; "
                    "provide files in the task image and configure: " + ", ".join(missing)
                )
        if self.config.network_access == "inherit" or provider.name == "opensandbox":
            return super()._restrict_owned_sandbox_network(spec, provider, body)
        if provider.name not in {"enroot", "local"}:
            raise AgentSeedRejectedError("Restricted networking requires a supported sandbox provider")
        if "network_policy" in (spec.provider_options or {}):
            raise AgentSeedRejectedError("Set network_access or provider network_policy, not both")
        if self.config.network_access == "model_only" and any(
            isinstance(access, MCPToolAccess) and access.required for access in self.effective_tool_accesses(body)
        ):
            raise AgentSeedRejectedError("model_only blocks required MCP tool grants; use model_and_tools")
        # Enroot/local use the namespace/proxy boundary at launch. Endpoint grants
        # come from the same effective tool accesses as the shared OpenCode agent.
        return spec

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> OpenCodeSandboxSession:
        if self.config.network_access != "inherit" and body.sandbox_access is not None:
            raise AgentSeedRejectedError("Restricted networking requires an owned sandbox")
        if body.continuation is not None:
            raise AgentSeedRejectedError("Multimodal sessions accept one initial task, not continuation inputs")
        state = await super()._seed_agent_session_state(body)
        try:
            # Owned sessions can narrow an inherit ceiling per request. Install
            # before activation even when this task ultimately inherits networking.
            if state.session.owns_sandbox and not await _NETWORK_RUNTIME.is_installed(state.session.sandbox):
                await _NETWORK_RUNTIME.install(
                    state.session.sandbox, timeout_s=self.config.sandbox_install_timeout_seconds
                )
            if self.config.network_access != "inherit":
                await self._check_session_network(state)
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                # Keep the handle so the shared lifecycle can retry cleanup.
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _check_session_network(self, state: MultimodalSandboxSession) -> None:
        if state.network_checked:
            return
        runner = f"{state.session.session_dir}/network_runner.py"
        await state.session.sandbox.upload(Path(__file__).with_name("network_runner.py"), runner)
        # Run the actual wrapper: this exercises util-linux, all four namespaces,
        # private procfs, loopback setup, and the Unix/TCP proxy listeners. No
        # request is sent to the placeholder origin and no model call is made.
        command = shlex.join(
            [
                _NETWORK_RUNTIME.python,
                "-I",
                runner,
                "--mode",
                "model_only",
                "--model-url",
                "http://127.0.0.1:9",
                "--",
                _NETWORK_RUNTIME.python,
                "-I",
                "-c",
                "from pathlib import Path; assert Path('/proc/self/status').is_file()",
            ]
        )
        result = await state.session.sandbox.exec(command, cwd="/", timeout_s=30)
        self._check_sandbox_setup(command, result)
        state.network_checked = True

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        return await super()._close_agent_session_state(state)

    def _agent_continuation_capabilities(self) -> None:
        return None

    def _validate_sandbox_options(self) -> None:
        if self.config.tool_rewards is not None or self.config.token_id_capture or self._token_id_capture_enabled():
            raise AgentSeedRejectedError(
                "Multimodal sessions are evaluation-only; use the compatibility path for training capture and rewards"
            )
        # The image adapter generates these options itself. Caller-provided plugins,
        # providers and env overrides could escape the capture route or use host paths.
        unsupported = self.config.opencode_config.keys() - {"permission", "tools", "agent"}
        if unsupported:
            raise AgentSeedRejectedError(f"Unsupported multimodal session options: {sorted(unsupported)}")
        if any(self.config.env.values()) or self.config.extra_args or self.config.command != "opencode":
            raise AgentSeedRejectedError("Session execution requires command=opencode and no env/extra_args overrides")
        if self.config.sandbox is not None:
            raise AgentSeedRejectedError(
                "Use sandbox_provider or a borrowed sandbox for sessions; sandbox selects the local Bubblewrap path"
            )

    def _session_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, str]:
        self._network_access_for_request(body)
        self._tools_for_request(body)
        if body.temperature is None or body.top_p is None:
            raise HTTPException(422, "Explicit temperature and top_p are required")
        if body.max_output_tokens is not None and not 0 < body.max_output_tokens <= self.config.context_window:
            raise HTTPException(422, "max_output_tokens must be positive and within context_window")
        # Keep the parent's strict validation of all other request fields and model routing.
        sanitized = type(body).model_validate(
            body.model_dump()
            | {
                "input": [{"role": "user", "content": "Validate multimodal request controls"}],
                "temperature": None,
                "top_p": None,
                "max_output_tokens": None,
                "metadata": {
                    key: value
                    for key, value in (body.metadata or {}).items()
                    if key not in {"network_access", "allowed_tools"}
                }
                or None,
            }
        )
        super()._session_input(sanitized)
        if isinstance(body.input, str):
            raise HTTPException(422, "Multimodal tasks require image inputs")
        texts, systems, images = [], [], 0
        for item in body.model_dump()["input"]:
            role = item.get("role")
            if role not in {"user", "system", "developer"}:
                raise HTTPException(422, "Multimodal sessions accept initial prompts, not replayed trajectories")
            content = item["content"]
            parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
            for part in parts:
                if part["type"] == "input_text":
                    (texts if role == "user" else systems).append(part["text"])
                elif part["type"] == "input_image" and role == "user":
                    images += 1
                else:
                    raise HTTPException(422, "Only user images and text prompts are supported")
        if images == 0 or (self.config.max_images is not None and images > self.config.max_images):
            raise HTTPException(422, "Initial image count is outside the configured limits")
        # Preserve the adapter's user-message prefix and input-system-message semantics.
        prompt = "\n\n".join(text for text in [self.config.system_prompt, *systems, *texts] if text)
        return prompt, body.instructions or ""

    async def _prepare_session_launch(
        self, state: OpenCodeSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, payload: dict[str, Any]
    ) -> None:
        assert isinstance(state, MultimodalSandboxSession)
        network_access = self._network_access_for_request(body)
        allowed_tools = self._tools_for_request(body)
        if network_access != "inherit" and state.request.sandbox_access is not None:
            raise HTTPException(422, "Restricted networking requires an owned sandbox")
        grants = [
            access for access in self.effective_tool_accesses(state.request) if isinstance(access, MCPToolAccess)
        ]
        if network_access == "model_only" and any(access.required for access in grants):
            raise HTTPException(422, "model_only blocks required MCP tool grants")
        if network_access != "inherit":
            # An inherit session may first request restricted execution here.
            await self._check_session_network(state)
        with TemporaryDirectory(prefix="gym-visual-session-") as temporary:
            directory = Path(temporary)
            try:
                _, _, images = await asyncio.to_thread(
                    stage_inputs,
                    body,
                    directory,
                    max_source_pixels=self.config.max_source_pixels,
                    max_images=self.config.max_images,
                    preserve_format=self.config.preserve_image_format,
                )
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            state.expected_images = len(images)
            image_paths = []
            for index, image in enumerate(images):
                path = f"{state.session.session_dir}/{self._input_image_name(image, index)}"
                await state.session.sandbox.upload(image, path)
                image_paths.append(path)
            native_config = copy.deepcopy(self.config.opencode_config)
            self._deep_merge(
                native_config, {"agent": {"build": {"temperature": body.temperature, "top_p": body.top_p}}}
            )
            runner = self.model_copy(
                update={
                    "config": self.config.model_copy(
                        update={
                            "network_access": network_access,
                            "allowed_tools": allowed_tools,
                            "opencode_config": native_config,
                            "max_output_tokens": body.max_output_tokens or self.config.max_output_tokens,
                        }
                    )
                }
            )
            runner._write_opencode_config(directory, state.request.episode_id.capture_key)
            config = json.loads((directory / "opencode.json").read_text())
            inherited = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
            plugins = []
            for plugin in config.get("plugin", []):
                local = directory / Path(plugin).name
                if not local.is_file():
                    raise ValueError(f"Session plugin was not generated by the adapter: {local.name}")
                remote = f"{state.session.session_dir}/{local.name}"
                await state.session.sandbox.upload(local, remote)
                plugins.append(Path(remote).as_uri())
            config["plugin"] = plugins + inherited.get("plugin", [])
            for key in ("mcp", "instructions"):
                if key in inherited:
                    config[key] = inherited[key]
            config.update(
                model=runner._effective_model(),
                small_model=runner._effective_model(),
                enabled_providers=["nemo"],
                autoupdate=False,
                share="disabled",
            )
            payload["env"]["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
            payload["env"].update(runner._web_tool_environment())
            payload["env"]["OPENCODE_ENABLE_EXA"] = "1" if config.get("tools", {}).get("websearch", False) else "0"
            payload["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(runner.config.max_output_tokens)
            payload["env"]["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = str(
                self.config.command_exec_timeout * 1000
            )
            payload["command"].extend(
                ["--model", runner._effective_model(), "--agent", "build", "--file", *image_paths, "--"]
            )
            if network_access != "inherit":
                network_runner = f"{state.session.session_dir}/network_runner.py"
                await state.session.sandbox.upload(Path(__file__).with_name("network_runner.py"), network_runner)
                payload["command"] = [
                    _NETWORK_RUNTIME.python,
                    "-I",
                    network_runner,
                    "--mode",
                    network_access,
                    *(
                        [arg for access in grants for arg in ("--tool-url", str(access.connection.url))]
                        if network_access == "model_and_tools"
                        else []
                    ),
                    "--",
                    *payload["command"],
                ]
            # Keep adapter-owned files out of a borrowed task repository.
            payload["prompt"] = "Image files: " + ", ".join(image_paths) + "\n\n" + payload["prompt"]

    def _session_output(
        self, export: dict[str, Any], observations: AgentObservationBundle
    ) -> list[NeMoGymResponseOutputItem]:
        invocation = next(
            (
                record
                for record in observations.records
                if isinstance(record, AgentInvocation) and record.invocation_id == export.get("session_id")
            ),
            None,
        )
        if invocation is None:
            # The parent preserves partial output and emits a gap when parsing fails.
            return super()._session_output(export, observations)
        output = [
            item.model_copy(deep=True) for item in invocation.conversation if getattr(item, "role", None) != "user"
        ]
        attachments = {}
        for message in export.get("messages", []):
            for part in message.get("parts", []):
                if part.get("type") != "tool":
                    continue
                images = []
                for attachment in part.get("state", {}).get("attachments", []):
                    if not attachment.get("mime", "").startswith("image/"):
                        continue
                    url = attachment.get("url", "")
                    if not url.startswith("data:image/"):
                        raise ValueError("Tool images must be embedded before sandbox cleanup")
                    images.append({"type": "input_image", "image_url": url, "detail": "auto"})
                if images:
                    attachments[part["callID"]] = images
        for item in output:
            if isinstance(item, NeMoGymFunctionCallOutput) and item.call_id in attachments:
                text = item.output if isinstance(item.output, list) else [{"type": "input_text", "text": item.output}]
                existing = {
                    part.get("image_url")
                    for part in text
                    if isinstance(part, dict) and part.get("type") == "input_image"
                }
                item.output = [
                    *text,
                    *(part for part in attachments.pop(item.call_id) if part["image_url"] not in existing),
                ]
        if attachments:
            raise ValueError("Image attachment has no matching tool result")
        return output
