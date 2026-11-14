# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional media setup around stock OpenCode's sandboxed agent loop."""

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path, PurePosixPath
from shlex import quote
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, field_serializer, model_validator

from nemo_gym.base_responses_api_agent import Body
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymFunctionCallOutput, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
    OpenCodeSandboxedAgentVerifyResponse,
)


RUNTIME = PurePosixPath("/tmp/nemo-gym-video/video_runtime")
LAUNCHER = RUNTIME / "launcher.json"
# Response metadata the video verifiers read (resources_servers/video_mcqa/scoring.py). A failed run is graded on
# an emptied copy while its row keeps the original response, so replaying that row reads the failure from here.
EXECUTION_FAILURE_METADATA_KEY = "video_execution_failure"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class VideoBinding(BaseModel):
    """Trusted dataset transport only: never forwarded to the policy model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: FilePath
    sandbox_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_serializer("source")
    def serialize_source(self, value: Path) -> str:
        # Stock sandbox seed transport calls model_dump() in Python mode.
        return str(value)

    @model_validator(mode="after")
    def relative_path(self):
        path = PurePosixPath(self.sandbox_path)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or any(char in self.sandbox_path for char in "\x00\r\n")
        ):
            raise ValueError("sandbox video binding must be a safe workspace-relative path")
        return self


class OpenCodeVideoAgentConfig(OpenCodeSandboxedAgentConfig):
    media_runtime_archive: FilePath
    media_runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allowed_video_roots: list[DirectoryPath] = Field(min_length=1)
    media_profile: dict[str, Any]
    max_output_tokens: int = Field(gt=0)
    capture_model_inputs: bool = False
    # "<provider>/<model>" the OpenCode config names the policy model with; OpenCode's system prompt
    # quotes it. None keeps the stock "nemo_gym/dummy_model". Model settings in opencode_config stay
    # under provider.nemo_gym.models.dummy_model and move with the model.
    opencode_model: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    # Absolute sandbox directory that holds the bound videos and is the media tools' workspace; it
    # contains the workdir (OpenCode's working directory). None is the workdir.
    media_root: str | None = None
    # Seconds the media launcher allows one visual tool call.
    media_tool_timeout_seconds: float = Field(default=180, gt=0)

    @model_validator(mode="after")
    def aligned_runtime(self):
        if self.opencode_version != "1.17.11" or not self.preinstalled_opencode:
            raise ValueError("Video runtime requires prepackaged stock OpenCode 1.17.11")
        if self.media_profile.get("max_context_tokens") != self.opencode_max_context_window:
            raise ValueError("Media profile and OpenCode context differ")
        if self.media_profile.get("reserved_output_tokens") != self.max_output_tokens:
            raise ValueError("Media profile and OpenCode output differ")
        if self.max_output_tokens >= self.opencode_max_context_window:
            raise ValueError("Output reserve must be smaller than the context")
        workdir = self.sandbox_config.get("workdir", "")
        if not PurePosixPath(workdir).is_absolute() or workdir == "/":
            raise ValueError("A dedicated absolute sandbox workdir is required")
        if self.media_root is not None:
            root = PurePosixPath(self.media_root)
            if not root.is_absolute() or root == PurePosixPath("/") or ".." in root.parts:
                raise ValueError("media_root must be a dedicated absolute sandbox directory")
            if not PurePosixPath(workdir).is_relative_to(root):
                raise ValueError("media_root must contain the sandbox workdir")
        if not self.artifacts_dir or not Path(self.artifacts_dir).is_absolute():
            raise ValueError("A dedicated absolute host artifacts_dir is required")
        # Copy before deriving environment settings; never mutate caller/global config.
        self.sandbox_config = deepcopy(self.sandbox_config)
        env = self.sandbox_config.setdefault("env", {})
        env["NEMOTRON_OPENCODE_MEDIA_LAUNCHER"] = str(LAUNCHER)
        env["NEMOTRON_OPENCODE_DEPENDENCY_ROOT"] = str(RUNTIME / "node_modules")
        env["PYTHONPATH"] = f"{RUNTIME}/packages:{RUNTIME}/site-packages"
        env["PATH"] = (
            f"{RUNTIME}/node_modules/.bin:{RUNTIME}/node/bin:{RUNTIME}/bin:{RUNTIME}/python/bin:/usr/local/bin:/usr/bin:/bin"
        )
        return self


class OpenCodeVideoRunRequest(OpenCodeSandboxedAgentRunRequest):
    setting: Literal["agentic"]
    video_binding: VideoBinding


def video_config(config: dict, *, output_tokens: int, model: str | None = None) -> dict:
    """Add only the validated media provider/plugin; do not replace native prompts/compact.

    The model's limit is its context and OUTPUT_TOKENS. MODEL ("<provider>/<model>") renames the
    stock provider and model, keeping their settings.
    """
    result = deepcopy(config)
    provider = result["provider"]["nemo_gym"]
    module = RUNTIME / "gym/responses_api_agents/opencode_agent"
    provider["npm"] = f"file://{module}/nemotron_media_provider.mjs"
    entry = provider["models"]["dummy_model"]
    entry.update(attachment=True, modalities={"input": ["text", "image", "video"], "output": ["text"]})
    entry["limit"] = {"context": entry["limit"]["context"], "output": output_tokens}
    if model is not None:
        provider_id, model_id = model.split("/")
        del result["provider"]["nemo_gym"]
        provider["models"] = {model_id: {**entry, "name": model_id}}
        result["provider"][provider_id] = provider
        result["model"] = model
    result.setdefault("plugin", []).append(f"file://{module}/nemotron_media_plugin.mjs")
    return result


def tool_output(state: dict, tool_name: str = "watch_video") -> str | list[dict]:
    """An observational export, not a reconstruction of post-compact model input."""
    if (state.get("time") or {}).get("compacted") is not None:
        return "[Old tool result content cleared]"
    text = state.get("output") if state.get("output") is not None else state.get("error", "")
    parts = []
    if state.get("status") == "completed":
        for attachment in state.get("attachments") or []:
            mime, url = attachment.get("mime", ""), attachment.get("url")
            if not isinstance(url, str) or not url.startswith("data:"):
                raise ValueError("Video tool export has an invalid media attachment")
            if mime == "video/mp4":
                parts.append({"type": "input_video", "video_url": url})
            elif mime.startswith("image/"):
                parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
            else:
                if tool_name in {"watch_video", "view_image", "view_video"}:
                    raise ValueError("Unsupported visual tool attachment MIME")
    # As media_bridge.mjs replays it: only view_image drops its text beside the image.
    if parts and text and tool_name != "view_image":
        parts.insert(0, {"type": "input_text", "text": str(text)})
    return parts if parts else str(text)


class OpenCodeVideoAgent(OpenCodeSandboxedAgent):
    config: OpenCodeVideoAgentConfig

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Keep the verifier's fixed-cohort metrics on the standard Gym route."""
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def _create_opencode_config(self, request: Request, *, image_input: bool = False) -> dict:
        return video_config(
            await super()._create_opencode_config(request, image_input=image_input),
            output_tokens=self.config.max_output_tokens,
            model=self.config.opencode_model,
        )

    def _opencode_export_to_output_items(self, export):
        output = super()._opencode_export_to_output_items(export)
        states = {}
        for message in export["messages"]:
            for part in message["parts"]:
                if part.get("type") == "tool":
                    if part["callID"] in states:
                        raise ValueError("Duplicate native tool call ID")
                    states[part["callID"]] = (part["state"], part["tool"])
        for index, item in enumerate(output):
            if item.type == "function_call_output" and item.call_id in states:
                state, name = states[item.call_id]
                output[index] = NeMoGymFunctionCallOutput(call_id=item.call_id, output=tool_output(state, name))
        return output

    async def _run(self, request: Request, body: OpenCodeVideoRunRequest) -> OpenCodeSandboxedAgentVerifyResponse:
        source = body.video_binding.source.resolve(strict=True)
        if not any(source.is_relative_to(root.resolve()) for root in self.config.allowed_video_roots):
            raise ValueError("Video source is outside the trusted dataset roots")
        if await asyncio.to_thread(digest, source) != body.video_binding.sha256:
            raise ValueError("Frozen source video bytes changed")
        if await asyncio.to_thread(digest, self.config.media_runtime_archive) != self.config.media_runtime_sha256:
            raise ValueError("Pinned video runtime bytes changed")
        request.state._ng_video_binding = body.video_binding
        try:
            return await super()._run(request, body)
        finally:
            del request.state._ng_video_binding

    async def run(self, request: Request, body: OpenCodeVideoRunRequest) -> OpenCodeSandboxedAgentVerifyResponse:
        # Keep the parent's concurrency and complete lifecycle; only the input schema differs.
        return await super().run(request, body)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        binding = getattr(request.state, "_ng_video_binding", None)
        if binding is None:
            raise ValueError("Video media requires a seeded /run request")
        sandbox = self._sandbox_id_to_sandbox[request.cookies["sandbox_id"]]
        workdir = PurePosixPath(self.config.sandbox_config["workdir"])
        root = PurePosixPath(self.config.media_root) if self.config.media_root else workdir
        destination = root / binding.sandbox_path
        # From the root: the image need not contain the workdir, which this creates.
        directories = " ".join(quote(str(path)) for path in (RUNTIME.parent, destination.parent, workdir))
        prepared = await sandbox.exec(
            command=f"umask 077 && mkdir -p {directories}",
            cwd="/",
            timeout_s=30,
        )
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError("Cannot prepare video sandbox directories")
        archive = RUNTIME.parent / "bundle.tar.gz"
        await sandbox.upload(self.config.media_runtime_archive, str(archive))
        unpacked = await sandbox.exec(
            command=f"printf '%s  %s\\n' {quote(self.config.media_runtime_sha256)} {quote(str(archive))} "
            f"| sha256sum --check --status && tar -xzf {quote(str(archive))} --no-same-owner -C {quote(str(RUNTIME.parent))}",
            timeout_s=180,
        )
        if unpacked.return_code != 0 or unpacked.error_type:
            raise RuntimeError("Cannot unpack pinned video runtime")
        await sandbox.upload(binding.source, str(destination))
        readonly = await sandbox.exec(command=f"chmod 0444 {quote(str(destination))}", timeout_s=30)
        if readonly.return_code != 0 or readonly.error_type:
            raise RuntimeError("Cannot mark sandbox video read-only")
        checked = await sandbox.exec(command=f"sha256sum {quote(str(destination))}", timeout_s=60)
        if checked.return_code != 0 or checked.error_type or not checked.stdout.startswith(binding.sha256 + " "):
            raise RuntimeError("Uploaded source video checksum mismatch")
        media = {
            "workspace_root": str(root),
            "artifact_root": str(RUNTIME / "archives"),
            "profile": self.config.media_profile,
        }
        launcher = {
            "command": [str(RUNTIME / "python/bin/python3.13"), "-m", "nemo_rl.environments.video_media_cli"],
            "config_path": str(RUNTIME / "media.json"),
            "dependency_root": str(RUNTIME),
            "timeout_seconds": self.config.media_tool_timeout_seconds,
            "max_output_bytes": 258 << 20,
            "seed_parts": [],
            "context_namespace": "opencode_" + uuid4().hex,
            "context_window": self.config.opencode_max_context_window,
            "max_output_tokens": self.config.max_output_tokens,
            "capture_model_inputs": self.config.capture_model_inputs,
        }
        with TemporaryDirectory(prefix="gym-video-launch-") as temporary:
            for name, payload in (("media.json", media), ("launcher.json", launcher)):
                local = Path(temporary) / name
                local.write_text(json.dumps(payload))
                await sandbox.upload(local, str(RUNTIME / name))
        module = RUNTIME / "gym/responses_api_agents/opencode_agent"
        validate_python = (
            "from pathlib import Path; from nemo_rl.environments.video_media_cli import MediaToolConfig; "
            f"MediaToolConfig.model_validate_json(Path({str(RUNTIME / 'media.json')!r}).read_bytes())"
        )
        validate_js = (
            f"await import('file://{module}/nemotron_media_provider.mjs'); "
            f"const p = await import('file://{module}/nemotron_media_plugin.mjs'); "
            "const hooks = await p.default(); if (!hooks.tool.watch_video || !hooks.tool.view_image) throw Error('missing visual tools');"
        )
        validated = await sandbox.exec(
            command=f"{quote(str(RUNTIME / 'python/bin/python3.13'))} -c {quote(validate_python)} "
            f"&& {quote(str(RUNTIME / 'node/bin/node'))} --input-type=module -e {quote(validate_js)}",
            timeout_s=120,
        )
        if validated.return_code != 0 or validated.error_type:
            raise RuntimeError("Pinned visual runtime could not initialize before model execution")
        response = await super().responses(request, body)
        if self.config.capture_model_inputs:
            evidence = "/tmp/nemo-gym-video-evidence.tar.gz"
            packed = await sandbox.exec(
                command=f"tar -czf {quote(evidence)} -C {quote(str(RUNTIME))} archives", timeout_s=180
            )
            if packed.return_code != 0 or packed.error_type:
                raise RuntimeError("Cannot preserve requested private media evidence")
            output = Path(self.config.artifacts_dir) / request.session[SESSION_ID_KEY] / "video-media-evidence.tar.gz"
            await sandbox.download(evidence, output)
        # A changed or missing source is a completed execution failure, never a valid answer.
        checked = await sandbox.exec(command=f"sha256sum {quote(str(destination))}", timeout_s=60)
        result = self._sandbox_id_to_run_result[request.session[SESSION_ID_KEY]]
        if checked.return_code != 0 or not checked.stdout.startswith(binding.sha256 + " "):
            result.update(opencode_failed=True, opencode_error_type="video_source_modified")
        if self.config.execution_failure_reward_zero and result.get("opencode_failed"):
            metadata = {**(response.metadata or {}), EXECUTION_FAILURE_METADATA_KEY: "true"}
            response = response.model_copy(update={"metadata": metadata})
        return response


if __name__ == "__main__":
    OpenCodeVideoAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeVideoAgent.run_webserver()
