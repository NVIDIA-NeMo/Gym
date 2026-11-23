# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-owned native media and Python state shared by evaluated harnesses.

The pinned external runtime provides initialization and execution. This module
imports that implementation; it does not vendor or reproduce native source.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import Request


class _ResourceKernelPool:
    """Retain the manager before startup so failed setup always has a close handle."""

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel

    async def acquire(self) -> Any:
        if not self.kernel.is_running:
            await self.kernel.start()
        return self.kernel

    async def release(self, kernel: Any) -> None:
        # One Resources session owns this manager until the episode closes.
        pass

    async def shutdown_all(self) -> None:
        await self.kernel.shutdown()


@dataclass
class SpatialClawSessionRuntime:
    """A single task's native kernel, retained until Resources closes it."""

    session_id: str
    sample_id: str
    workspace: Path
    workflow: Any = None
    kernel: Any = None
    spatial_config: Any = None
    context: dict[str, Any] | None = None
    accepted_answer: str | None = None
    execution_count: int = 0
    verified: bool = False

    @classmethod
    def new(cls, session_id: str, sample_id: str, workspace_root: str) -> SpatialClawSessionRuntime:
        token = hashlib.sha256(session_id.encode()).hexdigest()
        return cls(session_id, sample_id, Path(workspace_root).expanduser().resolve() / token)

    async def initialize(
        self, server: Any, sample: Any, benchmark: Any, capture_key: str, *, media: dict[str, str] | None = None
    ) -> None:
        # Reuse Gym's native preparation path rather than deriving frame sampling
        # or tool/object injection independently for the second harness.
        from benchmarks.spatialclaw.prepare import _instruction, _sample_metadata
        from nemo_gym.config import ResourcesServerRef
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.spatialclaw_agent.app import (
            SpatialClawAgent,
            SpatialClawAgentConfig,
            _install_gym_llm_transport,
        )

        cfg = server.config
        native = SpatialClawAgent(
            config=SpatialClawAgentConfig(
                host=cfg.host,
                port=cfg.port,
                name=f"{cfg.name}.task-runtime",
                entrypoint="app.py",
                resources_server=ResourcesServerRef(type="resources_servers", name=cfg.name),
                model_server=cfg.model_server,
                spatialclaw_url=cfg.spatialclaw_url,
                spatialclaw_commit=cfg.spatialclaw_commit,
                spatialclaw_root=cfg.spatialclaw_root,
                source_cache_root=cfg.source_cache_root,
                dataset_config=cfg.dataset_config,
                model_config_path=cfg.model_config_path,
                model_name=cfg.model_name,
                video_root=cfg.data_root or str(server._source_root / "data"),
                gpu_server_registry=cfg.gpu_server_registry,
                config_overrides=cfg.config_overrides,
                max_output_tokens=cfg.max_output_tokens,
                frame_cache_root=cfg.frame_cache_root,
                enable_logging=True,
            ),
            server_client=server.server_client,
        )
        root = await native._resolve_spatialclaw_root()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.session_id = self.workspace.name
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=_instruction(benchmark, sample),
            metadata=media or _sample_metadata(sample, Path(native.config.video_root).expanduser().resolve()),
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "headers": [],
                "path": f"/ng-rollout/{capture_key}/v1/responses",
                "path_params": {"rollout_id": capture_key},
            }
        )
        self.spatial_config = native._build_spatialclaw_config(root, self.workspace, params, request)
        instruction, images, metadata = await native._materialize_inputs(params, self.workspace, self.spatial_config)
        from spatial_agent.config import set_config
        from spatial_agent.nodes.init_node import init_node
        from spatial_agent.workflow import SpatialAgentWorkflow

        set_config(self.spatial_config)
        self.workflow = SpatialAgentWorkflow(self.spatial_config)
        from spatial_agent.kernel.manager import JupyterKernelManager

        # Publish before any awaited startup; the native pool otherwise keeps a
        # newly starting manager only in an acquire() local variable.
        self.kernel = JupyterKernelManager(timeout_sec=self.spatial_config.timeout_sec)
        self.workflow._kernel_pool = _ResourceKernelPool(self.kernel)
        _install_gym_llm_transport(
            self.workflow.llm_client, journal=self.workspace / "gym-model-responses.jsonl", cookies={}
        )
        runtime = self

        class InitializeOnly:
            async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
                runtime.kernel = config["configurable"]["kernel_manager"]
                initialized = await init_node(state, config)
                messages = []
                for message in initialized["messages"]:
                    role = {"system": "system", "human": "user", "ai": "assistant"}.get(message.type, message.type)
                    messages.append({"role": role, "content": message.content})
                runtime.context = {
                    "session_id": runtime.session_id,
                    "instruction": instruction,
                    "images": images,
                    "metadata": metadata,
                    "system_prompt": config["configurable"]["system_prompt"],
                    "messages": messages,
                }
                return {**state, **initialized}

        self.workflow.graph = InitializeOnly()
        await self.workflow.arun(
            instruction=instruction,
            images=images,
            answer=None,
            session_id=self.session_id,
            defer_report=True,
            **{
                key: metadata.get(key)
                for key in (
                    "frame_indices",
                    "video_source",
                    "fps",
                    "total_video_frames",
                    "duration_sec",
                    "image_groups",
                    "frame_indices_groups",
                    "fps_per_video",
                    "total_frames_per_video",
                    "duration_per_video",
                    "video_names",
                    "video_sources_per_video",
                    "ref_images",
                )
            },
        )
        if self.context is None or self.kernel is None:
            raise RuntimeError("Native SpatialClaw task initialization produced no context or kernel")

    async def capture_answer(self) -> Any:
        value = await self.kernel.check_sentinel("_return_answer_result")
        if isinstance(value, dict) and value.get("text") is not None:
            self.accepted_answer = str(value["text"])
        return value

    async def rpc(
        self, operation: str, *, code: str | None, timeout: float | None, name: str | None, answer: str | None = None
    ) -> Any:
        if operation == "finalize_answer":
            if answer is None:
                raise ValueError("finalize_answer requires answer")
            self.accepted_answer = answer
            return None
        kernel = self.kernel
        if kernel is None:
            raise RuntimeError("SpatialClaw task kernel is not initialized")
        if operation == "execute":
            result = await kernel.execute(code or "", timeout=timeout or self.spatial_config.timeout_sec)
            await self.capture_answer()
            return execution_dict(result)
        if operation == "start":
            return await kernel.start()
        if operation == "shutdown":
            return await kernel.shutdown()
        if operation == "reset_namespace":
            return await kernel.reset_namespace()
        if operation == "get_variables":
            return await kernel.get_variables()
        if operation == "check_sentinel":
            if name in (None, "_return_answer_result"):
                return await self.capture_answer()
            return await kernel.check_sentinel(name)
        if operation == "clear_sentinel":
            if name in (None, "_return_answer_result"):
                await self.capture_answer()
            return await kernel.clear_sentinel(name or "_return_answer_result")
        raise ValueError(f"Unsupported kernel operation: {operation}")

    async def execute(self, code: str, timeout: float | None) -> dict[str, Any]:
        from spatial_agent.kernel.safety import SecuritySandbox

        if self.execution_count >= self.spatial_config.max_steps:
            return tool_result({"error": "SpatialClaw execution-step budget exhausted"}, error=True)
        self.execution_count += 1
        rejection = SecuritySandbox.check(code)
        if rejection:
            return tool_result({"error": f"Security violation: {rejection}"}, error=True)
        limit = min(timeout or self.spatial_config.timeout_sec, self.spatial_config.timeout_sec)
        result = await self.kernel.execute(code, timeout=limit)
        await self.capture_answer()
        payload = execution_dict(result)
        payload["answer_submitted"] = self.accepted_answer is not None
        return tool_result(payload, images=execution_images(result), error=bool(payload.get("error")))

    async def close(self, *, timeout: float, keep_workspace: bool) -> None:
        # Keep references until every shutdown succeeds so an Environment close
        # retry can finish cleanup after a failed first attempt.
        async def release() -> None:
            if self.workflow is not None:
                await self.workflow._kernel_pool.shutdown_all()
                await self.workflow.llm_client.close()
            elif self.kernel is not None:
                await self.kernel.shutdown()

        await asyncio.wait_for(release(), timeout=timeout)
        if not keep_workspace and self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=False)


def execution_dict(result: Any) -> dict[str, Any]:
    """Serialize the external execution result without exposing its live objects."""
    value = (
        asdict(result)
        if is_dataclass(result)
        else {
            key: getattr(result, key, None)
            for key in ("stdout", "stderr", "error", "display_data", "execution_time_sec")
        }
    )
    return json.loads(json.dumps(value, default=str))


def context_result(context: dict[str, Any]) -> dict[str, Any]:
    images: list[dict[str, str]] = []
    messages = []
    for message in context["messages"]:
        content = message["content"]
        if isinstance(content, str):
            messages.append(message)
            continue
        texts = []
        for part in content:
            if part.get("type") == "text":
                texts.append(part)
            elif part.get("type") == "image_url":
                images.append({"image_url": part["image_url"]["url"]})
        messages.append({**message, "content": texts})
    # Images travel through Gym's tool-output contract. Keeping their base64 in
    # the JSON text as well would duplicate them in every subsequent model call.
    return tool_result({**context, "messages": messages}, images=images)


def execution_images(result: Any) -> list[dict[str, str]]:
    from spatial_agent.nodes.execute_node import _parse_show_markers

    images = []
    for value in getattr(result, "display_data", []) or []:
        data = value.get("data", value) if isinstance(value, dict) else {}
        for mime in ("image/png", "image/jpeg"):
            if mime in data:
                images.append({"image_url": f"data:{mime};base64,{data[mime]}"})
    for shown in _parse_show_markers(result.stdout or ""):
        for image in shown["images"]:
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            images.append(
                {
                    "image_url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
                    "label": shown.get("label") or None,
                }
            )
    return images


def tool_result(
    payload: dict[str, Any], *, images: list[dict[str, str]] | None = None, error: bool = False
) -> dict[str, Any]:
    from nemo_gym.core import TOOL_OUTPUT_IMAGES_KEY

    return {**payload, TOOL_OUTPUT_IMAGES_KEY: images or [], "is_error": error}
