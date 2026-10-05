# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation-first NeMo Gym harness for the SpatialClaw agent."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import traceback
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from nemo_gym.server_utils import request as http_request
from responses_api_agents.spatialclaw_agent.native_client import create_native_client


logger = logging.getLogger(__name__)


class SpatialClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    spatialclaw_root: str = Field(default_factory=lambda: os.environ.get("SPATIALCLAW_ROOT", ""))
    model_name: str = ""
    dataset_config: str | None = None
    model_config_path: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    video_mm_processor_kwargs: dict[str, Any] = Field(default_factory=dict)
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    # Upstream prompt/tool modules share a process-global configuration singleton.
    concurrency: int = Field(default=1, ge=1, le=1)
    timeout: int = Field(default=1800, gt=0)
    workspace_root: str = "outputs/spatialclaw_agent"
    frame_cache_root: str = "outputs/spatialclaw_frame_cache"
    keep_workspaces: bool = True
    enable_logging: bool = True


class SpatialClawAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SpatialClawAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    turns_used: int = 0
    termination_reason: str | None = None


def _dump_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported Responses input item: {type(item)!r}")


def _part_url(part: dict[str, Any]) -> str | None:
    for key in ("image_url", "video_url", "image", "video", "url"):
        value = part.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for nested_key in ("url", "file_url", "path"):
                nested = value.get(nested_key)
                if isinstance(nested, str):
                    return nested
    return None


def _suffix_for_url(url: str, default: str) -> str:
    if url.startswith("data:"):
        media_type = url[5:].split(";", 1)[0]
        subtype = media_type.split("/", 1)[-1].split("+", 1)[0]
        return f".{subtype}" if subtype else default
    return Path(urlparse(url).path).suffix or default


async def _materialize_url(url: str, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("file://"):
        return unquote(urlparse(url).path)
    if url.startswith("data:"):
        _, payload = url.split(",", 1)
        output_path.write_bytes(base64.b64decode(payload))
        return str(output_path)
    if url.startswith(("http://", "https://")):
        response = await http_request(method="GET", url=url)
        await raise_for_status(response)
        output_path.write_bytes(await response.read())
        return str(output_path)
    return str(Path(url).expanduser().resolve())


def _metadata(body: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
    metadata = body.metadata or {}
    raw = metadata.get("spatialclaw") if isinstance(metadata, dict) else None
    if raw is None:
        return {}
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("responses_create_params.metadata.spatialclaw must encode a JSON object")
    return parsed


def _session_id(value: Any) -> str:
    session_id = str(value or uuid4().hex)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", session_id):
        raise ValueError(
            "SpatialClaw metadata.session_id must be 1-128 filename-safe characters "
            "(letters, digits, dot, underscore, or hyphen)"
        )
    return session_id


def _extract_request_input(
    body: NeMoGymResponseCreateParamsNonStreaming,
) -> tuple[str, list[str], list[str]]:
    items = [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
    text_parts: list[str] = []
    image_urls: list[str] = []
    video_urls: list[str] = []
    for item in items:
        message = _dump_item(item)
        if message.get("role") not in {"user", "developer", "system"}:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        for raw_part in content or []:
            part = _dump_item(raw_part)
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                text_parts.append(str(part.get("text", "")))
            elif part_type in {"image", "image_url", "input_image"}:
                if url := _part_url(part):
                    image_urls.append(url)
            elif part_type in {"video", "video_url", "input_video"}:
                if url := _part_url(part):
                    video_urls.append(url)
    return "\n\n".join(part for part in text_parts if part), image_urls, video_urls


def _configure_video_role_preprocessing(spatial_config: Any, overrides: dict[str, Any]) -> None:
    """Apply explicit video processor overrides to every SpatialClaw model role."""
    for role_name in (
        "main_params",
        "planning_params",
        "general_params",
        "vlm_params",
        "vlm_grounding_params",
        "reflection_params",
    ):
        role_params = getattr(spatial_config, role_name, None)
        if role_params is None:
            continue
        kwargs = dict(getattr(role_params, "mm_processor_kwargs", {}) or {})
        kwargs.update(overrides)
        role_params.mm_processor_kwargs = kwargs


def _frame_cache_dir(video_path: str, cache_root: str, spatial_config: Any) -> Path:
    path = Path(video_path).expanduser().resolve()
    stat = path.stat()
    identity = json.dumps(
        {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "video_max_fps": getattr(spatial_config, "video_max_fps", None),
            "resize_short_edge": getattr(spatial_config, "video_frame_resize_short_edge", None),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return Path(cache_root).expanduser().resolve() / digest


def _extract_video_frames_locked(
    extract_video_frames: Any,
    video_path: str,
    cache_dir: Path,
    spatial_config: Any,
) -> tuple[list[str], list[int], float, int]:
    """Serialize population of a shared frame cache across rollout workers."""
    import fcntl

    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / ".extract.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return extract_video_frames(
            video_path,
            str(cache_dir),
            getattr(spatial_config, "video_max_fps", None),
            getattr(spatial_config, "video_frame_resize_short_edge", None),
        )


class SpatialClawAgent(SimpleResponsesAPIAgent):
    config: SpatialClawAgentConfig
    sem: Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _resolve_spatialclaw_root(self) -> Path:
        root_value = self.config.spatialclaw_root or os.environ.get("SPATIALCLAW_ROOT", "")
        if not root_value:
            raise RuntimeError("SPATIALCLAW_ROOT or agent spatialclaw_root is required")
        root = Path(root_value).expanduser().resolve()
        if not (root / "spatial_agent" / "workflow.py").is_file():
            raise RuntimeError(f"Invalid SpatialClaw checkout: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        pythonpath = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
        if str(root) not in pythonpath:
            os.environ["PYTHONPATH"] = os.pathsep.join([str(root), *pythonpath])
        return root

    def _model_base_url(self, request: Request) -> str:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id=rollout_id)

    @staticmethod
    def _config_path(root: Path, value: str | None, kind: str) -> str | None:
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / "spatial_agent" / "config" / kind / path
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        if not path.is_file():
            raise FileNotFoundError(f"SpatialClaw {kind} config not found: {path}")
        return str(path)

    def _build_spatialclaw_config(
        self,
        root: Path,
        run_metadata: dict[str, Any],
        session_dir: Path,
        body: NeMoGymResponseCreateParamsNonStreaming,
        request: Request,
    ) -> Any:
        from spatial_agent.config import SpatialAgentConfig

        spatial_config = SpatialAgentConfig()
        spatial_config._load_from_envs()
        dataset_path = self._config_path(
            root,
            run_metadata.get("dataset_config") or self.config.dataset_config,
            "dataset",
        )
        model_path = self._config_path(
            root,
            run_metadata.get("model_config") or self.config.model_config_path,
            "model",
        )
        if dataset_path:
            spatial_config.update_from_dataset_json(dataset_path)
        if model_path:
            spatial_config.update_from_model_json(model_path)

        overrides = copy.deepcopy(self.config.config_overrides)
        overrides.update(run_metadata.get("config_overrides") or {})
        for name, value in overrides.items():
            if not hasattr(spatial_config, name):
                raise ValueError(f"Unknown SpatialClaw config override: {name}")
            setattr(spatial_config, name, value)

        spatial_config.llm_base_url = self._model_base_url(request)
        spatial_config.llm_model = self.config.model_name or self.config.model_server.name
        spatial_config.llm_api_key = "gym"  # pragma: allowlist secret
        spatial_config.work_dir = str(session_dir)
        spatial_config.concurrency = 1
        spatial_config.generate_report = False
        spatial_config.enable_logging = bool(run_metadata.get("enable_logging", self.config.enable_logging))

        role_params = tuple(
            params
            for params in (
                getattr(spatial_config, "main_params", None),
                getattr(spatial_config, "planning_params", None),
                getattr(spatial_config, "general_params", None),
                getattr(spatial_config, "vlm_params", None),
                getattr(spatial_config, "vlm_grounding_params", None),
                getattr(spatial_config, "reflection_params", None),
            )
            if params is not None
        )
        temperature = body.temperature if body.temperature is not None else self.config.temperature
        top_p = body.top_p if body.top_p is not None else self.config.top_p
        requested_max = body.max_output_tokens
        configured_max = self.config.max_output_tokens
        max_output_tokens = (
            configured_max
            if requested_max is None
            else requested_max
            if configured_max is None
            else min(requested_max, configured_max)
        )
        for params in role_params:
            if temperature is not None:
                params.temperature = temperature
            if top_p is not None:
                params.top_p = top_p
            if max_output_tokens is not None:
                params.max_tokens = max_output_tokens
        return spatial_config

    async def _materialize_inputs(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        run_metadata: dict[str, Any],
        session_dir: Path,
        spatial_config: Any,
    ) -> tuple[str, list[str], dict[str, Any]]:
        instruction, image_urls, video_urls = _extract_request_input(body)
        input_dir = session_dir / "request_media"
        image_paths = [
            await _materialize_url(url, input_dir / f"image-{index}{_suffix_for_url(url, '.png')}")
            for index, url in enumerate(image_urls)
        ]
        video_paths = [
            await _materialize_url(url, input_dir / f"video-{index}{_suffix_for_url(url, '.mp4')}")
            for index, url in enumerate(video_urls)
        ]

        ref_image_urls = list(run_metadata.get("ref_images") or [])
        if ref_image_urls:
            run_metadata["ref_images"] = [
                await _materialize_url(
                    url,
                    input_dir / f"ref-image-{index}{_suffix_for_url(url, '.png')}",
                )
                for index, url in enumerate(ref_image_urls)
            ]

        if video_paths:
            from spatial_agent.evals.base import extract_video_frames

            video_frame_groups: list[list[str]] = []
            frame_indices_groups: list[list[int]] = []
            fps_per_video: list[float] = []
            total_frames_per_video: list[int] = []
            duration_per_video: list[float] = []
            for video_path in video_paths:
                cache_dir = _frame_cache_dir(video_path, self.config.frame_cache_root, spatial_config)
                frames, indices, fps, total = await asyncio.to_thread(
                    _extract_video_frames_locked,
                    extract_video_frames,
                    video_path,
                    cache_dir,
                    spatial_config,
                )
                video_frame_groups.append(frames)
                frame_indices_groups.append(indices)
                fps_per_video.append(float(fps))
                total_frames_per_video.append(int(total))
                duration_per_video.append(total / fps if fps else 0.0)

            image_paths.extend(frame for group in video_frame_groups for frame in group)
            run_metadata["video_sources_per_video"] = video_paths
            run_metadata["extracted_frame_counts"] = [len(group) for group in video_frame_groups]
            if len(video_paths) == 1:
                run_metadata["video_source"] = video_paths[0]
                run_metadata["frame_indices"] = frame_indices_groups[0]
                run_metadata["fps"] = fps_per_video[0]
                run_metadata["total_video_frames"] = total_frames_per_video[0]
                run_metadata["duration_sec"] = duration_per_video[0]
            else:
                run_metadata["image_group_sizes"] = [len(group) for group in video_frame_groups]
                run_metadata["frame_indices_groups"] = frame_indices_groups
                run_metadata["fps_per_video"] = fps_per_video
                run_metadata["total_frames_per_video"] = total_frames_per_video
                run_metadata["duration_per_video"] = duration_per_video
                run_metadata["video_names"] = [Path(path).name for path in video_paths]

        if not instruction:
            raise ValueError("SpatialClaw request has no text instruction")
        if not image_paths:
            raise ValueError("SpatialClaw request has no image or decodable video frames")
        run_metadata.setdefault("frame_indices", list(range(len(image_paths))))
        return instruction, image_paths, run_metadata

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        if self.sem is None:  # defensive for model_construct-based tests
            self.sem = Semaphore(self.config.concurrency)
        async with self.sem:
            root = self._resolve_spatialclaw_root()
            run_metadata = _metadata(body)
            session_id = _session_id(run_metadata.get("session_id"))
            session_dir = Path(self.config.workspace_root).expanduser().resolve() / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            workflow = None
            try:
                spatial_config = self._build_spatialclaw_config(root, run_metadata, session_dir, body, request)
                instruction, images, run_metadata = await self._materialize_inputs(
                    body, run_metadata, session_dir, spatial_config
                )
                video_processor_overrides = copy.deepcopy(self.config.video_mm_processor_kwargs)
                video_processor_overrides.update(run_metadata.get("video_mm_processor_kwargs") or {})
                if run_metadata.get("video_sources_per_video") and video_processor_overrides:
                    _configure_video_role_preprocessing(spatial_config, video_processor_overrides)

                from spatial_agent.config import set_config

                # SpatialClaw's prompt and tool modules read the package-level config
                # singleton. Every request in one agent instance uses the same benchmark
                # config; install it after all Gym overrides have been applied.
                set_config(spatial_config)
                from spatial_agent.workflow import SpatialAgentWorkflow

                workflow = SpatialAgentWorkflow(spatial_config)
                workflow.llm_client = create_native_client(spatial_config)
                group_sizes = list(run_metadata.get("image_group_sizes") or [])
                image_groups = None
                if group_sizes:
                    image_groups = []
                    offset = 0
                    for raw_size in group_sizes:
                        size = int(raw_size)
                        image_groups.append(images[offset : offset + size])
                        offset += size
                    if offset != len(images):
                        raise ValueError("image_group_sizes does not cover all request images")

                try:
                    result = await asyncio.wait_for(
                        workflow.arun(
                            instruction=instruction,
                            images=images,
                            answer=None,
                            session_id=session_id,
                            frame_indices=run_metadata.get("frame_indices"),
                            video_source=run_metadata.get("video_source"),
                            fps=run_metadata.get("fps"),
                            total_video_frames=run_metadata.get("total_video_frames"),
                            duration_sec=run_metadata.get("duration_sec"),
                            image_groups=image_groups,
                            frame_indices_groups=run_metadata.get("frame_indices_groups"),
                            fps_per_video=run_metadata.get("fps_per_video"),
                            total_frames_per_video=run_metadata.get("total_frames_per_video"),
                            duration_per_video=run_metadata.get("duration_per_video"),
                            video_names=run_metadata.get("video_names"),
                            video_sources_per_video=run_metadata.get("video_sources_per_video"),
                            ref_images=run_metadata.get("ref_images"),
                            defer_report=True,
                        ),
                        timeout=self.config.timeout,
                    )
                except asyncio.TimeoutError:
                    result = {
                        "final_answer": {"text": ""},
                        "termination_reason": "timeout",
                        "step_count": 0,
                        "total_tool_calls": 0,
                        "usage": {},
                    }
            except Exception:
                error_traceback = traceback.format_exc()
                logger.exception("SpatialClaw session %s failed", session_id)
                try:
                    (session_dir / "error.traceback.txt").write_text(error_traceback, encoding="utf-8")
                except OSError:
                    pass
                if not self.config.keep_workspaces:
                    shutil.rmtree(session_dir, ignore_errors=True)
                raise
            finally:
                if workflow is not None:
                    workflow.shutdown()

            final_answer = str((result.get("final_answer") or {}).get("text", ""))
            usage = result.get("usage") or {}
            input_tokens = int(usage.get("total_prompt_tokens", 0) or 0)
            output_tokens = int(usage.get("total_completion_tokens", 0) or 0)
            reasoning_tokens = int(usage.get("total_reasoning_tokens", 0) or 0)
            response = NeMoGymResponse(
                id=f"spatialclaw-{session_id}",
                created_at=int(time()),
                model=self.config.model_name or self.config.model_server.name,
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id=f"spatialclaw-answer-{session_id}",
                        content=[
                            NeMoGymResponseOutputText(
                                type="output_text",
                                text=final_answer,
                                annotations=[],
                            )
                        ],
                    )
                ],
                tool_choice=body.tool_choice,
                tools=body.tools,
                parallel_tool_calls=body.parallel_tool_calls,
                metadata={
                    "spatialclaw_final_answer": final_answer,
                    "spatialclaw_termination_reason": str(result.get("termination_reason") or ""),
                    "spatialclaw_turns": str(result.get("step_count", 0) or 0),
                    "spatialclaw_tool_calls": str(result.get("total_tool_calls", 0) or 0),
                },
                usage=NeMoGymResponseUsage(
                    input_tokens=input_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                    output_tokens=output_tokens,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                    total_tokens=input_tokens + output_tokens,
                ),
            )
            if not self.config.keep_workspaces:
                shutil.rmtree(session_dir, ignore_errors=True)
            return response

    async def run(
        self,
        request: Request,
        body: SpatialClawAgentRunRequest = Body(),
    ) -> SpatialClawAgentVerifyResponse:
        cookies = request.cookies
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed)
        cookies = seed.cookies

        agent_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(agent_response)
        cookies = agent_response.cookies
        agent_json = await get_response_json(agent_response)

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": agent_json},
            cookies=cookies,
        )
        await raise_for_status(verify)
        verify_json = await get_response_json(verify)
        metadata = agent_json.get("metadata") or {}
        return SpatialClawAgentVerifyResponse.model_validate(
            verify_json
            | {
                "turns_used": int(metadata.get("spatialclaw_turns", 0) or 0),
                "termination_reason": metadata.get("spatialclaw_termination_reason"),
            }
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SpatialClawAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SpatialClawAgent.run_webserver()
