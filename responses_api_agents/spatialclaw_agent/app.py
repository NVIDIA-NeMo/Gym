# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym adapter for the pinned SpatialClaw evaluation harness."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from asyncio import Semaphore
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import current_thread, main_thread
from time import time
from types import MethodType
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
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
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
from responses_api_agents.spatialclaw_agent.kernel_transport import _kernel_transport_loop


SPATIALCLAW_URL = "ssh://git@gitlab-master.nvidia.com:12051/ehosseiniasl/spatial_claw.git"
SPATIALCLAW_COMMIT = "946ac114dfcabf9df997629bfa8b6f2f66da1425"  # pragma: allowlist secret


class SpatialClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    spatialclaw_url: str = SPATIALCLAW_URL
    spatialclaw_commit: str = SPATIALCLAW_COMMIT
    spatialclaw_root: str | None = None
    source_cache_root: str | None = None
    dataset_config: str
    model_config_path: str | None = None
    model_name: str | None = None
    video_root: str
    gpu_server_registry: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    max_output_tokens: int | None = None
    concurrency: int = Field(default=1, ge=1, le=1)
    timeout: int = 1800
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


def _git_output(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def _checkout_name(url: str, commit: str) -> str:
    repo = re.sub(r"\W+", "_", url.rstrip("/").split("/")[-1].removesuffix(".git"))
    return f"{repo}_{commit[:12]}"


@contextmanager
def _checkout_lock(lock_path: Path):
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def _validate_spatialclaw_checkout(root: Path, commit: str) -> Path:
    root = root.expanduser().resolve()
    if not (root / "spatial_agent" / "workflow.py").is_file():
        raise RuntimeError(f"Invalid SpatialClaw checkout: {root}")
    try:
        head = _git_output("git", "-C", str(root), "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"SpatialClaw checkout is not a Git worktree: {root}") from exc
    if head != commit:
        raise RuntimeError(f"SpatialClaw checkout {root} is at {head}, expected pinned commit {commit}")
    return root


def ensure_spatialclaw_checkout(url: str, commit: str, cache_root: Path) -> Path:
    """Clone SpatialClaw once per pin and validate the exact source revision."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("spatialclaw_commit must be a full 40-character lowercase Git SHA")
    checkout = cache_root / _checkout_name(url, commit)
    lock_path = cache_root / f".{_checkout_name(url, commit)}.setup.lock"
    with _checkout_lock(lock_path):
        if not checkout.exists():
            subprocess.run(["git", "clone", url, str(checkout)], check=True)
        head = _git_output("git", "-C", str(checkout), "rev-parse", "HEAD")
        if head != commit:
            subprocess.run(["git", "-C", str(checkout), "fetch", "origin", commit], check=False)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
    return _validate_spatialclaw_checkout(checkout, commit)


def _install_source_path(root: Path) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    pythonpath = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if root_str not in pythonpath:
        os.environ["PYTHONPATH"] = os.pathsep.join([root_str, *pythonpath])


def _configure_gpu_registry(root: Path, registry: str | None) -> None:
    """Expose an external SpatialClaw GPU registry to spawned Jupyter kernels."""
    if not registry:
        return
    source = Path(registry).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SpatialClaw GPU server registry not found: {source}")
    target = root / "spatial_agent" / "logs" / "gpu_server.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() == source:
            return
        raise RuntimeError(f"SpatialClaw GPU registry already exists at {target}; refusing to replace it")
    try:
        target.symlink_to(source)
    except FileExistsError:
        if target.resolve() == source:
            return
        raise


def _dump_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported Responses input item: {type(item)!r}")


def _extract_instruction_and_images(
    body: NeMoGymResponseCreateParamsNonStreaming,
) -> tuple[str, list[str]]:
    items = [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
    texts: list[str] = []
    image_urls: list[str] = []
    if body.instructions:
        texts.append(body.instructions)
    for raw_item in items:
        item = _dump_item(raw_item)
        if item.get("role") not in {"user", "developer", "system"}:
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            texts.append(content)
            continue
        for raw_part in content or []:
            part = _dump_item(raw_part)
            if part.get("type") in {"text", "input_text"}:
                texts.append(str(part.get("text", "")))
            elif part.get("type") in {"image", "image_url", "input_image"}:
                value = part.get("image_url") or part.get("image") or part.get("url")
                if isinstance(value, Mapping):
                    value = value.get("url") or value.get("file_url") or value.get("path")
                if isinstance(value, str):
                    image_urls.append(value)
    return "\n\n".join(text for text in texts if text), image_urls


def _metadata_video_references(body: NeMoGymResponseCreateParamsNonStreaming) -> list[str]:
    metadata = dict(body.metadata or {})
    populated = [key for key in ("video_data", "video_path", "video_paths") if metadata.get(key)]
    if len(populated) > 1:
        raise ValueError(f"metadata video keys are mutually exclusive; got {populated}")
    if not populated:
        return []
    key = populated[0]
    value = metadata[key]
    if key != "video_paths":
        return [str(value)]
    paths = json.loads(value) if isinstance(value, str) else value
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("metadata.video_paths must be a JSON list of strings")
    return paths


def _metadata_json(body: NeMoGymResponseCreateParamsNonStreaming, key: str) -> Any:
    value = dict(body.metadata or {}).get(key)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata.{key} must contain valid JSON") from exc
    return value


def _metadata_path_list(body: NeMoGymResponseCreateParamsNonStreaming, key: str) -> list[str]:
    value = _metadata_json(body, key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(path, str) for path in value):
        raise ValueError(f"metadata.{key} must be a JSON list of strings")
    return value


def _metadata_path_groups(body: NeMoGymResponseCreateParamsNonStreaming, key: str) -> list[list[str]]:
    value = _metadata_json(body, key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(group, list) and all(isinstance(path, str) for path in group) for group in value
    ):
        raise ValueError(f"metadata.{key} must be a JSON list of string lists")
    return value


def _resolve_media_path(reference: str, media_root: str, kind: str) -> str:
    if reference.startswith(("data:", "http://", "https://", "file://")):
        return reference
    path = Path(reference).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        root = Path(media_root).expanduser().resolve()
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"metadata {kind} path escapes configured video_root: {reference!r}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"SpatialClaw {kind} not found: {resolved}")
    return str(resolved)


def _resolve_video_path(reference: str, video_root: str) -> str:
    """Backward-compatible wrapper retained for focused adapter tests."""
    return _resolve_media_path(reference, video_root, "video")


def _suffix_for_url(url: str, default: str) -> str:
    if url.startswith("data:"):
        media_type = url[5:].split(";", 1)[0]
        subtype = media_type.split("/", 1)[-1].split("+", 1)[0]
        return f".{subtype}" if subtype else default
    return Path(urlparse(url).path).suffix or default


async def _materialize_url(url: str, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("file://"):
        path = Path(unquote(urlparse(url).path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    if url.startswith("data:"):
        header, payload = url.split(",", 1)
        data = base64.b64decode(payload) if ";base64" in header else unquote(payload).encode()
        output_path.write_bytes(data)
        return str(output_path)
    if url.startswith(("http://", "https://")):
        response = await http_request(method="GET", url=url)
        await raise_for_status(response)
        output_path.write_bytes(await response.read())
        return str(output_path)
    path = Path(url).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _frame_cache_dir(video_path: str, cache_root: str, spatial_config: Any) -> Path:
    path = Path(video_path).resolve()
    stat = path.stat()
    identity = json.dumps(
        {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "video_max_fps": spatial_config.video_max_fps,
            "resize_short_edge": spatial_config.video_frame_resize_short_edge,
        },
        sort_keys=True,
    )
    return Path(cache_root).expanduser().resolve() / hashlib.sha256(identity.encode()).hexdigest()[:24]


def _extract_video_frames_locked(
    extract_video_frames: Any,
    video_path: str,
    cache_dir: Path,
    spatial_config: Any,
) -> tuple[list[str], list[int], float, int]:
    with _checkout_lock(cache_dir / ".extract.lock"):
        return extract_video_frames(
            video_path,
            str(cache_dir),
            spatial_config.video_max_fps,
            spatial_config.video_frame_resize_short_edge,
        )


class _AiohttpChatCompletions:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")
        self.cookies = None

    async def create(self, **kwargs: Any) -> NeMoGymChatCompletion:
        payload = dict(kwargs)
        extra_body = dict(payload.pop("extra_body", {}) or {})
        metadata = dict(payload.pop("metadata", {}) or {})
        chat_template_kwargs = extra_body.pop("chat_template_kwargs", None)
        if chat_template_kwargs:
            metadata["chat_template_kwargs"] = json.dumps(chat_template_kwargs)
            payload["chat_template_kwargs"] = chat_template_kwargs
        if extra_body:
            metadata["extra_body"] = json.dumps(extra_body)
        if metadata:
            payload["metadata"] = metadata
        if current_thread() is not main_thread():
            future = asyncio.run_coroutine_threadsafe(self._create(payload), _kernel_transport_loop())
            return await asyncio.wrap_future(future)
        return await self._create(payload)

    async def _create(self, payload: dict[str, Any]) -> NeMoGymChatCompletion:
        response = await http_request(
            method="POST",
            url=f"{self.endpoint}/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer gym"},  # pragma: allowlist secret
            cookies=self.cookies,
        )
        await raise_for_status(response)
        self.cookies = response.cookies
        return NeMoGymChatCompletion.model_validate(await get_response_json(response))


class _AiohttpOpenAIClient:
    def __init__(self, endpoint: str):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _AiohttpChatCompletions(endpoint)

    async def close(self) -> None:
        return None


def _gym_get_client(llm_client: Any, endpoint: str) -> _AiohttpOpenAIClient:
    client = llm_client._client_pool.get(endpoint)
    if client is None:
        client = _AiohttpOpenAIClient(endpoint)
        llm_client._client_pool[endpoint] = client
    return client


async def _gym_noop_rediscover(_llm_client: Any) -> None:
    return None


def _install_gym_llm_transport(llm_client: Any) -> None:
    """Keep SpatialClaw semantics while replacing its httpx OpenAI transport."""
    llm_client._client_pool.clear()
    llm_client._is_vllm = True
    llm_client._get_client = MethodType(_gym_get_client, llm_client)
    llm_client._maybe_rediscover = MethodType(_gym_noop_rediscover, llm_client)


def _session_id(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    metadata = dict(body.metadata or {})
    value = str(metadata.get("spatialclaw_session_id") or uuid4().hex)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("metadata.spatialclaw_session_id must be 1-128 filename-safe characters")
    return value


def _config_path(root: Path, value: str | None, kind: str) -> str | None:
    if not value:
        return None
    requested = Path(value).expanduser()
    candidates = (
        [requested]
        if requested.is_absolute()
        else [root / requested, root / "spatial_agent" / "config" / kind / requested]
    )
    for candidate in candidates:
        with_suffix = candidate if candidate.suffix == ".json" else candidate.with_suffix(".json")
        if with_suffix.is_file():
            return str(with_suffix.resolve())
    raise FileNotFoundError(f"SpatialClaw {kind} config not found: {value}")


async def _shutdown_workflow(workflow: Any) -> None:
    await asyncio.gather(workflow.llm_client.close(), workflow._kernel_pool.shutdown_all(), return_exceptions=True)


class SpatialClawAgent(SimpleResponsesAPIAgent):
    config: SpatialClawAgentConfig
    sem: Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def _resolve_spatialclaw_root(self) -> Path:
        if self.config.spatialclaw_root:
            root = await asyncio.to_thread(
                _validate_spatialclaw_checkout,
                Path(self.config.spatialclaw_root),
                self.config.spatialclaw_commit,
            )
        else:
            cache_root = (
                Path(self.config.source_cache_root).expanduser().resolve()
                if self.config.source_cache_root
                else Path(__file__).parent / ".sources"
            )
            root = await asyncio.to_thread(
                ensure_spatialclaw_checkout,
                self.config.spatialclaw_url,
                self.config.spatialclaw_commit,
                cache_root,
            )
        _install_source_path(root)
        _configure_gpu_registry(root, self.config.gpu_server_registry)
        return root

    def _model_base_url(self) -> str:
        model_server_config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.model_server.name,
        )
        return f"{self.server_client._build_server_base_url(model_server_config)}/v1"

    def _build_spatialclaw_config(
        self,
        root: Path,
        session_dir: Path,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> Any:
        from spatial_agent.config import SpatialAgentConfig

        spatial_config = SpatialAgentConfig()
        spatial_config._load_from_envs()
        spatial_config.update_from_dataset_json(_config_path(root, self.config.dataset_config, "dataset"))
        model_config = _config_path(root, self.config.model_config_path, "model")
        if model_config:
            spatial_config.update_from_model_json(model_config)
        for name, value in copy.deepcopy(self.config.config_overrides).items():
            if value is None:
                continue
            if not hasattr(spatial_config, name):
                raise ValueError(f"Unknown SpatialClaw config override: {name}")
            setattr(spatial_config, name, value)

        spatial_config.llm_base_url = self._model_base_url()
        spatial_config.llm_model = self.config.model_name or self.config.model_server.name
        spatial_config.llm_api_key = "gym"  # pragma: allowlist secret
        spatial_config.work_dir = str(session_dir)
        spatial_config.concurrency = 1
        spatial_config.generate_report = False
        spatial_config.enable_logging = self.config.enable_logging

        role_names = (
            "main_params",
            "planning_params",
            "general_params",
            "vlm_params",
            "vlm_grounding_params",
            "reflection_params",
        )
        for role_name in role_names:
            role = getattr(spatial_config, role_name, None)
            if role is None:
                continue
            if body.temperature is not None:
                role.temperature = body.temperature
            if body.top_p is not None:
                role.top_p = body.top_p
            requested_max = body.max_output_tokens
            if requested_max is not None or self.config.max_output_tokens is not None:
                values = [value for value in (requested_max, self.config.max_output_tokens) if value is not None]
                role.max_tokens = min(values)
        return spatial_config

    async def _materialize_inputs(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        session_dir: Path,
        spatial_config: Any,
    ) -> tuple[str, list[str], dict[str, Any]]:
        instruction, image_urls = _extract_instruction_and_images(body)
        if not instruction:
            raise ValueError("SpatialClaw request has no text instruction")

        media_dir = session_dir / "request_media"
        images = [
            await _materialize_url(url, media_dir / f"image-{index}{_suffix_for_url(url, '.png')}")
            for index, url in enumerate(image_urls)
        ]
        image_references = _metadata_path_list(body, "image_paths")
        explicit_image_groups = _metadata_path_groups(body, "image_groups")
        if image_references and explicit_image_groups:
            raise ValueError("metadata.image_paths and metadata.image_groups are mutually exclusive")
        if image_urls and explicit_image_groups:
            raise ValueError("input image blocks and metadata.image_groups are mutually exclusive")

        for reference in image_references:
            resolved = _resolve_media_path(reference, self.config.video_root, "image")
            images.append(
                await _materialize_url(
                    resolved,
                    media_dir / f"image-{len(images)}{_suffix_for_url(resolved, '.png')}",
                )
            )

        materialized_image_groups: list[list[str]] = []
        for group_index, group in enumerate(explicit_image_groups):
            materialized_group = []
            for image_index, reference in enumerate(group):
                resolved = _resolve_media_path(reference, self.config.video_root, "image")
                materialized_group.append(
                    await _materialize_url(
                        resolved,
                        media_dir / f"group-{group_index}-image-{image_index}{_suffix_for_url(resolved, '.png')}",
                    )
                )
            materialized_image_groups.append(materialized_group)
        if materialized_image_groups:
            images.extend(image for group in materialized_image_groups for image in group)

        ref_images = []
        for index, reference in enumerate(_metadata_path_list(body, "ref_image_paths")):
            resolved = _resolve_media_path(reference, self.config.video_root, "reference image")
            ref_images.append(
                await _materialize_url(
                    resolved,
                    media_dir / f"reference-{index}{_suffix_for_url(resolved, '.png')}",
                )
            )

        video_references = _metadata_video_references(body)
        video_paths = []
        for index, reference in enumerate(video_references):
            resolved = _resolve_video_path(reference, self.config.video_root)
            if resolved.startswith(("data:", "http://", "https://", "file://")):
                resolved = await _materialize_url(
                    resolved,
                    media_dir / f"video-{index}{_suffix_for_url(resolved, '.mp4')}",
                )
            video_paths.append(resolved)

        metadata: dict[str, Any] = {}
        if materialized_image_groups:
            metadata["image_groups"] = materialized_image_groups
        for key in (
            "frame_indices",
            "frame_indices_groups",
            "fps_per_video",
            "total_frames_per_video",
            "duration_per_video",
            "video_names",
            "video_sources_per_video",
        ):
            value = _metadata_json(body, key)
            if value is not None:
                metadata[key] = value
        for key in ("fps", "total_video_frames", "duration_sec", "video_source"):
            value = dict(body.metadata or {}).get(key)
            if value is not None and value != "":
                metadata[key] = value
        if video_paths:
            from spatial_agent.evals.base import extract_video_frames

            frame_groups: list[list[str]] = []
            index_groups: list[list[int]] = []
            fps_values: list[float] = []
            total_values: list[int] = []
            for video_path in video_paths:
                frames, indices, fps, total = await asyncio.to_thread(
                    _extract_video_frames_locked,
                    extract_video_frames,
                    video_path,
                    _frame_cache_dir(video_path, self.config.frame_cache_root, spatial_config),
                    spatial_config,
                )
                if not frames:
                    raise RuntimeError(f"SpatialClaw extracted no frames from {video_path}")
                frame_groups.append(frames)
                index_groups.append(indices)
                fps_values.append(float(fps))
                total_values.append(int(total))
            images.extend(frame for group in frame_groups for frame in group)
            metadata["video_sources_per_video"] = video_paths
            if len(video_paths) == 1:
                metadata.update(
                    video_source=video_paths[0],
                    frame_indices=index_groups[0],
                    fps=fps_values[0],
                    total_video_frames=total_values[0],
                    duration_sec=total_values[0] / fps_values[0] if fps_values[0] else 0.0,
                )
            else:
                metadata.update(
                    image_groups=frame_groups,
                    frame_indices_groups=index_groups,
                    fps_per_video=fps_values,
                    total_frames_per_video=total_values,
                    duration_per_video=[total / fps if fps else 0.0 for total, fps in zip(total_values, fps_values)],
                    video_names=[Path(path).name for path in video_paths],
                )
        if not images:
            raise ValueError("SpatialClaw request has no image or video input")
        metadata["ref_images"] = ref_images
        return instruction, images, metadata

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        if self.sem is None:
            self.sem = Semaphore(self.config.concurrency)
        async with self.sem:
            root = await self._resolve_spatialclaw_root()
            session_id = _session_id(body)
            session_dir = Path(self.config.workspace_root).expanduser().resolve() / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            workflow = None
            try:
                spatial_config = self._build_spatialclaw_config(root, session_dir, body)
                instruction, images, metadata = await self._materialize_inputs(body, session_dir, spatial_config)
                from spatial_agent.config import set_config
                from spatial_agent.workflow import SpatialAgentWorkflow

                set_config(spatial_config)
                workflow = SpatialAgentWorkflow(spatial_config)
                _install_gym_llm_transport(workflow.llm_client)
                try:
                    result = await asyncio.wait_for(
                        workflow.arun(
                            instruction=instruction,
                            images=images,
                            answer=None,
                            session_id=session_id,
                            frame_indices=metadata.get("frame_indices"),
                            video_source=metadata.get("video_source"),
                            fps=metadata.get("fps"),
                            total_video_frames=metadata.get("total_video_frames"),
                            duration_sec=metadata.get("duration_sec"),
                            image_groups=metadata.get("image_groups"),
                            frame_indices_groups=metadata.get("frame_indices_groups"),
                            fps_per_video=metadata.get("fps_per_video"),
                            total_frames_per_video=metadata.get("total_frames_per_video"),
                            duration_per_video=metadata.get("duration_per_video"),
                            video_names=metadata.get("video_names"),
                            video_sources_per_video=metadata.get("video_sources_per_video"),
                            ref_images=metadata.get("ref_images"),
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
                print(
                    f"[spatialclaw_agent] session {session_id} failed:\n{error_traceback}", file=sys.stderr, flush=True
                )
                try:
                    (session_dir / "error.traceback.txt").write_text(error_traceback, encoding="utf-8")
                except OSError:
                    pass
                raise
            finally:
                if workflow is not None:
                    await _shutdown_workflow(workflow)
                if not self.config.keep_workspaces:
                    shutil.rmtree(session_dir, ignore_errors=True)

            final = result.get("final_answer") or {}
            final_answer = str(final.get("text", "") if isinstance(final, Mapping) else final)
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
                        content=[NeMoGymResponseOutputText(text=final_answer, annotations=[])],
                    )
                ],
                tool_choice=body.tool_choice,
                tools=body.tools,
                parallel_tool_calls=body.parallel_tool_calls,
                metadata={
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

        response_params = body.responses_create_params.model_copy(deep=True)
        response_params.metadata = dict(response_params.metadata or {}) | {"spatialclaw_session_id": uuid4().hex}
        agent_response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=response_params,
            cookies=cookies,
        )
        await raise_for_status(agent_response)
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
