# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agentic setting's sandbox: each episode's seed lends the harness a sandbox that holds the episode's video.

The sandbox image (sandbox_image/Dockerfile) carries OpenCode, FFmpeg and the video tools: the OpenCode plugin
``video_tools.mjs`` and the ``video_tools`` program it runs. With ``sandbox_provider`` set, each seed

1. checks the row's ``video_binding``: its ``source`` must be a regular file under one of ``video_roots``;
2. starts the sandbox with the tools' configuration (``tools.json``) and two checksum lists;
3. uploads the video to ``sandbox_media_root/<sandbox_path>`` with ``nemo_gym.sandbox.upload_path``;
4. runs one command in the sandbox: the image's tool files must have this repository's SHA-256 and the video the
   row's SHA-256; then the video is made read-only, and the tools check their configuration and FFmpeg.

A row whose videos are too large to upload for every episode names ``preloaded_media`` instead: media its sandbox
image already holds, with a catalog of the files (task_data.py). Its seed starts the sandbox from the image that
``sandbox_preloaded_images`` maps the row's ``image`` to, with the row's other host ``files`` (checked like a video),
and uploads nothing; the command in the sandbox checks the image's tool files, the catalog's SHA-256 and every file the
catalog lists (a regular file of the listed size inside the media root), makes the row's files read-only, moves the
catalog out of the media root (it is the seed's, not the model's), and has the tools check their configuration.

A step that fails stops the sandbox. A row or an image that no retry can fix is refused with 422; any other failure
raises, and the Environment Server runs the episode again. The episode's close stops the sandbox. Grading never reads
the sandbox. Without ``sandbox_provider`` the server lends nothing and keeps no session state.
"""

import hashlib
import shlex
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BeforeValidator, Field, PrivateAttr, ValidationError, model_validator

from nemo_gym import PARENT_DIR
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxAccess, SandboxConfig, exec_as_root, upload_path
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    LentSandboxes,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    SimpleResourcesServer,
    reject_partial_typed_seed,
)
from resources_servers.video_mcqa.sandbox_image.video_tools.contract import ToolConfig, ToolSettings
from resources_servers.video_mcqa.task_data import FileBinding, PreloadedMedia, VideoBinding


SANDBOX_IMAGE_DIR = Path(__file__).parent / "sandbox_image"
# Where sandbox_image/Dockerfile puts the plugin and the video_tools program.
TOOLS_DIR = PurePosixPath("/opt/video-mcqa")
PLUGIN_PATH = TOOLS_DIR / "video_tools.mjs"
# The image's files that must match this repository, relative to TOOLS_DIR and to SANDBOX_IMAGE_DIR.
TOOL_FILES = (
    "video_tools.mjs",
    "video_tools/__init__.py",
    "video_tools/cli.py",
    "video_tools/contract.py",
    "video_tools/media.py",
)
# The tools' source indexes, outside the media root that the model's paths name.
CACHE_DIR = "/tmp/video-mcqa/index"
# Exit statuses of the seed check.
TOOLS_DIFFER, VIDEO_DIFFERS, CANNOT_PROTECT, TOOLS_FAIL, CATALOG_DIFFERS, MEDIA_DIFFERS = 10, 11, 12, 13, 14, 15
# Where the seed of a row with preloaded media keeps the image's catalog when it was inside the media root.
PRELOADED_CATALOG = "preloaded-catalog.json"
# Run by the image's Python with the catalog and the media root: every file the catalog lists is a regular file of
# the listed size inside the media root, else one line on stderr and status 1.
MEDIA_CHECK = r"""
import json, os, stat, sys
catalog, root = sys.argv[1], sys.argv[2].rstrip("/") + "/"
try:
    entries = [(video["path"], video["bytes"]) for video in json.load(open(catalog, encoding="utf-8"))["videos"]]
except (OSError, ValueError, KeyError, TypeError) as error:
    sys.exit(f"The catalog does not list videos with path and bytes: {error!r}")
if not entries:
    sys.exit("The catalog lists no videos")
for path, size in entries:
    if not isinstance(path, str) or os.path.normpath(path) != path or not path.startswith(root):
        sys.exit(f"The catalog lists {path!r}, which is not inside {root}")
    try:
        status = os.lstat(path)
    except OSError as error:
        sys.exit(f"Cannot read {path}: {error.strerror}")
    if not stat.S_ISREG(status.st_mode) or status.st_size != size:
        sys.exit(f"{path} is not a regular file of {size} bytes")
"""


def seed_check(*, video: str, tools: ToolConfig) -> str:
    """The seed's command in the sandbox: the image's tools are this repository's, the video is the row's; then the
    video becomes read-only and the tools check their configuration and FFmpeg."""
    check = shlex.join([*tools.command, "--config", "tools.json", "--check"])
    return (
        f"cd {shlex.quote(str(TOOLS_DIR))} || exit 1\n"
        f"sha256sum --check --strict --quiet tools.sha256 || exit {TOOLS_DIFFER}\n"
        f"sha256sum --check --strict --quiet video.sha256 || exit {VIDEO_DIFFERS}\n"
        f"chmod 0444 {shlex.quote(video)} || exit {CANNOT_PROTECT}\n"
        f"{check} >/dev/null || exit {TOOLS_FAIL}\n"
    )


def preloaded_seed_check(*, catalog: str, media_root: str, files: list[str], tools: ToolConfig) -> str:
    """The seed's command in the sandbox for preloaded media: the image's tools are this repository's and its catalog
    the row's, every file it lists is there; then the row's files become read-only, a catalog inside the media root
    moves next to the tools, and the tools check their configuration and FFmpeg."""
    check = shlex.join([*tools.command, "--config", "tools.json", "--check"])
    media_check = shlex.join(["python3", "-I", "-c", MEDIA_CHECK, catalog, media_root])
    lines = [
        f"cd {shlex.quote(str(TOOLS_DIR))} || exit 1",
        f"sha256sum --check --strict --quiet tools.sha256 || exit {TOOLS_DIFFER}",
        f"sha256sum --check --strict --quiet catalog.sha256 || exit {CATALOG_DIFFERS}",
        f"{media_check} || exit {MEDIA_DIFFERS}",
    ]
    if files:
        lines.append("sha256sum --check --strict --quiet files.sha256 || exit 1")
        lines.append(f"chmod 0444 {shlex.join(files)} || exit {CANNOT_PROTECT}")
    if PurePosixPath(catalog).is_relative_to(media_root):
        lines.append(f"mv {shlex.quote(catalog)} {PRELOADED_CATALOG} || exit 1")
    lines.append(f"{check} >/dev/null || exit {TOOLS_FAIL}")
    return "\n".join(lines) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sandbox_directory(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("must be an absolute sandbox directory other than /")
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError("must be an absolute sandbox directory other than /")
    return value


SandboxDirectory = Annotated[str, BeforeValidator(_sandbox_directory)]


class VideoSandboxImageConfig(SandboxConfig):
    # An OCI image built from sandbox_image/Dockerfile: the sandbox of a row with a video_binding.
    image: Optional[str] = None


class VideoSandboxServerConfig(BaseResourcesServerConfig):
    """The agentic setting's sandbox. Without ``sandbox_provider`` the server lends none."""

    sandbox_provider: Optional[str] = None
    sandbox_config: VideoSandboxImageConfig = Field(default_factory=VideoSandboxImageConfig)
    # OpenCode's working directory (WORKDIR of the image), inside sandbox_media_root so that the tools can read the
    # files the model writes there.
    sandbox_workdir: SandboxDirectory = "/testbed/work"
    # The tools' workspace: a row's video goes to sandbox_media_root/<video_binding.sandbox_path>.
    sandbox_media_root: SandboxDirectory = "/testbed"
    # model_only: the sandbox reaches only sandbox_model_server's host. inherit keeps the provider's network.
    sandbox_network_access: Literal["inherit", "model_only"] = "inherit"
    sandbox_model_server: Optional[ModelServerRef] = None
    # Host directories a row's video_binding.source and preloaded_media.files must be in; a seed reads no other file.
    # A relative root, like a relative source, starts at the Gym repository root.
    video_roots: list[Path] = Field(default_factory=list)
    # The images of rows with preloaded_media, by the name in the row's preloaded_media.image: images built from
    # sandbox_image/Dockerfile with the media added. A row whose name maps to no image is refused.
    sandbox_preloaded_images: dict[str, Optional[str]] = Field(default_factory=dict)
    # The video tools' frame profile (the model's grid, 28 pixels by default), limits and timeout.
    video_tools: ToolSettings = Field(default_factory=ToolSettings)
    # Seconds for the video's upload and for the seed check in the sandbox.
    video_upload_timeout_seconds: float = Field(default=900, gt=0)
    seed_check_timeout_seconds: float = Field(default=300, gt=0)

    @model_validator(mode="after")
    def lending_settings(self) -> "VideoSandboxServerConfig":
        if self.sandbox_provider is None:
            if self.sandbox_network_access != "inherit":
                raise ValueError("sandbox_network_access applies to a lent sandbox; set sandbox_provider")
            return self
        if not self.sandbox_config.image and not any(self.sandbox_preloaded_images.values()):
            raise ValueError(
                "Set sandbox_config.image to an image built from resources_servers/video_mcqa/sandbox_image, or "
                "sandbox_preloaded_images to such images with the rows' media added"
            )
        if not self.video_roots:
            raise ValueError("Set video_roots, the host directories of the videos that rows name")
        if not PurePosixPath(self.sandbox_workdir).is_relative_to(self.sandbox_media_root):
            raise ValueError("sandbox_workdir must be inside sandbox_media_root")
        self.tool_config()
        return self

    def tool_config(self) -> ToolConfig:
        """The tools.json that each seed writes to the sandbox."""
        return ToolConfig(workspace_root=self.sandbox_media_root, cache_dir=CACHE_DIR, **self.video_tools.model_dump())


class VideoSandboxServer(SimpleResourcesServer):
    """A video MCQA resources server that lends each agentic episode a sandbox with its video."""

    config: VideoSandboxServerConfig
    # The sandboxes of rows with a video_binding, from sandbox_config.image.
    _lent_sandboxes: LentSandboxes | None = PrivateAttr(default=None)
    # The sandboxes of rows with preloaded_media, by the image name the row gives.
    _preloaded_sandboxes: dict[str, LentSandboxes] = PrivateAttr(default_factory=dict)
    # The lender of each session a seed started, so that its close or a repeated seed reaches the same sandboxes.
    _lenders: dict[str, LentSandboxes] = PrivateAttr(default_factory=dict)
    # The episode and task each open session was seeded for, keyed by resources_session_id.
    _sessions: dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    _locks: ResourcesSessionLocks = PrivateAttr(default_factory=ResourcesSessionLocks)
    _fences: ResourcesSessionFences[EpisodeId] = PrivateAttr()
    _video_roots: list[Path] = PrivateAttr(default_factory=list)
    _seed_files: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._fences = ResourcesSessionFences(self.config.session_close_retry_window_seconds)
        if self.config.sandbox_provider is None:
            return
        model_server = self.config.sandbox_model_server

        def lender(image: str | None) -> LentSandboxes:
            return LentSandboxes(
                self.config.sandbox_provider,
                self.config.sandbox_config,
                image=image,
                workdir=self.config.sandbox_workdir,
                network_access=self.config.sandbox_network_access,
                model_server=model_server.name if model_server is not None else None,
            )

        self._lent_sandboxes = lender(self.config.sandbox_config.image)
        self._preloaded_sandboxes = {
            name: lender(image) for name, image in self.config.sandbox_preloaded_images.items() if image
        }
        self._video_roots = [(PARENT_DIR / root).resolve() for root in self.config.video_roots]
        self._seed_files = {
            str(TOOLS_DIR / "tools.json"): self.config.tool_config().model_dump_json(indent=2) + "\n",
            str(TOOLS_DIR / "tools.sha256"): "".join(
                f"{file_sha256(SANDBOX_IMAGE_DIR / name)}  {name}\n" for name in TOOL_FILES
            ),
        }

    def multiple_workers_unsupported_reason(self) -> str | None:
        if self._lent_sandboxes is not None:
            return "lent sandboxes live in one worker's memory"
        return super().multiple_workers_unsupported_reason()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    def _binding(self, task_data: Mapping[str, object]) -> tuple[VideoBinding, Path]:
        """The row's video binding and its source file, or a 422 refusal."""
        try:
            binding = VideoBinding.model_validate(task_data.get("video_binding"))
        except ValidationError as error:
            raise ResourcesSeedRejectedError(f"An agentic row needs a valid video_binding: {error}") from error
        return binding, self._source(binding, "video")

    def _source(self, binding: FileBinding, kind: str) -> Path:
        """The host file a binding names, or a 422 refusal."""
        try:
            source = (PARENT_DIR / binding.source).resolve(strict=True)
        except OSError as error:
            raise ResourcesSeedRejectedError(f"Cannot read the row's {kind} {binding.source}: {error}") from error
        if not source.is_file() or not any(source.is_relative_to(root) for root in self._video_roots):
            raise ResourcesSeedRejectedError(
                f"The row's {kind} is not a regular file under the configured video_roots"
            )
        return source

    def _preloaded(self, task_data: Mapping[str, object]) -> tuple[PreloadedMedia, LentSandboxes, dict[str, bytes]]:
        """The row's preloaded media, the lender of its image and the row's files by sandbox path, or a 422 refusal."""
        try:
            media = PreloadedMedia.model_validate(task_data.get("preloaded_media"))
        except ValidationError as error:
            raise ResourcesSeedRejectedError(f"An agentic row needs a valid preloaded_media: {error}") from error
        lender = self._preloaded_sandboxes.get(media.image)
        if lender is None:
            raise ResourcesSeedRejectedError(
                f"No image holds the row's preloaded media {media.image!r}: set it in sandbox_preloaded_images"
            )
        files = {}
        for binding in media.files:
            content = self._source(binding, "file").read_bytes()
            if hashlib.sha256(content).hexdigest() != binding.sha256:
                raise ResourcesSeedRejectedError(f"The row's file {binding.source} does not have its sha256")
            files[str(PurePosixPath(self.config.sandbox_media_root) / binding.sandbox_path)] = content
        return media, lender, files

    async def _stage_video(self, session_id: str, binding: VideoBinding, source: Path, target: str) -> None:
        """Upload the video and run the seed check; the caller stops the sandbox if this raises."""
        sandbox = self._lent_sandboxes.get(session_id)
        if sandbox is None:
            raise RuntimeError("The lent sandbox is gone before its video was staged")
        await upload_path(sandbox, source, target, timeout_s=self.config.video_upload_timeout_seconds)
        refusals = {VIDEO_DIFFERS: f"The video uploaded from {source} does not have the row's video_binding.sha256"}
        await self._check(sandbox, seed_check(video=target, tools=self.config.tool_config()), refusals)

    async def _check(self, sandbox: AsyncSandbox, command: str, refusals: Mapping[int, str]) -> None:
        """Run the seed's check in the sandbox: 422 for a status no retry can fix, RuntimeError for any other failure."""
        # exec_as_root raises for a set error_type, a result that did not run.
        result = await exec_as_root(sandbox, command, timeout_s=self.config.seed_check_timeout_seconds)
        # -1 is the sentinel of a command whose end the provider did not see (a signal), not an exit status.
        if result.return_code in (None, -1):
            raise RuntimeError("The sandbox did not report how the video check ended")
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        refusals = {
            TOOLS_DIFFER: "The sandbox image's video tools differ from this repository's; build the image from "
            "resources_servers/video_mcqa/sandbox_image",
            TOOLS_FAIL: "The video tools cannot run in the sandbox image",
            **refusals,
        }
        if result.return_code in refusals:
            raise ResourcesSeedRejectedError(f"{refusals[result.return_code]}: {detail}")
        if result.return_code != 0:
            raise RuntimeError(f"The sandbox video check failed with status {result.return_code}: {detail}")

    async def _seed_preloaded(
        self, session_id: str, task_id: TaskId, task_data: Mapping[str, object]
    ) -> SandboxAccess:
        """Start the sandbox from the image that holds the row's media and check it; nothing is uploaded."""
        media, lender, row_files = self._preloaded(task_data)
        media_root = self.config.sandbox_media_root
        files = {
            **self._seed_files,
            str(TOOLS_DIR / "catalog.sha256"): f"{media.catalog_sha256}  {media.catalog}\n",
            **row_files,
        }
        if row_files:
            files[str(TOOLS_DIR / "files.sha256")] = "".join(
                f"{binding.sha256}  {PurePosixPath(media_root) / binding.sandbox_path}\n" for binding in media.files
            )
        self._lenders[session_id] = lender
        access = await lender.lend(session_id, task_id, files=files)
        try:
            sandbox = lender.get(session_id)
            if sandbox is None:
                raise RuntimeError("The lent sandbox is gone before its media was checked")
            command = preloaded_seed_check(
                catalog=media.catalog, media_root=media_root, files=list(row_files), tools=self.config.tool_config()
            )
            refusals = {
                CATALOG_DIFFERS: f"The image {media.image!r} does not hold the row's catalog {media.catalog}",
                MEDIA_DIFFERS: f"The image {media.image!r} does not hold every file its catalog lists",
            }
            await self._check(sandbox, command, refusals)
        except BaseException:
            await lender.release(session_id)
            raise
        return access

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Lend the episode its sandbox when the server lends sandboxes; otherwise keep no state.

        A repeated seed gets the sandbox it already has. While the server lends sandboxes, a cookie seed is refused:
        an agent's own /run seeds that way and would run its harness outside the sandbox that holds the video.
        """
        if self._lent_sandboxes is None:
            return await super().seed_session(body)
        if not isinstance(body, ResourcesSeedSessionRequest):
            raise ResourcesSeedRejectedError(
                "This server lends a sandbox only to an Environment Server's episode seed"
            )
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            self._fences.raise_if_fenced(session_id)
            seeded = self._sessions.get(session_id)
            if seeded is not None and seeded != (body.episode_id, body.task_id):
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
            if seeded is not None:
                access = await self._lenders[session_id].lend(session_id, body.task_id)
            elif body.task_data.get("preloaded_media") is not None:
                if body.task_data.get("video_binding") is not None:
                    raise ResourcesSeedRejectedError("A row takes a video_binding or preloaded_media, not both")
                access = await self._seed_preloaded(session_id, body.task_id, body.task_data)
                self._sessions[session_id] = (body.episode_id, body.task_id)
            else:
                binding, source = self._binding(body.task_data)
                target = str(PurePosixPath(self.config.sandbox_media_root) / binding.sandbox_path)
                files = {**self._seed_files, str(TOOLS_DIR / "video.sha256"): f"{binding.sha256}  {target}\n"}
                self._lenders[session_id] = self._lent_sandboxes
                access = await self._lent_sandboxes.lend(session_id, body.task_id, files=files)
                try:
                    await self._stage_video(session_id, binding, source, target)
                except BaseException:
                    await self._lent_sandboxes.release(session_id)
                    raise
                self._sessions[session_id] = (body.episode_id, body.task_id)
            request.session[SESSION_ID_KEY] = session_id
            return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=access)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Stop the episode's sandbox. Safe to repeat, and for a session never seeded."""
        if self._lent_sandboxes is None:
            return await super().close_resources_session(body)
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            closed_episode = self._fences.get(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            seeded = self._sessions.get(session_id)
            if seeded is not None and seeded[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            # A failed stop raises before the fence, so the caller can close again.
            await self._lenders.get(session_id, self._lent_sandboxes).release(session_id)
            self._lenders.pop(session_id, None)
            self._sessions.pop(session_id, None)
            self._fences.fence(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def shutdown(self) -> None:
        if self._lent_sandboxes is not None:
            await self._lent_sandboxes.release_all()
        for lender in self._preloaded_sandboxes.values():
            await lender.release_all()
        self._lenders.clear()
        self._sessions.clear()
