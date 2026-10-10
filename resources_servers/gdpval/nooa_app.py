# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources-owned GDP task sandboxes, with the existing GDP verifier unchanged."""

import asyncio
import hashlib
import json
import logging
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from urllib.parse import urlsplit

from aiohttp import ClientTimeout
from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, PrivateAttr, SecretStr, field_validator

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from nemo_gym.server_utils import request as http_request
from resources_servers.gdpval.app import (
    GDPValResourcesServer,
    GDPValResourcesServerConfig,
    GDPValVerifyRequest,
    GDPValVerifyResponse,
    _safe_output_text,
)
from resources_servers.gdpval.comparison import IGNORE_FILES
from resources_servers.gdpval.nooa_tasks import INPUT_DIR, OUTPUT_DIR, WORKDIR, GDPFileTask, relative_file


LOG = logging.getLogger(__name__)


def _write_receipt(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


_LIST_OUTPUTS = """
import hashlib, json, os, pathlib, stat
root = pathlib.Path(__OUTPUT_DIR__)
if root.is_symlink() or not root.is_dir():
    raise RuntimeError('GDP output directory is missing or a symlink')
files = []
total = 0
def fail_listing(error):
    raise error
for directory, subdirectories, filenames in os.walk(root, followlinks=False, onerror=fail_listing):
    parent = pathlib.Path(directory)
    subdirectories.sort()
    for name in subdirectories:
        if not stat.S_ISDIR((parent / name).lstat().st_mode):
            raise RuntimeError('GDP output directories must not be symbolic links')
    for name in sorted(filenames):
        path = parent / name
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError('GDP outputs must be regular, non-linked files')
        total += info.st_size
        if len(files) >= __MAX_FILES__ or total > __MAX_BYTES__:
            raise RuntimeError('GDP deliverable limit exceeded')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files.append({'name': path.relative_to(root).as_posix(), 'size': info.st_size, 'sha256': digest})
print(json.dumps(files))
"""


class NOOAGDPConfig(GDPValResourcesServerConfig):
    """An explicitly selected audited GDP image and durable host artifact directory."""

    sandbox_provider: str = "sandbox"
    image: str = Field(min_length=1)
    deliverables_root: Path
    num_workers: int = 1
    hf_token: SecretStr | None = None
    execute_only: bool = False
    max_reference_bytes: int = Field(default=4 * 1024**3, gt=0)
    max_deliverable_bytes: int = Field(default=2 * 1024**3, gt=0)
    max_deliverable_files: int = Field(default=100, gt=0)
    sandbox_stop_timeout: float = Field(default=60, gt=0)

    @field_validator("deliverables_root")
    @classmethod
    def absolute_output(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("deliverables_root must be absolute")
        return value

    @field_validator("num_workers")
    @classmethod
    def one_worker(cls, value: int) -> int:
        if value != 1:
            raise ValueError("GDP sandbox sessions require num_workers=1")
        return value


@dataclass
class _Session:
    seed: ResourcesSeedSessionRequest
    sandbox: AsyncSandbox
    directory: Path
    ready: bool = False
    failed: bool = False
    stopped: bool = False
    deliverables: Path | None = None
    verification_key: str | None = None
    verdict: GDPValVerifyResponse | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class GDPGenerationResponse(GDPValVerifyResponse):
    """An explicitly ungraded artifact receipt, masked from quality metrics."""

    execute_only: bool = True
    generation_manifest: str


class NOOAGDPResourcesServer(GDPValResourcesServer):
    """Export NOOA deliverables for the existing judge; retain ownership until close."""

    config: NOOAGDPConfig
    _sessions: dict[str, _Session] = PrivateAttr(default_factory=dict)
    _closed: dict[str, EpisodeId] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent(app) as state:
                    yield state
            finally:
                for session in list(self._sessions.values()):
                    try:
                        async with session.lock:
                            await self._stop(session)
                    except Exception:
                        LOG.exception("Failed to stop GDP sandbox during shutdown")

        app.router.lifespan_context = lifespan
        return app

    async def _stop(self, session: _Session) -> None:
        if not session.stopped:
            async with asyncio.timeout(self.config.sandbox_stop_timeout):
                await session.sandbox.stop()
            session.stopped = True

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", session_id):
            raise HTTPException(422, "Invalid resources_session_id")
        task = GDPFileTask.model_validate(body.task_data)
        if task.task_id != body.task_id.task_id:
            raise HTTPException(422, "Task ID does not match task_data")
        if session_id in self._closed:
            raise HTTPException(409, "Resources session is already closed")
        session = self._sessions.get(session_id)
        if session is None:
            provider = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
            sandbox = AsyncSandbox(provider)
            self.config.deliverables_root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="gdp-", dir=self.config.deliverables_root))
            session = _Session(body.model_copy(deep=True), sandbox, directory)
            self._sessions[session_id] = session
        if session.seed != body:
            raise HTTPException(409, "Resources session is already bound to another request")
        async with session.lock:
            if session_id in self._closed or session.stopped or session.failed:
                raise HTTPException(409, "Resources session is closed or failed; create a new episode")
            try:
                if not session.ready:
                    await session.sandbox.start(SandboxSpec(image=self.config.image, workdir=WORKDIR))
                    result = await session.sandbox.exec(f"mkdir -p {INPUT_DIR} {OUTPUT_DIR}", timeout_s=30)
                    if result.return_code != 0:
                        raise RuntimeError("Could not prepare GDP sandbox directories")
                    await self._stage_references(session, task)
                    session.ready = True
                descriptor = await session.sandbox.serialize()
            except BaseException:
                session.failed = True
                try:
                    await self._stop(session)
                except BaseException:
                    # Ownership remains reachable for the Environment Server's cleanup retry.
                    LOG.exception("GDP seed cleanup failed; retaining session %s", session_id)
                raise
            request.session[SESSION_ID_KEY] = session_id
            return ResourcesSeedSessionResponse(
                resources_session_id=session_id,
                sandbox_access=SandboxAccess(
                    connection=DirectSandboxConnection(
                        provider_config_ref=self.config.sandbox_provider, descriptor=descriptor
                    ),
                    workdir=WORKDIR,
                ),
            )

    async def _stage_references(self, session: _Session, task: GDPFileTask) -> None:
        total = 0
        for name, url in zip(task.reference_files, task.reference_file_urls, strict=True):
            local = session.directory / "reference_files" / name.removeprefix("reference_files/")
            local.parent.mkdir(parents=True, exist_ok=True)
            headers = {}
            if self.config.hf_token and urlsplit(url).hostname in {"huggingface.co", "hf.co"}:
                headers["Authorization"] = f"Bearer {self.config.hf_token.get_secret_value()}"
            response = await http_request("GET", url, headers=headers, timeout=ClientTimeout(total=180))
            try:
                response.raise_for_status()
                with local.open("wb") as stream:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        total += len(chunk)
                        if total > self.config.max_reference_bytes:
                            raise RuntimeError("GDP reference-file total exceeds max_reference_bytes")
                        stream.write(chunk)
            finally:
                response.release()
            # The original host bytes remain outside the agent's sandbox for judging.
            await session.sandbox.upload(local, f"{INPUT_DIR}/{name}")

    async def _export_deliverables(self, session: _Session) -> Path:
        if session.deliverables is not None:
            return session.deliverables
        script = (
            _LIST_OUTPUTS.replace("__OUTPUT_DIR__", repr(OUTPUT_DIR))
            .replace("__MAX_FILES__", str(self.config.max_deliverable_files))
            .replace("__MAX_BYTES__", str(self.config.max_deliverable_bytes))
        )
        result = await session.sandbox.exec(f"python3 -c {quote(script)}", timeout_s=180)
        if result.return_code != 0:
            raise HTTPException(503, "GDP artifact export failed: " + (result.stderr or "listing failed")[-1000:])
        files = json.loads(result.stdout or "null")
        if not isinstance(files, list) or len(files) > self.config.max_deliverable_files:
            raise HTTPException(503, "Invalid GDP artifact listing")
        target = Path(tempfile.mkdtemp(prefix="deliverables-", dir=session.directory))
        total = 0
        seen = set()
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise HTTPException(503, "Invalid GDP artifact entry")
            name = relative_file(item["name"])
            size = item.get("size")
            digest = item.get("sha256")
            if (
                name in seen
                or Path(name).parts[0] in IGNORE_FILES
                or type(size) is not int
                or size < 0
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise HTTPException(503, "Invalid or reserved GDP artifact entry")
            seen.add(name)
            total += size
            if total > self.config.max_deliverable_bytes:
                raise HTTPException(503, "GDP artifact size limit exceeded")
            local = target / name
            local.parent.mkdir(parents=True, exist_ok=True)
            await session.sandbox.download(f"{OUTPUT_DIR}/{name}", local)
            if local.is_symlink() or not local.is_file() or local.stat().st_size != size:
                raise HTTPException(503, "GDP artifact changed during export")
            with local.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise HTTPException(503, "GDP artifact changed during export")
        references = session.directory / "reference_files"
        if references.exists():
            await asyncio.to_thread(shutil.copytree, references, target / "reference_files")
        _write_receipt(session.directory / "artifacts.json", json.dumps(files, indent=2))
        session.deliverables = target
        return target

    async def verify(
        self, request: Request, body: GDPValVerifyRequest
    ) -> GDPValVerifyResponse | GDPGenerationResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        session = self._sessions.get(session_id)
        if session is None or body.task_id != session.seed.task_id.task_id:
            raise HTTPException(409, "Verification does not match a GDP session")
        key = body.model_dump_json()
        async with session.lock:
            if session_id in self._closed or not session.ready or session.failed:
                raise HTTPException(409, "No ready GDP sandbox for this session")
            if session.verification_key is not None and session.verification_key != key:
                raise HTTPException(409, "Verification request changed for this episode")
            session.verification_key = key
            if session.verdict is not None:
                return session.verdict
            target = await self._export_deliverables(session)
            task = GDPFileTask.model_validate(session.seed.task_data)
            payload = GDPValVerifyRequest.model_validate(
                session.seed.task_data
                | {
                    "reference_file_urls": task.reference_file_urls,
                    "responses_create_params": body.responses_create_params,
                    "response": body.response,
                    "deliverables_dir": str(target),
                }
            )
            generation_manifest = session.directory / "generation.json"
            _write_receipt(
                generation_manifest,
                json.dumps(
                    {
                        "schema_version": 1,
                        "episode_id": session.seed.episode_id.model_dump(mode="json"),
                        "task_id": session.seed.task_id.model_dump(mode="json"),
                        "resources_session_id": session_id,
                        "verify_request": payload.model_dump(mode="json"),
                    },
                    indent=2,
                ),
            )
            # Existing GDP judging recognizes this completion marker. It records
            # NOOA's successful return, not an invented Stirrup finish-tool call.
            _write_receipt(
                target / "finish_params.json",
                json.dumps(
                    {
                        "summary": _safe_output_text(body.response),
                        "paths": [
                            str(path)
                            for path in sorted(target.rglob("*"))
                            if path.is_file() and path.relative_to(target).parts[0] not in IGNORE_FILES
                        ],
                        "submission_method": "nooa_final_response_output_directory",
                    },
                    indent=2,
                ),
            )
            # NOOA finish precedes verification; agent close must confirm worker
            # cleanup before Resources stops this sandbox. Judge only exported
            # host bytes and never mount private judge references into the sandbox.
            if self.config.execute_only:
                session.verdict = GDPGenerationResponse(
                    **payload.model_dump(),
                    reward=0.0,
                    mask_sample=True,
                    failure_reason="Generation only: artifacts saved, no judge score requested",
                    generation_manifest=str(generation_manifest),
                )
                return session.verdict
            verdict = await super().verify(payload)
            if verdict.invalid_judge_response:
                raise HTTPException(503, "GDP judge did not return a valid verdict")
            session.verdict = verdict
            # Persist the actual verifier result for recovery without another rollout.
            (session.directory / "verdict.json").write_text(verdict.model_dump_json())
            return verdict

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        if self.config.execute_only:
            return AggregateMetrics(
                agent_metrics={
                    "generation/exported": sum(bool(row.get("execute_only")) for row in body.verify_responses)
                }
            )
        return await super().aggregate_metrics(body)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        closed = self._closed.get(session_id)
        if closed is not None and closed != body.episode_id:
            raise HTTPException(409, "Close episode does not match")
        session = self._sessions.get(session_id)
        if session is not None:
            if session.seed.episode_id != body.episode_id:
                raise HTTPException(409, "Close episode does not match")
            async with session.lock:
                await self._stop(session)
                self._sessions.pop(session_id, None)
                self._closed[session_id] = body.episode_id
        else:
            # Fence a delayed seed after its caller timed out before receiving it.
            self._closed[session_id] = body.episode_id
        request.session.pop(SESSION_ID_KEY, None)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)


if __name__ == "__main__":
    NOOAGDPResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NOOAGDPResourcesServer.run_webserver()
