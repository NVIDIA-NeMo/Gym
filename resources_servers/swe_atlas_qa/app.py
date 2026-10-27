# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run SWE Atlas Q&A with any harness that attaches to a resource-owned sandbox."""

import asyncio
import json
import logging
import tempfile
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from omegaconf import ListConfig
from pydantic import Field, PrivateAttr, SecretStr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.base import ConnectableProvider, SandboxHandle, SandboxSpec
from nemo_gym.sandbox.providers.registry import create_provider
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.swe_atlas_qa.prepare import TASKS_DIR, load_task, validate_source


LOG = logging.getLogger(__name__)


class SWEAtlasQAResourcesServerConfig(BaseResourcesServerConfig):
    """Task assets, sandbox provider, and an OpenAI-compatible rubric judge."""

    REVERIFY_MODE = ReverifyMode.UNSUPPORTED
    tasks_dir: Path = TASKS_DIR
    sandbox_provider: str
    model_server: ModelServerRef
    sandbox_model_base_url: str | None = None
    sandbox_ttl_s: int = Field(default=14400, gt=0)
    judge_base_url: str
    judge_model: str
    judge_api_key: SecretStr = SecretStr("unused")
    verifier_image: str = (
        "docker.io/library/python@sha256:8fef26df932191825664e4957ff488c96dfe64918327634a357a55facbc994d3"
    )
    max_concurrent_verifiers: int = Field(default=8, ge=1)
    max_answer_bytes: int = Field(default=2 * 1024 * 1024, ge=1)


class SWEAtlasQARequest(BaseRunRequest):
    task_id: str


class SWEAtlasQASeedSessionResponse(BaseSeedSessionResponse):
    sandbox_descriptor: dict[str, Any]


class SWEAtlasQAVerifyRequest(SWEAtlasQARequest, BaseVerifyRequest):
    pass


class SWEAtlasQAVerifyResponse(BaseVerifyResponse):
    task_id: str
    evaluation_completed: bool
    answer: str | None = None
    agg_score: float = 0.0
    num_rubrics: int = 0
    num_scored: int = 0
    num_unscored: int = 0
    num_passed: int = 0
    all_rubrics_scored: bool = False


class SWEAtlasQAResourcesServer(SimpleResourcesServer):
    """Keep task execution and private upstream rubric evaluation in separate sandboxes."""

    config: SWEAtlasQAResourcesServerConfig
    _sessions: dict[str, tuple[str, SandboxHandle]] = PrivateAttr(default_factory=dict)
    _closed_sessions: set[str] = PrivateAttr(default_factory=set)
    _session_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=lambda: defaultdict(asyncio.Lock))
    _sandboxes: dict[str, SandboxHandle] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        validate_source(self.config.tasks_dir.parent.parent)
        global_config = self.server_client.global_config_dict
        self._provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        if not isinstance(self._provider, ConnectableProvider):
            raise TypeError("SWE Atlas Q&A requires a provider that supports resource-owned sandboxes")
        self._metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        self._verifier_slots = asyncio.Semaphore(self.config.max_concurrent_verifiers)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                try:
                    results = await asyncio.gather(
                        *(self._close(handle) for handle in list(self._sandboxes.values())), return_exceptions=True
                    )
                    for result in results:
                        if isinstance(result, BaseException):
                            raise result
                finally:
                    self._sessions.clear()
                    await self._provider.aclose()

        app.router.lifespan_context = lifespan
        return app

    async def _close(self, handle: SandboxHandle) -> None:
        close = asyncio.create_task(asyncio.wait_for(self._provider.close(handle), timeout=60))
        try:
            await asyncio.shield(close)
        except asyncio.CancelledError:
            try:
                await close
            except Exception as error:
                LOG.warning("Sandbox cleanup failed during cancellation (%s)", type(error).__name__)
            else:
                self._sandboxes.pop(handle.sandbox_id, None)
            raise
        self._sandboxes.pop(handle.sandbox_id, None)

    async def seed_session(self, request: Request, body: SWEAtlasQARequest) -> SWEAtlasQASeedSessionResponse:
        task = load_task(self.config.tasks_dir, body.task_id)
        session_id = request.session[SESSION_ID_KEY]
        async with self._session_locks[session_id]:
            if session_id in self._closed_sessions:
                raise ValueError("This SWE Atlas Q&A session has already closed")
            if session_id in self._sessions:
                task_id, handle = self._sessions[session_id]
                if task_id != body.task_id:
                    raise ValueError("The session was seeded for a different task")
                descriptor = await self._provider.serialize_handle(handle)
                return SWEAtlasQASeedSessionResponse(sandbox_descriptor=descriptor)

            model_url = self.config.sandbox_model_base_url
            if not model_url:
                model_config = get_first_server_config_dict(
                    self.server_client.global_config_dict, self.config.model_server.name
                )
                # harness_agent calls the Gym model server, not the policy endpoint behind it (`base_url`).
                model_url = self.server_client._build_server_base_url(model_config)
            if isinstance(model_url, (list, ListConfig)):
                model_url = model_url[0]
            model_host = urlsplit(str(model_url)).hostname
            if not model_host or model_host in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "::"}:
                raise ValueError(
                    "Configure a model server host reachable from the task sandbox (use_absolute_ip=true)"
                )
            environment = task["environment"]
            spec = SandboxSpec(
                image=environment["docker_image"],
                workdir="/app",
                entrypoint=["/bin/sh", "-c", "sleep infinity"],
                ttl_s=self.config.sandbox_ttl_s,
                resources={
                    "cpu": environment["cpus"],
                    "memory_mib": environment["memory_mb"],
                    "disk_gib": (environment["storage_mb"] + 1023) // 1024,
                },
                metadata=self._metadata | {"benchmark": "swe-atlas-qa"},
                provider_options={
                    "network_policy": {
                        "defaultAction": "deny",
                        "egress": [{"action": "allow", "target": model_host}],
                    }
                },
            )
            handle = await self._provider.create(spec)
            self._sandboxes[handle.sandbox_id] = handle
            try:
                descriptor = await self._provider.serialize_handle(handle)
            except BaseException:
                await self._close(handle)
                raise
            self._sessions[session_id] = (body.task_id, handle)
            return SWEAtlasQASeedSessionResponse(sandbox_descriptor=descriptor)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        cookie_id = request.session.get(SESSION_ID_KEY)
        if cookie_id != session_id and cookie_id in self._sessions:
            raise ValueError("The session cookie conflicts with resources_session_id")
        async with self._session_locks[session_id]:
            if session_id in self._sessions:
                handle = self._sessions[session_id][1]
                try:
                    await self._close(handle)
                finally:
                    if handle.sandbox_id not in self._sandboxes:
                        del self._sessions[session_id]
                        self._closed_sessions.add(session_id)
            self._closed_sessions.add(session_id)
            return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def _grade(self, task_id: str, answer_path: Path, directory: Path) -> dict[str, Any]:
        task = load_task(self.config.tasks_dir, task_id)
        async with self._verifier_slots:
            verifier = await self._provider.create(
                SandboxSpec(
                    image=self.config.verifier_image,
                    entrypoint=["/bin/sh", "-c", "sleep infinity"],
                    ttl_s=1800,
                    resources={"cpu": 2, "memory_mib": 2048, "disk_gib": 5},
                    metadata=self._metadata | {"benchmark": "swe-atlas-qa-verifier"},
                )
            )
            self._sandboxes[verifier.sandbox_id] = verifier
            try:
                setup = await self._provider.exec(
                    verifier,
                    "mkdir -p /tests /logs/agent /logs/verifier && "
                    "python -m pip install --disable-pip-version-check --quiet openai==2.7.2",
                    timeout_s=300,
                )
                if setup.return_code != 0:
                    raise RuntimeError("Could not install the upstream verifier's dependencies")
                tests_dir = self.config.tasks_dir / task_id / "tests"
                for name in (
                    "evaluate_answer.py",
                    "rubrics.json",
                    "prompt.txt",
                    "system_prompt.txt",
                    "user_prompt_template.txt",
                ):
                    await self._provider.upload_file(verifier, tests_dir / name, f"/tests/{name}")
                await self._provider.upload_file(verifier, answer_path, "/logs/agent/answer.txt")
                result = await self._provider.exec(
                    verifier,
                    "python /tests/evaluate_answer.py",
                    env={
                        "EVAL_BASE_URL": self.config.judge_base_url,
                        "EVAL_MODEL": self.config.judge_model,
                        "EVAL_API_KEY": self.config.judge_api_key.get_secret_value(),
                    },
                    timeout_s=task["verifier"]["timeout_sec"],
                )
                if result.return_code != 0:
                    raise RuntimeError("The upstream rubric evaluator failed")
                results_path, reward_path = directory / "evaluation_results.json", directory / "reward.txt"
                await self._provider.download_file(verifier, "/logs/verifier/evaluation_results.json", results_path)
                await self._provider.download_file(verifier, "/logs/verifier/reward.txt", reward_path)
                results = json.loads(results_path.read_text())
                reward = float(reward_path.read_text())
                if reward not in (0.0, 1.0) or reward != results["reward"]:
                    raise ValueError("The upstream verifier returned inconsistent rewards")
                if (
                    results["num_rubrics"] <= 0
                    or not 0 <= results["num_passed"] <= results["num_scored"] <= results["num_rubrics"]
                    or results["num_unscored"] != results["num_rubrics"] - results["num_scored"]
                    or not 0 <= results["agg_score"] <= 1
                ):
                    raise ValueError("The upstream verifier returned inconsistent rubric counts")
                if results["num_scored"] == 0:
                    raise RuntimeError("The upstream verifier did not score any rubrics")
                scores = {
                    key: results[key]
                    for key in ("reward", "agg_score", "num_rubrics", "num_scored", "num_unscored", "num_passed")
                }
                scores["all_rubrics_scored"] = results["num_unscored"] == 0
                return scores
            finally:
                await self._close(verifier)

    async def verify(self, request: Request, body: SWEAtlasQAVerifyRequest) -> SWEAtlasQAVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self._session_locks[session_id]:
            session = None if session_id in self._closed_sessions else self._sessions.pop(session_id, None)
            self._closed_sessions.add(session_id)
        result: dict[str, Any] = {"reward": 0.0, "evaluation_completed": True}
        if session is None:
            result.update(
                mask_sample=True, evaluation_completed=False, failure_reason="Task sandbox session is missing"
            )
        else:
            task_id, handle = session
            try:
                with tempfile.TemporaryDirectory() as temporary_dir:
                    directory = Path(temporary_dir)
                    answer_path = directory / "answer.txt"
                    try:
                        if task_id != body.task_id:
                            raise ValueError("Verification task does not match the seeded task")
                        answer_file = await self._provider.exec(
                            handle,
                            "test -f /logs/agent/answer.txt && test ! -L /logs/agent/answer.txt && "
                            f'test "$(wc -c < /logs/agent/answer.txt)" -le {self.config.max_answer_bytes}',
                            timeout_s=30,
                        )
                        if answer_file.error_type is not None or answer_file.return_code not in (0, 1):
                            raise RuntimeError("Agent answer inspection failed")
                        if answer_file.return_code == 0:
                            await self._provider.download_file(handle, "/logs/agent/answer.txt", answer_path)
                    finally:
                        # No agent process can run while private rubric files or judge credentials are staged.
                        try:
                            await self._close(handle)
                        finally:
                            if handle.sandbox_id in self._sandboxes:
                                self._sessions[session_id] = session
                    if answer_path.exists():
                        if answer_path.stat().st_size > self.config.max_answer_bytes:
                            raise ValueError("Agent answer exceeds the configured size limit")
                        result["answer"] = answer_path.read_text(errors="replace")
                        answer_text = result["answer"].strip()
                        if "<<FINAL_ANSWER>>" in answer_text:
                            answer_text = answer_text.split("<<FINAL_ANSWER>>")[1].strip()
                        if answer_text:
                            result.update(await self._grade(body.task_id, answer_path, directory))
            except Exception as error:
                LOG.warning("SWE Atlas Q&A verification failed (%s)", type(error).__name__)
                result.update(
                    reward=0.0,
                    evaluation_completed=False,
                    mask_sample=True,
                    failure_reason=f"SWE Atlas Q&A verification failed: {type(error).__name__}",
                )
        return SWEAtlasQAVerifyResponse.model_validate(body.model_dump() | result)


if __name__ == "__main__":
    SWEAtlasQAResourcesServer.run_webserver()
