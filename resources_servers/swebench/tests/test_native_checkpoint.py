# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native SWE ownership, checkpoint restore, and verification from a durable patch."""

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from omegaconf import DictConfig

from nemo_gym.base_resources_server import BaseSeedSessionRequest, BaseVerifyRequest
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.sandbox import owner as owner_module
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.scale_swe import app as scale
from resources_servers.swe_next import app as next_swe
from resources_servers.swe_rebench import app as rebench
from resources_servers.swebench.native_checkpoint import NativeSWESandboxConfig, NativeSWESandboxResourcesServer
from resources_servers.swebench.patch_capture import PatchCapture
from resources_servers.swemer_v1 import app as v1
from resources_servers.swemer_v2 import app as v2
from tests.unit_tests.test_sandbox_checkpoint import FakeSnapshotProvider


@dataclass
class Case:
    module: ModuleType
    server: type[NativeSWESandboxResourcesServer]
    config: type[NativeSWESandboxConfig]
    seed: type[BaseSeedSessionRequest]
    verify: type[BaseVerifyRequest]
    prepared: type[BaseVerifyRequest]


CASES = [
    Case(
        rebench,
        rebench.SWERebenchResourcesServer,
        rebench.SWERebenchResourcesServerConfig,
        rebench.SWERebenchSeedSessionRequest,
        rebench.SWERebenchVerifyRequest,
        rebench.SWERebenchPreparedVerifyRequest,
    ),
    Case(
        scale,
        scale.ScaleSWEResourcesServer,
        scale.ScaleSWEResourcesServerConfig,
        scale.ScaleSWESeedSessionRequest,
        scale.ScaleSWEVerifyRequest,
        scale.ScaleSWEPreparedVerifyRequest,
    ),
    Case(
        v1,
        v1.SwemerV1ResourcesServer,
        v1.SwemerV1ResourcesServerConfig,
        v1.SwemerV1SeedSessionRequest,
        v1.SwemerV1VerifyRequest,
        v1.SwemerV1PreparedVerifyRequest,
    ),
    Case(
        v2,
        v2.SwemerV2ResourcesServer,
        v2.SwemerV2ResourcesServerConfig,
        v2.SwemerV2SeedSessionRequest,
        v2.SwemerV2VerifyRequest,
        v2.SwemerV2PreparedVerifyRequest,
    ),
    Case(
        next_swe,
        next_swe.SWENextResourcesServer,
        next_swe.SWENextResourcesServerConfig,
        next_swe.SWENextSeedSessionRequest,
        next_swe.SWENextVerifyRequest,
        next_swe.SWENextPreparedVerifyRequest,
    ),
]
BODY = {
    "instance_id": "task-1",
    "repo": "org/repo",
    "workdir": "/repo",
    "base_commit": "dataset-base",
    "image_name": "image:1",
    "image_url": "image:1",
    "image_ref": "image:1",
    "test_framework": "pytest",
    "test_command": "pytest",
    "responses_create_params": {"input": []},
    "response": {
        "id": "r",
        "object": "response",
        "created_at": 0,
        "model": "m",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
}


class GitProvider(FakeSnapshotProvider):
    async def exec(self, handle: SandboxHandle, command: str, **kwargs: object) -> SandboxExecResult:
        result = await super().exec(handle, command, **kwargs)
        if "rev-parse HEAD" in command:
            return replace(result, stdout="original-base\n")
        elif "ls-files" in command:
            return replace(result, stdout="baseline.tmp\n")
        return result


@pytest.fixture(params=CASES, ids=[c.module.__name__.split(".")[-2] for c in CASES])
def case(request: pytest.FixtureRequest) -> Case:
    return request.param


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> GitProvider:
    provider = GitProvider()
    monkeypatch.setattr(owner_module, "resolve_provider_config", lambda *_: provider)
    return provider


def make_server(
    case: Case, provider: GitProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, golden: bool = False
) -> NativeSWESandboxResourcesServer:
    global_config = DictConfig(
        {
            "checkpoint": {
                "enabled": True,
                "control_auth_token": "test",
                "state_dir": str(tmp_path / "state"),
            }
        }
    )
    monkeypatch.setattr(owner_module, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(case.module, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(case.module, "resolve_provider_config", lambda *_: provider)
    monkeypatch.setattr(case.module, "resolve_provider_metadata", lambda *_: {})
    config = case.config(
        host="127.0.0.1",
        port=0,
        entrypoint="",
        name="native",
        num_workers=1,
        sandbox_provider="sandbox",
        sandbox_config={},
        apply_anti_cheating=False,
        is_verifying_golden_patch=golden,
        inconclusive_verification_retries=0,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    return case.server(config=config, server_client=client)


def request() -> Request:
    return Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "session"}, "app": FastAPI()})


async def test_snapshot_restore_keeps_checkout_and_verification_metadata(
    case: Case,
    provider: GitProvider,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = make_server(case, provider, tmp_path, monkeypatch)
    assert server.checkpoint_mode_in_effect() == "exported"
    seed = case.seed.model_validate(BODY)
    seeded = await server.seed_session(request(), seed)
    original = server.require_session_sandbox("session")
    if case.module is next_swe:
        commands = provider.files(original.handle.sandbox_id)
        link = commands.index("ln -sfn /workspace /repo")
        init = next(i for i, command in enumerate(commands) if "git init -q" in command)
        assert link < init  # SWE-Next images need this link before any repository setup.
    await original.exec("agent patch before checkpoint")
    states = await server.export_session_states(["session"], checkpoint_id="cut-1")
    assert states["session"]["data"]["pristine_untracked"] == ["baseline.tmp"]
    if case.module is not rebench:
        assert states["session"]["data"]["base_commit"] == "original-base"
    await original.exec("late agent edit")

    restored = make_server(case, provider, tmp_path, monkeypatch)
    await restored.restore_session_states(json.loads(json.dumps(states)))
    fork = restored.require_session_sandbox("session")
    assert fork.handle.sandbox_id != seeded.sandbox_handle
    assert "agent patch before checkpoint" in provider.files(fork.handle.sandbox_id)
    assert "late agent edit" not in provider.files(fork.handle.sandbox_id)
    commands_before = list(provider.files(fork.handle.sandbox_id))
    reseeded = await restored.seed_session(request(), seed)
    assert reseeded.sandbox_handle == fork.handle.sandbox_id
    assert provider.files(fork.handle.sandbox_id) == commands_before  # No reset, pre-commands, or history scrub.
    access = await restored.current_sandbox_access("session")
    assert access is not None and access.workdir == "/repo"

    capture = AsyncMock(return_value=PatchCapture.static("generated patch", "worktree", "worktree"))
    monkeypatch.setattr(case.module, "capture_model_patch", capture)
    prepared = await restored.prepare_verify(request(), case.verify.model_validate(BODY))
    assert prepared.capture.patch == "generated patch"
    assert capture.await_args.kwargs["pristine_untracked"] == frozenset({"baseline.tmp"})
    assert capture.await_args.args[2] == ("dataset-base" if case.module is rebench else "original-base")
    assert restored.session_sandbox("session") is None
    assert provider.snapshots  # Ending the rollout must not delete snapshots retained by a checkpoint.
    await server.retire_session_state("session")


async def test_prepared_patch_is_idempotent_and_verification_replays_without_agent_sandbox(
    case: Case,
    provider: GitProvider,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = make_server(case, provider, tmp_path, monkeypatch)
    await server.seed_session(request(), case.seed.model_validate(BODY))
    capture = AsyncMock(return_value=PatchCapture.static("generated patch", "worktree", "worktree"))
    monkeypatch.setattr(case.module, "capture_model_patch", capture)
    body = case.verify.model_validate(BODY)
    first, second = await asyncio.gather(
        server.prepare_verify(request(), body), server.prepare_verify(request(), body)
    )
    assert capture.await_count == 1
    assert first == second
    assert server.session_sandbox("session") is None
    wire = json.loads(json.dumps(first.model_dump(mode="json")))
    prepared_body = case.prepared.model_validate(body.model_dump(mode="json") | wire)
    result = case.module.VerificationResult(
        completed=True, resolved=True, patch_applied=True, test_results={}, test_output="ok", error=None
    )
    grade = AsyncMock(return_value=result)
    monkeypatch.setattr(case.module, "run_verification", grade)
    if case.module is rebench:
        monkeypatch.setattr(rebench, "resolve_parser", lambda *_args, **_kwargs: lambda _: {})
    monkeypatch.setattr(
        case.server, "require_session_sandbox", MagicMock(side_effect=AssertionError("agent state used"))
    )
    first_verdict = await server.verify_prepared(request(), prepared_body)
    second_verdict = await server.verify_prepared(request(), prepared_body)
    assert first_verdict.reward == second_verdict.reward == 1.0
    assert first_verdict.evaluation_completed and first_verdict.patch_source == "worktree"
    assert "capture" not in first_verdict.model_dump()
    assert grade.await_args_list[0].kwargs["inputs"].patch == "generated patch"
    assert grade.await_args_list[0].kwargs["log_dir"] != grade.await_args_list[1].kwargs["log_dir"]


async def test_preparation_retry_keeps_patch_if_cancelled_after_sandbox_shutdown(
    case: Case,
    provider: GitProvider,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = make_server(case, provider, tmp_path, monkeypatch)
    await server.seed_session(request(), case.seed.model_validate(BODY))
    capture = AsyncMock(return_value=PatchCapture.static("saved patch", "worktree", "worktree"))
    monkeypatch.setattr(case.module, "capture_model_patch", capture)
    end = case.server.end_session_sandbox
    stopped = asyncio.Event()
    calls = 0

    async def stop_and_hold(self, session_id: str) -> None:
        nonlocal calls
        await end(self, session_id)
        calls += 1
        if calls == 1:
            stopped.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(case.server, "end_session_sandbox", stop_and_hold)
    body = case.verify.model_validate(BODY)
    first = asyncio.create_task(server.prepare_verify(request(), body))
    await asyncio.wait_for(stopped.wait(), timeout=5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert server.session_sandbox("session") is None
    retried = await server.prepare_verify(request(), body)
    assert retried.capture.patch == "saved patch"
    assert retried.extraction_error is None
    assert capture.await_count == 1


def test_golden_patch_instance_holds_no_checkpointed_agent_state(
    case: Case,
    provider: GitProvider,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = make_server(case, provider, tmp_path, monkeypatch, golden=True)
    assert server.checkpoint_mode_in_effect() == "restart_only"
    assert "/verify_prepared" in server.checkpoint_replayable_paths
    schema = server.setup_webserver().openapi()
    assert "/prepare_verify" in schema["paths"] and "/verify_prepared" in schema["paths"]
