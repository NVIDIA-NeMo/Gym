# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint and replay SWE verification using immutable patches and the real HTTP participant."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import SandboxExecResult
from resources_servers.swebench import swebench_patches
from resources_servers.swebench.app import SWEBenchVerifyRequest
from resources_servers.swebench.tests.test_checkpoint import (
    AUTH,
    SEED,
    VERIFY,
    SnapshotProvider,
    control,
    make_server,
    ops,
)
from resources_servers.swebench.tests.test_checkpoint import provider as provider


async def test_preparation_retries_return_the_captured_patch_after_the_original_sandbox_is_stopped(
    provider: SnapshotProvider,
    tmp_path: Path,
) -> None:
    server, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        first = await client.post("/prepare_verify", json=VERIFY)
        second = await client.post("/prepare_verify", json=VERIFY)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"model_patch": "diff --git a/f b/f\n+fixed\n"}
    assert ops(provider, "create") == ["sb-1"], "capture must not create a verifier sandbox"
    assert provider.boxes["sb-1"]["state"] == "stopped"
    assert server._checkpointer().session_ids == []
    assert server._checkpoint.readiness().counts["sessions"] == 0
    assert sum("git --no-pager diff" in cmd for cmd in provider.boxes["sb-1"]["commands"]) == 1


async def test_failed_patch_capture_keeps_the_original_sandbox_available(
    provider: SnapshotProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        [session_id] = server._checkpointer().session_ids
        monkeypatch.setattr(
            provider,
            "exec",
            AsyncMock(return_value=SandboxExecResult(stdout="fatal: git failed", stderr=None, return_code=1)),
        )
        with pytest.raises(RuntimeError, match="Failed to capture the generated patch"):
            await server._prepare_verification_patch(session_id, SWEBenchVerifyRequest.model_validate(VERIFY))
    assert provider.boxes["sb-1"]["state"] == "running"
    assert server._checkpointer().session_ids == [session_id] and server._prepared_patches == {}


async def test_legacy_verification_still_blocks_prepare(
    provider: SnapshotProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    async def evaluate(**kwargs):
        started.set()
        await finish.wait()
        return {"resolved": True, "completed": True}

    monkeypatch.setattr("resources_servers.swebench.app.run_instance", evaluate)
    _, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        pending = asyncio.create_task(client.post("/ng-rollout/r-a1/verify", json=VERIFY))
        try:
            await asyncio.wait_for(started.wait(), 5)
            prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(timeout=0.1), headers=AUTH)
            assert not prepared.json()["report"]["ready"]
            assert prepared.json()["report"]["counts"]["inflight"] == 1
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_a_slow_prepared_verification_does_not_block_a_checkpoint_and_replays_without_the_original(
    provider: SnapshotProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    finish = asyncio.Event()
    patches = []
    run_ids = []

    async def evaluate(**kwargs):
        patches.append(kwargs["pred"]["model_patch"])
        run_ids.append(kwargs["run_id"])
        started.set()
        await finish.wait()
        return {"resolved": True, "completed": True}

    monkeypatch.setattr("resources_servers.swebench.app.run_instance", evaluate)
    server, client = make_server(tmp_path / "state")
    pending = None
    async with client:
        try:
            await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
            captured = await client.post("/prepare_verify", json=VERIFY)
            payload = VERIFY | captured.json()
            cookies = dict(client.cookies)
            pending = asyncio.create_task(client.post("/verify_prepared", json=payload))
            await asyncio.wait_for(started.wait(), 5)
            prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(timeout=0.2), headers=AUTH)
            assert prepared.json()["phase"] == "prepared", prepared.text
            assert not pending.done(), "tests must still be running when the checkpoint prepares"
            committed = await client.post(
                "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path / "ckpt")), headers=AUTH
            )
            assert committed.status_code == 200, committed.text
            assert committed.json()["manifest"]["record_count"] == 0
            assert ops(provider, "snapshot") == [], "verifying episodes no longer need an agent sandbox snapshot"
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert provider.boxes["sb-2"]["state"] == "stopped", "interrupted tests must free their sandbox"
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    finish.set()
    fresh_server, fresh = make_server(tmp_path / "state")
    async with fresh:
        fresh.cookies.update(cookies)
        restored = await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control(
                "restore", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r", "attempt": 1}]
            ),
            headers=AUTH,
        )
        assert restored.status_code == 200, restored.text
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("restore"), headers=AUTH)
        result = await fresh.post("/ng-rollout/r-a2/verify_prepared", json=payload)
    assert result.status_code == 200 and result.json()["reward"] == 1, result.text
    assert result.json()["model_patch"] == payload["model_patch"]
    assert patches == [payload["model_patch"], payload["model_patch"]]
    assert len(set(run_ids)) == 2, "a replay must not collide with the previous attempt's test logs"
    assert provider.boxes["sb-3"]["state"] == "stopped"
    assert fresh_server._checkpointer().session_ids == server._checkpointer().session_ids == []


@pytest.mark.parametrize("patch", [None, ""])
async def test_prepared_verification_requires_an_explicit_patch_but_accepts_an_empty_diff(
    provider: SnapshotProvider, tmp_path: Path, patch: str | None
) -> None:
    _, client = make_server(tmp_path / "state")
    async with client:
        payload = VERIFY if patch is None else VERIFY | {"model_patch": patch}
        result = await client.post("/verify_prepared", json=payload)
    assert result.status_code == (422 if patch is None else 200), result.text
    assert len(ops(provider, "create")) == (0 if patch is None else 1)


async def test_the_evaluation_harness_propagates_cancellation_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(swebench_patches, "RUN_EVALUATION_LOG_DIR", tmp_path)
    container = SimpleNamespace(copy=AsyncMock(side_effect=asyncio.CancelledError), cleanup=AsyncMock())
    spec = SimpleNamespace(instance_id="test", is_remote_image=True)
    with pytest.raises(asyncio.CancelledError):
        await swebench_patches.run_instance(
            spec, {"model_patch": "diff"}, False, False, container, "cancelled", timeout=1
        )
    container.cleanup.assert_awaited_once()
