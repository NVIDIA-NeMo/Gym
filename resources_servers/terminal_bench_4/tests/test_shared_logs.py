# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.sandbox import SandboxSpec
from resources_servers.terminal_bench_4 import shared_logs as module
from resources_servers.terminal_bench_4.collection import collect
from resources_servers.terminal_bench_4.shared_logs import SharedLogs
from resources_servers.terminal_bench_4.task import TaskSettings
from resources_servers.terminal_bench_4.tests.test_collection_verifier import environment
from resources_servers.terminal_bench_4.tests.test_environment import environment_config, make_environment
from resources_servers.terminal_bench_4.verifier import restore, run_verifier


class LocalHelper:
    """Execute the production helper scripts against a temporary EFS tree."""

    def __init__(self):
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self._handle = SimpleNamespace(sandbox_id="helper")

    async def exec(self, command, **kwargs):
        args = shlex.split(command)
        assert args[0] == "python3"
        process = await asyncio.create_subprocess_exec(
            sys.executable, *args[1:], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())


def shared(tmp_path, monkeypatch):
    helper = LocalHelper()
    constructor = MagicMock(return_value=helper)
    monkeypatch.setattr(module, "AsyncSandbox", constructor)
    mount = tmp_path / "efs"
    mount.mkdir(exist_ok=True)
    monkeypatch.setattr(SharedLogs, "mount", str(mount))
    config = environment_config(
        sandbox_provider={"opensandbox": {"connection": {"domain": "example.invalid"}}},
        efs_logs_host_path="/mnt/efs/data/shared",
    )
    env = SimpleNamespace(
        config=config,
        provider_config=config.sandbox_provider,
        pool="default",
        session_id="episode",
        build_spec=lambda: SandboxSpec(image="task", metadata={"tb4-session": "episode", "run": "test"}),
    )
    return SharedLogs(env), helper, constructor


def mount_role(env, logs, role):
    shutil.rmtree(env.main.path("/logs"))
    env.main.path("/logs").symlink_to(Path(logs.root) / role, target_is_directory=True)
    for directory in ("agent", "verifier", "artifacts"):
        env.main.path("/logs/" + directory).mkdir()
    env.shared_logs = logs


async def test_efs_snapshot_matches_host_restore_and_survives_agent_deletion(tmp_path, monkeypatch):
    logs, helper, _ = shared(tmp_path, monkeypatch)
    await logs.start()
    cfg = TaskSettings.model_validate(
        {
            "environment": {"docker_image": "agent"},
            "verifier": {"environment": {"docker_image": "verifier"}},
            "artifacts": [{"source": "/logs/artifacts", "exclude": ["*.secret"]}, {"source": "/app"}],
        }
    )
    agent, box, _ = environment(tmp_path / "agent", cfg)
    verifier, target, _ = environment(tmp_path / "verifier", cfg)
    mount_role(agent, logs, "agent")
    mount_role(verifier, logs, "verifier")
    box.path("/logs/agent/trajectory.json").write_text("private agent logs")
    box.path("/logs/verifier/reward.txt").write_text("999")
    box.path("/logs/undeclared.txt").write_text("not an artifact")
    box.path("/logs/artifacts/report.txt").write_bytes(b"original\x00bytes")
    box.path("/logs/artifacts/report.txt").chmod(0o751)
    box.path("/logs/artifacts/link").symlink_to("report.txt")
    box.path("/logs/artifacts/empty").mkdir()
    box.path("/logs/artifacts/omit.secret").write_text("excluded")
    box.path("/app/source.txt").write_text("normal artifact")
    artifacts = tmp_path / "artifacts"
    diagnostics = []
    await collect(agent, artifacts, diagnostics)
    assert logs.archive_digest
    assert not diagnostics
    # Neither subsequent writes nor deleting the workload can affect the
    # already collected snapshot. EFS persists independently of the sandbox.
    box.path("/logs/artifacts/report.txt").write_text("late mutation")
    shutil.rmtree(box.root)
    await logs.prepare_verifier()
    assert logs.restored_archive
    target.upload = AsyncMock(wraps=target.upload)
    await restore(verifier, artifacts)
    assert target.path("/logs/artifacts/report.txt").read_bytes() == b"original\x00bytes"
    assert target.path("/logs/artifacts/report.txt").stat().st_mode & 0o777 == 0o751
    assert target.path("/logs/artifacts/link").is_symlink()
    assert target.path("/logs/artifacts/link").read_bytes() == b"original\x00bytes"
    assert target.path("/logs/artifacts/empty").is_dir()
    assert not target.path("/logs/artifacts/omit.secret").exists()
    assert not target.path("/logs/undeclared.txt").exists()
    assert not list(target.path("/logs/agent").iterdir())
    assert not list(target.path("/logs/verifier").iterdir())
    assert not target.path(logs.restored_archive).exists()
    assert target.path("/app/source.txt").read_text() == "normal artifact"
    # Only /app was uploaded; /logs/artifacts came from the EFS snapshot.
    assert target.upload.await_count == 1
    target.path("/tests/test.sh").write_text(f"#!/bin/sh\necho 1 > {target.path('/logs/verifier/reward.txt')}\n")
    assert await run_verifier(verifier, tmp_path / "result", diagnostics) == {"rewards": {"reward": 1}}
    await logs.stop()
    await logs.stop()
    assert not Path(logs.root).exists()
    assert logs.closed
    helper.stop.assert_awaited_once()
    assert (artifacts / "logs/artifacts/report.txt").read_bytes() == b"original\x00bytes"


async def test_archive_tampering_uses_collected_host_fallback(tmp_path, monkeypatch):
    logs, _, _ = shared(tmp_path, monkeypatch)
    await logs.start()
    cfg = TaskSettings.model_validate(
        {"environment": {"docker_image": "a"}, "verifier": {"environment_mode": "separate"}}
    )
    agent, box, _ = environment(tmp_path / "agent", cfg)
    verifier, target, _ = environment(tmp_path / "verifier", cfg)
    mount_role(agent, logs, "agent")
    mount_role(verifier, logs, "verifier")
    box.path("/logs/artifacts/result.txt").write_text("snapshot")
    await collect(agent, tmp_path / "artifacts", [])
    box.path("/logs/" + logs.archive_name).write_text("changed after collection")
    await logs.prepare_verifier()
    assert logs.restored_archive is None
    await restore(verifier, tmp_path / "artifacts")
    assert target.path("/logs/artifacts/result.txt").read_text() == "snapshot"
    await logs.stop()


@pytest.mark.parametrize(
    "other",
    [
        {"source": "/logs"},
        {"source": "/logs/artifacts/nested"},
        {"source": "/logs/artifacts", "service": "db"},
        {"source": "/app", "destination": "logs/artifacts"},
    ],
)
def test_overlapping_artifacts_keep_ordered_host_restore(tmp_path, monkeypatch, other):
    logs, _, _ = shared(tmp_path, monkeypatch)
    cfg = TaskSettings.model_validate(
        {"environment": {"docker_image": "a"}, "verifier": {"environment_mode": "separate"}, "artifacts": [other]}
    )
    artifacts = cfg.collected_artifacts
    assert logs.collection_archive(artifacts[0], artifacts) is None


async def test_roles_and_episodes_are_isolated_and_writable(tmp_path, monkeypatch):
    first, _, create = shared(tmp_path, monkeypatch)
    second, _, _ = shared(tmp_path, monkeypatch)
    await first.start()
    await second.start()
    assert first.volume("agent")["subPath"] != first.volume("verifier")["subPath"]
    assert first.volume("agent")["subPath"] != second.volume("agent")["subPath"]
    assert first.volume("agent")["mountPath"] == "/logs"
    for role in ("agent", "verifier"):
        assert (Path(first.root) / role).stat().st_mode & 0o777 == 0o777
    spec = create.call_args.args[1]
    assert spec.resources.cpu == 1 and not spec.resources.gpu
    assert spec.metadata["run"] == "test"
    await first.stop()
    assert Path(second.root).is_dir()
    await second.stop()


async def test_partial_initialization_and_failed_delete_still_stop_helper(tmp_path, monkeypatch):
    logs, helper, _ = shared(tmp_path, monkeypatch)
    await logs.start()
    # Treat a partly created role tree as a setup failure.
    shutil.rmtree(Path(logs.root) / "verifier")
    await logs.stop()
    assert not Path(logs.root).exists()
    helper.stop.assert_awaited_once()
    failed, helper, _ = shared(tmp_path, monkeypatch)
    await failed.start()
    failed.python = AsyncMock(side_effect=RuntimeError("EFS unavailable"))
    with pytest.raises(ExceptionGroup, match="cleanup failed"):
        await failed.stop()
    helper.stop.assert_awaited_once()
    assert failed.cleanup_errors and not failed.closed


async def test_live_workload_mount_is_retained_on_failed_teardown(tmp_path, monkeypatch):
    logs, helper, _ = shared(tmp_path, monkeypatch)
    await logs.start()
    await logs.stop(remove_data=False)
    assert Path(logs.root).exists()
    helper.stop.assert_awaited_once()
    assert logs.resources[0]["efs_subpath"] == logs.relative


async def test_compose_mounts_logs_only_in_main(tmp_path, monkeypatch):
    env, _, _, compose = make_environment(tmp_path, monkeypatch, compose=True)
    logs, _, _ = shared(tmp_path, monkeypatch)
    logs.initialize_role = AsyncMock()
    env.shared_logs = logs
    await env.start()
    specs = compose.call_args.kwargs["service_specs"]
    assert specs["main"].provider_options["volumes"] == [logs.volume("agent")]
    assert "volumes" not in specs["db"].provider_options


async def test_log_owner_uses_image_identity_and_protects_parent(tmp_path, monkeypatch):
    logs, _, _ = shared(tmp_path, monkeypatch)
    await logs.start()
    env = SimpleNamespace(
        log_role="agent",
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout=f"{os.getuid()}\n{os.getgid()}\n")),
    )
    await logs.initialize_role(env)
    root = Path(logs.root) / "agent"
    assert root.stat().st_uid == os.getuid() and root.stat().st_mode & 0o777 == 0o755
    assert "user" not in env.exec.await_args.kwargs
    for result in (SimpleNamespace(return_code=0, stdout="invalid"), SimpleNamespace(return_code=1, stdout="0 0")):
        env.exec.return_value = result
        with pytest.raises(RuntimeError, match="task log owner"):
            await logs.initialize_role(env)
    await logs.prepare_verifier()  # No snapshot: normal host restore.
    await logs.stop()


@pytest.mark.parametrize("error", ["VOLUME::HOST_PATH_NOT_ALLOWED /mnt/efs/data/shared", "quota exceeded"])
async def test_only_explicit_unsupported_efs_mount_uses_original_lifecycle(tmp_path, monkeypatch, error):
    env, box, create, _ = make_environment(tmp_path, monkeypatch)
    logs, _, _ = shared(tmp_path, monkeypatch)
    logs.initialize_role = AsyncMock()
    env.shared_logs = logs
    box.start.side_effect = [RuntimeError(error), None]
    if error.startswith("quota"):
        with pytest.raises(RuntimeError, match="quota"):
            await env.start()
        assert env.shared_logs is logs and create.call_count == 1
    else:
        await env.start()
        assert env.shared_logs is None and env.efs_logs_fallback == error
        assert create.call_count == 2
        assert "volumes" not in create.call_args.args[1].provider_options
        box.stop.assert_awaited_once()
        logs.initialize_role.assert_not_awaited()


def test_conflicting_log_mount_rejected_before_provisioning(tmp_path, monkeypatch):
    env, _, create, _ = make_environment(
        tmp_path, monkeypatch, config={"sandbox_provider_options": {"volumes": [{"mountPath": "/logs/"}]}}
    )
    env.shared_logs, _, _ = shared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="conflict"):
        env.build_spec()
    create.assert_not_called()


async def test_cancelled_cleanup_continues_and_closes_helper(tmp_path, monkeypatch):
    logs, helper, _ = shared(tmp_path, monkeypatch)
    await logs.start()
    cleanup_started, proceed = asyncio.Event(), asyncio.Event()
    execute = logs.python

    async def delayed(*args):
        cleanup_started.set()
        await proceed.wait()
        return await execute(*args)

    logs.python = delayed
    pending = asyncio.create_task(logs.stop())
    await cleanup_started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert Path(logs.root).exists()
    proceed.set()
    await logs.stop()
    assert not Path(logs.root).exists()
    helper.stop.assert_awaited_once()


@pytest.mark.parametrize("agent_gpus,verifier_gpus,pool", [(0, 0, "cpu"), (1, 0, "gpu"), (0, 1, "gpu"), (1, 1, "gpu")])
@pytest.mark.parametrize("verifier", [False, True])
def test_logs_helper_uses_task_deployment_and_independent_role_mounts(
    tmp_path, monkeypatch, agent_gpus, verifier_gpus, pool, verifier
):
    for endpoint in ("CPU", "GPU"):
        monkeypatch.setenv(f"OPENSANDBOX_DOMAIN_{endpoint}", endpoint.lower() + ".invalid")
        monkeypatch.setenv(f"OPENSANDBOX_API_KEY_{endpoint}", endpoint + "-credential")
    env, *_ = make_environment(
        tmp_path,
        monkeypatch,
        config={"sandbox_split_endpoints": True},
        verifier=verifier,
        task_config={
            "environment": {"gpus": agent_gpus},
            "verifier": {"environment": {"docker_image": "public/verifier", "gpus": verifier_gpus}},
        },
    )
    env.config.efs_logs_host_path = "/mnt/efs/data/shared"
    create = MagicMock()
    monkeypatch.setattr(module, "AsyncSandbox", create)
    logs = SharedLogs(env)
    cfg, spec = create.call_args.args
    assert cfg["opensandbox"]["connection"]["domain"] == pool + ".invalid"
    assert cfg["opensandbox"]["connection"]["api_key"] == pool.upper() + "-credential"
    assert spec.metadata["nemo-gym.nvidia.com/resource-pool"] == pool
    assert spec.resources.gpu is None and not spec.env
    assert env.provider_config["opensandbox"]["connection"]["domain"] == pool + ".invalid"
    env.shared_logs = logs
    assert env.build_spec().provider_options["volumes"] == [logs.volume("verifier" if verifier else "agent")]
