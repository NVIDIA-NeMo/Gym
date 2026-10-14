# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shlex
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment_servers.harbor import app as harbor
from environment_servers.harbor.tests.test_app import FakeSandbox, make_server, run_episode
from environment_servers.harbor.tests.test_review_regressions import during_turn
from nemo_gym.sandbox.providers.base import SandboxExecResult


class LocalSandbox:
    """Run production transfer and verifier commands on a pytest-owned filesystem tree."""

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def upload(self, source, target):
        shutil.copyfile(source, target)

    async def download(self, source, target):
        shutil.copyfile(source, target)


@pytest.fixture
def verifier_tree(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    root.mkdir()
    logs, tests = root / "logs", root / "tests"
    monkeypatch.setattr(harbor, "TESTS_DIR", str(tests))
    monkeypatch.setattr(harbor, "VERIFIER_LOGS_DIR", str(logs / "verifier"))
    monkeypatch.setattr(harbor, "AGENT_LOGS_DIR", str(logs / "agent"))
    monkeypatch.setattr(harbor, "VERIFIER_STDOUT", str(logs / "verifier" / "test-stdout.txt"))
    return root, logs, tests


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
@pytest.mark.parametrize("linked", ["none", "tests", "verifier", "logs"])
async def test_linked_verifier_roots_cannot_preserve_planted_success(tmp_path, monkeypatch, verifier_tree, linked):
    root, logs, tests = verifier_tree
    planted = root / "planted"
    planted.mkdir()
    (planted / "reward.json").write_text('{"reward": 1}')
    (planted / "helper.sh").write_text("echo 1")
    if linked == "logs":
        (planted / "verifier").mkdir()
        (planted / "verifier" / "reward.json").write_text('{"reward": 1}')
        logs.symlink_to(planted, target_is_directory=True)
    else:
        logs.mkdir()
        if linked != "none":
            target = tests if linked == "tests" else logs / "verifier"
            target.symlink_to(planted, target_is_directory=True)
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests" / "test.sh").write_text(
        f"#!/bin/bash\nif test -f {shlex.quote(str(tests / 'helper.sh'))}; then\n"
        f"  bash {shlex.quote(str(tests / 'helper.sh'))}\nelse\n  echo 0\nfi "
        f"> {shlex.quote(str(logs / 'verifier' / 'reward.txt'))}\n"
    )
    outcome = await server._run_verifier(SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root)))
    assert outcome["reward"] == 0
    assert not outcome.get("mask_sample", False)
    assert not tests.is_symlink() and not logs.is_symlink() and not (logs / "verifier").is_symlink()
    assert not (tests / "helper.sh").exists()
    assert (planted / "reward.json").read_text() == '{"reward": 1}'
    await server.shutdown()


@pytest.mark.parametrize("sign", ["", "-"])
async def test_large_numeric_reward_is_measured_zero(tmp_path, monkeypatch, sign):
    sandbox = FakeSandbox(verifier_files={"reward.json": '{"reward": ' + sign + "9" * 400 + "}"})
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and result.mask_sample is False
    assert result.failure_kind == "harbor:invalid_reward"
    assert sum("test.sh" in call["command"] for call in sandbox.execs) == 1


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_unsafe_diagnostic_link_does_not_mask_valid_reward(tmp_path, monkeypatch, verifier_tree):
    root, logs, _ = verifier_tree
    outside = root / "private"
    outside.write_text("must not be copied")
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests" / "test.sh").write_text(
        f"#!/bin/bash\necho 1 > {shlex.quote(str(logs / 'verifier/reward.txt'))}\n"
        f"ln -s {shlex.quote(str(outside))} {shlex.quote(str(logs / 'verifier/diagnostic-link'))}\n"
        "echo useful-output\n"
    )
    outcome = await server._run_verifier(SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root)))
    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    artifacts = Path(outcome["verifier_logs_dir"])
    assert not (artifacts / "diagnostic-link").exists()
    assert "useful-output" in (artifacts / "test-stdout.txt").read_text()
    assert all("must not be copied" not in path.read_text() for path in artifacts.rglob("*") if path.is_file())
    await server.shutdown()


async def test_timeout_retains_logs_and_is_measured_zero(tmp_path, monkeypatch):
    sandbox = FakeSandbox(
        verifier_files={"test-stdout.txt": "verifier reached checkpoint"},
        test_result=SandboxExecResult(stdout=None, stderr="timeout", return_code=124, error_type="timeout"),
    )
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and result.mask_sample is False
    assert result.failure_kind == "harbor:verifier_timeout"
    assert result.verifier_return_code == 124
    assert "checkpoint" in (Path(result.verifier_logs_dir) / "test-stdout.txt").read_text()
    assert sum("test.sh" in call["command"] for call in sandbox.execs) == 1


def test_sandbox_lifetime_covers_episode_and_cleanup(tmp_path, monkeypatch):
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    monkeypatch.setattr(harbor, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    spec = server._sandbox_spec(task, "/app")
    assert spec.ttl_s >= server.config.default_episode_timeout_seconds + server.config.cleanup_timeout_seconds


async def test_optional_diagnostic_transfer_failure_preserves_score(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    download = sandbox.download
    calls = 0

    async def fail_diagnostics(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("diagnostic transfer failed")
        await download(source, target)

    monkeypatch.setattr(sandbox, "download", fail_diagnostics)
    outcome = await server._run_verifier(SimpleNamespace(task=task, sandbox=sandbox, workdir="/app"))
    assert calls == 2
    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    await server.shutdown()


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_unsafe_reward_link_still_masks_transfer_failure(tmp_path, monkeypatch, verifier_tree):
    root, logs, _ = verifier_tree
    outside = root / "untrusted-reward"
    outside.write_text("1")
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests/test.sh").write_text(
        f"ln -s {shlex.quote(str(outside))} {shlex.quote(str(logs / 'verifier/reward.txt'))}\n"
    )
    outcome = await server._run_verifier(SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root)))
    assert outcome["reward"] == 0 and outcome["mask_sample"] is True
    assert not list(server._artifacts_dir.rglob("reward.txt"))
    await server.shutdown()


@pytest.mark.parametrize("identifier", ["normal", "dots", "long"])
async def test_episode_ids_never_name_verifier_folders(tmp_path, monkeypatch, identifier):
    """Rollout IDs are caller input; verifier folders are reserved under the artifact root regardless of them."""
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    root = tmp_path / "work" / "artifacts"
    server._artifacts_dir = root
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "reward.txt"
    sentinel.write_text("untouched")
    rollout = {"normal": "r1", "dots": "a..b..", "long": "x" * 300}[identifier]
    response = await run_episode(server, task, rollout=rollout)
    assert response.episode_id.rollout_id == rollout
    result = response.result
    assert result.reward == 1 and result.mask_sample is False
    logs = Path(result.verifier_logs_dir).resolve()
    assert logs.parent == root.resolve() and rollout not in logs.name
    assert (logs / "reward.txt").read_text() == "1\n"
    assert sentinel.read_text() == "untouched"
    assert sum("test.sh" in call["command"] for call in sandbox.execs) == 1


async def test_artifact_root_redirected_after_startup_cannot_receive_verifier_files(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    during_turn(server, lambda: server._artifacts_dir.symlink_to(outside, target_is_directory=True))
    result = (await run_episode(server, task)).result
    assert result.mask_sample is True
    assert result.failure_kind == "persistence_failed"
    assert str(server._artifacts_dir) in result.failure_reason
    assert list(outside.iterdir()) == []
    assert not any("test.sh" in call["command"] for call in sandbox.execs)
    assert sandbox.stopped


async def test_reused_episode_id_does_not_read_old_verifier_output(tmp_path, monkeypatch):
    sandbox = FakeSandbox(verifier_files={"test-stdout.txt": "no reward was written"})
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    stale = server._artifacts_dir / "r1"
    stale.mkdir(parents=True)
    (stale / "reward.json").write_text('{"reward": 1}')
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and result.mask_sample is False
    assert result.failure_kind == "harbor:missing_reward"
    assert Path(result.verifier_logs_dir) != stale
    assert (stale / "reward.json").read_text() == '{"reward": 1}'


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
@pytest.mark.parametrize("umask", [0o002, 0o077])
async def test_verifier_logs_preserve_directory_permissions_after_transfers(
    tmp_path, monkeypatch, verifier_tree, umask
):
    root, logs, _ = verifier_tree
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    server._artifacts_dir.mkdir()
    server._artifacts_dir.chmod(0o2775)
    (task.path / "tests/test.sh").write_text(
        f"echo 1 > {shlex.quote(str(logs / 'verifier/reward.txt'))}\necho useful-output\n"
    )
    previous_umask = os.umask(umask)
    try:
        expected = server._artifacts_dir / "ordinary-directory"
        expected.mkdir()
        outcome = await server._run_verifier(SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root)))
    finally:
        os.umask(previous_umask)
        await server.shutdown()
    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    artifacts = Path(outcome["verifier_logs_dir"])
    assert artifacts.parent == server._artifacts_dir
    assert stat.S_IMODE(artifacts.stat().st_mode) == stat.S_IMODE(expected.stat().st_mode)
    assert (artifacts / "reward.txt").read_text() == "1\n"
    assert (artifacts / "test-stdout.txt").read_text() == "useful-output\n"


async def test_artifact_directory_creation_failure_is_a_storage_failure(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    during_turn(server, lambda: server._artifacts_dir.write_text("a file blocks the output directory"))
    result = (await run_episode(server, task)).result
    assert result.mask_sample is True
    assert result.failure_kind == "persistence_failed"
    assert str(server._artifacts_dir) in result.failure_reason
    assert server._artifacts_dir.read_text() == "a file blocks the output directory"
    assert not any("test.sh" in call["command"] for call in sandbox.execs)
