# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in completion contracts distinguish a measured zero from unavailable verification."""

import json
from pathlib import Path

import pytest

from environment_servers.harbor.tests.conftest import (
    SEPARATE_TOML,
    AgentSandbox,
    FakeSandbox,
    make_server,
    run_episode,
)
from nemo_gym.sandbox import SandboxExecResult


def prepared(tmp_path, monkeypatch, verifiers, *, agent=None, retries=1, completion_key="evaluation_completed"):
    agent = agent or AgentSandbox()
    server, task, _, _ = make_server(
        tmp_path,
        monkeypatch,
        agent,
        toml=SEPARATE_TOML,
        verifier_completion_key=completion_key,
        inconclusive_verification_retries=retries,
    )
    created = []

    async def create_verifier(task, *, episode_key=None):
        assert agent.stopped
        verifier = verifiers[len(created)]
        created.append(verifier)
        server._sandboxes[f"{episode_key}/verifier"] = verifier
        return verifier

    monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
    return server, task, agent, created


def verdict(reward, completed):
    return FakeSandbox(
        verifier_files={"reward.json": json.dumps({"reward": reward, "evaluation_completed": completed})}
    )


@pytest.mark.parametrize("reward", [0.0, 1.0])
async def test_completed_scores_are_measured_without_retry(tmp_path, monkeypatch, reward):
    server, task, agent, created = prepared(tmp_path, monkeypatch, [verdict(reward, 1)])
    result = (await run_episode(server, task)).result
    assert (result.reward, result.mask_sample, result.evaluation_completed) == (reward, False, True)
    assert len(result.verification_attempts) == len(created) == 1
    assert agent.stopped and created[0].stopped and not server._sandboxes


async def test_retry_uses_the_same_archives_and_keeps_both_results(tmp_path, monkeypatch):
    first, second = verdict(1, 0), verdict(0, 1)
    server, task, agent, created = prepared(tmp_path, monkeypatch, [first, second])
    result = (await run_episode(server, task)).result
    assert (result.reward, result.mask_sample, result.evaluation_completed) == (0, False, True)
    assert [attempt.mask_sample for attempt in result.verification_attempts] == [True, False]
    assert len(created) == 2 and all(verifier.stopped for verifier in created)
    assert not server._sandboxes
    assert agent.archived == ["/logs/artifacts", "/app/output"]
    assert sum("cp /app/state.db" in call["command"] for call in agent.execs) == 1
    for archive in agent.archives.values():
        assert archive in first.uploaded_archives and archive in second.uploaded_archives
    for attempt in result.verification_attempts:
        assert (Path(attempt.verifier_logs_dir) / "reward.json").is_file()
    assert not list((Path(result.verification_attempts[0].verifier_logs_dir) / "artifacts").glob("*.tar.gz"))


@pytest.mark.parametrize("kind", ["incomplete", "missing", "invalid", "missing-key", "timeout", "provider"])
async def test_inconclusive_verification_is_bounded_and_masked(tmp_path, monkeypatch, kind):
    verifiers = [verdict(1, 0), verdict(1, 0)]
    for verifier in verifiers:
        if kind == "missing":
            verifier.verifier_files = {}
        elif kind == "invalid":
            verifier.verifier_files = {"reward.json": "broken"}
        elif kind == "missing-key":
            verifier.verifier_files = {"reward.txt": "1"}
        elif kind in ("timeout", "provider"):
            # Even a valid-looking reward cannot override the provider's incomplete execution.
            verifier.verifier_files = {"reward.json": '{"reward":1,"evaluation_completed":1}'}
            verifier.test_result = SandboxExecResult(stdout="", stderr="", return_code=None, error_type=kind)
    server, task, _, created = prepared(tmp_path, monkeypatch, verifiers)
    result = (await run_episode(server, task)).result
    assert (result.reward, result.mask_sample, result.evaluation_completed) == (0, True, False)
    assert len(result.verification_attempts) == len(created) == 2
    assert all(verifier.stopped for verifier in created) and not server._sandboxes


@pytest.mark.parametrize(
    "error_type,return_code,masked", [("timeout", None, True), (None, None, True), (None, 1, False)]
)
async def test_capture_provider_failure_masks_but_git_exit_is_gradeable(
    tmp_path, monkeypatch, error_type, return_code, masked
):
    class CaptureFailure(AgentSandbox):
        async def exec(self, command, **kwargs):
            if command.startswith("cp /app/state.db"):
                return SandboxExecResult(stdout="", stderr="", return_code=return_code, error_type=error_type)
            return await super().exec(command, **kwargs)

    server, task, agent, created = prepared(tmp_path, monkeypatch, [verdict(0, 1)], agent=CaptureFailure())
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and result.mask_sample == masked
    assert result.evaluation_completed == (not masked)
    assert len(created) == (0 if masked else 1)
    assert agent.stopped and not server._sandboxes


async def test_default_harbor_missing_reward_remains_measured_zero(tmp_path, monkeypatch):
    server, task, _, created = prepared(tmp_path, monkeypatch, [FakeSandbox(verifier_files={})], completion_key=None)
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and not result.mask_sample
    assert len(created) == 1


@pytest.mark.parametrize("metrics, reward, masked", [({"accuracy": 0.5}, 0.5, False), ({}, 0, True)])
async def test_completion_metric_is_not_a_reward_component(tmp_path, monkeypatch, metrics, reward, masked):
    verifier = FakeSandbox(verifier_files={"reward.json": json.dumps(metrics | {"evaluation_completed": 1})})
    server, task, _, _ = prepared(tmp_path, monkeypatch, [verifier], retries=0)
    result = (await run_episode(server, task)).result
    assert result.reward == reward and result.mask_sample == masked
    if not masked:
        assert result.reward_components == metrics


def test_retry_allowance_counts_only_fresh_verification_not_collection(tmp_path, monkeypatch):
    server, task, _, _ = prepared(tmp_path, monkeypatch, [])
    with_retry = server._post_seed_budget_s(task)
    server.config.inconclusive_verification_retries = 0
    assert with_retry - server._post_seed_budget_s(task) == (
        server.config.sandbox_ready_timeout_s + task.config.verifier.timeout_sec + server.config.verifier_grace_s
    )


@pytest.mark.parametrize("failures", [1, 2])
async def test_retry_does_not_replace_a_verifier_that_cannot_be_stopped(tmp_path, monkeypatch, failures):
    class StopFails(FakeSandbox):
        stops = 0

        async def stop(self):
            self.stops += 1
            if self.stops <= failures:
                raise RuntimeError("provider could not stop verifier")
            await super().stop()

    first = StopFails(verifier_files={"reward.json": '{"reward":0,"evaluation_completed":0}'})
    second = verdict(1, 1)
    server, task, _, created = prepared(tmp_path, monkeypatch, [first, second])
    result = (await run_episode(server, task)).result
    assert first.stopped and first.stops == failures + 1
    assert not server._sandboxes
    if failures == 1:
        assert created == [first, second] and second.stopped
        assert result.reward == 1 and not result.mask_sample
    else:
        assert created == [first] and not second.stopped
        assert result.mask_sample and not result.evaluation_completed
        assert "previous turn's verifier sandbox could not be stopped" in result.failure_reason
