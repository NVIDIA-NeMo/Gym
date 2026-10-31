# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual verifier/lifecycle/typed verify, environment HTTP and Collection/Store boundaries; no services."""

import asyncio
import io
import json
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import httpx
import orjson
import pytest
from omegaconf import OmegaConf

import nemo_gym.rollout_collection as collector
from environment_servers.single_agent_turn.tests.test_app import (
    _agent_response,
    _environment_server,
    _request,
    _Response,
)
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollout_store import RolloutStore
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.terminal_bench_4 import lifecycle, verifier
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer, atomic_json
from resources_servers.terminal_bench_4.environment import Environment
from resources_servers.terminal_bench_4.models import AgentTermination, SandboxedVerifyRequest


class Sandbox:
    def __init__(self, case, reward):
        self.case, self.reward = case, reward
        self.commands = []
        self.excludes = False
        self.tests = 0

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command.startswith("chmod"):
            if self.case == "preparation":
                await asyncio.sleep(0.2)
            return SandboxExecResult("", "", 0)
        if command.startswith("/tests/test.sh"):
            self.tests += 1
            if self.case == "unknown_status":
                await asyncio.sleep(0.2)
            if self.case == "provider_timeout":
                raise TimeoutError("scripted provider status wait")
            if self.case == "typed_timeout":
                return SandboxExecResult("", "", 124, "timeout")
            return SandboxExecResult("", "", 124 if self.case == "native124" else 0)
        if command.startswith("if [ -f /logs/verifier"):
            return SandboxExecResult("reward.txt", "", 0)
        if command.startswith("tar "):
            self.excludes = "--exclude=reward" in command
            return SandboxExecResult("", "", 0)
        if command.startswith("rm -f") and self.case == "artifact_timeout":
            await asyncio.sleep(0.2)
        return SandboxExecResult("", "", 0)

    async def download(self, remote, local):
        local = Path(local)
        if remote.endswith("reward.txt"):
            if self.case == "reward_transfer":
                raise ConnectionError("scripted official reward transfer loss")
            local.write_text(str(self.reward))
            return local
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            if not self.excludes:
                raw = str(self.reward).encode()
                entry = tarfile.TarInfo("reward.txt")
                entry.size = len(raw)
                archive.addfile(entry, io.BytesIO(raw))
        local.write_bytes(data.getvalue())
        return local


def environment(sandbox, budget):
    env = NS(
        main=sandbox,
        config=NS(exec_shell=None, default_exec_timeout_s=budget),
        task_env={},
        uses_compose=False,
        sandbox=lambda _: sandbox,
        closed=False,
        start=AsyncMock(),
    )

    async def execute(command, **kwargs):
        return await Environment.exec(env, command, **kwargs)

    async def stop():
        env.closed = True

    env.exec = execute
    env.stop = AsyncMock(side_effect=stop)
    return env


async def produce(tmp_path, monkeypatch, case, reward):
    task = NS(
        name="task",
        config=NS(verifier=NS(timeout_sec=0.025, env={}, user="root"), environment=NS(build_timeout_sec=0.1)),
    )
    legacy = case.startswith("legacy_")
    sdk = Sandbox(case.removeprefix("legacy_"), reward)
    env = environment(sdk, 0.025)
    env.task = task
    agent = NS(main=object(), closed=False, quiesce_agent=AsyncMock())

    async def stop_agent():
        agent.closed = True

    agent.stop = AsyncMock(side_effect=stop_agent)
    session = lifecycle.Session(
        identity="fixture",
        owner="fixture",
        request=NS(task_name="task", capture_rollout_id="fixture-run"),
        session_id="fixture-session",
        directory=tmp_path,
    )
    session.task = task
    session.environment = agent
    session.verifier_environment = env
    session.termination = AgentTermination(reason="completed")
    session.seed_response = NS(termination=None)
    session.resources_request = None if legacy else NS(resources_session_id="resource-fixture")
    session.config = NS(environment=NS(verifier_timeout_sec=lambda _: task.config.verifier.timeout_sec))
    session.persist = lambda: atomic_json(
        tmp_path / "resource-result.json", session.result | {"diagnostics": session.diagnostics}
    )
    monkeypatch.setattr(lifecycle, "download_dir", AsyncMock())
    monkeypatch.setattr(lifecycle, "collect", AsyncMock())
    monkeypatch.setattr(lifecycle, "restore", AsyncMock())
    if case == "original_failure":
        session.termination = AgentTermination(reason="infrastructure_error", detail="original failure")
        session.result["exception_info"] = {
            "exception_type": "OriginalFailure",
            "exception_message": "original failure",
        }
        session.diagnostics.append(
            {"operation": "original", "event_id": "fixture-existing-id", "cause_id": "fixture-existing-cause"}
        )
        sdk.case = "preparation"
    resource = NS(_session=lambda *_: session, _check_expiry=lambda _: None, _remember_close=lambda *_: None)
    resource._verified_response = lambda s, extra: TerminalBench4ResourcesServer._verified_response(resource, s, extra)
    resource._finalize_session = lambda s: TerminalBench4ResourcesServer._finalize_session(resource, s)
    body = SandboxedVerifyRequest(
        session_id="fixture-session",
        responses_create_params={"input": "task"},
        response=_agent_response(),
        termination=session.termination,
        agent_started=True,
    )
    result = await TerminalBench4ResourcesServer.verify(resource, NS(), body)
    replay = await TerminalBench4ResourcesServer.verify(resource, NS(), body)
    assert replay == result
    return result, sdk, session, agent, env


@pytest.mark.parametrize(
    "case,reward,measured",
    [
        ("healthy", 1, True),
        ("wrong", 0, True),
        ("native124", 0, True),
        ("preparation", 0, False),
        ("unknown_status", 0, False),
        ("provider_timeout", 0, False),
        ("typed_timeout", 0, False),
        ("reward_transfer", 0, False),
        ("artifact_timeout", 1, True),
        ("original_failure", 0, False),
        ("legacy_unknown_status", 0, False),
        ("legacy_reward_transfer", 0, False),
    ],
)
def test_actual_persisted_boundary(tmp_path, monkeypatch, case, reward, measured):
    asyncio.run(persisted(tmp_path, monkeypatch, case, reward, measured))


async def persisted(tmp_path, monkeypatch, case, reward, measured):
    verified, sdk, session, agent, env = await produce(tmp_path, monkeypatch, case, reward)
    assert verified.reward == reward
    assert verified.mask_sample is (not measured)
    assert not any(k.startswith("_ng_") for k in verified.model_dump())
    assert verified.response == _agent_response()
    if case in ("healthy", "wrong", "native124", "artifact_timeout"):
        test = [k for c, k in sdk.commands if c.startswith("/tests/test.sh")]
        assert 0 < test[0]["timeout_s"] <= session.task.config.verifier.timeout_sec
    if case == "artifact_timeout":
        assert verified.verifier_diagnostics
    if case == "original_failure":
        assert verified.infrastructure_error == "OriginalFailure"
        assert verified.verifier_diagnostics[0]["event_id"] == "fixture-existing-id"
    environment_server, dependency = _environment_server()
    dependency.responses[4] = _Response(verified.model_dump(mode="json"))
    source = tmp_path / "input.jsonl"
    source.write_text(_request().task.model_dump_json() + "\n")
    cfg = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        environment_server_routes={"test:source": "environment"},
        disable_aggregation=False,
        disable_health_check=True,
        route_failures_to_sidecar=True,
    )
    calls = []

    class Reply:
        status = 200
        ok = True
        headers = {}

        def __init__(self, data):
            self.data = data

        def release(self):
            pass

        async def read(self):
            return orjson.dumps(self.data)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=environment_server.setup_webserver()), base_url="http://environment"
    ) as http:

        async def post(server_name, url_path, json, **kwargs):
            if url_path == "/aggregate_metrics":
                return Reply(compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())
            assert url_path == "/run"
            calls.append(json)
            response = await http.post(url_path, json=json)
            response.raise_for_status()
            return Reply(response.json())

        client = MagicMock()
        client.post = AsyncMock(side_effect=post)
        client.global_config_dict = OmegaConf.create({"environment": {"environment_servers": {"custom": {}}}})

        async def status(reply):
            assert reply.ok

        async def payload(reply):
            return reply.data

        monkeypatch.setattr(collector, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(collector, "setup_server_client_utils", lambda *a, **k: client)
        monkeypatch.setattr(collector, "raise_for_status", status)
        monkeypatch.setattr(collector, "get_response_json", payload)
        from nemo_gym.rollout_recovery import IncompleteEvaluationError

        for resume in (False, True):
            try:
                await RolloutCollectionHelper().run_from_config(cfg.model_copy(update={"resume_from_cache": resume}))
            except IncompleteEvaluationError:
                assert not measured
        store = RolloutStore.read(Path(cfg.output_jsonl_fpath))
        coverage = store.coverage()
        assert coverage["measured"] == int(measured) and coverage["masked"] == int(not measured)
        assert store.pending() == [] and len(calls) == 1
        rows = [json.loads(x) for x in Path(cfg.output_jsonl_fpath).read_text().splitlines()]
        assert rows[0]["response"] == _agent_response().model_dump(mode="json")
        assert rows[0]["reward"] == reward
        print(
            "PERSISTED",
            json.dumps(
                {
                    "case": case,
                    "coverage": coverage,
                    "run_calls": len(calls),
                    "test_calls": sdk.tests,
                    "agent_stops": agent.stop.await_count,
                    "verifier_stops": env.stop.await_count,
                }
            ),
        )


def test_cross_device_reward_staging(tmp_path, monkeypatch):
    async def run():
        if not Path("/dev/shm").is_dir():
            pytest.skip("separate tmpfs unavailable")
        with tempfile.TemporaryDirectory(dir="/dev/shm") as location:
            logs = Path(location)
            if logs.stat().st_dev == tmp_path.stat().st_dev:
                pytest.skip("no actual cross-device boundary")
            sdk = Sandbox("healthy", 1)
            env = environment(sdk, 0.1)
            diagnostics = []
            rewards = await verifier._download_reward(env, logs, lambda: 0.1, diagnostics)
            assert rewards["reward"] == 1 and (logs / "reward.txt").read_text() == "1"
            print(
                "CROSS_DEVICE",
                json.dumps({"logs_device": logs.stat().st_dev, "default_temp_device": tmp_path.stat().st_dev}),
            )

    asyncio.run(run())


def test_local_cleanup_failure_keeps_grade(tmp_path, monkeypatch):
    original = verifier.TemporaryDirectory
    created = []

    class BadCleanup:
        def __init__(self, *a, **k):
            self.real = original(*a, **k)
            self.name = self.real.name
            created.append(self.real)

        def cleanup(self):
            # Actual nonempty-directory filesystem failure, not a fabricated successful result.
            import os

            (Path(self.name) / "concurrent-cleanup-entry").write_text("fixture")
            os.rmdir(self.name)

    monkeypatch.setattr(verifier, "TemporaryDirectory", BadCleanup)

    async def run():
        sdk = Sandbox("healthy", 1)
        env = environment(sdk, 0.1)
        diagnostics = []
        rewards = await verifier._download_reward(env, tmp_path, lambda: 0.1, diagnostics)
        assert rewards["reward"] == 1
        assert any(x["operation"] == "verifier_reward_local_cleanup" for x in diagnostics)

    try:
        asyncio.run(run())
    finally:
        for item in created:
            item.cleanup()


def test_preparation_consumes_the_common_verifier_deadline(tmp_path):
    async def check():
        budget = 0.2

        class SlowPreparation(Sandbox):
            async def exec(self, command, **kwargs):
                self.commands.append((command, kwargs))
                if command.startswith("chmod"):
                    await asyncio.sleep(0.12)
                    return SandboxExecResult("", "", 0)
                if command.startswith("/tests/test.sh"):
                    self.tests += 1
                    await asyncio.sleep(0.5)
                raise AssertionError("No reward or optional transfer after unknown execution")

        sdk = SlowPreparation("elapsed", 1)
        env = environment(sdk, budget)
        env.task = NS(config=NS(verifier=NS(timeout_sec=budget, env={}, user="root")))
        diagnostics = []
        start = asyncio.get_running_loop().time()
        with pytest.raises(verifier.VerifierStatusUnknownError):
            await verifier.run_verifier(env, tmp_path, diagnostics)
        elapsed = asyncio.get_running_loop().time() - start
        execution = [kwargs for command, kwargs in sdk.commands if command.startswith("/tests/test.sh")]
        assert len(execution) == 1
        assert 0 < execution[0]["timeout_s"] < 0.1
        assert 0.18 <= elapsed < 0.28
        assert diagnostics[-1]["operation"] == "verifier_execution"
        assert [cause["exception_type"] for cause in diagnostics[-1]["causes"]] == ["TimeoutError", "CancelledError"]

    asyncio.run(check())


@pytest.mark.parametrize("transfer_phase", ["archive", "download", "cleanup"])
def test_optional_transfer_uses_remaining_budget_and_retains_reward(tmp_path, transfer_phase):
    async def check():
        budget = 0.2

        class SlowPreparation(Sandbox):
            async def exec(self, command, **kwargs):
                if command.startswith("chmod"):
                    await asyncio.sleep(0.12)
                if (transfer_phase == "archive" and command.startswith("tar ")) or (
                    transfer_phase == "cleanup" and command.startswith("rm -f")
                ):
                    self.commands.append((command, kwargs))
                    await asyncio.sleep(0.5)
                return await super().exec(command, **kwargs)

            async def download(self, remote, local):
                if transfer_phase == "download" and not remote.endswith("reward.txt"):
                    await asyncio.sleep(0.5)
                return await super().download(remote, local)

        sdk = SlowPreparation("elapsed", 1)
        env = environment(sdk, budget)
        env.task = NS(config=NS(verifier=NS(timeout_sec=budget, env={}, user="root")))
        diagnostics = []
        start = asyncio.get_running_loop().time()
        result = await verifier.run_verifier(env, tmp_path, diagnostics)
        elapsed = asyncio.get_running_loop().time() - start
        assert result["rewards"] == {"reward": 1}
        assert (tmp_path / "verifier/reward.txt").read_text() == "1"
        assert sdk.tests == 1
        assert 0.18 <= elapsed < 0.28
        for _, kwargs in sdk.commands[1:]:
            assert 0 < kwargs["timeout_s"] < 0.1
        assert diagnostics[-1]["operation"] == "verifier_artifact_transfer"
        if transfer_phase != "cleanup":
            assert not any(command.startswith("rm -f") for command, _ in sdk.commands)

    asyncio.run(check())


@pytest.mark.parametrize("transfer_phase", ["lookup", "download"])
def test_reward_transfer_cannot_restart_the_verifier_deadline(tmp_path, transfer_phase):
    async def check():
        budget = 0.2

        class SlowPreparation(Sandbox):
            async def exec(self, command, **kwargs):
                if command.startswith("chmod"):
                    await asyncio.sleep(0.12)
                if command.startswith("if [ -f") and transfer_phase == "lookup":
                    self.commands.append((command, kwargs))
                    await asyncio.sleep(0.5)
                return await super().exec(command, **kwargs)

            async def download(self, remote, local):
                if remote.endswith("reward.txt") and transfer_phase == "download":
                    await asyncio.sleep(0.5)
                return await super().download(remote, local)

        sdk = SlowPreparation("elapsed", 1)
        env = environment(sdk, budget)
        env.task = NS(config=NS(verifier=NS(timeout_sec=budget, env={}, user="root")))
        diagnostics = []
        start = asyncio.get_running_loop().time()
        with pytest.raises(verifier.VerifierStatusUnknownError):
            await verifier.run_verifier(env, tmp_path, diagnostics)
        elapsed = asyncio.get_running_loop().time() - start
        assert 0.18 <= elapsed < 0.28
        assert sdk.tests == 1
        assert diagnostics[-1]["operation"] == "verifier_reward_transfer"
        assert not (tmp_path / "verifier/reward.txt").exists()
        assert not any(command.startswith("tar ") for command, _ in sdk.commands)

    asyncio.run(check())
