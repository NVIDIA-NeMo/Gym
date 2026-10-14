# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import io
import json
import posixpath
import shutil
import tarfile
from pathlib import Path

import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from yarl import URL

from environment_servers.harbor.tests.conftest import (
    SOLVE_SH,
    TASK_TOML,
    FakeSandbox,
    episode,
    make_server,
    run_episode,
)
from environment_servers.harbor.verifier import parse_reward_file, select_reward
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.tasks.harbor import load_task


def uploaded_file(sandbox, name: str) -> list[bytes]:
    """The contents of ``name`` in every archive the server uploaded: a single file travels as a one-file archive."""
    found: list[bytes] = []
    for data in sandbox.uploaded_archives:
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile() and posixpath.basename(member.name) == name:
                        found.append(tar.extractfile(member).read())
        except tarfile.TarError:
            continue
    return found


class TestRewardFile:
    def test_text_and_json(self, tmp_path):
        (tmp_path / "reward.txt").write_text("0.5\n")
        assert parse_reward_file(tmp_path) == ({"reward": 0.5}, None)
        (tmp_path / "reward.json").write_text(json.dumps({"accuracy": 1, "speed": 0.25}))
        rewards, problem = parse_reward_file(tmp_path)
        assert problem is None and rewards == {"accuracy": 1.0, "speed": 0.25}
        assert select_reward(rewards) is None
        assert select_reward({"accuracy": 0.25}) == 0.25
        assert select_reward({"reward": 1.0, "other": 0.0}) == 1.0

    @pytest.mark.parametrize(
        ("name", "text", "fragment"),
        [
            ("reward.txt", "", "empty"),
            ("reward.txt", "yes", "not valid"),
            ("reward.json", "[1]", "non-empty JSON object"),
            ("reward.json", '{"reward": "1"}', "not a finite number"),
        ],
    )
    def test_problems(self, tmp_path, name, text, fragment):
        (tmp_path / name).write_text(text)
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None and fragment in problem

    def test_missing(self, tmp_path):
        assert parse_reward_file(tmp_path)[1] == "no reward.json or reward.txt was written"

    @pytest.mark.parametrize("contents", ["{not json", "[]", "", '{"reward": "1"}'])
    def test_invalid_json_does_not_fall_back_to_text(self, tmp_path, contents):
        (tmp_path / "reward.json").write_text(contents)
        (tmp_path / "reward.txt").write_text("0.5\n")
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None
        assert "reward.json" in problem

    def test_boolean_rewards_match_harbor(self, tmp_path):
        (tmp_path / "reward.json").write_text('{"passed": true, "failed": false}')
        assert parse_reward_file(tmp_path) == ({"passed": 1.0, "failed": 0.0}, None)

    def test_non_utf8_bytes_do_not_raise(self, tmp_path):
        (tmp_path / "reward.txt").write_bytes(b"\xff\xfe1\n")
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None and "reward.txt is not valid" in problem

        (tmp_path / "reward.json").write_bytes(b'{"reward": 1}\xff')
        (tmp_path / "reward.txt").write_text("0.25\n")
        assert parse_reward_file(tmp_path)[0] is None
        assert "reward.json is not valid" in parse_reward_file(tmp_path)[1]


class TestEpisode:
    async def test_episode_seeds_runs_the_agent_verifies_and_stops_the_sandbox(self, tmp_path, monkeypatch):
        server, task, sandbox, created = make_server(tmp_path, monkeypatch)
        client = server.server_client
        closes_before_verify: list[int] = []
        real_exec = sandbox.exec

        async def exec_recording_closes(command, **kwargs):
            if "test.sh" in command:
                closes_before_verify.append(client.paths().count("/v1/agent_sessions/close"))
            return await real_exec(command, **kwargs)

        monkeypatch.setattr(sandbox, "exec", exec_recording_closes)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        result = response.result
        assert result.reward == 1.0 and result.mask_sample is False and result.failure_kind is None
        assert result.verifier_rewards == {"reward": 1.0} and result.verifier_return_code == 0
        assert result.response.id == "response"
        assert result.responses_create_params.input[0].content == "Create hello.txt"
        # The sandbox was created for the task, prepared, and handed to the harness through its session seed.
        assert created == [("hello", "/app")]
        assert sandbox.execs[0]["command"] == "mkdir -p /app"
        assert client.paths() == ["/v1/agent_sessions", "/ng-rollout/r1/v1/responses", "/v1/agent_sessions/close"]
        assert client.seed_bodies()[0]["sandbox_access"] == {
            "connection": {
                "kind": "direct",
                "provider_config_ref": "sandbox",
                "descriptor": {"sandbox_id": "sb-1", "workdir": "/app"},
            },
            "workdir": "/app",
        }
        assert client.seed_bodies()[0]["task_id"] == {"taskset": "ds", "task_id": "hello"}
        # The task's own agent budget travels in the seed, so the harness can stop in time.
        assert client.seed_bodies()[0]["agent_timeout_sec"] == 300.0
        # The turn's cookie jar, not the seed's, travels to the close.
        assert client.calls[2][2]["cookies"] == {"session": "cookie-value"}
        # Verification ran only after the harness let go of the sandbox, then the sandbox was stopped.
        assert closes_before_verify == [1]
        assert sandbox.stopped and not server._sandboxes and not server._sessions
        assert (Path(result.verifier_logs_dir) / "reward.txt").read_text() == "1\n"

    def test_run_endpoint_returns_the_episode_without_a_sandbox_handle(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        with TestClient(server.setup_webserver()) as client:
            response = client.post("/run", json=episode(task).model_dump(mode="json"))
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["result"]["reward"] == 1.0 and payload["failure"] is None
        assert payload["episode_id"] == {"rollout_id": "r1", "attempt": 0}
        assert "sb-1" not in response.text and "sandbox" not in payload["result"]
        assert sandbox.stopped

    async def test_episodes_parse_and_hash_the_task_once(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        loads: list[Path] = []
        real_load_task = load_task

        def counting_load_task(folder):
            loads.append(Path(folder))
            return real_load_task(folder)

        monkeypatch.setattr("environment_servers.harbor.app.load_task", counting_load_task)

        assert (await run_episode(server, task)).result.reward == 1.0
        assert (await run_episode(server, task, rollout="r2")).result.reward == 1.0

        assert created == [("hello", "/app"), ("hello", "/app")]
        assert len(loads) == 1
        assert loads[0] != task.path
        assert load_task(loads[0]).digest == task.digest

    @pytest.mark.parametrize(
        ("selector", "fragment"),
        [
            ({"taskset": "nope"}, "not served"),
            ({"task_id": "missing"}, "not in taskset"),
            ({"digest": "0" * 64}, "does not match the taskset mapping"),
        ],
    )
    async def test_bad_identity_is_a_terminal_seed_failure(self, tmp_path, monkeypatch, selector, fragment):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        response = await run_episode(server, task, **selector)
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert fragment in failure.failure_reason
        assert created == [] and server.server_client.calls == []

    async def test_changed_folder_is_a_terminal_seed_failure(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        (task.path / "tests" / "test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and response.failure.terminal
        assert "changed since materialization" in response.failure.failure_reason
        assert created == []

    async def test_host_copy_failure_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        """A failed snapshot copy (disk full, I/O error) is not a task defect: the collector may retry it."""
        server, task, _, created = make_server(tmp_path, monkeypatch)

        def no_space(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(shutil, "copytree", no_space)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "No space left on device" in response.failure.failure_reason
        assert created == [] and server.server_client.calls == []

    async def test_malformed_task_folder_is_a_terminal_seed_failure(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text("[verifier\n")
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "Could not snapshot task" in response.failure.failure_reason
        assert created == []

    async def test_sandbox_failure_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)

        async def boom(task, workdir, *, episode_key):
            raise RuntimeError("pull failed")

        monkeypatch.setattr(server, "_create_sandbox", boom)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "pull failed" in response.failure.failure_reason
        assert server.server_client.calls == []

    async def test_shutdown_refuses_new_episodes(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        await server.shutdown()
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "shutting down" in response.failure.failure_reason and created == []

    def test_sandbox_spec_from_task(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}, "default_metadata": {"sandbox-api": "osb"}}},
        )

        spec = server._sandbox_spec(task, "/app")

        assert spec.image == "ubuntu:24.04"
        assert spec.workdir == "/app"
        assert spec.ttl_s == 21600 + 10 + server.config.sandbox_ttl_slack_s
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (1.0, 2048, 10)
        assert spec.metadata["sandbox-api"] == "osb"
        assert spec.metadata["harbor_task"] == "hello"
        assert spec.metadata["nemo_gym_environment_server"] == "harbor_environment_server"


class TestAgentStage:
    async def test_lost_agent_seed_is_a_retryable_agent_failure_and_stops_the_sandbox(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/v1/agent_sessions"

        response = await run_episode(server, task)

        assert response.failure.stage == "agent" and not response.failure.terminal
        assert "response lost" in response.failure.failure_reason
        # The harness never worked, so nothing was verified; the sandbox and the agent session are closed anyway.
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert server.server_client.paths() == ["/v1/agent_sessions", "/v1/agent_sessions/close"]
        assert sandbox.stopped and not server._sandboxes

    @pytest.mark.parametrize(("status", "terminal"), [(422, True), (503, False), (429, False)])
    async def test_agent_http_errors_are_terminal_only_for_client_faults(
        self, tmp_path, monkeypatch, status, terminal
    ):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        request_info = RequestInfo(URL("http://agent/v1/responses"), "POST", CIMultiDictProxy(CIMultiDict()))
        server.server_client.fail_with["/ng-rollout/r1/v1/responses"] = ClientResponseError(
            request_info, (), status=status, message="agent said no"
        )

        response = await run_episode(server, task)

        assert response.failure.stage == "agent" and response.failure.terminal is terminal
        assert "agent said no" in response.failure.failure_reason
        assert sandbox.stopped and not server._sandboxes

    async def test_failed_turn_closes_the_session_and_keeps_the_agents_evidence(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/ng-rollout/r1/v1/responses"
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}

        response = await run_episode(server, task)

        assert response.failure.stage == "agent" and not response.failure.terminal
        assert response.failure.agent_observations is not None
        assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert sandbox.stopped

    async def test_failed_close_is_a_cleanup_failure_with_the_turn_attached(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/v1/agent_sessions/close"

        response = await run_episode(server, task)

        assert response.failure.stage == "cleanup" and not response.failure.terminal
        assert response.failure.partial_response.id == "response"
        # Verification never ran: the harness may still have been working in the sandbox.
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        # The final unwind retried the close, then stopped the sandbox.
        assert server.server_client.paths().count("/v1/agent_sessions/close") == 2
        assert sandbox.stopped

    async def test_observations_travel_with_the_result(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}
        response = await run_episode(server, task)
        assert response.result.ng_agent_observations is not None
        assert response.result.model_dump()["ng_agent_observations"]["source"] == "harness"

    async def test_episode_deadline_stops_the_sandbox(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, default_episode_timeout_seconds=0.1)
        server.server_client.slow_path = "/ng-rollout/r1/v1/responses"

        response = await run_episode(server, task)

        assert response.failure is not None and "timed out" in response.failure.failure_reason
        assert not response.failure.terminal
        assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
        assert sandbox.stopped and not server._sandboxes


class TestVerify:
    async def verified(self, tmp_path, monkeypatch, sandbox=None, **kwargs):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, sandbox, **kwargs)
        response = await run_episode(server, task)
        return server, task, sandbox, response

    async def test_verify_runs_test_sh_and_reads_reward(self, tmp_path, monkeypatch):
        server, task, sandbox, response = await self.verified(tmp_path, monkeypatch)

        result = response.result
        assert result.reward == 1.0 and result.mask_sample is False and result.failure_kind is None
        assert result.verifier_rewards == {"reward": 1.0} and result.verifier_return_code == 0

        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert run["command"] == "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"
        assert run["cwd"] == "/app" and run["timeout_s"] == 120.0
        commands = [call["command"] for call in sandbox.execs]
        wipe = next(i for i, command in enumerate(commands) if "-mindepth 1 -delete" in command)
        upload = next(i for i, command in enumerate(commands) if "tar -xzf" in command and "/tests" in command)
        assert wipe < upload < commands.index(run["command"])
        # An image may have a non-root default even when the task has no user override.
        prepare = next(
            call for call in sandbox.execs if "chmod 777" in call["command"] and "/tests" in call["command"]
        )
        assert prepare["user"] == "root" and prepare["cwd"] == "/"
        # tests/ was uploaded as an archive and unpacked into /tests.
        assert any(remote.endswith(".tar.gz") for _, remote in sandbox.uploads)
        assert Path(result.verifier_logs_dir).is_absolute()
        assert (Path(result.verifier_logs_dir) / "reward.txt").read_text() == "1\n"

    async def test_prepare_runs_as_root_on_a_non_root_image(self, tmp_path, monkeypatch):
        class NonRootSandbox(FakeSandbox):
            """The image's default user may not touch root-owned /logs/verifier."""

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                result = await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
                if ("chmod" in command or "chown" in command) and user != "root":
                    return SandboxExecResult(stdout="", stderr="chmod: Permission denied", return_code=1)
                return result

        server, task, sandbox, response = await self.verified(
            tmp_path,
            monkeypatch,
            NonRootSandbox(),
            toml=TASK_TOML.replace("[environment]", '[environment]\ndocker_image = "nonroot:latest"'),
        )
        assert task.user is None
        assert response.result.reward == 1.0 and response.result.failure_kind is None
        prepare = next(
            call for call in sandbox.execs if "chmod 777" in call["command"] and "/tests" in call["command"]
        )
        assert prepare["user"] == "root"
        # test.sh itself still runs as the verifier's user, not root.
        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert run["user"] == task.config.verifier.user

    async def test_json_reward_with_components(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"reward": 0.5, "tests_passed": 3})})
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.5
        assert response.result.verifier_rewards == {"reward": 0.5, "tests_passed": 3.0}

    async def test_ambiguous_reward_is_a_terminal_verification_failure(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"a": 1, "b": 0})})
        _, _, sandbox, response = await self.verified(tmp_path, monkeypatch, sandbox)
        failure = response.failure
        assert failure.stage == "verification" and failure.terminal
        assert "several keys" in failure.failure_reason
        assert failure.partial_response.id == "response"
        assert sandbox.stopped

    async def test_missing_reward_scores_zero_and_is_measured(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={"test-stdout.txt": "pytest exploded"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=1),
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is False
        assert result.failure_kind == "harbor:missing_reward"
        assert "pytest exploded" in result.failure_reason
        assert result.verifier_return_code == 1

    async def test_invalid_reward_scores_zero(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.txt": "maybe"})
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.0 and response.result.failure_kind == "harbor:invalid_reward"

    async def test_verifier_timeout_scores_zero(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            test_result=SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout")
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.0 and response.result.mask_sample is False
        assert response.result.failure_kind == "harbor:verifier_timeout"

    async def test_sandbox_runtime_failure_masks(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            test_result=SandboxExecResult(stdout=None, stderr="gone", return_code=125, error_type="sandbox")
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.0 and response.result.mask_sample is True
        assert response.result.failure_kind == "verifier_error"

    async def test_transfer_exception_masks(self, tmp_path, monkeypatch):
        class BrokenUpload(FakeSandbox):
            async def upload(self, local_path, remote_path):
                raise ConnectionError("lost")

        _, _, _, response = await self.verified(tmp_path, monkeypatch, BrokenUpload())
        assert response.result.mask_sample is True
        assert response.result.failure_kind == "provider_unavailable"
        assert "lost" in response.result.failure_reason

    async def test_aggregate_metrics_scores_the_episode_results(self, tmp_path, monkeypatch):
        server, task, _, response = await self.verified(tmp_path, monkeypatch)
        row = response.result.model_dump(mode="json")
        rows = [
            row | {"_ng_task_index": 0, "_ng_rollout_index": 0},
            row | {"_ng_task_index": 0, "_ng_rollout_index": 1, "reward": 0.0},
        ]
        metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
        assert metrics.agent_metrics["mean/reward"] == 0.5


class TestOwnership:
    async def test_shutdown_stops_sandboxes_of_episodes_in_flight(self, tmp_path, monkeypatch):
        class Counting(FakeSandbox):
            stops = 0

            async def stop(self):
                self.stops += 1
                await super().stop()

        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, Counting())
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold():
            entered.set()
            await release.wait()

        server.server_client.turn_hooks.append(hold)
        running = asyncio.create_task(run_episode(server, task))
        await asyncio.wait_for(entered.wait(), 5)
        assert server._sandboxes and not sandbox.stopped
        await server.shutdown()
        assert sandbox.stopped and not server._sandboxes
        release.set()
        response = await running
        # The verifier does not score an episode that raced shutdown: shutdown removed the task snapshots, so
        # the verifier's upload fails and the sample is masked rather than trained on. The episode's own
        # cleanup finds nothing left to stop; shutdown's stop was the only one.
        assert response.result is not None and response.result.mask_sample is True
        assert response.result.failure_kind == "provider_unavailable"
        assert sandbox.stops == 1
        assert not server._sandboxes


class TestSeedGuards:
    """Defects from the first cluster run: a seed the episode deadline cancels leaves its sandbox to the cleanup."""

    async def test_cancelled_seed_leaves_the_stop_to_the_bounded_cleanup(self, tmp_path, monkeypatch):
        """The episode deadline fires during setup while the provider's stop hangs.

        The seed must not stop the sandbox inline (the deadline cancels only once, so that stop would run
        unbounded and keep the episode key active); the registered cleanup stops it under the cleanup
        timeout, the request returns, and the sandbox stays owned for the shutdown retry.
        """
        release = asyncio.Event()

        class HangingSandbox(FakeSandbox):
            stop_calls = 0

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command.startswith("mkdir"):
                    await asyncio.get_running_loop().create_future()
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

            async def stop(self):
                self.stop_calls += 1
                await release.wait()
                self.stopped = True

        sandbox = HangingSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox, default_episode_timeout_seconds=0.1)
        server.config.cleanup_timeout_seconds = 0.2

        response = await asyncio.wait_for(run_episode(server, task), 5)

        assert response.failure is not None and "timed out" in response.failure.failure_reason
        assert not response.failure.terminal
        assert task.task_id not in server._active and not server._active
        # The cleanup attempted the stop once, gave up at its bound, and kept ownership for a later retry.
        assert sandbox.stop_calls == 1 and not sandbox.stopped
        assert list(server._sandboxes.values()) == [sandbox]
        assert server.server_client.calls == []
        release.set()
        await server.shutdown()
        assert sandbox.stopped and not server._sandboxes


class TestOracleSolution:
    """The seed places the task's reference solution for the Harbor oracle, and for no other harness.

    Harbor's oracle (0.22.0 ``agents/oracle.py``) copies ``solution/`` to ``/solution`` and runs
    ``bash /solution/solve.sh``; here the environment server does the copy, as root like its other seed
    uploads, before the oracle's session opens. The oracle reads no task folder.
    """

    SOLUTION_TOML = (
        TASK_TOML
        + """
[solution.env]
GREETING = "hi"
"""
    )

    @staticmethod
    def solution_uploads(sandbox) -> list[dict]:
        return [c for c in sandbox.execs if "tar -xzf" in c["command"] and "-C /solution" in c["command"]]

    async def test_the_oracle_gets_the_solution_before_its_session_opens(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, toml=self.SOLUTION_TOML, agent_implementation="harbor_oracle_agent", solution=True
        )
        uploads_at_seed: list[int] = []
        post = type(server.server_client).post

        async def post_recording_uploads(client, server_name, url_path, **kwargs):
            if url_path == "/v1/agent_sessions":
                uploads_at_seed.append(len(self.solution_uploads(sandbox)))
            return await post(client, server_name, url_path, **kwargs)

        monkeypatch.setattr(type(server.server_client), "post", post_recording_uploads)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        # One archive, extracted as root into /solution, then opened to every user (the task user runs it).
        (upload,) = self.solution_uploads(sandbox)
        assert upload["user"] == "root"
        assert "chmod -R a+rX /solution" in upload["command"]
        # The whole folder travels with its tree, and nothing the task digest ignores.
        names = set()
        for data in sandbox.uploaded_archives:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                members = {m.name for m in tar.getmembers()}
                if "solve.sh" in members:
                    names = members
                    modes = {m.name: m.mode & 0o777 for m in tar.getmembers()}
        # The upload takes the folder as is here; the digest's ignore rule joins the folder uploads with the
        # Compose volumes work, which filters a stray __pycache__ out of every task upload.
        assert names == {"solve.sh", "data", "data/answer.txt", "__pycache__", "__pycache__/x.pyc"}
        assert modes["solve.sh"] == 0o700  # the archive keeps the author's mode
        assert uploaded_file(sandbox, "solve.sh") == [SOLVE_SH.encode()]
        # The upload preceded the agent seed, which carries the task user and the resolved [solution].env.
        assert uploads_at_seed == [1]
        seed = server.server_client.seed_bodies()[0]
        assert "user" not in seed["sandbox_access"]  # the task declares no user, so none travels
        assert seed["sandbox_access"]["env"] == {"GREETING": "hi"}
        assert seed["agent_timeout_sec"] == 300.0

    @pytest.mark.parametrize("implementation", ["hermes_agent", "test"])
    async def test_other_harnesses_never_see_the_solution(self, tmp_path, monkeypatch, implementation):
        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, toml=self.SOLUTION_TOML, agent_implementation=implementation, solution=True
        )

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert self.solution_uploads(sandbox) == []
        assert uploaded_file(sandbox, "solve.sh") == []
        assert not any("/solution" in c["command"] for c in sandbox.execs)
        seed = server.server_client.seed_bodies()[0]
        assert "env" not in seed["sandbox_access"]

    async def test_a_task_without_a_solution_uploads_nothing_for_the_oracle(self, tmp_path, monkeypatch):
        """The oracle finds no /solution/solve.sh and reports the task unvalidated; the seed is not at fault."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, agent_implementation="harbor_oracle_agent")
        assert not (task.path / "solution").exists()

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert self.solution_uploads(sandbox) == []
        assert not any("/solution" in c["command"] for c in sandbox.execs)
        assert "env" not in server.server_client.seed_bodies()[0]["sandbox_access"]

    async def test_the_oracles_solution_env_resolves_on_the_host_at_seed(self, tmp_path, monkeypatch):
        """`${VAR}` in [solution].env is the host's value for the oracle; a missing one is a terminal seed failure.

        A model harness never runs the solution, so the same task runs for it whatever the host lacks.
        """
        toml = self.SOLUTION_TOML.replace('GREETING = "hi"', 'GREETING = "${HARBOR_TEST_GREETING}"')
        monkeypatch.delenv("HARBOR_TEST_GREETING", raising=False)
        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, toml=toml, agent_implementation="harbor_oracle_agent", solution=True
        )

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and response.failure.terminal
        assert "HARBOR_TEST_GREETING" in response.failure.failure_reason
        assert server.server_client.paths() == []
        assert sandbox.stopped and not server._sandboxes

        monkeypatch.setenv("HARBOR_TEST_GREETING", "from host")
        response = await run_episode(server, task, rollout="r2")
        assert response.failure is None, response.failure
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["env"] == {"GREETING": "from host"}

        monkeypatch.delenv("HARBOR_TEST_GREETING")
        model_server, model_task, _, _ = make_server(
            tmp_path / "model", monkeypatch, toml=toml, agent_implementation="hermes_agent", solution=True
        )
        response = await run_episode(model_server, model_task)
        assert response.failure is None, response.failure

    async def test_a_private_solution_is_readable_and_runnable_by_the_non_root_task_user(self, tmp_path, monkeypatch):
        """The author's checkout kept ``solution/`` private (a 0700 script, a 0700 folder, a 0600 input) and the
        user the oracle runs as, the task's ``[agent].user`` or else the image's default user, is not root. The
        extraction runs as root and keeps those modes, so without the fix-up that user could neither read the
        script nor enter the folder; the seed opens the tree, and the oracle's commands as that user succeed."""

        class ModeAwareSandbox(FakeSandbox):
            """Models the extracted tree: root extracts with --no-same-owner (owner root, archive modes kept), a
            ``chmod -R a+rX`` opens it, and a command run as any other user is refused where ``other`` lacks the
            read bit on the file or the search bit on a folder on the way."""

            default_user = "runner"

            def __init__(self):
                super().__init__()
                self.modes: dict[str, tuple[int, bool]] = {}
                self.extracted: dict[str, int] = {}

            def _allowed(self, path: str) -> bool:
                parts = path.strip("/").split("/")
                for depth in range(1, len(parts)):
                    mode, _ = self.modes.get("/" + "/".join(parts[:depth]), (0o755, True))
                    if not mode & 0o001:
                        return False
                mode, is_dir = self.modes.get(path, (0, False))
                return bool(mode & 0o004) and not is_dir

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
                if "tar -xzf" in command and "-C /solution" in command:
                    assert user == "root"
                    self.modes["/solution"] = (0o755, True)
                    with tarfile.open(fileobj=io.BytesIO(self.uploaded_archives[-1]), mode="r:gz") as tar:
                        for member in tar.getmembers():
                            self.modes[f"/solution/{member.name}"] = (member.mode & 0o777, member.isdir())
                    self.extracted = {path: mode for path, (mode, _) in self.modes.items()}
                    if "chmod -R a+rX /solution" in command:
                        for path, (mode, is_dir) in list(self.modes.items()):
                            self.modes[path] = (mode | 0o444 | (0o111 if is_dir else 0), is_dir)
                    return SandboxExecResult(stdout="", stderr="", return_code=0)
                if (user or self.default_user) != "root":
                    for path in [token for token in command.split() if token.startswith("/solution/")]:
                        if not self._allowed(path):
                            return SandboxExecResult(stdout="", stderr=f"{path}: Permission denied", return_code=126)
                return SandboxExecResult(stdout="ok\n", stderr="", return_code=0)

        sandbox = ModeAwareSandbox()
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            sandbox,
            toml=self.SOLUTION_TOML,
            agent_implementation="harbor_oracle_agent",
            solution=True,
        )
        assert (task.user or sandbox.default_user) != "root"
        solution = task.path / "solution"
        (solution / "data").chmod(0o700)
        (solution / "data" / "answer.txt").chmod(0o600)
        assert (solution / "solve.sh").stat().st_mode & 0o777 == 0o700

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        # What landed before the fix-up is the author's private tree, exactly as the archive carried it.
        assert sandbox.extracted["/solution/solve.sh"] == 0o700
        assert sandbox.extracted["/solution/data"] == 0o700
        assert sandbox.extracted["/solution/data/answer.txt"] == 0o600
        # The oracle's commands run as the task user (or the image's default user), not root, and succeed.
        probe = await sandbox.exec("test -f /solution/solve.sh", user=task.user)
        run = await sandbox.exec("bash /solution/solve.sh", user=task.user)
        read = await sandbox.exec("cat /solution/data/answer.txt", user=task.user)
        assert (probe.return_code, run.return_code, read.return_code) == (0, 0, 0), (probe, run, read)
        assert sandbox.modes["/solution/solve.sh"] == (0o744, False)
        assert sandbox.modes["/solution/data"] == (0o755, True)
        # The check is real: a tree left as extracted would have refused the same user.
        sandbox.modes = {path: (mode, sandbox.modes[path][1]) for path, mode in sandbox.extracted.items()}
        assert (await sandbox.exec("bash /solution/solve.sh", user=task.user)).return_code == 126
        assert (await sandbox.exec("cat /solution/data/answer.txt", user=task.user)).return_code == 126

    async def test_a_sandbox_that_cannot_take_the_solution_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        """Like the other seed uploads: the transfer failed in the provider's sandbox, so the episode is retried."""

        class RefusesSolution(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if "-C /solution" in command:
                    self.execs.append({"command": command, "cwd": cwd, "env": env, "user": user, "seen": None})
                    return SandboxExecResult(
                        stdout="", stderr="tar: /solution: Cannot mkdir: Permission denied", return_code=1
                    )
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        sandbox = RefusesSolution()
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            sandbox,
            toml=self.SOLUTION_TOML,
            agent_implementation="harbor_oracle_agent",
            solution=True,
        )

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "Permission denied" in response.failure.failure_reason
        assert server.server_client.paths() == []
        assert sandbox.stopped and not server._sandboxes

    def test_the_implementation_is_read_from_the_agent_instance_block(self, tmp_path, monkeypatch):
        server, _, _, _ = make_server(tmp_path, monkeypatch, agent_implementation="harbor_oracle_agent")
        assert server._agent_implementation() == "harbor_oracle_agent"
        # An instance the merged config does not show in that shape is not the oracle.
        server.server_client.global_config_dict = OmegaConf.create({"agent": {"responses_api_agents": {}}})
        assert server._agent_implementation() is None
        server.server_client.global_config_dict = OmegaConf.create({"agent": {"host": "x"}})
        assert server._agent_implementation() is None
        server.server_client.global_config_dict = OmegaConf.create({})
        assert server._agent_implementation() is None
