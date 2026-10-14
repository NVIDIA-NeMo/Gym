# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import json
import shutil
from pathlib import Path

import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from environment_servers.harbor.sandboxes import SEED_CREATE_GRACE_S
from environment_servers.harbor.tests.conftest import TASK_TOML, FakeSandbox, episode, make_server, run_episode
from environment_servers.harbor.verifier import (
    EXIT_RECORD_NAME,
    TIMEOUT_WRAPPER_GUARD,
    parse_exit_record,
    parse_reward_file,
    select_reward,
    verifier_command,
)
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.tasks.harbor import load_task


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

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("137 1700000000 1700000120\n", (137, 120)),
            ("0 1700000000 1700000001", (0, 1)),
            # An image without `date` leaves empty timestamps: the exit code is still known.
            ("137  \n", (137, None)),
            ("137 x y", (137, None)),
            ("", None),
            ("nope", None),
        ],
    )
    def test_exit_record(self, tmp_path, text, expected):
        (tmp_path / EXIT_RECORD_NAME).write_text(text)
        assert parse_exit_record(tmp_path) == expected

    def test_exit_record_missing(self, tmp_path):
        assert parse_exit_record(tmp_path) is None

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


class TestSetupCommands:
    async def test_setup_commands_run_as_root_after_workdir(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.config.sandbox_setup_commands = ["sed -i s/a/b/ /etc/apt/sources.list", "apt-get update || true"]
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        commands = [call["command"] for call in sandbox.execs]
        assert commands[0] == "mkdir -p /app"
        assert commands[1:3] == server.config.sandbox_setup_commands
        assert all(call["user"] == "root" for call in sandbox.execs[1:3])
        # The harness only gets its session once the sandbox is fully set up.
        assert server.server_client.calls

    async def test_failing_setup_command_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        class Failing(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command == "false":
                    return SandboxExecResult(stdout="", stderr="nope", return_code=1)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, Failing())
        server.config.sandbox_setup_commands = ["false"]
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "nope" in response.failure.failure_reason and sandbox.stopped
        assert server.server_client.calls == []


class TestSeedWorkdirAndResources:
    async def test_image_workdir_used_when_task_sets_none(self, tmp_path, monkeypatch):
        server, task, sandbox, created = make_server(tmp_path, monkeypatch)
        # A task with a real Dockerfile and a prebuilt image declares no workdir; the image's WORKDIR is used.
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN true\n")
        (task.path / "task.toml").write_text(TASK_TOML.replace("cpus = 1", 'docker_image = "org/task:1"\ncpus = 1'))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        assert task.workdir is None

        class PwdSandbox(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command == "pwd":
                    return SandboxExecResult(stdout="/work\n", stderr="", return_code=0)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        pwd_sandbox = PwdSandbox()

        async def create(task, workdir, *, episode_key=None):
            created.append((task.task_id, workdir))
            return pwd_sandbox

        monkeypatch.setattr(server, "_create_sandbox", create)
        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert created == [("hello", None)]
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["workdir"] == "/work"

    async def test_root_workdir_is_neither_created_nor_handed_over(self, tmp_path, monkeypatch):
        """An image whose WORKDIR is `/` with a user override: `/` keeps root's ownership."""
        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER runner\n")
        assert task.user == "runner" and task.workdir is None

        class RootPwdSandbox(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command == "pwd":
                    return SandboxExecResult(stdout="/\n", stderr="", return_code=0)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        def workdir_commands(sandbox) -> list[str]:
            # The /logs folders are prepared by the same helper; only the working directory commands matter here.
            return [call["command"] for call in sandbox.execs if "/logs" not in call["command"]]

        sandbox = RootPwdSandbox()
        assert await server._prepare_workdir(sandbox, task, None) == "/"
        # Only the `pwd` probe ran (answered above the recording fake): no `mkdir -p /`, no `chown ... /`.
        assert workdir_commands(sandbox) == []

        # A named directory below the root is still created and handed over.
        named = RootPwdSandbox()
        assert await server._prepare_workdir(named, task, "/work") == "/work"
        assert workdir_commands(named) == ["mkdir -p /work && chown runner /work"]

    def test_resources_override_and_cpu_env(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.cpu == 1.0 and spec.env["OMP_NUM_THREADS"] == "1"

        server.config.sandbox_resources_override = {"cpu": 4, "memory_mib": 16384, "disk_gib": 30}
        spec = server._sandbox_spec(task, "/app")
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (4.0, 16384, 30)
        assert spec.env["OMP_NUM_THREADS"] == "4"

        # The override merges over the task's resources, so a GPU request survives a CPU/memory override.
        task.config.environment.gpus = 1
        task.config.environment.gpu_types = ["H100"]
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.gpu == 1 and spec.resources.cpu == 4.0
        task.config.environment.gpus = None
        task.config.environment.gpu_types = None

        server.config.derive_cpu_env = False
        assert "OMP_NUM_THREADS" not in server._sandbox_spec(task, "/app").env


TASK_TOOL_CONTEXT_DECLARATIONS = {
    "mcp_servers": '[[environment.mcp_servers]]\nname = "playwright"\ntransport = "sse"\nurl = "http://api:3080/sse"\n',
    "skills_dir": 'skills_dir = "/app/.skills"\n',
}


def task_tool_context_toml(declaration: str) -> str:
    """TASK_TOML whose ``[environment]`` table also carries the given tool declaration."""
    return TASK_TOML + TASK_TOOL_CONTEXT_DECLARATIONS[declaration]


class TestAgentStage:
    @pytest.mark.parametrize("declaration", sorted(TASK_TOOL_CONTEXT_DECLARATIONS))
    async def test_task_tool_context_is_refused_to_a_harness_that_did_not_declare_support(
        self, tmp_path, monkeypatch, declaration
    ):
        """Terminus 2 never sees the task's MCP servers or skills: the run would score a silent zero."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=task_tool_context_toml(declaration))
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        failure = response.failure
        assert response.result is None and failure.stage == "agent" and failure.terminal
        assert "declares MCP servers or a skills directory" in failure.failure_reason
        assert "'agent' does not support" in failure.failure_reason
        assert "supports_task_tool_context" in failure.failure_reason
        # The harness was seeded and is closed again, but never took its turn, and nothing was verified.
        assert server.server_client.paths() == ["/v1/agent_sessions", "/v1/agent_sessions/close"]
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert sandbox.stopped and not server._sandboxes

    @pytest.mark.parametrize("declaration", sorted(TASK_TOOL_CONTEXT_DECLARATIONS))
    async def test_task_tool_context_runs_on_a_harness_that_declared_support(self, tmp_path, monkeypatch, declaration):
        """mini-SWE reads the task's MCP servers and skills from the sandbox, so the task is run and verified."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=task_tool_context_toml(declaration))
        assert server.server_client.supports_task_tool_context is True

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        assert server.server_client.paths() == [
            "/v1/agent_sessions",
            "/ng-rollout/r1/v1/responses",
            "/v1/agent_sessions/close",
        ]

    async def test_a_task_without_tool_declarations_runs_on_any_harness(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0

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
        # A harness that returns no trajectory leaves the key out of the result, as before.
        assert "ng_trajectory" not in response.result.model_dump(mode="json")

    async def test_the_harness_trajectory_travels_with_the_result(self, tmp_path, monkeypatch):
        """The harness's own decisions ride on its close and reach the result in the slot rollout collection reads.

        Rebuilding turns from the captured model calls would count helper calls (context recovery, summarization)
        as decisions and lose the command-batch count behind each one.
        """
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}
        server.server_client.trajectory = {
            "task_id": "hello",
            "rollout_id": "r1",
            "turns": [
                {
                    "invocation_id": "harness_1",
                    "task_id": "hello",
                    "rollout_id": "r1",
                    "turn_no": 1,
                    "timestamp": 1.0,
                    "question": "do it",
                    "answer": [],
                    "step_count": 3,
                    "model_calls": [
                        {"model_ref": {"type": "responses_api_models", "name": "policy"}, "response_id": "resp_0"}
                    ],
                }
            ],
        }

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        [turn] = response.result.ng_trajectory.turns
        assert (turn.turn_no, turn.step_count, turn.model_calls[0].response_id) == (1, 3, "resp_0")
        dumped = response.result.model_dump(mode="json")
        assert dumped["ng_trajectory"]["turns"][0]["step_count"] == 3
        assert dumped["ng_agent_observations"]["source"] == "harness"

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
        assert run["command"] == verifier_command(120)
        # The portable short form (`--signal=` fails on BusyBox), guarded so a missing wrapper exits 127
        # before anything is redirected, and followed by the exit record the server reads the deadline from.
        assert run["command"].startswith(f"{TIMEOUT_WRAPPER_GUARD}; start=$(date +%s); ")
        assert (
            "timeout -s KILL 120 bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1; rc=$?; " in run["command"]
        )
        assert run["command"].endswith('echo "$rc $start $(date +%s)" > /logs/verifier/.ng-exit; exit $rc')
        assert "--signal" not in run["command"]
        assert run["cwd"] == "/app"
        # The exec itself is bounded by the budget plus the grace period; the provider keeps it alive that long.
        assert run["timeout_s"] == 120 + server.config.verifier_grace_s
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

    async def test_verifier_timeout_when_exit_code_never_appears(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={},
            test_result=SandboxExecResult(stdout=None, stderr=None, return_code=125, error_type="timeout"),
        )
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, sandbox)
        server.config.verifier_grace_s = 0
        (task.path / "task.toml").write_text(
            TASK_TOML.replace("timeout_sec = 120.0\n\n[agent]", "timeout_sec = 0.01\n\n[agent]")
        )
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        result = (await run_episode(server, task)).result
        assert result.reward == 0.0 and result.failure_kind == "harbor:verifier_timeout"
        assert result.mask_sample is False
        # A sub-second budget is rounded up to one whole second: `timeout 0` would mean no limit at all.
        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert "timeout -s KILL 1 bash /tests/test.sh" in run["command"]

    async def test_exit_137_before_the_deadline_keeps_the_reward(self, tmp_path, monkeypatch):
        """A grader that writes its reward and then ends 137 (a killed helper under `wait`) did not time out."""
        sandbox = FakeSandbox(
            verifier_files={"reward.txt": "1\n"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=137),
            test_elapsed=5,
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 1.0 and result.mask_sample is False and result.failure_kind is None
        assert result.verifier_return_code == 137

    async def test_exit_137_at_the_deadline_without_a_reward_is_a_timeout(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={"test-stdout.txt": "still running"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=137),
            test_elapsed=120,
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is False
        assert result.failure_kind == "harbor:verifier_timeout"
        assert "still running" in result.failure_reason

    async def test_reward_written_before_the_kill_still_counts(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={"reward.txt": "0.5\n"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=137),
            test_elapsed=120,
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.5 and response.result.failure_kind is None

    async def test_exit_137_without_timestamps_falls_back_to_the_exit_code(self, tmp_path, monkeypatch):
        """An image without `date` leaves no elapsed time: 137 then means the timer, as before."""
        sandbox = FakeSandbox(
            verifier_files={EXIT_RECORD_NAME: "137  \n"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=137),
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.failure_kind == "harbor:verifier_timeout"

    async def test_missing_timeout_wrapper_masks_the_sample(self, tmp_path, monkeypatch):
        """BusyBox without `timeout`, or an image without `bash`: the guard exits 127 before test.sh starts."""
        sandbox = FakeSandbox(
            verifier_files={},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=127),
            wrapper_ran=False,
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is True
        assert result.failure_kind == "verifier_error"
        assert "could not be started" in result.failure_reason and "exit 127" in result.failure_reason

    async def test_test_sh_exiting_127_itself_is_measured(self, tmp_path, monkeypatch):
        """The wrapper ran: a script that ends 127 (a missing command of its own) is an ordinary measured zero."""
        sandbox = FakeSandbox(verifier_files={}, test_result=SandboxExecResult(stdout="", stderr="", return_code=127))
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.mask_sample is False and result.failure_kind == "harbor:missing_reward"
        assert result.verifier_return_code == 127

    async def test_sandbox_runtime_failure_masks(self, tmp_path, monkeypatch):
        class BrokenLaunch(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if "test.sh" in command:
                    return SandboxExecResult(stdout=None, stderr="gone", return_code=125, error_type="sandbox")
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        _, _, _, response = await self.verified(tmp_path, monkeypatch, BrokenLaunch())
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
    """Defects from the first cluster run: requests above limits, and a create the ready timeout did not bound."""

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

    async def test_seed_gives_up_on_a_create_at_the_ready_timeout(self, tmp_path, monkeypatch):
        """The configured ready timeout is a seed deadline, whatever the provider's own retry budget."""
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox_ready_timeout_s=0.05)
        monkeypatch.setattr("environment_servers.harbor.sandboxes.SEED_CREATE_GRACE_S", 0.0)
        stopped = []

        class Hanging:
            async def stop(self):
                stopped.append(True)

        async def hang(task, workdir, *, episode_key):
            server._sandboxes[episode_key] = Hanging()
            await asyncio.get_running_loop().create_future()

        monkeypatch.setattr(server, "_create_sandbox", hang)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "not ready within 0.05s" in response.failure.failure_reason
        assert "sandbox_ready_timeout_s=0.05" in response.failure.failure_reason
        assert stopped == [True] and not server._sandboxes
        assert server.server_client.calls == []
        assert SEED_CREATE_GRACE_S > 0

    async def test_a_providers_own_timeout_inside_the_deadline_is_reported_as_is(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox_ready_timeout_s=100)

        async def provider_timeout(task, workdir, *, episode_key):
            raise TimeoutError("Timed out creating OpenSandbox sandbox after 100s")

        monkeypatch.setattr(server, "_create_sandbox", provider_timeout)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "Timed out creating OpenSandbox sandbox after 100s" in response.failure.failure_reason
