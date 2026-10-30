# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import inspect
import io
import json
import posixpath
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
import yaml
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from yarl import URL

import environment_servers.harbor.overlay as overlay
import nemo_gym.sandbox.providers.base as sandbox_base
from environment_servers.harbor.app import (
    HarborEnvironmentServer,
    HarborEnvironmentServerConfig,
    HarborEpisodeRequest,
)
from environment_servers.harbor.sandboxes import SEED_CREATE_GRACE_S, _is_image_pull_failure
from environment_servers.harbor.tests.conftest import (
    COMPOSE_YAML,
    IMAGE_CONFIGS,
    IMAGE_PATH,
    OK,
    OVERLAY_DOCKERFILE,
    SEPARATE_IN_TASK_IMAGE_TOML,
    SEPARATE_TOML,
    SOLVE_SH,
    TASK_TOML,
    VERIFIER_TESTS_DOCKERFILE,
    AgentSandbox,
    FakeCompose,
    FakeSandbox,
    OverlaySandbox,
    _fails,
    agent_response,
    archive_of,
    episode,
    make_server,
    record_verifier_build,
    run_episode,
    write_image_config,
    write_task,
)
from environment_servers.harbor.verifier import (
    COLLECT_HOOK_GRACE_S,
    EXIT_RECORD_NAME,
    TIMEOUT_WRAPPER_GUARD,
    _verifier_image,
    _verifier_runs_task_image,
    _verifier_start,
    parse_exit_record,
    parse_reward_file,
    reward_components,
    select_reward,
    summing_components,
    verifier_command,
)
from nemo_gym import failure_kinds
from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcomePolicyConfig, OutcomeAction
from nemo_gym.base_environment_server import BaseEnvironmentServerConfig
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, cpu_cap_env
from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS
from nemo_gym.server_utils import ServerClient
from nemo_gym.tasks.harbor import load_task
from nemo_gym.tasks.harbor.digest import DIGEST_KEY
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.models import HarborEnvironment
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE


def harness_trajectory(step_count: int) -> dict:
    """A close's trajectory: one decision the harness took, with its model call and command-batch count."""
    return {
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
                "step_count": step_count,
                "model_calls": [
                    {"model_ref": {"type": "responses_api_models", "name": "policy"}, "response_id": "resp_0"}
                ],
            }
        ],
    }


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
        # Without a `reward` key every key is a component and the reward is their sum; with one, the other keys
        # are metrics and there are no components to name.
        assert select_reward(rewards) == 1.25 and reward_components(rewards) == rewards
        assert select_reward({"accuracy": 0.25}) == 0.25 and reward_components({"accuracy": 0.25}) == {
            "accuracy": 0.25
        }
        assert select_reward({"reward": 1.0, "other": 0.0}) == 1.0
        assert reward_components({"reward": 1.0, "other": 0.0}) is None
        # Finite keys whose sum is not: no reward.
        assert select_reward({"a": 1e308, "b": 1e308}) is None

    def test_components_must_sum_to_the_reward(self):
        assert summing_components(1.0, {"a": 0.5, "b": 0.5}) == {"a": 0.5, "b": 0.5}
        assert summing_components(0.3, {"a": 0.1, "b": 0.2}) == {"a": 0.1, "b": 0.2}  # within tolerance
        assert summing_components(1.0, {"a": 0.5}) is None
        assert summing_components(1.0, None) is None

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
            if "test.sh" in command and "timeout" in command:
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
        # The transcript reached /logs/agent/trajectory.json before test.sh ran, and a copy sits with the logs.
        (trajectory_bytes,) = uploaded_file(sandbox, "trajectory.json")
        assert any("-C /logs/agent;" in c["command"] for c in sandbox.execs if "tar -xzf" in c["command"])
        trajectory = json.loads(trajectory_bytes)
        assert trajectory["schema_version"] == "ATIF-v1.7"
        assert [step["source"] for step in trajectory["steps"]] == ["user", "agent"]
        assert trajectory["steps"][1]["message"] == "echo hello > hello.txt"
        assert trajectory["agent"] == {
            "name": "agent",
            "version": trajectory["agent"]["version"],
            "model_name": "model",
        }
        assert (Path(result.verifier_logs_dir) / "trajectory.json").read_bytes() == trajectory_bytes
        assert result.verifier_network == "public"
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

    async def test_a_flat_single_turn_row_runs_as_the_canonical_one(self, tmp_path, monkeypatch):
        """Review (CI bot, round 34): a single-turn `/run` request with the task fields beside
        `responses_create_params` (the flat shape `SingleAgentTurnTaskInput` accepted before `HarborTaskInput`
        replaced it) was rejected by the request schema before execution. It is the canonical row once validated,
        and runs as it does."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        canonical = episode(task)
        flat = HarborEpisodeRequest.model_validate(
            {
                "episode_id": {"rollout_id": "r1", "attempt": 0},
                "task": {
                    "task_id": {"taskset": "ds", "task_id": "hello"},
                    "task_input": {
                        "responses_create_params": {"input": [{"role": "user", "content": "Create hello.txt"}]},
                        DIGEST_KEY: task.digest,
                    },
                },
            }
        )
        assert flat.task.task_input == canonical.task.task_input
        assert flat.model_dump(mode="json", exclude_unset=True) == canonical.model_dump(
            mode="json", exclude_unset=True
        )

        response = await server.run_request(flat)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_rewards == {"reward": 1.0}
        assert any("test.sh" in call["command"] and "timeout" in call["command"] for call in sandbox.execs)

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

    async def test_dataset_toml_edit_after_materialization_is_refused(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        dataset_file = task.path.parent / "dataset.toml"
        dataset_file.write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        # The rows were materialized with the multiplier at 2; the file is edited before the server runs.
        dataset_file.write_text("[gym.defaults]\ntimeout_multiplier = 4.0\n")
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "changed since materialization" in response.failure.failure_reason and created == []
        # Restored, the same mapping serves the task again.
        dataset_file.write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        assert (await run_episode(server, task, rollout="r2")).failure is None

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

    def test_sandbox_spec_starts_the_pinned_base_image(self, tmp_path, monkeypatch):
        """An overlay-mode task starts the immutable reference its base configuration was recorded at (the index
        digest behind the tag the Dockerfile wrote), with the recorded settings resolved from it."""
        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM debian:bookworm-slim\nRUN true")
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}, "default_metadata": {"sandbox-api": "osb"}}},
        )
        pinned = "library/debian@sha256:" + "d" * 64
        path = task.path.parent / IMAGE_CONFIGS_FILE
        recorded = json.loads(path.read_text())
        recorded["debian:bookworm-slim"]["image"] = pinned
        path.write_text(json.dumps(recorded))
        task = load_task(task.path)
        assert task.image == pinned and [run.command for run in task.overlay] == ["true"]

        spec = server._sandbox_spec(task, "/app")

        assert spec.image == pinned and spec.workdir == "/app"

    def test_sandbox_spec_from_task(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}, "default_metadata": {"sandbox-api": "osb"}}},
        )

        spec = server._sandbox_spec(task, "/app")

        assert spec.image == "ubuntu:24.04"
        assert spec.workdir == "/app"
        # The sandbox lives for the task's own budget (sandbox start, harness setup, agent, verifier and its
        # grace), which is far below the run-wide deadline here, plus the cleanup budget and the slack.
        assert server._task_budget_s(task) == 900 + 300 + 300 + 120 + 30
        assert spec.ttl_s == server._task_budget_s(task) + 10 + server.config.sandbox_ttl_slack_s
        server.config.enforce_task_timeouts = False
        assert server._sandbox_spec(task, "/app").ttl_s == 21600 + 10 + server.config.sandbox_ttl_slack_s
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (1.0, 2048, 10)
        assert spec.metadata["sandbox-api"] == "osb"
        assert spec.metadata["harbor_task"] == "hello"
        assert spec.metadata["nemo_gym_environment_server"] == "harbor_environment_server"
        assert spec.user is None  # no USER in the Dockerfile and no [agent].user: the image's own default

    def test_sandbox_spec_user_is_the_dockerfile_user_for_the_task_image_only(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER agent\n")
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        assert task.user == "agent"
        assert server._sandbox_spec(task, "/app").user == "agent"
        # A root-overlay step still asks for root explicitly when the sandbox default is agent.
        assert server._sandbox_spec(task, "/app", image=task.image).user == "agent"
        # A separate verifier in a different image keeps that image's own default user.
        assert server._sandbox_spec(task, None, image="org/verifier:1", role="verifier").user is None


class TestDefaultUser:
    async def test_numeric_agent_user_without_an_overlay_is_resolved_by_name(self, tmp_path, monkeypatch, caplog):
        toml = TASK_TOML.replace("[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "1000"')
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        assert task.user == "1000" and not task.overlay
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        # No overlay, but a raw uid is never the facade's default (OpenSandbox would `su` to it): the sandbox is
        # created without one and the seed resolves the uid to a name once, as root on the plain exec path.
        assert server._sandbox_spec(task, "/app").user is None
        assert (await run_episode(server, task)).failure is None
        lookups = [c for c in sandbox.execs if "getent" in c["command"]]
        assert [c["command"] for c in lookups] == ["getent passwd 1000 | cut -d: -f1"]
        assert lookups[0]["user"] is None  # the plain exec path: the image starts as root
        assert sandbox.default_user == "agent"

        # A uid that names no user is installed as the integer uid (OpenSandbox's native `uid=` path, not a
        # string it would `su` to), with a warning.
        sandbox.passwd.pop("1000")
        sandbox.default_user = None
        with caplog.at_level("WARNING"):
            assert (await run_episode(server, task, rollout="r2")).failure is None
        assert sandbox.default_user == 1000 and "USER 1000 names no user" in caplog.text

    async def test_an_unnamed_numeric_agent_user_reaches_the_harness_as_the_integer_uid(self, tmp_path, monkeypatch):
        """Review (CI bot, round 34): the harness was handed `[agent].user` as the task wrote it, the string
        `"4242"`, while the seed had installed the integer uid (no passwd entry) as the sandbox's default; OpenSandbox
        takes the string for an account name. The harness gets the user as installed: the integer."""
        toml = TASK_TOML.replace("[agent]\ntimeout_sec = 300.0", "[agent]\ntimeout_sec = 300.0\nuser = 4242")
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        assert task.user == "4242"
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == 4242
        (seed,) = server.server_client.seed_bodies()
        assert seed["sandbox_access"]["user"] == 4242 and type(seed["sandbox_access"]["user"]) is int

    async def test_named_user_is_set_as_the_default_after_seed(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER agent\n")
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == "agent"
        assert not any("getent" in c["command"] for c in sandbox.execs)


class TestVerifierUser:
    """test.sh runs as the verifier's own user: ``[verifier].user``, else the image's USER, never ``[agent].user``."""

    AGENT_OVERRIDE = TASK_TOML.replace("[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "agent"')

    @staticmethod
    def run_of(sandbox: FakeSandbox) -> dict:
        """The test.sh run itself, not the probe for the file that a separate verifier makes first."""
        return next(call for call in sandbox.execs if "bash /tests/test.sh" in call["command"])

    async def test_shared_verifier_runs_as_the_image_user_not_the_agent_override(self, tmp_path, monkeypatch):
        # A root-default image with an `agent` account: the agent's commands run as agent, test.sh as root.
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=self.AGENT_OVERRIDE)
        assert (task.user, task.entrypoint_user, task.config.verifier.user) == ("agent", None, None)
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == "agent"
        assert self.run_of(sandbox)["user"] == "root"

    async def test_shared_verifier_on_a_non_root_image_runs_as_that_user(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=self.AGENT_OVERRIDE,
            dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nUSER app",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "app"},
        )
        assert (task.user, task.entrypoint_user) == ("agent", "app")
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == "agent" and self.run_of(sandbox)["user"] == "app"

    async def test_explicit_verifier_user_still_wins(self, tmp_path, monkeypatch):
        toml = self.AGENT_OVERRIDE.replace("[verifier]\n", '[verifier]\nuser = "tester"\n')
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml)
        assert (await run_episode(server, task)).failure is None
        assert self.run_of(sandbox)["user"] == "tester"

    def separate(self, tmp_path, monkeypatch, *, toml, verifier=None, **kwargs):
        agent = AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=toml, **kwargs)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        verifier = verifier or FakeSandbox()
        specs: list[SandboxSpec] = []

        async def create_verifier(task, *, episode_key=None):
            # What the real `_create_verifier_sandbox` would create the verifier with.
            specs.append(server._sandbox_spec(task, task.workdir, image=_verifier_image(task), role="verifier"))
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier, specs

    @staticmethod
    def task_image_agent_override() -> str:
        # The task's own image declared: the loader admits the separate-in-task-image shape with a declared image
        # only. A method, as SEPARATE_IN_TASK_IMAGE_TOML comes later in the module.
        return SEPARATE_IN_TASK_IMAGE_TOML.replace(
            "[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "agent"'
        ).replace("[environment]\n", '[environment]\ndocker_image = "ubuntu:24.04"\n')

    async def test_separate_verifier_in_the_task_image_keeps_that_image_user(self, tmp_path, monkeypatch):
        toml = self.task_image_agent_override()
        server, task, agent, verifier, specs = self.separate(tmp_path, monkeypatch, toml=toml)
        assert _verifier_image(task) == task.image and task.user == "agent" and task.entrypoint_user is None
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        # The agent's sandbox got the override; the verifier's was created and kept with the image's own default.
        assert agent.default_user == "agent"
        assert specs[0].user is None and verifier.default_user is None
        assert self.run_of(verifier)["user"] == "root"

    async def test_fresh_separate_verifier_resolves_a_numeric_image_user_by_name(self, tmp_path, monkeypatch):
        # The image starts as uid 1000 (`svc` in its passwd); [agent].user is a different account. The verifier
        # is created without a raw uid as default, installs the name once it runs, and test.sh runs as the uid.
        toml = self.task_image_agent_override()
        verifier = FakeSandbox(passwd={"0": "root", "1000": "svc"}, image_user="svc")
        server, task, agent, verifier, specs = self.separate(
            tmp_path,
            monkeypatch,
            toml=toml,
            verifier=verifier,
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "1000"},
        )
        assert (task.user, task.entrypoint_user, task.sandbox_user) == ("agent", "1000", "1000")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert specs[0].user is None and verifier.default_user == "svc"
        lookup = next(c for c in verifier.execs if c["command"].startswith("getent passwd 1000"))
        assert lookup["user"] == "root"  # the image does not start as root, so root is asked for by name
        assert self.run_of(verifier)["user"] == 1000
        assert agent.default_user == "agent"

    @staticmethod
    def other_image_toml() -> str:
        # Defined here rather than as a class attribute: SEPARATE_IN_TASK_IMAGE_TOML comes later in the module.
        return SEPARATE_IN_TASK_IMAGE_TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n'

    async def test_separate_verifier_in_another_image_runs_as_that_image_default(self, tmp_path, monkeypatch):
        # The task image's USER is `app`; the verifier image is another one, whose own USER (maybe without an
        # `app` account) is the sandbox's default, so test.sh names no user, as `docker compose exec` would not.
        server, task, agent, verifier, specs = self.separate(
            tmp_path,
            monkeypatch,
            toml=self.other_image_toml(),
            dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nUSER app",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "app"},
        )
        assert _verifier_image(task) == "org/verifier:1" and not _verifier_runs_task_image(task)
        # Another declared image, which no Dockerfile's base record names: nothing is recorded for it, so nothing is
        # started in it (unchanged; only a declared image compose-images.json records starts, by that record).
        assert _verifier_start(task) is None
        assert (task.user, task.entrypoint_user) == ("app", "app")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert specs[0].user is None and verifier.default_user is None
        assert self.run_of(verifier)["user"] is None
        assert agent.default_user == "app"

    async def test_separate_verifier_in_another_image_ignores_the_agent_override(self, tmp_path, monkeypatch):
        # A root task image with [agent].user: the other-image verifier is still not asked for root by name.
        toml = self.other_image_toml().replace(
            "[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "agent"'
        )
        server, task, agent, verifier, specs = self.separate(tmp_path, monkeypatch, toml=toml)
        assert not _verifier_runs_task_image(task) and task.user == "agent" and task.entrypoint_user is None
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert specs[0].user is None and verifier.default_user is None
        assert self.run_of(verifier)["user"] is None
        assert agent.default_user == "agent"

    async def test_separate_verifier_in_another_image_still_takes_an_explicit_verifier_user(
        self, tmp_path, monkeypatch
    ):
        toml = self.other_image_toml().replace("[verifier]\n", '[verifier]\nuser = "tester"\n')
        server, task, _, verifier, specs = self.separate(tmp_path, monkeypatch, toml=toml)
        assert not _verifier_runs_task_image(task) and task.config.verifier.user == "tester"
        assert (await run_episode(server, task)).failure is None
        assert specs[0].user is None and self.run_of(verifier)["user"] == "tester"


class TestUserNormalisation:
    """``[agent].user`` spellings a provider cannot take raw are normalised before they become a default."""

    @staticmethod
    def with_agent_user(user: str) -> str:
        return TASK_TOML.replace("[agent]\ntimeout_sec = 300.0", f'[agent]\ntimeout_sec = 300.0\nuser = "{user}"')

    @staticmethod
    def stub_config(monkeypatch) -> None:
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

    async def test_uid_zero_is_root_and_needs_no_switch_on_a_root_image(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=self.with_agent_user("0"))
        self.stub_config(monkeypatch)
        assert task.user == "0"
        assert server._sandbox_spec(task, "/app").user is None
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user is None
        assert not any("getent" in c["command"] for c in sandbox.execs)

    async def test_root_override_on_a_non_root_image_is_asked_for_by_name(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=self.with_agent_user("0"),
            dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nUSER app",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "app"},
        )
        self.stub_config(monkeypatch)
        assert task.sandbox_user == "app"
        assert server._sandbox_spec(task, "/app").user == "root"
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == "root"

    async def test_a_declared_image_built_from_a_user_line_takes_the_root_override_by_name(
        self, tmp_path, monkeypatch
    ):
        """A declared docker_image with a pull-mode Dockerfile (`FROM python:3.12` / `USER app`) starts as `app`, the
        Dockerfile's final USER, not as the FROM image's root record says. So an `[agent] user = "root"` on it is a
        switch the sandbox has to ask for by name, as on any non-root image (`_normalised_user`), where a default
        derived from the FROM image would install no user at all and leave every exec running as `app`."""
        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            FakeSandbox(image_user="app"),
            toml=self.with_agent_user("root").replace(
                "[environment]\n", '[environment]\ndocker_image = "org/task:1"\n'
            ),
            dockerfile="FROM python:3.12\nUSER app\n",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "root"},
        )
        self.stub_config(monkeypatch)
        assert (task.image, task.sandbox_user, task.user) == ("org/task:1", "app", "root")
        assert server._sandbox_spec(task, "/app").user == "root"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert sandbox.default_user == "root"

    @staticmethod
    def docker_config(monkeypatch) -> None:
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"docker": {}}}
        )

    async def test_a_group_is_kept_on_docker(self, tmp_path, monkeypatch, caplog):
        """Review (round 14): `agent:shared` reached Docker as `--user agent`, with a warning, where Harbor's Docker
        environment passes `-u agent:shared` as written; the files readable only by the group were lost."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=self.with_agent_user("agent:shared"))
        self.docker_config(monkeypatch)
        with caplog.at_level("WARNING"):
            assert server._sandbox_spec(task, "/app").user == "agent:shared"
            assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == "agent:shared"
        assert "is not applied" not in caplog.text

    async def test_a_group_is_refused_on_a_provider_that_switches_user_by_name(self, tmp_path, monkeypatch):
        from environment_servers.harbor.users import HarborUserError

        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=self.with_agent_user("agent:shared"))
        self.stub_config(monkeypatch)
        # Refused before any sandbox exists (Daytona would create it for that user), and again where the
        # default is installed, as the episode's own terminal failure.
        with pytest.raises(
            HarborUserError, match="hello: user 'agent:shared' names group 'shared'; the 'opensandbox'"
        ):
            server._sandbox_spec(task, "/app")
        response = await run_episode(server, task)
        assert response.failure is not None and response.failure.terminal and response.failure.stage == "seed"
        reason = response.failure.failure_reason
        assert "hello: user 'agent:shared' names group 'shared'" in reason and "'opensandbox'" in reason
        assert "switches user by name only" in reason
        assert sandbox.default_user is None and sandbox.stopped

    async def test_a_numeric_uid_with_a_group_keeps_the_group_behind_the_name_on_docker(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=self.with_agent_user("1000:1000"))
        self.docker_config(monkeypatch)
        assert server._sandbox_spec(task, "/app").user is None  # a raw uid is never the creation default
        assert (await run_episode(server, task)).failure is None
        lookups = [c["command"] for c in sandbox.execs if "getent" in c["command"]]
        assert lookups == ["getent passwd 1000 | cut -d: -f1"]  # the name alone is looked up
        assert sandbox.default_user == "agent:1000"

        # A uid with no name keeps the pair as written: Docker's --user takes it.
        sandbox.passwd.pop("1000")
        sandbox.default_user = None
        assert (await run_episode(server, task, rollout="r2")).failure is None
        assert sandbox.default_user == "1000:1000"

    async def test_a_numeric_uid_with_a_group_is_refused_off_docker(self, tmp_path, monkeypatch):
        from environment_servers.harbor.users import HarborUserError

        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=self.with_agent_user("1000:1000"))
        self.stub_config(monkeypatch)
        with pytest.raises(HarborUserError, match="user '1000:1000' names group '1000'"):
            server._sandbox_spec(task, "/app")
        response = await run_episode(server, task)
        assert response.failure is not None and "names group '1000'" in response.failure.failure_reason
        assert not any("getent" in c["command"] for c in sandbox.execs)

    async def test_a_root_group_is_kept_on_docker(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=self.with_agent_user("root:1234"))
        self.docker_config(monkeypatch)
        assert server._sandbox_spec(task, "/app").user == "root:1234"
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == "root:1234"

    @staticmethod
    def verifier_in_a_group_image(tmp_path, monkeypatch):
        """A separate verifier graded in the task's own image, whose USER is `app:grp`; the agent runs as `agent`."""
        server, task, agent, verifier, specs = TestVerifierUser().separate(
            tmp_path,
            monkeypatch,
            toml=TestVerifierUser.task_image_agent_override(),
            dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nUSER app:grp",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "app:grp"},
        )
        assert _verifier_runs_task_image(task) and (task.user, task.entrypoint_user) == ("agent", "app:grp")
        return server, task, agent, verifier, specs

    async def test_a_verifier_image_group_is_refused_off_docker_before_the_rollout(self, tmp_path, monkeypatch):
        # Review (round 14): the verifier installs that image's own USER, `app:grp`, which OpenSandbox cannot run
        # as. The agent's own user passes the spec, but the verifier's is as static as the provider, so the refusal
        # is the seed's own terminal failure, before the agent's sandbox exists and the rollout runs, not a
        # verification failure after the whole rollout was paid for.
        server, task, agent, verifier, specs = self.verifier_in_a_group_image(tmp_path, monkeypatch)
        self.stub_config(monkeypatch)
        assert server._sandbox_spec(task, "/app").user == "agent"
        response = await run_episode(server, task)
        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert failure.partial_response is None
        assert (
            "hello: user 'app:grp' names group 'grp'" in failure.failure_reason
            and "'opensandbox'" in failure.failure_reason
        )
        # Nothing ran: no agent sandbox default, no exec, no verifier sandbox.
        assert agent.default_user is None and agent.execs == [] and specs == [] and verifier.execs == []

    async def test_a_verifier_image_group_is_kept_on_docker(self, tmp_path, monkeypatch):
        # The same task on Docker: the seed lets it through, and the verifier is created with and keeps the pair.
        server, task, agent, verifier, specs = self.verifier_in_a_group_image(tmp_path, monkeypatch)
        self.docker_config(monkeypatch)
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert agent.default_user == "agent"
        assert specs[0].user == "app:grp" and verifier.default_user == "app:grp"
        # test.sh runs as the pair too, not as `app` over the pair the default kept.
        assert TestVerifierUser.run_of(verifier)["user"] == "app:grp"

    @staticmethod
    def shared_in_a_group_image(tmp_path, monkeypatch, user: str, toml: str | None = None):
        """A shared-mode task whose image USER is the pair ``user`` and whose agent runs as `agent`."""
        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=toml or TestVerifierUser.AGENT_OVERRIDE,
            dockerfile=f"FROM ubuntu:24.04\nWORKDIR /app\nUSER {user}",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": user},
        )
        assert task.config.is_shared_verifier and (task.user, task.entrypoint_user) == ("agent", user)
        return server, task, sandbox

    @pytest.mark.parametrize("user", ["app:grp", "1000:grp"])
    async def test_shared_grading_runs_as_the_image_user_with_its_group_on_docker(self, tmp_path, monkeypatch, user):
        """Review (round 15): `_verifier_user` split the image USER at `:` and dispatched test.sh as `app` (a numeric
        one as the uid alone), over the `app:grp` default the sandbox had kept; Harbor's `docker compose exec -u`
        passes the pair."""
        server, task, sandbox = self.shared_in_a_group_image(tmp_path, monkeypatch, user)
        self.docker_config(monkeypatch)
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == "agent"
        assert TestVerifierUser.run_of(sandbox)["user"] == user

    async def test_shared_grading_as_a_group_user_is_refused_off_docker_before_the_rollout(
        self, tmp_path, monkeypatch
    ):
        # The agent's own user passes the spec; the image USER test.sh would run as is as static as the provider,
        # so the pair is refused at seed, as the separate verifier's is, not after the rollout at verification.
        server, task, sandbox = self.shared_in_a_group_image(tmp_path, monkeypatch, "app:grp")
        self.stub_config(monkeypatch)
        assert server._sandbox_spec(task, "/app").user == "agent"
        response = await run_episode(server, task)
        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "hello: user 'app:grp' names group 'grp'" in failure.failure_reason
        assert "'opensandbox'" in failure.failure_reason and "switches user by name only" in failure.failure_reason
        assert sandbox.execs == [] and sandbox.default_user is None

    @pytest.mark.parametrize("provider", ["docker", "opensandbox"])
    async def test_an_explicit_verifier_user_with_a_group_follows_the_provider(self, tmp_path, monkeypatch, provider):
        toml = TestVerifierUser.AGENT_OVERRIDE.replace("[verifier]\n", '[verifier]\nuser = "tester:grp"\n')
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml)
        assert task.config.verifier.user == "tester:grp" and task.entrypoint_user is None
        (self.docker_config if provider == "docker" else self.stub_config)(monkeypatch)
        response = await run_episode(server, task)
        if provider == "docker":
            assert response.failure is None and TestVerifierUser.run_of(sandbox)["user"] == "tester:grp"
        else:
            assert response.failure is not None and response.failure.stage == "seed"
            assert "hello: user 'tester:grp' names group 'grp'" in response.failure.failure_reason
            assert sandbox.execs == []


class TestSeedWorkdirAndResources:
    async def test_seed_creates_the_logs_folders_and_per_task_env(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        # Per-task environment comes from the dataset's own file, applied by the loader; the server sees plain tasks.
        (task.path.parent / "dataset.toml").write_text(
            '[gym.tasks."hello".environment]\nenv = { CIRCLE_NODE_TOTAL = "3" }\n[gym.tasks."other".environment]\nenv = { X = "1" }\n'
        )
        task = load_task(task.path)
        # The applied dataset slice is part of the digest the rows and the mapping carry.
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        seeded: list = []
        server.server_client.turn_hooks.append(lambda: seeded.append(server._sessions["r1"].task))
        assert (await run_episode(server, task)).failure is None
        commands = [c["command"] for c in sandbox.execs]
        assert commands[0] == "mkdir -p /app"
        logs = next(c for c in sandbox.execs if "/logs/artifacts" in c["command"])
        assert "chmod 777 /logs/agent /logs/verifier /logs/artifacts" in logs["command"] and logs["user"] == "root"
        # The server resolves the task from its own snapshot; the dataset's settings must reach that copy.
        assert seeded[0].path != task.path and seeded[0].env["CIRCLE_NODE_TOTAL"] == "3" and "X" not in seeded[0].env
        spec = server._sandbox_spec(seeded[0], "/app")
        assert spec.env["CIRCLE_NODE_TOTAL"] == "3" and "X" not in spec.env

    async def test_build_mode_task_seeds_from_its_recorded_image(self, tmp_path, monkeypatch):
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        # A Dockerfile that needs a build resolves to the image prepare recorded beside the task folders.
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY . /app\n")
        built = {context_digest(task.path / "environment"): {"image": "registry.example/ds-hello:abc"}}
        (task.path.parent / BUILT_IMAGES_FILE).write_text(json.dumps(built))
        task = load_task(task.path)
        assert task.image == "registry.example/ds-hello:abc"
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        seeded: list = []
        server.server_client.turn_hooks.append(lambda: seeded.append(server._sessions["r1"].task))

        response = await run_episode(server, task)

        # The server loads the task from its own snapshot, which must see the same record.
        assert response.failure is None, response.failure
        assert seeded[0].image == task.image

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
        assert task.user == "runner" and task.workdir in (None, "/")

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

    def test_resources_override_moves_the_cpu_hints_with_the_limit(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        spec = server._sandbox_spec(task, "/app")
        # The sandbox gets the task's [environment].env over the CPU-count hints sized from its limit, nothing else.
        assert spec.resources.cpu == 1.0 and spec.env == cpu_cap_env(1)

        server.config.sandbox_resources_override = {"cpu": 4, "memory_mib": 16384, "disk_gib": 30}
        spec = server._sandbox_spec(task, "/app")
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (4.0, 16384, 30)
        # The hints describe the limit the sandbox actually runs under, so an override moves them too.
        assert spec.env == cpu_cap_env(4) and spec.env["OMP_NUM_THREADS"] == "4"

        # The override merges over the task's resources, so a GPU request survives a CPU/memory override.
        task.config.environment.gpus = 1
        task.config.environment.gpu_types = ["H100"]
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.gpu == 1 and spec.resources.cpu == 4.0
        task.config.environment.gpus = None
        task.config.environment.gpu_types = None

    async def test_cpu_hints_reach_every_sandbox_and_the_tasks_own_env_wins(self, tmp_path, monkeypatch):
        """A task with `cpus = 2`: agent, Compose `main` and verifier specs all carry the hints for their own limit."""
        toml = SEPARATE_TOML.replace("cpus = 2\n", "cpus = 3\n").replace(
            "cpus = 1\n", 'cpus = 2\n\n[environment.env]\nOMP_NUM_THREADS = "8"\nTOOL = "x"\n'
        )
        server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        assert (task.config.environment.cpus, task.config.verifier.environment.cpus) == (2, 3)

        agent = server._sandbox_spec(task, "/app")
        assert agent.resources.cpu == 2.0 and agent.env["TOOL"] == "x"
        # Every hint names the task's two cores; the task's own OMP_NUM_THREADS wins over the hint.
        assert {name: agent.env[name] for name in CPU_CAP_ENV_VARS} == cpu_cap_env(2) | {"OMP_NUM_THREADS": "8"}
        assert agent.env == cpu_cap_env(2) | {"OMP_NUM_THREADS": "8", "TOOL": "x"}

        # The separate verifier is sized by [verifier.environment], and so are its hints; the agent's env is not its.
        verifier = server._sandbox_spec(
            task, None, environment=task.config.verifier.environment, image=_verifier_image(task), role="verifier"
        )
        assert verifier.resources.cpu == 3.0 and verifier.env == cpu_cap_env(3)
        assert verifier.env["OMP_NUM_THREADS"] == "3" and "TOOL" not in verifier.env

        # A Compose group's main service starts from the agent's spec, hints included; sidecars get none.
        (task.path / "task.toml").write_text(
            COMPOSE_TOML.replace("cpus = 1\n", "cpus = 2\n") + '\n[environment.env]\nOMP_NUM_THREADS = "8"\n'
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        compose_task = load_task(task.path)
        assert compose_task.config.environment.cpus == 2
        captured: dict = {}

        class CapturingCompose:
            def __init__(self, provider_config, path, *, service_specs, **kwargs):
                captured["specs"] = service_specs

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", CapturingCompose)
        await server._create_compose(compose_task, "r1/0")
        main, api = captured["specs"]["main"], captured["specs"]["api"]
        assert main.env == cpu_cap_env(2) | {"OMP_NUM_THREADS": "8"} and main.resources.cpu == 2.0
        assert api.env == {} and api.resources.cpu is None


TASK_TOOL_CONTEXT_DECLARATIONS = {
    "mcp_servers": '[[environment.mcp_servers]]\nname = "playwright"\ntransport = "sse"\nurl = "http://api:3080/sse"\n',
    "skills_dir": 'skills_dir = "/app/.skills"\n',
}
# A Compose sidecar's MCP server, the kind `task_mcp_grants` can grant (SWE-Interact's simulated user).
GRANTABLE_MCP_SERVER = (
    '[[environment.mcp_servers]]\nname = "user-server"\ntransport = "streamable-http"\n'
    'url = "http://user-server:8000/mcp"\n'
)


def http_error_with_body(status: int, body: bytes) -> ClientResponseError:
    """An HTTP error as ``raise_for_status`` raises it: the server's own reason is only in the response body."""
    request_info = RequestInfo(URL("http://agent/v1/agent_sessions"), "POST", CIMultiDictProxy(CIMultiDict()))
    error = ClientResponseError(request_info, (), status=status, message="Request failed")
    error.response_content = body
    return error


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

    @pytest.mark.parametrize("declaration", sorted(TASK_TOOL_CONTEXT_DECLARATIONS))
    async def test_the_oracle_declares_support_so_dataset_validate_runs_the_task(
        self, tmp_path, monkeypatch, declaration
    ):
        """`gym dataset validate` runs the oracle through this server; solve.sh reaches the task's tools by itself."""
        from unittest.mock import MagicMock

        from responses_api_agents.harbor_oracle_agent.app import OracleAgent, OracleAgentConfig

        oracle = OracleAgent(
            config=OracleAgentConfig(host="0.0.0.0", port=8081, entrypoint="", name="harbor_oracle_agent"),
            server_client=MagicMock(spec=ServerClient),
        )
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=task_tool_context_toml(declaration))
        server.server_client.supports_task_tool_context = oracle.supports_task_tool_context()

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        assert any("test.sh" in call["command"] for call in sandbox.execs)

    async def test_a_task_whose_mcp_servers_are_all_granted_as_required_runs_on_a_harness_without_task_json(
        self, tmp_path, monkeypatch
    ):
        """Hermes or Pi read no task.json, but take required MCP grants, or fail their seed when they cannot."""
        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, toml=TASK_TOML + GRANTABLE_MCP_SERVER, task_mcp_grants="required"
        )
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        (seed,) = server.server_client.seed_bodies()
        assert [(access["name"], access["required"]) for access in seed["tool_accesses"]] == [("user-server", True)]
        assert any("test.sh" in call["command"] for call in sandbox.execs)

    async def test_a_grant_the_tool_access_schema_rejects_is_a_staged_terminal_failure(self, tmp_path, monkeypatch):
        """The grant lookup runs inside the seed's try: a declaration the grant schema refuses (a name the session
        contract does not accept) fails the episode at stage agent, terminal, instead of escaping as a 500."""
        bad = GRANTABLE_MCP_SERVER.replace('name = "user-server"', 'name = "user server"')
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=TASK_TOML + bad, task_mcp_grants="required")
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        failure = response.failure
        assert response.result is None and failure.stage == "agent" and failure.terminal
        assert not server.server_client.seed_bodies()
        assert not any("test.sh" in call["command"] for call in sandbox.execs)

    @pytest.mark.parametrize(
        ("grants", "declaration", "unreached"),
        [
            ("off", None, "MCP server 'user-server'"),
            # An optional grant may be skipped by the harness without failing its seed.
            ("optional", None, "MCP server 'user-server'"),
            # Only streamable-http entries can be granted; an SSE server still needs task.json.
            ("required", "mcp_servers", "MCP server 'playwright'"),
            # A skills directory has no grant at all.
            ("required", "skills_dir", "skills_dir"),
        ],
    )
    async def test_task_tools_no_required_grant_covers_are_still_refused(
        self, tmp_path, monkeypatch, grants, declaration, unreached
    ):
        extra = TASK_TOOL_CONTEXT_DECLARATIONS[declaration] if declaration else ""
        # `skills_dir` belongs to the [environment] table, so it goes before the array-of-tables entry.
        toml = TASK_TOML + extra + GRANTABLE_MCP_SERVER
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml, task_mcp_grants=grants)
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        failure = response.failure
        assert response.result is None and failure.stage == "agent" and failure.terminal
        assert unreached in failure.failure_reason
        assert ("MCP server 'user-server'" in failure.failure_reason) is (grants != "required")
        # The grant hint is offered only when it would help: an ungranted streamable-http server and grants not
        # already required. An SSE server or a skills_dir under required grants gets no such advice.
        assert ("task_mcp_grants: required" in failure.failure_reason) is (grants != "required")
        assert not any("test.sh" in call["command"] for call in sandbox.execs)

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

    @pytest.mark.parametrize(
        ("path", "status", "stage", "terminal"),
        [
            # An agent refuses a seed it cannot serve (AgentSeedRejectedError) with a final 422.
            ("/v1/agent_sessions", 422, "agent", True),
            ("/ng-rollout/r1/v1/responses", 422, "agent", True),
            ("/v1/agent_sessions/close", 503, "cleanup", False),
        ],
    )
    async def test_agent_http_error_body_reaches_the_failure_reason(
        self, tmp_path, monkeypatch, path, status, stage, terminal
    ):
        """str(ClientResponseError) names only the status and URL; the agent's own reason is in the response body."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_with[path] = http_error_with_body(
            status, b'{"detail":"Simple Agent does not support required sandbox access"}'
        )

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == stage and failure.terminal is terminal
        assert "Simple Agent does not support required sandbox access" in failure.failure_reason
        assert sandbox.stopped and not server._sandboxes

    async def test_seed_http_error_body_reaches_the_failure_reason(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)

        async def unreachable_access(session):
            raise http_error_with_body(502, b'{"detail":"sandbox endpoint unreachable"}')

        monkeypatch.setattr(server, "_sandbox_access", unreachable_access)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "seed" and not failure.terminal
        assert failure.failure_reason.startswith("ClientResponseError: ")
        assert "sandbox endpoint unreachable" in failure.failure_reason
        assert sandbox.stopped and not server._sandboxes

    async def test_failed_turn_closes_the_session_and_keeps_the_agents_evidence(self, tmp_path, monkeypatch):
        """A turn that fails after a decision keeps the harness's observations and the decisions it took.

        The failure row carries the close's trajectory as a result does; without it rollout collection rebuilds
        one turn per captured model call and loses the model-call references and command-batch counts.
        """
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/ng-rollout/r1/v1/responses"
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}
        server.server_client.trajectory = harness_trajectory(step_count=3)

        response = await run_episode(server, task)

        assert response.failure.stage == "agent" and not response.failure.terminal
        assert response.failure.agent_observations is not None
        [turn] = response.failure.trajectory.turns
        assert (turn.turn_no, turn.step_count, turn.model_calls[0].response_id) == (1, 3, "resp_0")
        dumped = response.model_dump(mode="json")["failure"]
        assert (
            dumped["trajectory"]["turns"][0]["step_count"] == 3 and dumped["agent_observations"]["source"] == "harness"
        )
        assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert sandbox.stopped

    @staticmethod
    def failed_turn(outcome: str) -> dict:
        response = agent_response().model_dump(mode="json")
        response.update(
            status="failed",
            error={"code": "server_error", "message": "stream disconnected before completion"},
            metadata={AGENT_OUTCOME_METADATA_KEY: outcome},
        )
        return response

    @pytest.mark.parametrize(
        ("outcome", "terminal"),
        [('{"failure_kind":"unknown"}', False), ('{"failure_kind":"model_transport_error"}', False), ("{", True)],
        ids=["unknown-retried", "transport-retried", "malformed-failed"],
    )
    async def test_failed_agent_response_is_not_scored(self, tmp_path, monkeypatch, outcome, terminal):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.turn_response = self.failed_turn(outcome)
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}

        response = await run_episode(server, task)

        assert response.result is None
        assert response.failure.stage == "agent" and response.failure.terminal is terminal
        assert response.failure.failure_reason == "stream disconnected before completion"
        assert response.failure.partial_response.status == "failed"
        assert response.failure.agent_observations is not None
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert sandbox.stopped and not server._sandboxes

    async def test_failed_agent_response_with_a_verify_cause_is_scored(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.turn_response = self.failed_turn('{"failure_kind":"agent_limit"}')

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.response.status == "failed"
        assert any("test.sh" in call["command"] for call in sandbox.execs)

    async def test_agent_outcome_policy_is_configurable(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.config.agent_outcome_policy = AgentOutcomePolicyConfig(
            by_failure_kind={"unknown": OutcomeAction.VERIFY}
        )
        server.server_client.turn_response = self.failed_turn('{"failure_kind":"unknown"}')

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert any("test.sh" in call["command"] for call in sandbox.execs)

    async def test_failed_close_is_a_cleanup_failure_with_the_turn_attached(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        request_info = RequestInfo(
            URL("http://agent/v1/agent_sessions/close"), "POST", CIMultiDictProxy(CIMultiDict())
        )
        server.server_client.fail_with["/v1/agent_sessions/close"] = ClientResponseError(
            request_info, (), status=500, message="close exploded"
        )

        response = await run_episode(server, task)

        assert response.failure.stage == "cleanup" and not response.failure.terminal
        assert response.failure.partial_response.id == "response"
        # Verification never ran: the harness may still have been working in the sandbox.
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        # The final unwind retried the close, then stopped the sandbox.
        assert server.server_client.paths().count("/v1/agent_sessions/close") == 2
        assert sandbox.stopped

    async def test_close_that_only_times_out_still_scores_the_turn(self, tmp_path, monkeypatch):
        """A slow harness teardown must not discard a finished turn; its observations are lost with the reply."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/v1/agent_sessions/close"
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.response.id == "response"
        assert response.result.ng_agent_observations is None
        assert any("test.sh" in call["command"] for call in sandbox.execs)
        # The final unwind retried the close (which then succeeded), then stopped the sandbox.
        assert server.server_client.paths().count("/v1/agent_sessions/close") == 2
        assert sandbox.stopped and not server._sandboxes

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
        server.server_client.trajectory = harness_trajectory(step_count=3)

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

        run = next(call for call in sandbox.execs if "test.sh" in call["command"] and "timeout" in call["command"])
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

    async def test_prepare_clears_the_reserved_trajectory_destination_before_the_upload(self, tmp_path, monkeypatch):
        """Shared-mode preparation keeps the agent's logs, except the one path the server writes: an entry the
        agent planted there (a link, a directory) would be followed by, or fail, the transcript upload."""
        _, _, sandbox, response = await self.verified(tmp_path, monkeypatch)

        assert response.result.reward == 1.0
        commands = [call["command"] for call in sandbox.execs]
        prepare = next(command for command in commands if "-mindepth 1 -delete" in command)
        steps = prepare.split(" && ")
        clear = steps.index("rm -rf -- /logs/agent/trajectory.json")
        # After the roots exist as directories (a linked /logs/agent was unlinked and remade just before), and
        # before the wipe of the verifier's folders; nothing else under /logs/agent is deleted.
        assert steps[clear - 1] == "mkdir -p /tests /logs/verifier /logs/agent"
        # The whole of /logs/verifier goes with /tests before test.sh (round 38 review, P1): keeping everything but
        # the grader's four names kept an agent-planted test report too. A sidecar's input survives only when the
        # task declares it (`[verifier].inputs`; `test_prepare_keeps_only_the_declared_sidecar_inputs`).
        assert steps[clear + 1] == "find /tests /logs/verifier -mindepth 1 -delete"
        assert not any("/logs/agent" in step and "delete" in step for step in steps)
        trajectory_upload = next(i for i, c in enumerate(commands) if "tar -xzf" in c and "-C /logs/agent" in c)
        assert commands.index(prepare) < trajectory_upload

    @pytest.mark.parametrize("planted", ["dangling_symlink", "directory"])
    async def test_prepare_removes_a_planted_trajectory_entry(self, tmp_path, monkeypatch, planted):
        """The prepare line the server sends, run by a real shell against a temporary root."""
        _, _, sandbox, _ = await self.verified(tmp_path, monkeypatch)
        prepare = next(call["command"] for call in sandbox.execs if "-mindepth 1 -delete" in call["command"])
        root = tmp_path / "fs"
        agent_logs = root / "logs" / "agent"
        agent_logs.mkdir(parents=True)
        (root / "tests").mkdir(parents=True)
        (root / "tests" / "stale").write_text("old")
        (root / "logs" / "verifier").mkdir(parents=True)
        (root / "logs" / "verifier" / "reward.json").write_text('{"reward": 1}')
        (root / "logs" / "verifier" / "actions.jsonl").write_text("{}\n")  # an undeclared sidecar input: removed
        (agent_logs / "agent.log").write_text("kept")
        if planted == "dangling_symlink":
            (agent_logs / "trajectory.json").symlink_to(root / "nowhere")
        else:
            (agent_logs / "trajectory.json" / "sub").mkdir(parents=True)
        command = prepare.replace("/tests", str(root / "tests")).replace("/logs", str(root / "logs"))

        subprocess.run(["sh", "-c", command], check=True)

        planted_path = agent_logs / "trajectory.json"
        assert not planted_path.exists() and not planted_path.is_symlink()
        assert (agent_logs / "agent.log").read_text() == "kept"
        assert list((root / "tests").iterdir()) == []
        assert list((root / "logs" / "verifier").iterdir()) == []

    @pytest.mark.parametrize("planted_as", ["directory", "symlink"])
    async def test_prepare_keeps_only_the_declared_sidecar_inputs(self, tmp_path, monkeypatch, planted_as):
        """Round 38 review (P1): the prepare line empties /logs/verifier except the task's declared sidecar inputs
        (`[verifier].inputs`, from dataset.toml), each kept as a regular file at that name alone, run by a real shell
        against a temporary root. An agent-planted all-pass results.xml, a reward.json and a subfolder go; a planted
        directory or link under a declared name goes too."""
        (tmp_path / "datasets" / "ds").mkdir(parents=True)
        (tmp_path / "datasets" / "ds" / "dataset.toml").write_text(
            '[gym.tasks."hello".verifier]\ninputs = ["actions.jsonl", "planner.log"]\n'
        )
        _, task, sandbox, response = await self.verified(tmp_path, monkeypatch)
        assert task.config.verifier.inputs == ["actions.jsonl", "planner.log"] and response.result.reward == 1.0
        prepare = next(call["command"] for call in sandbox.execs if "-mindepth 1 -delete" in call["command"])
        assert "find /logs/verifier -mindepth 1 ! \\( \\( -path /logs/verifier/actions.jsonl -o " in prepare
        assert "-path /logs/verifier/planner.log \\) -type f \\) -delete" in prepare
        root = tmp_path / "fs"
        verifier_dir = root / "logs" / "verifier"
        verifier_dir.mkdir(parents=True)
        (root / "logs" / "agent").mkdir()
        (root / "tests").mkdir()
        (verifier_dir / "actions.jsonl").write_text('{"action": "click"}\n')
        (verifier_dir / "results.xml").write_text('<testsuite failures="0"/>')
        (verifier_dir / "reward.json").write_text('{"reward": 1}')
        (verifier_dir / "sub").mkdir()
        (verifier_dir / "sub" / "actions.jsonl").write_text("nested, not a top-level input")
        if planted_as == "directory":
            (verifier_dir / "planner.log").mkdir()
            (verifier_dir / "planner.log" / "x").write_text("x")
        else:
            (verifier_dir / "planner.log").symlink_to(root / "tests")
        command = prepare.replace("/tests", str(root / "tests")).replace("/logs", str(root / "logs"))

        subprocess.run(["sh", "-c", command], check=True)

        assert [path.name for path in verifier_dir.iterdir()] == ["actions.jsonl"]
        assert (verifier_dir / "actions.jsonl").read_text() == '{"action": "click"}\n'
        assert not (verifier_dir / "planner.log").is_symlink() and (root / "tests").is_dir()

    async def test_prepare_runs_as_root_on_a_non_root_image(self, tmp_path, monkeypatch):
        class NonRootSandbox(FakeSandbox):
            """The image's default user may not touch root-owned /logs/verifier."""

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                result = await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
                if ("chmod" in command or "chown" in command) and user != "root":
                    return SandboxExecResult(stdout="", stderr="chmod: Permission denied", return_code=1)
                return result

        # The recorded base image runs as `app`, as the NonRootSandbox models.
        server, task, sandbox, response = await self.verified(
            tmp_path,
            monkeypatch,
            NonRootSandbox(),
            dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nUSER app",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "app"},
        )
        assert task.user == "app" and task.sandbox_user == "app"
        assert response.result.reward == 1.0 and response.result.failure_kind is None
        # Harbor's `chmod +x` on test.sh is root's as well; set it aside so the lines below read the run itself.
        chmod = next(call for call in sandbox.execs if call["command"] == "chmod +x /tests/test.sh")
        assert chmod["user"] == "root"
        sandbox.execs.remove(chmod)
        prepare = next(
            call for call in sandbox.execs if "chmod 777" in call["command"] and "/tests" in call["command"]
        )
        assert prepare["user"] == "root"
        # test.sh itself still runs as the image's own user (no [verifier].user), named explicitly, not root.
        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert task.config.verifier.user is None and run["user"] == "app"

    async def test_json_reward_key_makes_the_other_keys_metrics(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"reward": 0.5, "tests_passed": 3})})
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.5
        assert response.result.verifier_rewards == {"reward": 0.5, "tests_passed": 3.0}
        # `tests_passed` does not sum to the reward, so the row names no components at all (not a null).
        assert response.result.reward_components is None
        assert "reward_components" not in response.result.model_dump(mode="json")

    async def test_several_keys_without_reward_are_summed_components(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"a": 0.75, "b": 0.25})})
        _, _, sandbox, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert response.failure is None and result.reward == 1.0 and result.failure_kind is None
        assert result.reward_components == {"a": 0.75, "b": 0.25}
        assert result.verifier_rewards == {"a": 0.75, "b": 0.25}
        dumped = result.model_dump(mode="json")
        assert sum(dumped["reward_components"].values()) == dumped["reward"]
        assert sandbox.stopped

    async def test_one_key_without_reward_is_its_own_component(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"accuracy": 0.25})})
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.25 and response.result.reward_components == {"accuracy": 0.25}
        # reward.txt is a `reward` key.
        _, _, _, response = await self.verified(tmp_path, monkeypatch, FakeSandbox(verifier_files={"reward.txt": "1"}))
        assert response.result.reward == 1.0 and response.result.reward_components is None

    async def test_components_that_overflow_are_an_invalid_reward(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"a": 1e308, "b": 1e308})})
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is False
        assert (
            result.failure_kind == "harbor:invalid_reward" and "do not sum to a finite reward" in result.failure_reason
        )
        assert result.reward_components is None and result.verifier_return_code == 0

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
        run = next(call for call in sandbox.execs if "test.sh" in call["command"] and "timeout" in call["command"])
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
                if "test.sh" in command and "timeout" in command:
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


class TestSeparateVerification:
    def prepared(self, tmp_path, monkeypatch, *, toml=SEPARATE_TOML, verifier=None, agent=None):
        agent = agent or AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent)
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        verifier = verifier or FakeSandbox()
        created = []

        async def create_verifier(task, *, episode_key=None):
            assert agent.stopped, "the agent sandbox must be stopped before the verifier is created"
            created.append(task.task_id)
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier, created

    async def test_collects_artifacts_then_verifies_in_a_fresh_sandbox(self, tmp_path, monkeypatch):
        server, task, agent, verifier, created = self.prepared(tmp_path, monkeypatch)

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.reward == 1.0 and result.verifier_mode == "separate"
        assert created == ["hello"]
        # The main-container collect hook ran in the agent sandbox; the sidecar hook was skipped.
        hooks = [c for c in agent.execs if "state.db" in c["command"] or "kafka-dump" in c["command"]]
        assert [h["command"] for h in hooks] == ["cp /app/state.db /logs/artifacts/state.db"]
        # /logs/artifacts (a directory) and the report (a file) were pulled from the agent sandbox, each
        # under its own staging root. Verifier logs live in a fresh per-verification folder; the row names it.
        artifacts = Path(result.verifier_logs_dir) / "artifacts"
        assert (artifacts / "00" / "logs" / "artifacts" / "state.db").read_text() == "db"
        assert (artifacts / "01" / "app" / "output" / "report.json").read_bytes() == b'{"ok": true}'
        # ...the agent sandbox was stopped before test.sh ran, and the verifier got tests plus artifacts back.
        assert agent.stopped
        # tests/, /logs/artifacts, the report and the trajectory all travel as archives, so modes travel too.
        assert len(verifier.uploads) == 4 and all(remote.endswith(".tar.gz") for _, remote in verifier.uploads)
        unpacked = [c["command"] for c in verifier.execs if "tar -xzf" in c["command"]]
        assert any("-C /logs/artifacts;" in c for c in unpacked) and any("-C /app/output;" in c for c in unpacked)
        # The two artifacts reach the verifier as the very archives the agent sandbox packed (the modes the agent
        # left, not the host staging copy's widened ones), extracted there by root tar; the staged trees stay as
        # diagnostics and the archives are gone once restored.
        assert agent.archives["/logs/artifacts"] in verifier.uploaded_archives
        assert agent.archives["/app/output"] in verifier.uploaded_archives
        assert sorted(p.name for p in artifacts.iterdir()) == ["00", "01", "manifest.json"]
        # The pass is recorded as Harbor's artifact handler records its own: every declared entry, in staging order.
        manifest = json.loads((artifacts / "manifest.json").read_text())
        assert [(e["index"], e["source"], e["origin"], e["status"], e["type"]) for e in manifest] == [
            (0, "/logs/artifacts", "convention", "ok", "directory"),
            (1, "/app/output/report.json", "task", "ok", "file"),
            (2, "/var/log/api", "task", "skipped", None),
        ]
        assert manifest[1]["destination"] == "artifacts/01/app/output/report.json" and manifest[2]["service"] == "api"
        # A restored directory replaces what the image ships there and is world-writable, as Harbor leaves
        # it, so a verifier that drops privileges can still write scratch files next to the agent's output.
        assert any(
            c["command"]
            == "mkdir -p /logs/artifacts && find /logs/artifacts -mindepth 1 -delete && chmod 777 /logs/artifacts"
            for c in verifier.execs
        )
        # A file's parent is created world-writable when missing and widened when it exists; when the chmod is
        # refused (an image that cannot switch to root) the parent only has to be writable.
        prepare = (
            "if [ -d /app/output ]; then chmod a+rwx /app/output 2>/dev/null || test -w /app/output; "
            "else mkdir -p /app/output && chmod 777 /app/output; fi"
        )
        assert any(c["command"] == prepare for c in verifier.execs)
        run = next(c for c in verifier.execs if "test.sh" in c["command"] and "timeout" in c["command"])
        assert run["command"] == verifier_command(300)
        assert (run["env"], run["user"], run["cwd"]) == ({"CHECK": "strict"}, "root", None)
        assert run["timeout_s"] == 300 + server.config.verifier_grace_s
        assert verifier.stopped
        # The episode's final cleanup found nothing left to stop: the agent sandbox is no longer owned.
        assert not server._sandboxes and not server._sessions

    CLEAR_TRAJECTORY = (
        "if test -L /logs/agent; then rm -- /logs/agent; fi && mkdir -p /logs/agent "
        "&& rm -rf -- /logs/agent/trajectory.json"
    )

    def with_artifacts(self, *sources: str) -> str:
        entries = ", ".join(f'"{source}"' for source in sources)
        return SEPARATE_TOML.replace(
            'artifacts = ["/app/output/report.json", { source = "/var/log/api", service = "api" }]',
            f"artifacts = [{entries}]",
        )

    @pytest.mark.parametrize(
        ("source", "planted"),
        [("/logs/agent", "trajectory.json/sub"), ("/logs", "agent/trajectory.json/sub")],
        ids=["agent_logs", "logs"],
    )
    async def test_a_restored_artifact_over_the_agent_logs_does_not_carry_a_planted_trajectory_entry(
        self, tmp_path, monkeypatch, source, planted
    ):
        """A task that collects /logs/agent (or /logs) restores the agent's tree into the fresh verifier before
        the transcript goes up: a directory the agent planted at trajectory.json would make that upload's tar fail
        and mask the sample. The reserved path is cleared between the restore and the upload."""
        kept = planted.replace("trajectory.json/sub", "agent.log")
        agent = AgentSandbox(
            present={"/logs/artifacts": "dir", source: "dir"},
            archives={
                "/logs/artifacts": archive_of({"state.db": "db"}),
                source: archive_of({planted: "planted", kept: "kept"}),
            },
        )
        server, task, agent, verifier, _ = self.prepared(
            tmp_path, monkeypatch, toml=self.with_artifacts(source), agent=agent
        )

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is False and result.failure_kind is None and result.reward == 1.0
        # The planted directory travelled with the artifact and was restored at its source...
        staged = Path(result.verifier_logs_dir) / "artifacts" / "01" / source.lstrip("/")
        assert (staged / planted).is_file() and (staged / kept).read_text() == "kept"
        commands = [c["command"] for c in verifier.execs]
        restore = next(i for i, c in enumerate(commands) if "tar -xzf" in c and f"-C {source};" in c)
        # ...then the reserved path was cleared, as root, before the transcript's own archive was unpacked there.
        clear = commands.index(self.CLEAR_TRAJECTORY)
        upload = next(i for i, c in enumerate(commands) if "tar -xzf" in c and "-C /logs/agent;" in c and i > clear)
        assert restore < clear < upload
        assert verifier.execs[clear]["user"] == "root"
        assert commands.count(self.CLEAR_TRAJECTORY) == 1
        assert len(uploaded_file(verifier, "trajectory.json")) == 1

    @pytest.mark.parametrize("planted", ["directory", "dangling_symlink", "linked_agent_logs"])
    def test_the_clear_line_removes_what_the_agent_planted_and_keeps_its_logs(self, tmp_path, planted):
        """The clear line the server sends, run by a real shell against a temporary root."""
        root = tmp_path / "fs"
        agent_logs = root / "logs" / "agent"
        if planted == "linked_agent_logs":
            # /logs itself was the artifact: /logs/agent came back as a link to a folder the agent chose.
            (root / "logs").mkdir(parents=True)
            (root / "elsewhere").mkdir()
            (root / "elsewhere" / "trajectory.json").write_text("planted")
            agent_logs.symlink_to(root / "elsewhere")
        else:
            agent_logs.mkdir(parents=True)
            (agent_logs / "agent.log").write_text("kept")
            if planted == "directory":
                (agent_logs / "trajectory.json" / "sub").mkdir(parents=True)
            else:
                (agent_logs / "trajectory.json").symlink_to(root / "nowhere")
        command = self.CLEAR_TRAJECTORY.replace("/logs", str(root / "logs"))

        subprocess.run(["sh", "-c", command], check=True)

        destination = agent_logs / "trajectory.json"
        assert not destination.exists() and not destination.is_symlink()
        assert agent_logs.is_dir() and not agent_logs.is_symlink()
        if planted == "linked_agent_logs":
            # The link's target was not followed: the agent's folder is untouched.
            assert (root / "elsewhere" / "trajectory.json").read_text() == "planted"
        else:
            assert (agent_logs / "agent.log").read_text() == "kept"

    @pytest.mark.parametrize(
        ("restored", "cleared"),
        [
            ((), False),
            (("/logs/artifacts", "/app/output/report.json", "/var/log/api"), False),
            (("/logs/agents",), False),
            (("/logs/agent",), True),
            (("/logs/agent/",), True),
            (("/logs",), True),
            (("//logs",), True),
            (("/",), True),
            (("/logs/agent/trajectory.json",), True),
            (("/app", "/logs/agent"), True),
        ],
    )
    async def test_the_trajectory_upload_clears_its_destination_only_after_a_restore_that_covers_it(
        self, tmp_path, monkeypatch, restored, cleared
    ):
        """Separate mode's common case keeps its single round trip: the sandbox is fresh at /logs/agent unless a
        restored artifact was that folder, an ancestor, or the file itself."""
        server, _, _, _ = make_server(tmp_path, monkeypatch)
        verifier = FakeSandbox()
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        await server._upload_trajectory(verifier, b"{}", logs_dir, restored=restored)

        commands = [c["command"] for c in verifier.execs]
        assert (self.CLEAR_TRAJECTORY in commands) is cleared
        assert (logs_dir / "trajectory.json").read_bytes() == b"{}"
        assert len(uploaded_file(verifier, "trajectory.json")) == 1
        if cleared:
            assert commands.index(self.CLEAR_TRAJECTORY) < next(i for i, c in enumerate(commands) if "tar -xzf" in c)

    async def test_a_clear_the_sandbox_refuses_masks_the_sample_as_an_outage(self, tmp_path, monkeypatch):
        clear_line = self.CLEAR_TRAJECTORY

        class RefusingVerifier(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                result = await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
                if command == clear_line:
                    return _fails("rm: cannot remove '/logs/agent/trajectory.json': Read-only file system")
                return result

        agent = AgentSandbox(
            present={"/logs/artifacts": "dir", "/logs/agent": "dir"},
            archives={"/logs/artifacts": archive_of({}), "/logs/agent": archive_of({"agent.log": "x"})},
        )
        server, task, _, verifier, _ = self.prepared(
            tmp_path, monkeypatch, toml=self.with_artifacts("/logs/agent"), agent=agent, verifier=RefusingVerifier()
        )

        result = (await run_episode(server, task)).result

        assert result.mask_sample is True and result.failure_kind == failure_kinds.PROVIDER_UNAVAILABLE
        assert "Could not clear the restored /logs/agent/trajectory.json" in result.failure_reason
        assert not any("test.sh" in c["command"] and "timeout" in c["command"] for c in verifier.execs)

    def test_sandbox_spec_applies_dockerfile_env(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nENV FOO=bar\nENV MODE=image\n"
        )
        (task.path / "task.toml").write_text(TASK_TOML + '\n[environment.env]\nMODE = "toml"\n')
        task = load_task(task.path)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        spec = server._sandbox_spec(task, "/app")

        # Dockerfile ENV reaches the agent's sandbox, and task.toml [environment.env] wins over it.
        assert spec.env["FOO"] == "bar" and spec.env["MODE"] == "toml"

        verifier_environment = HarborEnvironment(docker_image="org/verifier:1", env={"ONLY": "verifier"})
        verifier_spec = server._sandbox_spec(task, None, environment=verifier_environment, role="verifier")
        assert "FOO" not in verifier_spec.env and verifier_spec.env["ONLY"] == "verifier"

    def test_verifier_sandbox_spec_uses_the_verifier_environment(self, tmp_path, monkeypatch):
        server, task, _, _, _ = self.prepared(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        spec = server._sandbox_spec(
            task,
            None,
            environment=task.config.verifier.environment,
            image=_verifier_image(task),
            role="verifier",
            ttl=42,
        )
        assert spec.image == "org/verifier:1"
        assert (spec.resources.cpu, spec.resources.memory_mib) == (2, 4096)
        assert spec.ttl_s == 42 and spec.metadata["harbor_role"] == "verifier"

    async def test_a_verifier_block_without_an_image_and_no_tests_dockerfile_is_refused(self, tmp_path, monkeypatch):
        """Harbor's verifier environment dir is always tests/: with no Dockerfile there and no docker_image in the
        block its environment validation fails, so the loader refuses the task (and the seed with it), naming the
        two remedies rather than a rebuild that could not help."""
        server, task, _, created = make_server(tmp_path, monkeypatch)
        toml = SEPARATE_TOML.replace('docker_image = "org/verifier:1"\n', "")
        (task.path / "task.toml").write_text(toml)
        with pytest.raises(HarborTaskError, match="has no environment definition") as info:
            load_task(task.path)
        assert "Set [verifier.environment].docker_image or add tests/Dockerfile" in str(info.value)
        assert "prepare" not in str(info.value)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "has no environment definition" in response.failure.failure_reason and created == []

    async def test_verifier_failure_masks_and_stops_both_sandboxes(self, tmp_path, monkeypatch):
        class Broken(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                raise RuntimeError("verifier sandbox lost")

        broken = Broken()
        server, task, agent, verifier, _ = self.prepared(tmp_path, monkeypatch, verifier=broken)
        result = (await run_episode(server, task)).result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert agent.stopped and verifier.stopped

    async def test_a_probe_the_sandbox_cannot_answer_masks_but_absence_does_not(self, tmp_path, monkeypatch):
        server, task, agent, verifier, created = self.prepared(tmp_path, monkeypatch)
        agent.probe_result = SandboxExecResult(
            stdout="", stderr="container gone", return_code=125, error_type="sandbox"
        )
        result = (await run_episode(server, task)).result
        # The outage is Gym's, not an empty artifact: masked, and no verifier was started for it.
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert "SandboxUnavailableError" in result.failure_reason and "container gone" in result.failure_reason
        assert created == [] and verifier.execs == [] and agent.stopped

        server, task, agent, verifier, _ = self.prepared(tmp_path, monkeypatch)
        agent.probe_result = SandboxExecResult(stdout="none\n", stderr="", return_code=0)
        result = (await run_episode(server, task)).result
        # A plain "none" is still absence: nothing restored, the verifier runs and scores.
        assert result.reward == 1.0 and not result.mask_sample
        assert not any("tar -xzf" in c["command"] and "/logs/artifacts" in c["command"] for c in verifier.execs)

    async def test_a_collect_hook_the_sandbox_cannot_run_masks(self, tmp_path, monkeypatch):
        class HookLost(AgentSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command.startswith("cp /app/state.db"):
                    return SandboxExecResult(
                        stdout="", stderr="container gone", return_code=None, error_type="sandbox"
                    )
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        agent = HookLost()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=SEPARATE_TOML)
        result = (await run_episode(server, task)).result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert "collect hook could not run: sandbox" in result.failure_reason and agent.stopped

    async def test_a_collect_hook_that_times_out_warns_and_collection_goes_on(self, tmp_path, monkeypatch, caplog):
        """Harbor's hooks are best effort: one that runs past its timeout is logged, never an outage. The artifacts
        are still collected, the verifier runs and the sample is scored."""

        class HookSlow(AgentSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command.startswith("cp /app/state.db"):
                    return SandboxExecResult(stdout="", stderr="", return_code=None, error_type="timeout")
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        agent = HookSlow()
        server, task, _, verifier, created = self.prepared(tmp_path, monkeypatch, agent=agent)
        with caplog.at_level("WARNING"):
            result = (await run_episode(server, task)).result
        assert result.reward == 1.0 and not result.mask_sample, result.failure_reason
        assert "collect hook timed out after 10s: 'cp /app/state.db /logs/artifacts/state.db'" in caplog.text
        assert created == ["hello"] and agent.archives["/logs/artifacts"] in verifier.uploaded_archives
        assert any("test.sh" in c["command"] and "timeout" in c["command"] for c in verifier.execs)
        assert agent.stopped and verifier.stopped

    async def test_an_archive_step_the_sandbox_cannot_run_masks(self, tmp_path, monkeypatch):
        """The probe answered, then the provider lost the container under `tar -czf` (error_type set, exit 125):
        an outage, not an artifact to skip, so nothing is collected and the sample is masked."""

        class LostUnderTar(AgentSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if "tar -czf" in command:
                    self.execs.append(
                        {"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user}
                    )
                    return SandboxExecResult(stdout="", stderr="container gone", return_code=125, error_type="sandbox")
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        agent = LostUnderTar()
        server, task, _, verifier, created = self.prepared(tmp_path, monkeypatch, agent=agent)
        result = (await run_episode(server, task)).result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert "SandboxUnavailableError" in result.failure_reason and "container gone" in result.failure_reason
        assert created == [] and verifier.execs == [] and agent.stopped

    async def test_shared_mode_runs_collect_hooks_before_preparing_tests(self, tmp_path, monkeypatch):
        toml = (
            TASK_TOML
            + '\n[[verifier.collect]]\ncommand = "cp /app/state.db /logs/artifacts/state.db"\nuser = "root"\n'
        )
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml)
        result = (await run_episode(server, task)).result
        assert result.reward == 1.0 and result.verifier_mode == "shared"
        commands = [c["command"] for c in sandbox.execs]
        hook = commands.index("cp /app/state.db /logs/artifacts/state.db")
        prepare = next(i for i, command in enumerate(commands) if "-mindepth 1 -delete" in command)
        assert hook < prepare
        assert sandbox.execs[hook]["user"] == "root" and sandbox.execs[hook]["cwd"] == "/app"
        assert sandbox.execs[hook]["timeout_s"] == 60 + 30

    async def test_artifact_exclusions_reach_the_archive_command(self, tmp_path, monkeypatch):
        toml = SEPARATE_TOML.replace(
            'artifacts = ["/app/output/report.json", { source = "/var/log/api", service = "api" }]',
            'artifacts = [{ source = "/app/build", exclude = ["node_modules", "./cache", "*.log"] }]',
        )
        agent = AgentSandbox(
            present={"/logs/artifacts": "dir", "/app/build": "dir"},
            archives={"/logs/artifacts": archive_of({}), "/app/build": archive_of({"a.out": "elf"})},
        )
        server, task, _, _, _ = self.prepared(tmp_path, monkeypatch, toml=toml, agent=agent)
        result = (await run_episode(server, task)).result
        assert result.reward == 1.0, result.failure_reason
        build = next(c["command"] for c in agent.execs if "tar -czf" in c["command"] and "/app/build" in c["command"])
        # Verbatim, unanchored, as Harbor passes them: `node_modules` at any depth, `./cache` at the top only.
        assert " --exclude=node_modules --exclude=./cache --exclude='*.log' -C /app/build ." in build
        artifacts_archive = next(
            c["command"] for c in agent.execs if "tar -czf" in c["command"] and "/logs/artifacts" in c["command"]
        )
        assert "--exclude" not in artifacts_archive

    @pytest.mark.parametrize("member", ["device_node", "long_name"])
    async def test_an_artifact_tree_this_host_cannot_hold_is_not_collected_and_does_not_mask(
        self, tmp_path, monkeypatch, caplog, member
    ):
        """Real archive extraction with a simulated provider: a character device in the agent's tree raised
        PermissionError and a 300-character name ENAMETOOLONG out of the per-artifact handler, masking the sample
        as provider_unavailable. Both are the agent's tree: that artifact is skipped, the other is restored and the
        verifier scores."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            if member == "device_node":
                info = tarfile.TarInfo(name="./null")
                info.type, info.devmajor, info.devminor = tarfile.CHRTYPE, 1, 3
            else:
                info = tarfile.TarInfo(name="./" + "n" * 300)
                info.size = 2
            tar.addfile(info, io.BytesIO(b"ok") if member == "long_name" else None)
        agent = AgentSandbox(
            archives={"/logs/artifacts": buffer.getvalue(), "/app/output": archive_of({"report.json": "{}"})}
        )
        server, task, _, verifier, created = self.prepared(tmp_path, monkeypatch, agent=agent)
        with caplog.at_level("WARNING"):
            result = (await run_episode(server, task)).result
        assert result.reward == 1.0 and not result.mask_sample, result.failure_reason
        assert "artifact '/logs/artifacts' not collected" in caplog.text
        artifacts = Path(result.verifier_logs_dir) / "artifacts"
        # Nothing of the refused artifact reached the verifier or stayed behind as an archive; the report did.
        assert agent.archives["/logs/artifacts"] not in verifier.uploaded_archives
        assert agent.archives["/app/output"] in verifier.uploaded_archives
        assert not list(artifacts.glob("*.tar.gz")) and not list((artifacts / "00").rglob("null"))
        assert created == ["hello"] and agent.stopped and verifier.stopped

    async def test_each_artifact_stages_under_its_own_root(self, tmp_path, monkeypatch):
        """A link kept from one artifact is never a path component of another artifact's destination."""
        victim = tmp_path / "victim"
        victim.mkdir()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            link = tarfile.TarInfo(name="./output")
            link.type, link.linkname = tarfile.SYMTYPE, str(victim)
            tar.addfile(link)
        toml = SEPARATE_TOML.replace(
            'artifacts = ["/app/output/report.json", { source = "/var/log/api", service = "api" }]',
            'artifacts = ["/app", "/app/output/report.json"]',
        )
        agent = AgentSandbox(
            present={"/logs/artifacts": "none", "/app": "dir", "/app/output/report.json": "file"},
            archives={"/app": buffer.getvalue(), "/app/output": archive_of({"report.json": '{"ok": true}'})},
        )
        server, task, _, _, _ = self.prepared(tmp_path, monkeypatch, toml=toml, agent=agent)
        result = (await run_episode(server, task)).result
        assert result.reward == 1.0, result.failure_reason
        artifacts = Path(result.verifier_logs_dir) / "artifacts"
        # The link travelled as a link under the first artifact's root; the report landed under the second.
        assert (artifacts / "01" / "app" / "output").is_symlink()
        assert (artifacts / "02" / "app" / "output" / "report.json").read_bytes() == b'{"ok": true}'
        assert list(victim.iterdir()) == []


COMPOSE_TOML = """
schema_version = "1.4"

artifacts = [{ source = "/var/log/api", service = "api" }]

[verifier]
timeout_sec = 300.0
environment_mode = "separate"

[verifier.environment]
docker_image = "org/verifier:1"

[[verifier.collect]]
command = "dump-topics > /logs/artifacts/topics.txt"
service = "kafka"

[agent]
timeout_sec = 120.0

[environment]
docker_image = "org/agent:1"
cpus = 1
skills_dir = "/app/.skills"

[[environment.mcp_servers]]
name = "playwright"
transport = "sse"
url = "http://api:3080/sse"

[environment.healthcheck]
command = "test -f /tmp/ready"
interval_sec = 0.01
start_interval_sec = 0.01
timeout_sec = 5.0
retries = 3
"""


@dataclass
class HealthySandbox(AgentSandbox):
    """Main service whose healthcheck passes on the third probe."""

    probes: int = 0

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command == "test -f /tmp/ready":
            self.probes += 1
            self.execs.append({"command": command, "user": user})
            return SandboxExecResult(stdout="", stderr="", return_code=0 if self.probes >= 3 else 1)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


class NetworkedStubProvider:
    """The least a provider needs for the Compose adapter's validation of a multi-service document."""

    name = "stub"

    def validate_networking(self) -> None:
        pass

    async def network_address(self, handle):
        return "192.0.2.1"

    async def set_hosts(self, handle, hosts):
        pass


@dataclass
class MainWatchingSandbox(AgentSandbox):
    """A sidecar that notes, at each collect hook and artifact lookup, whether the main container was stopped."""

    main: FakeSandbox | None = None
    main_stopped_when_seen: list[bool] = field(default_factory=list)

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "dump-topics" in command or command.startswith("if [ -d "):
            self.main_stopped_when_seen.append(self.main.stopped)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


def _record_stop(stops: list[str], name: str, sandbox: FakeSandbox):
    async def stop():
        stops.append(name)
        sandbox.stopped = True

    return stop


class StubProvider:
    """A sandbox provider whose sandboxes exist only as records and answer every exec with success, enough for
    the real Compose adapter to start a group through it."""

    def __init__(self, name):
        self.name = name
        self.created: list[SandboxSpec] = []
        self.closed: list[str] = []

    async def create(self, spec):
        self.created.append(spec)
        return SandboxHandle(str(len(self.created)), self.name, None)

    async def exec(self, handle, command, **kwargs):
        return SandboxExecResult("", "", 0)

    async def upload_file(self, handle, source_path, target_path):
        pass

    async def close(self, handle):
        self.closed.append(handle.sandbox_id)

    async def aclose(self):
        pass

    def validate_networking(self):
        pass

    async def network_address(self, handle):
        return f"10.0.{len(self.name)}.{handle.sandbox_id}"

    async def set_hosts(self, handle, hosts):
        pass


class TestComposeAndRouting:
    def compose_server(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        main, api, kafka = HealthySandbox(), FakeSandbox(), FakeSandbox()
        compose = FakeCompose({"main": main, "api": api, "kafka": kafka})
        documents = []
        verifier = FakeSandbox()

        async def start_compose(task, episode_key):
            documents.append(server._compose_document(task))
            return compose

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_compose", start_compose)
        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, compose, documents, verifier

    def test_compose_document_follows_the_requested_sandbox_platform(self, tmp_path, monkeypatch):
        """arm64 acceptance defect: a deployment asking for arm64 pods (``sandbox_provider_options.platform``)
        runs the arm64 images prepare built; the default deployment still runs amd64 ones and refuses the rest."""
        arm64 = {image: {**record, "architecture": "arm64"} for image, record in IMAGE_CONFIGS.items()}
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        def compose_task(root, records, **config):
            server, task, _, _ = make_server(root, monkeypatch, **config)
            (task.path / "task.toml").write_text(COMPOSE_TOML)
            (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
            (task.path.parent / "compose-images.json").write_text(json.dumps(records))
            return server, load_task(task.path)

        platform = {"platform": {"os": "linux", "arch": "arm64"}, "resource_requests": {"cpu": 0.5}}
        server, task = compose_task(tmp_path / "arm", arm64, sandbox_provider_options=platform)
        document = server._compose_document(task)
        assert set(document["services"]) == {"main", "api", "kafka"}
        assert document["services"]["kafka"]["image"] == arm64["org/kafka:1"]["image"]
        # The requested platform is the one the options still carry for the sandboxes themselves.
        assert server._sandbox_spec(task, "/app").provider_options["platform"] == platform["platform"]

        server, task = compose_task(tmp_path / "amd", arm64)
        with pytest.raises(ValueError, match="Service 'main': the recorded image .* is linux/arm64, not linux/amd64"):
            server._compose_document(task)
        server, task = compose_task(tmp_path / "mixed", IMAGE_CONFIGS, sandbox_provider_options=platform)
        with pytest.raises(ValueError, match="Service 'main': the recorded image .* is linux/amd64, not linux/arm64"):
            server._compose_document(task)

    async def test_compose_document_resolves_variables_and_env_file_from_the_task_then_the_process_environment(
        self, tmp_path, monkeypatch
    ):
        # M19: the server reads the Compose file through the same resolver prepare used, so ${MAIN_IMAGE_NAME},
        # ${CONTEXT_DIR} and an env_file resolve from the task. G3: a variable the task does not set is read from
        # the server's process environment, as Harbor's docker compose reads it, so prepare and the server must
        # run with the same variables.
        server, task, compose, documents, _ = self.compose_server(tmp_path, monkeypatch)
        monkeypatch.setenv("KAFKA_TOPIC", "from-the-shell")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n"
            "  main:\n    image: ${MAIN_IMAGE_NAME}\n    volumes: ['${CONTEXT_DIR}/seed.json:/app/seed.json:ro']\n"
            "  api:\n    image: org/api:1\n    expose: ['3080']\n    env_file: api.env\n"
            "  kafka:\n    image: org/kafka:1\n    environment: ['KAFKA_TOPIC=${KAFKA_TOPIC:-events}']\n"
        )
        (task.path / "environment" / "api.env").write_text("API_MODE=replay\n")
        (task.path / "environment" / "seed.json").write_text("{}")
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        services = documents[0]["services"]
        assert services["main"]["image"] == IMAGE_CONFIGS["org/agent:1"]["image"]
        # The server resolves the task from its snapshot, so the bind mount (an upload since M22) points into
        # that copy.
        (upload,) = services["main"]["x-sandbox"]["uploads"]
        assert upload["target"] == "/app/seed.json" and upload["source"].endswith("/hello/environment/seed.json")
        assert upload["source"].startswith(server._snapshots.name) and not upload["source"].startswith(str(task.path))
        assert "volumes" not in services["main"]
        assert services["api"]["environment"] == {"API_MODE": "replay"} and "env_file" not in services["api"]
        assert services["kafka"]["environment"] == {"KAFKA_TOPIC": "from-the-shell"}

    MAIN_BUILD_SHAPES = {
        "args-and-target": "    build:\n      context: ${CONTEXT_DIR}\n      target: prod\n      args:\n        PYTHON_VERSION: '3.12'\n",
        "dockerfile_inline": "    build:\n      context: ${CONTEXT_DIR}\n      dockerfile_inline: |\n        FROM python:3.12\n        RUN pip install x\n",
    }

    @pytest.mark.parametrize("main_build", list(MAIN_BUILD_SHAPES.values()), ids=list(MAIN_BUILD_SHAPES))
    def test_a_built_tasks_main_build_section_is_dropped_at_seed(self, tmp_path, monkeypatch, main_build):
        """Prepare builds ``main`` from the task's Compose file (its build args, target or dockerfile_inline apply
        there) and every sidecar with a ``build:`` section (G2), and records the images by service; at seed the
        recorded images stand in, so the ``build:`` and ``pull_policy: build`` keys that only describe those builds are
        dropped before the Compose adapter, which starts images only, and the sidecar's entrypoint, command and
        ports come from its recorded configuration like any pulled image's."""
        from nemo_gym.sandbox import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE
        from nemo_gym.tasks.harbor.task import built_service_images

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY . /app\n")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    pull_policy: build\n"
            + main_build
            + "    depends_on:\n      world:\n        condition: service_healthy\n"
            + "  api:\n    image: org/api:1\n"
            + "  world:\n    build:\n      context: .\n      dockerfile: world/Dockerfile\n    pull_policy: build\n"
            + "    expose: ['8972']\n    healthcheck:\n      test: ['CMD', 'true']\n      interval: 2s\n"
        )
        built = "registry.example/ds/hb__" + "0" * 32 + "@sha256:" + "d" * 64
        world = "registry.example/ds/hb__" + "0" * 32 + "-world@sha256:" + "e" * 64
        (task.path.parent / BUILT_IMAGES_FILE).write_text(
            json.dumps(
                {
                    context_digest(task.path / "environment"): {
                        "image": built,
                        "services": {"main": {"image": built}, "world": {"image": world}},
                    }
                }
            )
        )
        records = json.loads((task.path.parent / IMAGE_CONFIGS_FILE).read_text())
        records[built] = {
            "os": "linux",
            "architecture": "amd64",
            "image": built,
            "config": {"WorkingDir": "/app"},
            "record_version": RECORD_VERSION,
        }
        records[world] = {
            "os": "linux",
            "architecture": "amd64",
            "image": world,
            "config": {"Cmd": ["python3", "/opt/world/server.py"], "ExposedPorts": {"8972/tcp": {}}},
            "record_version": RECORD_VERSION,
        }
        records["org/api:1"] = IMAGE_CONFIGS["org/api:1"]
        (task.path.parent / IMAGE_CONFIGS_FILE).write_text(json.dumps(records))
        task = load_task(task.path)
        assert task.built_image == built == task.image and task.workdir == "/app"
        assert built_service_images(task.path) == {"main": built, "world": world}
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        document = server._compose_document(task)

        main = document["services"]["main"]
        assert main["image"] == built and main["working_dir"] == "/app"
        assert "build" not in main and "pull_policy" not in main
        assert document["services"]["api"]["image"] == IMAGE_CONFIGS["org/api:1"]["image"]
        sidecar = document["services"]["world"]
        assert sidecar["image"] == world and "build" not in sidecar and "pull_policy" not in sidecar
        assert sidecar["command"] == ["python3", "/opt/world/server.py"] and sidecar["expose"] == ["8972", "8972/tcp"]
        assert main["depends_on"] == {"world": {"condition": "service_healthy"}}
        # The adapter's own validation accepts the document: the fields that failed every episode are gone.
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml")
        group.document = document
        assert set(group._validate()) == {"main", "api", "world"}

    def test_seed_reads_the_build_record_once_at_load_and_never_hashes_the_environment(self, tmp_path, monkeypatch):
        """``load_task`` (once per task, in a thread) hashes ``environment/`` once and carries Harbor's context id
        and the build record on the task; ``_compose_document``, which runs per episode, hashes nothing, whatever
        the folder's size, and still substitutes the recorded images and defines ``COMPOSE_PROJECT_NAME``."""
        import nemo_gym.tasks.harbor.digest as digest_module
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY . /app\n")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  world:\n    build: ./world\n    environment: ['PROJECT=${COMPOSE_PROJECT_NAME}']\n"
        )
        (task.path / "environment" / "world").mkdir()
        (task.path / "environment" / "world" / "Dockerfile").write_text("FROM python:3.12\n")
        expected = context_digest(task.path / "environment")
        built = f"registry.example/ds/hb__{expected}@sha256:" + "d" * 64
        world = f"registry.example/ds/hb__{expected}-world@sha256:" + "e" * 64
        record = {"image": built, "services": {"main": {"image": built}, "world": {"image": world}}}
        (task.path.parent / BUILT_IMAGES_FILE).write_text(json.dumps({expected: record}))
        records = json.loads((task.path.parent / IMAGE_CONFIGS_FILE).read_text())
        records[built] = {
            "os": "linux",
            "architecture": "amd64",
            "image": built,
            "config": {},
            "record_version": RECORD_VERSION,
        }
        records[world] = {
            "os": "linux",
            "architecture": "amd64",
            "image": world,
            "config": {"Cmd": ["serve"]},
            "record_version": RECORD_VERSION,
        }
        (task.path.parent / IMAGE_CONFIGS_FILE).write_text(json.dumps(records))
        task = load_task(task.path)
        assert task.context_id == expected and task.built_images == {"main": built, "world": world}
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        # Every caller of context_digest reaches this function (patching task.context_digest alone would miss
        # compose_variables' COMPOSE_PROJECT_NAME); the counter is proven live before the seed path is measured.
        hashed: list[Path] = []
        real = digest_module.environment_content_hash
        monkeypatch.setattr(
            digest_module, "environment_content_hash", lambda path: (hashed.append(path), real(path))[1]
        )
        assert context_digest(task.path / "environment") == expected and len(hashed) == 1
        hashed.clear()

        for _ in range(2):
            document = server._compose_document(task)
            assert document["services"]["main"]["image"] == built
            sidecar = document["services"]["world"]
            assert sidecar["image"] == world and "build" not in sidecar and sidecar["command"] == ["serve"]
            assert sidecar["environment"] == {"PROJECT": f"hb__{expected}"}
        assert hashed == []

    @pytest.mark.parametrize(
        ("dockerfile", "build"),
        [
            ("FROM ubuntu:24.04\nWORKDIR /app", "      context: ${CONTEXT_DIR}\n      pull: true\n"),
            ("FROM ubuntu:24.04\nRUN echo hi > /hi\n", "      dockerfile: Dockerfile\n"),
        ],
    )
    def test_a_restated_main_build_is_dropped_in_pull_and_overlay_mode(self, tmp_path, monkeypatch, dockerfile, build):
        """Harbor's own template already builds ``main`` from ``environment/Dockerfile`` with ``context: ${CONTEXT_DIR}``,
        so a task file that only restates that (orca-bench, terminal_bench_aai) changes nothing there. Gym runs the
        Dockerfile without a build in pull and overlay mode, so the section is dropped and the document starts."""
        from nemo_gym.sandbox import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.image_configs import check_task_compose_fields, load_compose_document

        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile=dockerfile)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n" + build
        )
        task = load_task(task.path)
        assert task.built_image is None and task.image == "ubuntu:24.04"
        assert bool(task.overlay) == ("RUN" in dockerfile)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        document = server._compose_document(task)
        assert "build" not in document["services"]["main"]
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml")
        group.document = document
        assert group._validate() == ["main"]
        assert "build" not in check_task_compose_fields(task, load_compose_document(task))["services"]["main"]

    def test_a_main_build_key_gym_cannot_apply_without_a_build_is_a_task_error(self, tmp_path, monkeypatch):
        """A ``target`` (or ``args``, ``dockerfile_inline``, another ``dockerfile`` or ``context``) would change the image
        Harbor builds; without a build Gym cannot honour it, so the task is an error naming the key, as prepare already
        reported it into ``skipped.json``, instead of starting on the wrong image."""
        from nemo_gym.tasks.harbor.errors import HarborTaskError
        from nemo_gym.tasks.harbor.image_configs import check_task_compose_fields, load_compose_document
        from nemo_gym.tasks.harbor.registry import ImageConfigError

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: ${CONTEXT_DIR}\n"
            "      target: prod\n"
        )
        task = load_task(task.path)
        assert task.built_image is None
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        expected = "main.build.target cannot be applied in pull or overlay mode (Gym does not build this Dockerfile)"
        with pytest.raises(HarborTaskError, match=re.escape(expected)):
            server._compose_document(task)
        with pytest.raises(ImageConfigError) as info:
            check_task_compose_fields(task, load_compose_document(task))
        assert str(info.value) == f"hello: environment/docker-compose.yaml: {expected}"

    def test_a_services_pull_policy_reaches_the_adapter_and_its_provider(self, tmp_path, monkeypatch):
        """``pull_policy`` says when an image is pulled before it starts. A task's runtime policy (``always``,
        ``missing`` with its spellings, ``never``) reaches the adapter in the server's document, which records it per
        service and hands the provider's own form of it to the sandbox create where the provider can apply one; an
        invalid value is the task's error, as prepare reports it (review: the key was dropped from every service
        before the field check, by prepare and again here, so the advertised support never saw a task's setting).
        Only ``pull_policy: build`` leaves, with the build it names (``test_built_images_stand_in...``)."""
        from nemo_gym.sandbox import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.image_configs import check_task_compose_fields, load_compose_document
        from nemo_gym.tasks.harbor.registry import ImageConfigError

        class PullPolicyStubProvider(NetworkedStubProvider):
            def pull_policy_options(self, policy):
                return {"extensions": {"pull": policy}}

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        compose = task.path / "environment" / "docker-compose.yaml"
        compose.write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    pull_policy: if_not_present\n"
            "  api:\n    image: org/api:1\n    pull_policy: always\n    expose: ['8080']\n"
        )
        records = json.loads((task.path.parent / IMAGE_CONFIGS_FILE).read_text())
        records["org/api:1"] = IMAGE_CONFIGS["org/api:1"]
        (task.path.parent / IMAGE_CONFIGS_FILE).write_text(json.dumps(records))
        task = load_task(task.path)
        assert task.built_image is None
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        document = server._compose_document(task)

        assert document["services"]["main"]["pull_policy"] == "missing"
        assert document["services"]["api"]["pull_policy"] == "always"
        assert document["services"]["api"]["expose"] == ["8080"]
        provider = PullPolicyStubProvider()
        group = AsyncSandboxCompose(provider, "compose.yaml")
        group.document = document
        assert set(group._validate()) == {"main", "api"}
        assert group.pull_policies == {"main": "missing", "api": "always"}
        assert group._service_options == {
            "main": provider.pull_policy_options("missing"),
            "api": provider.pull_policy_options("always"),
        }
        # Prepare's check normalizes the same document the same way.
        checked = check_task_compose_fields(task, load_compose_document(task))
        assert {name: service["pull_policy"] for name, service in checked["services"].items()} == group.pull_policies
        # A value Compose does not define is the task's error here as at prepare, not a silently dropped key.
        compose.write_text(compose.read_text().replace("always", "whenever"))
        with pytest.raises(
            ImageConfigError, match=re.escape("service 'api' pull_policy 'whenever' is not a Compose value")
        ):
            server._compose_document(task)

    def test_a_built_services_runtime_pull_policy_reaches_the_adapter_with_its_recorded_image(
        self, tmp_path, monkeypatch
    ):
        """Build mode: a service prepare built keeps its runtime ``pull_policy`` (``always``, ``missing``, ``never``)
        beside the recorded image that stands in for its ``build:``, so the server's document carries it to the
        adapter and prepare's check normalizes it the same way; only ``pull_policy: build``, the policy naming that
        build, leaves with it (review: ``with_built_images`` dropped every policy of a built service, so with a
        recorded build preparation succeeded and the server and the adapter received no policy)."""
        from nemo_gym.sandbox import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.image_configs import check_task_compose_fields, load_compose_document

        class PullPolicyStubProvider(NetworkedStubProvider):
            def pull_policy_options(self, policy):
                return {"extensions": {"pull": policy}}

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY . /app\n")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            'services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build: {context: "${CONTEXT_DIR}"}\n'
            "    pull_policy: always\n"
            "  api:\n    build: ./api\n    pull_policy: never\n    expose: ['8080']\n"
            "  cache:\n    image: ${MAIN_IMAGE_NAME}_cache\n    build: ./cache\n    pull_policy: build\n"
        )
        digest = context_digest(task.path / "environment")
        images = {
            name: f"registry.example/ds/hb__{digest}{suffix}@sha256:" + fill * 64
            for name, suffix, fill in (("main", "", "d"), ("api", "-api", "e"), ("cache", "-cache", "f"))
        }
        (task.path.parent / BUILT_IMAGES_FILE).write_text(
            json.dumps(
                {
                    digest: {
                        "tasks": ["hello"],
                        "tool": "docker",
                        "pushed": True,
                        "services": {name: {"image": image} for name, image in images.items()},
                    }
                }
            )
        )
        for name, image in images.items():
            write_image_config(task.path.parent, image, Cmd=[name])
        task = load_task(task.path)
        assert task.built_image == images["main"]
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        document = server._compose_document(task)

        services = document["services"]
        assert {name: service["image"] for name, service in services.items()} == images
        assert not any("build" in service for service in services.values())
        assert {name: service.get("pull_policy") for name, service in services.items()} == {
            "main": "always",
            "api": "never",
            "cache": None,
        }
        assert services["api"]["expose"] == ["8080"]
        provider = PullPolicyStubProvider()
        group = AsyncSandboxCompose(provider, "compose.yaml")
        group.document = document
        assert set(group._validate()) == {"main", "api", "cache"}
        assert group.pull_policies == {"main": "always", "api": "never"}
        assert group._service_options == {
            "main": provider.pull_policy_options("always"),
            "api": provider.pull_policy_options("never"),
        }
        # Prepare's check, on the loader's document, keeps and normalizes the same policies.
        checked = check_task_compose_fields(task, load_compose_document(task))
        assert {name: service.get("pull_policy") for name, service in checked["services"].items()} == {
            "main": "always",
            "api": "never",
            "cache": None,
        }

    @staticmethod
    def capture_compose_specs(monkeypatch) -> dict:
        captured = {}

        class FakeComposeGroup:
            def __init__(self, provider_config, path, *, service_specs, timeout_s, **options):
                captured["specs"] = service_specs
                captured["options"] = options

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FakeComposeGroup)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        return captured

    def test_a_sidecar_without_deploy_resources_is_warned_about_once_per_task(self, tmp_path, monkeypatch, caplog):
        """G7 round 24, attempts 6 and 7: a Compose sidecar without `deploy.resources` gets the provider's default
        sandbox size (`_sidecar_spec` hands the adapter no task resources), where Harbor's `docker compose` gives it
        the host's, and the WebShop sidecar died at start for it. The default is unchanged; the seed now says, once
        per task, which sidecars run at it and where a size goes, and a sized sidecar is not named."""
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  api:\n    image: org/api:1\n    expose: ['8080']\n"
            "  kafka:\n    image: org/kafka:1\n    deploy:\n      resources:\n        limits:\n          memory: 2g\n"
            "  cache:\n    image: org/api:1\n    cpus: 1\n"
        )
        for image in ("org/api:1", "org/kafka:1"):
            write_image_config(task.path.parent, image, **IMAGE_CONFIGS[image]["config"])
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        captured = self.capture_compose_specs(monkeypatch)

        with caplog.at_level("WARNING", logger="environment_servers.harbor.sandboxes"):
            asyncio.run(server._create_compose(task, "r1"))
            asyncio.run(server._create_compose(task, "r2"))

        warnings = [r.getMessage() for r in caplog.records if "deploy.resources" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].startswith("hello: Compose service(s) api declare no deploy.resources")
        assert "sandbox provider's default size" in warnings[0] and "deploy.resources.limits" in warnings[0]
        assert "kafka" not in warnings[0] and "cache" not in warnings[0]
        # The sizes themselves are the adapter's to apply from the document: the specs still carry no task resources.
        from nemo_gym.sandbox import SandboxResources

        assert captured["specs"]["api"].resources == SandboxResources()
        assert captured["specs"]["kafka"].resources == SandboxResources()

    def test_main_env_layers_image_env_under_the_compose_file_under_the_host_resolved_task_env(
        self, tmp_path, monkeypatch
    ):
        """Harbor's order for ``main``: image ENV (here the pull-mode Dockerfile's ``ENV``, which the loader folds into
        the task) < the task's Compose file < the host-resolved ``[environment].env`` (its env override file). A key
        the file sets beats the Dockerfile's, a ``${FOO:-}`` that interpolated to empty at prepare does not hide the
        host value, and a sidecar keeps its file's environment."""
        from nemo_gym.sandbox import AsyncSandboxCompose

        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM ubuntu:24.04\nENV MODE=image\nENV ONLY_IMAGE=i\n",
            toml=TASK_TOML + '\n[environment.env]\nFOO = "${FOO:-}"\n',
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "    environment:\n      MODE: compose\n      FOO: ${FOO:-}\n      ONLY_COMPOSE: c\n"
            "  api:\n    image: org/api:1\n    environment:\n      MODE: sidecar\n    expose: ['8080']\n"
        )
        write_image_config(task.path.parent, "org/api:1", **IMAGE_CONFIGS["org/api:1"]["config"])
        task = load_task(task.path)
        assert task.built_image is None and not task.overlay
        # The base image's own ENV stays in the image; the loader folds in the Dockerfile's ENV lines only.
        assert task.env == {"MODE": "image", "ONLY_IMAGE": "i", "FOO": "${FOO:-}"}
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setenv("FOO", "host")
        captured = self.capture_compose_specs(monkeypatch)

        asyncio.run(server._create_compose(task, "r1"))

        specs = captured["specs"]
        # Under the three layers sit the CPU-count hints for the task's `cpus = 1`; a sidecar gets none.
        expected = cpu_cap_env(1) | {"MODE": "compose", "ONLY_IMAGE": "i", "FOO": "host", "ONLY_COMPOSE": "c"}
        assert specs["main"].env == expected and specs["api"].env == {}
        # The adapter's own plan for the same specs and document gives the same three layers.
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml", service_specs=specs)
        group.document = server._compose_document(task)
        group._validate()
        asyncio.run(group._prepare())
        assert group._plans["main"]["spec"].env == expected
        assert group._plans["api"]["spec"].env == {"MODE": "sidecar"}

    def test_a_built_images_recorded_env_sits_under_the_compose_file_too(self, tmp_path, monkeypatch):
        """Build mode: the recorded built image's whole ``Env`` is the image layer, under the file and under the
        host-resolved ``[environment].env``; a key only the image names survives."""
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE

        server, task, _, _ = make_server(
            tmp_path, monkeypatch, toml=TASK_TOML + '\n[environment.env]\nTOK = "${TOK:-fromtoml}"\n'
        )
        (task.path / "environment" / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nCOPY . /app\nENV ISSUE_ID=from_dockerfile\n"
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(
            'services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build: {context: "${CONTEXT_DIR}"}\n'
            "    environment:\n      ISSUE_ID: from_compose\n      LC_ALL: C.UTF-8\n      TOK: ${TOK:-}\n"
        )
        digest = context_digest(task.path / "environment")
        image = f"registry.example/ds/hb__{digest}@sha256:" + "d" * 64
        (task.path.parent / BUILT_IMAGES_FILE).write_text(
            json.dumps({digest: {"image": image, "tasks": ["hello"], "tool": "docker", "pushed": True}})
        )
        write_image_config(
            task.path.parent, image, Env=["ISSUE_ID=from_image", "LC_ALL=POSIX", "PATH=/usr/bin"], Cmd=["bash"]
        )
        task = load_task(task.path)
        assert task.built_image == image
        assert task.env == {"ISSUE_ID": "from_image", "LC_ALL": "POSIX", "PATH": "/usr/bin", "TOK": "${TOK:-fromtoml}"}
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setenv("TOK", "hosttok")
        captured = self.capture_compose_specs(monkeypatch)

        asyncio.run(server._create_compose(task, "r1"))

        assert captured["specs"]["main"].env == cpu_cap_env(1) | {
            "ISSUE_ID": "from_compose",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin",
            "TOK": "hosttok",
        }

    def test_sidecar_compose_environment_takes_the_host_resolved_task_env_at_seed(self, tmp_path, monkeypatch):
        """Harbor interpolates the task's Compose file for every service against the host-resolved
        ``[environment].env``; prepare could only use the dataset-time default. At seed the sidecars' ``environment``
        is read again with the host values (tau3 / hil-bench shape); a literal entry and the structure are untouched,
        and without the variable the default still applies."""
        from nemo_gym.tasks.harbor.image_configs import load_compose_document

        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=TASK_TOML
            + '\n[environment.env]\nTAU2_USER_MODEL = "${TAU2_USER_MODEL:-gpt-5.2}"\nOPENAI_API_KEY = "${OPENAI_API_KEY:-}"\n',
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  user-sim:\n    image: org/api:1\n    expose: ['8000']\n    environment:\n"
            "      - TAU2_USER_MODEL=${TAU2_USER_MODEL:-gpt-5.2}\n      - OPENAI_API_KEY=${OPENAI_API_KEY:-}\n"
            "      - LITERAL=x\n"
        )
        write_image_config(task.path.parent, "org/api:1", **IMAGE_CONFIGS["org/api:1"]["config"])
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        prepared = load_compose_document(task)["services"]["user-sim"]["environment"]
        assert prepared == {"TAU2_USER_MODEL": "gpt-5.2", "OPENAI_API_KEY": "", "LITERAL": "x"}

        monkeypatch.setenv("TAU2_USER_MODEL", "some-local-model")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-host")
        document = server._compose_document(task)
        assert document["services"]["user-sim"]["environment"] == {
            "TAU2_USER_MODEL": "some-local-model",
            "OPENAI_API_KEY": "sk-host",
            "LITERAL": "x",
        }
        assert document["services"]["user-sim"]["image"] == "org/api:1"
        assert "TAU2_USER_MODEL" not in document["services"]["main"].get("environment", {})

        monkeypatch.delenv("TAU2_USER_MODEL")
        monkeypatch.delenv("OPENAI_API_KEY")
        document = server._compose_document(task)
        assert document["services"]["user-sim"]["environment"] == prepared

    def test_overlay_compose_task_starts_without_the_task_user_and_switches_after_the_overlay(
        self, tmp_path, monkeypatch
    ):
        """The compose adapter's bootstrap execs run through the facade default before any overlay RUN line has
        created the task's user, so an overlay task starts with no default user and gets it afterwards."""
        import environment_servers.harbor.sandboxes as sandboxes_module
        from nemo_gym.sandbox import SandboxHandle

        class DockerLikeProvider:
            """exec behaves like `docker exec --user <name>`: it fails until the user exists in the container."""

            name = "fake"

            def __init__(self):
                self.created, self.execs, self.users = [], [], {"root"}

            def __contains__(self, key):  # `_compose_document` asks `"opensandbox" in provider`
                return False

            async def create(self, spec):
                self.created.append(spec)
                return SandboxHandle(str(len(self.created)), self.name, None)

            async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
                self.execs.append({"user": user, "command": command})
                if user not in (None, "root", 0) and str(user) not in self.users:
                    return SandboxExecResult("", f"unable to find user {user}", 126, error_type="sandbox")
                if "useradd" in command:
                    self.users.add("agent")
                return SandboxExecResult("", "", 0)

            async def upload_file(self, handle, source_path, target_path):
                pass

            def validate_networking(self):
                pass

            async def network_address(self, handle):
                return "192.0.2.1"

            async def set_hosts(self, handle, hosts):
                pass

            async def close(self, handle):
                pass

            async def aclose(self):
                pass

        server, task, _, _ = make_server(
            tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nRUN useradd -m agent\nUSER agent\n"
        )
        (task.path / "task.toml").write_text(TASK_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    command: ['sleep', 'infinity']\n"
        )
        task = load_task(task.path)
        assert task.user == "agent" and task.overlay and task.needs_compose
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        provider = DockerLikeProvider()
        monkeypatch.setattr(sandboxes_module, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
        monkeypatch.setattr(sandboxes_module, "resolve_provider_config", lambda ref, cfg: provider)

        async def run():
            # The overlay runs inside the group's start (round 13), once main exists and before its command.
            compose = await server._create_compose(task, "r1")
            await asyncio.sleep(0)  # the command's exec is a task the adapter started; let it record itself
            main = compose.services["main"]
            useradd = next(i for i, e in enumerate(provider.execs) if "useradd" in e["command"])
            run_sh = next(i for i, e in enumerate(provider.execs) if "run.sh" in e["command"])
            assert useradd < run_sh
            assert provider.created[0].user is None and main.default_user is None
            # Nothing before the overlay created the user ran as it; the seed installs the default afterwards.
            assert "agent" not in {e["user"] for e in provider.execs[:useradd]}, provider.execs
            names = server._overlay_names.pop("r1")
            await server._set_default_user(main, task, names)
            assert main.default_user == "agent"
            # From here every exec that names no user runs as the task's user.
            await main.exec("id")
            assert provider.execs[-1]["user"] == "agent"

        asyncio.run(run())

    def test_overlay_run_lines_precede_mains_uploads_and_command_in_a_compose_group(self, tmp_path, monkeypatch):
        """Review (round 13): for an overlay Compose task, main's bind-mount uploads and its command ran inside the
        group's start, before the seed applied the overlay, so a `RUN` writing defaults clobbered the uploaded
        file and `RUN mkdir /data` failed on the folder the upload had created. The overlay now runs as main's
        setup inside the start: RUN lines, then the uploads, then the command (Harbor's build, then `compose up`
        with the mounts). A failing RUN line is still the seed's 422 reason, and leaves no names behind."""
        import environment_servers.harbor.sandboxes as sandboxes_module
        from nemo_gym.sandbox import SandboxHandle

        class RecordingProvider:
            name = "fake"

            def __init__(self):
                self.events, self.created, self.closed = [], [], []

            def __contains__(self, key):
                return False

            async def create(self, spec):
                self.created.append(spec)
                return SandboxHandle(str(len(self.created)), self.name, None)

            async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
                self.events.append(("exec", handle.sandbox_id, command))
                if "exit 7" in command:
                    return SandboxExecResult("", "boom", 7)
                return SandboxExecResult("", "", 0)

            async def upload_file(self, handle, source_path, target_path):
                self.events.append(("upload", handle.sandbox_id, str(target_path)))

            def validate_networking(self):
                pass

            async def network_address(self, handle):
                return "192.0.2.1"

            async def set_hosts(self, handle, hosts):
                pass

            async def close(self, handle):
                self.closed.append(handle.sandbox_id)

            async def aclose(self):
                pass

        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM ubuntu:24.04\nRUN mkdir -p /data && echo default > /data/seed.csv\n",
        )
        (task.path / "task.toml").write_text(TASK_TOML)
        (task.path / "environment" / "seed.csv").write_text("uploaded")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    command: ['sleep', 'infinity']\n"
            "    volumes:\n      - ./seed.csv:/data/seed.csv\n      - ${HOST_VERIFIER_LOGS_PATH}:/logs/verifier\n"
            "  side:\n    image: org/api:1\n    command: ['sleep', 'infinity']\n"
        )
        write_image_config(task.path.parent, "org/api:1", Cmd=["serve"])  # beside the base image's record
        task = load_task(task.path)
        assert task.overlay and task.needs_compose
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        provider = RecordingProvider()
        monkeypatch.setattr(sandboxes_module, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
        monkeypatch.setattr(sandboxes_module, "resolve_provider_config", lambda ref, cfg: provider)

        async def run():
            compose = await server._create_compose(task, "r1")
            await asyncio.sleep(0)  # the command's exec is a task the adapter started; let it record itself
            main_id = compose.services["main"]._require_handle().sandbox_id
            events = [(kind, what) for kind, sandbox_id, what in provider.events if sandbox_id == main_id]
            overlay = next(i for i, (kind, what) in enumerate(events) if kind == "exec" and "echo default" in what)
            upload = next(i for i, (kind, what) in enumerate(events) if kind == "upload" and what == "/data/seed.csv")
            logs = next(
                i
                for i, (kind, what) in enumerate(events)
                if kind == "exec" and what.startswith("mkdir -p /logs/verifier")
            )
            command = next(i for i, (kind, what) in enumerate(events) if kind == "exec" and "run.sh >" in what)
            assert overlay < upload < command and logs < command, events
            # The overlay ran once, on main only, and its result waits for the seed under the episode key.
            assert sum(1 for kind, _, what in provider.events if kind == "exec" and "echo default" in what) == 1
            assert server._overlay_names.pop("r1") == {}
            # The group's start budget carries the overlay's own on top of the sandbox ready timeout.
            assert (
                compose.timeout_s == server.config.sandbox_ready_timeout_s + task.config.environment.build_timeout_sec
            )
            await server._stop_sandbox("r1")

        asyncio.run(run())
        assert server._sandboxes == {}
        # A RUN line that fails is the task's Dockerfile failing: HarborOverlayError (the seed's 422), nothing kept.
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN exit 7\n")
        failing = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = failing.digest
        provider = RecordingProvider()
        with pytest.raises(overlay.HarborOverlayError, match="RUN line 1/1"):
            asyncio.run(server._create_compose(failing, "r2"))
        assert "r2" not in server._overlay_names and sorted(provider.closed) == sorted(h for h in provider.closed)
        assert len(provider.closed) == len(provider.created) >= 1
        # No upload or command ever ran on main after the failed line.
        assert not any(kind == "upload" and what == "/data/seed.csv" for kind, _, what in provider.events)

    def test_sidecar_specs_do_not_inherit_the_task_user(self, tmp_path, monkeypatch):
        # Main's own user and working directory, neither of which a sidecar image has: a sidecar spec that
        # repeated them would be inheritance, not the image's own defaults (review, #4689).
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER agent\nWORKDIR /main-only\n"
        )
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        assert task.user == "agent" and task.workdir == "/main-only"
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        captured = {}

        class FakeComposeGroup:
            def __init__(self, provider_config, path, *, service_specs, timeout_s, **kwargs):
                captured["specs"] = service_specs

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FakeComposeGroup)
        server.config.artifacts_dir = tmp_path / "artifacts"
        asyncio.run(server._create_compose(task, "r1"))
        specs = captured["specs"]
        assert specs["main"].user == "agent"
        assert {name: spec.user for name, spec in specs.items() if name != "main"} == {"api": None, "kafka": None}
        # Nor main's working directory: a sidecar's image decides its own (the adapter applies the recorded one).
        assert specs["main"].workdir == "/main-only"
        assert {name: spec.workdir for name, spec in specs.items() if name != "main"} == {"api": None, "kafka": None}

    def test_compose_document_turns_task_bind_mounts_into_uploads_from_the_task_copy(self, tmp_path, monkeypatch):
        # M22: the task-aw pattern, a sidecar bind-mounting ./seed.json from the task folder. The server plans it
        # as an upload whose source is the file in its own task copy; named and external volumes resolve through
        # the provider block's volume_sources, which also reaches the Compose adapter.
        server, task, _, documents, _ = self.compose_server(tmp_path, monkeypatch)
        (task.path / "environment" / "seed.json").write_text('{"records": []}')
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: org/agent:1\n    volumes:\n      - ${HOST_VERIFIER_LOGS_PATH}:/logs/verifier\n"
            "  api:\n    image: org/api:1\n    volumes:\n      - ./seed.json:/app/seed.json:ro\n      - models:/models\n"
            "  kafka:\n    image: org/kafka:1\n    volumes:\n      - pgdata:/var/lib/kafka\n"
            "volumes:\n  models:\n    external: true\n  pgdata:\n"
        )
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        with pytest.raises(RuntimeError, match="external volume 'models' needs a `volume_sources` entry"):
            server._compose_document(task)
        # The deployment also maps `pgdata`, another task's external volume; this task's `pgdata` is ordinary.
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {
                "sandbox": {
                    "opensandbox": {},
                    "volume_sources": {"models": "shared/models", "pgdata": "datasets/pgdata"},
                }
            },
        )
        document = server._compose_document(task)
        api, main = document["services"]["api"], document["services"]["main"]
        assert api["x-sandbox"]["uploads"] == [
            {"source": str(task.path / "environment" / "seed.json"), "target": "/app/seed.json"}
        ]
        assert api["volumes"] == [{"type": "volume", "source": "models", "target": "/models", "read_only": False}]
        # The /logs tree is not a mount: the adapter creates it in main before main's command (round 13).
        assert "volumes" not in main
        assert main["x-sandbox"] == {"logs": [{"source": "/logs/verifier", "target": "/logs/verifier"}]}
        captured = {}

        class FakeComposeGroup:
            def __init__(
                self, provider_config, path, *, service_specs, timeout_s, volume_sources, upload_root, **kwargs
            ):
                captured["volume_sources"] = volume_sources
                captured["upload_root"] = upload_root
                captured["document"] = yaml.safe_load(Path(path).read_text())
                captured["kwargs"] = kwargs

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FakeComposeGroup)
        asyncio.run(type(server)._create_compose(server, task, "r1"))  # compose_server() replaced the instance's
        # Review (round 13): only what this document resolves externally reaches the adapter, which treats every
        # name in its mapping as externally managed; the task's ordinary `pgdata` stays the group's own.
        assert captured["volume_sources"] == {"models": "shared/models"}
        assert captured["document"]["services"]["api"]["x-sandbox"]["uploads"] == api["x-sandbox"]["uploads"]
        # The adapter may only read upload sources from the server's own task copy, the one the planner saw.
        assert captured["upload_root"] == task.path
        # No sidecar on another block, and no overlay: nothing per service, no setup callback.
        assert captured["kwargs"]["service_volume_sources"] == {} and captured["kwargs"]["service_setup"] == {}

    @pytest.mark.parametrize("service", ["main", "api"])
    async def test_compose_document_refuses_a_task_authored_x_sandbox(self, tmp_path, monkeypatch, service):
        # Review blocker: a task file that writes the planner's output form itself would have the server upload
        # any readable host file into the sandbox. The seed fails naming the field; nothing starts.
        server, task, compose, documents, _ = self.compose_server(tmp_path, monkeypatch)
        smuggled = "    x-sandbox:\n      uploads:\n        - {source: /etc/passwd, target: /tmp/passwd}\n"
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: org/agent:1\n"
            + (smuggled if service == "main" else "")
            + "  api:\n    image: org/api:1\n"
            + (smuggled if service == "api" else "")
            + "  kafka:\n    image: org/kafka:1\n"
        )
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        with pytest.raises(RuntimeError, match=f"service {service!r} declares x-sandbox"):
            server._compose_document(task)
        response = await run_episode(server, task)
        assert response.failure is not None and response.failure.stage == "seed"
        assert "x-sandbox" in response.failure.failure_reason
        assert documents == []

    async def test_compose_task_seeds_a_group_and_leaves_task_context(self, tmp_path, monkeypatch):
        server, task, compose, documents, _ = self.compose_server(tmp_path, monkeypatch)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        document = documents[0]
        assert set(document["services"]) == {"main", "api", "kafka"}
        # The non-root sidecar keeps its image user and gets no host injection.
        assert "user" not in document["services"]["kafka"] and document["services"]["kafka"]["x-sandbox"] == {
            "hosts": []
        }
        main = compose.services["main"]
        # The healthcheck was polled until it passed, then the task context was written for the harness.
        assert main.probes == 3
        context_write = next(c for c in main.execs if "/tmp/.nemo-gym/task.json" in c["command"])
        assert context_write["user"] == "root"
        assert "playwright" in context_write["command"] and "/app/.skills" in context_write["command"]
        # Everything above happened before the harness got its session.
        seed = server.server_client.seed_bodies()[0]
        assert seed["sandbox_access"]["connection"]["provider_config_ref"] == "sandbox"

    async def test_separate_verifier_collects_from_sidecars_and_stops_the_group(self, tmp_path, monkeypatch):
        server, task, compose, _, verifier = self.compose_server(tmp_path, monkeypatch)
        main = compose.services["main"]
        api = compose.services["api"] = MainWatchingSandbox(
            main=main, present={"/var/log/api": "dir"}, archives={"/var/log/api": archive_of({"api.log": "ok"})}
        )
        kafka = compose.services["kafka"] = MainWatchingSandbox(main=main)

        result = (await run_episode(server, task)).result

        assert result.reward == 1.0, result
        hook = next(c for c in kafka.execs if "dump-topics" in c["command"])
        assert hook["command"].startswith("sh -c ")
        # Harbor's phased collection: the agent's container was already stopped when the sidecar hook ran
        # and when the sidecar's artifact was looked up, while the sidecars themselves were still up.
        assert kafka.main_stopped_when_seen == [True] and api.main_stopped_when_seen == [True]
        assert any("/var/log/api" in c["command"] for c in api.execs)
        # The main container's own /logs/artifacts was looked up (before it was stopped: it answered).
        assert any(c["command"].startswith("if [ -d /logs/artifacts") for c in main.execs)
        assert compose.stopped and verifier.stopped
        # The group left the ownership table when it was stopped; the final cleanup had nothing to do.
        assert not server._sandboxes and not server._sessions

    async def test_a_separate_verifier_without_sidecar_work_stops_the_agent_side_once(self, tmp_path, monkeypatch):
        server, task, compose, _, verifier = self.compose_server(tmp_path, monkeypatch)
        toml = COMPOSE_TOML.replace('artifacts = [{ source = "/var/log/api", service = "api" }]\n', "").replace(
            '[[verifier.collect]]\ncommand = "dump-topics > /logs/artifacts/topics.txt"\nservice = "kafka"\n', ""
        )
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        assert not task.config.artifacts and not task.config.verifier.collect
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        stops: list[str] = []
        compose.services["main"].stop = _record_stop(stops, "main", compose.services["main"])
        original_stop = compose.stop

        async def stop_group():
            stops.append("group")
            await original_stop()

        compose.stop = stop_group

        result = (await run_episode(server, task)).result

        assert result.reward == 1.0, result
        # Main is stopped on its own first, then the group once; nothing stops a second time.
        assert stops == ["main", "group"] and verifier.stopped

    async def test_a_check_started_inside_the_start_period_is_not_a_counted_failure(self, tmp_path, monkeypatch):
        toml = (
            TASK_TOML + '\n[environment.healthcheck]\ncommand = "test -f /tmp/ready"\nstart_period_sec = 0.05\n'
            "start_interval_sec = 0.01\ninterval_sec = 0.01\nretries = 1\n"
        )
        server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml)

        @dataclass
        class SlowFirstProbe(FakeSandbox):
            probes: int = 0

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                self.probes += 1
                if self.probes == 1:
                    # The attempt begins inside the start period and ends after it, failing.
                    await asyncio.sleep(0.1)
                    return SandboxExecResult(stdout="", stderr="", return_code=1)
                return OK

        sandbox = SlowFirstProbe()
        # With retries = 1 a counted failure would raise here; Harbor decides the start period before the attempt.
        await server._wait_healthy(sandbox, task)
        assert sandbox.probes == 2
        # After the start period a failing attempt counts, and retries = 1 ends the wait.
        late = SlowFirstProbe(probes=5)
        late.spec_env = {}

        async def failing(command, **kwargs):
            return SandboxExecResult(stdout="", stderr="", return_code=1)

        late.exec = failing
        await asyncio.sleep(0.06)
        with pytest.raises(RuntimeError, match="Healthcheck failed 1 times"):
            await server._wait_healthy(late, task)

    async def test_compose_document_is_resolved_and_written_once_per_task(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        parses: list[str] = []
        resolve = server._compose_document

        def counting(task):
            parses.append(task.task_id)
            if parses == ["hello"]:
                raise RuntimeError("compose-images.json unreadable")
            return resolve(task)

        monkeypatch.setattr(server, "_compose_document", counting)
        # A failed resolution is not cached: the next episode tries again.
        with pytest.raises(RuntimeError, match="unreadable"):
            await server._compose_document_for(task)
        first, second = await asyncio.gather(server._compose_document_for(task), server._compose_document_for(task))
        third = await server._compose_document_for(task)
        # Concurrent first episodes shared one resolution, later ones reuse it: one parse, one file.
        assert parses == ["hello", "hello"] and first is second is third
        document, path = first
        assert set(document["services"]) == {"main", "api", "kafka"}
        assert yaml.safe_load(path.read_text()) == document and path.is_relative_to(server._artifacts_dir)

    async def test_gpu_tasks_route_to_the_gpu_provider(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(TASK_TOML.replace("cpus = 1", 'cpus = 1\ngpus = 1\ngpu_types = ["H100"]'))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        server.config.gpu_sandbox_provider = "sandbox_gpu"
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        assert server._provider_ref(task) == "sandbox_gpu"
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.gpu == 1 and spec.resources.gpu_type is None
        server.config.request_gpu_type = True
        assert server._sandbox_spec(task, "/app").resources.gpu_type == "H100"
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        seed = server.server_client.seed_bodies()[0]
        assert seed["sandbox_access"]["connection"]["provider_config_ref"] == "sandbox_gpu"

    def gpu_sidecar_server(self, tmp_path, monkeypatch, extra: str = ""):
        """A CPU task whose kafka sidecar reserves one GPU (plus ``extra`` kafka lines); ``sandbox_gpu`` is the
        GPU block, with its own metadata and networking."""
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(COMPOSE_TOML.replace("cpus = 1", 'cpus = 1\ngpu_types = ["H100"]'))
        (task.path / "environment" / "docker-compose.yaml").write_text(
            COMPOSE_YAML
            + "    deploy:\n      resources:\n        limits:\n          memory: 2g\n        reservations:\n"
            "          devices:\n            - driver: nvidia\n              capabilities: [gpu]\n              count: 1\n"
            + extra
        )
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        server.config.gpu_sandbox_provider = "sandbox_gpu"
        server.config.request_gpu_type = True
        server.config.artifacts_dir = tmp_path / "artifacts"
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {
                "sandbox": {"opensandbox": {}},
                "sandbox_gpu": {
                    "opensandbox": {"networking": {"enabled": True}},
                    "default_metadata": {"pool": "h100"},
                },
            },
        )
        return server, task

    def test_gpu_sidecars_route_to_the_gpu_provider_per_service(self, tmp_path, monkeypatch):
        """M28: a sidecar that reserves GPUs through deploy.resources runs on the GPU block, the task's main
        service stays where the task's own declaration puts it, and the adapter gets per-service providers."""
        import yaml

        server, task = self.gpu_sidecar_server(tmp_path, monkeypatch)
        captured = {}

        class FakeComposeGroup:
            def __init__(
                self, provider_config, path, *, service_specs, timeout_s, service_providers, platform, **kwargs
            ):
                captured.update(
                    provider=provider_config,
                    specs=service_specs,
                    providers=service_providers,
                    platform=platform,
                    document=yaml.safe_load(Path(path).read_text()),
                )

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FakeComposeGroup)
        asyncio.run(server._create_compose(task, "r1"))
        # The task declares no GPUs itself: main stays on the task's block, kafka alone goes to the GPU block.
        assert server._provider_ref(task) == "sandbox" and captured["provider"] == {"opensandbox": {}}
        assert captured["providers"] == {"kafka": {"opensandbox": {"networking": {"enabled": True}}}}
        kafka, api = captured["specs"]["kafka"], captured["specs"]["api"]
        # The adapter applies deploy.resources; the server adds the task's gpu_type and the GPU block's metadata.
        assert kafka.resources.gpu is None and kafka.resources.gpu_type == "H100"
        assert kafka.metadata["pool"] == "h100" and kafka.metadata["harbor_role"] == "agent"
        assert "pool" not in api.metadata and api.resources.gpu_type is None
        assert captured["document"]["services"]["kafka"]["deploy"]["resources"] == {
            "limits": {"memory": 2 * 1024**3},
            "reservations": {"gpus": 1},
        }
        assert captured["platform"] == "linux/amd64"

    def test_a_gpu_sidecar_resolves_its_external_volume_through_the_gpu_blocks_mapping(self, tmp_path, monkeypatch):
        """Review (round 13): the reviewer's probe showed the GPU-routed sidecar handed `cpu/models`. The planner
        checks it against the GPU block's mapping and the adapter gets that mapping for the sidecar alone, while
        the group's own mapping stays the task's block's, narrowed to what the document uses."""
        server, task = self.gpu_sidecar_server(
            tmp_path,
            monkeypatch,
            extra="    volumes:\n      - models:/models\nvolumes:\n  models:\n    external: true\n",
        )

        def blocks(gpu_mapping):
            return {
                "sandbox": {"opensandbox": {}, "volume_sources": {"models": "cpu/models", "pgdata": "pg"}},
                "sandbox_gpu": {
                    "opensandbox": {"networking": {"enabled": True}},
                    "default_metadata": {"pool": "h100"},
                    **({"volume_sources": gpu_mapping} if gpu_mapping is not None else {}),
                },
            }

        # Mapped in the task's block only: the sidecar runs on the GPU block, so the seed fails naming the volume.
        monkeypatch.setattr("environment_servers.harbor.sandboxes.get_global_config_dict", lambda: blocks(None))
        with pytest.raises(RuntimeError, match="external volume 'models' needs a `volume_sources` entry"):
            server._compose_document(task)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: blocks({"models": "gpu/models"})
        )
        captured = {}

        class FakeComposeGroup:
            def __init__(self, provider_config, path, *, volume_sources, service_volume_sources, **kwargs):
                captured.update(volume_sources=volume_sources, service_volume_sources=service_volume_sources)

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FakeComposeGroup)
        asyncio.run(server._create_compose(task, "r1"))
        assert captured["service_volume_sources"] == {"kafka": {"models": "gpu/models"}}
        # The group's mapping is the task block's, kept to what the document resolves through it (not `pgdata`).
        assert captured["volume_sources"] == {"models": "cpu/models"}

    async def test_gpu_sidecar_group_starts_through_the_real_adapter(self, tmp_path, monkeypatch):
        """The document the server writes (deploy normalized once) passes the adapter's own pass over the reloaded
        file, and the group starts with the GPU sidecar created on the GPU block's provider: the real
        AsyncSandboxCompose, with only the provider construction replaced."""
        server, task = self.gpu_sidecar_server(tmp_path, monkeypatch)
        cpu, gpu = StubProvider("opensandbox"), StubProvider("opensandbox-gpu")
        monkeypatch.setattr(
            "nemo_gym.sandbox.adapters.docker_compose.create_provider",
            lambda config: gpu if config == {"opensandbox": {"networking": {"enabled": True}}} else cpu,
        )
        compose = await server._create_compose(task, "r1")
        try:
            assert compose._ready and server._sandboxes["r1"] is compose
            assert compose.services["main"]._provider is cpu and compose.services["kafka"]._provider is gpu
            (kafka,) = gpu.created
            # deploy.resources applied by the adapter, the task's gpu_type and the GPU block's metadata by the server.
            assert (kafka.resources.gpu, kafka.resources.gpu_type, kafka.resources.memory_mib) == (1, "H100", 2048)
            assert kafka.metadata["pool"] == "h100" and kafka.image == IMAGE_CONFIGS["org/kafka:1"]["image"]
            assert [spec.image for spec in cpu.created] == [
                IMAGE_CONFIGS["org/agent:1"]["image"],
                IMAGE_CONFIGS["org/api:1"]["image"],
            ]
        finally:
            await compose.stop()
        assert gpu.closed == ["1"] and sorted(cpu.closed) == ["1", "2"]

    def test_platform_at_seed_is_read_from_the_provider_options_as_at_prepare(self, tmp_path, monkeypatch):
        from nemo_gym.tasks.harbor.registry import ImageConfigError

        server, task, _, _, _ = self.compose_server(tmp_path, monkeypatch)

        def declaring(platform):
            (task.path / "environment" / "docker-compose.yaml").write_text(
                COMPOSE_YAML + f"    platform: {platform}\n"
            )
            loaded = load_task(task.path)
            server.config.tasksets["ds"].tasks["hello"] = loaded.digest
            return loaded

        arm = declaring("linux/arm64")
        with pytest.raises(
            ImageConfigError,
            match="service 'kafka' platform 'linux/arm64' does not match the deployment platform linux/amd64",
        ):
            server._compose_document(arm)
        # The same setting prepare reads (compose_platform_from_overrides) admits the task at seed, once the
        # recorded images are for that platform too (the loader-side image check reads the same option).
        server.config.sandbox_provider_options = {"platform": {"os": "linux", "arch": "arm64"}}
        (task.path.parent / "compose-images.json").write_text(
            json.dumps({image: {**record, "architecture": "arm64"} for image, record in IMAGE_CONFIGS.items()})
        )
        assert server._compose_platform() == "linux/arm64"
        assert server._compose_document(arm)["services"]["kafka"]["platform"] == "linux/arm64"
        with pytest.raises(ImageConfigError, match="does not match the deployment platform linux/arm64"):
            server._compose_document(declaring("linux/amd64"))

    async def test_extra_hosts_on_a_non_root_sidecar_fail_prepare_and_seed_for_the_same_reason(
        self, tmp_path, monkeypatch
    ):
        from nemo_gym.tasks.harbor.image_configs import record_compose_images
        from nemo_gym.tasks.harbor.registry import ImageConfigError

        server, task, compose, documents, _ = self.compose_server(tmp_path, monkeypatch)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            COMPOSE_YAML + "    extra_hosts: ['licence.example:198.51.100.7']\n"
        )
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        reason = (
            "hello: environment/docker-compose.yaml: service 'kafka' extra_hosts cannot be applied without host "
            "injection: the image runs as user 'appuser', so the sidecar takes no host entries"
        )
        # Prepare: the recorded configurations say kafka runs as appuser, so the task fails there, naming the service.
        with pytest.raises(ImageConfigError) as excinfo:
            record_compose_images([task], task.path.parent, sandbox_provider="opensandbox")
        assert str(excinfo.value) == reason
        # Seed: the server decides the host opt-out with the same predicate and refuses the same document with the
        # same words, in its precheck: a terminal 422 before any group, not a failure inside the start retried per
        # episode (the resolution path raises the same, as the last line).
        with pytest.raises(ImageConfigError) as excinfo:
            server._compose_document(task)
        assert str(excinfo.value) == reason
        response = await run_episode(server, task)
        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        assert response.failure.failure_reason == f"Compose group of task 'hello': {reason}"
        assert documents == [] and not compose.stopped and not server._sandboxes
        # A root kafka image takes the entries on both paths.
        configs = {**IMAGE_CONFIGS, "org/kafka:1": {**IMAGE_CONFIGS["org/kafka:1"], "config": {"Cmd": ["kafka"]}}}
        (task.path.parent / "compose-images.json").write_text(json.dumps(configs))
        assert record_compose_images([task], task.path.parent, sandbox_provider="opensandbox") is not None
        kafka = server._compose_document(task)["services"]["kafka"]
        assert kafka["extra_hosts"] == {"licence.example": "198.51.100.7"} and "x-sandbox" not in kafka

    def test_a_gpu_sidecars_options_are_checked_against_the_gpu_block_at_seed(self, tmp_path, monkeypatch):
        from nemo_gym.tasks.harbor.registry import ImageConfigError

        server, task = self.gpu_sidecar_server(tmp_path, monkeypatch, extra="    tmpfs: [/run]\n")
        # Docker task block, OpenSandbox GPU block: kafka runs on the GPU block, which applies no tmpfs.
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"docker": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        with pytest.raises(
            ImageConfigError, match="service 'kafka' tmpfs is not supported by the opensandbox sandbox provider"
        ):
            server._compose_document(task)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"docker": {}}},
        )
        assert server._compose_document(task)["services"]["kafka"]["tmpfs"] == ["/run"]

    def test_compose_document_normalizes_the_example_fields_and_names_what_the_provider_cannot_apply(
        self, tmp_path, monkeypatch
    ):
        from nemo_gym.tasks.harbor.registry import ImageConfigError

        server, task, _, _, _ = self.compose_server(tmp_path, monkeypatch)
        # alibaba-ctf style: restart always and a custom network, which need nothing else.
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: org/agent:1\n    restart: always\n    networks: [agent_network]\n"
            "  api:\n    image: org/api:1\n    restart: always\n    networks:\n      agent_network:\n"
            "        aliases: [challenge]\n  kafka:\n    image: org/kafka:1\n"
            "networks:\n  agent_network:\n    driver: bridge\n"
        )
        task = load_task(task.path)
        services = server._compose_document(task)["services"]
        assert services["main"]["restart"] == "always" and services["main"]["networks"] == {"agent_network": {}}
        assert services["api"]["networks"] == {"agent_network": {"aliases": ["challenge"]}}
        # A field OpenSandbox cannot apply fails before any sandbox, naming the field and the provider.
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: org/agent:1\n  api:\n    image: org/api:1\n"
            "    sysctls:\n      net.core.somaxconn: 1024\n"
        )
        task = load_task(task.path)
        with pytest.raises(
            ImageConfigError,
            match="hello: environment/docker-compose.yaml: service 'api' sysctls is not supported by the opensandbox",
        ):
            server._compose_document(task)

    async def test_failed_healthcheck_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        server, task, compose, _, _ = self.compose_server(tmp_path, monkeypatch)
        compose.services["main"].probes = -100  # never reaches 3 within 3 retries
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "Healthcheck failed" in response.failure.failure_reason
        assert compose.stopped and not server._sandboxes


class TestOracleSolution:
    """The seed places the task's reference solution for the Harbor oracle, and for no other harness.

    Harbor's oracle (0.22.0 ``agents/oracle.py``) copies ``solution/`` to ``/solution`` and runs
    ``bash /solution/solve.sh``; here the environment server does the copy, as root like its other seed
    uploads, readable by the task user, before the oracle's session opens. The oracle reads no task folder.
    """

    SOLUTION_TOML = (
        TASK_TOML.replace("timeout_sec = 300.0\n", 'timeout_sec = 300.0\nuser = "agent"\n')
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
        assert names == {"solve.sh", "data", "data/answer.txt"}
        assert modes["solve.sh"] == 0o700  # the archive keeps the author's mode; the chmod opens it in place
        assert uploaded_file(sandbox, "solve.sh") == [SOLVE_SH.encode()]
        # The upload preceded the agent seed, which carries the task user and the resolved [solution].env.
        assert uploads_at_seed == [1]
        seed = server.server_client.seed_bodies()[0]
        assert seed["sandbox_access"]["user"] == "agent"
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
                    return _fails("tar: /solution: Cannot mkdir: Permission denied")
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


def test_server_config_holds_only_deployment_settings():
    """A setting belongs here only if every dataset on the same deployment wants the same value.

    Task facts stay in task.toml, dataset settings in dataset.toml's [gym] table, image mirrors and
    setup commands in the sandbox provider. Adding a field means changing this set in the same PR.
    """
    own = set(HarborEnvironmentServerConfig.model_fields) - set(BaseEnvironmentServerConfig.model_fields)
    assert own == {
        "agent_server",
        "agent_outcome_policy",
        "tasksets",
        "sandbox_provider",
        "sandbox_ready_timeout_s",
        "sandbox_ttl_slack_s",
        "sandbox_provider_options",
        "sandbox_metadata",
        "sandbox_resources_override",
        "gpu_sandbox_provider",
        "request_gpu_type",
        "verifier_grace_s",
        "artifacts_dir",
        "secrets",
        "enforce_task_timeouts",
        "harness_setup_allowance_s",
        "step_setup_timeout_s",
        "image_pull_retries",
        "separate_verifier_for_artifacts",
        "task_mcp_grants",
        "turn_sessions",
        "compose_variables",
        "compose_overlay",
    }
    source = inspect.getsource(HarborEnvironmentServerConfig).lower()
    assert not any(word in source for word in ("terminal", "tb4", "tb2", "swe", "nextjs", "hello-world"))


class TestSeedGuards:
    """Defects from the first cluster run: requests above limits, and a create the ready timeout did not bound."""

    def test_sandbox_spec_clamps_requests_above_the_tasks_limits(self, tmp_path, monkeypatch, caplog):
        toml = TASK_TOML.replace("storage_mb = 10240", "storage_mb = 4096")
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=toml,
            sandbox_provider_options={"resource_requests": {"cpu": 0.5, "memory_mib": 512, "disk_gib": 10}},
        )
        monkeypatch.setattr("environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {}})

        with caplog.at_level("WARNING"):
            spec = server._sandbox_spec(task, "/app")

        assert spec.resources.disk_gib == 4
        assert spec.provider_options["resource_requests"] == {"cpu": 0.5, "memory_mib": 512, "disk_gib": 4}
        assert "resource_requests.disk_gib=10 exceeds the disk_gib limit 4 task hello (agent)" in caplog.text
        # The configured options are not mutated, and a task within the requests gets them unchanged.
        assert server.config.sandbox_provider_options["resource_requests"]["disk_gib"] == 10
        caplog.clear()
        server2, task2, _, _ = make_server(
            tmp_path / "two", monkeypatch, sandbox_provider_options={"resource_requests": {"disk_gib": 10}}
        )
        spec2 = server2._sandbox_spec(task2, "/app")
        assert spec2.provider_options == {"resource_requests": {"disk_gib": 10}} and caplog.text == ""
        # "limits" and absent requests pass through untouched.
        server3, task3, _, _ = make_server(
            tmp_path / "three", monkeypatch, toml=toml, sandbox_provider_options={"resource_requests": "limits"}
        )
        assert server3._sandbox_spec(task3, "/app").provider_options == {"resource_requests": "limits"}

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
        # Budgets wide enough for a loaded CI runner (30 server suites share one small host): the deadline must fire
        # only once the fake sandbox exists and `mkdir` hangs, else the cleanup has nothing to stop and the test
        # fails for timing alone; the assertions below do not depend on the exact figures.
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox, default_episode_timeout_seconds=0.5)
        server.config.cleanup_timeout_seconds = 0.5

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
        # The grace covers the provider's orphan reap, so the seed sees what the reap found, not a bare timeout.
        assert SEED_CREATE_GRACE_S >= sandbox_base.ORPHAN_REAP_TIMEOUT_S

    async def test_separate_verifier_create_is_bounded_by_the_ready_timeout(self, tmp_path, monkeypatch):
        """The same deadline bounds the verifier's create: the finished rollout is kept, masked, not held
        for the provider's retry budget."""
        agent = AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, sandbox_ready_timeout_s=0.05)
        (task.path / "task.toml").write_text(SEPARATE_TOML)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr("environment_servers.harbor.sandboxes.SEED_CREATE_GRACE_S", 0.0)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        stopped = []

        class HangingStart:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                await asyncio.get_running_loop().create_future()

            async def stop(self):
                stopped.append(True)

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", HangingStart)
        started = asyncio.get_running_loop().time()
        response = await run_episode(server, task)
        assert asyncio.get_running_loop().time() - started < 5
        assert response.failure is None, response.failure
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is True
        assert result.failure_kind == "provider_unavailable"
        assert "HarborSeedTimeoutError: the sandbox was not ready within 0.05s" in result.failure_reason
        # The agent's sandbox was stopped before the verifier's create; the verifier's entry is released too.
        assert agent.stopped and stopped == [True] and not server._sandboxes

    async def test_a_providers_own_timeout_inside_the_deadline_is_reported_as_is(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox_ready_timeout_s=100)

        async def provider_timeout(task, workdir, *, episode_key):
            raise TimeoutError("Timed out creating OpenSandbox sandbox after 100s")

        monkeypatch.setattr(server, "_create_sandbox", provider_timeout)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "Timed out creating OpenSandbox sandbox after 100s" in response.failure.failure_reason


@dataclass
class VerifierSandbox(FakeSandbox):
    """A separate verifier's sandbox whose image holds ``/tests/test.sh`` unless told otherwise."""

    owns_tests: bool = True

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command.startswith("if [ -d "):
            self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
            path = command.split("if [ -d ")[1].split(" ]")[0].strip("'")
            kind = "file" if path == "/tests/test.sh" and self.owns_tests else "none"
            return SandboxExecResult(stdout=f"{kind}\n", stderr="", return_code=0)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


class TestBuiltVerifierImage:
    """D1: a separate verifier grades in the image prepare built from ``tests/Dockerfile``."""

    BUILT = "registry.example/r/hb__0123456789abcdef0123456789abcdef-verifier@sha256:feed"

    def built(self, tmp_path, monkeypatch, *, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=None):
        """A separate-mode task with an overlay-mode agent Dockerfile and a ``tests/Dockerfile``, built and recorded;
        ``_create_verifier_sandbox`` is replaced by one handing out ``verifier``."""
        agent = OverlaySandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, dockerfile=OVERLAY_DOCKERFILE)
        (task.path / "task.toml").write_text(toml + '\n[environment.env]\nAPP_MODE = "${HARBOR_TEST_MODE:-grade}"\n')
        (task.path / "tests" / "Dockerfile").write_text(VERIFIER_TESTS_DOCKERFILE)
        record_verifier_build(task.path, self.BUILT)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        verifier = verifier or VerifierSandbox()
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier

    async def test_the_verifier_sandbox_starts_from_the_built_image_under_the_environment_block(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HARBOR_TEST_MODE", "strict")
        server, task, _, _ = self.built(tmp_path, monkeypatch)
        verifier = VerifierSandbox()
        specs = []

        async def start(spec):
            specs.append(spec)

        monkeypatch.setattr(verifier, "start", start, raising=False)
        monkeypatch.setattr("environment_servers.harbor.sandboxes.resolve_provider_config", lambda *args: {})
        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", lambda config: verifier)
        monkeypatch.delattr(server, "_create_verifier_sandbox")  # the real one, not the test double
        assert await server._create_verifier_sandbox(task) is verifier
        (spec,) = specs
        # Harbor's effective verifier environment is a copy of [environment]: its env (resolved) and resources
        # reach the verifier sandbox, the agent image's Dockerfile ENV and WORKDIR do not, and the image keeps its
        # own default user and working directory.
        assert spec.image == self.BUILT and spec.metadata["harbor_role"] == "verifier"
        # ... under the CPU-count hints for the copied `cpus = 1`.
        assert spec.env == cpu_cap_env(1) | {"APP_MODE": "strict"} and "PATH" not in spec.env
        assert spec.workdir is None and spec.user is None
        assert (spec.resources.cpu, spec.resources.memory_mib) == (1, 2048)
        # The verifier keeps the episode-sized lifetime the agent's sandbox gets, not timeout_sec plus slack.
        assert spec.ttl_s == server._sandbox_spec(task, task.workdir).ttl_s

    async def test_the_verifier_runs_test_sh_in_the_built_image_without_uploading_tests(self, tmp_path, monkeypatch):
        server, task, agent, verifier = self.built(tmp_path, monkeypatch)
        # The agent image is overlaid (its Dockerfile has RUN lines); the verifier image was built, so it is not.
        assert task.verifier_image == self.BUILT and task.overlay and task.image == "debian:bookworm-slim"

        result = (await run_episode(server, task)).result

        assert result.reward == 1.0 and result.verifier_mode == "separate", result
        assert result.mask_sample is False and result.failure_kind is None
        # The agent's sandbox took the overlay; the verifier's did not, and no tests archive went up: the image
        # owns /tests, as Harbor's skip_tests_upload assumes. Only the ATIF trajectory the verifier reads goes up.
        assert any("apt-get" in c["command"] for c in agent.execs)
        assert not any("apt-get" in c["command"] for c in verifier.execs)
        assert len(verifier.uploads) == 1 and len(uploaded_file(verifier, "trajectory.json")) == 1
        assert not uploaded_file(verifier, "test.sh")
        run = next(c for c in verifier.execs if "test.sh" in c["command"] and "timeout" in c["command"])
        assert "bash /tests/test.sh" in run["command"]
        assert agent.stopped and verifier.stopped

    async def test_a_built_image_without_test_sh_gets_the_upload_and_one_warning(self, tmp_path, monkeypatch, caplog):
        server, task, _, verifier = self.built(tmp_path, monkeypatch, verifier=VerifierSandbox(owns_tests=False))
        with caplog.at_level("WARNING", logger="environment_servers.harbor.verifier"):
            assert (await run_episode(server, task)).result.reward == 1.0
            assert (await run_episode(server, task, rollout="r2")).result.reward == 1.0
        # tests/ was uploaded both times (the trajectory travels as its own archive); the image was named once.
        assert len(uploaded_file(verifier, "test.sh")) == 2
        warnings = [r for r in caplog.records if "has no /tests/test.sh" in r.getMessage()]
        assert len(warnings) == 1 and self.BUILT in warnings[0].getMessage()

    async def test_a_verifier_environment_block_without_an_image_seeds_the_built_image_with_its_settings(
        self, tmp_path, monkeypatch
    ):
        toml = (
            TASK_TOML
            + '\n[verifier.environment]\ncpus = 3\nworkdir = "/grade"\n\n[verifier.environment.env]\nK = "v"\n'
        )
        server, task, _, verifier = self.built(tmp_path, monkeypatch, toml=toml)
        assert task.verifier_image == self.BUILT
        spec = server._sandbox_spec(
            task,
            task.config.verifier.environment.workdir,
            environment=task.config.verifier.environment,
            image=self.BUILT,
        )
        assert spec.image == self.BUILT and spec.env == cpu_cap_env(3) | {"K": "v"} and spec.workdir == "/grade"
        assert spec.resources.cpu == 3
        assert (await run_episode(server, task)).result.reward == 1.0
        assert len(verifier.uploads) == 1 and len(uploaded_file(verifier, "trajectory.json")) == 1

    @pytest.mark.parametrize("shape", ["no_block", "block_without_image"])
    async def test_without_a_record_the_task_is_refused_by_the_loader_and_at_seed(self, tmp_path, monkeypatch, shape):
        """Harbor grades in the image built from tests/Dockerfile; without a record nothing stands in for it: not the
        agent image (where the tests' tooling is usually missing, so every episode would score 0 with no error) and
        not a refusal that only comes at seed. The loader refuses the task like an unrecorded environment/ build."""
        toml = (
            SEPARATE_IN_TASK_IMAGE_TOML if shape == "no_block" else TASK_TOML + "\n[verifier.environment]\ncpus = 2\n"
        )
        agent = OverlaySandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, dockerfile=OVERLAY_DOCKERFILE)
        (task.path / "task.toml").write_text(toml)
        (task.path / "tests" / "Dockerfile").write_text(VERIFIER_TESTS_DOCKERFILE)
        with pytest.raises(HarborTaskError) as info:
            load_task(task.path)
        reason = str(info.value)
        assert reason.startswith(f"{task.path / 'tests' / 'Dockerfile'} is the separate verifier's image")
        assert str(task.path.parent / BUILT_IMAGES_FILE) in reason and "`builds:` block" in reason
        assert "declare [verifier.environment].docker_image" in reason
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "is the separate verifier's image" in response.failure.failure_reason
        assert agent.execs == [] and not server._sandboxes

    async def test_the_seed_still_checks_the_verifier_image_invariant(self, tmp_path, monkeypatch):
        """A task with a [verifier.environment] without docker_image and no verifier image cannot come from the
        loader; the seed refuses such an object all the same, naming the record file and the remedies."""
        server, task, agent, _ = self.built(
            tmp_path, monkeypatch, toml=TASK_TOML + "\n[verifier.environment]\ncpus = 2\n"
        )
        broken = replace(task, verifier_image=None)
        assert _verifier_image(task) == self.BUILT and _verifier_image(broken) is None

        async def resolve(task_id, task_data):
            return broken

        monkeypatch.setattr(server, "_resolve_task", resolve)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "has no image for its separate verifier" in response.failure.failure_reason
        assert "built-images.json" in response.failure.failure_reason
        assert "tests/Dockerfile" in response.failure.failure_reason
        assert agent.execs == []


SECRET_TOML = (
    TASK_TOML
    + """
[verifier.env]
JUDGE = "${JUDGE_KEY}"
MODE = "strict"
"""
)


class TestGeneralization:
    """M27: secrets, the ATIF transcript, pull retries, task deadlines, forced separate mode, nested tasks."""

    async def test_secrets_reach_the_test_sh_process_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JUDGE_KEY", raising=False)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, toml=SECRET_TOML, secrets={"JUDGE_KEY": "sk-judge", "OTHER": "o"}
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        # The test.sh run itself, not the root `chmod +x` that precedes it.
        verifier_exec = next(
            call for call in sandbox.execs if "test.sh" in call["command"] and "timeout" in call["command"]
        )
        # Templates in [verifier].env resolve from the secrets block; a secret the task does not name is not
        # passed, so a test.sh that dumps its environment cannot leak a secret set for another taskset.
        assert verifier_exec["env"] == {"JUDGE": "sk-judge", "MODE": "strict"}
        assert "OTHER" not in verifier_exec["env"] and "o" not in verifier_exec["env"].values()
        # Nothing else saw them: the sandbox's own environment, the seed and the agent's turn.
        assert "sk-judge" not in str(server._sandbox_spec(task, "/app").env)
        assert "sk-judge" not in str([call[2] for call in server.server_client.calls])

    async def test_secret_templates_outside_the_verifier_are_not_filled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JUDGE_KEY", raising=False)
        toml = TASK_TOML + '\n[environment.env]\nTOKEN = "${JUDGE_KEY}"\n'
        server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml, secrets={"JUDGE_KEY": "sk-judge"})
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and response.failure.terminal
        assert "JUDGE_KEY" in response.failure.failure_reason

    async def test_a_configured_secret_wins_over_a_same_named_process_variable(self, tmp_path, monkeypatch):
        """resolve_env_templates consults the server's secrets before os.environ: a judge key set for the
        deployment is not shadowed by a stale one in the server process's environment."""
        from environment_servers.harbor.session import resolve_env_templates

        monkeypatch.setenv("JUDGE_KEY", "sk-from-the-process")
        # Without a configured secret the process variable fills the template, so the next assertions are about
        # precedence, not about the variable being ignored.
        assert resolve_env_templates({"JUDGE": "${JUDGE_KEY}"}) == {"JUDGE": "sk-from-the-process"}
        assert resolve_env_templates({"JUDGE": "${JUDGE_KEY}"}, {"JUDGE_KEY": "sk-judge"}) == {"JUDGE": "sk-judge"}
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, toml=SECRET_TOML, secrets={"JUDGE_KEY": "sk-judge"}
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        verifier_exec = next(
            call for call in sandbox.execs if "test.sh" in call["command"] and "timeout" in call["command"]
        )
        assert verifier_exec["env"] == {"JUDGE": "sk-judge", "MODE": "strict"}
        assert "sk-from-the-process" not in str(sandbox.execs)

    async def test_failed_trajectory_export_masks_the_turn_before_test_sh(self, tmp_path, monkeypatch):
        from nemo_gym.atif_export import AtifExportError

        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)

        def broken_export(request, agent_response):
            raise AtifExportError("transcript[3]: unsupported transcript item")

        monkeypatch.setattr(server, "_export_trajectory", broken_export)
        response = await run_episode(server, task)

        result = response.result
        assert result.mask_sample is True and result.failure_kind == "harbor:trajectory_export_failed"
        assert "unsupported transcript item" in result.failure_reason
        assert result.verifier_mode is None and result.response.id == "response"
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert sandbox.stopped and not server._sandboxes

    async def test_a_refusal_turn_exports_as_text_and_the_episode_is_still_graded(self, tmp_path, monkeypatch):
        """A refusal is model behaviour, not a converter defect: the Terminus loop goes on after one, and
        the episode reaches test.sh with the refusal in the transcript instead of being masked."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        turn = agent_response().model_dump(mode="json")
        answer = turn["output"][0]
        turn["output"] = [
            {**answer, "id": "message-1", "content": [{"type": "refusal", "refusal": "I can't help with that."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "(terminal unchanged)"}]},
            answer,
        ]
        server.server_client.turn_response = turn

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is False and result.failure_kind is None
        assert result.reward == 1.0 and result.verifier_rewards == {"reward": 1.0}
        assert any("test.sh" in call["command"] for call in sandbox.execs)
        trajectory = json.loads((Path(result.verifier_logs_dir) / "trajectory.json").read_text())
        steps = trajectory["steps"]
        assert [step["source"] for step in steps] == ["user", "agent", "agent"]
        assert steps[1]["message"] == "I can't help with that." and steps[1]["extra"]["nemo_gym"]["refusal"] is True
        assert steps[1]["observation"]["results"][0]["content"] == "(terminal unchanged)"
        assert steps[2]["message"] == "echo hello > hello.txt"
        assert trajectory["extra"]["nemo_gym"]["conversion"]["reasons"] == ["refusal"]

    async def test_terminus_shaped_transcript_exports_user_turns_as_observations(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        turn = agent_response().model_dump(mode="json")
        turn["output"] = [
            turn["output"][0],
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello.txt created"}]},
            {
                **turn["output"][0],
                "id": "message-2",
                "content": [{**turn["output"][0]["content"][0], "text": "Done."}],
            },
        ]
        server.server_client.turn_response = turn

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        (trajectory_bytes,) = uploaded_file(sandbox, "trajectory.json")
        steps = json.loads(trajectory_bytes)["steps"]
        assert [step["source"] for step in steps] == ["user", "agent", "agent"]
        assert steps[1]["observation"]["results"][0]["content"] == "hello.txt created"
        assert steps[1]["extra"]["nemo_gym"]["observation_source"] == "user_turn"

    async def test_sandbox_access_names_the_agent_user(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, toml=TASK_TOML.replace("[agent]\n", '[agent]\nuser = "agent"\n')
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        await run_episode(server, task)
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["user"] == "agent"

    async def test_task_deadline_ends_an_overrunning_episode(self, tmp_path, monkeypatch):
        toml = TASK_TOML.replace("timeout_sec = 300.0", "timeout_sec = 0.05").replace(
            "timeout_sec = 120.0", "timeout_sec = 0.05"
        )
        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=toml,
            sandbox_ready_timeout_s=0.01,
            harness_setup_allowance_s=0,
            verifier_grace_s=0,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        assert server._task_budget_s(task) == pytest.approx(0.11)
        # The deadline itself starts when the sandbox is ready and leaves the sandbox start out.
        assert server._post_seed_budget_s(task) == pytest.approx(0.1)
        server.server_client.slow_path = "/ng-rollout/r1/v1/responses"

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "agent" and failure.failure_kind == "harbor:task_deadline" and not failure.terminal
        assert (
            "time budget of task 'hello'" in failure.failure_reason
            and "enforce_task_timeouts" in failure.failure_reason
        )
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        # The final unwind still closed the agent session and stopped the sandbox.
        assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
        assert sandbox.stopped and not server._sandboxes and server._active == set()

        # The same slow harness finishes when the run ignores task timeouts.
        server.config.enforce_task_timeouts = False
        server.server_client.slow_path = "/ng-rollout/r2/v1/responses"
        assert (await run_episode(server, task, rollout="r2")).result.reward == 1.0

    @pytest.mark.parametrize("mode", ["shared", "separate"])
    async def test_collect_hooks_count_toward_the_task_deadline(self, tmp_path, monkeypatch, mode):
        """A hook within its own `timeout_sec` must not trip the episode deadline: Harbor gives each hook its
        declared time outside any trial-wide limit, so the budget counts the hooks' limits in both modes."""
        toml = TASK_TOML.replace("timeout_sec = 300.0", "timeout_sec = 0.05").replace(
            "timeout_sec = 120.0", "timeout_sec = 0.05"
        )
        toml += '\n[[verifier.collect]]\ncommand = "sleep 0.25"\ntimeout_sec = 1.0\n'
        if mode == "separate":
            toml = toml.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
            toml += '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n'

        # Separate mode collects /logs/artifacts from the agent's sandbox before the verifier starts.
        @dataclass
        class SlowHook(AgentSandbox if mode == "separate" else FakeSandbox):
            async def exec(self, command, **kwargs):
                if command.startswith("sleep "):
                    await asyncio.sleep(0.25)
                return await super().exec(command, **kwargs)

        agent = SlowHook()
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            agent,
            toml=toml,
            sandbox_ready_timeout_s=0.01,
            harness_setup_allowance_s=0,
            verifier_grace_s=0,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        verifier = FakeSandbox()

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        # Agent and verifier limits, the hook's own limit with the exec grace, and a separate verifier's start.
        separate_start = 0.01 if mode == "separate" else 0
        assert server._post_seed_budget_s(task) == pytest.approx(0.1 + 1 + COLLECT_HOOK_GRACE_S + separate_start)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == mode
        hook = next(call for call in agent.execs if call["command"] == "sleep 0.25")
        assert hook["timeout_s"] == 1 + COLLECT_HOOK_GRACE_S

    async def test_task_deadline_covers_verification_too(self, tmp_path, monkeypatch):
        toml = TASK_TOML.replace("timeout_sec = 300.0", "timeout_sec = 0.05").replace(
            "timeout_sec = 120.0", "timeout_sec = 0.05"
        )

        @dataclass
        class SlowVerifier(FakeSandbox):
            async def exec(self, command, **kwargs):
                if "test.sh" in command:
                    await asyncio.sleep(0.5)
                return await super().exec(command, **kwargs)

        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            SlowVerifier(),
            toml=toml,
            sandbox_ready_timeout_s=0.01,
            harness_setup_allowance_s=0,
            verifier_grace_s=0,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest

        response = await run_episode(server, task)

        assert response.failure.stage == "verification" and response.failure.failure_kind == "harbor:task_deadline"
        assert response.failure.partial_response.id == "response"
        assert sandbox.stopped and not server._sandboxes

    def retrying_server(self, tmp_path, monkeypatch, failures):
        """A server whose real `_create_sandbox` runs against a provider that fails `failures` starts in turn."""
        from nemo_gym.sandbox import SandboxCreateError

        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, image_pull_retries=2)
        monkeypatch.setattr(server, "_create_sandbox", HarborEnvironmentServer._create_sandbox.__get__(server))
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.resolve_provider_config", lambda ref, config: {"fake": {}}
        )
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.sandboxes._pull_retry_delay", lambda attempt: 0)
        starts: list[str] = []

        class FlakyProvider:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                starts.append(spec.image)
                if len(starts) <= len(failures):
                    raise SandboxCreateError(failures[len(starts) - 1])

            def __getattr__(self, name):
                return getattr(sandbox, name)

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", FlakyProvider)
        return server, task, sandbox, starts

    async def test_image_pull_failures_are_retried_in_the_server(self, tmp_path, monkeypatch):
        server, task, sandbox, starts = self.retrying_server(
            tmp_path, monkeypatch, ["pod failed: ErrImagePull", "create failed: ImagePullBackOff for ubuntu:24.04"]
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert starts == ["ubuntu:24.04"] * 3 and response.result.reward == 1.0

    async def test_exhausted_pull_retries_are_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        server, task, sandbox, starts = self.retrying_server(tmp_path, monkeypatch, ["ErrImagePull"] * 3)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "ErrImagePull" in response.failure.failure_reason and len(starts) == 3
        assert not server._sandboxes

    async def test_seed_time_is_not_charged_against_the_task_deadline(self, tmp_path, monkeypatch):
        """A start that recovered on a pull retry begins the task with its whole budget (review fix).

        The loop's clock is the test's: the seed jumps it by more than the whole task budget, so a deadline armed
        before the sandbox was ready would have expired, while the post-seed path keeps its full budget in real
        time. The second episode jumps the clock during the agent's turn instead and does hit the deadline.
        """
        from nemo_gym.sandbox import SandboxCreateError

        toml = TASK_TOML.replace("timeout_sec = 300.0", "timeout_sec = 10.0").replace(
            "timeout_sec = 120.0", "timeout_sec = 10.0"
        )
        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=toml,
            image_pull_retries=2,
            sandbox_ready_timeout_s=1,
            harness_setup_allowance_s=0,
            verifier_grace_s=0,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        assert server._post_seed_budget_s(task) == pytest.approx(20)
        assert server._task_budget_s(task) == pytest.approx(21)
        loop = asyncio.get_running_loop()
        real_time, offset = loop.time, 0.0
        monkeypatch.setattr(loop, "time", lambda: real_time() + offset)

        def advance(seconds: float) -> None:
            # asyncio.timeout and the loop's timers read loop.time(), so a jump expires every deadline it passes.
            nonlocal offset
            offset += seconds

        monkeypatch.setattr(server, "_create_sandbox", HarborEnvironmentServer._create_sandbox.__get__(server))
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.resolve_provider_config", lambda ref, config: {"fake": {}}
        )
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.sandboxes._pull_retry_delay", lambda attempt: 0)
        starts: list[str] = []

        class SlowFlakyProvider:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                starts.append(spec.image)
                if len(starts) == 1:
                    # The provider's own create retries: longer than the whole task budget before the pull
                    # failure surfaces (within the ready deadline, which bounds the whole create).
                    advance(server._task_budget_s(task) + 1)
                    raise SandboxCreateError("pod failed: ErrImagePull")
                advance(server.config.sandbox_ready_timeout_s)

            def __getattr__(self, name):
                return getattr(sandbox, name)

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", SlowFlakyProvider)

        response = await run_episode(server, task)

        assert starts == ["ubuntu:24.04"] * 2
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and sandbox.stopped and not server._sandboxes

        # The same jump after the sandbox is ready is charged: the deadline ends the agent's turn.
        async def overrun_the_turn():
            advance(server._post_seed_budget_s(task) + 1)
            await asyncio.sleep(1)  # Cancelled by the expired deadline at the loop's next pass, not slept.

        server.server_client.turn_hooks.append(overrun_the_turn)
        failure = (await run_episode(server, task, rollout="r2")).failure
        assert failure.stage == "agent" and failure.failure_kind == "harbor:task_deadline" and not failure.terminal

    async def test_other_start_failures_are_not_retried(self, tmp_path, monkeypatch):
        server, task, sandbox, starts = self.retrying_server(tmp_path, monkeypatch, ["quota exceeded for namespace"])
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert len(starts) == 1 and not server._sandboxes

    async def test_compose_group_pull_failures_are_retried_in_the_server(self, tmp_path, monkeypatch):
        from nemo_gym.sandbox import SandboxCreateError

        server, task, _, _ = make_server(tmp_path, monkeypatch, image_pull_retries=2)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.sandboxes._pull_retry_delay", lambda attempt: 0)
        groups: list = []
        verifier = FakeSandbox()

        class FlakyComposeGroup:
            def __init__(
                self, provider_config, path, *, service_specs, timeout_s, service_providers, platform, **kwargs
            ):
                groups.append(self)
                self.provider_config = provider_config
                self.services: dict = {}
                self.stopped = False

            async def start(self):
                if len(groups) == 1:
                    # The first group fails on a sidecar's pull, half-started: it is stopped whole.
                    self.services = {"main": HealthySandbox()}
                    raise SandboxCreateError("create failed: ImagePullBackOff for org/kafka:1")
                self.services = {"main": HealthySandbox(), "api": FakeSandbox(), "kafka": FakeSandbox()}

            async def stop(self):
                self.stopped = True

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FlakyComposeGroup)
        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert len(groups) == 2 and groups[0].stopped and groups[1].stopped
        assert response.result.reward == 1.0 and not server._sandboxes
        # Each attempt is a fresh group on the task's block, whose registry name the seed-time field check ran against.
        assert [g.provider_config for g in groups] == [{"opensandbox": {}}] * 2

    async def test_exhausted_compose_pull_retries_are_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        from nemo_gym.sandbox import SandboxCreateError

        server, task, _, _ = make_server(tmp_path, monkeypatch, image_pull_retries=1)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.sandboxes._pull_retry_delay", lambda attempt: 0)
        starts = []

        class FailingComposeGroup:
            services: dict = {}

            def __init__(
                self, provider_config, path, *, service_specs, timeout_s, service_providers, platform, **kwargs
            ):
                pass

            async def start(self):
                starts.append(1)
                raise SandboxCreateError("ErrImagePull")

            async def stop(self):
                pass

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FailingComposeGroup)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "ErrImagePull" in response.failure.failure_reason and len(starts) == 2
        assert not server._sandboxes

    def test_pull_failure_is_found_inside_an_exception_group(self):
        """A Compose group whose cleanup failed too raises an ExceptionGroup; the pull failure is still found."""
        from nemo_gym.sandbox import SandboxCreateError

        pull = SandboxCreateError("create failed: ImagePullBackOff for org/kafka:1")
        cleanup = RuntimeError("Shared volume cleanup failed: no space left")
        assert _is_image_pull_failure(ExceptionGroup("Compose cleanup failed", [cleanup, pull]))
        assert _is_image_pull_failure(ExceptionGroup("outer", [ExceptionGroup("inner", [pull])]))
        # The live shape: the group of cleanup errors is raised while the pull failure is being handled.
        try:
            try:
                raise pull
            except SandboxCreateError:
                raise ExceptionGroup("Compose cleanup failed", [cleanup])
        except ExceptionGroup as group:
            assert group.__context__ is pull and _is_image_pull_failure(group)
        assert not _is_image_pull_failure(ExceptionGroup("Compose cleanup failed", [cleanup]))
        assert not _is_image_pull_failure(ExceptionGroup("x", [RuntimeError("quota exceeded")]))

    async def test_compose_pull_failure_whose_cleanup_also_failed_is_retried(self, tmp_path, monkeypatch):
        from nemo_gym.sandbox import SandboxCreateError

        server, task, _, _ = make_server(tmp_path, monkeypatch, image_pull_retries=1)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.sandboxes._pull_retry_delay", lambda attempt: 0)
        groups: list = []
        verifier = FakeSandbox()

        class FlakyComposeGroup:
            def __init__(
                self, provider_config, path, *, service_specs, timeout_s, service_providers, platform, **kwargs
            ):
                groups.append(self)
                self.services: dict = {}
                self.stopped = 0

            async def start(self):
                if len(groups) == 1:
                    # The group's own stop() after the failed pull failed too: the cleanup errors surface as a
                    # group over the pull failure (docker_compose.py start/stop), not as the pull failure itself.
                    raise ExceptionGroup(
                        "Compose cleanup failed",
                        [RuntimeError("Shared volume cleanup failed"), SandboxCreateError("ErrImagePull")],
                    )
                self.services = {"main": HealthySandbox(), "api": FakeSandbox(), "kafka": FakeSandbox()}

            async def stop(self):
                self.stopped += 1

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FlakyComposeGroup)
        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert len(groups) == 2 and [g.stopped for g in groups] == [1, 1]
        assert response.result.reward == 1.0 and not server._sandboxes

    def verifier_retrying_server(self, tmp_path, monkeypatch, failures):
        """A separate-mode server whose real `_create_verifier_sandbox` runs against a provider that fails
        `failures` verifier starts in turn; the agent's sandbox is the usual fake."""
        from nemo_gym.sandbox import SandboxCreateError

        agent = AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, image_pull_retries=2)
        (task.path / "task.toml").write_text(SEPARATE_TOML)
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.resolve_provider_config", lambda ref, config: {"fake": {}}
        )
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.sandboxes._pull_retry_delay", lambda attempt: 0)
        verifier = FakeSandbox()
        starts: list[str] = []
        gate_values: list[int] = []
        providers: list = []

        class FlakyProvider:
            def __init__(self, provider_config):
                providers.append(self)
                self.stopped = 0

            async def start(self, spec):
                starts.append(spec.image)
                gate_values.append(server._pull_retry_gate._value)
                # Owned under the verifier key from before start(), so a half-started one is stopped whole.
                assert server._sandboxes.get(f"{next(iter(server._sessions))}/verifier") is self
                if len(starts) <= len(failures):
                    raise SandboxCreateError(failures[len(starts) - 1])

            async def stop(self):
                self.stopped += 1
                verifier.stopped = True

            def __getattr__(self, name):
                return getattr(verifier, name)

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", FlakyProvider)
        return server, task, starts, gate_values, providers

    async def test_verifier_sandbox_pull_failures_are_retried_in_the_server(self, tmp_path, monkeypatch):
        server, task, starts, gates, providers = self.verifier_retrying_server(
            tmp_path, monkeypatch, ["pod failed: ErrImagePull"]
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        # The first start is not gated; the retry queues behind the per-server gate.
        assert starts == ["org/verifier:1"] * 2 and gates == [4, 3]
        assert [p.stopped for p in providers] == [1, 1]
        assert not server._sandboxes and server._pull_retry_gate._value == 4

    async def test_exhausted_verifier_pull_retries_mask_and_leave_nothing_owned(self, tmp_path, monkeypatch):
        server, task, starts, gates, providers = self.verifier_retrying_server(
            tmp_path, monkeypatch, ["ErrImagePull"] * 3
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        result = response.result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert "ErrImagePull" in result.failure_reason
        assert len(starts) == 3 and gates == [4, 3, 3]
        # The last provider (not stopped by the helper) is released by the episode's own close.
        assert [p.stopped for p in providers] == [1, 1, 1]
        assert not server._sandboxes and server._pull_retry_gate._value == 4

    async def test_verifier_sandbox_other_start_failures_are_not_retried(self, tmp_path, monkeypatch):
        server, task, starts, _, _ = self.verifier_retrying_server(
            tmp_path, monkeypatch, ["quota exceeded for namespace"]
        )
        response = await run_episode(server, task)
        result = response.result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert len(starts) == 1 and not server._sandboxes and server._pull_retry_gate._value == 4

    def test_nested_compose_task_reads_image_configs_from_the_dataset_root(self, tmp_path, monkeypatch):
        """prepare records compose-images.json at the dataset root; the snapshot of a nested task finds it there."""
        server, _, _, _ = make_server(tmp_path, monkeypatch)
        root = tmp_path / "datasets" / "ds"
        web = write_task(root / "00-services" / "web", toml=COMPOSE_TOML)
        (web / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (root / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(web, root=root)
        assert task.needs_compose

        snapshot = server._snapshot_task(web, task.digest, root)

        assert snapshot.path.parent != web.parent
        assert server._compose_image_configs(snapshot) == IMAGE_CONFIGS

    async def test_turn_cut_off_inside_reasoning_is_verified_not_masked(self, tmp_path, monkeypatch):
        """The live Terminus shape after a reasoning-only max_output_tokens response: exported and graded."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        turn = agent_response().model_dump(mode="json")
        answer = turn["output"][0]

        def reasoning(item_id, text):
            return {"type": "reasoning", "id": item_id, "summary": [{"type": "summary_text", "text": text}]}

        def observed(text):
            return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}

        turn["output"] = [
            reasoning("r1", "plan"),
            answer,
            observed("hello.txt created"),
            reasoning("r2", "half a thou"),
            observed("(no command)"),
            reasoning("r3", "finish"),
            {**answer, "id": "message-2", "content": [{**answer["content"][0], "text": "Done."}]},
        ]
        server.server_client.turn_response = turn

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        result = response.result
        assert result.mask_sample is False and result.failure_kind is None and result.reward == 1.0
        assert any("test.sh" in call["command"] for call in sandbox.execs)
        (trajectory_bytes,) = uploaded_file(sandbox, "trajectory.json")
        trajectory = json.loads(trajectory_bytes)
        steps = trajectory["steps"]
        assert [step["source"] for step in steps] == ["user", "agent", "agent", "agent"]
        cut = steps[2]
        assert cut["message"] == "" and cut["reasoning_content"] == "half a thou"
        assert cut["extra"]["nemo_gym"] == {"incomplete": "max_output_tokens", "observation_source": "user_turn"}
        assert cut["observation"]["results"][0]["content"] == "(no command)"
        assert trajectory["extra"]["nemo_gym"]["conversion"]["reasons"] == ["trailing_reasoning"]

    def test_runs_can_force_separate_mode_for_tasks_with_artifacts(self, tmp_path, monkeypatch):
        with_artifacts = TASK_TOML.replace("[verifier]\n", 'artifacts = ["/app/out.txt"]\n\n[verifier]\n', 1)
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, toml=with_artifacts, separate_verifier_for_artifacts=True
        )
        task = load_task(task.path)
        assert task.config.is_shared_verifier
        assert server._separate_verifier(task)
        # Module helpers follow the forced mode: the verifier runs in the task's own image.
        assert _verifier_image(task, True) == task.image == "ubuntu:24.04" and _verifier_image(task) is None
        # The budget and the sandbox lifetime count the second sandbox start.
        plain_server, plain_task, _, _ = make_server(tmp_path / "plain", monkeypatch, toml=with_artifacts)
        assert (
            server._task_budget_s(task)
            == plain_server._task_budget_s(plain_task) + server.config.sandbox_ready_timeout_s
        )
        # A task without artifacts keeps Harbor's shared mode.
        _, bare, _, _ = make_server(tmp_path / "bare", monkeypatch, separate_verifier_for_artifacts=True)
        assert not server._separate_verifier(bare)

    async def test_forced_separate_mode_verifies_in_a_fresh_sandbox(self, tmp_path, monkeypatch):
        agent = AgentSandbox()
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            agent,
            toml=TASK_TOML.replace("[verifier]\n", 'artifacts = ["/app/output/report.json"]\n\n[verifier]\n', 1),
            separate_verifier_for_artifacts=True,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        verifier = FakeSandbox()

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.verifier_mode == "separate" and response.result.reward == 1.0
        assert any("test.sh" in call["command"] for call in verifier.execs)
        assert not any("test.sh" in call["command"] for call in agent.execs)
        assert len(uploaded_file(verifier, "trajectory.json")) == 1
        assert agent.stopped and verifier.stopped and not server._sandboxes

    async def test_nested_tasks_resolve_through_the_mapping_path_and_the_dataset_root(self, tmp_path, monkeypatch):
        server, _, _, _ = make_server(tmp_path, monkeypatch)
        root = tmp_path / "datasets" / "ds"
        (root / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        deep = write_task(root / "group" / "deep")
        # The base-image record sits at the dataset root, where prepare writes it, not beside the nested task.
        (root / "group" / IMAGE_CONFIGS_FILE).rename(tmp_path / "unused.json")
        task = load_task(deep, root=root)
        assert task.config.agent.timeout_sec == 600.0
        server.config.tasksets["ds"].tasks["deep"] = task.digest
        server.config.tasksets["ds"].paths["deep"] = "group/deep"

        response = await run_episode(server, task, task_id="deep")

        assert response.failure is None, response.failure
        assert server.server_client.seed_bodies()[0]["agent_timeout_sec"] == 600.0
        # A mapping path that escapes the taskset folder is refused before anything is provisioned.
        server.config.tasksets["ds"].paths["deep"] = "../deep"
        escaped = await run_episode(server, task, task_id="deep", rollout="r2")
        assert (
            escaped.failure.stage == "seed"
            and escaped.failure.terminal
            and "invalid path" in escaped.failure.failure_reason
        )


class TestVerifierNetwork:
    """``[verifier].network_mode`` is the separate verifier sandbox's egress policy, applied when it is created.

    OpenSandbox cannot switch a running sandbox from public to restricted (its egress API only patches rules
    under the ``defaultAction`` the sandbox was created with), so the agent's sandbox stays as it is and the
    phase policy shapes the verifier's own sandbox; any provider that cannot enforce it is refused at seed.
    """

    NO_NETWORK_TOML = SEPARATE_TOML.replace("[verifier]\n", '[verifier]\nnetwork_mode = "no-network"\n')
    ALLOWLIST_TOML = SEPARATE_TOML.replace(
        "[verifier]\n",
        '[verifier]\nnetwork_mode = "allowlist"\nallowed_hosts = ["PyPI.org", "*.nvidia.com", "pypi.org", "10.0.0.7"]\n',
    )

    def prepared(self, tmp_path, monkeypatch, *, toml, provider="opensandbox", **config):
        agent = AgentSandbox()
        server, task, _, created = make_server(tmp_path, monkeypatch, agent, **config)
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {provider: {}}}
        )
        verifier = FakeSandbox()
        specs: list[SandboxSpec] = []

        async def start_with_pull_retries(task, *, key, what, make, start):
            # The real `_create_verifier_sandbox` built the spec; it reaches the provider through `start`.
            class Starter:
                async def start(self, spec):
                    specs.append(spec)

            assert agent.stopped, "the agent sandbox must be stopped before the verifier is created"
            await start(Starter())
            server._sandboxes[key] = verifier
            return verifier

        monkeypatch.setattr(server, "_start_with_pull_retries", start_with_pull_retries)
        return server, task, agent, verifier, specs, created

    async def test_no_network_verifier_is_created_under_a_deny_all_policy(self, tmp_path, monkeypatch):
        server, task, agent, verifier, specs, _ = self.prepared(tmp_path, monkeypatch, toml=self.NO_NETWORK_TOML)
        response = await run_episode(server, task)
        result = response.result
        assert result is not None, response.failure
        assert (result.reward, result.verifier_mode, result.verifier_network) == (1.0, "separate", "no-network")
        (spec,) = specs
        assert spec.image == "org/verifier:1"
        assert spec.provider_options == {"network_policy": {"defaultAction": "deny", "egress": []}}
        # The agent's own sandbox is created as before: public, with the deployment's options only.
        assert "network_policy" not in server._sandbox_spec(task, "/app").provider_options
        assert verifier.stopped and agent.stopped

    async def test_allowlist_hosts_become_allow_rules_in_order_without_repeats(self, tmp_path, monkeypatch):
        server, task, _, _, specs, _ = self.prepared(tmp_path, monkeypatch, toml=self.ALLOWLIST_TOML)
        response = await run_episode(server, task)
        assert response.result is not None, response.failure
        assert response.result.verifier_network == "allowlist"
        assert specs[0].provider_options["network_policy"] == {
            "defaultAction": "deny",
            "egress": [
                {"action": "allow", "target": "pypi.org"},
                {"action": "allow", "target": "*.nvidia.com"},
                {"action": "allow", "target": "10.0.0.7"},
            ],
        }

    @pytest.mark.parametrize(
        "deny",
        [
            {"defaultAction": "deny", "egress": [{"action": "allow", "target": "mirror.internal"}]},
            # OpenSandbox reads a policy that names no defaultAction as deny.
            {"egress": [{"action": "allow", "target": "mirror.internal"}]},
            # Even a deployment that already allows every host the task names: the rule has no host matching.
            {
                "defaultAction": "deny",
                "egress": [{"action": "allow", "target": t} for t in ("pypi.org", "*.nvidia.com", "10.0.0.7")],
            },
        ],
        ids=["explicit-deny", "implicit-deny", "hosts-already-allowed"],
    )
    async def test_an_allowlist_cannot_widen_a_deny_by_default_deployment(self, tmp_path, monkeypatch, deny):
        """The task's policy replaces the deployment's for the verifier sandbox, so under a deny-by-default
        deployment a task allowlist would open hosts the operator closed; refused at seed, before any sandbox,
        naming the hosts and the one remedy (an allow-by-default deployment), rather than intersected: there is
        no host matching to get subtly wrong, so a deployment that happens to allow the same hosts is refused too."""
        server, task, _, _, specs, created = self.prepared(
            tmp_path, monkeypatch, toml=self.ALLOWLIST_TOML, sandbox_provider_options={"network_policy": deny}
        )
        response = await run_episode(server, task)
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert "[verifier] network policy" in failure.failure_reason
        assert "allowed_hosts pypi.org, *.nvidia.com, 10.0.0.7 would widen this deployment's deny-by-default" in (
            failure.failure_reason
        )
        assert "run the task on an allow-by-default deployment" in failure.failure_reason
        assert created == [] and specs == []
        # The deployment's options are not mutated.
        assert server.config.sandbox_provider_options["network_policy"] == deny

    async def test_an_empty_allowlist_is_no_network_and_passes_under_a_deny_by_default_deployment(
        self, tmp_path, monkeypatch
    ):
        toml = self.ALLOWLIST_TOML.replace('["PyPI.org", "*.nvidia.com", "pypi.org", "10.0.0.7"]', "[]")
        assert "allowed_hosts = []" in toml, toml
        deny = {"defaultAction": "deny", "egress": []}
        server, task, _, _, specs, _ = self.prepared(
            tmp_path, monkeypatch, toml=toml, sandbox_provider_options={"network_policy": deny}
        )
        response = await run_episode(server, task)
        assert response.result is not None, response.failure
        assert specs[0].provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}

    async def test_an_empty_network_policy_is_deny_all_and_refuses_an_allowlist_but_not_no_network(
        self, tmp_path, monkeypatch
    ):
        """Review (round 36): `network_policy: {}` is deny-all in the 0.1.15 SDK (no `defaultAction` reads as deny,
        no rules), so it is a present deny-by-default policy, not an absent one: a task allowlist is refused with the
        same message, and `no-network` only narrows it and passes."""
        server, task, _, _, specs, created = self.prepared(
            tmp_path, monkeypatch, toml=self.ALLOWLIST_TOML, sandbox_provider_options={"network_policy": {}}
        )
        response = await run_episode(server, task)
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert "allowed_hosts pypi.org, *.nvidia.com, 10.0.0.7 would widen this deployment's deny-by-default" in (
            failure.failure_reason
        )
        assert created == [] and specs == []
        assert server.config.sandbox_provider_options["network_policy"] == {}
        server, task, _, _, specs, _ = self.prepared(
            tmp_path / "no-network",
            monkeypatch,
            toml=self.NO_NETWORK_TOML,
            sandbox_provider_options={"network_policy": {}},
        )
        response = await run_episode(server, task)
        assert response.result is not None, response.failure
        assert response.result.verifier_network == "no-network"
        assert specs[0].provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}

    async def test_a_network_policy_that_is_not_a_mapping_is_refused_at_seed(self, tmp_path, monkeypatch):
        server, task, _, _, specs, created = self.prepared(
            tmp_path, monkeypatch, toml=self.ALLOWLIST_TOML, sandbox_provider_options={"network_policy": "deny"}
        )
        response = await run_episode(server, task)
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert "sandbox_provider_options.network_policy must be a mapping, got str" in failure.failure_reason
        assert created == [] and specs == []

    async def test_no_network_only_narrows_a_deny_by_default_deployment_and_passes(self, tmp_path, monkeypatch):
        deny = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "mirror.internal"}]}
        server, task, _, _, specs, _ = self.prepared(
            tmp_path, monkeypatch, toml=self.NO_NETWORK_TOML, sandbox_provider_options={"network_policy": deny}
        )
        response = await run_episode(server, task)
        assert response.result is not None, response.failure
        assert response.result.verifier_network == "no-network"
        assert specs[0].provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}

    async def test_the_phase_policy_replaces_the_deployments_for_the_verifier_only(self, tmp_path, monkeypatch):
        """Under an allow-by-default deployment (or none) the task's policy replaces the deployment's for the
        verifier sandbox alone; a deny-by-default deployment admits only a narrowing policy (the tests above)."""
        baseline = {"defaultAction": "allow", "egress": [{"action": "deny", "target": "evil.example"}]}
        server, task, _, _, specs, _ = self.prepared(
            tmp_path, monkeypatch, toml=self.NO_NETWORK_TOML, sandbox_provider_options={"network_policy": baseline}
        )
        assert (await run_episode(server, task)).result is not None
        assert server._sandbox_spec(task, "/app").provider_options["network_policy"] == baseline
        assert specs[0].provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}
        # The deployment's options are not mutated.
        assert server.config.sandbox_provider_options["network_policy"] == baseline

    @pytest.mark.parametrize("provider", ["docker", "daytona"])
    async def test_providers_without_an_egress_policy_are_refused_at_seed(self, tmp_path, monkeypatch, provider):
        server, task, _, _, specs, created = self.prepared(
            tmp_path, monkeypatch, toml=self.NO_NETWORK_TOML, provider=provider
        )
        response = await run_episode(server, task)
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert "[verifier] network policy" in failure.failure_reason
        assert "requires the OpenSandbox network-policy provider" in failure.failure_reason
        # Refused before anything was provisioned: no agent sandbox, no verifier.
        assert created == [] and specs == []

    async def test_hosts_the_policy_cannot_express_are_refused_at_seed(self, tmp_path, monkeypatch):
        toml = self.ALLOWLIST_TOML.replace('"10.0.0.7"', '"10.0.0.0/8"')
        server, task, _, _, _, created = self.prepared(tmp_path, monkeypatch, toml=toml)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal and created == []
        assert "CIDR ranges are not supported" in response.failure.failure_reason

    async def test_a_restricted_verifier_in_the_agents_sandbox_is_refused_at_seed(self, tmp_path, monkeypatch):
        # The loader admits the policy for separate-mode tasks only; the seed keeps the invariant for a task
        # object that did not come from it (here: separate mode switched off under the server).
        server, task, _, _, _, created = self.prepared(tmp_path, monkeypatch, toml=self.NO_NETWORK_TOML)
        monkeypatch.setattr(server, "_separate_verifier", lambda task: False)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal and created == []
        assert "network policy cannot be changed after it started" in response.failure.failure_reason

    def test_public_verifier_phase_adds_no_policy(self, tmp_path, monkeypatch):
        server, task, _, _, _, _ = self.prepared(tmp_path, monkeypatch, toml=SEPARATE_TOML)
        assert server._verifier_network_policy(task) is None
        spec = server._sandbox_spec(task, None, image="org/verifier:1", role="verifier", network_policy=None)
        assert "network_policy" not in spec.provider_options
