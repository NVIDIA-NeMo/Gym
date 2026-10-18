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
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from yarl import URL

import nemo_gym.sandbox.providers.base as sandbox_base
from environment_servers.harbor.app import (
    HarborEnvironmentServerConfig,
)
from environment_servers.harbor.sandboxes import SEED_CREATE_GRACE_S
from environment_servers.harbor.tests.conftest import (
    COMPOSE_YAML,
    IMAGE_CONFIGS,
    IMAGE_PATH,
    OK,
    SEPARATE_IN_TASK_IMAGE_TOML,
    SEPARATE_TOML,
    SOLVE_SH,
    TASK_TOML,
    AgentSandbox,
    FakeCompose,
    FakeSandbox,
    _fails,
    archive_of,
    episode,
    make_server,
    run_episode,
    write_image_config,
)
from environment_servers.harbor.verifier import (
    EXIT_RECORD_NAME,
    TIMEOUT_WRAPPER_GUARD,
    _verifier_image,
    _verifier_runs_task_image,
    parse_exit_record,
    parse_reward_file,
    select_reward,
    verifier_command,
)
from nemo_gym.base_environment_server import BaseEnvironmentServerConfig
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS, cpu_cap_env
from nemo_gym.server_utils import ServerClient
from nemo_gym.tasks.harbor import load_task
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.models import HarborEnvironment
from nemo_gym.tasks.harbor.records import IMAGE_CONFIGS_FILE


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
        assert spec.ttl_s == 21600 + 10 + server.config.sandbox_ttl_slack_s
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
        # tests/, /logs/artifacts and the report all travel as archives, so the report keeps its mode bits.
        assert len(verifier.uploads) == 3 and all(remote.endswith(".tar.gz") for _, remote in verifier.uploads)
        unpacked = [c["command"] for c in verifier.execs if "tar -xzf" in c["command"]]
        assert any("-C /logs/artifacts;" in c for c in unpacked) and any("-C /app/output;" in c for c in unpacked)
        # The two artifacts reach the verifier as the very archives the agent sandbox packed (the modes the agent
        # left, not the host staging copy's widened ones), extracted there by root tar; the staged trees stay as
        # diagnostics and the archives are gone once restored.
        assert agent.archives["/logs/artifacts"] in verifier.uploaded_archives
        assert agent.archives["/app/output"] in verifier.uploaded_archives
        assert sorted(p.name for p in artifacts.iterdir()) == ["00", "01"]
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

    async def test_seed_rejects_a_verifier_that_must_be_built(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        toml = SEPARATE_TOML.replace('docker_image = "org/verifier:1"\n', "")
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "tests/Dockerfile" in response.failure.failure_reason and created == []

    async def test_verifier_failure_masks_and_stops_both_sandboxes(self, tmp_path, monkeypatch):
        class Broken(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                raise RuntimeError("verifier sandbox lost")

        broken = Broken()
        server, task, agent, verifier, _ = self.prepared(tmp_path, monkeypatch, verifier=broken)
        result = (await run_episode(server, task)).result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert agent.stopped and verifier.stopped

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
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        services = documents[0]["services"]
        assert services["main"]["image"] == IMAGE_CONFIGS["org/agent:1"]["image"]
        # The server resolves the task from its snapshot, so the bind mount points into that copy.
        (volume,) = services["main"]["volumes"]
        assert volume.endswith("/hello/environment/seed.json:/app/seed.json:ro")
        assert volume.startswith(server._snapshots.name) and not volume.startswith(str(task.path))
        assert services["api"]["environment"] == {"API_MODE": "replay"} and "env_file" not in services["api"]
        assert services["kafka"]["environment"] == {"KAFKA_TOPIC": "from-the-shell"}

    MAIN_BUILD_SHAPES = {
        "args-and-target": "    build:\n      context: ${CONTEXT_DIR}\n      target: prod\n      args:\n        PYTHON_VERSION: '3.12'\n",
        "dockerfile_inline": "    build:\n      context: ${CONTEXT_DIR}\n      dockerfile_inline: |\n        FROM python:3.12\n        RUN pip install x\n",
    }

    @pytest.mark.parametrize("main_build", list(MAIN_BUILD_SHAPES.values()), ids=list(MAIN_BUILD_SHAPES))
    def test_a_built_tasks_main_build_section_is_dropped_at_seed(self, tmp_path, monkeypatch, main_build):
        """Prepare builds ``main`` from the task's Compose file (its build args, target or dockerfile_inline apply
        there) and records the image; at seed the recorded image stands in, so the ``build:`` and ``pull_policy``
        keys that only describe that build are dropped before the Compose adapter, which starts images only."""
        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY . /app\n")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    pull_policy: build\n"
            + main_build
            + "  api:\n    image: org/api:1\n"
        )
        built = "registry.example/ds/hb__" + "0" * 32 + "@sha256:" + "d" * 64
        (task.path.parent / BUILT_IMAGES_FILE).write_text(
            json.dumps({context_digest(task.path / "environment"): {"image": built}})
        )
        records = json.loads((task.path.parent / IMAGE_CONFIGS_FILE).read_text())
        records[built] = {
            "os": "linux",
            "architecture": "amd64",
            "image": built,
            "config": {"WorkingDir": "/app"},
            "record_version": RECORD_VERSION,
        }
        records["org/api:1"] = IMAGE_CONFIGS["org/api:1"]
        (task.path.parent / IMAGE_CONFIGS_FILE).write_text(json.dumps(records))
        task = load_task(task.path)
        assert task.built_image == built == task.image and task.workdir == "/app"
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        document = server._compose_document(task)

        main = document["services"]["main"]
        assert main["image"] == built and main["working_dir"] == "/app"
        assert "build" not in main and "pull_policy" not in main
        assert document["services"]["api"]["image"] == IMAGE_CONFIGS["org/api:1"]["image"]
        # The adapter's own validation accepts the document: the fields that failed every episode are gone.
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml")
        group.document = document
        assert set(group._validate()) == {"main", "api"}

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
        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.prepare_builds import unsupported_compose_fields_error

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
        assert unsupported_compose_fields_error(task) is None

    def test_a_main_build_key_gym_cannot_apply_without_a_build_is_a_task_error(self, tmp_path, monkeypatch):
        """A ``target`` (or ``args``, ``dockerfile_inline``, another ``dockerfile`` or ``context``) would change the image
        Harbor builds; without a build Gym cannot honour it, so the task is an error naming the key, as prepare already
        reported it into ``skipped.json``, instead of starting on the wrong image."""
        from nemo_gym.tasks.harbor.errors import HarborTaskError
        from nemo_gym.tasks.harbor.prepare_builds import unsupported_compose_fields_error

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
        assert str(unsupported_compose_fields_error(task)) == f"docker-compose.yaml: {expected}"

    def test_pull_policy_on_any_service_is_dropped_so_the_document_starts(self, tmp_path, monkeypatch):
        """``pull_policy`` says when ``docker compose`` pulls an image before starting it; the sandbox has no pull
        step to configure, so the key is dropped from every service in every image mode (here pull mode, where
        ``main`` keeps everything else) instead of the adapter refusing it. Prepare's check sees the same document."""
        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.prepare_builds import unsupported_compose_fields_error

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    pull_policy: missing\n"
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

        assert all("pull_policy" not in service for service in document["services"].values())
        assert document["services"]["api"]["expose"] == ["8080"]
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml")
        group.document = document
        assert set(group._validate()) == {"main", "api"}
        assert unsupported_compose_fields_error(task) is None

    @staticmethod
    def capture_compose_specs(monkeypatch) -> dict:
        captured = {}

        class FakeComposeGroup:
            def __init__(self, provider_config, path, *, service_specs, timeout_s):
                captured["specs"] = service_specs

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FakeComposeGroup)
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        return captured

    def test_main_env_layers_image_env_under_the_compose_file_under_the_host_resolved_task_env(
        self, tmp_path, monkeypatch
    ):
        """Harbor's order for ``main``: image ENV (here the pull-mode Dockerfile's ``ENV``, which the loader folds into
        the task) < the task's Compose file < the host-resolved ``[environment].env`` (its env override file). A key
        the file sets beats the Dockerfile's, a ``${FOO:-}`` that interpolated to empty at prepare does not hide the
        host value, and a sidecar keeps its file's environment."""
        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose

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
        from nemo_gym.sandbox.providers.base import SandboxHandle

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
            compose = await server._create_compose(task, "r1")
            main = compose.services["main"]
            bootstrap_users = {e["user"] for e in provider.execs}
            assert provider.created[0].user is None and main.default_user is None
            assert "agent" not in bootstrap_users, provider.execs
            names = await server._apply_overlay(main, task)
            await server._set_default_user(main, task, names)
            assert any("useradd" in e["command"] for e in provider.execs)
            assert main.default_user == "agent"
            # From here every exec that names no user runs as the task's user.
            await main.exec("id")
            assert provider.execs[-1]["user"] == "agent"

        asyncio.run(run())

    def test_sidecar_specs_do_not_inherit_the_task_user(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER agent\n")
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        assert task.user == "agent"
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        captured = {}

        class FakeComposeGroup:
            def __init__(self, provider_config, path, *, service_specs, timeout_s):
                captured["specs"] = service_specs

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", FakeComposeGroup)
        server.config.artifacts_dir = tmp_path / "artifacts"
        asyncio.run(server._create_compose(task, "r1"))
        specs = captured["specs"]
        assert specs["main"].user == "agent"
        assert {name: spec.user for name, spec in specs.items() if name != "main"} == {"api": None, "kafka": None}

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
        # The upload takes the folder as is here; the digest's ignore rule joins the folder uploads with the
        # Compose volumes work, which filters a stray __pycache__ out of every task upload.
        assert names == {"solve.sh", "data", "data/answer.txt", "__pycache__", "__pycache__/x.pyc"}
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
        "task_mcp_grants",
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
