# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import io
import tarfile
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from environment_servers.harbor.app import HarborEnvironmentServer
from environment_servers.harbor.session import _task_sandbox_env, resolve_env_templates
from environment_servers.harbor.task_data import TaskData
from environment_servers.harbor.tests import conftest
from environment_servers.harbor.tests.conftest import (
    SEPARATE_IN_TASK_IMAGE_TOML,
    SEPARATE_TOML,
    TASK_TOML,
    AgentSandbox,
    FakeSandbox,
    episode,
    make_server,
    run_episode,
)
from nemo_gym.sandbox.transfer import SandboxTransferError
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.tasks.harbor import DIGEST_KEY, load_task
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.materialize import materialize_task


def refresh_task(server, task):
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    return task


def during_turn(server, hook):
    """Run ``hook`` when the harness takes its turn, between the seed and the verification of one episode."""
    server.server_client.turn_hooks.append(hook)


@pytest.mark.parametrize("section", ["environment", "verifier"])
async def test_missing_env_fails_before_provisioning(tmp_path, monkeypatch, section):
    monkeypatch.delenv("HARBOR_TEST_REQUIRED", raising=False)
    toml = TASK_TOML + f'\n[{section}.env]\nTOKEN = "${{HARBOR_TEST_REQUIRED}}"\n'
    server, task, _, created = make_server(tmp_path, monkeypatch, toml=toml)
    response = await run_episode(server, task)
    assert response.failure.stage == "seed" and response.failure.terminal
    assert "HARBOR_TEST_REQUIRED" in response.failure.failure_reason
    assert created == []


async def test_templates_resolve_at_use_without_changing_materialized_task(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "agent-secret")
    monkeypatch.setenv("HARBOR_TEST_EMPTY", "")
    monkeypatch.delenv("HARBOR_TEST_UNSET", raising=False)
    template = {
        "token": "${HARBOR_TEST_TOKEN}",
        "empty": "${HARBOR_TEST_EMPTY:-fallback}",
        "default": "${HARBOR_TEST_UNSET:-fallback}",
        "literal": "prefix-${HARBOR_TEST_TOKEN}",
    }
    assert resolve_env_templates(template) == {
        "token": "agent-secret",
        "empty": "",
        "default": "fallback",
        "literal": "prefix-${HARBOR_TEST_TOKEN}",
    }
    toml = (
        TASK_TOML
        + '\n[environment.env]\nTOKEN = "${HARBOR_TEST_TOKEN}"\n[verifier.env]\nTOKEN = "${HARBOR_TEST_TOKEN}"\n'
    )
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml)
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    assert server._sandbox_spec(task, "/app").env == cpu_cap_env(1) | {"TOKEN": "agent-secret"}
    # The verifier resolves the template when it runs, not when the sandbox was created.
    during_turn(server, lambda: monkeypatch.setenv("HARBOR_TEST_TOKEN", "verifier-secret"))
    snapshots: list[Path] = []
    during_turn(server, lambda: snapshots.append(server._sessions["r1"].task.path))
    response = await run_episode(server, task)
    assert response.result.reward == 1
    assert "agent-secret" not in response.model_dump_json() and "verifier-secret" not in response.model_dump_json()
    run = next(call for call in sandbox.execs if "test.sh" in call["command"] and "timeout" in call["command"])
    assert run["env"] == {"TOKEN": "verifier-secret"}
    assert task.env["TOKEN"] == "${HARBOR_TEST_TOKEN}"
    assert "agent-secret" not in materialize_task(task, "ds").model_dump_json()
    assert "verifier-secret" not in (snapshots[0] / "task.toml").read_text()


@pytest.mark.parametrize("probe", [None, "host-value"])
async def test_dockerfile_env_literal_is_not_a_host_template(tmp_path, monkeypatch, probe):
    """A Dockerfile ``ENV`` value is baked, as ``docker build`` leaves it; only ``[environment.env]`` is a host template."""
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "agent-secret")
    if probe is None:
        # No such host variable: the seed must not refuse the task for a value that was never a template.
        monkeypatch.delenv("GYM4256_LITERAL_PROBE", raising=False)
    else:
        monkeypatch.setenv("GYM4256_LITERAL_PROBE", probe)
    dockerfile = "FROM debian:bookworm-slim\nENV LITERAL='${GYM4256_LITERAL_PROBE}'\nRUN echo probe\n"
    toml = TASK_TOML + '\n[environment.env]\nTOKEN = "${HARBOR_TEST_TOKEN}"\n'
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml, dockerfile=dockerfile)
    # The loader keeps the single-quoted value literal and stores the task.toml template unresolved.
    assert task.overlay and task.env["LITERAL"] == "${GYM4256_LITERAL_PROBE}"
    assert task.env["TOKEN"] == "${HARBOR_TEST_TOKEN}"
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    specs = []

    async def create(task, workdir, *, episode_key=None):
        # The sandbox starts with the spec's environment, as the provider's container would.
        specs.append(server._sandbox_spec(task, workdir))
        sandbox.spec_env = dict(specs[-1].env)
        sandbox.created_in(workdir)
        return sandbox

    monkeypatch.setattr(server, "_create_sandbox", create)
    response = await run_episode(server, task)
    assert response.failure is None and response.result.reward == 1
    (spec,) = specs
    # The agent's sandbox gets the Dockerfile value as written and the task.toml template resolved on the host.
    assert spec.env["LITERAL"] == "${GYM4256_LITERAL_PROBE}" and spec.env["TOKEN"] == "agent-secret"
    assert _task_sandbox_env(task) == {**task.env, "TOKEN": "agent-secret"}
    assert spec.env == cpu_cap_env(1) | _task_sandbox_env(task)
    # The overlay RUN line sees the same literal, as a build step would.
    assert sandbox.run("echo probe")["seen"]["env"]["LITERAL"] == "${GYM4256_LITERAL_PROBE}"
    assert "host-value" not in response.model_dump_json()


async def test_live_folder_edits_do_not_change_seeded_verifier(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    original = (task.path / "tests" / "test.sh").read_bytes()
    assert (await run_episode(server, task)).result.reward == 1
    (task.path / "tests" / "test.sh").write_text("tampered verifier")
    snapshots: list[Path] = []
    during_turn(server, lambda: snapshots.append(server._sessions["r2"].task.path))
    assert (await run_episode(server, task, rollout="r2")).result.reward == 1
    # Every upload is an archive (the trajectory travels as a one-file archive too): the last one holding
    # test.sh is the second episode's tests/.
    scripts = []
    for data in sandbox.uploaded_archives:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            scripts += [
                archive.extractfile(name).read() for name in archive.getnames() if name.lstrip("./") == "test.sh"
            ]
    assert scripts and scripts[-1] == original
    # Both episodes ran from the one snapshot taken at the first seed.
    assert len(server._tasks) == 1 and snapshots[0] == server._tasks[next(iter(server._tasks))].result().path


async def test_concurrent_first_seeds_share_snapshot_even_if_one_is_cancelled(tmp_path, monkeypatch):
    server, task, _, created = make_server(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    copies = []
    original = server._snapshot_task

    def snapshot(folder, expected, root=None):
        copies.append(folder)
        entered.set()
        assert release.wait(10)
        return original(folder, expected, root)

    monkeypatch.setattr(server, "_snapshot_task", snapshot)
    first = asyncio.create_task(server.run_request(episode(task)))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(server.run_request(episode(task, rollout="r2")))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    finally:
        release.set()
    assert (await second).result.reward == 1
    assert copies == [task.path] and len(created) == 1
    await server.shutdown()
    assert not server._tasks and not server._sandboxes


@pytest.mark.parametrize("stage", ["start", "prepare", "serialize"])
async def test_cancelled_episode_stops_owned_sandbox(tmp_path, monkeypatch, stage):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    entered = asyncio.Event()

    async def block(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    if stage == "start":
        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", lambda config: sandbox)
        monkeypatch.setattr("environment_servers.harbor.sandboxes.resolve_provider_config", lambda *args: {})
        monkeypatch.setattr("environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {})
        monkeypatch.setattr(server, "_sandbox_spec", lambda *args: None)
        monkeypatch.setattr(sandbox, "start", block, raising=False)
        monkeypatch.setattr(server, "_create_sandbox", HarborEnvironmentServer._create_sandbox.__get__(server))
    elif stage == "prepare":
        monkeypatch.setattr(server, "_prepare_workdir", block)
    else:
        monkeypatch.setattr(sandbox, "serialize", block)
    running = asyncio.create_task(server.run_request(episode(task)))
    await asyncio.wait_for(entered.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert sandbox.stopped and not server._sandboxes and not server._sessions
    assert server.server_client.calls == []
    await server.shutdown()


@pytest.mark.parametrize("retry", ["final cleanup", "shutdown"])
async def test_failed_stop_keeps_ownership_until_a_stop_succeeds(tmp_path, monkeypatch, retry, caplog):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    attempts = []
    # The seed's own stop fails; so does the episode's final cleanup when the retry is left to shutdown.
    failing_stops = 1 if retry == "final cleanup" else 2

    async def fail_setup(*args):
        raise RuntimeError("setup failed")

    async def stop():
        attempts.append(1)
        if len(attempts) <= failing_stops:
            raise RuntimeError("stop temporarily unavailable")
        sandbox.stopped = True

    monkeypatch.setattr(server, "_prepare_workdir", fail_setup)
    monkeypatch.setattr(sandbox, "stop", stop)
    response = await run_episode(server, task)
    assert response.failure.stage == "seed" and not response.failure.terminal
    assert "setup failed" in response.failure.failure_reason
    assert "Failed to stop Harbor sandbox" in caplog.text
    if retry == "final cleanup":
        assert sandbox.stopped and not server._sandboxes and len(attempts) == 2
    else:
        assert server._sandboxes["r1"] is sandbox and not sandbox.stopped and len(attempts) == 2
        # A repeat of the same episode (a lost reply redelivered) first retries the stop of the sandbox the
        # old attempt left behind and is refused while that keeps failing; its own final cleanup retries too.
        monkeypatch.setattr(server, "_prepare_workdir", HarborEnvironmentServer._prepare_workdir.__get__(server))
        failing_stops = 3
        refused = await run_episode(server, task)
        assert refused.failure.stage == "seed" and not refused.failure.terminal
        assert "could not be stopped yet" in refused.failure.failure_reason
        assert sandbox.stopped and not server._sandboxes and len(attempts) == 4
        # With the old sandbox gone, the next repeat provisions its own and runs.
        again = await run_episode(server, task)
        assert again.result.reward == 1 and len(attempts) == 5 and not server._sandboxes
    await server.shutdown()
    assert sandbox.stopped and not server._sandboxes


@pytest.mark.parametrize("artifacts", [None, Path("custom/verifier")])
async def test_logs_use_global_results_root_across_working_directories(tmp_path, monkeypatch, artifacts):
    old, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    results = tmp_path / "caller" / "results"
    component = tmp_path / "component"
    component.mkdir()
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"results_dir": str(results)}
    )
    monkeypatch.setattr("environment_servers.harbor.app.get_global_config_dict", lambda: {"results_dir": str(results)})
    monkeypatch.chdir(component)
    config = old.config.model_copy(update={"artifacts_dir": artifacts})
    server = HarborEnvironmentServer(config=config, server_client=old.server_client)
    monkeypatch.setattr(server, "_create_sandbox", old._create_sandbox)
    response = await run_episode(server, task)
    expected = results / (artifacts or "harbor/verifier")
    assert Path(response.result.verifier_logs_dir).parent == expected
    monkeypatch.chdir(tmp_path)
    assert (Path(response.result.verifier_logs_dir) / "reward.txt").read_text() == "1\n"
    assert not (component / "results").exists()


async def test_in_flight_duplicate_episode_does_not_stop_the_live_attempt(tmp_path, monkeypatch):
    """A redelivered episode key refused while the first attempt is working must leave that attempt's sandbox alone."""
    server, task, sandbox, created = make_server(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold():
        entered.set()
        await release.wait()

    during_turn(server, hold)
    first = asyncio.create_task(server.run_request(episode(task)))
    await asyncio.wait_for(entered.wait(), 5)
    server.server_client.turn_hooks.clear()
    duplicate = await server.run_request(episode(task))
    assert duplicate.failure.stage == "seed" and not duplicate.failure.terminal
    assert "still running" in duplicate.failure.failure_reason
    # The duplicate was refused before it registered anything: the first attempt still owns its sandbox.
    assert not sandbox.stopped and server._sandboxes["r1"] is sandbox and len(created) == 1
    release.set()
    assert (await first).result.reward == 1
    assert sandbox.stopped and not server._sandboxes and not server._sessions


async def test_serialize_failure_after_seed_is_a_retryable_seed_failure(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)

    async def broken_serialize(*args, **kwargs):
        raise RuntimeError("descriptor unavailable")

    monkeypatch.setattr(sandbox, "serialize", broken_serialize)
    response = await run_episode(server, task)
    assert response.failure.stage == "seed" and not response.failure.terminal
    assert "descriptor unavailable" in response.failure.failure_reason
    assert server.server_client.calls == [] and sandbox.stopped and not server._sandboxes


async def test_hung_agent_close_does_not_pin_episode_key(tmp_path, monkeypatch):
    """A close that hangs past the cleanup budget is abandoned; the sandbox stop and the key release still run.

    Each cleanup callback has its own budget (``CleanupContext.aclose``); under one shared budget the hung
    close starved both, the sandbox ran until its TTL and every redelivery of the episode was refused forever.
    The finished turn is still scored: a close that fails only by timeout does not discard it.
    """

    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    server.config = server.config.model_copy(update={"cleanup_timeout_seconds": 0.2})
    original = conftest.FakeAgentClient.post

    async def hanging_close(self, server_name, url_path, **kwargs):
        if url_path == "/v1/agent_sessions/close":
            await asyncio.sleep(60)
        return await original(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(conftest.FakeAgentClient, "post", hanging_close)
    response = await asyncio.wait_for(run_episode(server, task), 10)
    assert response.failure is None and response.result.reward == 1 and response.result.ng_agent_observations is None
    assert sandbox.stopped and not server._sandboxes and not server._sessions and server._active == set()
    # The client's redelivery of the episode is admitted again, not refused as "still running".
    again = await asyncio.wait_for(run_episode(server, task), 10)
    assert "still running" not in (again.failure.failure_reason if again.failure else "")


async def test_separate_verifier_sandbox_is_owned_until_its_stop_succeeds(tmp_path, monkeypatch, caplog):
    from environment_servers.harbor.tests.conftest import SEPARATE_TOML, AgentSandbox, FakeSandbox

    agent = AgentSandbox()
    server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=SEPARATE_TOML)
    verifier = FakeSandbox()
    attempts: list[int] = []

    async def failing_stop():
        attempts.append(1)
        # The verifier's own finally and the episode's final cleanup both fail; shutdown's retry succeeds.
        if len(attempts) <= 2:
            raise RuntimeError("stop temporarily unavailable")
        verifier.stopped = True

    async def start(spec):
        return None

    monkeypatch.setattr(verifier, "stop", failing_stop)
    monkeypatch.setattr(verifier, "start", start, raising=False)
    monkeypatch.setattr("environment_servers.harbor.sandboxes.resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", lambda config: verifier)
    response = await run_episode(server, task)
    assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
    assert "Could not stop the verifier sandbox" in caplog.text
    # The agent side was stopped once by the verifier flow and the episode key released without a second stop...
    assert agent.stopped and "r1" not in server._sandboxes and not server._sessions
    # ...and the verifier sandbox whose stop failed is still owned, so shutdown retries it.
    assert server._sandboxes == {"r1/verifier": verifier} and not verifier.stopped
    await server.shutdown()
    assert verifier.stopped and not server._sandboxes and len(attempts) == 3


async def test_missing_verifier_environment_env_fails_before_provisioning(tmp_path, monkeypatch):
    """``[verifier.environment.env]`` templates are checked at seed like the other two env tables."""
    monkeypatch.delenv("HARBOR_TEST_REQUIRED", raising=False)
    toml = TASK_TOML.replace(
        "[verifier]\ntimeout_sec = 120.0\n",
        '[verifier]\ntimeout_sec = 120.0\nenvironment_mode = "separate"\n\n[verifier.environment]\n'
        'docker_image = "org/verifier:1"\n\n[verifier.environment.env]\nTOKEN = "${HARBOR_TEST_REQUIRED}"\n',
    )
    server, task, _, created = make_server(tmp_path, monkeypatch, toml=toml)
    response = await run_episode(server, task)
    assert response.failure.stage == "seed" and response.failure.terminal
    assert "HARBOR_TEST_REQUIRED" in response.failure.failure_reason
    assert created == [] and server.server_client.calls == []


@pytest.mark.parametrize("error", ["transfer", "long_name", "provider", "host"])
async def test_artifact_collection_is_best_effort_but_provider_failures_still_mask(
    tmp_path, monkeypatch, caplog, error
):
    agent = AgentSandbox()
    server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=SEPARATE_TOML)
    verifier = FakeSandbox()

    async def create_verifier(task, *, episode_key=None):
        assert agent.stopped, "the agent sandbox must be stopped before the verifier is created"
        server._sandboxes[f"{episode_key}/verifier"] = verifier
        return verifier

    monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
    attempted = []

    async def failing_download(sandbox, source, target, **kwargs):
        attempted.append((source, kwargs.get("links")))
        if error == "transfer":
            raise SandboxTransferError(f"Failed to archive {source}: tar: Cannot open")
        if error == "long_name":
            # The agent's tree holds a path this host's filesystem cannot: the artifact's business, not an outage.
            raise OSError(errno.ENAMETOOLONG, "File name too long")
        if error == "host":
            raise OSError(errno.ENOSPC, "No space left on device")
        raise RuntimeError("sandbox is gone")

    monkeypatch.setattr("environment_servers.harbor.verifier.download_path", failing_download)
    response = await run_episode(server, task)
    # Artifacts travel with their links kept: the restore target is another sandbox.
    assert attempted and all(links == "keep" for _, links in attempted)
    if error in ("transfer", "long_name"):
        # Every artifact was attempted, none stopped the episode, and verification went ahead and scored.
        assert len(attempted) == 2 and response.result.reward == 1.0 and not response.result.mask_sample
        assert "not collected" in caplog.text
        assert any("test.sh" in call["command"] for call in verifier.execs)
        assert verifier.stopped
    else:
        # A dead sandbox or a full host disk is Gym's outage, not the agent's output: the sample is masked,
        # not scored, and no verifier sandbox was ever started for it.
        assert len(attempted) == 1 and response.result.mask_sample
        assert response.result.failure_kind == "provider_unavailable" and verifier.execs == []
    assert agent.stopped and not server._sandboxes


def separate_episode(tmp_path, monkeypatch, agent, artifacts: str):
    """A separate-mode episode whose task collects ``artifacts`` (a TOML array) from ``agent`` into a fake verifier."""
    toml = SEPARATE_TOML.replace(
        'artifacts = ["/app/output/report.json", { source = "/var/log/api", service = "api" }]',
        f"artifacts = {artifacts}",
    )
    server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=toml)
    verifier = FakeSandbox()

    async def create_verifier(task, *, episode_key=None):
        assert agent.stopped, "the agent sandbox must be stopped before the verifier is created"
        server._sandboxes[f"{episode_key}/verifier"] = verifier
        return verifier

    monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
    return server, task, verifier


async def test_a_restore_over_the_verifier_outputs_is_followed_by_their_reset_before_test_sh(tmp_path, monkeypatch):
    """``artifacts = ["/logs"]`` brings the agent's ``/logs/verifier`` back into the fresh sandbox; that directory is
    emptied again right after the restore, as root, and ``test.sh`` runs after that. Only the restore that covers it
    is followed by a reset, and each restore's target is resolved in the sandbox before it is prepared, so a
    refusal would have touched nothing."""
    agent = AgentSandbox(
        present={"/logs/artifacts": "dir", "/logs": "dir"},
        archives={
            "/logs/artifacts": conftest.archive_of({"state.db": "db"}),
            "/logs": conftest.archive_of({"verifier/reward.json": '{"reward": 1}', "agent/agent.log": "x"}),
        },
    )
    server, task, verifier = separate_episode(tmp_path, monkeypatch, agent, '["/logs"]')

    result = (await run_episode(server, task)).result

    assert result.reward == 1.0 and result.verifier_mode == "separate" and not result.mask_sample
    commands = [c["command"] for c in verifier.execs]
    reset = "rm -rf -- /logs/verifier && mkdir -p /logs/verifier && chmod 777 /logs/verifier"
    assert commands.count(reset) == 1 and verifier.execs[commands.index(reset)]["user"] == "root"
    restores = [
        i for i, c in enumerate(commands) if "tar -xzf" in c and ("-C /logs;" in c or "-C /logs/artifacts;" in c)
    ]
    run = next(i for i, c in enumerate(commands) if "test.sh" in c and "timeout" in c)
    assert len(restores) == 2 and max(restores) < commands.index(reset) < run
    for target in ("/logs/artifacts", "/logs"):
        prepared = commands.index(f"mkdir -p {target} && find {target} -mindepth 1 -delete && chmod 777 {target}")
        resolved = next(i for i, c in enumerate(commands) if c.startswith(f"p={target}; ") and c.endswith("pwd -P"))
        extracted = next(i for i, c in enumerate(commands) if "tar -xzf" in c and f"-C {target};" in c)
        assert resolved < prepared < extracted
    assert agent.archives["/logs"] in verifier.uploaded_archives and verifier.stopped


async def test_artifacts_at_the_tests_or_the_verifier_outputs_are_collected_but_never_restored(
    tmp_path, monkeypatch, caplog
):
    """The trusted tests directory and anything at or inside the verifier's output directory are refused at the
    restore by their declared path (the loader refuses the literal ``/tests`` and ``/logs/verifier``, so the
    server's directories are moved for this episode): their targets are never prepared, which would have emptied
    an image's own tests, nor extracted into; the warning names each; the other artifact is restored and the
    sample scored."""
    monkeypatch.setattr("environment_servers.harbor.verifier.TESTS_DIR", "/trusted/tests")
    monkeypatch.setattr("environment_servers.harbor.verifier.VERIFIER_LOGS_DIR", "/trusted/verifier")
    agent = AgentSandbox(
        present={
            "/logs/artifacts": "dir",
            "/trusted/tests": "dir",
            "/trusted/verifier/reward.json": "file",
            "/app/output/report.json": "file",
        },
        archives={
            "/logs/artifacts": conftest.archive_of({}),
            "/trusted/tests": conftest.archive_of({"test.sh": "echo 1 > /trusted/verifier/reward.txt"}),
            "/trusted/verifier": conftest.archive_of({"reward.json": '{"reward": 1}'}),
            "/app/output": conftest.archive_of({"report.json": "{}"}),
        },
    )
    server, task, verifier = separate_episode(
        tmp_path, monkeypatch, agent, '["/trusted/tests", "//trusted/verifier/reward.json", "/app/output/report.json"]'
    )

    with caplog.at_level("WARNING"):
        result = (await run_episode(server, task)).result

    assert result.reward == 1.0 and not result.mask_sample, result.failure_reason
    assert "artifact '/trusted/tests' not restored: it would replace the grader's /trusted/tests" in caplog.text
    assert (
        "artifact '/trusted/verifier/reward.json' not restored: it would replace the grader's /trusted/verifier"
        in caplog.text
    )
    commands = [c["command"] for c in verifier.execs]
    assert not any("find /trusted/tests -mindepth 1 -delete" in c or "-C /trusted/verifier;" in c for c in commands)
    # The one extraction into the tests directory is the package's own tests/; the agent's archives never went up.
    assert sum("tar -xzf" in c and "-C /trusted/tests;" in c for c in commands) == 1
    assert agent.archives["/trusted/tests"] not in verifier.uploaded_archives
    assert agent.archives["/trusted/verifier"] not in verifier.uploaded_archives
    assert agent.archives["/app/output"] in verifier.uploaded_archives
    # Both were collected on the host under their own staging roots, as diagnostics.
    artifacts = Path(result.verifier_logs_dir) / "artifacts"
    assert (artifacts / "01" / "trusted" / "tests" / "test.sh").is_file()
    assert (artifacts / "02" / "trusted" / "verifier" / "reward.json").is_file()
    assert agent.stopped and verifier.stopped


@pytest.mark.parametrize("source", ["/tests", "/tests/test.sh", "//tests", "/", "/logs/verifier", "//logs/verifier/x"])
async def test_an_artifact_at_the_tests_or_the_verifier_outputs_is_refused_when_the_task_loads(
    tmp_path, monkeypatch, source
):
    """Deterministic per task, so refused once, at load, naming the artifact, rather than warned about per episode."""
    toml = SEPARATE_TOML.replace(
        'artifacts = ["/app/output/report.json", { source = "/var/log/api", service = "api" }]',
        f'artifacts = ["{source}"]',
    )
    server, task, _, _ = make_server(tmp_path, monkeypatch, AgentSandbox())
    (task.path / "task.toml").write_text(toml)
    settled = "/" + source.lstrip("/")
    with pytest.raises(
        HarborTaskError, match=f"{settled!r} would replace the verifier's /(tests|logs/verifier)"
    ) as info:
        load_task(task.path)
    assert str(task.path / "task.toml") in str(info.value)


def test_within_and_protected_restore_settle_the_spelling_of_sandbox_paths():
    """A declared ``//tests`` reaches the same directory as ``/tests`` and is judged as such; a sibling that only
    shares the prefix is not inside; a trailing slash changes nothing."""
    within = HarborEnvironmentServer._within
    assert within("//tests", "/tests") and within("/tests/", "//tests") and within("/tests/test.sh", "//tests/")
    assert within("/tests", "/") and not within("/", "/tests")
    assert not within("/tests-data", "/tests") and not within("/logs/verifier-cache", "/logs/verifier")
    protected = HarborEnvironmentServer._protected_restore
    assert protected("//tests") == "/tests" and protected("/tests/x") == "/tests" and protected("//") == "/tests"
    assert (
        protected("//logs/verifier/") == "/logs/verifier"
        and protected("/logs/verifier/reward.json") == "/logs/verifier"
    )
    assert protected("//logs") is None and protected("/logs/agent") is None and protected("/tests-data") is None


async def test_repeat_attempt_retries_the_previous_verifier_stop(tmp_path, monkeypatch):
    """A verifier sandbox left owned by a failed stop is stopped before a redelivered episode provisions."""
    agent = AgentSandbox()
    server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=SEPARATE_TOML)
    first, second = FakeSandbox(), FakeSandbox()
    first_attempts = []

    async def flaky_stop():
        first_attempts.append(1)
        # Fails in the verifier's own finally and in the episode's final cleanup; the redelivery's retry succeeds.
        if len(first_attempts) <= 2:
            raise RuntimeError("stop temporarily unavailable")
        first.stopped = True

    async def start(spec):
        return None

    monkeypatch.setattr(first, "stop", flaky_stop)
    for sandbox in (first, second):
        monkeypatch.setattr(sandbox, "start", start, raising=False)
    monkeypatch.setattr("environment_servers.harbor.sandboxes.resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    verifiers = iter([first, second])
    monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", lambda config: next(verifiers))
    assert (await run_episode(server, task)).result.reward == 1.0
    assert server._sandboxes == {"r1/verifier": first} and not first.stopped
    # The redelivery (a lost reply) retries that stop first, then runs with its own verifier.
    agent.stopped = False
    again = await run_episode(server, task)
    assert again.result.reward == 1.0 and len(first_attempts) == 3 and first.stopped
    assert second.stopped and not server._sandboxes and not server._sessions


async def test_interrupted_verifier_start_is_stopped_by_the_episode(tmp_path, monkeypatch):
    agent = AgentSandbox()
    server, task, _, _ = make_server(
        tmp_path, monkeypatch, agent, toml=SEPARATE_TOML, default_episode_timeout_seconds=0.5
    )
    verifier = FakeSandbox()

    async def hang(spec):
        await asyncio.Future()

    monkeypatch.setattr(verifier, "start", hang, raising=False)
    monkeypatch.setattr("environment_servers.harbor.sandboxes.resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", lambda config: verifier)
    response = await asyncio.wait_for(run_episode(server, task), 10)
    assert response.failure is not None
    # The episode's own cleanup stops the half-started verifier; it does not wait for shutdown.
    assert verifier.stopped and agent.stopped and not server._sandboxes


@pytest.mark.parametrize("mode", ["task_image", "explicit", "explicit_workdir"])
async def test_separate_verifier_spec_follows_the_task_image_or_the_explicit_environment(tmp_path, monkeypatch, mode):
    """Built through ``_create_verifier_sandbox`` itself, not a test double of it."""
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "secret")
    explicit = mode != "task_image"
    # The task image: Harbor's effective verifier environment is a copy of [environment], so the task must declare
    # the image there (without one and without tests/Dockerfile the loader refuses the task, as Harbor does).
    toml = (
        SEPARATE_TOML
        if explicit
        else SEPARATE_IN_TASK_IMAGE_TOML.replace("[environment]\n", '[environment]\ndocker_image = "org/agent:1"\n')
        + '\n[environment.env]\nAPP_MODE = "${HARBOR_TEST_TOKEN}"\n'
    )
    if mode == "explicit_workdir":
        toml = toml.replace(
            'docker_image = "org/verifier:1"\n', 'docker_image = "org/verifier:1"\nworkdir = "/work"\n'
        )
    server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml)
    verifier = FakeSandbox()
    specs = []

    async def start(spec):
        specs.append(spec)

    monkeypatch.setattr(verifier, "start", start, raising=False)
    monkeypatch.setattr("environment_servers.harbor.sandboxes.resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", lambda config: verifier)
    assert await server._create_verifier_sandbox(task) is verifier
    (spec,) = specs
    assert spec.metadata["harbor_role"] == "verifier"
    # The verifier lives as long as the agent's sandbox: readiness, transfers, test.sh with its grace and reward
    # collection all fit, and so does the TB4 provider's 1800 s renew interval, which the old `timeout_sec + slack`
    # (300 + 900) did not clear.
    assert spec.ttl_s == server._sandbox_spec(task, task.workdir).ttl_s >= 1800 + task.config.verifier.timeout_sec
    if explicit:
        # An explicit verifier image starts in its own WORKDIR, or the one `[verifier.environment]` names,
        # with its own variables.
        # ... plus the CPU-count hints for its own `cpus = 2`.
        assert spec.image == "org/verifier:1" and spec.env == cpu_cap_env(2)
        assert spec.workdir == ("/work" if mode == "explicit_workdir" else None)
    else:
        # The task image: the agent's working directory and resolved environment, Dockerfile ENV included.
        assert task.workdir == "/app"
        assert spec.image == task.image and spec.workdir == "/app"
        assert spec.env == cpu_cap_env(1) | _task_sandbox_env(task) and spec.env["APP_MODE"] == "secret"


def test_task_data_requires_digest_and_accepts_materialized_row(tmp_path, monkeypatch):
    _, task, _, _ = make_server(tmp_path, monkeypatch)
    row = materialize_task(task, "ds").model_dump(mode="json")
    data = TaskData.model_validate(row["task_input"]["task_data"])
    assert data.ng_digest == task.digest
    assert DIGEST_KEY in TaskData.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        TaskData.model_validate({})
