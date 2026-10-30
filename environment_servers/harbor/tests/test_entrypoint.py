# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the image ENTRYPOINT launcher: keepalive detection, the launch beside the provider
keepalive, healthcheck gating, exit status recording, late exits during the rollout, the
entrypoint log collection and the user the entrypoint runs as.
"""

import asyncio
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from environment_servers.harbor.entrypoint import (
    ENTRYPOINT_DIR,
    ENTRYPOINT_FAILED_KIND,
    entrypoint_to_run,
    image_healthcheck_settings,
    is_keepalive_entrypoint,
)
from environment_servers.harbor.tests.conftest import (
    HEALTHCHECK_TOML,
    IMAGE_PATH,
    OK,
    OVERLAY_DOCKERFILE,
    SEPARATE_IN_TASK_IMAGE_TOML,
    TASK_TOML,
    VERIFIER_TESTS_DOCKERFILE,
    FakeCompose,
    FakeSandbox,
    archive_of,
    make_server,
    record_verifier_build,
    run_episode,
    write_image_config,
)
from environment_servers.harbor.verifier import _verifier_image, _verifier_runs_task_image, _verifier_start
from nemo_gym import failure_kinds
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.tasks.harbor import load_task


BUILT_IMAGE = "registry.example/ds-hello:abc"
ENTRYPOINT_CONFIG = {
    "Env": [f"PATH={IMAGE_PATH}"],
    "Entrypoint": ["/entrypoint.sh"],
    "User": "app",
    "WorkingDir": "/app",
}


@dataclass
class EntrypointSandbox(FakeSandbox):
    """A sandbox whose image entrypoint the seed starts: answers the markers the launcher polls.

    The launch exec (``sh DIR/run.sh ...``) waits on ``release`` (the steady state: an entrypoint that
    never returns) unless ``launch_result`` is set, in which case it returns at once (the provider could
    not run it). ``exit_status`` is what ``DIR/exit`` holds (``None``: no exit file yet); it becomes
    ``exit_after_probes_status`` once the healthcheck has been probed ``exit_after_probes`` times.
    ``health_failures`` is how many healthcheck probes fail before one passes, ``log`` the entrypoint's
    output. ``download_fails`` makes the log transfer fail. ``exit_after_exit_probes`` makes the exit
    file appear right after the N-th exit probe (a death between two probes). ``release_after_probes``
    ends the launch exec after the N-th healthcheck probe with ``hold_result`` (the provider capped the
    exec, or ``run.sh`` finished).
    """

    exit_status: str | None = None
    exit_after_probes: int | None = None
    exit_after_probes_status: str = "2"
    exit_after_exit_probes: int | None = None
    exit_after_exit_probes_status: str = "1"
    release_after_probes: int | None = None
    hold_result: SandboxExecResult = OK
    log: str = "LAB-Bench Runtime Setup - Shuffling choices...\nUsing random seed=843121323\n"
    launch_result: SandboxExecResult | None = None
    health_failures: int = 0
    download_fails: bool = False
    launched: bool = False
    launch_cancelled_before_stop: bool | None = None
    log_downloaded_before_stop: bool | None = None
    default_user_at_launch: str | None = None
    health_probes: int = 0
    exit_probes: int = 0
    log_requested: bool = False
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        record = {"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user, "seen": None}
        if command.startswith(f"sh {ENTRYPOINT_DIR}/run.sh"):
            self.execs.append(record)
            self.launched = True
            self.default_user_at_launch = self.default_user
            if self.launch_result is not None:
                return self.launch_result
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.launch_cancelled_before_stop = not self.stopped
                raise
            return self.hold_result
        if command == f"test -f {ENTRYPOINT_DIR}/started":
            self.execs.append(record)
            ok = self.launched and self.launch_result is None
            return SandboxExecResult(stdout="", stderr="", return_code=0 if ok else 1)
        if command.startswith(f"if test -f {ENTRYPOINT_DIR}/exit"):
            self.execs.append(record)
            self.exit_probes += 1
            status = self.exit_status
            if self.exit_after_exit_probes is not None and self.exit_probes >= self.exit_after_exit_probes:
                self.exit_status = self.exit_after_exit_probes_status
            return SandboxExecResult(stdout=status or "", stderr="", return_code=0)
        if command.startswith("tail -c "):
            self.execs.append(record)
            return SandboxExecResult(stdout=self.log[-1500:], stderr="", return_code=0)
        if command.startswith(f"if [ -d {ENTRYPOINT_DIR}/log ]"):
            self.execs.append(record)
            return SandboxExecResult(stdout="file\n" if self.launched else "none\n", stderr="", return_code=0)
        if command.startswith(f"cd {ENTRYPOINT_DIR} && ") and "tar -czf" in command:
            # The log travels as a one-file archive of the entrypoint folder, so its mode comes with it.
            self.execs.append(record)
            self.log_requested = True
            return OK
        probe = command
        if command.startswith(("/bin/sh -c ", "/bin/bash ")):
            # An image healthcheck script arrives wrapped in the image's Shell (/bin/sh -c, or the recorded one).
            words = shlex.split(command)
            if len(words) >= 3 and words[-2] == "-c":
                probe = words[-1]
        if probe in ("test -f /app/ready.txt", "test -f /tmp/healthy"):
            self.execs.append(record)
            self.health_probes += 1
            if self.exit_after_probes is not None and self.health_probes >= self.exit_after_probes:
                self.exit_status = self.exit_after_probes_status
            if self.release_after_probes is not None and self.health_probes >= self.release_after_probes:
                self.release.set()
            healthy = self.health_probes > self.health_failures
            return SandboxExecResult(stdout="", stderr="", return_code=0 if healthy else 1)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    async def download(self, remote_path, local_path):
        if self.log_requested:
            self.log_requested = False
            if self.download_fails:
                raise RuntimeError("transfer lost")
            self.log_downloaded_before_stop = not self.stopped
            Path(local_path).write_bytes(archive_of({"log": self.log}))
            return
        await super().download(remote_path, local_path)

    def launch(self) -> dict | None:
        launches = [c for c in self.execs if c["command"].startswith(f"sh {ENTRYPOINT_DIR}/run.sh")]
        assert len(launches) <= 1, launches
        return launches[0] if launches else None

    def index_of(self, fragment: str) -> int:
        return next(i for i, c in enumerate(self.execs) if fragment in c["command"])

    def run_script(self) -> str | None:
        for index, (_, remote) in enumerate(self.uploads):
            if remote == f"{ENTRYPOINT_DIR}/run.sh":
                return self.uploaded_archives[index].decode()
        return None


@dataclass
class LateExitSandbox(EntrypointSandbox):
    """The entrypoint exits once the sandbox is handed to the harness (after the seed's last probe).

    ``late_returns`` is whether ``run.sh`` finished too, so the exec holding the entrypoint returns, or
    only the exit file appears (what an agent could write into the world-writable runtime folder).
    """

    late_status: str = "1"
    late_returns: bool = True

    async def serialize(self, *, scope=None):
        self.exit_status = self.late_status
        self.log = "odoo: database connection lost\n"
        if self.late_returns:
            self.release.set()
            await asyncio.sleep(0)  # the holding exec returns while the harness works, as in a real rollout
        return await super().serialize(scope=scope)


AGENT_USER_TOML = TASK_TOML.replace("[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "agent"')
WORKDIR_TOML = AGENT_USER_TOML.replace("storage_mb = 10240", 'storage_mb = 10240\nworkdir = "/work"')
AGENT_USER_HEALTH_TOML = HEALTHCHECK_TOML.replace(
    "[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "agent"'
)
SEPARATE_HEALTH_TOML = HEALTHCHECK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
NO_USER_CONFIG = {"Env": [f"PATH={IMAGE_PATH}"], "Entrypoint": ["/entrypoint.sh"], "WorkingDir": "/app"}
IMAGE_HEALTHCHECK = {"Test": ["CMD-SHELL", "test -f /tmp/healthy"], "Interval": 10_000_000}
# IMAGE_HEALTHCHECK as the seed runs it: a CMD-SHELL script through the image's Shell, /bin/sh -c when it sets none.
IMAGE_SHELL_PROBE = "/bin/sh -c 'test -f /tmp/healthy'"


def build_mode(task, *, record: dict | None = ENTRYPOINT_CONFIG, image: str = BUILT_IMAGE):
    """Turn the test task into a build-mode one: a Dockerfile that needs a build, resolved to ``image``.

    ``record`` is the built image's recorded configuration (``None``: nothing recorded, as for an image
    prepare did not inspect).
    """
    from nemo_gym.tasks.harbor.digest import context_digest
    from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE

    (task.path / "environment" / "Dockerfile").write_text(
        'FROM ubuntu:24.04\nCOPY . /app\nENTRYPOINT ["/entrypoint.sh"]\n'
    )
    record_path = task.path.parent / BUILT_IMAGES_FILE
    built = json.loads(record_path.read_text()) if record_path.is_file() else {}
    built[context_digest(task.path / "environment")] = {"image": image}
    record_path.write_text(json.dumps(built))
    if record is not None:
        write_image_config(task.path.parent, image, **record)
    return load_task(task.path)


class TestEntrypoint:
    """G4: a plain task's image ENTRYPOINT runs beside the keepalive, as Harbor's Compose start leaves it."""

    def entrypoint_server(self, tmp_path, monkeypatch, *, sandbox=None, toml=TASK_TOML, record=ENTRYPOINT_CONFIG):
        sandbox = sandbox or EntrypointSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox, toml=toml)
        task = build_mode(task, record=record)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        return server, task, sandbox

    async def test_built_image_entrypoint_runs_as_the_image_user_between_logs_and_healthcheck(
        self, tmp_path, monkeypatch
    ):
        toml = HEALTHCHECK_TOML.replace("storage_mb = 10240", 'storage_mb = 10240\nskills_dir = "/skills"')
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=toml)
        assert task.entrypoint == ("/entrypoint.sh",) and task.user == "app"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        script = sandbox.run_script()
        assert script is not None
        assert script.splitlines() == [
            f"touch {ENTRYPOINT_DIR}/started",
            "/entrypoint.sh sh -c 'sleep infinity'",
            "status=$?",
            f"printf '%s' \"$status\" > {ENTRYPOINT_DIR}/exit",
        ]
        launch = sandbox.launch()
        assert launch["command"] == f"sh {ENTRYPOINT_DIR}/run.sh >{ENTRYPOINT_DIR}/log 2>&1 </dev/null"
        assert (launch["user"], launch["cwd"], launch["timeout_s"]) == ("app", "/app", None)
        # The runtime folder is made as root and opened up, so the image user can write the markers.
        runtime = next(c for c in sandbox.execs if f"chmod 777 {ENTRYPOINT_DIR}" in c["command"])
        assert runtime["command"] == f"mkdir -p {ENTRYPOINT_DIR} && chmod 777 {ENTRYPOINT_DIR}"
        assert runtime["user"] == "root"
        # Order: /logs exists before the entrypoint starts; the healthcheck and the task context come after it.
        logs = sandbox.index_of("/logs/artifacts")
        started = sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh")
        assert logs < sandbox.index_of(f"mkdir -p {ENTRYPOINT_DIR}") < started
        # The started marker was polled until it existed, then the exit status probed once, before the healthcheck.
        polls = [i for i, c in enumerate(sandbox.execs) if c["command"] == f"test -f {ENTRYPOINT_DIR}/started"]
        assert polls and polls[-1] < sandbox.index_of(f"if test -f {ENTRYPOINT_DIR}/exit")
        assert started < sandbox.index_of("test -f /app/ready.txt") < sandbox.index_of("/tmp/.nemo-gym/task.json")
        # The entrypoint's exec was cancelled before the sandbox was stopped, and nothing stays owned.
        assert sandbox.launch_cancelled_before_stop is True and sandbox.stopped
        assert not server._sandboxes and not server._sessions

    @pytest.mark.parametrize(
        "record",
        [
            pytest.param(None, id="no-record"),
            pytest.param({"Env": [f"PATH={IMAGE_PATH}"]}, id="no-entrypoint"),
            pytest.param({"Entrypoint": []}, id="cleared"),
            pytest.param({"Entrypoint": ["sleep", "infinity"]}, id="sleep"),
        ],
    )
    async def test_nothing_is_launched_without_a_real_entrypoint(self, tmp_path, monkeypatch, record):
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, record=record)
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.launch() is None and sandbox.run_script() is None
        assert not any(ENTRYPOINT_DIR in c["command"] for c in sandbox.execs)

    def test_compose_tasks_leave_the_entrypoint_to_the_adapter(self, tmp_path, monkeypatch):
        server, task, _ = self.entrypoint_server(tmp_path, monkeypatch)
        assert entrypoint_to_run(task) == ["/entrypoint.sh"]
        (task.path / "environment" / "docker-compose.yaml").write_text("services:\n  main:\n    image: x\n")
        task = build_mode(task)  # the Compose file is part of the build context, so the record is keyed anew
        assert task.needs_compose and task.entrypoint == ("/entrypoint.sh",)
        # The adapter started main (the seed says so); a plain sandbox of the same image starts it itself.
        assert entrypoint_to_run(task, compose_managed=True) is None
        assert entrypoint_to_run(task) == ["/entrypoint.sh"]

    @pytest.mark.parametrize(
        ("argv", "keepalive"),
        [
            (["sleep", "infinity"], True),
            (["/usr/bin/tail", "-f", "/dev/null"], True),
            (["bash", "-c", "exec sleep infinity"], True),
            (["sh", "-c", "sleep infinity"], True),
            (["/bin/bash", "-c", "exec tail -f /dev/null"], True),
            (["sh", "-c", "while :; do sleep 30; done"], True),
            (["sh", "-c", "while true; do sleep 1; done"], True),
            (["/entrypoint.sh"], False),
            (["bash", "/opt/blobfish-world/entrypoint.sh"], False),
            (["sh", "-c", "  exec sleep infinity  "], True),
            (["sh", "-c", "sleep 5; python app.py"], False),  # compound: the services after the sleep must start
            (["sh", "-c", "sleep 2; service redis-server start; exec sleep infinity"], False),
            (["sh", "-c", "sleep 5 && service postgresql start && exec sleep infinity"], False),
            (["bash", "-c", "tail -n0 -f /var/log/app.log | ./watcher"], False),
            (["sh", "-c", "python app.py && sleep infinity"], False),
            (["bash"], False),
            ([], False),
        ],
    )
    def test_keepalive_rule(self, argv, keepalive):
        assert is_keepalive_entrypoint(argv) is keepalive

    async def test_overlay_task_launches_the_base_entrypoint_after_the_run_lines(self, tmp_path, monkeypatch):
        dockerfile = OVERLAY_DOCKERFILE.replace("ENTRYPOINT []\n", "")
        config = {"Env": [f"PATH={IMAGE_PATH}"], "Entrypoint": ["/entrypoint.sh"]}
        sandbox = EntrypointSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox, dockerfile=dockerfile, image_config=config)
        # The base image's entrypoint, started with the Dockerfile's final USER and WORKDIR.
        assert task.entrypoint == ("/entrypoint.sh",)
        assert (task.entrypoint_user, task.entrypoint_workdir) == ("agent", "/app")

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        launch = sandbox.launch()
        assert (launch["user"], launch["cwd"]) == ("agent", "/app")
        last_run = max(i for i, c in enumerate(sandbox.execs) if c["seen"] and "as agent" in c["seen"]["command"])
        assert last_run < sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh")
        assert sandbox.default_user_at_launch == "agent"

        # The same Dockerfile with ENTRYPOINT [] clears it: nothing is launched.
        cleared = EntrypointSandbox()
        server, task, _, _ = make_server(
            tmp_path / "b", monkeypatch, cleared, image_config=config, dockerfile=OVERLAY_DOCKERFILE
        )
        assert task.entrypoint is None and task.entrypoint_user == "agent"
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert cleared.launch() is None

    async def test_entrypoint_exiting_non_zero_is_a_terminal_seed_failure_with_its_output(self, tmp_path, monkeypatch):
        sandbox = EntrypointSandbox(exit_status="1", log="LAB-Bench Runtime Setup\nTraceback: setup failed\n")
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "Image ENTRYPOINT failed for 'hello'" in failure.failure_reason
        assert "ENTRYPOINT /entrypoint.sh exited 1" in failure.failure_reason
        assert "Traceback: setup failed" in failure.failure_reason
        assert sandbox.launch_cancelled_before_stop is True and sandbox.stopped
        assert not server._sandboxes and not server._sessions
        assert server.server_client.paths() == []  # never handed to the harness

    async def test_entrypoint_returning_zero_warns_once_and_continues(self, tmp_path, monkeypatch, caplog):
        sandbox = EntrypointSandbox(exit_status="0", health_failures=2)
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, toml=HEALTHCHECK_TOML)

        with caplog.at_level("WARNING", logger="environment_servers.harbor.entrypoint"):
            response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        warnings = [r.getMessage() for r in caplog.records if "exited 0" in r.getMessage()]
        assert len(warnings) == 1
        assert "hello: ENTRYPOINT /entrypoint.sh exited 0 without running its keepalive argument" in warnings[0]
        assert "LAB-Bench Runtime Setup" in warnings[0]
        # The healthcheck was still polled to a pass; the exit was not probed again once known.
        assert sandbox.health_probes == 3 and sandbox.exit_probes == 1

    async def test_launcher_that_never_starts_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        sandbox = EntrypointSandbox(
            launch_result=SandboxExecResult(stdout="", stderr="", return_code=125, error_type="sandbox")
        )
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and not failure.terminal
        assert "the entrypoint launcher exited before starting: sandbox" in failure.failure_reason
        assert sandbox.stopped and not server._sandboxes

    async def test_launcher_shell_failing_before_the_marker_is_a_terminal_seed_failure(self, tmp_path, monkeypatch):
        # `su`/`chdir` for the image's user or directory failed before run.sh's first line: the image's fault,
        # like a RUN line exiting non-zero; another sandbox would repeat it.
        sandbox = EntrypointSandbox(
            launch_result=SandboxExecResult(stdout="", stderr="su: user 1000 does not exist\n", return_code=1)
        )
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "Image ENTRYPOINT failed for 'hello'" in failure.failure_reason
        assert "(as app in /app) exited 1 before starting it" in failure.failure_reason
        assert "su: user 1000 does not exist" in failure.failure_reason
        assert sandbox.stopped and not server._sandboxes and server.server_client.paths() == []

    async def test_launcher_start_timeout_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        class NeverStarted(EntrypointSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command == f"test -f {ENTRYPOINT_DIR}/started":
                    self.execs.append({"command": command})
                    return SandboxExecResult(stdout="", stderr="", return_code=1)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        monkeypatch.setattr("environment_servers.harbor.entrypoint.ENTRYPOINT_START_TIMEOUT_S", 0.05)
        monkeypatch.setattr("environment_servers.harbor.entrypoint.ENTRYPOINT_POLL_INTERVAL_S", 0.01)
        sandbox = NeverStarted()
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "did not start within 0.05s" in response.failure.failure_reason
        assert sandbox.launch_cancelled_before_stop is True and sandbox.stopped

    async def test_healthcheck_loop_fails_fast_when_the_entrypoint_dies(self, tmp_path, monkeypatch):
        # The healthcheck never passes; the entrypoint exits after the first probe. Without the probe the
        # seed would wait out the three retries.
        sandbox = EntrypointSandbox(health_failures=100, exit_after_probes=1, log="odoo: connection refused\n")
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, toml=HEALTHCHECK_TOML)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "seed" and failure.terminal
        assert "exited 2" in failure.failure_reason and "odoo: connection refused" in failure.failure_reason
        assert sandbox.health_probes == 1
        assert "Healthcheck failed" not in failure.failure_reason

    def test_image_healthcheck_settings(self):
        ns = 1_000_000_000
        settings = image_healthcheck_settings(
            {
                "Test": ["CMD-SHELL", "curl -f http://localhost/ || exit 1"],
                "Interval": 10 * ns,
                "Timeout": 3 * ns,
                "StartPeriod": 2 * ns,
                "Retries": 5,
            }
        )
        # The script runs through the image's Shell, Docker's /bin/sh -c when none is recorded (moby daemon/health.go).
        assert settings.command == "/bin/sh -c 'curl -f http://localhost/ || exit 1'"
        assert (settings.interval_sec, settings.timeout_sec, settings.start_period_sec, settings.retries) == (
            10,
            3,
            2,
            5,
        )
        assert settings.start_interval_sec == 5.0  # Docker's default when the Dockerfile sets none
        pipefail = ("/bin/bash", "-o", "pipefail", "-c")
        configured = image_healthcheck_settings({"Test": ["CMD-SHELL", "false | true"]}, pipefail)
        assert configured.command == "/bin/bash -o pipefail -c 'false | true'"
        assert (
            image_healthcheck_settings({"Test": ["CMD-SHELL", "false | true"]}, None).command
            == "/bin/sh -c 'false | true'"
        )
        # A CMD vector runs as written: the image's shell plays no part in it.
        exec_form = image_healthcheck_settings({"Test": ["CMD", "pg_isready", "-U", "odoo user"]}, pipefail)
        assert exec_form.command == "pg_isready -U 'odoo user'"
        assert (exec_form.interval_sec, exec_form.timeout_sec, exec_form.retries) == (30.0, 30.0, 3)
        assert image_healthcheck_settings({"Test": ["NONE"]}) is None
        assert image_healthcheck_settings({"Test": []}) is None
        assert image_healthcheck_settings({}) is None and image_healthcheck_settings(None) is None

    @pytest.mark.parametrize("passes", [True, False])
    async def test_recorded_dockerfile_healthcheck_is_polled(self, tmp_path, monkeypatch, passes):
        ns = 1_000_000_000
        record = ENTRYPOINT_CONFIG | {
            "Healthcheck": {
                "Test": ["CMD-SHELL", "test -f /tmp/healthy"],
                "Interval": ns // 100,
                "Timeout": 5 * ns,
                "Retries": 3,
            }
        }
        sandbox = EntrypointSandbox(health_failures=2 if passes else 100)
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, record=record)
        assert task.image_healthcheck == record["Healthcheck"]

        response = await run_episode(server, task)

        if passes:
            assert response.failure is None, response.failure
            assert sandbox.health_probes == 3
            # The image sets no SHELL: the script runs through Docker's default, named, not the provider's own shell.
            probe = next(c for c in sandbox.execs if c["command"] == IMAGE_SHELL_PROBE)
            assert probe["timeout_s"] == 10  # the recorded 5 s timeout plus the server's margin
            assert sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < sandbox.index_of("test -f /tmp/healthy")
        else:
            assert response.failure.stage == "seed" and not response.failure.terminal
            assert (
                f"Healthcheck failed 3 times: {IMAGE_SHELL_PROBE!r} (the image HEALTHCHECK)"
                in response.failure.failure_reason
            )
            assert sandbox.health_probes == 3 and sandbox.stopped and not server._sandboxes

    @pytest.mark.parametrize(
        "configured_shell", [None, ["/bin/bash", "-o", "pipefail", "-c"]], ids=["default", "pipefail"]
    )
    async def test_cmd_shell_healthcheck_runs_through_the_image_shell(self, tmp_path, monkeypatch, configured_shell):
        """Docker runs a CMD-SHELL check through the image's configured Shell (moby daemon/health.go). With
        ``SHELL ["/bin/bash", "-o", "pipefail", "-c"]`` and ``HEALTHCHECK CMD false | true`` the check fails under
        Docker; under the provider's default ``sh -c`` the pipeline's last command decides, and it passed. The
        recorded Shell travels through the loader and the healthcheck loop, and the probe here runs the generated
        command the way the DockerProvider does, ``<exec shell> -c <command>``, on this host instead of a container."""
        ns = 1_000_000_000
        record = ENTRYPOINT_CONFIG | {
            "Healthcheck": {"Test": ["CMD-SHELL", "false | true"], "Interval": ns // 100, "Retries": 1},
            **({"Shell": configured_shell} if configured_shell else {}),
        }

        @dataclass
        class HostShellSandbox(EntrypointSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if "false | true" in command:
                    self.execs.append({"command": command, "cwd": cwd, "user": user, "timeout_s": timeout_s})
                    self.health_probes += 1
                    run = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
                    return SandboxExecResult(stdout=run.stdout, stderr=run.stderr, return_code=run.returncode)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        sandbox = HostShellSandbox()
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, record=record)
        assert task.image_shell == tuple(configured_shell or ("/bin/sh", "-c"))

        response = await run_episode(server, task)

        probes = [c for c in sandbox.execs if "false | true" in c["command"]]
        if configured_shell is None:
            # An image without SHELL still runs its script through /bin/sh -c, as the daemon does, and passes.
            assert response.failure is None, response.failure
            assert [c["command"] for c in probes] == ["/bin/sh -c 'false | true'"]
        else:
            # The image's whole shell argument list runs the script, so pipefail fails readiness as under Docker.
            assert response.failure is not None and response.failure.stage == "seed"
            assert [c["command"] for c in probes] == ["/bin/bash -o pipefail -c 'false | true'"]
            assert (
                "Healthcheck failed 1 times: \"/bin/bash -o pipefail -c 'false | true'\" (the image HEALTHCHECK)"
                in response.failure.failure_reason
            )
            assert sandbox.stopped and not server._sandboxes
        assert all(c["cwd"] == "/app" and c["user"] == "app" for c in probes)

    async def test_compose_main_leaves_its_image_healthcheck_to_the_adapter(self, tmp_path, monkeypatch):
        # The Compose adapter runs main's recorded HEALTHCHECK (through the image's Shell too) before the seed sees
        # the sandbox; the seed polls only the task's [environment.healthcheck] there, whatever the image records.
        record = ENTRYPOINT_CONFIG | {
            "Healthcheck": {"Test": ["CMD-SHELL", "false | true"], "Retries": 1},
            "Shell": ["/bin/bash", "-o", "pipefail", "-c"],
        }
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=HEALTHCHECK_TOML, record=record)
        (task.path / "environment" / "docker-compose.yaml").write_text("services:\n  main:\n    image: x\n")
        task = build_mode(task, record=record)
        assert task.needs_compose and task.image_shell == ("/bin/bash", "-o", "pipefail", "-c")
        await server._wait_healthy(sandbox, task, compose_managed=True)
        assert not any("false | true" in c["command"] for c in sandbox.execs)
        assert sandbox.health_probes == 1 and sandbox.execs[-1]["command"] == "test -f /app/ready.txt"

    async def test_image_healthcheck_runs_before_the_task_healthcheck_as_the_image_user(self, tmp_path, monkeypatch):
        # Harbor's order: `up --wait` gates on the image HEALTHCHECK, then run_healthcheck() runs the task's, both
        # as the container's user (the agent user is set afterwards): the daemon runs the image check in the image
        # WorkingDir, Harbor's exec runs the task check in the task workdir; the seed names both directories (the
        # task's is the prepared workdir, the image's own /app for a task that declares none).
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK}
        server, task, sandbox = self.entrypoint_server(
            tmp_path, monkeypatch, toml=AGENT_USER_HEALTH_TOML, record=record
        )
        assert (task.user, task.entrypoint_user) == ("agent", "app")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        image = sandbox.index_of("test -f /tmp/healthy")
        task_probe = sandbox.index_of("test -f /app/ready.txt")
        assert sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < image < task_probe
        assert (sandbox.execs[image]["user"], sandbox.execs[image]["cwd"]) == ("app", "/app")
        assert (sandbox.execs[task_probe]["user"], sandbox.execs[task_probe]["cwd"]) == ("app", "/app")

    @pytest.mark.parametrize(
        ("toml", "user"), [(HEALTHCHECK_TOML, None), (AGENT_USER_HEALTH_TOML, "root")], ids=["root", "agent-user"]
    )
    async def test_root_image_healthchecks_run_as_root_whatever_the_agent_user(
        self, tmp_path, monkeypatch, toml, user
    ):
        record = NO_USER_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK}
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=toml, record=record)
        assert task.entrypoint_user is None
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        image = sandbox.execs[sandbox.index_of("test -f /tmp/healthy")]
        task_probe = sandbox.execs[sandbox.index_of("test -f /app/ready.txt")]
        assert (image["user"], image["cwd"]) == (user, "/app")
        assert (task_probe["user"], task_probe["cwd"]) == (user, "/app")

    async def test_slow_service_passes_when_the_image_healthcheck_absorbs_the_wait(self, tmp_path, monkeypatch):
        # domainbench-24's shape: no image USER, [agent] user = "agent", a task check with retries 3 and no start
        # period, an image HEALTHCHECK with retries 60, a service that listens after five probes. In Harbor's
        # order the image check absorbs the wait and the task check then passes; task-first failed after 3 s.
        ns = 1_000_000_000
        record = NO_USER_CONFIG | {
            "Healthcheck": {"Test": ["CMD", "test", "-f", "/tmp/healthy"], "Interval": ns // 100, "Retries": 60}
        }
        sandbox = EntrypointSandbox(health_failures=5)
        server, task, sandbox = self.entrypoint_server(
            tmp_path, monkeypatch, sandbox=sandbox, toml=AGENT_USER_HEALTH_TOML, record=record
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.health_probes == 7
        image_probes = [i for i, c in enumerate(sandbox.execs) if c["command"] == "test -f /tmp/healthy"]
        assert len(image_probes) == 6 and image_probes[-1] < sandbox.index_of("test -f /app/ready.txt")

    VERIFIER_IMAGE = "registry.example/ds-hello-verifier:abc"

    def separate_server(
        self, tmp_path, monkeypatch, *, toml, agent=None, verifier=None, record=ENTRYPOINT_CONFIG, verifier_record=None
    ):
        """A separate-mode task in build mode: its agent image is ``record``'s, and (D1) its verifier grades in the
        image prepare built from ``tests/Dockerfile``, recorded with ``verifier_record`` (``record`` by default, so
        both images start alike). A declared ``[verifier.environment].docker_image`` in ``toml`` takes precedence."""
        agent = agent or EntrypointSandbox()
        verifier = verifier or EntrypointSandbox()
        # Written in shared mode first: the loader refuses a separate-mode task before its verifier image is recorded.
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent)
        (task.path / "task.toml").write_text(toml)
        (task.path / "tests" / "Dockerfile").write_text(VERIFIER_TESTS_DOCKERFILE)
        record_verifier_build(
            task.path, self.VERIFIER_IMAGE, **(record if verifier_record is None else verifier_record)
        )
        task = build_mode(task, record=record)
        server.config.tasksets["ds"].tasks["hello"] = task.digest

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier

    async def test_separate_verifier_in_the_built_image_runs_its_entrypoint_before_test_sh(
        self, tmp_path, monkeypatch
    ):
        server, task, agent, verifier = self.separate_server(tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML)
        # The loader read the verifier image's record (D1 built it from tests/Dockerfile) beside the task image's.
        assert task.verifier_image == self.VERIFIER_IMAGE and task.entrypoint == ("/entrypoint.sh",)
        assert task.verifier_image_config.entrypoint == ("/entrypoint.sh",)
        assert (task.verifier_image_config.user, task.verifier_image_config.workdir) == ("app", "/app")

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        assert agent.launch() is not None and verifier.launch() is not None
        # The verifier's /logs folders exist before its entrypoint starts, as in the seed; test.sh comes after.
        launched = verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh")
        assert (
            verifier.index_of("mkdir -p /tests /logs/verifier /logs/agent /logs/artifacts")
            < launched
            < verifier.index_of("test.sh")
        )
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("app", "/app")
        # Both launchers were cancelled before their sandbox was stopped.
        assert agent.launch_cancelled_before_stop is True and agent.stopped
        assert verifier.launch_cancelled_before_stop is True and verifier.stopped
        assert not server._sandboxes and not server._sessions

    async def test_built_verifier_image_starts_with_its_own_record_not_the_task_images(self, tmp_path, monkeypatch):
        # Harbor `up --wait`s the image built from tests/Dockerfile: its entrypoint, as its user in its WorkingDir,
        # and its HEALTHCHECK, whatever the agent image's are.
        verifier_record = {
            "Entrypoint": ["/verify.sh"],
            "User": "grader",
            "WorkingDir": "/tests",
            "Healthcheck": IMAGE_HEALTHCHECK,
        }
        verifier = EntrypointSandbox(health_failures=1)
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier, verifier_record=verifier_record
        )
        assert task.image_healthcheck is None and task.verifier_image_config.healthcheck == IMAGE_HEALTHCHECK

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        assert (agent.launch()["user"], agent.launch()["cwd"]) == ("app", "/app")
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("grader", "/tests")
        assert "/verify.sh sh -c 'sleep infinity'" in verifier.run_script()
        assert "/entrypoint.sh" not in verifier.run_script()
        assert verifier.health_probes == 2 and agent.health_probes == 0
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < verifier.index_of("test -f /tmp/healthy")
        assert verifier.index_of("test -f /tmp/healthy") < verifier.index_of("bash /tests/test.sh")
        # A verifier image whose record has no entrypoint starts nothing, while the agent image's still runs.
        quiet = EntrypointSandbox()
        server, task, agent, quiet = self.separate_server(
            tmp_path / "b",
            monkeypatch,
            toml=SEPARATE_IN_TASK_IMAGE_TOML,
            verifier=quiet,
            verifier_record={"Env": [f"PATH={IMAGE_PATH}"], "WorkingDir": "/tests"},
        )
        assert task.verifier_image_config.entrypoint is None
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert agent.launch() is not None and quiet.launch() is None
        assert not any(ENTRYPOINT_DIR in c["command"] for c in quiet.execs)

    async def test_separate_verifier_in_the_task_image_runs_its_entrypoint_before_test_sh(self, tmp_path, monkeypatch):
        # Harbor's prebuilt verifier: no [verifier.environment], so the effective verifier environment is a copy of
        # [environment] with its docker_image, and the verifier grades in the task's own image, whose recorded start
        # (here the pull-mode Dockerfile's base image's) is what Harbor's `up --wait` runs.
        toml = SEPARATE_IN_TASK_IMAGE_TOML.replace("[environment]\n", '[environment]\ndocker_image = "org/app:1"\n')
        agent, verifier = EntrypointSandbox(), EntrypointSandbox()
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            agent,
            toml=toml,
            dockerfile="FROM org/app:1\nWORKDIR /app\n",
            image_config=ENTRYPOINT_CONFIG,
        )
        assert task.image == "org/app:1" and task.verifier_image is None and task.verifier_image_config is None
        assert task.entrypoint == ("/entrypoint.sh",) and (task.entrypoint_user, task.entrypoint_workdir) == (
            "app",
            "/app",
        )

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        assert agent.launch() is not None and verifier.launch() is not None
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("app", "/app")
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < verifier.index_of("test.sh")
        assert verifier.launch_cancelled_before_stop is True and verifier.stopped and not server._sandboxes

    async def test_separate_verifier_in_another_image_gets_no_launcher(self, tmp_path, monkeypatch):
        toml = SEPARATE_IN_TASK_IMAGE_TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n'
        server, task, agent, verifier = self.separate_server(tmp_path, monkeypatch, toml=toml)
        assert _verifier_image(task) == "org/verifier:1" and task.entrypoint == ("/entrypoint.sh",)
        assert task.verifier_image_config is None  # a declared verifier image nothing recorded: no start
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert agent.launch() is not None and verifier.launch() is None
        assert not any(ENTRYPOINT_DIR in c["command"] for c in verifier.execs)

    async def test_verifier_entrypoint_failure_masks_the_sample_and_keeps_the_rollout(self, tmp_path, monkeypatch):
        verifier = EntrypointSandbox(exit_status="3", log="cannot bind port 8069\n")
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier
        )

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == ENTRYPOINT_FAILED_KIND
        assert "exited 3" in result.failure_reason and "cannot bind port 8069" in result.failure_reason
        assert result.response is not None  # the rollout is kept
        assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert verifier.stopped and agent.stopped and not server._sandboxes

    async def test_entrypoint_log_is_collected_in_shared_mode(self, tmp_path, monkeypatch):
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch)
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        logs_dir = Path(response.result.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == sandbox.log
        assert not (logs_dir / "verifier-entrypoint-stdout.txt").exists()

    async def test_entrypoint_logs_are_collected_in_separate_mode(self, tmp_path, monkeypatch):
        agent = EntrypointSandbox(log="agent side setup\n")
        verifier = EntrypointSandbox(log="verifier side setup\n")
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, agent=agent, verifier=verifier
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        logs_dir = Path(response.result.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == "agent side setup\n"
        assert (logs_dir / "verifier-entrypoint-stdout.txt").read_text() == "verifier side setup\n"
        # The agent's log was taken before its sandbox was stopped.
        assert agent.log_downloaded_before_stop is True and agent.stopped

    async def test_entrypoint_log_collection_failure_only_warns(self, tmp_path, monkeypatch, caplog):
        sandbox = EntrypointSandbox(download_fails=True)
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)
        with caplog.at_level("WARNING", logger="environment_servers.harbor.entrypoint"):
            response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.mask_sample is False
        assert "Could not collect the entrypoint log for hello" in caplog.text
        assert not (Path(response.result.verifier_logs_dir) / "entrypoint-stdout.txt").exists()

    async def test_stop_during_the_episode_cancels_the_launcher_before_the_sandbox(self, tmp_path, monkeypatch):
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch)
        seen: list = []

        async def observe():
            session = server._sessions["r1"]
            seen.append((session.entrypoint is not None, session.entrypoint.process.done()))
            await server._stop_sandbox("r1")
            seen.append((sandbox.launch_cancelled_before_stop, sandbox.stopped))

        server.server_client.turn_hooks.append(observe)
        await run_episode(server, task)
        assert seen == [(True, False), (True, True)]

    # -- the exit status after the seed's first probe -------------------------------------------

    async def test_entrypoint_dying_before_the_hand_over_is_still_a_seed_failure(self, tmp_path, monkeypatch):
        # The probe right after the started marker sees nothing (a setup script still importing); the
        # entrypoint dies during the seed's remaining steps and the probe before the hand-over catches it.
        sandbox = EntrypointSandbox(exit_after_exit_probes=1, log="setup.py: missing data file\n")
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "ENTRYPOINT /entrypoint.sh exited 1" in failure.failure_reason
        assert "missing data file" in failure.failure_reason
        probes = [
            i for i, c in enumerate(sandbox.execs) if c["command"].startswith(f"if test -f {ENTRYPOINT_DIR}/exit")
        ]
        # Once after the started marker, once at the end of the seed (the output tail is the last exec).
        assert len(probes) == 2 and probes[-1] == len(sandbox.execs) - 2
        assert sandbox.execs[-1]["command"].startswith("tail -c ")
        assert server.server_client.paths() == []  # never handed to the harness
        assert sandbox.stopped and not server._sandboxes and not server._sessions

    async def test_entrypoint_dying_during_the_rollout_masks_the_sample_when_its_exec_returned(
        self, tmp_path, monkeypatch
    ):
        # Under Harbor the entrypoint is PID 1: its exit kills the container and the trial errors instead of
        # scoring. run.sh finished (the holding exec returned) with a non-zero status: masked, rollout kept.
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=LateExitSandbox())

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == ENTRYPOINT_FAILED_KIND
        assert "ENTRYPOINT /entrypoint.sh exited 1 during the rollout" in result.failure_reason
        assert "database connection lost" in result.failure_reason
        assert result.response is not None  # the rollout is kept
        assert not any("test.sh" in c["command"] for c in sandbox.execs)
        logs_dir = Path(result.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == sandbox.log
        assert (logs_dir / "entrypoint-exit.txt").read_text() == "1\n"
        assert sandbox.stopped and not server._sandboxes

    async def test_exit_file_alone_is_recorded_and_the_sample_still_scored(self, tmp_path, monkeypatch, caplog):
        # The runtime folder is world-writable and the agent had the sandbox: an exit file the holding exec
        # does not corroborate is kept and logged, never an agent-controlled mask.
        sandbox = LateExitSandbox(late_returns=False)
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        with caplog.at_level("ERROR", logger="environment_servers.harbor.entrypoint"):
            response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.reward == 1.0 and result.mask_sample is False
        assert any("test.sh" in c["command"] for c in sandbox.execs)
        assert (Path(result.verifier_logs_dir) / "entrypoint-exit.txt").read_text() == "1\n"
        errors = [r.getMessage() for r in caplog.records if "exit status 1 recorded" in r.getMessage()]
        assert len(errors) == 1 and "did not return (still held)" in errors[0]
        assert sandbox.launch_cancelled_before_stop is True

    async def test_late_exit_zero_warns_and_the_sample_is_scored(self, tmp_path, monkeypatch, caplog):
        sandbox = LateExitSandbox(late_status="0")
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)
        with caplog.at_level("WARNING", logger="environment_servers.harbor.entrypoint"):
            response = await run_episode(server, task)
        result = response.result
        assert result is not None, response.failure
        assert result.reward == 1.0 and result.mask_sample is False
        assert (Path(result.verifier_logs_dir) / "entrypoint-exit.txt").read_text() == "0\n"
        warnings = [r.getMessage() for r in caplog.records if "exited 0 during the rollout" in r.getMessage()]
        assert len(warnings) == 1

    async def test_entrypoint_dying_during_the_rollout_masks_in_separate_mode_too(self, tmp_path, monkeypatch):
        agent = LateExitSandbox()
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, agent=agent
        )

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == ENTRYPOINT_FAILED_KIND
        assert result.response is not None
        assert verifier.launch() is None and not verifier.execs  # no verifier sandbox was needed
        logs_dir = Path(result.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == agent.log
        assert (logs_dir / "entrypoint-exit.txt").read_text() == "1\n"
        assert agent.log_downloaded_before_stop is True and agent.stopped and not server._sandboxes

    async def test_entrypoint_dying_during_the_rollout_masks_a_separate_step_before_its_verifier(
        self, tmp_path, monkeypatch
    ):
        """Round 38 review (P2): a step verified in a sandbox of its own skipped the agent-entrypoint death check the
        single-step paths run, so this fixture scored unmasked through it. The check runs before the collection pass
        stops `main` or the step's verifier sandbox is created, while the agent's sandbox is still up, and masks the
        turn as in shared mode, with the entrypoint log and exit status kept beside the turn's logs."""
        from environment_servers.harbor.tests.test_multi_turn import request_for, wire_verifier_sandboxes, write_steps

        agent = LateExitSandbox()
        toml = TASK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n') + (
            '[verifier.environment]\ndocker_image = "org/verifier:1"\n[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
        )
        write_steps(tmp_path / "datasets" / "ds" / "hello", {"plan": "P", "build": "B"})
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=toml)
        task = build_mode(task)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        assert task.entrypoint == ("/entrypoint.sh",) and [t.verifier_mode for t in task.turns] == ["separate"] * 2
        verifiers, _ = wire_verifier_sandboxes(server, monkeypatch)

        response = await server.run_request(request_for(task))

        result = response.result
        assert result is not None, response.failure
        (turn,) = result.turns  # the masked turn ends the episode
        assert turn.mask_sample is True and turn.failure_kind == ENTRYPOINT_FAILED_KIND
        assert turn.verifier_mode == "separate" and turn.response is not None
        assert "ENTRYPOINT /entrypoint.sh exited 1 during the rollout" in turn.failure_reason
        assert "database connection lost" in turn.failure_reason
        assert result.mask_sample is True and result.stopped_after_turn == "plan"
        # No verifier sandbox was created for the masked turn, and test.sh ran nowhere.
        assert verifiers == [] and not any("test.sh" in c["command"] for c in agent.execs)
        logs_dir = Path(turn.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == agent.log
        assert (logs_dir / "entrypoint-exit.txt").read_text() == "1\n"
        assert agent.log_downloaded_before_stop is True and agent.stopped and not server._sandboxes

    # -- the exec that holds the entrypoint --------------------------------------------------

    async def test_launcher_capped_by_the_provider_is_logged_once_and_the_episode_goes_on(
        self, tmp_path, monkeypatch, caplog
    ):
        # Docker ends the holding exec at exec.default_timeout_s with the process still running.
        sandbox = EntrypointSandbox(
            health_failures=2,
            release_after_probes=1,
            hold_result=SandboxExecResult(stdout="", stderr="", return_code=125, error_type="timeout"),
        )
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, toml=HEALTHCHECK_TOML)

        with caplog.at_level("WARNING", logger="environment_servers.harbor.entrypoint"):
            response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.mask_sample is False
        warnings = [r.getMessage() for r in caplog.records if "the exec holding ENTRYPOINT" in r.getMessage()]
        assert len(warnings) == 1
        assert "hello: the exec holding ENTRYPOINT /entrypoint.sh ended (timeout)" in warnings[0]
        assert sandbox.health_probes == 3 and sandbox.launch_cancelled_before_stop is None

    async def test_health_failure_names_the_launcher_the_provider_ended(self, tmp_path, monkeypatch):
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK | {"Retries": 3}}
        sandbox = EntrypointSandbox(
            health_failures=100,
            release_after_probes=1,
            hold_result=SandboxExecResult(stdout="", stderr="", return_code=1, error_type="sandbox"),
        )
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, record=record)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and not failure.terminal
        assert f"Healthcheck failed 3 times: {IMAGE_SHELL_PROBE!r} (the image HEALTHCHECK)" in failure.failure_reason
        assert "after the exec holding the entrypoint ended (sandbox)" in failure.failure_reason

    # -- the launcher's user and working directory -------------------------------------------

    async def test_entrypoint_runs_as_the_image_user_in_the_image_workdir_not_the_task_toml_ones(
        self, tmp_path, monkeypatch
    ):
        # Image User root (no User recorded), [agent].user and [environment].workdir both override.
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=WORKDIR_TOML, record=NO_USER_CONFIG)
        assert (task.user, task.workdir) == ("agent", "/work")
        assert (task.entrypoint_user, task.entrypoint_workdir) == (None, "/app")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        launch = sandbox.launch()
        assert launch is not None
        assert (launch["user"], launch["cwd"], launch["timeout_s"]) == ("root", "/app", None)
        # A built image holds its WorkingDir: nothing creates it here.
        assert not any(c["command"].startswith("mkdir -p /app") for c in sandbox.execs)

    async def test_agent_user_does_not_move_the_entrypoint_off_the_image_user(self, tmp_path, monkeypatch):
        server, task, sandbox = self.entrypoint_server(
            tmp_path, monkeypatch, toml=AGENT_USER_TOML, record=ENTRYPOINT_CONFIG
        )
        assert (task.user, task.entrypoint_user) == ("agent", "app")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        launch = sandbox.launch()
        assert (launch["user"], launch["cwd"]) == ("app", "/app")

    async def test_root_image_user_without_an_agent_user_uses_the_plain_exec(self, tmp_path, monkeypatch):
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=TASK_TOML, record=NO_USER_CONFIG)
        assert task.user is None and task.entrypoint_user is None
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        launch = sandbox.launch()
        assert (launch["user"], launch["cwd"]) == (None, "/app")

    async def test_numeric_image_user_runs_the_entrypoint_by_name(self, tmp_path, monkeypatch):
        # OpenSandbox switches user by name (`su ... 1000` fails); the uid is mapped once, for the default
        # user and the launcher alike, as the overlay maps a numeric USER.
        record = ENTRYPOINT_CONFIG | {"User": "1000"}
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, record=record)
        assert task.entrypoint_user == "1000"
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.launch()["user"] == "agent" and sandbox.default_user == "agent"
        assert len([c for c in sandbox.execs if c["command"].startswith("getent passwd")]) == 1

    async def test_numeric_image_user_without_a_name_stays_numeric_with_a_warning(self, tmp_path, monkeypatch, caplog):
        sandbox = EntrypointSandbox(passwd={"0": "root"})
        record = ENTRYPOINT_CONFIG | {"User": "1000"}
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, record=record)
        with caplog.at_level("WARNING", logger="environment_servers.harbor.users"):
            response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.launch()["user"] == "1000"
        warnings = [r.getMessage() for r in caplog.records if "USER 1000 names no user" in r.getMessage()]
        assert len(warnings) == 1

    async def test_resolved_dockerfile_workdir_is_created_for_the_entrypoint(self, tmp_path, monkeypatch):
        # A trailing WORKDIR with no RUN after it: `docker build` would have created it in the image, and so does
        # the overlay (after its last RUN line), before the task's workdir, which task.toml points elsewhere.
        dockerfile = "FROM debian:bookworm-slim\nRUN useradd agent\nUSER agent\nWORKDIR /app\n"
        config = {"Env": [f"PATH={IMAGE_PATH}"], "Entrypoint": ["/entrypoint.sh"]}
        toml = TASK_TOML.replace("storage_mb = 10240", 'storage_mb = 10240\nworkdir = "/work"')
        sandbox = EntrypointSandbox()
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, sandbox, dockerfile=dockerfile, image_config=config, toml=toml
        )
        assert (task.workdir, task.entrypoint_workdir, task.entrypoint_user) == ("/work", "/app", "agent")

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        made = sandbox.index_of("test -d /app || (mkdir -p /app && chown agent /app)")
        assert made < sandbox.index_of("mkdir -p /work") < sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh")
        assert sandbox.execs[made]["user"] == "root"
        assert sandbox.launch()["cwd"] == "/app" and sandbox.dirs["/app"] == "agent"

    # -- the separate verifier's start -------------------------------------------------------

    async def test_separate_verifier_never_runs_the_task_healthcheck(self, tmp_path, monkeypatch):
        # Harbor builds the separate verifier from tests/Dockerfile and only `up --wait`s it; run_healthcheck()
        # is for the agent's environment. The task image's own HEALTHCHECK is the substitute, nothing more.
        server, task, agent, verifier = self.separate_server(tmp_path, monkeypatch, toml=SEPARATE_HEALTH_TOML)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        assert agent.health_probes == 1 and verifier.health_probes == 0
        assert verifier.launch() is not None and any("test.sh" in c["command"] for c in verifier.execs)

    async def test_separate_verifier_waits_for_the_image_healthcheck_before_test_sh(self, tmp_path, monkeypatch):
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK}
        verifier = EntrypointSandbox(health_failures=2)
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier, record=record
        )

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        assert verifier.health_probes == 3
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < verifier.index_of("test -f /tmp/healthy")
        assert verifier.index_of("test -f /tmp/healthy") < verifier.index_of("test.sh")
        # The verifier's exit status is probed once more right before test.sh.
        probes = [
            i for i, c in enumerate(verifier.execs) if c["command"].startswith(f"if test -f {ENTRYPOINT_DIR}/exit")
        ]
        assert verifier.index_of("test -f /tmp/healthy") < probes[-1] < verifier.index_of("bash /tests/test.sh")

    async def test_verifier_image_healthcheck_that_never_passes_masks_as_provider_unavailable(
        self, tmp_path, monkeypatch
    ):
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK | {"Retries": 3}}
        verifier = EntrypointSandbox(health_failures=10**6)
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier, record=record
        )

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == failure_kinds.PROVIDER_UNAVAILABLE
        assert "Healthcheck failed 3 times" in result.failure_reason
        assert result.response is not None  # the rollout is kept
        assert verifier.health_probes == 3
        assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert verifier.stopped and agent.stopped and not server._sandboxes

    @pytest.mark.parametrize("passes", [True, False], ids=["healthy", "never-healthy"])
    async def test_compose_agent_with_a_built_verifier_starts_it_by_plain_sandbox_rules(
        self, tmp_path, monkeypatch, passes
    ):
        """The agent runs as a Compose group (the adapter started main's entrypoint and waited for its healthcheck),
        and the verifier is a standalone sandbox prepare built from tests/Dockerfile, as Harbor's ``up --wait`` of the
        verifier image is. Its startup is its own decision: the verifier image's ENTRYPOINT is launched there and its
        HEALTHCHECK gates test.sh, whatever the agent's environment was."""
        agent = EntrypointSandbox()
        verifier = EntrypointSandbox(health_failures=1 if passes else 10**6)
        # Written in shared mode first: the loader refuses a separate-mode task before its verifier image is recorded.
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=HEALTHCHECK_TOML)
        (task.path / "task.toml").write_text(SEPARATE_HEALTH_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text("services:\n  main:\n    image: x\n")
        (task.path / "tests" / "Dockerfile").write_text(VERIFIER_TESTS_DOCKERFILE)
        record_verifier_build(
            task.path,
            self.VERIFIER_IMAGE,
            Env=[f"PATH={IMAGE_PATH}"],
            Entrypoint=["/start-db.sh"],
            User="grader",
            WorkingDir="/tests",
            Healthcheck=IMAGE_HEALTHCHECK | {"Retries": 3},
            Shell=["/bin/bash", "-o", "pipefail", "-c"],
        )
        task = build_mode(task, record=ENTRYPOINT_CONFIG)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        assert task.needs_compose and task.verifier_image == self.VERIFIER_IMAGE
        assert task.verifier_image_config.entrypoint == ("/start-db.sh",)
        compose = FakeCompose({"main": agent})

        async def start_compose(task, episode_key):
            return compose

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_compose", start_compose)
        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        # The seed checks the Compose document against the deployment's provider block before the (stubbed) start.
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        response = await run_episode(server, task)

        # The agent side was the adapter's: no entrypoint launched by the seed, no image probe, the task's check only.
        assert agent.launch() is None and agent.health_probes == 1
        assert agent.execs[agent.index_of("test -f /app/ready.txt")]["command"] == "test -f /app/ready.txt"
        # The verifier's side is a plain sandbox's: its image entrypoint runs as its user in its WorkingDir, and its
        # HEALTHCHECK, through the verifier image's own Shell, is polled before anything is graded.
        assert verifier.launch() is not None and "/start-db.sh sh -c 'sleep infinity'" in verifier.run_script()
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("grader", "/tests")
        probe = "/bin/bash -o pipefail -c 'test -f /tmp/healthy'"
        probes = [i for i, c in enumerate(verifier.execs) if c["command"] == probe]
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < probes[0]
        if passes:
            assert response.failure is None, response.failure
            assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
            assert verifier.health_probes == 2 and probes[-1] < verifier.index_of("bash /tests/test.sh")
        else:
            result = response.result
            assert result is not None, response.failure
            assert result.mask_sample is True and result.failure_kind == failure_kinds.PROVIDER_UNAVAILABLE
            assert f"Healthcheck failed 3 times: {probe!r} (the image HEALTHCHECK)" in result.failure_reason
            assert verifier.health_probes == 3
            assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert compose.stopped and verifier.stopped and not server._sandboxes

    @pytest.mark.parametrize("passes", [True, False], ids=["healthy", "never-healthy"])
    async def test_a_declared_verifier_image_that_is_the_task_image_starts_by_its_record(
        self, tmp_path, monkeypatch, passes
    ):
        """Review (M24): ``[verifier.environment].docker_image`` names the task's own ``[environment].docker_image``,
        whose start (ENTRYPOINT, user, WorkingDir, HEALTHCHECK) is recorded as the pull-mode Dockerfile's base. M20
        keyed the verifier's start on that image identity; M24's rewrite of ``_verifier_runs_task_image`` to the
        configuration shape (no explicit block) lost it, and a declared image had no ``verifier_image_config``, so the
        verifier sandbox ran test.sh with the image's entrypoint never started and its healthcheck never polled. Harbor
        ``up --wait``s a declared verifier image like any other. The loader now reads the declared image's own record
        (``verifier_image_config``), so the start is that image's; the sandbox is still created from the explicit
        block (its workdir and resources), the real ``_create_verifier_sandbox`` here with the provider's sandbox
        replaced, and the block's user default and overlay rules are untouched."""
        toml = SEPARATE_IN_TASK_IMAGE_TOML.replace("[environment]\n", '[environment]\ndocker_image = "org/app:1"\n')
        toml += '\n[verifier.environment]\ndocker_image = "org/app:1"\ncpus = 3\nworkdir = "/grade"\n'
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK | {"Retries": 3}}
        agent = EntrypointSandbox()
        verifier = EntrypointSandbox(health_failures=1 if passes else 10**6)
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, agent, toml=toml, dockerfile="FROM org/app:1\nWORKDIR /app", image_config=record
        )
        assert task.image == task.verifier_image == "org/app:1" and task.verifier_image_config is not None
        assert task.entrypoint == ("/entrypoint.sh",) and task.image_healthcheck == record["Healthcheck"]
        start = _verifier_start(task)
        assert start is not None and start is not task and not _verifier_runs_task_image(task)
        assert (start.image, start.entrypoint, start.overlay) == ("org/app:1", ("/entrypoint.sh",), ())
        assert (start.entrypoint_user, start.entrypoint_workdir) == ("app", "/app")
        assert start.image_healthcheck == record["Healthcheck"]
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        specs = []

        async def start(spec):
            specs.append(spec)

        verifier.start = start
        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", lambda provider_config: verifier)

        response = await run_episode(server, task)

        # Created from [verifier.environment]: the declared image, its workdir and its resources.
        (spec,) = specs
        assert spec.image == "org/app:1" and spec.workdir == "/grade" and spec.resources.cpu == 3
        # Started by the task's record: the image entrypoint as its user in its WorkingDir, its HEALTHCHECK polled
        # (through /bin/sh, the image records no Shell) before anything is graded.
        assert verifier.launch() is not None and "/entrypoint.sh sh -c 'sleep infinity'" in verifier.run_script()
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("app", "/app")
        probes = [i for i, c in enumerate(verifier.execs) if c["command"] == IMAGE_SHELL_PROBE]
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < probes[0]
        if passes:
            assert response.failure is None, response.failure
            assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
            assert verifier.health_probes == 2 and probes[-1] < verifier.index_of("bash /tests/test.sh")
        else:
            result = response.result
            assert result is not None, response.failure
            assert result.mask_sample is True and result.failure_kind == failure_kinds.PROVIDER_UNAVAILABLE
            assert (
                f"Healthcheck failed 3 times: {IMAGE_SHELL_PROBE!r} (the image HEALTHCHECK)" in result.failure_reason
            )
            assert result.response is not None  # the rollout is kept
            assert verifier.health_probes == 3
            assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert agent.stopped and verifier.stopped and not server._sandboxes

    async def test_a_declared_verifier_image_that_is_an_overlay_tasks_base_starts_by_the_base_record(
        self, tmp_path, monkeypatch
    ):
        """Review follow-up (M24): the task is an overlay (``FROM org/app:1`` plus ``RUN`` lines, a ``USER`` and a
        ``WORKDIR`` of its own) and ``[verifier.environment].docker_image`` is that same base image. Harbor
        ``up --wait``s ``org/app:1`` as it is for the verifier, so its entrypoint runs there as the image's own user in
        its own WorkingDir and its HEALTHCHECK gates test.sh. The first fix keyed the start on image identity minus
        the overlay and skipped this case; the start is now the declared image's own record, so the agent's sandbox
        runs the overlay and starts the entrypoint as the Dockerfile's final USER in its WORKDIR, while the verifier's
        gets no RUN line and starts the entrypoint as the record's ``User`` in its ``WorkingDir``."""
        toml = SEPARATE_IN_TASK_IMAGE_TOML + '\n[verifier.environment]\ndocker_image = "org/app:1"\ncpus = 3\n'
        dockerfile = "FROM org/app:1\nRUN useradd -m grader\nWORKDIR /work\nUSER grader\n"
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK | {"Retries": 3}}
        agent, verifier = EntrypointSandbox(), EntrypointSandbox(health_failures=1)
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, agent, toml=toml, dockerfile=dockerfile, image_config=record
        )
        assert task.overlay and task.verifier_image == "org/app:1" and task.verifier_image_config is not None
        assert (task.entrypoint_user, task.entrypoint_workdir) == ("grader", "/work")
        start = _verifier_start(task)
        assert start is not None and not _verifier_runs_task_image(task) and not start.overlay
        assert (start.image, start.entrypoint, start.image_healthcheck) == (
            "org/app:1",
            ("/entrypoint.sh",),
            record["Healthcheck"],
        )
        assert (start.entrypoint_user, start.entrypoint_workdir) == ("app", "/app")
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        specs = []

        async def create(spec):
            specs.append(spec)

        verifier.start = create
        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", lambda provider_config: verifier)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        # The agent: the overlay's RUN line, then the entrypoint as the Dockerfile's final USER in its WORKDIR.
        assert any(c["seen"] and "useradd -m grader" in c["seen"]["command"] for c in agent.execs)
        assert (agent.launch()["user"], agent.launch()["cwd"]) == ("grader", "/work")
        # The verifier: created from [verifier.environment], no RUN line, the image's own entrypoint as its own user
        # in its own WorkingDir, its HEALTHCHECK polled (through /bin/sh) before test.sh.
        (spec,) = specs
        assert spec.image == "org/app:1" and spec.resources.cpu == 3
        assert not any(c["seen"] and "useradd" in c["seen"]["command"] for c in verifier.execs)
        assert verifier.launch() is not None and "/entrypoint.sh sh -c 'sleep infinity'" in verifier.run_script()
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("app", "/app")
        probes = [i for i, c in enumerate(verifier.execs) if c["command"] == IMAGE_SHELL_PROBE]
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < probes[0] < verifier.index_of("bash /tests/test.sh")
        assert verifier.health_probes == 2
        assert agent.stopped and verifier.stopped and not server._sandboxes

    async def test_verifier_entrypoint_dying_before_test_sh_masks_the_sample(self, tmp_path, monkeypatch):
        # Dies after the launcher's probe, during the uploads: the probe right before test.sh catches it.
        verifier = EntrypointSandbox(exit_after_exit_probes=1, log="cannot bind port 8069\n")
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier
        )
        response = await run_episode(server, task)
        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == ENTRYPOINT_FAILED_KIND
        assert "exited 1" in result.failure_reason and "cannot bind port 8069" in result.failure_reason
        assert not any("bash /tests/test.sh" in c["command"] for c in verifier.execs)
        assert verifier.exit_probes == 2 and verifier.stopped and agent.stopped and not server._sandboxes

    @pytest.mark.parametrize(
        ("launch_result", "kind", "fragment"),
        [
            pytest.param(
                SandboxExecResult(stdout="", stderr="", return_code=125, error_type="sandbox"),
                failure_kinds.PROVIDER_UNAVAILABLE,
                "HarborEntrypointUnavailable",
                id="provider",
            ),
            pytest.param(
                SandboxExecResult(stdout="", stderr="sh: not found\n", return_code=127),
                ENTRYPOINT_FAILED_KIND,
                "exited 127 before starting it",
                id="launcher-shell",
            ),
        ],
    )
    async def test_verifier_launcher_that_cannot_start_masks_the_sample(
        self, tmp_path, monkeypatch, launch_result, kind, fragment
    ):
        verifier = EntrypointSandbox(launch_result=launch_result)
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier
        )
        response = await run_episode(server, task)
        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == kind
        assert fragment in result.failure_reason
        assert result.response is not None and agent.launch() is not None
        assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert verifier.stopped and agent.stopped and not server._sandboxes
