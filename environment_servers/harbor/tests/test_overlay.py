# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the Dockerfile overlay applied at seed: RUN, WORKDIR, ENV and USER replay, overlay
failures and retries, and the overlay's interaction with Compose groups and the separate
verifier.
"""

import asyncio
import json
import shlex
import shutil
import subprocess

import pytest

from environment_servers.harbor.overlay import overlay_shell_command, overlay_workdir_command
from environment_servers.harbor.tests.conftest import (
    COMPOSE_YAML,
    HEALTHCHECK_TOML,
    IMAGE_CONFIGS,
    IMAGE_PATH,
    OVERLAY_DOCKERFILE,
    SEPARATE_IN_TASK_IMAGE_TOML,
    TASK_TOML,
    FakeCompose,
    FakeSandbox,
    OverlaySandbox,
    make_server,
    run_episode,
    write_image_config,
)
from environment_servers.harbor.verifier import _verifier_image
from nemo_gym.sandbox import SandboxExecResult, SandboxSpec, cpu_cap_env
from nemo_gym.tasks.harbor import load_task
from nemo_gym.tasks.harbor.dockerfile import OverlayRun
from nemo_gym.tasks.harbor.errors import HarborTaskError


# The CPU-count hints every sandbox of a `cpus = 1` task (TASK_TOML) is created with, under the task's own variables.
HINTS = cpu_cap_env(1)


class TestOverlay:
    def overlay_server(
        self,
        tmp_path,
        monkeypatch,
        *,
        dockerfile=OVERLAY_DOCKERFILE,
        toml=TASK_TOML,
        sandbox=None,
        image_config=None,
        provider="opensandbox",
        **config_overrides,
    ):
        sandbox = sandbox or OverlaySandbox()
        server, task, _, created = make_server(
            tmp_path, monkeypatch, sandbox, dockerfile=dockerfile, image_config=image_config, **config_overrides
        )
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {provider: {}}}
        )

        async def create(task, workdir, *, episode_key=None):
            # The sandbox starts with the spec's environment, as the provider's container would.
            sandbox.spec = server._sandbox_spec(task, workdir)
            sandbox.spec_env = dict(sandbox.spec.env)
            sandbox.created_in(workdir)
            created.append((task.task_id, workdir))
            return sandbox

        monkeypatch.setattr(server, "_create_sandbox", create)
        return server, task, sandbox, created

    def test_shell_command_exports_literal_values_and_only_enters_the_workdir(self):
        step = OverlayRun(
            command="make all", workdir="/app src", env={"PATH": "/opt/bin:/usr/bin", "Q": 'say "hi" `x` $NOT'}
        )
        # Values are already resolved: they are quoted so nothing expands again, `$NOT` included.
        assert overlay_shell_command(step) == (
            "export PATH=/opt/bin:/usr/bin; export Q='say \"hi\" `x` $NOT'; cd '/app src' || exit 1; "
            "exec /bin/sh -c 'make all'"
        )
        # Keys the sandbox was created with that this step does not have yet are unset first.
        assert overlay_shell_command(step, {"PATH": "/x", "LATE": "1", "ZED": "2"}) == (
            "unset LATE; unset ZED; export PATH=/opt/bin:/usr/bin; export Q='say \"hi\" `x` $NOT'; "
            "cd '/app src' || exit 1; exec /bin/sh -c 'make all'"
        )
        assert overlay_shell_command(OverlayRun(command="true")) == "cd / || exit 1; exec /bin/sh -c true"

    def test_workdir_command_creates_as_root_and_hands_the_directory_to_the_user(self):
        assert overlay_workdir_command("/app", None) == "mkdir -p /app"
        assert overlay_workdir_command("/app", "root") == overlay_workdir_command("/app", "0") == "mkdir -p /app"
        # root with a group of its own is not plain root: the directory gets that group, as in a build.
        assert overlay_workdir_command("/app", "0:0") == "test -d /app || (mkdir -p /app && chown 0:0 /app)"
        assert overlay_workdir_command("/home/agent/w", "agent") == (
            "test -d /home/agent/w || (mkdir -p /home/agent/w && chown agent /home/agent/w)"
        )
        assert overlay_workdir_command("/a b", "agent:staff") == (
            "test -d '/a b' || (mkdir -p '/a b' && chown agent:staff '/a b')"
        )

    async def test_run_lines_see_the_environment_directory_and_user_of_their_point_in_the_dockerfile(
        self, tmp_path, monkeypatch
    ):
        server, task, sandbox, created = self.overlay_server(tmp_path, monkeypatch, toml=HEALTHCHECK_TOML)
        assert len(task.overlay) == 3 and task.user == "agent"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        # An overlay task's sandbox starts without the task workdir: the overlay's WORKDIR step makes it.
        assert created == [("hello", None)]
        # The sandbox is created with the Dockerfile's final environment, resolved: no live `$` reaches the provider.
        # The CPU-count hints sit under it (the task declares `cpus = 1`); a RUN line keeps them, they are not
        # part of the Dockerfile environment the per-line `unset` restores.
        assert sandbox.spec.env == HINTS | {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "MSG": "hi there"}
        assert not any("$" in value for value in sandbox.spec.env.values())
        first = sandbox.run("apt-get update")
        second = sandbox.run("echo ok > ready.txt")
        third = sandbox.run("as agent")
        # The line before `ENV PATH=...` sees the image's PATH and no MSG; the lines after see the extended PATH.
        extended = HINTS | {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "MSG": "hi there"}
        assert first["seen"] == {
            "command": "apt-get update && apt-get install -y curl",
            "env": HINTS | {"PATH": IMAGE_PATH},
            "user": "root",
            "cwd": "/",
        }
        assert second["seen"] == {
            "command": "mkdir -p /opt/tools/bin && echo ok > ready.txt",
            "env": extended,
            "user": "root",
            "cwd": "/app",
        }
        assert third["seen"] == {
            "command": 'echo "as agent" > /tmp/who',
            "env": extended,
            "user": "agent",
            "cwd": "/app",
        }
        # Lines before `USER agent` run as root. The image's own default user is root, but the sandbox was
        # created with the Dockerfile's final USER (agent) as its exec default, so root is asked for explicitly.
        assert task.sandbox_user is None and task.user == "agent"
        assert [c["user"] for c in (first, second, third)] == ["root", "root", "agent"]
        assert all(c["cwd"] == "/" and c["env"] is None for c in (first, second, third))
        commands = [c["command"] for c in sandbox.execs]
        # /app is made as root when the first RUN after `WORKDIR /app` comes up, once, then only entered.
        assert commands[:4] == [first["command"], "mkdir -p /app", second["command"], third["command"]]
        assert first["command"] == (
            f"unset MSG; export PATH={IMAGE_PATH}; cd / || exit 1; exec /bin/sh -c {shlex.quote(first['seen']['command'])}"
        )
        assert sandbox.execs[1]["user"] == "root" and commands.count("mkdir -p /app") == 1
        # The overlay shares the task's build budget: the first line sees all of it, later ones what is left.
        assert first["timeout_s"] == pytest.approx(task.config.environment.build_timeout_sec, abs=1)
        assert second["timeout_s"] <= first["timeout_s"] and third["timeout_s"] <= second["timeout_s"]
        # Only then does Gym prepare the agent's workdir, poll the healthcheck and hand the sandbox over.
        assert commands[4] == "mkdir -p /app && chown agent /app" and sandbox.dirs["/app"] == "agent"
        assert commands.index("test -f /app/ready.txt") > commands.index(third["command"])
        # The harness got its session only after all of it.
        assert server.server_client.seed_bodies()

    async def test_root_image_without_user_keeps_the_plain_exec_path(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nWORKDIR /app\nRUN true\nRUN false\n"
        )
        assert (await run_episode(server, task)).failure is None
        prepare, *runs = sandbox.execs[:3]
        assert prepare["command"] == "mkdir -p /app"
        # The image's own PATH is exported again, harmlessly: a step exports its whole resolved environment.
        assert [c["command"] for c in runs] == [
            f"export PATH={IMAGE_PATH}; cd /app || exit 1; exec /bin/sh -c true",
            f"export PATH={IMAGE_PATH}; cd /app || exit 1; exec /bin/sh -c false",
        ]
        assert [c["seen"]["cwd"] for c in runs] == ["/app", "/app"]
        # No USER anywhere: root is the image's default user, so no override is needed (every provider supports this).
        assert [c["user"] for c in (prepare, *runs)] == [None, None, None]

    async def test_workdir_after_user_is_made_as_root_and_owned_by_the_user(self, tmp_path, monkeypatch):
        # The fake enforces what a container would: a non-root user cannot create under a root-owned directory.
        assert OverlaySandbox()._mkdir("/home/agent/work", "agent").return_code == 1
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER agent\nWORKDIR /home/agent/work\nRUN touch made\n"
        )
        assert task.user == "agent" and task.workdir == "/home/agent/work"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        prepare, run = sandbox.execs[:2]
        assert prepare["command"] == (
            "test -d /home/agent/work || (mkdir -p /home/agent/work && chown agent /home/agent/work)"
        )
        assert prepare["user"] == "root"  # the sandbox's exec default is the Dockerfile's final USER (agent)
        assert sandbox.dirs["/home/agent/work"] == "agent" and sandbox.dirs["/home/agent"] == "root"
        assert run["seen"] == {
            "command": "touch made",
            "env": HINTS | {"PATH": IMAGE_PATH},
            "user": "agent",
            "cwd": "/home/agent/work",
        }

    async def test_numeric_user_is_resolved_to_its_name_in_the_image(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 1000\nWORKDIR /home/agent/work\nRUN id\nRUN id -g\n"
        )
        assert task.user == "1000"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        lookup, prepare, first, second = sandbox.execs[:4]
        assert lookup["command"] == "getent passwd 1000 | cut -d: -f1" and lookup["user"] == "root"
        # The name serves the directory's owner and both lines; the lookup happens once.
        assert prepare["command"].endswith("chown agent /home/agent/work)") and prepare["user"] == "root"
        assert [c["seen"]["user"] for c in (first, second)] == ["agent", "agent"]
        assert sum("getent" in c["command"] for c in sandbox.execs) == 1
        # The resolved name, not the uid, becomes the sandbox's default user after the overlay.
        assert sandbox.default_user == "agent"

    async def test_root_steps_on_a_non_root_image_override_the_image_user(self, tmp_path, monkeypatch):
        # jupyter/base-notebook style: the image runs as jovyan, the Dockerfile switches to root and stays there.
        # The sandbox's default exec user is the image's, so every root step needs the override, including the
        # WORKDIR directory and the agent's workdir, both made by root.
        sandbox = OverlaySandbox(image_user="jovyan")
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM img\nUSER root\nWORKDIR /opt/tools\nRUN apt-get install -y gcc\n",
            sandbox=sandbox,
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "jovyan"},
        )
        assert task.user == "root" and task.sandbox_user == "jovyan"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        # The overlay's WORKDIR step and the agent workdir preparation both create it, as root.
        assert [c["user"] for c in sandbox.execs if "mkdir -p /opt/tools" in c["command"]] == ["root", "root"]
        run = sandbox.run("apt-get install")
        assert run["user"] == "root" and run["seen"]["user"] == "root" and run["seen"]["cwd"] == "/opt/tools"
        assert sandbox.dirs["/opt/tools"] == "root"

    async def test_steps_on_a_non_root_image_without_a_user_line_run_as_the_image_user(self, tmp_path, monkeypatch):
        sandbox = OverlaySandbox(image_user="jovyan")
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM img\nWORKDIR /home/jovyan/work\nRUN touch made\n",
            sandbox=sandbox,
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "jovyan"},
        )
        assert task.user == "jovyan" and task.sandbox_user == "jovyan"
        sandbox.dirs["/home/jovyan"] = "jovyan"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        prepare, run = sandbox.execs[:2]
        # The directory is still made as root (with the override) and handed to the image user; the line runs
        # as that user with no override, as it would in a build.
        assert prepare["user"] == "root" and prepare["command"].endswith("chown jovyan /home/jovyan/work)")
        assert run["user"] == "jovyan" and run["seen"]["user"] == "jovyan"
        assert sandbox.dirs["/home/jovyan/work"] == "jovyan"

    async def test_a_numeric_user_without_a_passwd_entry_is_passed_through_as_a_uid(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242\nRUN id\n"
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        lookup, run = sandbox.execs[:2]
        # The uid goes to the exec as a number: Docker's --user and OpenSandbox's uid take it, where the string
        # "4242" would be looked up as an account name by the su-backed providers and fail.
        assert lookup["command"] == "getent passwd 4242 | cut -d: -f1"
        assert run["user"] == 4242 and isinstance(run["user"], int) and run["seen"]["user"] == "4242"
        # A uid:gid pair stays as written, on the line and as the default: only Docker's --user takes the pair.
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242:4242\nRUN id\n", provider="docker"
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.execs[1]["user"] == "4242:4242" and sandbox.default_user == "4242:4242"
        # Every other provider switches user by name only: the pair is refused before the sandbox exists.
        server, task, sandbox, created = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242:4242\nRUN id\n"
        )
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert (
            "user '4242:4242' names group '4242'; the 'opensandbox' sandbox provider"
            in response.failure.failure_reason
        )
        assert created == [] and sandbox.execs == []
        # A provider that refuses the uid fails the line itself, with its message: the Dockerfile does not apply.
        sandbox = OverlaySandbox(
            failing="id", failure=SandboxExecResult(stdout="", stderr="unable to find user 4242\n", return_code=125)
        )
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242\nRUN id\n", sandbox=sandbox
        )
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "RUN line 1/1 'id' exited 125" in response.failure.failure_reason
        assert "unable to find user 4242" in response.failure.failure_reason and sandbox.stopped

    @pytest.mark.parametrize("user", ["root:1234", "0:0"])
    async def test_root_with_a_group_keeps_the_group_on_its_workdir_and_on_its_lines_on_docker(
        self, tmp_path, monkeypatch, user
    ):
        """Docker's --user takes `root:1234`, so a RUN line under it runs with gid 1234 as a build would; root's own
        group (`0:0`) is plain root. Running the line as plain root would give the files it creates gid 0."""
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile=f"FROM img\nUSER {user}\nWORKDIR /srv/data\nRUN id -g\n",
            provider="docker",
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        prepare, run = sandbox.execs[:2]
        # The directory gets the group a build would give it, made as root (the plain exec path on a root image).
        assert prepare["command"] == f"test -d /srv/data || (mkdir -p /srv/data && chown {user} /srv/data)"
        assert prepare["user"] is None and sandbox.dirs["/srv/data"] == user
        # The line runs as the pair as written (Docker's `--user root:1234`); root's own group stays plain root.
        expected = "root:1234" if user == "root:1234" else None
        assert run["user"] == expected and run["seen"]["user"] == (expected or "root")
        # The default Docker then applies to every exec keeps the group too.
        assert sandbox.default_user == expected
        # On a non-root image the pair is still asked for as written; plain root takes the root override.
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile=f"FROM img\nUSER {user}\nRUN id -g\n",
            sandbox=OverlaySandbox(image_user="jovyan"),
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "jovyan"},
            provider="docker",
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.execs[0]["user"] == (expected or "root") and sandbox.execs[0]["seen"]["user"] == (
            expected or "root"
        )
        assert sandbox.default_user == (expected or "root")

    @pytest.mark.parametrize("user", ["root:1234", "0:0"])
    async def test_root_with_a_group_off_docker_is_refused_unless_it_is_roots_own_group(
        self, tmp_path, monkeypatch, user
    ):
        # OpenSandbox cannot run as `root:1234` (it would run as gid 0 where Harbor's Docker gives gid 1234), so
        # the task is refused before its sandbox exists; `0:0` names no group Docker's --user would add.
        server, task, sandbox, created = self.overlay_server(
            tmp_path, monkeypatch, dockerfile=f"FROM img\nUSER {user}\nWORKDIR /srv/data\nRUN id -g\n"
        )
        response = await run_episode(server, task)
        if user == "0:0":
            assert response.failure is None, response.failure
            assert sandbox.default_user is None and sandbox.dirs["/srv/data"] == "0:0"
        else:
            assert response.failure.stage == "seed" and response.failure.terminal
            assert "user 'root:1234' names group '1234'; the 'opensandbox'" in response.failure.failure_reason
            assert created == [] and sandbox.execs == []

    @pytest.mark.parametrize("user", ["root:1234", "app:1234", "1000:1000"])
    async def test_a_run_line_under_a_user_with_a_group_is_refused_off_docker_before_the_sandbox_exists(
        self, tmp_path, monkeypatch, user
    ):
        """The pair of an intermediate USER (the final USER is plain root here) is refused the same way: on a
        provider that switches user by name the line would run with the group lost or fail half-way through the
        overlay with `su`'s message, so the task is refused whole, before any sandbox is created."""
        dockerfile = f"FROM img\nUSER {user}\nRUN id -g\nUSER root\nRUN id -g\n"
        server, task, sandbox, created = self.overlay_server(tmp_path, monkeypatch, dockerfile=dockerfile)
        assert task.user == "root" and [step.user for step in task.overlay] == [user, "root"]
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        group = user.partition(":")[2]
        assert f"user {user!r} names group {group!r}; the 'opensandbox'" in response.failure.failure_reason
        assert created == [] and sandbox.execs == []
        # Docker takes the pair, on the line as written (a uid by the name the image gives it, as for a bare uid),
        # and the final USER root leaves the default alone.
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, dockerfile=dockerfile, provider="docker")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        first, second = [call for call in sandbox.execs if call.get("seen") and "id -g" in call["seen"]["command"]]
        assert first["user"] == ("agent:1000" if user == "1000:1000" else user)
        assert second["user"] is None and sandbox.default_user is None

    async def test_the_final_workdir_belongs_to_the_user_of_its_instruction(self, tmp_path, monkeypatch):
        # node:20 style: USER node, then WORKDIR, then a RUN that writes there as node. A sandbox created with the
        # task workdir already has it, root-owned, and the RUN line fails; the overlay must make it instead.
        sandbox = OverlaySandbox(passwd={"0": "root", "1000": "node"})
        sandbox.dirs["/home/node"] = "node"
        server, task, sandbox, created = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM node:20\nUSER node\nWORKDIR /home/node/app\nRUN npm init -y\n",
            sandbox=sandbox,
        )
        assert task.user == "node" and task.workdir == "/home/node/app"
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert created == [("hello", None)]
        run = sandbox.run("npm init")
        assert run["seen"]["user"] == "node" and run["seen"]["cwd"] == "/home/node/app"
        assert run["cwd_owner"] == "node", "the directory was root-owned when the RUN line ran"
        prepare = next(c for c in sandbox.execs if c["command"].endswith("chown node /home/node/app)"))
        assert sandbox.execs.index(prepare) < sandbox.execs.index(run)
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["workdir"] == "/home/node/app"

    async def test_the_compose_main_service_of_an_overlay_task_starts_without_the_task_workdir_too(
        self, tmp_path, monkeypatch
    ):
        # The same node:20 shape as a Compose task: the main service's spec must not name the task workdir
        # either, or the provider creates it root-owned before the overlay's WORKDIR step can.
        sandbox = OverlaySandbox(passwd={"0": "root", "1000": "node"})
        sandbox.dirs["/home/node"] = "node"
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            sandbox,
            dockerfile="FROM node:20\nUSER node\nWORKDIR /home/node/app\nRUN npm init -y\n",
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        write_image_config(task.path.parent, "node:20", Env=[f"PATH={IMAGE_PATH}"])
        task = load_task(task.path)
        assert task.needs_compose and task.overlay and task.workdir == "/home/node/app" and task.user == "node"
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        specs: dict[str, SandboxSpec] = {}

        class CapturingCompose(FakeCompose):
            def __init__(self, provider_config, path, *, service_specs, timeout_s, service_setup, **kwargs):
                super().__init__({"main": sandbox, "api": FakeSandbox(), "kafka": FakeSandbox()})
                specs.update(service_specs)
                self.setup = dict(service_setup)
                # What a provider does with the spec's workdir: it exists, root-owned, before anything runs.
                sandbox.created_in(service_specs["main"].workdir)

            async def start(self):
                # As the adapter does: the caller's setup (the overlay) runs on main once it exists.
                for name, setup in self.setup.items():
                    await setup(self.services[name])

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", CapturingCompose)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        # Neither main nor the sidecars derived from it start in the task workdir.
        assert specs["main"].workdir is None and specs["api"].workdir is None
        run = sandbox.run("npm init")
        assert run["seen"]["user"] == "node" and run["seen"]["cwd"] == "/home/node/app"
        assert run["cwd_owner"] == "node", "the directory was root-owned when the RUN line ran"
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["workdir"] == "/home/node/app"

    async def test_every_workdir_instruction_is_created_in_order_even_without_a_run_after_it(
        self, tmp_path, monkeypatch
    ):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM img\nWORKDIR /tmp/a\nWORKDIR /\nRUN test -d /tmp/a\nUSER agent\nWORKDIR /home/agent/last\n",
        )
        assert [(w.path, w.user, w.before) for w in task.overlay_workdirs] == [
            ("/tmp/a", None, 0),
            ("/", None, 0),
            ("/home/agent/last", "agent", 1),
        ]
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        commands = [c["command"] for c in sandbox.execs]
        run = sandbox.run("test -d /tmp/a")
        # /tmp/a was made (as root, the user of its instruction) before the RUN line that checks it...
        assert commands.index("mkdir -p /tmp/a") < sandbox.execs.index(run) and sandbox.dirs["/tmp/a"] == "root"
        # ...and the WORKDIR after the last RUN line exists too, owned by agent, before the workdir preparation.
        trailing = next(c for c in sandbox.execs if c["command"].endswith("chown agent /home/agent/last)"))
        assert sandbox.execs.index(run) < sandbox.execs.index(trailing) and sandbox.dirs["/home/agent/last"] == "agent"
        assert commands.index("mkdir -p /home/agent/last && chown agent /home/agent/last") > sandbox.execs.index(
            trailing
        )

    @pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sh") is None, reason="sh and bash are required")
    def test_an_inherited_shell_runs_the_line_and_exec_form_bypasses_it(self):
        pipefail = OverlayRun(command="false | true", shell=("/bin/bash", "-o", "pipefail", "-c"))
        line = overlay_shell_command(pipefail)
        assert line == "cd / || exit 1; exec /bin/bash -o pipefail -c 'false | true'"
        # The build failed this line under bash -o pipefail; the provider's sh would have passed it.
        assert subprocess.run(["sh", "-c", line]).returncode != 0
        assert subprocess.run(["sh", "-c", overlay_shell_command(OverlayRun(command="false | true"))]).returncode == 0
        exec_form = OverlayRun(command=shlex.join(["bash", "-lc", "echo hi"]), shell=None)
        assert overlay_shell_command(exec_form) == "cd / || exit 1; exec bash -lc 'echo hi'"
        assert subprocess.run(["sh", "-c", overlay_shell_command(exec_form)], capture_output=True).stdout == b"hi\n"

    @pytest.mark.skipif(
        shutil.which("bash") is None or shutil.which("dash") is None, reason="dash and bash are required"
    )
    def test_the_default_shell_is_invoked_explicitly_so_the_line_runs_under_the_images_sh(self):
        """The provider's exec runs the line under bash where the image has it. A RUN written for the image's
        `/bin/sh` (dash: `[[` is not a builtin) must fail as it did in the build, so the default shell is named."""
        line = overlay_shell_command(OverlayRun(command="[[ -d /tmp ]]"))
        assert line == "cd / || exit 1; exec /bin/sh -c '[[ -d /tmp ]]'"
        # Stand the image's /bin/sh in with dash whatever this host's sh is: the line fails under it...
        dash_line = line.replace("exec /bin/sh -c", f"exec {shutil.which('dash')} -c")
        assert subprocess.run(["bash", "-c", dash_line], capture_output=True).returncode != 0
        # ...while the unwrapped command the provider's bash used to get would have passed.
        assert subprocess.run(["bash", "-c", "cd / || exit 1; [[ -d /tmp ]]"]).returncode == 0

    async def test_failing_run_is_a_terminal_task_error_with_the_end_of_its_output(self, tmp_path, monkeypatch):
        stdout = "\n".join(f"line {number}" for number in range(1, 401)) + "\n"
        sandbox = OverlaySandbox(
            failing="apt-get install",
            failure=SandboxExecResult(stdout=stdout, stderr="E: no package\n", return_code=100),
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "seed" and failure.terminal
        assert "Dockerfile overlay failed for 'hello'" in failure.failure_reason
        assert "RUN line 1/3 'apt-get update && apt-get install -y curl' exited 100" in failure.failure_reason
        # The tail keeps the end of a long log, where the error is, and drops the start.
        assert "line 400\nE: no package" in failure.failure_reason and "line 1\nline 2\n" not in failure.failure_reason
        # Nothing after the failing line ran, the sandbox is gone and the harness was never engaged.
        assert len(sandbox.execs) == 1 and sandbox.stopped and not server._sandboxes
        assert server.server_client.calls == []

    async def test_provider_failure_during_a_run_is_retriable(self, tmp_path, monkeypatch):
        sandbox = OverlaySandbox(
            failing="apt-get install",
            failure=SandboxExecResult(stdout=None, stderr="connection reset", return_code=-1, error_type="sandbox"),
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "seed" and not failure.terminal
        assert "Could not set up sandbox for 'hello'" in failure.failure_reason
        assert (
            "RUN line 1/3 'apt-get update && apt-get install -y curl' could not run in the sandbox: sandbox"
            in failure.failure_reason
        )
        assert len(sandbox.execs) == 1 and sandbox.stopped
        assert server.server_client.calls == []

    async def test_timed_out_run_names_the_budget(self, tmp_path, monkeypatch):
        sandbox = OverlaySandbox(
            failing="echo ok", failure=SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="timeout")
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "RUN line 2/3" in response.failure.failure_reason
        assert "exceeded [environment].build_timeout_sec=600" in response.failure.failure_reason
        assert sandbox.stopped

    async def test_exhausted_budget_skips_the_remaining_lines(self, tmp_path, monkeypatch):
        class SlowSandbox(OverlaySandbox):
            async def exec(self, command, **kwargs):
                await asyncio.sleep(0.05)  # longer than the whole build budget below
                return await super().exec(command, **kwargs)

        toml = TASK_TOML.replace("cpus = 1", "build_timeout_sec = 0.01\ncpus = 1")
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, toml=toml, sandbox=SlowSandbox())
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "RUN line 2/3" in response.failure.failure_reason
        assert "build_timeout_sec=0.01 is used up" in response.failure.failure_reason
        assert len(sandbox.execs) == 1 and sandbox.stopped

    def test_sandbox_ttls_stay_episode_sized_when_there_is_an_overlay(self, tmp_path, monkeypatch):
        server, task, _, _ = self.overlay_server(tmp_path, monkeypatch)
        config, slack = task.config, server.config.sandbox_ttl_slack_s
        build, ready = config.environment.build_timeout_sec, server.config.sandbox_ready_timeout_s
        cleanup = server.config.cleanup_timeout_seconds
        assert build == 600 and task.overlay
        # The agent sandbox lives for the task's whole episode budget: one sandbox start, the overlay, harness
        # setup, the agent and the shared verifier with its grace; the overlay needs no budget of its own.
        budget = ready + build + server.config.harness_setup_allowance_s
        budget += config.agent.timeout_sec + config.verifier.timeout_sec + server.config.verifier_grace_s
        assert server._task_budget_s(task) == budget
        assert server._sandbox_spec(task, "/app").ttl_s == budget + cleanup + slack

        # A separate verifier in the task's own image: Harbor's effective verifier environment is a copy of
        # [environment] carrying its docker_image, a prebuilt image that resolves no overlay (the loader admits the
        # task-image shape with a declared image only), so no build budget is added and the verifier starts from the
        # task image with the agent's environment.
        toml = SEPARATE_IN_TASK_IMAGE_TOML.replace(
            "[environment]\n", '[environment]\ndocker_image = "debian:bookworm-slim"\n'
        )
        server, task, _, _ = self.overlay_server(tmp_path, monkeypatch, toml=toml)
        assert not task.overlay and task.verifier_image is None
        # No overlay, so no build budget anywhere; the separate verifier adds its own sandbox start.
        separate_budget = budget - build + ready
        assert server._task_budget_s(task) == separate_budget
        assert server._sandbox_spec(task, "/app").ttl_s == separate_budget + cleanup + slack
        specs = []

        class RecordingSandbox:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                specs.append(spec)

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", RecordingSandbox)
        asyncio.run(server._create_verifier_sandbox(task))
        (verifier_spec,) = specs
        # The verifier gets the same episode-sized lifetime: `timeout_sec + slack` would not clear a provider's
        # renew interval (TB4's 1800 s), which masks every separate-mode episode (M10 smVVP).
        agent_spec = server._sandbox_spec(task, "/app")
        assert verifier_spec.ttl_s == agent_spec.ttl_s >= 1800 + config.verifier.timeout_sec
        assert verifier_spec.image == task.image == "debian:bookworm-slim"
        assert verifier_spec.env == agent_spec.env

    def test_forced_separate_verifier_of_an_overlay_task_starts_without_the_workdir(self, tmp_path, monkeypatch):
        """A provider pre-creates the spec's workdir root-owned. The agent's sandbox of an overlay task is created
        without it so the overlay's WORKDIR step makes it for the USER in effect; the verifier sandbox that
        replays the same overlay must be created the same way, or the RUN under USER node gets Permission denied."""
        with_artifacts = TASK_TOML.replace("[verifier]\n", 'artifacts = ["/home/node/app/out.txt"]\n\n[verifier]\n', 1)
        dockerfile = "FROM img\nUSER node\nWORKDIR /home/node/app\nRUN touch out.txt\n"
        server, task, sandbox, created = self.overlay_server(
            tmp_path, monkeypatch, dockerfile=dockerfile, toml=with_artifacts, separate_verifier_for_artifacts=True
        )
        assert task.overlay and task.workdir == "/home/node/app" and server._separate_verifier(task)
        specs = []

        class RecordingSandbox:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                specs.append(spec)

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandbox", RecordingSandbox)
        asyncio.run(server._create_verifier_sandbox(task))
        (verifier_spec,) = specs
        assert verifier_spec.workdir is None and verifier_spec.image == task.image
        # The fake provider would otherwise have made the directory root-owned before the overlay's WORKDIR step.
        sandbox.created_in(verifier_spec.workdir)
        assert "/home/node/app" not in sandbox.dirs
        # A task without an overlay keeps the agent's working directory for its verifier, as before.
        specs.clear()
        toml = with_artifacts.replace(
            "[environment]\n", '[environment]\ndocker_image = "org/app:1"\nworkdir = "/home/node/app"\n'
        )
        server, task, _, _ = self.overlay_server(
            tmp_path / "plain", monkeypatch, dockerfile=dockerfile, toml=toml, separate_verifier_for_artifacts=True
        )
        assert not task.overlay and task.workdir == "/home/node/app"
        asyncio.run(server._create_verifier_sandbox(task))
        assert specs[0].workdir == "/home/node/app"

    async def test_forced_separate_verifier_of_an_overlay_task_runs_test_sh_in_the_task_workdir(
        self, tmp_path, monkeypatch
    ):
        """The verifier sandbox of a forced-separate overlay task is created without the task workdir (the test
        above), so its default directory is the base image's own, `/` for `FROM ubuntu:24.04`, and replaying the
        overlay does not change it. A test.sh that checks a relative path (`result.txt` under `WORKDIR /app`) would
        score a correct artifact 0 unless test.sh is sent to the task's workdir, as the shared verifier's exec is."""
        with_artifacts = TASK_TOML.replace("[verifier]\n", 'artifacts = ["/app/result.txt"]\n\n[verifier]\n', 1)
        dockerfile = "FROM img\nWORKDIR /app\nRUN echo ok > result.txt\n"
        server, task, _, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile=dockerfile, toml=with_artifacts, separate_verifier_for_artifacts=True
        )
        assert task.overlay and task.workdir == "/app" and server._separate_verifier(task)

        class RelativeTestsVerifier(OverlaySandbox):
            """Grades as a test.sh reading `result.txt` relative to where it runs would: 1 in /app, 0 anywhere else."""

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if "test.sh" in command and "timeout" in command:
                    self.verifier_files = {"reward.txt": "1\n" if cwd == "/app" else "0\n"}
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        verifier = RelativeTestsVerifier()

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.verifier_mode == "separate" and response.result.reward == 1.0
        (run,) = [c for c in verifier.execs if "test.sh" in c["command"] and "timeout" in c["command"]]
        assert run["cwd"] == "/app"
        # The overlay's WORKDIR step made the directory in the verifier sandbox before test.sh ran there.
        assert verifier.run("echo ok > result.txt")["seen"]["cwd"] == "/app" and "/app" in verifier.dirs

    async def test_forced_separate_verifier_of_an_overlay_task_makes_the_environment_workdir_before_test_sh(
        self, tmp_path, monkeypatch
    ):
        """`task.workdir` is `[environment].workdir` when task.toml sets one and the Dockerfile's final WORKDIR only
        otherwise, while the overlay makes the Dockerfile's directories alone. The agent's sandbox gets the task
        workdir from the seed (`_prepare_workdir`); the verifier sandbox that replays the overlay has to make it the
        same way before test.sh is sent there, since a Docker exec into a missing directory fails before test.sh
        starts, where the base image's directory at least ran it."""
        toml = TASK_TOML.replace("[verifier]\n", 'artifacts = ["/app/result.txt"]\n\n[verifier]\n', 1)
        toml = toml.replace("[environment]\n", '[environment]\nworkdir = "/work"\n', 1)
        dockerfile = "FROM img\nWORKDIR /app\nRUN echo ok > result.txt\n"
        server, task, agent, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile=dockerfile, toml=toml, separate_verifier_for_artifacts=True
        )
        assert task.overlay and task.workdir == "/work" and server._separate_verifier(task)
        assert [instruction.path for instruction in task.overlay_workdirs] == ["/app"]

        class ExistingDirVerifier(OverlaySandbox):
            """Scores as a test.sh started by `docker exec -w <cwd>` would: 1 from /work once it exists, else 0."""

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if "test.sh" in command and "timeout" in command:
                    self.verifier_files = {"reward.txt": "1\n" if cwd == "/work" and cwd in self.dirs else "0\n"}
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        verifier = ExistingDirVerifier()

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.verifier_mode == "separate" and response.result.reward == 1.0
        (run,) = [c for c in verifier.execs if "test.sh" in c["command"] and "timeout" in c["command"]]
        assert run["cwd"] == "/work"
        # The seed made /work in the agent's sandbox; the verifier made it as root, the same way, before test.sh,
        # next to the /app the overlay's WORKDIR step made.
        (made,) = [c for c in verifier.execs if c["command"].startswith("mkdir -p /work")]
        assert made["ran_as"] == "root" and verifier.execs.index(made) < verifier.execs.index(run)
        assert "/work" in agent.dirs and verifier.dirs["/work"] == agent.dirs["/work"] and "/app" in verifier.dirs

    def separate_verifier(self, tmp_path, monkeypatch, *, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=None):
        server, task, agent, _ = self.overlay_server(tmp_path, monkeypatch, toml=toml)
        verifier = verifier or OverlaySandbox()

        async def create_verifier(task, *, episode_key=None):
            verifier.spec_env = dict(
                server._sandbox_spec(task, None, environment=task.config.verifier.environment).env
            )
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier

    async def test_a_separate_verifier_with_no_image_and_no_tests_dockerfile_is_refused_as_harbor_refuses_it(
        self, tmp_path, monkeypatch
    ):
        """Harbor's verifier environment dir is always tests/: this shape fails its environment validation, so the
        loader refuses it (prepare leaves it out) and the server refuses it at seed. Nothing grades in the agent
        image, overlay or not."""
        agent = OverlaySandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, dockerfile=OVERLAY_DOCKERFILE)
        (task.path / "task.toml").write_text(SEPARATE_IN_TASK_IMAGE_TOML)
        with pytest.raises(HarborTaskError, match="has no environment definition"):
            load_task(task.path)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "has no environment definition" in response.failure.failure_reason
        assert "Set [verifier.environment].docker_image or add tests/Dockerfile" in response.failure.failure_reason
        assert agent.execs == [] and not server._sandboxes

    async def test_prebuilt_verifier_image_gets_no_overlay(self, tmp_path, monkeypatch):
        toml = TASK_TOML.replace(
            "[verifier]\n",
            '[verifier]\nenvironment_mode = "separate"\n\n[verifier.environment]\ndocker_image = "org/verifier:1"\n',
        )
        server, task, _, verifier = self.separate_verifier(tmp_path, monkeypatch, toml=toml)
        assert _verifier_image(task) == "org/verifier:1"
        assert (await run_episode(server, task)).result.reward == 1.0
        assert not any("apt-get" in c["command"] for c in verifier.execs)
        # Nor the agent image's Dockerfile environment.
        assert verifier.spec_env == {}
