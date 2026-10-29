# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M30 at the server: the job-level Compose overlay turns an image-only task into a group with the overlay's
services, its MCP entries reach the harness merged with the task's in the task context and the grant list, the
prompt template wraps the first turn's instruction, and the merged document stays private."""

import asyncio
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from environment_servers.harbor import app as harbor_app
from environment_servers.harbor import sandboxes as harbor_sandboxes
from environment_servers.harbor.entrypoint import entrypoint_to_run
from environment_servers.harbor.tests.conftest import (
    COMPOSE_YAML,
    IMAGE_CONFIGS,
    IMAGE_PATH,
    TASK_TOML,
    FakeCompose,
    FakeSandbox,
    episode,
    make_server,
    run_episode,
    write_image_config,
)
from environment_servers.harbor.tests.test_app import COMPOSE_TOML, StubProvider
from environment_servers.harbor.tests.test_multi_turn import MCP_TOML, THREE_STEPS, multi_turn_server
from nemo_gym.data import RECORD_VERSION, HarborComposeOverlay, load_compose_document, load_task
from nemo_gym.data.tasks.harbor.dockerfile import OverlayWorkdir
from nemo_gym.data.tasks.harbor.image_configs import record_compose_images
from nemo_gym.data.tasks.harbor.materialize import materialize_task


# A memory stack the way the memory team ships one: a standalone file with host ports and a global network name,
# then a `.harbor.yaml` glue that resets both and attaches `main`.
STANDALONE = """
services:
  memory:
    image: org/memory:1
    ports:
      - "9050:9000"
    volumes:
      - memory_data:/data
    restart: unless-stopped
    networks:
      - memory_net
volumes:
  memory_data:
networks:
  memory_net:
    name: ${MEMORY_NETWORK_NAME:-memory_net}
"""
GLUE = """
services:
  memory:
    ports: !reset []
  main:
    depends_on:
      memory:
        condition: service_healthy
    networks:
      - default
      - memory_net
networks:
  memory_net:
    name: !reset null
"""
TEMPLATE = "You have a memory server.\n\n## Task Instruction\n\n{{ instruction }}\n"


def memory_overlay(
    tmp_path: Path, *, template: bool = False, mcp: bool = True, files: bool = True
) -> HarborComposeOverlay:
    folder = tmp_path / "overlay"
    folder.mkdir(exist_ok=True)
    (folder / "memory.yaml").write_text(STANDALONE)
    (folder / "memory.harbor.yaml").write_text(GLUE)
    (folder / "memory_prompt.md.j2").write_text(TEMPLATE)
    return HarborComposeOverlay(
        files=[folder / "memory.yaml", folder / "memory.harbor.yaml"] if files else [],
        mcp_servers=[{"name": "memory", "transport": "streamable-http", "url": "http://memory:9000/mcp"}]
        if mcp
        else [],
        prompt_template=folder / "memory_prompt.md.j2" if template else None,
    )


def overlay_server(
    tmp_path, monkeypatch, *, toml: str = TASK_TOML, overlay: HarborComposeOverlay | None = None, **config
):
    """An image-only task (the conftest's pull-mode Dockerfile) with the overlay's sidecar image recorded, and a
    ``_create_compose`` stub that keeps the resolved document and hands back a fake group."""
    server, task, sandbox, created = make_server(tmp_path, monkeypatch, toml=toml, **config)
    write_image_config(task.path.parent, "org/memory:1", Cmd=["memory-server"], User="memory")
    server.config.compose_overlay = overlay if overlay is not None else memory_overlay(tmp_path)
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    documents: list[dict] = []
    memory = FakeSandbox()
    compose = FakeCompose({"main": sandbox, "memory": memory})

    async def start_compose(task, episode_key):
        documents.append(server._compose_document(task, episode_key))
        sandbox.created_in(task.workdir)
        created.append((task.task_id, task.workdir))
        return compose

    monkeypatch.setattr(server, "_create_compose", start_compose)
    return server, task, sandbox, compose, documents


def volume_free_overlay(tmp_path: Path) -> HarborComposeOverlay:
    """The memory stack without its named volume, which the stub provider has no shared storage for, with the
    `service_healthy` dependency the glue adds; the sidecar's recorded healthcheck is written by `start_real_group`."""
    folder = tmp_path / "sidecar"
    folder.mkdir(exist_ok=True)
    (folder / "memory.yaml").write_text(
        "services:\n  memory:\n    image: org/memory:1\n"
        "  main:\n    depends_on:\n      memory:\n        condition: service_healthy\n"
    )
    return HarborComposeOverlay(files=[folder / "memory.yaml"], mcp_servers=[], prompt_template=None)


class DirectoryProvider(StubProvider):
    """A StubProvider whose execs reach a FakeSandbox per service (``main_sandbox`` for the task image): directories
    exist once something made them, and each exec records the user and directory the container ran it as."""

    def __init__(self, task, main_sandbox: FakeSandbox):
        super().__init__("opensandbox")
        self.task, self.main_sandbox = task, main_sandbox
        self.sandboxes: dict[str, FakeSandbox] = {}

    async def create(self, spec):
        handle = await super().create(spec)
        self.sandboxes[handle.sandbox_id] = self.main_sandbox if spec.image == self.task.image else FakeSandbox()
        return handle

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        return await self.sandboxes[handle.sandbox_id].exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


async def start_real_group(monkeypatch, server, task, main_sandbox: FakeSandbox | None = None):
    """The server's own ``_create_compose`` (``overlay_server`` stubs the instance's) through the real
    ``AsyncSandboxCompose`` and a ``DirectoryProvider``; returns the group, the provider and main's fake sandbox.
    The `service_healthy` dependency needs the sidecar's healthcheck for the real adapter's validation."""
    from nemo_gym.sandbox.adapters import docker_compose

    write_image_config(
        task.path.parent, "org/memory:1", Cmd=["memory-server"], User="memory", Healthcheck={"Test": ["CMD", "true"]}
    )
    main_sandbox = main_sandbox if main_sandbox is not None else FakeSandbox()
    provider = DirectoryProvider(task, main_sandbox)
    monkeypatch.setattr(docker_compose, "create_provider", lambda config: provider)
    compose = await harbor_app.HarborEnvironmentServer._create_compose(server, task, "r1")
    # The adapter runs each service's command (main's run.sh) as a background task; the fake sandboxes never
    # suspend, so the loop gets to run it only once the start has returned.
    for _ in range(3):
        await asyncio.sleep(0)
    return compose, provider, main_sandbox


def main_spec_of(provider: DirectoryProvider, task):
    (spec,) = [spec for spec in provider.created if spec.image == task.image]
    return spec


def entrypoint_exec_of(sandbox: FakeSandbox) -> dict:
    """The adapter's exec of main's ``run.sh`` (its command, entrypoint included)."""
    (record,) = [
        r for r in sandbox.execs if r["command"].startswith("sh ") and r["command"].split()[1].endswith("/run.sh")
    ]
    return record


class TestGroup:
    def test_an_image_only_task_becomes_the_one_service_group_plus_the_overlay_services(self, tmp_path, monkeypatch):
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch)
        assert not task.needs_compose and server._uses_compose(task)

        document = server._compose_document(task)

        assert list(document["services"]) == ["main", "memory"]
        main = document["services"]["main"]
        # Harbor's own `main`: the task image, Harbor's keep-alive command, the glue's dependency and network.
        assert main["image"] == "ubuntu:24.04" and main["command"] == ["sh", "-c", "sleep infinity"]
        assert main["depends_on"] == {"memory": {"condition": "service_healthy"}}
        assert main["networks"] == {"default": {}, "memory_net": {}}
        memory = document["services"]["memory"]
        # The overlay's service with its recorded start, the standalone file's host ports reset by the glue, its
        # named volume kept for the provider's shared storage, and the network declared without the global name.
        # The non-root sidecar keeps its image user (not copied onto the service) and gets no host injection.
        assert memory["command"] == ["memory-server"] and "user" not in memory
        assert memory["x-sandbox"] == {"hosts": []}
        assert "ports" not in memory and memory["restart"] == "unless-stopped"
        assert memory["volumes"] == [
            {"type": "volume", "source": "memory_data", "target": "/data", "read_only": False}
        ]
        assert document["volumes"] == {"memory_data": {}} and document["networks"] == {"memory_net": {}}

    @pytest.mark.parametrize(
        ("dockerfile", "entrypoint"),
        [
            ("FROM ubuntu:24.04\nWORKDIR /app\nENTRYPOINT []", []),
            ("FROM ubuntu:24.04\nWORKDIR /app\nUSER app", ["/entry.sh"]),
        ],
        ids=["cleared", "kept"],
    )
    def test_main_keeps_the_tasks_cleared_entrypoint_when_an_overlay_is_attached(
        self, tmp_path, monkeypatch, dockerfile, entrypoint
    ):
        """A pull-mode task whose Dockerfile wrote ``ENTRYPOINT []`` starts its single sandbox without the base
        image's entrypoint (``entrypoint_to_run`` is ``None``); the group an overlay turns it into starts ``main``
        the same way, not with the base's recorded entrypoint put back by the image resolution. A Dockerfile that
        keeps it starts ``main`` with it, as the single sandbox and Docker would. Either way ``main``'s
        ``working_dir`` is the Dockerfile's final ``WORKDIR``, the directory the single sandbox runs the entrypoint
        in, not the base image's recorded ``WorkingDir``; see
        test_a_pull_mode_tasks_main_starts_in_the_workdir_while_the_seed_works_in_the_task_workdir for
        an ``[environment].workdir`` of its own. ``main`` carries no ``user``: the
        adapter runs its command, entrypoint included, as the sandbox's default user, the task's resolved user
        (``task.user``, the Dockerfile's final ``USER`` here, ``app``), which is the user the single sandbox gives
        the entrypoint (``task.entrypoint_user``) unless ``[agent].user`` names another."""
        server, task, _, _, _ = overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile=dockerfile,
            image_config={
                "Env": [f"PATH={IMAGE_PATH}"],
                "Entrypoint": ["/entry.sh"],
                "WorkingDir": "/srv",
                "User": "svc",
            },
        )
        assert task.entrypoint == (tuple(entrypoint) or None)
        assert entrypoint_to_run(task) == (entrypoint or None)
        assert task.entrypoint_workdir == "/app"
        assert task.entrypoint_user == ("app" if entrypoint else "svc")

        main = server._compose_document(task)["services"]["main"]

        assert main["entrypoint"] == entrypoint
        assert main["working_dir"] == task.entrypoint_workdir == task.workdir == "/app"
        assert "user" not in main
        assert main["image"] == "ubuntu:24.04" and main["command"] == ["sh", "-c", "sleep infinity"]
        # No user on main: the adapter runs its command as the sandbox's default, the task's resolved user, which is
        # the entrypoint's user here (the Dockerfile's USER app, else the base's svc); only [agent].user would differ.
        assert server._sandbox_spec(task, task.workdir).user == task.user == task.entrypoint_user

    async def test_a_pull_mode_tasks_main_starts_in_the_workdir_while_the_seed_works_in_the_task_workdir(
        self, tmp_path, monkeypatch
    ):
        """A pull-mode task whose ``[environment].workdir`` (``/other``) is not the Dockerfile's final ``WORKDIR``
        (``/app``), with an image ``HEALTHCHECK`` and a task ``[environment.healthcheck]``. The directories are
        Harbor's: its built image carries the ``WORKDIR`` as the container's directory (``docker-compose-build.yaml``
        writes no ``working_dir`` for ``main``), so the entrypoint and the image healthcheck run in ``/app``, while
        ``[environment].workdir`` is the ``cwd`` of Harbor's execs (harbor 0.22.0 ``environments/docker/docker.py``
        1104), the task healthcheck's among them. The synthesized ``main`` writes the ``WORKDIR`` as ``working_dir``,
        which the adapter makes as root, as Docker makes a missing ``working_dir``, and the server's setup hook hands
        to the entrypoint's user before main's command, as the single sandbox makes the ``WORKDIR`` app's before its
        entrypoint starts (``_start_entrypoint``):
        the adapter creates main's sandbox in ``/app`` and runs its command, entrypoint included, and its recorded
        healthcheck there (it gives neither a ``cwd`` of its own) as the sandbox's default user (``task.user``,
        ``app``), as the single sandbox runs both; the seed then prepares ``/other`` for the agent, as it does in a
        single sandbox, and gives the task healthcheck that directory explicitly (``_wait_healthy(workdir=...)``),
        as every exec of the seed, the turns and the verifier names its ``cwd`` (review, round 32)."""
        toml = TASK_TOML.replace("[environment]\n", '[environment]\nworkdir = "/other"\n')
        toml += (
            '\n[environment.healthcheck]\ncommand = "test -f ready"\ninterval_sec = 0.01\nstart_interval_sec = 0.01\n'
        )
        server, task, _, _, _ = overlay_server(
            tmp_path,
            monkeypatch,
            toml=toml,
            dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nUSER app",
            image_config={
                "Env": [f"PATH={IMAGE_PATH}"],
                "Entrypoint": ["/entry.sh"],
                "WorkingDir": "/srv",
                "Healthcheck": {"Test": ["CMD", "test", "-f", "/tmp/healthy"]},
            },
            overlay=volume_free_overlay(tmp_path),
        )
        assert not task.overlay and task.workdir == "/other" and task.entrypoint_workdir == "/app"
        assert task.user == task.entrypoint_user == "app"
        # The single sandbox is created in the task workdir (its execs' default) and gives the entrypoint and the
        # image check /app explicitly (`_start_entrypoint`, `_wait_healthy`).
        assert server._sandbox_spec(task, None if task.overlay else task.workdir).workdir == "/other"

        main = server._compose_document(task)["services"]["main"]
        assert main["entrypoint"] == ["/entry.sh"] and main["working_dir"] == "/app" and "user" not in main
        assert main["healthcheck"] == {"test": ["CMD", "test", "-f", "/tmp/healthy"]}

        compose, provider, main_sandbox = await start_real_group(monkeypatch, server, task)
        try:
            assert compose._ready
            spec = main_spec_of(provider, task)
            assert spec.workdir == "/app" and spec.user == "app"
            # The adapter made the WORKDIR as root, as Docker creates a missing working_dir, and the server's setup
            # hook handed it to the entrypoint's user, as root, before main's command started; nothing has made
            # /other yet.
            entry = entrypoint_exec_of(main_sandbox)
            commands = [r["command"] for r in main_sandbox.execs]
            handed = commands.index("chown app /app")
            assert commands.index("mkdir -p /app") < handed < commands.index(entry["command"])
            assert main_sandbox.execs[handed]["ran_as"] == "root"
            assert main_sandbox.dirs["/app"] == "app" and "/other" not in main_sandbox.dirs
            # The entrypoint and the image healthcheck ran as the sandbox's default user in the spec's directory, the
            # user's own by then: the facade fills the cwd the adapter leaves out with it, as the daemon runs both in
            # the container's.
            assert entry["ran_as"] == "app" and entry["cwd"] == "/app" and entry["cwd_owner"] == "app"
            (image_check,) = [r for r in main_sandbox.execs if r["command"] == "test -f /tmp/healthy"]
            assert image_check["ran_as"] == "app" and image_check["cwd"] == "/app"
            assert not [r for r in main_sandbox.execs if r["command"] == "test -f ready"]
            # The seed's side, on the group's main: the task workdir is prepared for the agent (made and handed to
            # app), as in a single sandbox, while main keeps the WORKDIR as its own directory, and the task
            # healthcheck runs in the prepared directory, named explicitly, as the image's user.
            main_service = compose.services["main"]
            prepared = await server._prepare_workdir(main_service, task, task.workdir)
            assert prepared == "/other" and main_sandbox.dirs["/other"] == "app"
            assert main_spec_of(provider, task).workdir == "/app"
            await server._wait_healthy(main_service, task, user="app", workdir=prepared, compose_managed=True)
            (task_check,) = [r for r in main_sandbox.execs if r["command"] == "test -f ready"]
            assert task_check["ran_as"] == "app" and task_check["cwd"] == "/other"
        finally:
            await compose.stop()

    async def test_an_overlay_tasks_main_leaves_its_final_workdir_to_the_overlay(self, tmp_path, monkeypatch):
        """An overlay task (a Dockerfile with ``RUN`` lines) whose final ``WORKDIR`` is a new directory under a
        non-root ``USER``: the synthesized ``main`` carries no ``working_dir`` and its spec no workdir, as its
        single sandbox starts (``_create_sandbox_within_deadline``). The adapter creates a service's ``working_dir``
        as root before the overlay runs, and the overlay's ``WORKDIR`` step leaves an existing directory as it is, so
        the ``RUN`` line in it would run in a root-owned directory and fail. Through the real adapter the overlay's
        ``WORKDIR`` step finds the directory missing, creates it owned by ``app``, and the ``RUN`` line runs there as
        ``app`` (review, round 32)."""
        server, task, _, _, _ = overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM ubuntu:24.04\nRUN useradd -m app\nUSER app\nWORKDIR /home/app/work\nRUN touch marker\n",
            image_config={"Env": [f"PATH={IMAGE_PATH}"]},
            overlay=volume_free_overlay(tmp_path),
        )
        assert task.overlay_workdirs == (OverlayWorkdir("/home/app/work", "app", before=1),)
        assert task.workdir == task.entrypoint_workdir == "/home/app/work" and task.user == "app"

        document = server._compose_document(task)

        main = document["services"]["main"]
        assert main["entrypoint"] == [] and "working_dir" not in main and "user" not in main
        assert server._sandbox_spec(task, None if task.overlay else task.workdir).workdir is None

        @dataclass
        class WatchingSandbox(FakeSandbox):
            """Notes, at the overlay's WORKDIR step, whether anything had made the directory already."""

            existed_at_workdir_step: list[bool] = field(default_factory=list)

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command.startswith("test -d /home/app/work || "):
                    self.existed_at_workdir_step.append("/home/app/work" in self.dirs)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        compose, provider, main_sandbox = await start_real_group(monkeypatch, server, task, WatchingSandbox())
        try:
            assert compose.services["main"]._provider is provider and compose._ready
            assert main_spec_of(provider, task).workdir is None
            # The adapter made no directory for main before the overlay: the WORKDIR step found it missing and
            # made it, owned by app, so the RUN line in it ran as app in app's directory.
            assert "mkdir -p /home/app/work" not in [record["command"] for record in main_sandbox.execs]
            assert main_sandbox.existed_at_workdir_step == [False]
            assert main_sandbox.dirs["/home/app/work"] == "app"
            marker = main_sandbox.run("touch marker")
            assert marker["seen"]["user"] == "app" and marker["seen"]["cwd"] == "/home/app/work"
            assert marker["cwd_owner"] == "app", "the directory was root-owned when the RUN line ran"
        finally:
            await compose.stop()

    async def test_an_overlay_tasks_entrypoint_starts_as_the_base_image_runs_it(self, tmp_path, monkeypatch):
        """The gap ``synthesized_group`` states for an overlay task, pinned so the statement stays true: the base
        image's recorded ``ENTRYPOINT`` is kept (the Dockerfile does not clear it) and the single sandbox runs it as
        the Dockerfile's final ``USER`` in its final ``WORKDIR`` (``_start_entrypoint`` names ``task.entrypoint_user``
        and ``task.entrypoint_workdir``), as Harbor's built image starts it. The group's ``main`` carries neither (the
        directory would be root-made before the overlay, above), so the adapter starts its command, entrypoint
        included, before ``_set_default_user`` runs, as the base image's own user in the base image's recorded
        ``WorkingDir`` that the resolution filled in. The agent's commands run as ``task.user`` in ``task.workdir``
        once the seed installed both."""
        server, task, _, _, _ = overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM ubuntu:24.04\nRUN useradd -m app\nUSER app\nWORKDIR /home/app/work\n",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "Entrypoint": ["/entry.sh"], "WorkingDir": "/srv"},
            overlay=volume_free_overlay(tmp_path),
        )
        assert task.overlay and entrypoint_to_run(task) == ["/entry.sh"]
        assert task.entrypoint_user == "app" and task.entrypoint_workdir == "/home/app/work"

        main = server._compose_document(task)["services"]["main"]
        # The base image's record fills the directory the synthesized main left out; no user anywhere.
        assert main["entrypoint"] == ["/entry.sh"] and main["working_dir"] == "/srv" and "user" not in main

        compose, provider, main_sandbox = await start_real_group(monkeypatch, server, task)
        try:
            spec = main_spec_of(provider, task)
            assert spec.workdir == "/srv" and spec.user is None
            # No default user yet (`_set_default_user` runs after the group started): the base image's own user, in
            # the base's recorded WorkingDir, not app in /home/app/work as the single sandbox runs it.
            entry = entrypoint_exec_of(main_sandbox)
            assert entry["user"] is None and main_sandbox.default_user is None
            assert entry["ran_as"] == main_sandbox.image_user == "root" and entry["cwd"] == "/srv"
            assert main_sandbox.dirs["/srv"] == "root" and main_sandbox.dirs["/home/app/work"] == "app"
        finally:
            await compose.stop()

    def test_without_overlay_files_an_image_only_task_stays_a_single_sandbox(self, tmp_path, monkeypatch):
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch, overlay=memory_overlay(tmp_path, files=False))
        assert not server._uses_compose(task)
        server.config.compose_overlay = None
        assert not server._uses_compose(task)

    async def test_the_episode_seeds_the_group_and_the_overlay_services_stop_with_it(self, tmp_path, monkeypatch):
        server, task, sandbox, compose, documents = overlay_server(tmp_path, monkeypatch)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        assert len(documents) == 1 and set(documents[0]["services"]) == {"main", "memory"}
        # The group, overlay services included, was stopped with the episode and released.
        assert compose.stopped and not server._sandboxes and not server._sessions

    async def test_the_merged_document_stays_private(self, tmp_path, monkeypatch):
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch)
        started: list[tuple[Path, str]] = []

        class RecordingCompose:
            def __init__(self, provider, path, *, service_specs, timeout_s, **adapter_kwargs):
                self.path = Path(path)
                self.services = {name: object() for name in service_specs}

            async def start(self):
                started.append((self.path, self.path.read_text()))

        monkeypatch.setattr("environment_servers.harbor.sandboxes.AsyncSandboxCompose", RecordingCompose)

        await harbor_app.HarborEnvironmentServer._create_compose(server, task, "r1")

        ((path, text),) = started
        assert "org/memory" in text
        assert not path.exists() and not path.is_relative_to(server._artifacts_dir)
        assert not list(server._artifacts_dir.rglob("compose.yaml"))


class TestSeedPrecheck:
    async def test_the_preflight_reads_its_files_off_the_loop_thread(self, tmp_path, monkeypatch):
        """The seed's Compose preflight is file work (the task's Compose file, the overlay files, their env files, the
        image records): it runs in a worker thread, one read of the record file, so a slow filesystem stalls the one
        episode and not the loop that serves every other. The group's own resolution (`_compose_document`, the start
        stub here) comes after it, on the loop."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        loop_thread = threading.get_ident()
        threads: dict[str, list[int]] = {"load": [], "check": []}
        load = harbor_sandboxes.load_compose_document
        check = server._check_compose_group

        def recording_load(*args, **kwargs):
            threads["load"].append(threading.get_ident())
            return load(*args, **kwargs)

        def recording_check(*args, **kwargs):
            threads["check"].append(threading.get_ident())
            return check(*args, **kwargs)

        monkeypatch.setattr(harbor_sandboxes, "load_compose_document", recording_load)
        monkeypatch.setattr(server, "_check_compose_group", recording_check)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert len(documents) == 1 and compose.stopped
        # The preflight's load and group check, on one worker thread; then the start's own (the stub's
        # `_compose_document`), on the loop.
        assert threads["load"][0] != loop_thread and threads["check"][0] == threads["load"][0]
        assert threads["load"][1:] == [loop_thread] and threads["check"][1:] == [loop_thread]

    async def test_an_overlay_image_prepare_did_not_record_is_refused_terminally_before_any_group(
        self, tmp_path, monkeypatch
    ):
        """A dataset prepared without this ``compose_overlay`` has no record for the overlay's images: a configuration
        error the seed refuses (422, terminal) naming the services, not a group start that fails and is retried."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        extra = tmp_path / "overlay" / "extra.yaml"
        extra.write_text("services:\n  extra:\n    image: org/extra:1\n  other:\n    image: org/other:1\n")
        server.config.compose_overlay = HarborComposeOverlay(files=[*server.config.compose_overlay.files, extra])

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        reason = response.failure.failure_reason
        assert "service 'extra' ('org/extra:1')" in reason and "service 'other' ('org/other:1')" in reason
        assert "'memory'" not in reason and "prepare the dataset again with this compose_overlay" in reason
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions

    @pytest.mark.parametrize(
        ("patch", "was"),
        [
            ({"architecture": "arm64"}, f"linux/arm64, record version {RECORD_VERSION}"),
            ({"record_version": RECORD_VERSION - 1}, f"linux/amd64, record version {RECORD_VERSION - 1}"),
        ],
        ids=["other-platform", "stale-record-version"],
    )
    async def test_a_record_this_deployment_cannot_use_is_refused_terminally_before_any_group(
        self, tmp_path, monkeypatch, patch, was
    ):
        """A record that exists but is not reusable for this deployment (another platform: the dataset prepared on x86
        and served on arm64; or an old record version) is refused by prepare's own rule at seed, terminally, naming
        what was recorded, where ``resolve_compose`` would raise inside the group's start and be retried every
        episode."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        record = task.path.parent / "compose-images.json"
        recorded = json.loads(record.read_text())
        recorded["org/memory:1"].update(patch)
        record.write_text(json.dumps(recorded))

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        reason = response.failure.failure_reason
        assert reason.startswith(
            "Compose group of task 'hello': no image configuration this deployment can use recorded in "
        )
        assert (
            f"for service 'memory' ('org/memory:1' is recorded for {was}; this deployment runs linux/amd64 and reads "
            f"record version {RECORD_VERSION}); prepare the dataset again with this compose_overlay, for this platform"
        ) in reason
        assert "'main'" not in reason
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions

    async def test_an_image_reference_only_a_seed_time_variable_sets_is_refused_naming_the_prepare_rule(
        self, tmp_path, monkeypatch
    ):
        """Prepare interpolates the overlay without the server's `compose_variables` (a name only they set is empty, or
        its `${X:-default}` default), so `org/memory:${MEMORY_TAG:-1}` is recorded as org/memory:1; the seed, with
        MEMORY_TAG=2 in hand, resolves org/memory:2, which no record covers. Preparing again with the overlay cannot
        change that, so the refusal says what does: the value has to be in prepare's process environment too."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        (tmp_path / "overlay" / "memory.yaml").write_text(
            STANDALONE.replace("org/memory:1", "org/memory:${MEMORY_TAG:-1}")
        )
        # What prepare recorded: the overlay read as prepare reads it, the tag at its default.
        prepared = load_compose_document(task, overlay=server.config.compose_overlay)
        assert prepared["services"]["memory"]["image"] == "org/memory:1"
        server.config.compose_variables = {"MEMORY_TAG": harbor_app.ComposeVariable.model_validate({"value": "2"})}

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        reason = response.failure.failure_reason
        assert "for service 'memory' ('org/memory:2'); prepare the dataset again with this compose_overlay" in reason
        assert "'org/memory:1'" not in reason
        assert reason.endswith(
            "so its images are recorded. An image reference, or a variable in one, that only a seed-time "
            "compose_variables value sets resolves to its default at prepare, so set the value in prepare's process "
            "environment too: image references and the variables they use must resolve at preparation"
        )
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions

    async def test_a_missing_image_record_is_refused_terminally(self, tmp_path, monkeypatch):
        """An image-only dataset prepared before the overlay was set has no record file at all."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        record = task.path.parent / "compose-images.json"
        record.rename(record.with_suffix(".json.aside"))

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        reason = response.failure.failure_reason
        assert reason.startswith("Compose group of task 'hello' needs the image configurations prepare records at ")
        assert reason.endswith(
            "compose-images.json; prepare the dataset again, with this compose_overlay when one is set"
        )
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            (
                "services:\n  memory:\n    privileged: true\n",
                "Service 'memory': unsupported Compose fields ['privileged']",
            ),
            (
                "services:\n  memory:\n    volumes: ['{outside}/conf:/etc/memory']\n",
                "from outside the task; map it under `volume_sources` in the sandbox block",
            ),
            (
                # Relative to the project directory, the task's environment folder, as under Harbor; not to the
                # overlay file's folder, where a ``conf`` exists.
                "services:\n  memory:\n    volumes: ['./conf:/etc/memory']\n",
                "binds './conf', which the task folder does not contain",
            ),
            ("services: !reset\n", "the merged document has no services mapping"),
            (
                # The recorded org/memory:1 runs as user `memory`: the group starts it without host injection, so
                # prepare refuses host entries on it once the images are recorded; the seed refuses them the same way.
                "services:\n  memory:\n    extra_hosts: ['licence.example:198.51.100.7']\n",
                "service 'memory' extra_hosts cannot be applied without host injection: the image runs as user "
                "'memory', so the sidecar takes no host entries",
            ),
        ],
        ids=["field", "foreign-bind-source", "relative-bind-source", "no-services", "extra-hosts-on-non-root-sidecar"],
    )
    async def test_an_overlay_the_deployment_cannot_run_is_refused_terminally_before_any_group(
        self, tmp_path, monkeypatch, text, reason
    ):
        """The checks prepare ran (field check, volume planner, the non-root sidecar's extra_hosts rule) run at seed
        on the merged document too: an overlay file that no longer passes them (the overlay is a run-config value,
        changed between prepare and serve) is a 422, terminal, naming the overlay files, not a failure inside the
        group's start retried for every episode. Prepare, run on the same overlay, skips the task with the same
        reason."""
        server, task, _, compose, documents = overlay_server(tmp_path, monkeypatch)
        (tmp_path / "overlay" / "conf").mkdir()
        extra = tmp_path / "overlay" / "extra.yaml"
        extra.write_text(text.format(outside=tmp_path / "overlay"))
        server.config.compose_overlay = HarborComposeOverlay(files=[*server.config.compose_overlay.files, extra])

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        failure_reason = response.failure.failure_reason
        assert failure_reason.startswith("Compose group of task 'hello': hello: ") or failure_reason.startswith(
            "Compose file of task 'hello': hello: "
        )
        assert reason in failure_reason and str(extra) in failure_reason
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions
        skipped: dict[str, str] = {}
        record_compose_images(
            [task],
            task.path.parent,
            skipped=skipped,
            overlay=server.config.compose_overlay,
            sandbox_provider="opensandbox",
        )
        assert set(skipped) == {"hello"} and failure_reason.endswith(skipped["hello"]), skipped

    async def test_an_overlay_that_resets_a_service_the_tasks_hooks_name_is_refused_as_prepare_refuses_it(
        self, tmp_path, monkeypatch
    ):
        """`services: {kafka: !reset}` against a task whose `[[verifier.collect]]` hook runs in `kafka`: the task is
        valid on its own and the overlay merges, but the group would start without the service the hook needs.
        Prepare skips the task naming the overlay files and the service, and the seed refuses it terminally with the
        same reason, before any group (the check used to live only inside the group's start, retried per episode)."""
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        assert [hook.service for hook in task.config.verifier.collect] == ["kafka"]
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.sandboxes.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        reset = tmp_path / "overlay" / "drop_kafka.yaml"
        reset.parent.mkdir()
        reset.write_text("services:\n  kafka: !reset\n")
        server.config.compose_overlay = HarborComposeOverlay(files=[reset])
        documents: list[dict] = []
        compose = FakeCompose({"main": FakeSandbox(), "api": FakeSandbox()})

        async def start_compose(task, episode_key):
            documents.append(server._compose_document(task, episode_key))
            return compose

        monkeypatch.setattr(server, "_create_compose", start_compose)
        skipped: dict[str, str] = {}
        record_compose_images(
            [task],
            task.path.parent,
            skipped=skipped,
            overlay=server.config.compose_overlay,
            sandbox_provider="opensandbox",
        )
        reason = (
            f"hello: environment/docker-compose.yaml merged with Compose overlay files [{reset}]: artifacts or collect "
            "hooks name Compose services the group does not have: ['kafka']"
        )
        assert skipped == {"hello": reason}

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        assert response.failure.failure_reason == f"Compose group of task 'hello': {reason}"
        assert documents == [] and not compose.stopped and not server._sandboxes and not server._sessions
        # Without the reset file the task's own group has the service, and the same checks pass the document.
        server.config.compose_overlay = None
        assert set(server._compose_document(task)["services"]) == {"main", "api", "kafka"}


class TestMcpGrants:
    async def test_overlay_entries_reach_the_harness_merged_with_the_tasks_in_the_same_list(
        self, tmp_path, monkeypatch
    ):
        overlay = HarborComposeOverlay(
            mcp_servers=[
                # Replaces the task's `legacy` entry by name (Harbor's rule) and adds `memory`.
                {"name": "legacy", "transport": "streamable-http", "url": "http://legacy:9000/mcp"},
                {"name": "memory", "transport": "sse", "url": "http://memory:9000/sse"},
            ]
        )
        server, task, sandbox, _, _ = overlay_server(
            tmp_path, monkeypatch, toml=TASK_TOML + MCP_TOML, overlay=overlay, task_mcp_grants="optional"
        )

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        context_write = next(c for c in sandbox.execs if "/tmp/.nemo-gym/task.json" in c["command"])
        written = json.loads(context_write["command"].split("printf %s ", 1)[1].split(" > ", 1)[0].strip("'"))
        assert [(s["name"], s["transport"], s["url"]) for s in written["mcp_servers"]] == [
            ("user-server", "streamable-http", "http://user-server:8000/mcp"),
            ("legacy", "streamable-http", "http://legacy:9000/mcp"),
            ("memory", "sse", "http://memory:9000/sse"),
        ]
        (seed,) = server.server_client.seed_bodies()
        # One grant list: the task's streamable-http entry and the overlay's; the sse entry cannot be granted.
        assert [(a["name"], a["connection"]["url"]) for a in seed["tool_accesses"]] == [
            ("user-server", "http://user-server:8000/mcp"),
            ("legacy", "http://legacy:9000/mcp"),
        ]

    async def test_an_overlay_entry_that_cannot_be_granted_is_warned_about_once_per_run(
        self, tmp_path, monkeypatch, caplog
    ):
        """A job-level `sse` entry (the memory team's Redis stack is `sse` by design) is the operator's choice for the
        whole run: one warning per entry per run, not one per turn of every episode. The task's own `sse` entry keeps
        its warning at every seed, the task author's signal. Both still reach the harness in task.json."""
        overlay = HarborComposeOverlay(
            mcp_servers=[{"name": "redis-memory", "transport": "sse", "url": "http://mcp:9000/sse"}]
        )
        server, task, sandbox, _, _ = overlay_server(
            tmp_path, monkeypatch, toml=TASK_TOML + MCP_TOML, overlay=overlay, task_mcp_grants="optional"
        )
        caplog.set_level("WARNING", logger="environment_servers.harbor.app")

        for rollout in ("r1", "r2", "r3"):
            assert (await run_episode(server, task, rollout=rollout)).failure is None

        warnings = [record.getMessage() for record in caplog.records if "cannot be granted" in record.getMessage()]
        assert (
            warnings.count("hello: MCP server 'legacy' (sse) cannot be granted; only streamable-http URL entries can")
            == 3
        )
        assert (
            warnings.count(
                "compose_overlay: MCP server 'redis-memory' (sse) cannot be granted; only streamable-http URL entries can; "
                "it reaches the harness in /tmp/.nemo-gym/task.json (said once per run)"
            )
            == 1
        )
        assert len(warnings) == 4
        for seed in server.server_client.seed_bodies():
            assert [a["name"] for a in seed["tool_accesses"]] == ["user-server"]
        writes = [c["command"] for c in sandbox.execs if "/tmp/.nemo-gym/task.json" in c["command"]]
        assert len(writes) == 3 and all("http://mcp:9000/sse" in w and "http://legacy:9000/sse" in w for w in writes)

    async def test_an_overlay_only_sse_entry_is_refused_under_required_grants(self, tmp_path, monkeypatch):
        """The grant and the refusal read the same merged list: a job-level `sse` entry cannot be granted, so a
        harness that reads no task.json is refused naming it even when `task_mcp_grants: required` granted every
        streamable-http server, and an overlay-only streamable-http entry granted as required lets the task run."""
        overlay = HarborComposeOverlay(
            mcp_servers=[{"name": "redis-memory", "transport": "sse", "url": "http://mcp:9000/sse"}]
        )
        server, task, sandbox, _, _ = overlay_server(
            tmp_path, monkeypatch, overlay=overlay, task_mcp_grants="required"
        )
        assert not task.config.environment.mcp_servers
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        failure = response.failure
        assert response.result is None and failure.stage == "agent" and failure.terminal
        assert "MCP server 'redis-memory'" in failure.failure_reason
        assert "task_mcp_grants: required" not in failure.failure_reason
        assert not any("test.sh" in call["command"] for call in sandbox.execs)

        (tmp_path / "ok").mkdir()
        granted = overlay_server(
            tmp_path / "ok",
            monkeypatch,
            overlay=memory_overlay(tmp_path / "ok", files=False),
            task_mcp_grants="required",
        )
        server2, task2, _, _, _ = granted
        server2.server_client.supports_task_tool_context = False
        response = await run_episode(server2, task2, rollout="r2")
        assert response.failure is None, response.failure
        (seed,) = server2.server_client.seed_bodies()
        assert [(a["name"], a["required"]) for a in seed["tool_accesses"]] == [("memory", True)]

    async def test_overlay_entries_alone_make_the_task_one_with_tools(self, tmp_path, monkeypatch):
        server, task, sandbox, _, _ = overlay_server(
            tmp_path, monkeypatch, overlay=memory_overlay(tmp_path, files=False)
        )
        assert not task.config.environment.mcp_servers
        server.server_client.supports_task_tool_context = False

        response = await run_episode(server, task)

        # A harness that cannot drive MCP servers is refused, exactly as for a task's own entries.
        assert response.result is None and response.failure.stage == "agent" and response.failure.terminal
        assert "declares MCP servers or a skills directory" in response.failure.failure_reason
        server.server_client.supports_task_tool_context = True
        response = await run_episode(server, task, rollout="r2")
        assert response.failure is None
        context_write = next(c for c in sandbox.execs if "/tmp/.nemo-gym/task.json" in c["command"])
        assert "http://memory:9000/mcp" in context_write["command"]


class TestPromptTemplate:
    async def test_the_first_turn_instruction_is_rendered_into_the_template(self, tmp_path, monkeypatch):
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch, overlay=memory_overlay(tmp_path, template=True))

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        rendered = "You have a memory server.\n\n## Task Instruction\n\nCreate hello.txt"
        (turn,) = [call[2]["json"] for call in server.server_client.calls if call[1].endswith("/v1/responses")]
        assert [(item.role, item.content) for item in turn.input] == [("user", rendered)]
        # The request to the harness goes out as `ServerClient.post` serializes it (set fields only): the row's own
        # keys with the rendered content, none of the model's defaults (`tools: []`, `tool_choice`, `phase`, ...)
        # made explicit by the rendering round trip.
        row = episode(task).task.task_input.responses_create_params.model_dump(exclude_unset=True)
        assert row == {"input": [{"role": "user", "content": "Create hello.txt"}]}
        assert turn.model_dump(exclude_unset=True) == {"input": [{"role": "user", "content": rendered}]}
        # The result and the transcript carry what the harness saw.
        assert response.result.responses_create_params.input[0].content == rendered

    async def test_only_the_first_turn_of_a_multi_step_task_is_wrapped(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = multi_turn_server(
            tmp_path, monkeypatch, THREE_STEPS, {"plan": "Plan it.", "build": "Build it.", "check": "Check it."}
        )
        server.config.compose_overlay = memory_overlay(tmp_path, template=True, files=False, mcp=False)
        # The request as the wire carries it (set fields only): each turn's params are `input` and `metadata`.
        request = harbor_app.HarborEpisodeRequest.model_validate(
            {
                "episode_id": {"rollout_id": "r1"},
                "task": materialize_task(task, "ds").model_dump(mode="json", exclude_unset=True),
            }
        )
        rows = [turn.responses_create_params.model_dump(exclude_unset=True) for turn in request.task.task_input.turns]
        assert [set(row) for row in rows] == [{"input", "metadata"}] * 3

        response = await server.run_request(request)

        assert response.failure is None, response.failure
        bodies = [
            call[2]["json"].model_dump(exclude_unset=True)
            for call in server.server_client.calls
            if call[1].endswith("/v1/responses")
        ]
        turns = [body["input"][0]["content"] for body in bodies]
        assert turns == ["You have a memory server.\n\n## Task Instruction\n\nPlan it.", "Build it.", "Check it."]
        # Each turn reaches the harness as the row wrote it, less the row's step name (the server's bookkeeping,
        # which the sandbox harnesses would refuse); the rendered first turn differs in its content only (no
        # default of the params model made explicit by the rendering round trip).
        sent = [{**row, "metadata": {k: v for k, v in row["metadata"].items() if k != "harbor_step"}} for row in rows]
        assert bodies == [{**sent[0], "input": [{**sent[0]["input"][0], "content": turns[0]}]}, sent[1], sent[2]]
        # The multi-turn record keeps the last turn's params, so the rendered first turn is in its own transcript
        # only (the turn results carry no params; recording them is a follow-up with M34).
        assert response.result.responses_create_params.input[0].content == "Check it."
        assert [turn.name for turn in response.result.turns] == ["plan", "build", "check"]
        assert "memory server" not in json.dumps(response.result.model_dump(mode="json")["turns"])

    async def test_a_template_that_does_not_render_fails_the_episode_before_any_sandbox(self, tmp_path, monkeypatch):
        # The block compiles its template when it is read (an undefined variable passes then, as Harbor's
        # StrictUndefined only fails the render), so the bad text is written before the block is built.
        template = tmp_path / "overlay" / "bad_prompt.md.j2"
        template.parent.mkdir(exist_ok=True)
        template.write_text("{{ instruction }} for {{ user_name }}\n")
        overlay = HarborComposeOverlay(prompt_template=template)
        server, task, _, _, _ = overlay_server(tmp_path, monkeypatch, overlay=overlay)
        created = []
        monkeypatch.setattr(server, "_create_sandbox", lambda *a, **k: created.append(a))

        response = await run_episode(server, task)

        assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
        assert (
            "undefined variables" in response.failure.failure_reason and "user_name" in response.failure.failure_reason
        )
        assert created == [] and server.server_client.paths() == [] and not server._sandboxes

    def test_the_block_is_a_deployment_setting_of_the_server(self, tmp_path, monkeypatch):
        server, _, _, _, _ = overlay_server(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="Prompt template not found"):
            HarborComposeOverlay(prompt_template=tmp_path / "missing.j2")
        assert server.config.compose_overlay.configured
