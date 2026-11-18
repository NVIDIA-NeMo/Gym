# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Docker provider's Compose capabilities (shared storage and networking between sandboxes), on their own and
under the Compose adapter. No Docker daemon is needed: ``DockerProvider._run``, the one place every CLI call goes
through, is replaced by a recorder or by a fake daemon whose volumes and containers are folders."""

import asyncio
import json
import os
import re
import shlex
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from nemo_gym.sandbox.adapters.docker_compose import PLACEMENT_GROUP_KEY, AsyncSandboxCompose
from nemo_gym.sandbox.providers.base import (
    SandboxHandle,
    SandboxSpec,
    SupportsSandboxNetwork,
    SupportsSandboxSharedStorage,
)
from nemo_gym.sandbox.providers.docker import provider as docker_provider
from nemo_gym.sandbox.providers.docker.provider import DockerVolumeMount


pytestmark = pytest.mark.sandbox

FAKE_BINARY = "/usr/bin/docker"
VOLUME = docker_provider.DEFAULT_SHARED_VOLUME


@pytest.fixture(autouse=True)
def fake_binary(monkeypatch):
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: FAKE_BINARY)


def make_provider(monkeypatch, responder, **kwargs):
    """A provider whose CLI calls go to ``responder(argv)``; the calls are recorded on ``provider.calls``."""
    kwargs.setdefault("exec", {"exec_shell": "sh"})
    provider = docker_provider.DockerProvider(**kwargs)
    provider.calls = []

    async def run(argv, *, timeout_s, stdin=None):
        provider.calls.append(list(argv))
        return responder(list(argv))

    monkeypatch.setattr(provider, "_run", run)
    return provider


def handle(name="nemo-gym-x"):
    return SandboxHandle(name, "docker", docker_provider._DockerContainer(name=name, image="img"))


def test_the_provider_implements_both_compose_capabilities():
    provider = docker_provider.DockerProvider()
    assert isinstance(provider, SupportsSandboxSharedStorage)
    assert isinstance(provider, SupportsSandboxNetwork)


# --------------------------------------------------------------------------- #
# Shared storage
# --------------------------------------------------------------------------- #
def test_shared_volume_options_mount_a_directory_of_the_shared_volume():
    provider = docker_provider.DockerProvider()
    (mount,) = provider.shared_volume_options("compose-1/data", "/opt/data", read_only=True)["volumes"]
    assert mount == DockerVolumeMount(volume=VOLUME, target="/opt/data", subpath="compose-1/data", read_only=True)
    assert mount.mount_arg() == (
        f"type=volume,source={VOLUME},target=/opt/data,volume-subpath=compose-1/data,readonly,volume-nocopy"
    )
    # The adapter's helper mounts the whole volume to create and remove the collections' directories.
    (root,) = provider.shared_volume_options(None, "/tmp/compose-volumes")["volumes"]
    assert root.mount_arg() == f"type=volume,source={VOLUME},target=/tmp/compose-volumes,volume-nocopy"
    assert provider.shared_volume_metadata() == {}
    # A spelling of the same directory mounts it, not something else.
    assert provider.shared_volume_options("./compose-1//data", "/d")["volumes"][0].subpath == "compose-1/data"


def test_the_shared_volume_is_configurable_and_must_be_a_volume_name():
    provider = docker_provider.DockerProvider(create={"shared_volume": "gym-compose"})
    assert provider.shared_volume_options("p/v", "/v")["volumes"][0].volume == "gym-compose"
    for name in ("", "-leading", "has space", "a"):
        with pytest.raises(ValueError, match="shared_volume"):
            docker_provider.DockerCreateConfig(shared_volume=name)


@pytest.mark.parametrize(
    "source", ["", ".", "./", "/absolute", "../escape", "p/../escape", "p\\q", "p,readonly", 'p"q', "p\nq"]
)
def test_shared_volume_refuses_an_unsafe_source(source):
    with pytest.raises(ValueError, match="source"):
        docker_provider.DockerProvider().shared_volume_options(source, "/data")


@pytest.mark.parametrize("target", ["", "relative", "/data/../etc", "/data,readonly=false", "/a\x00b"])
def test_shared_volume_refuses_an_unsafe_target(target):
    with pytest.raises(ValueError, match="target"):
        docker_provider.DockerProvider().shared_volume_options("p/v", target)


def daemon_responder(*, api="1.47", volume_exists=False, create_code=0):
    def respond(argv):
        if argv[1] == "version":
            return 0, api + "\n", ""
        if argv[1:3] == ["volume", "inspect"]:
            return (0, VOLUME + "\n", "") if volume_exists else (1, "", f"Error response from daemon: get {VOLUME}")
        if argv[1:3] == ["volume", "create"]:
            return (create_code, "", "" if create_code == 0 else "permission denied")
        if argv[1] == "exec":
            return 0, docker_provider.READY_PROBE_EXPECTED, ""
        return 0, "", ""

    return respond


@pytest.mark.asyncio
async def test_create_with_shared_storage_checks_the_daemon_and_creates_the_labeled_volume_once(monkeypatch):
    provider = make_provider(monkeypatch, daemon_responder())
    options = provider.shared_volume_options("compose-1/data", "/data", read_only=True)
    spec = SandboxSpec(image="img", provider_options={**options, "run_args": ["--dns", "1.1.1.1"]})
    await provider.create(spec)
    await provider.create(spec)
    version, inspect, create, run = provider.calls[:4]
    assert version == [FAKE_BINARY, "version", "--format", "{{.Server.APIVersion}}"]
    assert inspect == [FAKE_BINARY, "volume", "inspect", "--format", "{{.Name}}", VOLUME]
    assert create == [FAKE_BINARY, "volume", "create", "--label", "nemo-gym.shared-storage=1", VOLUME]
    mount = run.index("--mount")
    assert run[mount + 1] == options["volumes"][0].mount_arg()
    assert "-v" not in run
    # Once per provider: the second create goes straight to `docker run`.
    assert [call[1] for call in provider.calls].count("version") == 1
    assert [call[1:3] for call in provider.calls].count(["volume", "create"]) == 1


@pytest.mark.asyncio
async def test_an_existing_shared_volume_is_used_as_it_is(monkeypatch):
    provider = make_provider(monkeypatch, daemon_responder(volume_exists=True))
    await provider.create(SandboxSpec(image="img", provider_options=provider.shared_volume_options(None, "/root")))
    assert not any(call[1:3] == ["volume", "create"] for call in provider.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responder,message",
    [
        # Engine 25 would ignore volume-subpath and mount the whole volume, every collection's directories with it.
        (daemon_responder(api="1.44"), "Engine 26.0"),
        (daemon_responder(api="Cannot connect"), "API version"),
        (daemon_responder(create_code=1), "cannot create the shared-storage volume"),
    ],
)
async def test_shared_storage_fails_clearly_before_any_container_is_created(monkeypatch, responder, message):
    provider = make_provider(monkeypatch, responder)
    spec = SandboxSpec(image="img", provider_options=provider.shared_volume_options("p/v", "/v"))
    with pytest.raises(docker_provider.DockerCreateError, match=message):
        await provider.create(spec)
    assert not any(call[1] == "run" for call in provider.calls)


@pytest.mark.asyncio
async def test_plain_volumes_need_no_daemon_check(monkeypatch):
    provider = make_provider(monkeypatch, daemon_responder())
    await provider.create(SandboxSpec(image="img", provider_options={"volumes": "/h:/c:ro"}))
    assert [call[1] for call in provider.calls] == ["run", "exec"]
    assert provider.calls[0][provider.calls[0].index("-v") + 1] == "/h:/c:ro"


@pytest.mark.asyncio
async def test_metadata_becomes_container_labels_beside_the_ownership_label(monkeypatch, caplog):
    provider = make_provider(monkeypatch, daemon_responder())
    metadata = {
        PLACEMENT_GROUP_KEY: "compose-abc",
        "note": "a=b c",
        docker_provider.SANDBOX_LABEL: "0",
        # The docker CLI splits `--label` on the first `=`, so this key would set the ownership label to `0=x`.
        f"{docker_provider.SANDBOX_LABEL}=0": "x",
        # The docker CLI refuses a label name that is empty or holds whitespace.
        "": "empty",
        "has space": "x",
        "tab\tkey": "x",
    }
    await provider.create(SandboxSpec(image="img", metadata=metadata))
    run = provider.calls[0]
    labels = [run[index + 1] for index, token in enumerate(run) if token == "--label"]
    # Reconciliation trusts the ownership label, so metadata cannot replace it; unusable keys are skipped.
    assert labels == [f"{docker_provider.SANDBOX_LABEL}=1", f"{PLACEMENT_GROUP_KEY}=compose-abc", "note=a=b c"]
    assert [label.split("=", 1)[0] for label in labels].count(docker_provider.SANDBOX_LABEL) == 1
    assert caplog.text.count("is not used as a container label") == 5


# --------------------------------------------------------------------------- #
# Networking
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("network", [None, "gym-net", "bridge"])
def test_networking_works_on_the_bridge_and_named_networks(network):
    docker_provider.DockerProvider(create={"network": network}).validate_networking()


@pytest.mark.asyncio
@pytest.mark.parametrize("network", ["none", "host", "container:other"])
async def test_networking_is_refused_where_a_container_has_no_address_of_its_own(monkeypatch, network):
    provider = make_provider(monkeypatch, daemon_responder(), create={"network": network})
    with pytest.raises(NotImplementedError, match=re.escape(repr(network))):
        provider.validate_networking()
    with pytest.raises(NotImplementedError):
        await provider.network_address(handle())
    with pytest.raises(NotImplementedError):
        await provider.set_hosts(handle(), {"db": "172.17.0.3"})
    assert provider.calls == []


def inspect_responder(networks, code=0):
    return lambda argv: (code, json.dumps(networks) if code == 0 else "", "" if code == 0 else "No such object: x")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "network,networks,expected",
    [
        (None, {"bridge": {"IPAddress": "172.17.0.4", "GlobalIPv6Address": ""}}, "172.17.0.4"),
        (
            "gym-net",
            {"bridge": {"IPAddress": "172.17.0.4"}, "gym-net": {"IPAddress": "10.9.0.2"}},
            "10.9.0.2",
        ),
        (None, {"v6": {"IPAddress": "", "GlobalIPv6Address": "fd00::5"}}, "fd00::5"),
    ],
)
async def test_network_address_is_the_containers_ip_from_inspect(monkeypatch, network, networks, expected):
    provider = make_provider(monkeypatch, inspect_responder(networks), create={"network": network})
    assert await provider.network_address(handle("nemo-gym-a")) == expected
    assert provider.calls == [[FAKE_BINARY, "inspect", "--format", "{{json .NetworkSettings.Networks}}", "nemo-gym-a"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responder,message",
    [
        (inspect_responder({"bridge": {"IPAddress": ""}}), "no IP address"),
        (inspect_responder(None), "no IP address"),
        (inspect_responder({}, code=1), "docker inspect failed"),
        (lambda argv: (0, "not json", ""), "unreadable"),
    ],
)
async def test_network_address_fails_when_the_container_has_none(monkeypatch, responder, message):
    provider = make_provider(monkeypatch, responder)
    with pytest.raises(RuntimeError, match=message):
        await provider.network_address(handle())


@pytest.mark.asyncio
async def test_set_hosts_appends_the_names_to_etc_hosts_as_root(monkeypatch):
    provider = make_provider(monkeypatch, lambda argv: (0, "", ""))
    await provider.set_hosts(
        handle("nemo-gym-a"), {"db": "172.17.0.3", "cache.internal": "fd00::1", "my_svc": "1.2.3.4"}
    )
    (call,) = provider.calls
    assert call[:5] == [FAKE_BINARY, "exec", "--user", "0", "nemo-gym-a"]
    command = call[-1]
    assert command.endswith(" >> /etc/hosts")
    assert {"172.17.0.3 db\n", "fd00::1 cache.internal\n", "1.2.3.4 my_svc\n"} <= set(shlex.split(command))
    provider.calls.clear()
    await provider.set_hosts(handle(), {})
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hosts",
    [{"bad\nname": "10.2.3.4"}, {"db": "127.0.0.1;touch /x"}, {"-bad": "10.2.3.4"}, {"two names": "10.2.3.4"}],
)
async def test_set_hosts_refuses_what_is_not_a_hostname_and_an_address(monkeypatch, hosts):
    provider = make_provider(monkeypatch, lambda argv: (0, "", ""))
    with pytest.raises(ValueError):
        await provider.set_hosts(handle(), hosts)
    assert provider.calls == []


@pytest.mark.asyncio
async def test_set_hosts_fails_when_the_file_cannot_be_written(monkeypatch):
    provider = make_provider(monkeypatch, lambda argv: (1, "", "sh: can't create /etc/hosts: Read-only file system"))
    with pytest.raises(RuntimeError, match="Read-only file system"):
        await provider.set_hosts(handle(), {"db": "172.17.0.3"})


# --------------------------------------------------------------------------- #
# The Compose adapter on the Docker provider
# --------------------------------------------------------------------------- #
# Absolute paths in a command: at its start or after whitespace, a quote, `=`, `(` or a redirection.
_PATH = re.compile(r"(?<![^\s'\"=(<>])/[^\s'\";|&()<>]*")


class FakeDockerDaemon:
    """Answers ``DockerProvider._run`` as a Docker Engine 26 daemon would, with real files.

    A named volume is a folder. ``docker run`` makes a container folder from its image's files and records its
    ``--mount`` targets, each backed by its volume's folder or the ``volume-subpath`` inside it, which must exist.
    ``docker exec`` runs the command in a host shell with every absolute path mapped through the container's mounts,
    else into its folder, which is how the copy-up and the cleanup reach the volume; ``docker exec -d`` (a service's
    command) returns once the command started, and ``docker rm`` kills what it left running.
    """

    def __init__(self, root: Path, images: dict[str, dict[str, str]]):
        self.root = root
        self.images = images
        self.volumes: dict[str, dict[str, str]] = {}
        self.containers: dict[str, Any] = {}
        self.runs: list[list[str]] = []
        self.removed: list[str] = []

    def volume_dir(self, name: str) -> Path:
        return self.root / "volumes" / name

    def host_path(self, container, path: str) -> str:
        for target in sorted(container.mounts, key=len, reverse=True):
            if path == target or path.startswith(target.rstrip("/") + "/"):
                return str(container.mounts[target]) + path[len(target) :]
        return str(container.fs) + path

    def rewrite(self, container, text: str) -> str:
        return _PATH.sub(
            lambda match: match[0] if match[0].startswith("/dev/") else self.host_path(container, match[0]), text
        )

    async def __call__(self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None):
        command = argv[1]
        if command == "version":
            return 0, "1.47\n", ""
        if argv[1:3] == ["volume", "inspect"]:
            name = argv[-1]
            return (0, name + "\n", "") if name in self.volumes else (1, "", f"Error: No such volume: {name}")
        if argv[1:3] == ["volume", "create"]:
            labels = dict(argv[index + 1].split("=", 1) for index, token in enumerate(argv) if token == "--label")
            self.volumes[argv[-1]] = labels
            self.volume_dir(argv[-1]).mkdir(parents=True)
            return 0, argv[-1] + "\n", ""
        if command == "run":
            return self.run(argv)
        if command == "exec":
            return await self.exec(argv)
        if command == "cp":
            name, path = argv[3].split(":", 1)
            container = self.containers[name]
            Path(self.host_path(container, path)).write_text(self.rewrite(container, Path(argv[2]).read_text()))
            return 0, "", ""
        if command == "inspect":
            container = self.containers[argv[-1]]
            return 0, json.dumps({"bridge": {"IPAddress": container.address, "GlobalIPv6Address": ""}}), ""
        if command == "rm":
            for process in self.containers.pop(argv[-1]).detached:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                await process.wait()
            self.removed.append(argv[-1])
            return 0, argv[-1], ""
        raise AssertionError(f"unexpected docker command: {argv}")

    def run(self, argv: list[str]):
        self.runs.append(argv)
        name = argv[argv.index("--name") + 1]
        entrypoint = argv.index("--entrypoint")
        image = argv[entrypoint + 2]
        options = argv[:entrypoint]
        fs = self.root / "containers" / name
        (fs / "etc").mkdir(parents=True)
        (fs / "etc" / "hosts").write_text("127.0.0.1\tlocalhost\n")
        for path, content in self.images[image].items():
            (fs / path.lstrip("/")).parent.mkdir(parents=True, exist_ok=True)
            (fs / path.lstrip("/")).write_text(content)
        mounts = {}
        for index, token in enumerate(options):
            if token != "--mount":
                continue
            fields = dict(field.partition("=")[::2] for field in options[index + 1].split(","))
            if fields["source"] not in self.volumes:
                return 125, "", f"docker: Error response from daemon: volume {fields['source']} not found"
            backing = self.volume_dir(fields["source"]) / fields.get("volume-subpath", "")
            if not backing.is_dir():
                return 125, "", f"docker: Error response from daemon: subpath {backing} does not exist"
            mounts[fields["target"]] = backing
        workdir = options[options.index("-w") + 1] if "-w" in options else "/"
        labels = dict(options[index + 1].split("=", 1) for index, token in enumerate(options) if token == "--label")
        container = SimpleNamespace(
            image=image,
            fs=fs,
            mounts=mounts,
            workdir=workdir,
            labels=labels,
            address=f"172.17.0.{len(self.runs) + 1}",
            detached=[],
        )
        self.containers[name] = container
        Path(self.host_path(container, workdir)).mkdir(parents=True, exist_ok=True)
        return 0, name, ""

    async def exec(self, argv: list[str]):
        flags, index = {"env": {}}, 2
        while argv[index].startswith("-"):
            if argv[index] in ("-i", "-d"):
                flags[argv[index]] = True
                index += 1
                continue
            value = argv[index + 1]
            if argv[index] == "--env":
                key, _, item = value.partition("=")
                flags["env"][key] = item
            else:
                flags[argv[index]] = value
            index += 2
        container = self.containers[argv[index]]
        command = self.rewrite(container, argv[index + 3])
        cwd = self.host_path(container, flags.get("-w", container.workdir))
        output = asyncio.subprocess.DEVNULL if "-d" in flags else asyncio.subprocess.PIPE
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            cwd=cwd,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **flags["env"]},
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
        if "-d" in flags:
            container.detached.append(process)
            return 0, "", ""
        try:
            out, err = await process.communicate()
        except asyncio.CancelledError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            raise
        return process.returncode, out.decode(), err.decode()


FIXTURE_IMAGE = "registry.example/fixture-task@sha256:" + "f" * 64
API_IMAGE = "registry.example/api:1"
IMAGES = {
    "alpine:3.22": {},
    FIXTURE_IMAGE: {"/opt/gym-fixture/bundled-output-hashes.json": '{"sample": "abc"}\n'},
    API_IMAGE: {},
}


def collection(daemon: FakeDockerDaemon, monkeypatch, document: dict) -> AsyncSandboxCompose:
    """A collection on a fresh Docker provider whose CLI calls reach ``daemon`` (the Harbor server makes a provider
    per collection from its config), started from ``document`` as the server resolves it."""
    provider = docker_provider.DockerProvider(exec={"exec_shell": "sh"})
    monkeypatch.setattr(provider, "_run", daemon)
    group = AsyncSandboxCompose(provider, "compose.yaml", poll_interval_s=0.01)
    group._load = Mock(return_value=document)
    return group


def fixture_volume_document() -> dict:
    """A Compose file with a read-only fixture volume, as the server resolves it: ``main`` from the built image, the fixture volume read-only."""
    return {
        "services": {
            "main": {
                "image": FIXTURE_IMAGE,
                "command": ["true"],
                "working_dir": "/workspace",
                "environment": {"GYM_AGENT": "codex_agent", "GYM_MODEL": "openai/gpt"},
                "volumes": [
                    {"type": "volume", "source": "gym_fixture", "target": "/opt/gym-fixture", "read_only": True}
                ],
            }
        },
        "volumes": {"gym_fixture": {}},
    }


@pytest.mark.asyncio
async def test_a_read_only_named_volume_is_seeded_from_the_image_and_removed_with_the_collection(
    tmp_path, monkeypatch
):
    daemon = FakeDockerDaemon(tmp_path, IMAGES)
    group = collection(daemon, monkeypatch, fixture_volume_document())
    project_dir = daemon.volume_dir(VOLUME) / group.project / "gym_fixture"
    async with group:
        # The volume was created once, labeled for an operator to find.
        assert daemon.volumes == {VOLUME: {"nemo-gym.shared-storage": "1"}}
        helper, seed, main = daemon.runs
        assert (
            helper[helper.index("--mount") + 1]
            == f"type=volume,source={VOLUME},target=/tmp/compose-volumes,volume-nocopy"
        )
        # Copy-up: a sandbox of main's own image mounts the collection's directory and copies the image's files in.
        assert seed[seed.index("--entrypoint") + 2] == FIXTURE_IMAGE
        assert seed[seed.index("--mount") + 1] == (
            f"type=volume,source={VOLUME},target=/.nemo-gym-copy-up,"
            f"volume-subpath={group.project}/gym_fixture,volume-nocopy"
        )
        assert (project_dir / "bundled-output-hashes.json").read_text() == '{"sample": "abc"}\n'
        # main mounts that directory read-only where the Compose file put the volume, and reads the image's files.
        assert main[main.index("--mount") + 1] == (
            f"type=volume,source={VOLUME},target=/opt/gym-fixture,"
            f"volume-subpath={group.project}/gym_fixture,readonly,volume-nocopy"
        )
        result = await group.services["main"].exec("cat /opt/gym-fixture/bundled-output-hashes.json")
        assert (result.return_code, result.stdout) == (0, '{"sample": "abc"}\n')
        # Every container of the collection is labeled with its project, for an operator or a reaper to find.
        for run in daemon.runs:
            assert f"{docker_provider.SANDBOX_LABEL}=1" in run
            assert f"{PLACEMENT_GROUP_KEY}={group.project}" in run
        # The seed is gone as soon as it copied.
        assert daemon.removed == [seed[seed.index("--name") + 1]]
    # Stopping removes the collection's directory and every container; the shared volume itself stays.
    assert not (daemon.volume_dir(VOLUME) / group.project).exists()
    assert daemon.volume_dir(VOLUME).is_dir()
    assert daemon.containers == {}
    assert len(daemon.removed) == 3


@pytest.mark.asyncio
async def test_every_attempt_gets_fresh_volumes(tmp_path, monkeypatch):
    """Harbor runs each trial under its own Compose project and removes its volumes with it: a retry or a concurrent
    rollout of the same task never sees another attempt's volume. A volume without copy-up starts empty."""
    daemon = FakeDockerDaemon(tmp_path, IMAGES)
    document = fixture_volume_document()
    document["services"]["main"]["volumes"].append(
        {"type": "volume", "source": "docker-data", "target": "/var/lib/docker", "volume": {"nocopy": True}}
    )
    document["volumes"]["docker-data"] = {}
    first, second = (collection(daemon, monkeypatch, json.loads(json.dumps(document))) for _ in range(2))
    async with first, second:
        assert first.project != second.project
        written = await first.services["main"].exec("echo first > /var/lib/docker/marker")
        assert written.return_code == 0
        listed = await second.services["main"].exec("ls -A /var/lib/docker")
        assert (listed.return_code, listed.stdout) == (0, "")
        for group in (first, second):
            assert (daemon.volume_dir(VOLUME) / group.project / "gym_fixture" / "bundled-output-hashes.json").is_file()
        # The volume was created by the first provider; the second found it.
        assert daemon.volumes == {VOLUME: {"nemo-gym.shared-storage": "1"}}
    assert list(daemon.volume_dir(VOLUME).iterdir()) == []


@pytest.mark.asyncio
async def test_services_reach_each_other_by_name(tmp_path, monkeypatch):
    daemon = FakeDockerDaemon(tmp_path, IMAGES)
    document = {
        "services": {
            "main": {
                "image": FIXTURE_IMAGE,
                "command": ["true"],
                "depends_on": {"api": {"condition": "service_started"}},
            },
            "api": {
                "image": API_IMAGE,
                "command": ["true"],
                "networks": {"default": {"aliases": ["api.internal"]}},
            },
        }
    }
    group = collection(daemon, monkeypatch, document)
    async with group:
        addresses = {
            name: daemon.containers[sandbox._require_handle().sandbox_id].address
            for name, sandbox in group.services.items()
        }
        for name, sandbox in group.services.items():
            hosts = (daemon.containers[sandbox._require_handle().sandbox_id].fs / "etc" / "hosts").read_text()
            assert f"{addresses['main']} main\n" in hosts
            assert f"{addresses['api']} api\n" in hosts
            assert f"{addresses['api']} api.internal\n" in hosts
    assert daemon.containers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service,options,names",
    [
        ("worker_", {}, ["worker_"]),
        ("worker", {"networks": {"default": {"aliases": ["cache-"]}}}, ["worker", "cache-"]),
    ],
)
async def test_every_name_the_adapter_admits_is_written_to_etc_hosts(tmp_path, monkeypatch, service, options, names):
    """A service name or alias may end in ``_`` or ``-``: the adapter admits it, so the hosts file takes it too."""
    daemon = FakeDockerDaemon(tmp_path, IMAGES)
    document = {"services": {service: {"image": API_IMAGE, "command": ["true"], **options}}}
    group = collection(daemon, monkeypatch, document)
    async with group:
        container = daemon.containers[group.services[service]._require_handle().sandbox_id]
        hosts = (container.fs / "etc" / "hosts").read_text()
        for name in names:
            assert f"{container.address} {name}\n" in hosts
    assert daemon.containers == {}
