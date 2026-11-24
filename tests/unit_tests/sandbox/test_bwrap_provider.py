# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the bwrap provider. They run anywhere: a fake ``bwrap`` script strips the
bwrap flags and runs the command on the host, so argv, env, process tracking and file
transfer are exercised without namespaces. ``test_bwrap_integration`` runs the real binary."""

import asyncio
import errno
import json
import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.bwrap import (
    BwrapCreateError,
    BwrapCreateVerificationError,
    BwrapPathError,
    BwrapProvider,
)
from nemo_gym.sandbox.providers.bwrap import provider as bwrap_module
from nemo_gym.sandbox.providers.registry import create_provider, get_provider_class


pytestmark = pytest.mark.sandbox

FAKE_BWRAP = """#!/bin/bash
# bash, not sh: dash (Debian/Ubuntu /bin/sh) rejects the multi-digit --info-fd in `>&"$info"`.
# Drop every bwrap option, then run the command after `--` on the host. Like bwrap, report the
# pid of the namespace init (here: the command itself, after exec) on --info-fd.
info=
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
  if [ "$1" = "--info-fd" ]; then info=$2; shift; fi
  shift
done
shift
if [ -n "$info" ]; then
  printf '{\\n    "child-pid": %d\\n}\\n' "$$" >&"$info"
  eval "exec $info>&-"
fi
exec "$@"
"""


@pytest.fixture
def system(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in ("usr", "bin", "etc_ssl", "resolv.conf", "venv"):
        path = tmp_path / "sys" / name
        if "." in name:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("nameserver 127.0.0.1\n")
        else:
            path.mkdir(parents=True)
        paths[name] = path
    for name in ("ld.so.cache", "ld.so.conf"):
        (tmp_path / "sys" / name).write_text("")
    bwrap = tmp_path / "bwrap"
    bwrap.write_text(FAKE_BWRAP)
    bwrap.chmod(0o755)
    paths["bwrap"] = bwrap
    paths["base"] = tmp_path / "base"
    paths["sys"] = tmp_path / "sys"
    return paths


def make_config(system: dict[str, Path], **overrides) -> dict:
    config = {
        "bwrap_path": str(system["bwrap"]),
        "base_dir": str(system["base"]),
        "ro_binds": [
            str(system["usr"]),
            str(system["bin"]),
            {"path": str(system["sys"] / "lib64"), "optional": True},
            {"path": str(system["sys"] / "ld.so.*"), "optional": True},
        ],
        "network_host_ro_binds": [
            str(system["resolv.conf"]),
            str(system["etc_ssl"]),
            {"path": str(system["sys"] / "pki"), "optional": True},
        ],
        "env_passthrough": ["PATH"],
        "probe_on_create": False,
    }
    config.update(overrides)
    return config


def make_provider(system: dict[str, Path], **overrides) -> BwrapProvider:
    return BwrapProvider(**make_config(system, **overrides))


async def start(provider: BwrapProvider, **spec) -> AsyncSandbox:
    return await AsyncSandbox(provider, owns_provider=False).start(SandboxSpec(**spec))


def bind_pairs(argv: list[str], flag: str) -> list[tuple[str, str]]:
    return [(argv[i + 1], argv[i + 2]) for i, arg in enumerate(argv) if arg == flag]


# ----------------------------------------------------------------------------- config / argv


def test_registry_resolves_bwrap(system) -> None:
    assert get_provider_class("bwrap") is BwrapProvider
    provider = create_provider({"bwrap": make_config(system)})
    assert isinstance(provider, BwrapProvider)


def test_shipped_config_resolves(system, monkeypatch) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.sandbox.config import resolve_provider_config

    monkeypatch.setenv("NEMO_GYM_BWRAP_PATH", str(system["bwrap"]))
    path = Path(__file__).parents[3] / "nemo_gym/sandbox/providers/bwrap/configs/bwrap.yaml"
    config = resolve_provider_config("sandbox", OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    settings = bwrap_module.BwrapConfig(**config["bwrap"])
    assert settings.network == "none" and settings.remount_root_ro and settings.base_dir is None
    assert [bind.path for bind in settings.ro_binds][:3] == ["/usr", "/bin", "/lib"]
    assert "/venv" in [bind.path for bind in settings.ro_binds]
    assert "/etc/ca-certificates" in [bind.path for bind in settings.network_host_ro_binds]


async def test_argv_network_none(system) -> None:
    provider = make_provider(system)
    sandbox = await start(provider, workdir="/workspace")
    inst = sandbox._handle.raw
    argv = provider.bwrap_argv(inst, "echo hi")
    assert argv[0] == str(system["bwrap"])
    assert argv[1:8] == [
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
    ]
    assert argv[8:10] == ["--cap-drop", "ALL"]
    sys_dir = system["sys"]
    assert bind_pairs(argv, "--ro-bind") == [
        (str(system["usr"]), str(system["usr"])),
        (str(system["bin"]), str(system["bin"])),
        (str(sys_dir / "ld.so.cache"), str(sys_dir / "ld.so.cache")),
        (str(sys_dir / "ld.so.conf"), str(sys_dir / "ld.so.conf")),
    ]
    # The workspace's host-path bind exists only outside sandbox-writable space (tmp_path is under /tmp on Linux).
    alias = [(str(inst.workspace), str(inst.workspace))] if provider._binds_workspace_host_path(inst) else []
    assert (
        bind_pairs(argv, "--bind")
        == [
            (str(inst.tmp), "/tmp"),
            (str(inst.home), "/home/sandbox"),
            (str(inst.workspace), "/workspace"),
        ]
        + alias
    )
    assert ["--proc", "/proc"] == argv[argv.index("--proc") : argv.index("--proc") + 2]
    assert ["--dev", "/dev"] == argv[argv.index("--dev") : argv.index("--dev") + 2]
    # The root is remounted read-only after the last bind, before the command.
    remount = argv.index("--remount-ro")
    assert argv[remount + 1] == "/"
    assert remount > max(i for i, arg in enumerate(argv) if arg in ("--bind", "--ro-bind"))
    assert argv[-6:] == ["--chdir", "/workspace", "--", "/bin/bash", "-c", "echo hi"]
    assert "--clearenv" not in argv  # the environment is passed whole to bwrap instead
    assert provider.bwrap_argv(inst, "x", cwd="/tmp")[-6:-4] == ["--chdir", "/tmp"]
    await sandbox.stop()


@pytest.mark.parametrize("euid, kept", [(0, True), (1000, False)])
async def test_nested_userns_keeps_only_setfcap_and_only_for_root(system, monkeypatch, euid, kept) -> None:
    provider = make_provider(system, allow_nested_userns=True)
    sandbox = await start(provider, workdir="/workspace")
    monkeypatch.setattr("nemo_gym.sandbox.providers.bwrap.provider.os.geteuid", lambda: euid)
    argv = provider.bwrap_argv(sandbox._handle.raw, "true")
    # A sandboxed root maps uid 0 into a nested user namespace only with CAP_SETFCAP.
    assert (argv[8:12] == ["--cap-drop", "ALL", "--cap-add", "CAP_SETFCAP"]) is kept
    assert argv.count("--cap-add") == int(kept)
    assert "--cap-add" not in make_provider(system).bwrap_argv(sandbox._handle.raw, "true")
    await sandbox.stop()


async def test_argv_network_host_adds_resolution_binds(system) -> None:
    provider = make_provider(system, network="host", remount_root_ro=False, extra_args=["--hostname", "box"])
    sandbox = await start(provider)
    argv = provider.bwrap_argv(sandbox._handle.raw, "true")
    assert "--unshare-net" not in argv
    ro = [src for src, _ in bind_pairs(argv, "--ro-bind")]
    assert str(system["resolv.conf"]) in ro and str(system["etc_ssl"]) in ro
    assert not any(src.endswith("/pki") for src in ro)  # optional and absent
    assert "--remount-ro" not in argv
    assert argv[-6:-4] == ["--hostname", "box"]  # extra_args go last, before the command
    assert (await sandbox.serialize())["network"] == "host"
    await sandbox.stop()


def test_required_bind_missing_and_invalid_config(system) -> None:
    with pytest.raises(FileNotFoundError, match="Required bwrap read-only bind"):
        make_provider(system, ro_binds=[str(system["sys"] / "absent")])
    # Host-network binds are only required when network is host.
    make_provider(system, network_host_ro_binds=["/definitely/absent"])
    with pytest.raises(FileNotFoundError):
        make_provider(system, network="host", network_host_ro_binds=["/definitely/absent"])
    with pytest.raises(ValueError, match="absolute"):
        make_provider(system, bwrap_path="bwrap")
    with pytest.raises(RuntimeError, match="not an executable"):
        make_provider(system, bwrap_path=str(system["sys"] / "nope"))
    with pytest.raises(ValueError, match="network"):
        make_provider(system, network="bridge")
    with pytest.raises(ValueError, match="workdir"):
        make_provider(system, workdir="/")
    with pytest.raises(ValueError, match="POSIX"):
        make_provider(system, env={"BAD-NAME": "x"})
    with pytest.raises(TypeError):
        make_provider(system, unknown_option=True)


def test_mount_targets_inside_read_only_binds_are_refused(system) -> None:
    usr = str(system["usr"])
    with pytest.raises(ValueError, match="inside the read-only bind"):
        make_provider(system, mounts=[{"src": str(system["venv"]), "dst": f"{usr}/codex-runtime"}])
    with pytest.raises(ValueError, match="inside the read-only bind"):
        make_provider(system, workdir=f"{usr}/work")
    with pytest.raises(ValueError, match="'src:dst"):
        make_provider(system, mounts=["a:b:c:d"])
    with pytest.raises(ValueError, match="without '..'"):
        make_provider(system, mounts=[{"src": str(system["venv"]), "dst": "/x/../y"}])
    with pytest.raises(FileNotFoundError):
        make_provider(system, mounts=[{"src": "/definitely/absent", "dst": "/codex-runtime"}])


async def test_mounts_from_config_and_provider_options(system) -> None:
    provider = make_provider(system, mounts=[{"src": str(system["venv"]), "dst": "/codex-runtime"}])
    sandbox = await start(
        provider,
        workdir="/workspace",
        provider_options={"mounts": [f"{system['bin']}:/extra-bin:rw", {"src": str(system["usr"]), "dst": "/u2"}]},
    )
    argv = provider.bwrap_argv(sandbox._handle.raw, "true")
    assert (str(system["venv"]), "/codex-runtime") in bind_pairs(argv, "--ro-bind")
    assert (str(system["usr"]), "/u2") in bind_pairs(argv, "--ro-bind")
    assert bind_pairs(argv, "--bind")[-1] == (str(system["bin"]), "/extra-bin")
    # Extra mounts come after the sandbox's own binds, so they overlay them.
    assert argv.index("/codex-runtime") > argv.index("/workspace")
    await sandbox.stop()
    for options, error in (
        ({"mounts": [{"src": str(system["venv"]), "dst": f"{system['usr']}/x"}]}, "read-only bind"),
        ({"mounts": ["/definitely/absent:/x"]}, "does not exist"),
        ({"binds": []}, "unsupported bwrap provider_options"),
    ):
        with pytest.raises(BwrapCreateError, match=error):
            await start(provider, provider_options=options)


async def test_bwraps_own_setup_failure_is_a_sandbox_error(system, tmp_path) -> None:
    # A bwrap that fails before starting the command never reports a child-pid on --info-fd.
    failing = tmp_path / "failing-bwrap"
    failing.write_text('#!/bin/sh\necho "bwrap: Can\'t mount proc on /newroot/proc" >&2\nexit 1\n')
    failing.chmod(0o755)
    provider = make_provider(system, bwrap_path=str(failing))
    sandbox = await start(provider)
    result = await sandbox.exec("true")
    assert result.error_type == "sandbox" and result.return_code == 125 and "Can't mount proc" in result.stderr
    await sandbox.stop()


async def test_mount_targets_overlapping_the_sandboxs_own_dirs_are_refused(system) -> None:
    # Inside /tmp, home or the workdir, bwrap would create the mount point in sandbox-writable space (and follow
    # a symlink planted there); at or above one, the mount would shadow it.
    for dst in ("/tmp/codex-runtime", "/tmp", "/home", "/home/sandbox/x", "/workspace/lib"):
        with pytest.raises(ValueError, match="overlaps the sandbox's own"):
            make_provider(system, mounts=[{"src": str(system["venv"]), "dst": dst}])
    provider = make_provider(system)
    for dst, workdir in (("/tmp/x", None), ("/task/sub", "/task"), ("/app", "/app")):
        with pytest.raises(BwrapCreateError, match="overlaps the sandbox's own"):
            await start(provider, workdir=workdir, provider_options={"mounts": [f"{system['venv']}:{dst}"]})
    # A config mount is checked against each sandbox's own workdir too.
    provider = make_provider(system, mounts=[{"src": str(system["venv"]), "dst": "/task/runtime"}])
    with pytest.raises(BwrapCreateError, match="overlaps the sandbox's own"):
        await start(provider, workdir="/task")


async def test_workspace_host_path_is_bound_only_outside_sandbox_writable_dirs(system) -> None:
    provider = make_provider(system)
    sandbox = await start(provider, workdir="/workspace")
    inst = sandbox._handle.raw
    host = str(inst.workspace)
    argv = provider.bwrap_argv(inst, "true")
    # tmp_path is under /tmp on Linux (no alias) and elsewhere on macOS (alias).
    expected = not host.startswith("/tmp/")
    assert provider._binds_workspace_host_path(inst) is expected
    assert ((host, host) in bind_pairs(argv, "--bind")) is expected
    # The default base_dir is under the host's /tmp: there the alias would need a mount point in sandbox space.
    for inside in ("/tmp/nemo-gym-bwrap-0/sb/workspace", "/home/sandbox/w/workspace", "/workspace/w/workspace"):
        stand_in = SimpleNamespace(workspace=Path(inside), workdir="/workspace")
        assert provider._binds_workspace_host_path(stand_in) is False
    await sandbox.stop()


# ----------------------------------------------------------------------------- env


async def test_environment_is_whitelisted_and_gpus_hidden(system, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/host/bin:/usr/bin:/bin")
    monkeypatch.setenv("SECRET_TOKEN", "leak")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    provider = make_provider(system, env={"LANG": "en_US.UTF-8", "TRITON_CACHE_DIR": "/tmp/triton"})
    sandbox = await start(provider, env={"OMP_NUM_THREADS": "2"})
    env = provider._environment(sandbox._handle.raw, {"EXTRA": "1", "CUDA_VISIBLE_DEVICES": "3"})
    assert env == {
        "PATH": "/host/bin:/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        "TRITON_CACHE_DIR": "/tmp/triton",
        "HOME": "/home/sandbox",
        "OMP_NUM_THREADS": "2",
        "EXTRA": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
    }
    # The real process sees exactly that environment.
    result = await sandbox.exec("env | sort", env={"EXTRA": "1"})
    seen = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert "SECRET_TOKEN" not in seen
    assert seen["OMP_NUM_THREADS"] == "2" and seen["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert seen["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(ValueError, match="POSIX"):
        await sandbox.exec("true", env={"A B": "x"})
    await sandbox.stop()
    monkeypatch.delenv("PATH")
    defaults = make_provider(system, hide_gpus=False)
    sandbox = await start(defaults)
    env = defaults._environment(sandbox._handle.raw, None)
    assert env["PATH"] == bwrap_module.DEFAULT_PATH and env["LANG"] == "C.UTF-8"
    assert "CUDA_VISIBLE_DEVICES" not in env
    await sandbox.stop()


# ----------------------------------------------------------------------------- exec


async def test_exec_runs_and_classifies_failures(system) -> None:
    provider = make_provider(system)
    sandbox = await start(provider)
    result = await sandbox.exec("printf out; printf err >&2; exit 3")
    assert (result.stdout, result.stderr, result.return_code, result.error_type) == ("out", "err", 3, None)
    with pytest.raises(ValueError, match="user"):
        await sandbox.exec("true", user="root")
    timed_out = await sandbox.exec("sleep 30", timeout_s=0.3)
    assert timed_out.error_type == "timeout" and timed_out.return_code == 125
    assert not provider._processes and not list(sandbox._handle.raw.run_dir.iterdir())
    # A command that prints bwrap-like text fails as itself: bwrap had reported its namespace init.
    spoofed = await sandbox.exec('echo "bwrap: Can\'t mount proc on /newroot/proc" >&2; exit 3')
    assert spoofed.error_type is None and spoofed.return_code == 3
    await sandbox.stop()
    stopped = await provider.exec(sandbox._handle, "true")
    assert stopped.error_type == "sandbox" and "stopped" in stopped.stderr


async def test_own_directories_must_be_disjoint(system) -> None:
    # Nested, a later bind hides an earlier one from commands while transfers still resolve to it.
    provider = make_provider(system)
    for workdir in ("/home", "/home/sandbox", "/home/sandbox/w", "/tmp/w"):
        with pytest.raises(BwrapCreateError, match="overlap"):
            await start(provider, workdir=workdir)
    with pytest.raises(ValueError, match="overlap"):
        make_provider(system, workdir="/task", home_path="/task/home")
    with pytest.raises(BwrapCreateError, match="overlap"):
        await start(make_provider(system, home_path="/task/home"), workdir="/task")
    sandbox = await start(provider, workdir="/task")  # disjoint custom workdir
    await sandbox.stop()


async def test_a_pending_exec_of_another_instance_blocks_the_proof_and_the_stop(system, tmp_path) -> None:
    # Provider A queues an exec for a shared sandbox; borrower B must neither prove it empty nor stop past it.
    owner = make_provider(system, exec_concurrency=1, kill_timeout_s=1)
    shared = await start(owner)
    other = await start(owner)
    borrower = make_provider(system, kill_timeout_s=1)
    borrowed = await borrower.connect(await shared.serialize())
    ran = tmp_path / "queued-exec-ran"
    holder = asyncio.create_task(owner.exec(other._handle, "sleep 30"))
    while not owner._processes.get(other._handle.sandbox_id):
        await asyncio.sleep(0.01)
    queued = asyncio.create_task(owner.exec(shared._handle, f"touch {ran}"))
    while not [e for e in shared._handle.raw.run_dir.iterdir() if e.name.startswith("pending-")]:
        await asyncio.sleep(0.01)
    assert await borrower.exec_processes_terminated(borrowed) is False
    await borrower.close(borrowed)  # waits (bounded) for the pending exec, then stops
    await owner.close(other._handle)
    result = await asyncio.wait_for(queued, 10)
    assert result.error_type == "sandbox" and "stopped" in result.stderr
    assert not ran.exists()
    await holder


async def test_a_dead_owners_pending_record_does_not_block_the_proof(system) -> None:
    provider = make_provider(system)
    sandbox = await start(provider)
    record = sandbox._handle.raw.run_dir / "pending-stale"
    record.write_text("2147483647 1")  # an owner pid that cannot exist
    assert await provider.exec_processes_terminated(sandbox._handle) is True
    assert not record.exists()
    await sandbox.stop()


async def test_transfers_resolve_to_the_bind_a_command_sees(system, tmp_path) -> None:
    hidden, visible = tmp_path / "hidden", tmp_path / "visible"
    for root, text in ((hidden, "hidden"), (visible, "visible")):
        (root / "sub").mkdir(parents=True)
        (root / "f.txt").write_text(text)
        (root / "sub" / "g.txt").write_text(text)
    # A later read-only mount at the same destination hides the earlier writable one.
    provider = make_provider(system, mounts=[{"src": str(hidden), "dst": "/tools", "ro": False}])
    sandbox = await start(provider, provider_options={"mounts": [{"src": str(visible), "dst": "/tools", "ro": True}]})
    await sandbox.download("/tools/f.txt", tmp_path / "got.txt")
    assert (tmp_path / "got.txt").read_text() == "visible"
    with pytest.raises(BwrapPathError, match="read-only"):
        await sandbox.upload(_text(tmp_path / "src", "x"), "/tools/new.txt")
    await sandbox.stop()
    # A later parent mount hides an earlier child mount.
    provider = make_provider(system, mounts=[{"src": str(hidden / "sub"), "dst": "/tools/sub"}])
    sandbox = await start(provider, provider_options={"mounts": [{"src": str(visible), "dst": "/tools", "ro": True}]})
    await sandbox.download("/tools/sub/g.txt", tmp_path / "got2.txt")
    assert (tmp_path / "got2.txt").read_text() == "visible"
    await sandbox.stop()


async def test_a_regular_file_mount_is_transferable_at_its_destination(system, tmp_path) -> None:
    tool = _text(tmp_path / "tool", "#!/bin/sh\necho tool\n")
    provider = make_provider(system, mounts=[{"src": str(tool), "dst": "/tool", "ro": True}])
    sandbox = await start(provider)
    await sandbox.download("/tool", tmp_path / "got")
    assert (tmp_path / "got").read_text() == tool.read_text()
    assert await provider.exists(sandbox._handle, "/tool")
    with pytest.raises(BwrapPathError, match="read-only"):
        await sandbox.upload(tool, "/tool")
    with pytest.raises(BwrapPathError, match="below a file mount"):
        await sandbox.download("/tool/child", tmp_path / "got2")
    await sandbox.stop()


async def test_an_exec_queued_for_a_slot_never_starts_after_stop(system, tmp_path) -> None:
    provider = make_provider(system, exec_concurrency=1)
    sandbox = await start(provider)
    handle = sandbox._handle
    ran = tmp_path / "queued-exec-ran"
    holder = asyncio.create_task(provider.exec(handle, "sleep 30"))
    while not provider._processes.get(handle.sandbox_id):
        await asyncio.sleep(0.01)
    queued = asyncio.create_task(provider.exec(handle, f"touch {ran}"))
    await asyncio.sleep(0.05)
    # Queued: a process may still run, so the provider cannot prove the sandbox empty.
    assert await provider.exec_processes_terminated(handle) is False
    await provider.close(handle)
    result = await asyncio.wait_for(queued, 10)
    assert result.error_type == "sandbox" and "stopped" in result.stderr
    assert not ran.exists()
    assert (await holder).return_code != 0


async def test_exec_concurrency_is_bounded(system) -> None:
    provider = make_provider(system, exec_concurrency=2)
    sandbox = await start(provider)
    marker = sandbox._handle.raw.tmp
    started = asyncio.get_running_loop().time()
    results = await asyncio.gather(*(sandbox.exec(f"sleep 0.3; touch {marker}/{i}") for i in range(4)))
    assert all(result.return_code == 0 for result in results)
    assert asyncio.get_running_loop().time() - started >= 0.55
    await sandbox.stop()


async def test_create_probe_failure_cleans_up(system, tmp_path) -> None:
    failing = tmp_path / "failing-bwrap"
    failing.write_text("#!/bin/sh\necho 'bwrap: setting up uid map: Permission denied' >&2\nexit 1\n")
    failing.chmod(0o755)
    provider = make_provider(system, bwrap_path=str(failing), probe_on_create=True)
    with pytest.raises(BwrapCreateVerificationError, match="uid map"):
        await start(provider)
    assert list(system["base"].iterdir()) == []
    ok = await start(make_provider(system, probe_on_create=True))
    assert await ok.status() == SandboxStatus.RUNNING
    await ok.stop()


async def test_create_refuses_unsupported_spec_and_writes_default_files(system) -> None:
    provider = make_provider(system, default_files={"/tmp/runtime/ready": "", "/workspace/AGENTS.md": "hi"})
    with pytest.raises(BwrapCreateError, match="entrypoint"):
        await start(provider, entrypoint=["sleep", "inf"])
    with pytest.raises(BwrapCreateError, match="CPU-only"):
        await start(provider, resources={"gpu": 1})
    with pytest.raises(BwrapCreateError, match="workdir"):
        await start(provider, workdir="/proc/x")
    sandbox = await start(provider, workdir="/workspace", files={"/workspace/problem/a.txt": "A"})
    inst = sandbox._handle.raw
    assert (inst.tmp / "runtime" / "ready").read_text() == ""
    assert (inst.workspace / "AGENTS.md").read_text() == "hi"
    assert (inst.workspace / "problem" / "a.txt").read_text() == "A"
    assert oct(inst.root.stat().st_mode & 0o777) == "0o700"
    await sandbox.stop()


# ----------------------------------------------------------------------------- connect


async def test_descriptor_round_trip(system, tmp_path) -> None:
    creator = make_provider(system, mounts=[{"src": str(system["venv"]), "dst": "/codex-runtime"}])
    sandbox = await start(
        creator,
        workdir="/workspace",
        env={"KF_EVAL_TIMEOUT": "60"},
        provider_options={"mounts": [f"{system['bin']}:/extra"]},
    )
    descriptor = json.loads(json.dumps(await sandbox.serialize()))
    assert descriptor == {
        "provider": "bwrap",
        "host": socket.gethostname(),
        "uid": os.getuid(),
        "base_dir": str(Path(os.path.realpath(system["base"]))),
        "sandbox_id": sandbox._handle.sandbox_id,
        "workdir": "/workspace",
        "network": "none",
    }
    borrower_provider = make_provider(system, mounts=[{"src": str(system["venv"]), "dst": "/codex-runtime"}])
    borrowed = await AsyncSandbox.connect(descriptor, provider=borrower_provider)
    original, received = sandbox._handle.raw, borrowed._handle.raw
    assert received.root == original.root
    assert (received.workdir, received.network, received.env) == ("/workspace", "none", {"KF_EVAL_TIMEOUT": "60"})
    assert received.mounts == original.mounts
    assert borrower_provider.bwrap_argv(received, "true") == creator.bwrap_argv(original, "true")
    assert (await borrowed.exec("echo $KF_EVAL_TIMEOUT")).stdout == "60\n"
    await borrowed.upload(_text(tmp_path / "f", "shared"), "/workspace/f.txt")
    assert (original.workspace / "f.txt").read_text() == "shared"
    await borrowed.disconnect()
    assert original.root.is_dir() and await sandbox.status() == SandboxStatus.RUNNING
    await sandbox.stop()
    assert not original.root.exists()


async def test_connect_refusals(system, tmp_path, monkeypatch) -> None:
    provider = make_provider(system)
    sandbox = await start(provider, workdir="/workspace")
    good = await sandbox.serialize()
    root = sandbox._handle.raw.root

    async def refused(match: str, **changes) -> None:
        with pytest.raises(ValueError, match=match):
            await provider.connect({**good, **changes})

    await refused("produced by the bwrap provider", provider="apptainer")
    await refused("same host and uid", host="other-node")
    await refused("same host and uid", uid=os.getuid() + 1)
    await refused("base_dir", base_dir=str(tmp_path / "elsewhere"))
    await refused("invalid bwrap sandbox id", sandbox_id="../../etc")
    await refused("invalid bwrap sandbox id", sandbox_id="bwrap-" + "0" * 31)
    await refused("does not exist", sandbox_id="bwrap-" + "0" * 32)
    # The descriptor's workdir is the facade's default cwd, not the workspace bind: a moved one connects.
    assert (await provider.connect({**good, "workdir": "/workspace/project"})).raw.workdir == "/workspace"
    await refused("network", network="host")
    other = make_provider(system, base_dir=str(tmp_path / "other-base"))
    with pytest.raises(ValueError, match="base_dir"):
        await other.connect(good)
    (root / "sandbox.json").write_text("{not json")
    await refused("state file")
    (root / "sandbox.json").write_text(json.dumps({"sandbox_id": "bwrap-" + "1" * 32}))
    await refused("does not match")
    (root / "closing").touch()
    await refused("stopped")
    (root / "closing").unlink()
    (root / "sandbox.json").write_text(json.dumps(sandbox._handle.raw.state()))
    assert (await provider.connect(good)).raw.root == root
    await sandbox.stop()
    await refused("does not exist")


# ----------------------------------------------------------------------------- files


def _text(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


async def test_transfer_path_mapping(system, tmp_path) -> None:
    provider = make_provider(system, mounts=[{"src": str(system["venv"]), "dst": "/codex-runtime"}])
    (system["venv"] / "bin").mkdir()
    (system["venv"] / "bin" / "python").write_text("runtime")
    sandbox = await start(provider, workdir="/workspace")
    inst = sandbox._handle.raw
    source = _text(tmp_path / "src.sh", "#!/bin/sh\n")
    source.chmod(0o755)
    for target, host in (
        ("/workspace/a/b.sh", inst.workspace / "a" / "b.sh"),
        ("rel/c.txt", inst.workspace / "rel" / "c.txt"),
        *(
            [(f"{inst.workspace}/d.txt", inst.workspace / "d.txt")]
            if provider._binds_workspace_host_path(inst)
            else []
        ),
        ("/tmp/nemo-gym-codex-sessions/x/input.json", inst.tmp / "nemo-gym-codex-sessions" / "x" / "input.json"),
        ("/home/sandbox/.codex/config.toml", inst.home / ".codex" / "config.toml"),
        ("//workspace//./e.txt", inst.workspace / "e.txt"),
    ):
        await sandbox.upload(source, target)
        assert host.read_text() == "#!/bin/sh\n", target
        assert await provider.exists(sandbox._handle, target)
        out = tmp_path / "out" / host.name
        await sandbox.download(target, out)
        assert out.read_text() == "#!/bin/sh\n"
    assert (inst.workspace / "a" / "b.sh").stat().st_mode & 0o111  # mode is preserved
    await sandbox.download("/codex-runtime/bin/python", tmp_path / "python")
    assert (tmp_path / "python").read_text() == "runtime"
    assert not await provider.exists(sandbox._handle, "/workspace/missing")
    await sandbox.stop()


async def test_transfer_containment(system, tmp_path) -> None:
    provider = make_provider(system, mounts=[{"src": str(system["venv"]), "dst": "/codex-runtime"}])
    sandbox = await start(provider, workdir="/workspace")
    inst = sandbox._handle.raw
    source = _text(tmp_path / "src", "payload")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("secret")
    (inst.workspace / "escape").symlink_to(outside)
    (inst.workspace / "link.txt").symlink_to(outside / "secret")
    (inst.workspace / "sub").mkdir()
    for path, match in (
        ("/workspace/../etc/passwd", "'..'"),
        ("../x", "'..'"),
        ("/etc/passwd", "outside the sandbox"),
        ("/workspacex/a", "outside the sandbox"),
        ("/codex-runtime/new", "read-only mount"),
        ("/workspace/escape/secret", "symlink"),
        ("/workspace/link.txt", "symlink"),
    ):
        with pytest.raises(BwrapPathError, match=match):
            await sandbox.upload(source, path)
    for path in ("/workspace/escape/secret", "/workspace/link.txt", "/etc/passwd", "/workspace/sub/../../x"):
        with pytest.raises(BwrapPathError):
            await sandbox.download(path, tmp_path / "got")
    with pytest.raises(IsADirectoryError):
        await sandbox.download("/workspace/sub", tmp_path / "got")
    with pytest.raises(IsADirectoryError):
        await sandbox.upload(source, "/workspace")
    with pytest.raises(FileNotFoundError):
        await sandbox.download("/workspace/nope/x", tmp_path / "got")
    assert not await provider.exists(sandbox._handle, "/workspace/escape/secret")
    assert (outside / "secret").read_text() == "secret"
    assert not (outside / "new").exists()
    await sandbox.stop()


async def test_a_download_through_a_symlink_is_refused_with_eloop(system, tmp_path) -> None:
    """The errno is what lets a caller (``read_file_bytes``) tell a planted link from an outage."""
    provider = make_provider(system)
    sandbox = await start(provider, workdir="/workspace")
    inst = sandbox._handle.raw
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "reward.txt").write_text("1")
    (inst.workspace / "linked").symlink_to(outside)
    (inst.workspace / "reward.txt").symlink_to(outside / "reward.txt")
    for path in ("/workspace/reward.txt", "/workspace/linked/reward.txt"):
        with pytest.raises(BwrapPathError) as refused:
            await sandbox.download(path, tmp_path / "got")
        assert refused.value.errno == errno.ELOOP, path
    assert not (tmp_path / "got").exists()
    await sandbox.stop()


async def test_transfers_refuse_a_planted_fifo_without_blocking(system, tmp_path) -> None:
    provider = make_provider(system)
    sandbox = await start(provider, workdir="/workspace")
    inst = sandbox._handle.raw
    os.mkfifo(inst.workspace / "stderr.log")  # sandboxed code runs as our uid and can plant one
    source = _text(tmp_path / "src", "payload")
    # A blocking open would hang its worker thread forever; these must fail at once.
    with pytest.raises(IsADirectoryError, match="not a regular file"):
        await asyncio.wait_for(sandbox.download("/workspace/stderr.log", tmp_path / "got"), 5)
    with pytest.raises(BwrapPathError, match="not a regular file"):
        await asyncio.wait_for(sandbox.upload(source, "/workspace/stderr.log"), 5)
    assert not (tmp_path / "got").exists()
    await sandbox.stop()


# ----------------------------------------------------------------------------- stop


async def test_stop_kills_tracked_processes_and_removes_directory(system, tmp_path) -> None:
    provider = make_provider(system)
    sandbox = await start(provider, workdir="/workspace")
    inst = sandbox._handle.raw
    running = asyncio.create_task(sandbox.exec("sleep 60", timeout_s=120))
    for _ in range(100):
        if provider._processes.get(inst.sandbox_id):
            break
        await asyncio.sleep(0.02)
    (pid,) = provider._processes[inst.sandbox_id]
    assert (inst.run_dir / str(pid)).exists()
    # A process recorded by another provider instance (e.g. a borrower in another server).
    foreign = subprocess.Popen(["sleep", "60"], start_new_session=True)
    (inst.run_dir / str(foreign.pid)).write_text(bwrap_module._process_start_time(foreign.pid) or "")
    # Reap it as its real parent (another server) would, so it does not linger as a zombie.
    reaper = asyncio.create_task(asyncio.to_thread(foreign.wait, 30))
    # Sandboxed code may leave unwritable directories behind.
    locked = inst.workspace / "locked"
    (locked / "inner").mkdir(parents=True)
    (locked / "inner" / "f").write_text("x")
    locked.chmod(0o500)
    await sandbox.stop()
    result = await asyncio.wait_for(running, timeout=10)
    assert result.return_code != 0
    assert await reaper == -9
    assert not inst.root.exists()
    assert not provider._processes
    assert await provider.status(sandbox._handle) == SandboxStatus.STOPPED
    await provider.close(sandbox._handle)  # idempotent


async def test_stop_skips_reused_pids(system) -> None:
    provider = make_provider(system)
    sandbox = await start(provider)
    inst = sandbox._handle.raw
    bystander = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        # A record whose start time does not match is a PID that was reused: leave it alone.
        (inst.run_dir / str(bystander.pid)).write_text("1")
        await sandbox.stop()
        assert bystander.poll() is None
        assert not inst.root.exists()
    finally:
        bystander.kill()
        bystander.wait()


async def test_aclose_kills_running_execs_but_keeps_directories(system) -> None:
    provider = make_provider(system)
    sandbox = await start(provider)
    running = asyncio.create_task(sandbox.exec("sleep 60", timeout_s=120))
    for _ in range(100):
        if provider._processes:
            break
        await asyncio.sleep(0.02)
    await provider.aclose()
    assert (await asyncio.wait_for(running, timeout=10)).return_code != 0
    assert sandbox._handle.raw.root.is_dir()
    await sandbox.stop()


async def test_cancelled_exec_kills_the_process(system) -> None:
    provider = make_provider(system)
    sandbox = await start(provider)
    marker = sandbox._handle.raw.tmp / "pid"
    running = asyncio.create_task(sandbox.exec(f"echo $$ > {marker}; sleep 60", timeout_s=120))
    for _ in range(200):
        if marker.exists() and marker.read_text().strip():
            break
        await asyncio.sleep(0.02)
    pid = int(marker.read_text())
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    with pytest.raises(ProcessLookupError):
        for _ in range(100):
            os.kill(pid, 0)
            await asyncio.sleep(0.02)
    await sandbox.stop()


# ----------------------------------------------------------------------------- exec-scoped processes


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.02)


async def test_exec_processes_terminated_tracks_execs_and_their_namespace_inits(system) -> None:
    provider = make_provider(system, kill_timeout_s=0.3)
    sandbox = await start(provider)
    inst = sandbox._handle.raw
    assert await sandbox.exec_processes_terminated()

    running = asyncio.create_task(sandbox.exec("sleep 60", timeout_s=120))
    await _wait_for(lambda: list(inst.run_dir.glob("*.init")))
    (pid,) = provider._processes[inst.sandbox_id]
    (init,) = inst.run_dir.glob("*.init")
    assert init.name == f"{pid}.init"
    assert int(init.read_text().split()[0]) == pid  # the fake's "namespace init" is the command itself
    assert not await sandbox.exec_processes_terminated()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert await sandbox.exec_processes_terminated()
    assert not list(inst.run_dir.iterdir())

    # A normal exec leaves no record behind once its process is gone.
    assert (await sandbox.exec("true")).return_code == 0
    assert not list(inst.run_dir.iterdir())
    await sandbox.stop()


async def test_a_live_namespace_init_blocks_the_proof_until_it_is_gone(system) -> None:
    provider = make_provider(system, kill_timeout_s=0.3)
    sandbox = await start(provider)
    inst = sandbox._handle.raw
    # An init that outlives its bwrap (another instance's exec, or the instant after bwrap exits).
    init = subprocess.Popen(["sleep", "60"], start_new_session=True)
    record = inst.run_dir / f"999999{bwrap_module.INIT_RECORD_SUFFIX}"
    record.write_text(f"{init.pid} {bwrap_module._process_start_time(init.pid) or ''}")
    try:
        assert not await sandbox.exec_processes_terminated()
        init.kill()
        init.wait()
        assert await sandbox.exec_processes_terminated()
        assert not record.exists()
    finally:
        init.kill()
        init.wait()
    # Unprovable states are never reported as terminated.
    record.write_text("not a record")
    assert not await sandbox.exec_processes_terminated()
    record.unlink()
    provider._untracked_inits.add(inst.sandbox_id)
    assert not await sandbox.exec_processes_terminated()
    await sandbox.stop()


async def test_stop_kills_a_recorded_namespace_init(system) -> None:
    provider = make_provider(system)
    sandbox = await start(provider)
    inst = sandbox._handle.raw
    init = subprocess.Popen(["sleep", "60"], start_new_session=True)
    (inst.run_dir / f"999999{bwrap_module.INIT_RECORD_SUFFIX}").write_text(
        f"{init.pid} {bwrap_module._process_start_time(init.pid) or ''}"
    )
    reaper = asyncio.create_task(asyncio.to_thread(init.wait, 30))
    await sandbox.stop()
    assert await reaper == -9
    assert not inst.root.exists()


async def test_providers_without_exec_scoped_processes_never_prove_termination() -> None:
    class Plain:
        name = "plain"

    sandbox = AsyncSandbox(Plain(), owns_provider=False)
    assert not await sandbox.exec_processes_terminated()
