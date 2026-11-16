# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bz2
import copy
import gzip
import io
import json
import lzma
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from resources_servers.swe_gym.clean_image import build_rootfs, git, sha256_file
from resources_servers.swe_gym.image_admission import OFFLINE_NETWORK_PROBE, admit_agent_image


SOLUTION = b"return the_distinctive_reference_solution_for_this_private_task(value)"
HIDDEN = b"assert task_result == distinctive_private_hidden_test_expected_value"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


def runtime_tar(path, files):
    with tarfile.open(path, "w") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mtime = 123456789
            archive.addfile(member, io.BytesIO(data))


@pytest.fixture
def task(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet", "--initial-branch=main", "--template=")
    (source / "app.py").write_text("def answer():\n    return 0\n")
    git(source, "add", "app.py")
    git(source, "-c", "commit.gpgsign=false", "commit", "-qm", "public base")
    base = git(source, "rev-parse", "HEAD").decode().strip()
    # A future solution reachable from packed remote refs/tags, and a separate unreachable object.
    (source / "app.py").write_bytes(SOLUTION + b"\n")
    git(source, "add", "app.py")
    git(source, "-c", "commit.gpgsign=false", "commit", "-qm", "future solution")
    fix = git(source, "rev-parse", "HEAD").decode().strip()
    git(source, "update-ref", "refs/remotes/origin/main", fix)
    git(source, "tag", "future-fix")
    git(source, "pack-refs", "--all")
    git(source, "repack", "-ad")
    unreachable = git(source, "hash-object", "-w", "--stdin", input_bytes=HIDDEN).decode().strip()
    git(source, "pack-objects", str(source / ".git/objects/pack/pack"), input_bytes=(unreachable + "\n").encode())
    (source / ".git/objects" / unreachable[:2] / unreachable[2:]).unlink()
    git(source, "reset", "--hard", base)
    (source / ".cache").mkdir()
    (source / ".cache/reference.txt").write_bytes(SOLUTION)
    (source / "future.patch").write_bytes(SOLUTION)
    runtime = tmp_path / "runtime.tar"
    runtime_tar(runtime, {"usr/bin/runtime-fixture": b"trusted task-free runtime"})
    reference = tmp_path / "reference.patch"
    reference.write_bytes(b"diff --git a/app.py b/app.py\n+" + SOLUTION + b"\n")
    hidden = tmp_path / "hidden.py"
    hidden.write_bytes(HIDDEN)
    kwargs = dict(
        runtime=runtime,
        runtime_sha256=sha256_file(runtime),
        repo=source,
        base_commit=base,
        references=[reference, hidden],
        task_id="example-1",
        output=tmp_path / "rootfs.tar",
    )
    return kwargs, fix, unreachable


def test_reconstruction_removes_packed_unreachable_and_cached_solutions(task, tmp_path):
    kwargs, fix, unreachable = task
    assert len(list((kwargs["repo"] / ".git/objects/pack").glob("*.pack"))) >= 2
    assert SOLUTION in git(kwargs["repo"], "show", "origin/main:app.py")
    assert HIDDEN in git(kwargs["repo"], "cat-file", "-p", unreachable)
    manifest = build_rootfs(**kwargs)
    extracted = tmp_path / "extracted"
    with tarfile.open(kwargs["output"]) as archive:
        assert all(member.mtime == 0 for member in archive)
        archive.extractall(extracted, filter="data")
    repo = extracted / "testbed"
    assert not (repo / "future.patch").exists()
    assert not (repo / ".cache").exists()
    assert not (repo / ".git/packed-refs").exists()
    assert not (repo / ".git/logs").exists()
    assert git(repo, "for-each-ref", "--format=%(refname)") == b"refs/heads/main\n"
    assert git(repo, "rev-list", "--count", "--all") == b"1\n"
    for oid in [fix, unreachable]:
        with pytest.raises(subprocess.CalledProcessError):
            git(repo, "cat-file", "-p", oid)
    objects = git(repo, "cat-file", "--batch-all-objects", "--batch")
    assert SOLUTION not in objects and HIDDEN not in objects
    assert git(repo, "fsck", "--full", "--no-reflogs", "--unreachable") == b""
    # The initial commit supports ordinary edits, tests and patch capture without source history.
    (repo / "app.py").write_text("def answer():\n    return 42\n")
    assert b"+    return 42" in git(repo, "diff", manifest["agent_base_commit"])
    assert manifest["base_commit"] == kwargs["base_commit"]
    assert manifest["rootfs_sha256"] == sha256_file(kwargs["output"])


@pytest.mark.parametrize(
    "name,data",
    [
        ("root/.cache/fix.txt", b"cache"),
        ("usr/local/go/pkg/mod/fix.txt", SOLUTION),
        ("root/.netrc", b"credentials"),
        ("usr/share/blob", SOLUTION),
        ("usr/share/blob.gz", gzip.compress(SOLUTION)),
        ("usr/share/blob.bz2", bz2.compress(SOLUTION)),
        ("usr/share/blob.xz", lzma.compress(SOLUTION)),
        ("usr/share/renamed-pack", b"PACK" + b"compressed git data"),
        ("testbed/.git/objects/pack/pack-abc.pack", b"PACK"),
        ("opt/copy/.git/objects/pack/pack-abc.pack", b"PACK"),
        ("tmp/reference.patch", b"hidden"),
        ("etc/private-key", b"-----BEGIN PRIVATE KEY-----"),
    ],
)
def test_runtime_leaks_rejected_before_output(task, name, data):
    kwargs, _, _ = task
    runtime_tar(kwargs["runtime"], {name: data})
    kwargs["runtime_sha256"] = sha256_file(kwargs["runtime"])
    with pytest.raises(ValueError):
        build_rootfs(**kwargs)
    assert not kwargs["output"].exists()


def test_pinned_runtime_and_existing_output_preserved(task):
    kwargs, _, _ = task
    kwargs["runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        build_rootfs(**kwargs)
    kwargs["output"].write_text("existing artifact")
    with pytest.raises(ValueError, match="already exists"):
        build_rootfs(**kwargs)
    assert kwargs["output"].read_text() == "existing artifact"


@pytest.fixture
def admission(task):
    kwargs, _, _ = task
    manifest = build_rootfs(**kwargs) | {"image_id": "sha256:" + "a" * 64}
    path = kwargs["output"].with_suffix(".json")
    path.write_text(json.dumps(manifest))
    return dict(
        manifest_path=str(path),
        task_id=kwargs["task_id"],
        base_commit=kwargs["base_commit"],
        provider_config={
            "docker": {
                "create": {
                    "network": "none",
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges"],
                    "pids_limit": 256,
                }
            }
        },
        sandbox_config={"ttl_s": 600, "resources": {"cpu": 2, "memory_mib": 2048}},
    )


def test_admission_binds_task_base_and_immutable_image(admission):
    admitted = admit_agent_image(**admission)
    assert admitted.image_id == "sha256:" + "a" * 64
    for field in ["task_id", "base_commit"]:
        bad = admission | {field: "wrong"}
        with pytest.raises(ValueError):
            admit_agent_image(**bad)
    manifest = json.loads(Path(admission["manifest_path"]).read_text())
    manifest["image_id"] = "image:latest"
    Path(admission["manifest_path"]).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="image_id"):
        admit_agent_image(**admission)


@pytest.mark.parametrize("provider", ["local", "apptainer", "coreweave", "opensandbox", "e2b"])
def test_unsupported_provider_fails_closed(admission, provider):
    with pytest.raises(ValueError, match="offline Docker"):
        admit_agent_image(**(admission | {"provider_config": {provider: {}}}))


@pytest.mark.parametrize(
    "change",
    [
        {"network": "host"},
        {"network": "bridge"},
        {"cap_drop": []},
        {"security_opt": []},
        {"pids_limit": 0},
        {"extra_run_args": ["--privileged"]},
    ],
)
def test_network_and_privilege_bypasses_rejected(admission, change):
    bad = copy.deepcopy(admission)
    bad["provider_config"]["docker"]["create"].update(change)
    with pytest.raises(ValueError):
        admit_agent_image(**bad)


def test_host_mounts_rejected(admission):
    admission["sandbox_config"]["provider_options"] = {"volumes": ["/tmp:/tmp"]}
    with pytest.raises(ValueError, match="mounts"):
        admit_agent_image(**admission)


def test_probe_rejects_network_interface_before_policy_code():
    # Exercise the actual probe's fail-closed interface check, with no host network traffic.
    class Socket:
        @staticmethod
        def if_nameindex():
            return [(1, "lo"), (2, "eth0")]

    import builtins

    original = builtins.__import__

    def import_probe(name, *args, **kwargs):
        return Socket if name == "socket" else original(name, *args, **kwargs)

    with pytest.raises(RuntimeError, match="non-loopback"):
        exec(OFFLINE_NETWORK_PROBE, {"__builtins__": vars(builtins) | {"__import__": import_probe}})


@pytest.mark.parametrize("escape", [None, "ip", "udp", "tcp"])
def test_probe_checks_direct_ip_and_actual_and_external_dns(monkeypatch, escape):
    import builtins
    import socket

    calls = []

    class FakeSocket:
        def __init__(self, family, kind):
            self.kind = kind

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def settimeout(self, value):
            pass

        def connect_ex(self, target):
            calls.append(("ip", target))
            return 0 if escape == "ip" else 101

        def connect(self, target):
            calls.append(("tcp", target))
            if escape != "tcp":
                raise OSError("blocked")

        def sendto(self, data, target):
            calls.append(("udp", target))

        def recvfrom(self, size):
            if escape == "udp":
                return b"DNS REFUSED still proves egress", ("1.1.1.1", 53)
            raise TimeoutError("blocked")

    original_open = builtins.open
    monkeypatch.setattr(socket, "if_nameindex", lambda: [(1, "lo")])
    monkeypatch.setattr(socket, "socket", FakeSocket)
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *a, **kw: (
            io.StringIO("nameserver 10.0.0.53\n") if path == "/etc/resolv.conf" else original_open(path, *a, **kw)
        ),
    )
    if escape:
        with pytest.raises(RuntimeError, match="egress succeeded"):
            exec(OFFLINE_NETWORK_PROBE, {})
    else:
        exec(OFFLINE_NETWORK_PROBE, {})
        assert ("ip", ("1.1.1.1", 443)) in calls
        assert ("ip", ("2606:4700:4700::1111", 443)) in calls
        for resolver in ["10.0.0.53", "1.1.1.1", "8.8.8.8"]:
            assert ("udp", (resolver, 53)) in calls
            assert ("tcp", (resolver, 53)) in calls


@pytest.mark.asyncio
async def test_server_selects_audited_agent_image_and_leaves_verifier_image(admission, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from resources_servers.swe_gym import app

    admission["sandbox_config"]["provider_options"] = None
    body = SimpleNamespace(
        instance_id=admission["task_id"],
        base_commit=admission["base_commit"],
        image_name="untrusted-row:latest",
        language="python",
    )
    config = SimpleNamespace(
        sandbox_provider="sandbox",
        sandbox_config=admission["sandbox_config"],
        agent_image_manifests={body.instance_id: admission["manifest_path"]},
        name="swe_gym",
    )
    server = SimpleNamespace(
        config=config, _session_id_to_clean_base={}, _session_id_to_sandbox={}, _verifier_sandboxes={}
    )
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *args: admission["provider_config"])
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *args: {})
    sandbox = SimpleNamespace(start=AsyncMock())
    factory = lambda *args: sandbox
    monkeypatch.setattr(app, "AsyncSandbox", factory)
    await app.SWEGymResourcesServer._create_sandbox(server, body, session_id="agent")
    spec = sandbox.start.call_args.args[0]
    assert spec.image == "sha256:" + "a" * 64
    assert not spec.files and not spec.provider_options
    assert spec.metadata["image-scanner"] == "1"
    assert server._session_id_to_clean_base["agent"] == admit_agent_image(**admission).agent_base_commit
    # The companion verifier defense owns this path; no new image policy or files are imposed here.
    await app.SWEGymResourcesServer._create_sandbox(server, body, files={"hidden": "server only"})
    verifier_spec = sandbox.start.call_args.args[0]
    assert verifier_spec.image == body.image_name
    assert verifier_spec.files == {"hidden": "server only"}
    server.config.agent_image_manifests = {}
    sandbox.start.reset_mock()
    with pytest.raises(ValueError, match="no approved"):
        await app.SWEGymResourcesServer._create_sandbox(server, body, session_id="missing")
    sandbox.start.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe_result",
    [
        (0, None, "offline-network-admitted"),
        (1, None, ""),
        (0, "timeout", "offline-network-admitted"),
    ],
)
async def test_server_refuses_policy_access_after_failed_probe(monkeypatch, probe_result):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from resources_servers.swe_gym import app

    code, error, stdout = probe_result
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=code, error_type=error, stdout=stdout))
    )
    server = SimpleNamespace(
        config=SimpleNamespace(agent_image_manifests={}, apply_anti_cheating=True),
        _forget_task_sandbox_state=Mock(),
        _create_sandbox=AsyncMock(return_value=sandbox),
        _session_id_to_sandbox={},
        _session_id_to_pristine_untracked={},
        _pristine_untracked_files=AsyncMock(return_value=frozenset()),
    )
    identity = AsyncMock()
    scrub = AsyncMock()
    monkeypatch.setattr(app, "prepare_git_for_commits", identity)
    monkeypatch.setattr(app, "apply_anti_cheat_setup", scrub)
    body = SimpleNamespace(instance_id="task")
    if code or error:
        with pytest.raises(RuntimeError, match="network admission"):
            await app.SWEGymResourcesServer._start_task_sandbox(server, "agent", body)
        identity.assert_not_awaited()
    else:
        assert await app.SWEGymResourcesServer._start_task_sandbox(server, "agent", body) == "/testbed"
        identity.assert_awaited_once()
    scrub.assert_not_awaited()
    assert server._session_id_to_sandbox["agent"] is sandbox  # retained for existing failed-seed cleanup
    assert "python -c" in sandbox.exec.call_args.args[0]


@pytest.mark.asyncio
async def test_patch_capture_uses_reconstructed_commit(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from resources_servers.swe_gym import app

    capture = AsyncMock()
    monkeypatch.setattr(app, "capture_model_patch", capture)
    server = SimpleNamespace(
        _session_id_to_sandbox={"agent": object()},
        _session_id_to_clean_base={"agent": "fresh-base"},
        _session_id_to_pristine_untracked={},
        config=SimpleNamespace(patch_capture_mode="worktree"),
    )
    await app.SWEGymResourcesServer._extract_model_patch(server, "agent", "/testbed", "source-base")
    assert capture.call_args.args[2] == "fresh-base"
