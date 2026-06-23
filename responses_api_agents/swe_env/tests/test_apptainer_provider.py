# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apptainer provider tests (mocked subprocess — apptainer not installed here)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nemo_gym.sandbox import SandboxSpec
from responses_api_agents.swe_env.providers.apptainer_provider import ApptainerSandboxProvider


def _patch_run(provider, scripted):
    """Replace the provider's subprocess runner with a recording stub.

    Args:
        provider: The ApptainerSandboxProvider whose ``_run`` is patched.
        scripted: A callable that maps the argv list to a ``(return_code, stdout,
            stderr)`` tuple returned by the stubbed runner.

    Returns:
        A list that accumulates the argv list of each call, for later assertions.
    """
    calls: list[list[str]] = []

    async def fake_run(*args, timeout_s=None):
        calls.append(list(args))
        return scripted(list(args))

    provider._run = fake_run  # type: ignore[assignment]
    return calls


def test_resolve_sif_direct_path(tmp_path: Path):
    """A spec image that is a direct ``.sif`` path resolves to that path.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    sif = tmp_path / "image.sif"
    sif.write_text("x")
    provider = ApptainerSandboxProvider()
    spec = SandboxSpec(image=str(sif))
    assert provider._resolve_sif(spec) == str(sif)


def test_resolve_sif_glob(tmp_path: Path):
    """A configured ``image_glob`` resolves the image against the image root.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    (tmp_path / "myrepo__inst.sif").write_text("x")
    provider = ApptainerSandboxProvider(image_root=str(tmp_path))
    spec = SandboxSpec(image="inst", provider_options={"image_glob": "*.sif"})
    assert provider._resolve_sif(spec).endswith("myrepo__inst.sif")


def test_resolve_sif_fuzzy_restricts_to_sif(tmp_path: Path):
    """The fuzzy fallback only matches ``.sif`` files, ignoring other host files.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    # The fuzzy term is restricted to ``*.sif`` so an unrelated host file matching
    # the image substring is never mistaken for a container.
    (tmp_path / "myrepo__inst.sif").write_text("x")
    (tmp_path / "myrepo__inst.log").write_text("not a container")
    provider = ApptainerSandboxProvider(image_root=str(tmp_path))
    spec = SandboxSpec(image="inst")  # no explicit image_glob -> fuzzy path
    assert provider._resolve_sif(spec).endswith("myrepo__inst.sif")


def test_resolve_sif_fuzzy_lowercases_search_term(tmp_path: Path):
    """A mixed-case image matches a lowercased ``.sif`` on disk via the fuzzy fallback.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    (tmp_path / "repo__myinst.sif").write_text("x")
    provider = ApptainerSandboxProvider(image_root=str(tmp_path))
    spec = SandboxSpec(image="MyInst")  # uppercase; on-disk file is lowercase
    assert provider._resolve_sif(spec).endswith("repo__myinst.sif")


def test_default_instance_args_restore_legacy_flags():
    """The default launch flags include the expected apptainer exec flags."""
    # The default launch flags are --pid and --no-mount home,tmp,bind-paths, on top
    # of --writable-tmpfs --cleanenv. They remain overridable via the instance_args
    # kwarg.
    provider = ApptainerSandboxProvider()
    assert provider._instance_args == [
        "--writable-tmpfs",
        "--cleanenv",
        "--pid",
        "--no-mount",
        "home,tmp,bind-paths",
    ]


def test_instance_args_remain_overridable():
    """An explicit ``instance_args`` value overrides the default launch flags."""
    provider = ApptainerSandboxProvider(instance_args=["--nv"])
    assert provider._instance_args == ["--nv"]
    # An explicit empty list disables the defaults entirely (distinct from None).
    assert ApptainerSandboxProvider(instance_args=[])._instance_args == []


def test_create_issues_legacy_default_flags(tmp_path: Path):
    """Creating an instance issues the default launch flags in the start argv.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    sif = tmp_path / "image.sif"
    sif.write_text("x")
    provider = ApptainerSandboxProvider()
    calls = _patch_run(provider, lambda args: (0, "out", ""))
    asyncio.run(provider.create(SandboxSpec(image=str(sif), metadata={"instance_id": "i"})))
    start_argv = calls[0]
    assert "--pid" in start_argv
    assert "--no-mount" in start_argv
    assert "home,tmp,bind-paths" in start_argv


def test_create_and_exec_issue_expected_argv(tmp_path: Path):
    """Create and exec build the expected ``instance start`` and ``exec`` argv.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    sif = tmp_path / "image.sif"
    sif.write_text("x")
    provider = ApptainerSandboxProvider()
    calls = _patch_run(provider, lambda args: (0, "out", ""))

    handle = asyncio.run(
        provider.create(SandboxSpec(image=str(sif), workdir="/testbed", metadata={"instance_id": "i"}))
    )
    assert handle.provider_name == "apptainer"
    start_argv = calls[0]
    assert start_argv[:2] == ["instance", "start"]
    assert str(sif) in start_argv

    asyncio.run(provider.exec(handle, "echo hi", cwd="/testbed"))
    exec_argv = calls[-1]
    assert exec_argv[0] == "exec"
    assert any(a.startswith("instance://") for a in exec_argv)
    assert "--pwd" in exec_argv and "/testbed" in exec_argv


def test_create_binds_provider_option_mounts(tmp_path: Path):
    """Each ``provider_options['mounts']`` entry becomes a ``--bind`` argument.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    sif = tmp_path / "image.sif"
    sif.write_text("x")
    # Real host setup dir so the canonical self-bind (src == dst) is a genuine
    # host path and is NOT dropped by the dataset self-bind guard.
    setup = tmp_path / "setup"
    setup.mkdir()
    provider = ApptainerSandboxProvider()
    calls = _patch_run(provider, lambda args: (0, "out", ""))

    spec = SandboxSpec(
        image=str(sif),
        workdir="/testbed",
        metadata={"instance_id": "i"},
        provider_options={
            "mounts": [
                {"src": "/host/data.jsonl", "dst": "/root/dataset/data.jsonl"},
                {"src": str(setup), "dst": "/swebench_setup"},
                {"src": str(setup), "dst": str(setup)},
                {"src": "/host/ro", "dst": "/ro", "ro": True},
            ]
        },
    )
    asyncio.run(provider.create(spec))
    start_argv = calls[0]
    assert "--bind" in start_argv
    binds = [start_argv[i + 1] for i, a in enumerate(start_argv) if a == "--bind"]
    assert "/host/data.jsonl:/root/dataset/data.jsonl" in binds
    assert f"{setup}:/swebench_setup" in binds
    # An existing host self-bind survives unchanged.
    assert f"{setup}:{setup}" in binds
    # Read-only mounts get the :ro suffix (src != dst, so the guard never applies).
    assert "/host/ro:/ro:ro" in binds


def test_create_skips_dataset_self_bind_when_src_missing(tmp_path: Path):
    """A self-bind whose host source does not exist is dropped from the binds.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    # The dataset default mount falls back to src == dst == the in-container
    # dataset path (not a host path) when no real dataset path is provisioned.
    # Binding a missing host src would shadow the real dataset with an empty dir,
    # so a self-bind whose source does not exist is skipped.
    sif = tmp_path / "image.sif"
    sif.write_text("x")
    provider = ApptainerSandboxProvider()
    calls = _patch_run(provider, lambda args: (0, "out", ""))

    spec = SandboxSpec(
        image=str(sif),
        provider_options={
            "mounts": [
                # Self-bind of an in-container-only path (no host counterpart).
                {"src": "/root/dataset/data.jsonl", "dst": "/root/dataset/data.jsonl"},
            ]
        },
    )
    asyncio.run(provider.create(spec))
    start_argv = calls[0]
    binds = [start_argv[i + 1] for i, a in enumerate(start_argv) if a == "--bind"]
    # Only the scratch I/O mount survives; the phantom dataset self-bind is dropped.
    assert len(binds) == 1
    assert not any("data.jsonl" in b for b in binds)


def test_create_skips_incomplete_mounts(tmp_path: Path):
    """Mounts missing a src or dst are skipped, not emitted as half-specified binds.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    sif = tmp_path / "image.sif"
    sif.write_text("x")
    provider = ApptainerSandboxProvider()
    calls = _patch_run(provider, lambda args: (0, "out", ""))

    spec = SandboxSpec(
        image=str(sif),
        provider_options={"mounts": [{"src": "/only-src"}, {"dst": "/only-dst"}, {}]},
    )
    asyncio.run(provider.create(spec))
    start_argv = calls[0]
    binds = [start_argv[i + 1] for i, a in enumerate(start_argv) if a == "--bind"]
    # Only the scratch I/O mount survives; no half-specified binds slip through.
    assert len(binds) == 1
    assert not any("only-src" in b or "only-dst" in b for b in binds)


def test_mount_binds_empty_when_no_mounts():
    """``_mount_binds`` returns an empty list when no mounts are configured."""
    provider = ApptainerSandboxProvider()
    assert provider._mount_binds(SandboxSpec(image="x")) == []
    assert provider._mount_binds(SandboxSpec(image="x", provider_options={"mounts": None})) == []


def test_exec_timeout_returns_typed_result(tmp_path: Path):
    """A timed-out exec returns a typed result with return code 124 and a timeout kind.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    provider = ApptainerSandboxProvider()

    async def timeout_run(*args, timeout_s=None):
        raise asyncio.TimeoutError

    provider._run = timeout_run  # type: ignore[assignment]
    from nemo_gym.sandbox import SandboxHandle

    handle = SandboxHandle(sandbox_id="x", provider_name="apptainer", raw={"workdir": "/t", "scratch": str(tmp_path)})
    result = asyncio.run(provider.exec(handle, "sleep 100", timeout_s=1))
    assert result.return_code == 124
    assert result.error_type == "timeout"
