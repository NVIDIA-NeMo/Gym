# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import refresh_locks
from prepare import UV_VERSION, require_uv_version


@pytest.mark.parametrize("actual", ["0.11.29", "0.13.5"])
def test_uv_requires_exact_baseline_toolchain(actual, monkeypatch):
    monkeypatch.setattr(refresh_locks.subprocess, "check_output", lambda *args, **kwargs: f"uv {actual}\n")
    with pytest.raises(ValueError, match=f"baseline pins uv {UV_VERSION}"):
        require_uv_version("uv")


def test_uv_accepts_current_baseline_toolchain(monkeypatch):
    monkeypatch.setattr(refresh_locks.subprocess, "check_output", lambda *args, **kwargs: f"uv {UV_VERSION}\n")
    require_uv_version("uv")


def test_publication_restores_every_previous_file_after_failure(tmp_path, monkeypatch):
    (tmp_path / "core.lock").write_bytes(b"previous core")
    (tmp_path / "lock-inputs.json").write_bytes(b"previous manifest")
    replacements = {
        "core.lock": b"new core",
        "sandbox.lock": b"new sandbox",
        "lock-inputs.json": b"new manifest",
    }
    original = refresh_locks.os.replace
    attempted = []

    def fail_manifest(source, destination):
        attempted.append(Path(destination).name)
        if Path(destination).name == "lock-inputs.json":
            raise OSError("simulated publication failure")
        return original(source, destination)

    monkeypatch.setattr(refresh_locks.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="publication failure"):
        refresh_locks.publish(replacements, tmp_path)
    assert attempted[:3] == ["core.lock", "sandbox.lock", "lock-inputs.json"]
    assert (tmp_path / "core.lock").read_bytes() == b"previous core"
    assert (tmp_path / "lock-inputs.json").read_bytes() == b"previous manifest"
    assert not (tmp_path / "sandbox.lock").exists()


def generation_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in refresh_locks.INPUTS:
        (source / name).write_text("source input")
    setup = source / "scripts/ci/setup_dev.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text(f"https://astral.sh/uv/{UV_VERSION}/install.sh")
    destination = tmp_path / "assets"
    destination.mkdir()
    (destination / "core.lock").write_text("existing core")
    monkeypatch.setattr(refresh_locks, "clean_source", lambda source: "source-sha")
    monkeypatch.setattr(refresh_locks, "require_uv_version", lambda uv, expected: None)
    monkeypatch.setattr(refresh_locks, "interpreter_environment", lambda *args: {})

    def prepare(source, checkout, *, uv):
        assets = checkout / "scripts/pants/native_core"
        assets.mkdir(parents=True)
        for group in refresh_locks.EXTRAS:
            (assets / f"{group}.requirements.txt").write_text("example==1\n")
            (assets / f"{group}.lock").write_text("generated lock")
        (checkout / "pants.native.toml").write_text(
            '[GLOBAL]\npants_version="2.30.2"\n[pex-cli]\nversion="v2.101.6"\n'
        )

    monkeypatch.setattr(refresh_locks, "prepare", prepare)
    monkeypatch.setattr(refresh_locks.subprocess, "run", lambda *args, **kwargs: None)
    return source, destination


def test_second_resolve_failure_does_not_publish_first(tmp_path, monkeypatch):
    source, destination = generation_fixture(tmp_path, monkeypatch)
    validated = []

    def validate(requirements, lock):
        validated.append(lock.name)
        if lock.name == "sandbox.lock":
            raise ValueError("sandbox dependency mismatch")
        return {"distributions": 1}

    monkeypatch.setattr(refresh_locks, "validate_lock", validate)
    with pytest.raises(ValueError, match="sandbox dependency mismatch"):
        refresh_locks.refresh(source=source, pants="pants", uv="uv", destination=destination)
    assert validated == ["core.lock", "sandbox.lock"]
    assert (destination / "core.lock").read_text() == "existing core"
    assert not (destination / "sandbox.lock").exists()
    assert not (destination / ".refresh-locks.lock").exists()


def test_success_publishes_both_resolves_and_exact_inputs(tmp_path, monkeypatch):
    source, destination = generation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(refresh_locks, "validate_lock", lambda *args: {"distributions": 1})
    manifest = refresh_locks.refresh(source=source, pants="pants", uv="uv", destination=destination)
    assert manifest["inputs"] == refresh_locks.fingerprint(source)
    assert manifest["uv_version"] == UV_VERSION
    assert set(manifest["resolves"]) == {"core", "sandbox"}
    assert (destination / "core.lock").read_text() == "generated lock"
    assert (destination / "sandbox.lock").read_text() == "generated lock"
    assert json.loads((destination / "lock-inputs.json").read_text()) == manifest


def test_source_change_during_generation_blocks_publication(tmp_path, monkeypatch):
    source, destination = generation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(refresh_locks, "validate_lock", lambda *args: {"distributions": 1})

    def generate(*args, **kwargs):
        (source / "uv.lock").write_text("changed input")

    monkeypatch.setattr(refresh_locks.subprocess, "run", generate)
    with pytest.raises(ValueError, match="changed during"):
        refresh_locks.refresh(source=source, pants="pants", uv="uv", destination=destination)
    assert (destination / "core.lock").read_text() == "existing core"
    assert not (destination / "lock-inputs.json").exists()


def test_interpreter_setup_uses_sampled_checkout_not_controller(tmp_path, monkeypatch):
    (tmp_path / ".python-version").write_text("3.13.14")
    calls = []
    monkeypatch.setattr(refresh_locks.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setattr(
        refresh_locks.subprocess,
        "check_output",
        lambda command, **kw: (calls.append((command, kw)) or "/runtime/bin/python\n"),
    )
    environment = refresh_locks.interpreter_environment(tmp_path, "historical-uv")
    assert len(calls) == 2
    assert all(options["cwd"] == tmp_path for _, options in calls)
    assert environment["PATH"].startswith("/runtime/bin:")
