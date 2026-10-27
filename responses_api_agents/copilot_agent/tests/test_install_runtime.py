# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest


def load_installer():
    path = Path(__file__).parents[1] / "install_copilot_runtime.py"
    spec = importlib.util.spec_from_file_location("install_copilot_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Stands in for the standalone binary: the first run unpacks into $HOME/.cache like Copilot 1.0.103.
FAKE_BINARY = """#!/bin/sh
if [ -n "${COPILOT_CLI_DIST_DIR:-}" ]; then
  [ -d "$COPILOT_CLI_DIST_DIR" ] || exit 9
else
  mkdir -p "$HOME/.cache/copilot/pkg/linux-x64/1.0.103/builtin"
fi
echo "GitHub Copilot CLI 1.0.103."
"""


def package(binary=FAKE_BINARY) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        data = binary.encode()
        info = tarfile.TarInfo("package/copilot")
        info.size = len(data)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()


@pytest.fixture
def installer(monkeypatch):
    module = load_installer()
    tarball = package()
    integrity = "sha512-" + base64.b64encode(hashlib.sha512(tarball).digest()).decode()
    downloads = []

    def fetch(url):
        downloads.append(url)
        if url.endswith("/@github/copilot-linux-x64/1.0.103"):
            return json.dumps({"dist": {"tarball": "https://registry.test/pkg.tgz", "integrity": integrity}}).encode()
        assert url == "https://registry.test/pkg.tgz"
        return module_tarball[0]

    module_tarball = [tarball]
    monkeypatch.setattr(module, "package_name", lambda: "copilot-linux-x64")
    monkeypatch.setattr(module, "fetch", fetch)
    return module, downloads, module_tarball


def test_install_verifies_unpacks_privately_and_reuses(installer, tmp_path):
    module, downloads, _ = installer
    runtime = str(tmp_path / "nemo-gym-copilot-1.0.103")
    record = module.install(runtime, "1.0.103", "https://registry.test")
    assert record["binary"] == f"{runtime}/copilot"
    assert record["dist_dir"] == f"{runtime}/home/.cache/copilot/pkg/linux-x64/1.0.103"
    assert os.path.isfile(f"{runtime}/ready") and os.access(record["binary"], os.X_OK)
    assert not [path for path in os.listdir(tmp_path) if ".partial-" in path]
    assert len(downloads) == 2
    assert module.install(runtime, "1.0.103", "https://registry.test") == record
    assert len(downloads) == 2


def test_integrity_mismatch_is_refused_without_a_runtime(installer, tmp_path):
    module, _, tarball = installer
    tarball[0] = package(FAKE_BINARY.replace("1.0.103.", "1.0.103!"))
    runtime = str(tmp_path / "runtime")
    with pytest.raises(SystemExit, match="Integrity mismatch"):
        module.install(runtime, "1.0.103", "https://registry.test")
    assert not os.path.exists(runtime)
    assert not [path for path in os.listdir(tmp_path) if ".partial-" in path]


def test_runtime_with_another_version_is_replaced(installer, tmp_path):
    module, downloads, _ = installer
    runtime = str(tmp_path / "runtime")
    module.install(runtime, "1.0.103", "https://registry.test")
    record_path = Path(runtime) / "runtime.json"
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps(record | {"version": "1.0.0"}))
    module.install(runtime, "1.0.103", "https://registry.test")
    assert len(downloads) == 4


@pytest.mark.parametrize(
    "machine,glibc,musl,expected",
    [
        ("x86_64", "glibc 2.31", False, "copilot-linux-x64"),
        ("aarch64", "glibc 2.35", False, "copilot-linux-arm64"),
        ("x86_64", None, True, "copilot-linuxmusl-x64"),
    ],
)
def test_package_matches_the_sandbox_platform(monkeypatch, machine, glibc, musl, expected):
    module = load_installer()
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.platform, "machine", lambda: machine)
    monkeypatch.setattr(module.os, "confstr", lambda name: glibc)
    monkeypatch.setattr(module.os, "listdir", lambda path: ["ld-musl-x86_64.so.1"] if musl else ["libc.so.6"])
    assert module.package_name() == expected


@pytest.mark.parametrize(
    "platform_name,machine,glibc,reason",
    [
        ("darwin", "x86_64", "glibc 2.31", "Linux only"),
        ("linux", "ppc64le", "glibc 2.31", "x86_64 and aarch64"),
        ("linux", "x86_64", "glibc 2.17", "glibc 2.28 or newer"),
    ],
)
def test_unsupported_platforms_are_refused(monkeypatch, platform_name, machine, glibc, reason):
    module = load_installer()
    monkeypatch.setattr(module.sys, "platform", platform_name)
    monkeypatch.setattr(module.platform, "machine", lambda: machine)
    monkeypatch.setattr(module.os, "confstr", lambda name: glibc)
    with pytest.raises(SystemExit, match=reason):
        module.package_name()


@pytest.mark.parametrize("state", ["missing-ready", "missing-record", "broken-record"])
def test_interrupted_install_is_replaced(installer, tmp_path, state):
    module, downloads, _ = installer
    runtime = str(tmp_path / "runtime")
    module.install(runtime, "1.0.103", "https://registry.test")
    if state == "missing-ready":
        Path(runtime, "ready").unlink()
    elif state == "missing-record":
        Path(runtime, "runtime.json").unlink()
    else:
        Path(runtime, "runtime.json").write_text("{")
    recovered = module.install(runtime, "1.0.103", "https://registry.test")
    assert module.check(recovered, "1.0.103")
    assert Path(runtime, "ready").is_file()
    assert len(downloads) == 4
