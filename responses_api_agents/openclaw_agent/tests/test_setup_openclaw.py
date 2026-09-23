# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import tarfile
import zipfile
from pathlib import Path

import pytest

from responses_api_agents.openclaw_agent import setup_openclaw


# Every (sys.platform, platform.machine()) pair we claim to support, mapped to the
# archive nodejs.org actually publishes. Verified against
# https://nodejs.org/dist/v24.21.0/SHASUMS256.txt.
SUPPORTED = [
    ("linux", "x86_64", "node-v24.21.0-linux-x64.tar.xz"),
    ("linux", "aarch64", "node-v24.21.0-linux-arm64.tar.xz"),
    ("darwin", "x86_64", "node-v24.21.0-darwin-x64.tar.xz"),
    ("darwin", "arm64", "node-v24.21.0-darwin-arm64.tar.xz"),
    ("win32", "AMD64", "node-v24.21.0-win-x64.zip"),
    ("win32", "ARM64", "node-v24.21.0-win-arm64.zip"),
]


@pytest.fixture
def fake_platform(monkeypatch):
    """Pretend to run on an arbitrary (sys.platform, machine) pair."""

    def _set(sys_platform: str, machine: str) -> None:
        monkeypatch.setattr(setup_openclaw.sys, "platform", sys_platform)
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: machine)
        # On Windows the module asks _windows_machine(), not platform.machine(),
        # because platform.machine() is unreliable under ARM64 x64 emulation.
        monkeypatch.setattr(setup_openclaw, "_windows_machine", lambda: machine)

    return _set


class _FakeWin32Function:
    """Callable Win32 export that accepts ctypes signature attributes."""

    def __init__(self, implementation):
        self._implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._implementation(*args)


class _FakeKernel32:
    """kernel32 double whose IsWow64Process2 reports a fixed native machine."""

    def __init__(self, native_machine: int, result: int = 1):
        self.current_process_handle = object()
        self.received_handle = None
        self.GetCurrentProcess = _FakeWin32Function(lambda: self.current_process_handle)

        def _is_wow64_process2(handle, process_machine_ref, native_machine_ref):
            self.received_handle = handle
            native_machine_ref._obj.value = native_machine
            return result

        self.IsWow64Process2 = _FakeWin32Function(_is_wow64_process2)


class _FakeWindll:
    """Stands in for ``ctypes.windll`` exposing a fake kernel32."""

    def __init__(self, kernel32):
        self.kernel32 = kernel32


class _Kernel32WithoutIsWow64Process2:
    """kernel32 double for pre-Windows-10 kernels: no IsWow64Process2 export."""

    def __init__(self):
        self.GetCurrentProcess = _FakeWin32Function(lambda: object())


def _patch_kernel32(monkeypatch, kernel32) -> None:
    """Point ``setup_openclaw.ctypes.windll.kernel32`` at a test double."""
    monkeypatch.setattr(setup_openclaw.ctypes, "windll", _FakeWindll(kernel32))


class TestNodeDistUrl:
    @pytest.mark.parametrize(("sys_platform", "machine", "archive"), SUPPORTED)
    def test_builds_url_published_by_nodejs_org(self, fake_platform, sys_platform, machine, archive):
        fake_platform(sys_platform, machine)
        assert setup_openclaw._node_dist_url("24.21.0") == f"https://nodejs.org/dist/v24.21.0/{archive}"

    def test_windows_uses_zip_and_others_use_tar_xz(self, fake_platform):
        fake_platform("win32", "AMD64")
        assert setup_openclaw._node_dist_url("24.21.0").endswith(".zip")
        fake_platform("linux", "x86_64")
        assert setup_openclaw._node_dist_url("24.21.0").endswith(".tar.xz")

    def test_machine_spelling_is_case_insensitive(self, fake_platform):
        fake_platform("linux", "X86_64")
        assert "-linux-x64." in setup_openclaw._node_dist_url("24.21.0")

    def test_version_is_interpolated(self, fake_platform):
        fake_platform("linux", "x86_64")
        assert setup_openclaw._node_dist_url("26.1.0") == (
            "https://nodejs.org/dist/v26.1.0/node-v26.1.0-linux-x64.tar.xz"
        )

    def test_unsupported_os_raises_actionable_error(self, fake_platform):
        fake_platform("freebsd14", "x86_64")
        with pytest.raises(RuntimeError, match="freebsd14"):
            setup_openclaw._node_dist_url("24.21.0")

    def test_unsupported_arch_raises_actionable_error(self, fake_platform):
        fake_platform("linux", "riscv64")
        with pytest.raises(RuntimeError, match="riscv64"):
            setup_openclaw._node_dist_url("24.21.0")


class TestWindowsMachine:
    """platform.machine() on Windows reports the *interpreter's* architecture.

    An x64 python.exe running under emulation on ARM64 silicon gets "AMD64",
    and CPython 3.13+ flips between WMI truth (ARM64) and the emulated env var
    (AMD64) depending on whether the WMI service answers in time. The module
    therefore must not trust platform.machine() for the download choice.
    """

    def test_prefers_iswow64process2_over_platform_machine(self, monkeypatch):
        """The native machine and documented current-process handle drive detection."""
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "AMD64")
        kernel32 = _FakeKernel32(native_machine=setup_openclaw._IMAGE_FILE_MACHINE_ARM64)
        _patch_kernel32(monkeypatch, kernel32)
        assert setup_openclaw._windows_machine() == "ARM64"
        assert kernel32.received_handle is kernel32.current_process_handle
        assert kernel32.GetCurrentProcess.argtypes == []
        assert kernel32.GetCurrentProcess.restype is setup_openclaw.wintypes.HANDLE
        assert kernel32.IsWow64Process2.argtypes == [
            setup_openclaw.wintypes.HANDLE,
            setup_openclaw.ctypes.POINTER(setup_openclaw.wintypes.USHORT),
            setup_openclaw.ctypes.POINTER(setup_openclaw.wintypes.USHORT),
        ]
        assert kernel32.IsWow64Process2.restype is setup_openclaw.wintypes.BOOL

    def test_recognised_amd64_native_machine(self, monkeypatch):
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "ARM64")
        _patch_kernel32(monkeypatch, _FakeKernel32(native_machine=setup_openclaw._IMAGE_FILE_MACHINE_AMD64))
        assert setup_openclaw._windows_machine() == "AMD64"

    def test_unrecognised_machine_code_raises_actionable_error(self, monkeypatch):
        """A valid-but-unmapped kernel answer is authoritative truth we cannot
        translate; falling back to platform.machine() (the lying primitive) would
        be worse than surfacing the gap, so the caller sees the actionable error."""
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "AMD64")
        _patch_kernel32(monkeypatch, _FakeKernel32(native_machine=0xFFFF))
        with pytest.raises(RuntimeError, match="architecture"):
            setup_openclaw._node_platform()

    def test_api_absent_falls_back_to_platform_machine(self, monkeypatch):
        """Pre-Windows-10 kernels have no IsWow64Process2; behave as before."""
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "AMD64")
        _patch_kernel32(monkeypatch, _Kernel32WithoutIsWow64Process2())
        assert setup_openclaw._windows_machine() == "AMD64"

    def test_api_failure_falls_back_to_platform_machine(self, monkeypatch):
        monkeypatch.setattr(setup_openclaw.platform, "machine", lambda: "AMD64")
        _patch_kernel32(
            monkeypatch,
            _FakeKernel32(native_machine=setup_openclaw._IMAGE_FILE_MACHINE_ARM64, result=0),
        )
        assert setup_openclaw._windows_machine() == "AMD64"

    def test_emulated_interpreter_gets_native_download(self, fake_platform):
        """End-to-end trap: x64-emulated interpreter on an ARM64 host must pick
        the arm64 zip, not the x64 zip its own image would suggest."""
        fake_platform("win32", "ARM64")
        assert setup_openclaw._node_dist_url("24.21.0").endswith("node-v24.21.0-win-arm64.zip")


class TestNodeBinDir:
    def test_windows_launchers_sit_at_the_root(self, fake_platform):
        """The win-x64 zip has node.exe/npm.cmd at the top level, with no bin/."""
        fake_platform("win32", "AMD64")
        assert setup_openclaw._node_bin_dir(Path("/prefix")) == Path("/prefix")

    @pytest.mark.parametrize(("sys_platform", "machine"), [("linux", "x86_64"), ("darwin", "arm64")])
    def test_posix_uses_bin_subdirectory(self, fake_platform, sys_platform, machine):
        fake_platform(sys_platform, machine)
        assert setup_openclaw._node_bin_dir(Path("/prefix")) == Path("/prefix/bin")


class TestExtractNodeArchive:
    def test_extracts_tar_xz(self, tmp_path):
        payload = tmp_path / "node-v24.21.0-linux-x64" / "bin"
        payload.mkdir(parents=True)
        (payload / "node").write_text("#!/bin/sh\n")
        archive = tmp_path / "node.tar.xz"
        with tarfile.open(archive, "w:xz") as tf:
            tf.add(payload.parent, arcname="node-v24.21.0-linux-x64")

        dest = tmp_path / "dest"
        dest.mkdir()
        setup_openclaw._extract_node_archive(archive, dest)

        assert (dest / "node-v24.21.0-linux-x64" / "bin" / "node").is_file()

    def test_extracts_zip(self, tmp_path):
        archive = tmp_path / "node.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("node-v24.21.0-win-x64/node.exe", "binary")

        dest = tmp_path / "dest"
        dest.mkdir()
        setup_openclaw._extract_node_archive(archive, dest)

        assert (dest / "node-v24.21.0-win-x64" / "node.exe").read_text() == "binary"


class TestFlattenExtractedNode:
    def test_hoists_payload_into_prefix(self, tmp_path):
        nested = tmp_path / "node-v24.21.0-darwin-arm64" / "bin"
        nested.mkdir(parents=True)
        (nested / "node").write_text("x")

        setup_openclaw._flatten_extracted_node(tmp_path)

        assert (tmp_path / "bin" / "node").is_file()
        assert not (tmp_path / "node-v24.21.0-darwin-arm64").exists()

    def test_raises_when_archive_layout_is_unexpected(self, tmp_path):
        (tmp_path / "unrelated").mkdir()
        with pytest.raises(RuntimeError, match="node-\\*"):
            setup_openclaw._flatten_extracted_node(tmp_path)


class TestResolveVersions:
    def test_config_value_is_used_when_env_is_unset(self, monkeypatch):
        monkeypatch.delenv(setup_openclaw.OPENCLAW_VERSION_ENV, raising=False)
        assert setup_openclaw.resolve_openclaw_version("2026.6.11") == "2026.6.11"

    def test_defaults_when_nothing_is_supplied(self, monkeypatch):
        monkeypatch.delenv(setup_openclaw.OPENCLAW_VERSION_ENV, raising=False)
        monkeypatch.delenv(setup_openclaw.NODE_VERSION_ENV, raising=False)
        assert setup_openclaw.resolve_openclaw_version(None) == setup_openclaw.DEFAULT_OPENCLAW_VERSION
        assert setup_openclaw.resolve_node_version() == setup_openclaw.DEFAULT_NODE_VERSION

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv(setup_openclaw.OPENCLAW_VERSION_ENV, "2026.9.5")
        monkeypatch.setenv(setup_openclaw.NODE_VERSION_ENV, "26.1.0")
        assert setup_openclaw.resolve_openclaw_version("2026.6.11") == "2026.9.5"
        assert setup_openclaw.resolve_node_version() == "26.1.0"

    def test_empty_env_falls_back_to_config(self, monkeypatch):
        monkeypatch.setenv(setup_openclaw.OPENCLAW_VERSION_ENV, "")
        assert setup_openclaw.resolve_openclaw_version("2026.6.11") == "2026.6.11"

    def test_default_node_satisfies_default_openclaw_engine_range(self):
        """openclaw 2026.9.4 declares engines.node '>=24.16.0 <25 || >=26.1.0'."""
        major, minor, _ = (int(p) for p in setup_openclaw.DEFAULT_NODE_VERSION.split("."))
        assert (major == 24 and minor >= 16) or (major, minor) >= (26, 1)
