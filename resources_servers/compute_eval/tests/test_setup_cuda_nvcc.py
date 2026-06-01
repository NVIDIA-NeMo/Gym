# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for setup_cuda_nvcc.ensure_cuda_nvcc resolution order.

Locks in the parity guard: when a caller pins a prefix (via the env var or
the shared-lustre default), the function must NOT fall back to PATH-nvcc.
A container shipping /usr/local/cuda/bin/nvcc at a different CUDA version
would otherwise silently win and introduce toolchain drift between Skills
and Gym.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import setup_cuda_nvcc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prefix_with_nvcc(tmp_path: Path) -> Path:
    """Build a fake install prefix containing env/bin/nvcc + a targets dir."""
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "nvcc").touch()
    # _wire_paths walks targets/<arch>-linux/ — give it one
    arch = env / "targets" / "x86_64-linux"
    (arch / "lib").mkdir(parents=True)
    (arch / "include").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def clean_env(monkeypatch):
    """Strip COMPUTE_EVAL_CUDA_NVCC_PREFIX out of the env for each test."""
    monkeypatch.delenv("COMPUTE_EVAL_CUDA_NVCC_PREFIX", raising=False)


# ---------------------------------------------------------------------------
# Branch 1: prefix has env/bin/nvcc → reuse (no install, no PATH check)
# ---------------------------------------------------------------------------


class TestExistingInstall:
    def test_reuses_existing_install_even_when_nvcc_on_path(self, tmp_path, clean_env):
        prefix = _make_prefix_with_nvcc(tmp_path)
        # PATH-nvcc would normally short-circuit, but we should hit branch 1 first
        with (
            patch.object(setup_cuda_nvcc.shutil, "which", return_value="/usr/local/cuda/bin/nvcc"),
            patch.object(setup_cuda_nvcc, "_download_micromamba") as dl,
            patch.object(setup_cuda_nvcc, "_run_micromamba_install") as inst,
        ):
            nvcc = setup_cuda_nvcc.ensure_cuda_nvcc(prefix)
        assert nvcc == prefix / "env" / "bin" / "nvcc"
        # Neither install path was taken
        dl.assert_not_called()
        inst.assert_not_called()


# ---------------------------------------------------------------------------
# Branch 2: pinned prefix without an existing install → install, NEVER PATH
# ---------------------------------------------------------------------------


class TestPinnedPrefixInstalls:
    def test_pinned_via_env_var_installs_even_when_nvcc_on_path(self, tmp_path, monkeypatch):
        """The container has nvcc but the user pinned a specific prefix via
        env var — must install into the pinned prefix, not use PATH nvcc."""
        prefix = tmp_path / "explicit_prefix"
        monkeypatch.setenv("COMPUTE_EVAL_CUDA_NVCC_PREFIX", str(prefix))
        with (
            patch.object(setup_cuda_nvcc.shutil, "which", return_value="/usr/local/cuda/bin/nvcc"),
            patch.object(setup_cuda_nvcc, "_download_micromamba") as dl,
            patch.object(setup_cuda_nvcc, "_run_micromamba_install") as inst,
            patch.object(setup_cuda_nvcc, "_wire_paths", return_value=prefix / "env" / "bin" / "nvcc") as wire,
        ):
            setup_cuda_nvcc.ensure_cuda_nvcc(prefix)
        # Install fired even though PATH-nvcc was available
        dl.assert_called_once_with(prefix)
        inst.assert_called_once()
        wire.assert_called_once_with(prefix / "env")

    def test_shared_lustre_default_installs_even_when_nvcc_on_path(self, clean_env):
        """When the caller passes the shared-lustre default (the cluster
        production path), PATH-nvcc must not short-circuit."""
        with (
            patch.object(setup_cuda_nvcc.shutil, "which", return_value="/usr/local/cuda/bin/nvcc"),
            patch.object(setup_cuda_nvcc, "_download_micromamba") as dl,
            patch.object(setup_cuda_nvcc, "_run_micromamba_install") as inst,
            patch.object(setup_cuda_nvcc, "_wire_paths", return_value=Path("/dev/null")),
        ):
            setup_cuda_nvcc.ensure_cuda_nvcc(setup_cuda_nvcc._SHARED_PREFIX_DEFAULT)
        dl.assert_called_once()
        inst.assert_called_once()


# ---------------------------------------------------------------------------
# Branch 3: unpinned (local) prefix + nvcc on PATH → use PATH (dev convenience)
# ---------------------------------------------------------------------------


class TestUnpinnedFallback:
    def test_unpinned_prefix_uses_path_nvcc_when_available(self, tmp_path, clean_env):
        """On dev / CI without a shared lustre install, fall back to PATH-nvcc
        so test runs don't pay the ~5 min micromamba install on every invocation."""
        # Use the local default to signal "unpinned"
        local_prefix = setup_cuda_nvcc._LOCAL_PREFIX_DEFAULT
        path_nvcc = tmp_path / "fake-host-nvcc"
        path_nvcc.touch()
        with (
            patch.object(setup_cuda_nvcc.shutil, "which", return_value=str(path_nvcc)),
            patch.object(setup_cuda_nvcc, "_download_micromamba") as dl,
            patch.object(setup_cuda_nvcc, "_run_micromamba_install") as inst,
        ):
            result = setup_cuda_nvcc.ensure_cuda_nvcc(local_prefix)
        assert result == path_nvcc
        dl.assert_not_called()
        inst.assert_not_called()


# ---------------------------------------------------------------------------
# Branch 4: no prefix, no PATH-nvcc → install
# ---------------------------------------------------------------------------


class TestColdInstall:
    def test_installs_when_neither_prefix_nor_path_has_nvcc(self, tmp_path, clean_env):
        local_prefix = setup_cuda_nvcc._LOCAL_PREFIX_DEFAULT
        with (
            patch.object(setup_cuda_nvcc.shutil, "which", return_value=None),
            patch.object(setup_cuda_nvcc, "_download_micromamba") as dl,
            patch.object(setup_cuda_nvcc, "_run_micromamba_install") as inst,
            patch.object(setup_cuda_nvcc, "_wire_paths", return_value=Path("/dev/null")),
        ):
            setup_cuda_nvcc.ensure_cuda_nvcc(local_prefix)
        dl.assert_called_once_with(local_prefix)
        inst.assert_called_once()
