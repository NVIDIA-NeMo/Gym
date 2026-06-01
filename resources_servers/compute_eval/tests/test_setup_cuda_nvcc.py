# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for setup_cuda_nvcc.

Locks in the two-branch resolution and the parity guard: ``ensure_cuda_nvcc``
never falls back to ``which("nvcc")``. A container-bundled nvcc at a
different CUDA version would otherwise silently win and introduce toolchain
drift between consumers that share a pinned prefix.
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
# Branch 1: prefix has env/bin/nvcc → reuse
# ---------------------------------------------------------------------------


class TestExistingInstall:
    def test_reuses_existing_install(self, tmp_path, clean_env):
        prefix = _make_prefix_with_nvcc(tmp_path)
        with (
            patch.object(setup_cuda_nvcc, "_download_micromamba") as dl,
            patch.object(setup_cuda_nvcc, "_run_micromamba_install") as inst,
        ):
            nvcc = setup_cuda_nvcc.ensure_cuda_nvcc(prefix)
        assert nvcc == prefix / "env" / "bin" / "nvcc"
        dl.assert_not_called()
        inst.assert_not_called()


# ---------------------------------------------------------------------------
# Branch 2: no existing install → install via micromamba.
# In every case the install fires; we never inspect or use PATH-nvcc.
# ---------------------------------------------------------------------------


class TestAlwaysInstallsWithoutExistingPrefix:
    def test_installs_into_default_prefix(self, tmp_path, clean_env):
        prefix = tmp_path / "fresh_prefix"
        with (
            patch.object(setup_cuda_nvcc, "_download_micromamba") as dl,
            patch.object(setup_cuda_nvcc, "_run_micromamba_install") as inst,
            patch.object(setup_cuda_nvcc, "_wire_paths", return_value=prefix / "env" / "bin" / "nvcc"),
        ):
            setup_cuda_nvcc.ensure_cuda_nvcc(prefix)
        dl.assert_called_once_with(prefix)
        inst.assert_called_once()

    def test_env_var_override_routes_install_to_custom_prefix(self, tmp_path, monkeypatch):
        """COMPUTE_EVAL_CUDA_NVCC_PREFIX redirects the install. Used when
        multiple consumers want to share a single install — they all point
        at the same prefix and the first to boot performs the install."""
        custom = tmp_path / "shared_prefix"
        monkeypatch.setenv("COMPUTE_EVAL_CUDA_NVCC_PREFIX", str(custom))
        # Re-import to recompute DEFAULT_PREFIX. Simpler: just verify
        # _resolve_default_prefix picks up the env var.
        assert setup_cuda_nvcc._resolve_default_prefix() == custom


# ---------------------------------------------------------------------------
# Parity guard: PATH-nvcc MUST be ignored even when available.
# ---------------------------------------------------------------------------


class TestPathFallbackDisabled:
    """If a container ships /usr/local/cuda/bin/nvcc at a different CUDA
    version, ensure_cuda_nvcc must not pick it up — install into prefix
    regardless. setup_cuda_nvcc.py doesn't even import shutil any more, so
    the only way PATH could leak in would be via subprocess.run downstream.
    """

    def test_setup_does_not_import_shutil(self):
        import importlib

        importlib.reload(setup_cuda_nvcc)
        assert not hasattr(setup_cuda_nvcc, "shutil"), (
            "setup_cuda_nvcc must not depend on shutil — "
            "a previous version used shutil.which('nvcc') to short-circuit "
            "the install, which is the parity hazard this guard prevents."
        )

    def test_install_fires_even_with_a_plausible_path_nvcc(self, tmp_path, clean_env, monkeypatch):
        """Smoke test: even if PATH contains a directory holding a binary
        called 'nvcc', the install path is taken because the function
        never consults PATH."""
        fake_nvcc_dir = tmp_path / "fakebin"
        fake_nvcc_dir.mkdir()
        (fake_nvcc_dir / "nvcc").touch(mode=0o755)
        monkeypatch.setenv("PATH", str(fake_nvcc_dir))

        prefix = tmp_path / "fresh_prefix"
        with (
            patch.object(setup_cuda_nvcc, "_download_micromamba") as dl,
            patch.object(setup_cuda_nvcc, "_run_micromamba_install") as inst,
            patch.object(setup_cuda_nvcc, "_wire_paths", return_value=prefix / "env" / "bin" / "nvcc"),
        ):
            setup_cuda_nvcc.ensure_cuda_nvcc(prefix)
        dl.assert_called_once()
        inst.assert_called_once()
