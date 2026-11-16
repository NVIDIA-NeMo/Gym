# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep directly consumed runtime locks usable after wheel packaging."""

import runpy
import shutil
import tarfile
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_wheel_retains_installer_locks_and_archives_component_locks(tmp_path: Path) -> None:
    """Exercise the packaging hook against real runtime locks and an ordinary component."""
    paths = (
        Path("resources_servers/chemcotbench/molopt/uv.lock"),
        Path("responses_api_agents/osworld_agent/runtime/uv.lock"),
        Path("responses_api_agents/simple_agent/uv.lock"),
    )
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / relative, destination)
    setuptools = pytest.importorskip("setuptools")
    build_py = pytest.importorskip("setuptools.command.build_py").build_py
    with patch("setuptools.setup"):
        builder = runpy.run_path(str(ROOT / "setup.py"))["BuildPy"](setuptools.Distribution())
    builder.build_lib = str(tmp_path)
    with patch.object(build_py, "run"):
        builder.run()
    for relative in paths[:2]:
        assert (tmp_path / relative).read_bytes() == (ROOT / relative).read_bytes()
        assert tomllib.loads((tmp_path / relative).read_text())["package"]
    ordinary_lock = tmp_path / paths[2]
    assert ordinary_lock.stat().st_size < 100
    with tarfile.open(tmp_path / "nemo_gym/resources/component-locks.tar.xz") as archive:
        for relative in paths:
            assert archive.extractfile(relative.as_posix()).read() == (ROOT / relative).read_bytes()
