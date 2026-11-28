# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared modules directly under ``benchmarks/`` ship in the wheel, because the benchmark ``prepare.py`` files import them."""

import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
# Every module directly under ``benchmarks/``: a benchmark that adds a shared helper needs no change here.
SHARED_MODULES = {path.stem for path in (ROOT / "benchmarks").glob("*.py") if path.stem != "__init__"}


def test_shared_benchmark_modules_are_packaged() -> None:
    # ``setuptools`` is not in the locked dev environment (only the ``vllm`` extra reaches it).
    pytest.importorskip("setuptools")
    from setuptools import find_namespace_packages
    from setuptools.command.build_py import build_py
    from setuptools.dist import Distribution

    discovery = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"]["find"]
    packages = find_namespace_packages(where=str(ROOT), include=discovery["include"], exclude=discovery["exclude"])
    assert "benchmarks" in packages
    assert "audio_manifest" in SHARED_MODULES
    distribution = Distribution(
        {"script_name": "setup.py", "packages": ["benchmarks"], "package_dir": {"": str(ROOT)}}
    )
    command = build_py(distribution)
    command.finalize_options()
    packaged = {module for package, module, _ in command.find_all_modules() if package == "benchmarks"}
    assert SHARED_MODULES <= packaged
    for module in SHARED_MODULES:
        assert (ROOT / "benchmarks" / f"{module}.py").is_file()
