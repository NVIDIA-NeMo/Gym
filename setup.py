# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ship source metadata needed by component locks when Gym is installed as a wheel."""

from pathlib import Path
from shutil import copyfile

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self) -> None:
        super().run()
        destination = Path(self.build_lib) / "nemo_gym/resources/locked_project"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            copyfile(name, destination / name)


setup(cmdclass={"build_py": BuildPy})
