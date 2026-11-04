# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ship source metadata needed by component locks when Gym is installed as a wheel."""

import tarfile
from pathlib import Path
from shutil import copyfile

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    """Deduplicate component locks while retaining directly consumed runtime locks."""

    def run(self) -> None:
        """Deduplicate component locks in the wheel and retain their source metadata."""
        super().run()
        destination = Path(self.build_lib) / "nemo_gym/resources/locked_project"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            copyfile(name, destination / name)
        build_root = Path(self.build_lib)
        locks = sorted(build_root.rglob("uv.lock"))
        archive = destination.parent / "component-locks.tar.xz"
        runtime_locks = {
            Path("resources_servers/chemcotbench/molopt/uv.lock"),
            Path("responses_api_agents/osworld_agent/runtime/uv.lock"),
        }
        with tarfile.open(archive, "w:xz") as stream:
            for path in locks:
                relative = path.relative_to(build_root)
                info = stream.gettarinfo(str(relative), arcname=relative.as_posix())
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with relative.open("rb") as source:
                    stream.addfile(info, source)
                if relative not in runtime_locks:
                    path.write_text("# Stored in nemo_gym/resources/component-locks.tar.xz\n")


setup(cmdclass={"build_py": BuildPy})
