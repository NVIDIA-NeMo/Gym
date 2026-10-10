# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Building and packing the ``auto`` agent runtime bundle."""

import tarfile
from pathlib import Path
from unittest.mock import patch

from responses_api_agents.anyswe_agent import app as anyswe_app
from responses_api_agents.anyswe_agent.app import GymAgentHarnessProcessor, _pack_runtime
from responses_api_agents.anyswe_agent.tests.test_app import _config


class TestRuntimeBundle:
    def test_recipe_changes_when_nemo_gym_changes(self, tmp_path: Path) -> None:
        (tmp_path / "nemo_gym" / "token_id_capture").mkdir(parents=True)
        library_file = tmp_path / "nemo_gym" / "token_id_capture" / "lineage.py"
        library_file.write_text("VERSION = 1\n")
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        processor = GymAgentHarnessProcessor(config=_config(agent_server_module="responses_api_agents.x_agent.app"))

        with patch.object(anyswe_app, "PARENT_DIR", tmp_path):
            before = processor._recipe()
            assert processor._recipe() == before
            library_file.write_text("VERSION = 2\n")
            after_library_change = processor._recipe()
            (tmp_path / "pyproject.toml").write_text("[project]\nversion = '2'\n")
            after_pyproject_change = processor._recipe()

        assert len({before, after_library_change, after_pyproject_change}) == 3

    def test_concurrent_archive_rebuilds_do_not_collide(self, tmp_path: Path) -> None:
        deps = tmp_path / "deps"
        (deps / "bin").mkdir(parents=True)
        (deps / "bin" / "python").write_text("#!/bin/sh\n")
        archive_path = tmp_path / ".deps.tar.gz"
        real_open = tarfile.open
        raced = []

        class _RacingArchive:
            """Another server rebuilds the same archive between this one's write and its replace()."""

            def __init__(self, archive: tarfile.TarFile) -> None:
                self.archive = archive

            def __enter__(self) -> tarfile.TarFile:
                return self.archive.__enter__()

            def __exit__(self, *exc_info) -> None:
                self.archive.__exit__(*exc_info)
                if not raced:
                    raced.append(True)
                    _pack_runtime(deps, archive_path)

        def open_and_race(*args, **kwargs):
            return _RacingArchive(real_open(*args, **kwargs))

        with patch.object(anyswe_app.tarfile, "open", open_and_race):
            _pack_runtime(deps, archive_path)

        assert raced
        with tarfile.open(archive_path) as archive:
            assert "./bin/python" in archive.getnames()
        assert sorted(path.name for path in tmp_path.iterdir()) == [".deps.tar.gz", "deps"]

    def test_setup_script_reinstalls_nemo_gym_from_the_checkout(self) -> None:
        helper = (Path(__file__).parent.parent / "setup_scripts" / "_portable_python.sh").read_text()
        assert 'install_python_packages --no-deps --force-reinstall "$NEMO_GYM_ROOT"' in helper
        assert 'install_python_packages --no-deps --reinstall "$NEMO_GYM_ROOT"' in helper
