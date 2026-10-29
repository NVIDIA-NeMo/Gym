# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of gym dataset init.

The task folder scaffold, the dataset.toml it writes and the refusals on existing folders.
"""

import subprocess
import tomllib

import pytest

from nemo_gym.data.tasks.harbor import discover_tasks
from nemo_gym.data.tasks.harbor.materialize import write_rows
from tests.unit_tests.test_harbor_tasks import fake_registry  # noqa: F401 (fixture)


@pytest.mark.usefixtures("fake_registry")
class TestDatasetInit:
    """The scaffold is read back by the loader, so `gym dataset init` cannot drift from what runs."""

    def test_scaffold_loads_validates_and_materializes(self, tmp_path):
        from nemo_gym.data.tasks.harbor.dataset_config import read_dataset_config
        from nemo_gym.data.tasks.harbor.scaffold import init_dataset

        folder = init_dataset(tmp_path, "my-dataset", image="ubuntu:24.04")
        assert folder == tmp_path / "my-dataset"
        # Harbor 0.22.0 requires `org/name` in the manifest and in [task]: a bare name gets the `local` org.
        assert tomllib.loads((folder / "dataset.toml").read_text())["dataset"]["name"] == "local/my-dataset"
        (task,) = discover_tasks(folder)
        assert task.task_id == "hello" and task.image == "ubuntu:24.04" and task.has_solution
        assert task.config.task["name"] == "local/hello"
        assert task.config.agent.timeout_sec == 600 and task.config.environment.cpus == 1
        assert read_dataset_config(folder).is_empty  # defaults only, and the per-task example is a comment
        rows = write_rows([task], folder.name, tmp_path / "out" / "tasks.jsonl")
        assert rows[0]["task_id"] == {"taskset": "my-dataset", "task_id": "hello"}
        for script in (folder / "hello" / "tests" / "test.sh", folder / "hello" / "solution" / "solve.sh"):
            assert script.stat().st_mode & 0o111
            assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
        with pytest.raises(FileExistsError):
            init_dataset(tmp_path, "my-dataset")

    def test_org_name_is_kept_and_the_folder_is_the_short_name(self, tmp_path):
        """`gym dataset init acme/my-dataset`: the manifest keeps the org, the folder is the name without it."""
        from nemo_gym.data.tasks.harbor.dataset_config import read_dataset_config
        from nemo_gym.data.tasks.harbor.scaffold import init_dataset

        folder = init_dataset(tmp_path, "acme/my-dataset")
        assert folder == tmp_path / "my-dataset" and not (tmp_path / "acme").exists()
        assert tomllib.loads((folder / "dataset.toml").read_text())["dataset"]["name"] == "acme/my-dataset"
        (task,) = discover_tasks(folder)
        assert task.task_id == "hello" and task.config.task["name"] == "acme/hello"
        assert read_dataset_config(folder).is_empty
        with pytest.raises(FileExistsError):  # same folder as the bare name, since the org is not a directory
            init_dataset(tmp_path, "my-dataset")

    @pytest.mark.parametrize("name", ["a/b/c", ".hidden", "org/.hidden", "bad..name", "with space", "org/", "/name"])
    def test_names_harbor_rejects_are_refused_before_anything_is_written(self, tmp_path, name):
        from nemo_gym.data.tasks.harbor.scaffold import init_dataset

        with pytest.raises(ValueError, match="org/name"):
            init_dataset(tmp_path, name)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("name", ["with space", "my-dataset"])
    def test_cli_reports_a_refused_or_existing_name_as_a_usage_error(self, tmp_path, monkeypatch, capsys, name):
        """`gym dataset init` turns the scaffold's ValueError and FileExistsError into argparse's usage error
        (exit 2, message on stderr, no traceback), as the other `gym` commands report bad input."""
        import sys

        from nemo_gym.cli.main import main

        (tmp_path / "my-dataset").mkdir()  # makes the second name an existing folder
        monkeypatch.setattr(sys, "argv", ["gym", "dataset", "init", name, "--root", str(tmp_path)])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        out, err = capsys.readouterr()
        assert "Wrote" not in out
        assert "gym dataset init: error:" in err and ("org/name" in err or "already exists" in err)
        assert [p.name for p in tmp_path.iterdir()] == ["my-dataset"]  # nothing written for either name
        assert list((tmp_path / "my-dataset").iterdir()) == []

    @pytest.mark.parametrize("root", ["plain", "My Datasets"])
    def test_cli_next_steps_quote_the_folder_the_shell_would_split(self, tmp_path, monkeypatch, capsys, root):
        """`gym dataset init` accepts a root with a space in it, so the validate and eval commands it prints must
        name that folder as one argument when pasted back; a plain path prints unquoted as before."""
        import shlex
        import sys

        from nemo_gym.cli.main import main

        (tmp_path / root).mkdir()
        monkeypatch.setattr(sys, "argv", ["gym", "dataset", "init", "my-dataset", "--root", str(tmp_path / root)])
        main()
        folder = tmp_path / root / "my-dataset"
        out = capsys.readouterr().out
        validate_line, eval_line = (
            line for line in out.splitlines() if "gym dataset validate" in line or "gym eval run" in line
        )
        assert shlex.split(validate_line.split("  #", 1)[0]) == ["Next:", "gym", "dataset", "validate", str(folder)]
        assert shlex.split(eval_line)[:5] == ["gym", "eval", "run", str(folder), "--agent"]
        if root == "plain":
            assert f"Next: gym dataset validate {folder}  #" in out  # no quotes on a path that needs none
        else:
            assert f"'{folder}'" in validate_line and f"'{folder}'" in eval_line

    def test_harbor_accepts_the_scaffold_and_its_serializer_drops_the_gym_table(self, tmp_path):
        """The caveat the scaffold comment and the docs state, checked against Harbor itself: its manifest model
        parses what `gym dataset init` writes, and `harbor add/remove/sync` (which write `to_toml()`) lose [gym]."""
        # Import the library modules directly: a bare `importorskip("harbor")` can resolve to Gym's own
        # `harbor` test package when this file is collected with environment_servers/harbor/tests.
        pytest.importorskip("harbor.models.dataset.manifest")
        pytest.importorskip("harbor.models.task.config")
        pytest.importorskip("harbor.models.task.task")
        from harbor.models.dataset.manifest import DatasetManifest
        from harbor.models.task.config import TaskConfig
        from pydantic import ValidationError

        from nemo_gym.data.tasks.harbor.scaffold import init_dataset

        folder = init_dataset(tmp_path, "my-dataset")
        manifest = DatasetManifest.from_toml((folder / "dataset.toml").read_text())
        assert manifest.dataset.name == "local/my-dataset"
        assert TaskConfig.model_validate(tomllib.loads((folder / "hello" / "task.toml").read_text())).task.name == (
            "local/hello"
        )
        with pytest.raises(ValidationError, match="org/name"):  # the bare name the scaffold used to write
            DatasetManifest.from_toml('[dataset]\nname = "my-dataset"\n')
        rewritten = tomllib.loads(manifest.to_toml())
        assert "gym" not in rewritten and rewritten["dataset"]["name"] == "local/my-dataset"
