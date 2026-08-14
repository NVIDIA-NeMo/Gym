# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for AppWorld setup (isolated venv, install, download) and data prep.

The real installation is exercised once by the end-to-end test in
``test_app.py``; here the subprocess calls are stubbed so the branching — fast
path, uv vs. stdlib venv, verification failures — is covered without network.
"""

import json
import stat
import sys
from pathlib import Path

import pytest

from resources_servers.appworld import prepare_appworld, setup_appworld
from resources_servers.appworld.setup_appworld import (
    APPWORLD_REQUIREMENT,
    AppWorldInstall,
    ensure_appworld,
    is_installed,
)


def fake_venv(tmp_path: Path, exit_code: int = 0) -> Path:
    """A venv-shaped directory whose ``python`` exits with ``exit_code``."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    (venv / "bin" / "appworld").write_text("#!/bin/sh\n")
    return venv


def populate_data(root: Path) -> None:
    (root / "data" / "datasets").mkdir(parents=True)
    (root / "data" / "datasets" / "train.txt").write_text("aaa_1\naaa_2\n")


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------


def test_run_surfaces_a_failing_command(tmp_path):
    with pytest.raises(RuntimeError, match="failed \\(exit 3\\)"):
        setup_appworld._run(["sh", "-c", "echo nope >&2; exit 3"], str(tmp_path), str(tmp_path))


def test_run_exports_the_appworld_root_to_the_child(tmp_path):
    marker = tmp_path / "root.txt"

    setup_appworld._run(["sh", "-c", f'printf "%s" "$APPWORLD_ROOT" > {marker}'], str(tmp_path), "/the/root")

    assert marker.read_text() == "/the/root"


# ---------------------------------------------------------------------------
# is_installed
# ---------------------------------------------------------------------------


def test_is_installed_requires_the_appworld_apps_to_import(tmp_path):
    assert is_installed(fake_venv(tmp_path, exit_code=0)) is True


def test_is_installed_is_false_when_the_bundles_are_not_unpacked(tmp_path):
    assert is_installed(fake_venv(tmp_path, exit_code=1)) is False


def test_is_installed_is_false_without_the_cli(tmp_path):
    venv = fake_venv(tmp_path)
    (venv / "bin" / "appworld").unlink()

    assert is_installed(venv) is False


# ---------------------------------------------------------------------------
# _create_venv
# ---------------------------------------------------------------------------


def test_create_venv_uses_uv_with_no_config(tmp_path, monkeypatch):
    """--no-config is load-bearing: gym's [tool.uv] excludes deps AppWorld needs."""
    commands = []
    monkeypatch.setattr(setup_appworld.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(setup_appworld, "_run", lambda args, cwd, root: commands.append(args))

    setup_appworld._create_venv(tmp_path / "venv")

    assert commands[0][:3] == ["uv", "venv", "--no-config"]
    assert commands[1][:4] == ["uv", "pip", "install", "--no-config"]
    assert commands[1][-1] == APPWORLD_REQUIREMENT


def test_create_venv_falls_back_to_the_stdlib_venv(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(setup_appworld.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup_appworld, "_run", lambda args, cwd, root: commands.append(args))

    setup_appworld._create_venv(tmp_path / "venv")

    assert commands[0] == [sys.executable, "-m", "venv", str(tmp_path / "venv")]
    assert commands[1][-4:] == ["-m", "pip", "install", APPWORLD_REQUIREMENT]


# ---------------------------------------------------------------------------
# ensure_appworld
# ---------------------------------------------------------------------------


def test_ensure_appworld_is_a_noop_when_everything_is_in_place(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    populate_data(root)
    venv = fake_venv(tmp_path)
    monkeypatch.setattr(setup_appworld, "_run", lambda *args, **kwargs: pytest.fail("should not run"))

    install = ensure_appworld(root, venv)

    assert install == AppWorldInstall(root=str(root), executable=str(venv / "bin" / "appworld"))
    assert setup_appworld.os.environ["APPWORLD_ROOT"] == str(root)


def test_ensure_appworld_installs_and_downloads_when_missing(tmp_path, monkeypatch):
    root = tmp_path / "root"
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    commands = []
    installed = {"value": False}

    def _run(args, cwd, root_arg):
        commands.append(args)
        if args[-1] == "install":
            installed["value"] = True
        if args[-1] == "data":
            populate_data(Path(root_arg))

    monkeypatch.setattr(setup_appworld, "_run", _run)
    monkeypatch.setattr(setup_appworld, "_create_venv", lambda path: (path / "bin" / "appworld").touch())
    monkeypatch.setattr(setup_appworld, "is_installed", lambda path: installed["value"])

    install = ensure_appworld(root, venv)

    assert [args[-1] for args in commands] == ["install", "data"]
    assert install.executable == str(venv / "bin" / "appworld")


def test_ensure_appworld_reports_a_failed_install(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_appworld, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup_appworld, "_create_venv", lambda path: None)
    monkeypatch.setattr(setup_appworld, "is_installed", lambda path: False)

    with pytest.raises(RuntimeError, match="cannot import appworld.apps"):
        ensure_appworld(tmp_path / "root", tmp_path / "venv")


def test_ensure_appworld_reports_a_failed_download(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_appworld, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup_appworld, "is_installed", lambda path: True)

    with pytest.raises(RuntimeError, match="data is still empty"):
        ensure_appworld(tmp_path / "root", tmp_path / "venv")


def test_ensure_appworld_honours_the_environment(tmp_path, monkeypatch):
    root = tmp_path / "env-root"
    root.mkdir()
    populate_data(root)
    venv = fake_venv(tmp_path)
    monkeypatch.setenv("APPWORLD_ROOT", str(root))
    monkeypatch.setenv("APPWORLD_VENV", str(venv))

    install = ensure_appworld()

    assert install.root == str(root)
    assert install.executable == str(venv / "bin" / "appworld")


# ---------------------------------------------------------------------------
# prepare_appworld
# ---------------------------------------------------------------------------


def test_make_row_carries_no_appworld_content():
    row = prepare_appworld.make_row("82e2fac_1", "train")

    # Only the id and split: task text is fetched at seed time, never shipped.
    assert row == {
        "task_id": "82e2fac_1",
        "split": "train",
        "responses_create_params": {"input": []},
    }


def test_write_jsonl_creates_parents(tmp_path):
    output = tmp_path / "nested" / "train.jsonl"

    prepare_appworld.write_jsonl([{"a": 1}, {"b": 2}], output)

    assert [json.loads(line) for line in output.read_text().splitlines()] == [{"a": 1}, {"b": 2}]


def test_main_writes_every_split_and_the_example_file(tmp_path, monkeypatch):
    root = tmp_path / "root"
    monkeypatch.setattr(
        prepare_appworld,
        "ensure_appworld",
        lambda *args: AppWorldInstall(root=str(root), executable="appworld"),
    )
    monkeypatch.setattr(
        prepare_appworld,
        "load_task_ids",
        lambda _root, split: [f"{split}_{index}" for index in range(7)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_appworld.py", "--splits", "train", "dev", "--output-dir", str(tmp_path / "data")],
    )

    prepare_appworld.main()

    data_dir = tmp_path / "data"
    assert len((data_dir / "train_appworld.jsonl").read_text().splitlines()) == 7
    assert len((data_dir / "dev_appworld.jsonl").read_text().splitlines()) == 7
    # example.jsonl is cut from the first requested split only.
    example_rows = [json.loads(line) for line in (data_dir / "example.jsonl").read_text().splitlines()]
    assert len(example_rows) == 5
    assert all(row["split"] == "train" for row in example_rows)


def test_main_can_skip_the_example_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare_appworld,
        "ensure_appworld",
        lambda *args: AppWorldInstall(root=str(tmp_path), executable="appworld"),
    )
    monkeypatch.setattr(prepare_appworld, "load_task_ids", lambda _root, split: ["aaa_1"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_appworld.py",
            "--splits",
            "test_normal",
            "--output-dir",
            str(tmp_path / "data"),
            "--example-rows",
            "0",
        ],
    )

    prepare_appworld.main()

    assert not (tmp_path / "data" / "example.jsonl").exists()


def test_every_split_has_a_filename():
    assert set(prepare_appworld.SPLIT_FILENAMES) == set(prepare_appworld.ALL_SPLITS)
