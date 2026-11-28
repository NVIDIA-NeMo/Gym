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
"""`scripts/harbor_task_format_snapshot.py` records a Harbor checkout's task format and reports a change."""

import importlib.util
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "harbor_task_format_snapshot.py"
_spec = importlib.util.spec_from_file_location("harbor_task_format_snapshot", SCRIPT)
snapshot_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snapshot_script)

TASK_CONFIG = """
import toml  # as Harbor's module does, to write task.toml files; the script must not need it installed
from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    docker_image: str | None = Field(default=None, description="The image.")
    description: str = ""
    {extra_field}


class TaskConfig(BaseModel):
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
"""


def write_checkout(
    root: Path, *, extra_field: str = "", sanitizer_body: str = "return name.lower()", image_prefix: str = "hb__"
) -> Path:
    """A folder with the files the script reads from a Harbor checkout, at a commit of its own."""
    files = {
        "pyproject.toml": '[project]\nversion = "9.9.9"\n',
        "src/harbor/__init__.py": "raise RuntimeError('the package itself is never imported')\n",
        "src/harbor/models/__init__.py": "",
        "src/harbor/models/task/__init__.py": "",
        "src/harbor/models/task/config.py": TASK_CONFIG.format(extra_field=extra_field),
        "src/harbor/environments/definition.py": (
            "_CONTENT_HASH_IGNORE_NAMES = frozenset({'.git'})\n\n\n"
            "def environment_content_hash(environment_dir):\n    return 'hash'\n"
        ),
        "src/harbor/environments/docker/docker.py": (
            f"def _sanitize_docker_image_name(name):\n    {sanitizer_body}\n\n\n"
            "def main_image_name(self):\n"
            f'    return _sanitize_docker_image_name(f"{image_prefix}{{self.environment_id}}")\n'
        ),
        "examples/tasks/plain/task.toml": '[environment]\ndocker_image = "ubuntu:24.04"\n',
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "checkout"], check=True)
    return root


@pytest.fixture(autouse=True)
def forget_harbor_modules():
    """The script imports the checkout's task model as ``harbor``; each test reads its own checkout."""
    yield
    for name in [name for name in sys.modules if name == "harbor" or name.startswith("harbor.")]:
        del sys.modules[name]


@pytest.fixture
def snapshot_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "harbor_task_format.json"
    monkeypatch.setattr(snapshot_script, "SNAPSHOT_PATH", path)
    monkeypatch.setattr(snapshot_script, "REPOSITORY_ROOT", tmp_path)
    return path


def test_definition_hash_ignores_layout_and_a_functions_name():
    upstream = "def _sanitize(name):\n    # lower-case it\n    return name.lower()\n"
    copy = "import re\n\n\ndef sanitize(name):\n    return name.lower()\n"
    assert snapshot_script.definition_hash(upstream, "_sanitize") == snapshot_script.definition_hash(copy, "sanitize")
    changed = copy.replace("lower", "upper")
    assert snapshot_script.definition_hash(changed, "sanitize") != snapshot_script.definition_hash(copy, "sanitize")
    assert snapshot_script.definition_hash("A = 1\nB = 2\n", "B") != snapshot_script.definition_hash("B = 3\n", "B")
    # An annotated assignment is found too, and is a change of the definition.
    assert snapshot_script.definition_hash("B: int = 2\n", "B") != snapshot_script.definition_hash("B = 2\n", "B")
    with pytest.raises(LookupError, match="No top-level definition of missing"):
        snapshot_script.definition_hash(copy, "missing")


def test_snapshot_records_the_revision_and_the_schema_without_prose(tmp_path, snapshot_path, capsys):
    checkout = write_checkout(tmp_path / "harbor")

    assert snapshot_script.main([str(checkout)]) == 0

    snapshot = json.loads(snapshot_path.read_text())
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert (snapshot["version"], snapshot["commit"]) == ("9.9.9", commit)
    # No release tag in this checkout: the revision falls back to the abbreviated commit.
    assert commit.startswith(snapshot["revision"])
    environment = snapshot["task_config_schema"]["$defs"]["EnvironmentConfig"]
    # The field named `description` stays; the description of `docker_image` and every title go.
    assert sorted(environment["properties"]) == ["description", "docker_image"]
    assert environment["properties"]["docker_image"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "default": None,
    }
    assert "title" not in environment
    assert sorted(snapshot["vendored_definitions"]) == sorted(snapshot_script.VENDORED_DEFINITIONS)
    assert snapshot["image_name_expression"] == snapshot_script.IMAGE_NAME_EXPRESSION
    # Reading the checkout leaves nothing in it, and the stand-in for `toml` does not outlive the read.
    assert not list(checkout.rglob("__pycache__"))
    assert ("toml" in sys.modules) == (importlib.util.find_spec("toml") is not None)
    assert "Harbor 9.9.9" in capsys.readouterr().out


def test_check_passes_on_the_recorded_format_and_at_a_later_commit(tmp_path, snapshot_path, capsys):
    snapshot_script.main([str(write_checkout(tmp_path / "harbor"))])
    recorded = snapshot_path.read_text()
    later = write_checkout(tmp_path / "later")
    (later / "README.md").write_text("unrelated\n")
    subprocess.run(["git", "-C", str(later), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(later), "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "more"],
        check=True,
    )

    assert snapshot_script.main([str(later), "--check"]) == 0
    assert "matches the committed snapshot" in capsys.readouterr().out
    assert snapshot_path.read_text() == recorded


def test_check_fails_and_names_a_field_harbor_added(tmp_path, snapshot_path, capsys):
    snapshot_script.main([str(write_checkout(tmp_path / "harbor"))])
    recorded = snapshot_path.read_text()
    changed = write_checkout(tmp_path / "changed", extra_field="tpu: str | None = None")

    assert snapshot_script.main([str(changed), "--check"]) == 1

    out = capsys.readouterr().out
    assert "Harbor's task format changed since 9.9.9" in out
    assert '+          "tpu": {' in out
    assert snapshot_path.read_text() == recorded


def test_check_fails_when_a_vendored_definition_changed(tmp_path, snapshot_path, capsys):
    snapshot_script.main([str(write_checkout(tmp_path / "harbor"))])
    changed = write_checkout(tmp_path / "changed", sanitizer_body="return name.upper()")

    assert snapshot_script.main([str(changed), "--check"]) == 1
    out = capsys.readouterr().out
    assert '"sanitize_docker_image_name"' in out and '"task_config_schema"' not in out


def test_check_fails_when_harbor_names_the_task_image_another_way(tmp_path, snapshot_path, capsys):
    snapshot_script.main([str(write_checkout(tmp_path / "harbor"))])
    changed = write_checkout(tmp_path / "changed", image_prefix="harbor__")

    assert snapshot_script.main([str(changed), "--check"]) == 1
    out = capsys.readouterr().out
    assert '-  "image_name_expression": "_sanitize_docker_image_name' in out
    assert '+  "image_name_expression": null' in out


def test_check_fails_when_the_mirror_does_not_read_an_example_task(tmp_path, snapshot_path, capsys):
    checkout = write_checkout(tmp_path / "harbor")
    snapshot_script.main([str(checkout)])
    (checkout / "examples" / "tasks" / "next").mkdir()
    (checkout / "examples" / "tasks" / "next" / "task.toml").write_text(
        textwrap.dedent("""
            [environment]
            cpus = "many"
            teleport = true
        """)
    )

    assert snapshot_script.main([str(checkout), "--check"]) == 1

    out = capsys.readouterr().out
    assert "examples/tasks/next/task.toml: unknown key `environment.teleport`" in out
    assert "examples/tasks/next/task.toml: " in out and "environment.cpus" in out
    assert "examples/tasks/plain" not in out

    # A file that is not TOML is a named problem too, not a traceback.
    (checkout / "examples" / "tasks" / "next" / "task.toml").write_text("[environment\n")
    assert snapshot_script.main([str(checkout), "--check"]) == 1
    assert (
        "Gym's mirror does not read Harbor's example tasks: examples/tasks/next/task.toml: " in capsys.readouterr().out
    )


def test_a_folder_that_is_not_a_harbor_checkout_is_refused(tmp_path, snapshot_path):
    with pytest.raises(SystemExit, match="is not a Harbor checkout"):
        snapshot_script.task_config_schema(tmp_path)


def test_check_fails_when_harbor_has_no_example_tasks_where_they_are_expected(tmp_path, snapshot_path, capsys):
    checkout = write_checkout(tmp_path / "harbor")
    snapshot_script.main([str(checkout)])
    (checkout / "examples").rename(checkout / "samples")

    assert snapshot_script.main([str(checkout), "--check"]) == 1
    assert "no task.toml under examples/tasks" in capsys.readouterr().out


def test_a_checkout_without_git_history_or_a_vendored_definition_is_refused_by_name(tmp_path, snapshot_path):
    checkout = write_checkout(tmp_path / "harbor")
    (checkout / "src" / "harbor" / "environments" / "docker" / "docker.py").write_text("def renamed(name): ...\n")
    with pytest.raises(SystemExit, match="no longer defines `_sanitize_docker_image_name` in src/harbor/environments"):
        snapshot_script.read_snapshot(checkout)

    export = tmp_path / "export"
    shutil.copytree(write_checkout(tmp_path / "other"), export, ignore=shutil.ignore_patterns(".git"))
    with pytest.raises(SystemExit, match="must be a git checkout of Harbor"):
        snapshot_script.read_snapshot(export)
