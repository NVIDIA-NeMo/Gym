# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from swebench.harness.log_parsers.python import parse_log_sphinx
from swebench.harness.test_spec.test_spec import TestSpec as SWEBenchTestSpec

from nemo_gym.server_utils import ServerClient
from resources_servers.swebench.app import SwebenchResourcesServer, SwebenchResourcesServerConfig


@pytest.fixture
def original_spec() -> SWEBenchTestSpec:
    return SWEBenchTestSpec(
        instance_id="sphinx-doc__sphinx-9711",
        repo="sphinx-doc/sphinx",
        version="4.2",
        repo_script_list=[],
        eval_script_list=[
            "python -m pip install -e '.[test]'",
            "git apply -v - <<'EOF'\n+tox -- example\nEOF",
            "tox --current-env -epy39 -v -- tests/test_extension.py",
            "git checkout abc123",
        ],
        env_script_list=[],
        arch="x86_64",
        FAIL_TO_PASS=["tests/test_extension.py::test_needs_extensions"],
        PASS_TO_PASS=[],
        language="py",
        docker_specs={},
        namespace="swebench",
    )


@pytest.fixture
def server() -> SwebenchResourcesServer:
    config = SwebenchResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="", sandbox_provider="test", sandbox_config={}
    )
    return SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_sphinx_test_spec_preserves_selection_and_images(
    monkeypatch: pytest.MonkeyPatch, server: SwebenchResourcesServer, original_spec: SWEBenchTestSpec
) -> None:
    monkeypatch.setattr("resources_servers.swebench.app.make_test_spec", lambda *args, **kwargs: original_spec)
    result = server._make_test_spec(MagicMock())
    assert result.eval_script_list[2] == "tox --current-env -epy39 -v -- -rA tests/test_extension.py"
    assert original_spec.eval_script_list[2] == "tox --current-env -epy39 -v -- tests/test_extension.py"
    assert replace(result, eval_script_list=original_spec.eval_script_list) == original_spec
    assert result.instance_image_key == original_spec.instance_image_key
    assert result.env_image_key == original_spec.env_image_key
    assert result.eval_script_list[:2] == original_spec.eval_script_list[:2]
    assert result.eval_script_list[3:] == original_spec.eval_script_list[3:]


def test_other_repositories_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, server: SwebenchResourcesServer, original_spec: SWEBenchTestSpec
) -> None:
    original_spec.repo = "django/django"
    monkeypatch.setattr("resources_servers.swebench.app.make_test_spec", lambda *args, **kwargs: original_spec)
    assert server._make_test_spec(MagicMock()) is original_spec


@pytest.mark.parametrize(("passes", "status", "exit_code"), [(True, "PASSED", 0), (False, "FAILED", 1)])
def test_sphinx_pytest_results_are_parseable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: SwebenchResourcesServer,
    original_spec: SWEBenchTestSpec,
    passes: bool,
    status: str,
    exit_code: int,
) -> None:
    monkeypatch.setattr("resources_servers.swebench.app.make_test_spec", lambda *args, **kwargs: original_spec)
    spec = server._make_test_spec(MagicMock())
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_extension.py").write_text(f"def test_needs_extensions():\n    assert {passes}\n")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}

    def run_pytest(command: str) -> subprocess.CompletedProcess[str]:
        # Tox forwards the arguments after -- to pytest. Exercise real output,
        # without recreating the historical Sphinx Python/tox environment.
        arguments = shlex.split(command.split(" -- ", 1)[1])
        return subprocess.run(
            [sys.executable, "-m", "pytest", "--color=no", *arguments],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    if passes:
        before = run_pytest(original_spec.eval_script_list[2])
        assert before.returncode == 0, before.stdout + before.stderr
        assert "1 passed" in before.stdout
        assert parse_log_sphinx(before.stdout, original_spec) == {}
    result = run_pytest(spec.eval_script_list[2])
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert parse_log_sphinx(result.stdout, spec) == {"tests/test_extension.py::test_needs_extensions": status}
