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
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from swebench.harness.log_parsers.python import parse_log_sphinx
from swebench.harness.test_spec.test_spec import TestSpec as SWEBenchTestSpec

from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS
from nemo_gym.server_utils import ServerClient
from resources_servers.swebench.app import (
    DockerContainer,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
    SWEBenchVerifyResponse,
)
from resources_servers.swebench.swebench_patches import patch_swebench_multilingual_sandbox


def make_sandbox(
    *,
    exec_result: SandboxExecResult | None = None,
    exec_error: Exception | None = None,
    upload_error: Exception | None = None,
    stop_error: Exception | None = None,
) -> MagicMock:
    sandbox = MagicMock()
    sandbox._handle = SandboxHandle(sandbox_id="sandbox-123", provider_name="test-provider", raw=None)
    sandbox.exec = AsyncMock(return_value=exec_result, side_effect=exec_error)
    sandbox.upload = AsyncMock(side_effect=upload_error)
    sandbox.stop = AsyncMock(side_effect=stop_error)
    return sandbox


class TestApp:
    def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            is_verifying_golden_patch=True,
        )
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()

        client = TestClient(app)

        eval_sandbox = make_sandbox()
        monkeypatch.setattr(
            "resources_servers.swebench.app.SwebenchResourcesServer._create_sandbox",
            AsyncMock(return_value=eval_sandbox),
        )
        monkeypatch.setattr(
            "resources_servers.swebench.app.run_instance",
            AsyncMock(return_value=dict(resolved=True, completed=True)),
        )

        res = client.post(
            "/verify",
            json={
                "repo": "astropy/astropy",
                "instance_id": "my instance_id",
                "base_commit": "my base_commit",
                "patch": "my patch",
                "test_patch": "my test_patch",
                "problem_statement": "my problem_statement",
                "hints_text": "",
                "created_at": "my created_at",
                "version": "4.3",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "environment_setup_commit": "my environment_setup_commit",
                "difficulty": "my difficulty",
                "responses_create_params": {"input": []},
                "response": {
                    "output": [],
                    "id": "",
                    "created_at": 0,
                    "model": "",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
                "subset": "my subset",
                "split": "my split",
            },
        )
        assert res.status_code == 200
        observation = res.json()["verifier_sandbox_observation"]
        assert observation.pop("wall_time_s") >= 0
        assert observation == {
            "kind": "sandbox",
            "role": "verifier",
            "provider": "test-provider",
            "sandbox_id": "sandbox-123",
            "outcome": "completed",
            "exit_code": None,
            "cpu_time_s": None,
            "peak_memory_mib": None,
            "resource_usage_source": None,
            "error_type": None,
        }

    async def test_create_sandbox_derives_cpu_cap_env_from_cpu_limit(self, monkeypatch: MonkeyPatch) -> None:
        sandbox = MagicMock()
        sandbox.start_with_setup = AsyncMock(side_effect=lambda spec, setup: sandbox)
        monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_: {})
        monkeypatch.setattr("resources_servers.swebench.app.AsyncSandbox", MagicMock(return_value=sandbox))
        monkeypatch.setattr("resources_servers.swebench.app.patch_swebench_multilingual_sandbox", AsyncMock())
        test_spec = SimpleNamespace(
            instance_image_key="img:key", instance_id="astropy__astropy-12907", repo="astropy/astropy"
        )

        async def created_spec(sandbox_config: dict[str, Any]) -> Any:
            config = SwebenchResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                sandbox_provider="test",
                sandbox_config=sandbox_config,
                is_verifying_golden_patch=True,
            )
            server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            await server._create_sandbox(test_spec)
            return sandbox.start_with_setup.await_args.args[0]

        # Floored to whole cores; explicit sandbox_config.env keys win over the derived caps.
        spec = await created_spec({"resources": {"cpu": 2.7}, "env": {"OMP_NUM_THREADS": "16"}})
        assert spec.env["OMP_NUM_THREADS"] == "16"
        derived = [name for name in CPU_CAP_ENV_VARS if name != "OMP_NUM_THREADS"]
        assert {name: spec.env[name] for name in derived} == {name: "2" for name in derived}

        # Opt-out and no-cpu-limit paths omit only the derived caps; the
        # SWE-bench Verified Python environment is applied independently.
        for sandbox_config in (
            {"resources": {"cpu": 2}, "derive_cpu_env": False},
            {"resources": {"memory_mib": 1024}},
        ):
            spec = await created_spec(sandbox_config)
            assert not set(CPU_CAP_ENV_VARS) & spec.env.keys()
            assert spec.env["CONDA_DEFAULT_ENV"] == "testbed"

    def test_unobserved_response_omits_optional_field(self) -> None:
        response = SWEBenchVerifyResponse.model_construct(verifier_sandbox_observation=None)

        assert "verifier_sandbox_observation" not in response.model_dump()

    async def test_eval_exit_code_is_observed_without_treating_failed_tests_as_sandbox_failure(self) -> None:
        sandbox = make_sandbox(exec_result=SandboxExecResult(stdout="test output", stderr=None, return_code=7))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        test_output, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=3.5, evaluation_completed=True)

        assert test_output == "test output"
        assert timed_out is False
        assert observation.outcome == "completed"
        assert observation.exit_code == 7
        assert observation.wall_time_s == 3.5

    async def test_timeout_is_observed_without_changing_harness_timeout_behavior(self) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(
                stdout=None,
                stderr="backend failed",
                return_code=125,
                error_type="sandbox",
            )
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.exec_run("git apply patch.diff")
        sandbox.exec.side_effect = TimeoutError("timed out")
        test_output, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=60.0, evaluation_completed=False)

        assert test_output == ""
        assert timed_out is True
        assert observation.outcome == "timeout"
        assert observation.exit_code is None
        assert observation.error_type == "TimeoutError"

    async def test_runtime_error_is_observed_and_still_propagates(self) -> None:
        sandbox = make_sandbox(exec_error=RuntimeError("Sandbox was OOM-killed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        with pytest.raises(RuntimeError, match="OOM-killed"):
            await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)

        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)
        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"
        assert observation.exit_code is None

    @pytest.mark.parametrize(
        ("error_type", "expected_outcome"),
        [("sandbox", "sandbox_error"), ("TimeoutError", "timeout")],
    )
    async def test_provider_error_does_not_report_sentinel_as_process_exit_code(
        self, error_type: str, expected_outcome: str
    ) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(stdout=None, stderr="backend failed", return_code=125, error_type=error_type)
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        _, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)

        assert timed_out is False
        assert observation.outcome == expected_outcome
        assert observation.error_type == error_type
        assert observation.exit_code is None

    async def test_pre_eval_provider_error_is_observed(self) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(stdout=None, stderr="backend failed", return_code=125, error_type="sandbox")
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.exec_run("git apply patch.diff")
        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)

        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "sandbox"
        assert observation.exit_code is None

    async def test_upload_error_is_observed(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(upload_error=RuntimeError("upload failed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        with pytest.raises(RuntimeError, match="upload failed"):
            await container.copy(tmp_path / "patch.diff", Path("/tmp/patch.diff"))

        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)
        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"

    async def test_cleanup_error_is_fail_open_and_observed(self) -> None:
        sandbox = make_sandbox(stop_error=RuntimeError("stop failed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.cleanup()
        observation = container.observation(wall_time_s=2.0, evaluation_completed=True)

        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"


class TestMultilingualMavenSettings:
    @pytest.mark.parametrize("overwrite", [True, False])
    async def test_maven_settings_upload_follows_config(self, overwrite: bool) -> None:
        sandbox = make_sandbox()
        await patch_swebench_multilingual_sandbox(
            "google/gson", "google__gson-1", sandbox, overwrite_maven_settings=overwrite
        )

        destinations = [call.args[1] for call in sandbox.upload.await_args_list]
        assert ("/root/.m2/settings.xml" in destinations) is overwrite
        assert "/root/.gradle/init.d/maven_central_mirror.gradle" in destinations


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
