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
import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pytest import MonkeyPatch
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from swebench.harness.run_evaluation import make_test_spec

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.sandbox import CPU_CAP_ENV_VARS, SandboxExecResult, SandboxHandle
from resources_servers.swebench.app import (
    DockerContainer,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
    SWEBenchSeedSessionRequest,
    SWEBenchVerifyRequest,
    SWEBenchVerifyResponse,
    patch_extraction_command,
    strip_binary_diffs,
)
from resources_servers.swebench.swebench_patches import (
    EVAL_SCRIPT_COMMAND,
    patch_swebench_multilingual_sandbox,
    run_instance,
)
from resources_servers.swebench.task_data import TaskData


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
        wall_time_s = observation["wall_time_s"]
        assert wall_time_s >= 0
        assert observation == {
            "kind": "sandbox",
            "role": "verifier",
            "provider": "test-provider",
            "sandbox_id": "sandbox-123",
            "outcome": "completed",
            "stage": "verification",
            "wall_time_s": wall_time_s,
            "exit_code": None,
            "cpu_time_s": None,
            "peak_memory_mib": None,
            "resource_usage_source": None,
            "error_type": None,
            "error_message": None,
            "failure_id": None,
            "observed_client_interval_s": None,
            "teardown_confirmed": None,
            "cleanup_errors": [],
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


class TestPatchExtraction:
    def test_command_stages_tracked_changes_and_diffs_the_index_against_the_seed_commit(self) -> None:
        cmd = patch_extraction_command("/testbed", "abc123")
        assert cmd.startswith("cd /testbed && git add -u ")
        assert "add -A" not in cmd  # untracked build output must not ride along
        assert cmd.endswith("cd /testbed && git --no-pager diff --cached abc123")

    def test_command_quotes_odd_paths(self) -> None:
        assert "'/work dir'" in patch_extraction_command("/work dir", "deadbeef")

    def test_strip_binary_diffs_drops_only_binary_sections(self) -> None:
        text = (
            "diff --git a/a.py b/a.py\nindex 1..2 100644\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
            "diff --git a/img.png b/img.png\nindex 3..4 100644\nBinary files a/img.png and b/img.png differ\n"
            "diff --git a/b.py b/b.py\nnew file mode 100644\n--- /dev/null\n+++ b/b.py\n@@ -0,0 +1 @@\n+z\n"
        )
        out = strip_binary_diffs(text)
        assert "img.png" not in out and "a/a.py" in out and "b/b.py" in out
        assert strip_binary_diffs("") == ""


VERIFY_BODY = {
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
}


class TestVerdictMasking:
    """An evaluation that did not complete has no verdict and is masked as a verifier error; a patch the harness
    could not apply is the model's own graded zero."""

    async def _verify(
        self, monkeypatch: MonkeyPatch, run_result: dict, *, sandbox_error: str | None = None, golden: bool = True
    ) -> tuple[SWEBenchVerifyResponse, list]:
        server = SwebenchResourcesServer(
            config=SwebenchResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                sandbox_provider="test",
                sandbox_config=dict(),
                is_verifying_golden_patch=golden,
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        monkeypatch.setattr(
            "resources_servers.swebench.app.SwebenchResourcesServer._create_sandbox",
            AsyncMock(
                return_value=make_sandbox(exec_result=SandboxExecResult(return_code=0, stdout="/testbed\n", stderr=""))
            ),
        )

        async def fake_run_instance(*, client, **kwargs):
            client._sandbox_error_type = sandbox_error
            return run_result

        monkeypatch.setattr("resources_servers.swebench.app.run_instance", fake_run_instance)
        observed = []
        monkeypatch.setattr("resources_servers.swebench.app.observe_failure", observed.append)
        if not golden:
            # The agent's sandbox is gone: extracting the patch raises.
            server._session_id_to_sandbox["sid"] = make_sandbox(exec_error=RuntimeError("sandbox lost"))
        response = await server.verify(
            SimpleNamespace(session={SESSION_ID_KEY: "sid"}), SWEBenchVerifyRequest.model_validate(VERIFY_BODY)
        )
        return response, observed

    async def test_a_completed_evaluation_is_a_verdict_with_the_patch_digest(self, monkeypatch: MonkeyPatch) -> None:
        response, observed = await self._verify(monkeypatch, {"completed": True, "resolved": True})
        assert response.evaluation_completed and response.reward == 1 and not response.mask_sample
        assert response.failure_kind is None and response.failure_reason is None and observed == []
        assert response.model_patch_sha256 == hashlib.sha256(b"my patch").hexdigest()

    async def test_a_patch_that_does_not_apply_is_the_models_graded_zero(self, monkeypatch: MonkeyPatch) -> None:
        response, observed = await self._verify(
            monkeypatch,
            {"completed": False, "resolved": False, "patch_apply_failed": True, "error": "EvaluationError: apply"},
        )
        assert response.reward == 0 and not response.mask_sample and response.failure_kind is None
        assert observed == []

    async def test_an_evaluation_error_is_masked_as_a_verifier_error(self, monkeypatch: MonkeyPatch) -> None:
        response, observed = await self._verify(
            monkeypatch, {"completed": False, "resolved": False, "patch_apply_failed": False, "error": "ValueError: x"}
        )
        assert not response.evaluation_completed and response.reward == 0 and response.mask_sample
        assert response.failure_kind == "verifier_error" and response.failure_reason == "ValueError: x"
        assert response.model_patch_sha256 == hashlib.sha256(b"my patch").hexdigest()
        [event] = observed
        assert event.failure_kind == "verifier_error" and event.failure_reason == "ValueError: x"

    @pytest.mark.parametrize("apply_failed", [False, True])
    async def test_a_sandbox_error_is_masked_even_when_the_patch_did_not_apply(
        self, monkeypatch: MonkeyPatch, apply_failed: bool
    ) -> None:
        # A failed sandbox operation can itself make `git apply` or the tests fail, so neither is the model's result.
        response, observed = await self._verify(
            monkeypatch,
            {"completed": apply_failed is False, "resolved": False, "patch_apply_failed": apply_failed},
            sandbox_error="TimeoutError",
        )
        assert response.mask_sample and response.failure_kind == "verifier_error"
        assert response.failure_reason == "Verification sandbox error: TimeoutError" and len(observed) == 1

    async def test_a_patch_that_could_not_be_extracted_is_masked(self, monkeypatch: MonkeyPatch) -> None:
        # The empty patch then fails to apply, but the agent's patch was never collected: no verdict on it.
        response, observed = await self._verify(
            monkeypatch,
            {"completed": False, "resolved": False, "patch_apply_failed": True, "error": "EvaluationError: apply"},
            golden=False,
        )
        assert response.mask_sample and response.failure_kind == "verifier_error"
        assert response.failure_reason == "Failed to extract the model patch from the agent sandbox"
        assert response.model_patch_sha256 is None and response.model_patch is None and len(observed) == 1


PNG_A = b"\x89PNG\r\n\x1a\n" + b"reference-a"
PNG_B = b"\x89PNG\r\n\x1a\n" + b"reference-b"
CHART_TEST_ASSETS = [
    {
        "path": "test/fixtures/controller.bar/minBarLength/horizontal-neg.png",
        "url": "https://example.com/chartjs/horizontal-neg.png",
        "content_base64": base64.b64encode(PNG_A).decode("ascii"),
    },
    {
        "path": "test/fixtures/controller.bar/minBarLength/vertical-pos.png",
        "url": "https://example.com/chartjs/vertical-pos.png",
        "content_base64": base64.b64encode(PNG_B).decode("ascii"),
    },
]
CHART_F2P = "Chart.controllers.bar auto /base/test/fixtures/controller.bar/minBarLength/horizontal-neg.js "
CHART_ROW = VERIFY_BODY | {
    "repo": "chartjs/Chart.js",
    "instance_id": "chartjs__Chart.js-10157",
    "base_commit": "abc123",
    "version": "3.7",
    "test_patch": (
        "diff --git a/test/specs/controller.bar.tests.js b/test/specs/controller.bar.tests.js\n"
        "--- a/test/specs/controller.bar.tests.js\n+++ b/test/specs/controller.bar.tests.js\n@@ -1 +1 @@\n-a\n+b\n"
    ),
    "FAIL_TO_PASS": json.dumps([CHART_F2P]),
    "PASS_TO_PASS": "[]",
    "image_assets": {
        "problem_statement": ["https://example.com/issue.png"],
        "patch": [],
        "test_patch": CHART_TEST_ASSETS,
    },
}


def make_server(*, golden: bool = True) -> SwebenchResourcesServer:
    return SwebenchResourcesServer(
        config=SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            is_verifying_golden_patch=golden,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


class TestImageAssets:
    """SWE-bench Multimodal's test-patch binary files (Chart.js reference images) reach the evaluation script."""

    def test_the_parsed_request_restores_each_test_asset_where_swebench_downloads_it(self) -> None:
        request = SWEBenchVerifyRequest.model_validate(CHART_ROW)
        assert request.model_dump()["image_assets"]["test_patch"][0]["path"] == CHART_TEST_ASSETS[0]["path"]

        upstream = make_test_spec(CHART_ROW, namespace="swebench").eval_script_list
        script = make_server()._make_test_spec(request).eval_script_list

        assert not [line for line in script if line.startswith("curl ")]
        for index, asset in enumerate(CHART_TEST_ASSETS):
            position = upstream.index(f"curl -o {asset['path']} {asset['url']}")
            assert script[position] == f"cp /tmp/nemo-gym-swebench-test-asset-{index} {asset['path']}"
        # Only the downloads change: the copies run after the test files are reset and before the test patch applies.
        assert [line for line in script if not line.startswith("cp ")] == [
            line for line in upstream if not line.startswith("curl ")
        ]

    def test_a_javascript_row_without_image_assets_builds_the_upstream_script(self) -> None:
        # SWE-bench Multilingual JavaScript rows have no image_assets.
        # SWE-bench's JavaScript harness reads the key whenever it is present,
        # so the request's null default must not reach it.
        row = {key: value for key, value in CHART_ROW.items() if key != "image_assets"}
        request = SWEBenchVerifyRequest.model_validate(row)
        assert request.image_assets is None

        script = make_server()._make_test_spec(request).eval_script_list

        assert script == make_test_spec(row, namespace="swebench").eval_script_list

    def test_image_assets_stored_as_a_json_string_are_read(self) -> None:
        request = SWEBenchVerifyRequest.model_validate(
            CHART_ROW | {"image_assets": json.dumps(CHART_ROW["image_assets"])}
        )
        assert [asset.content for asset in request.test_assets()] == [PNG_A, PNG_B]
        assert request.image_assets.model_dump()["patch"] == []
        assert SWEBenchVerifyRequest.model_validate(VERIFY_BODY).test_assets() == []

    @pytest.mark.parametrize("schema", [SWEBenchVerifyRequest, TaskData])
    def test_a_test_asset_without_its_bytes_is_rejected(self, schema: type) -> None:
        without_bytes = [{"path": "test/fixtures/a.png", "url": "https://example.com/a.png"}]
        with pytest.raises(ValidationError, match="benchmarks/swebench/multimodal/prepare.py"):
            schema.model_validate(CHART_ROW | {"image_assets": {"test_patch": without_bytes}})
        not_base64 = [without_bytes[0] | {"content_base64": "not base64!"}]
        with pytest.raises(ValidationError):
            schema.model_validate(CHART_ROW | {"image_assets": {"test_patch": not_base64}})

    async def test_only_the_verifier_sandbox_receives_the_test_assets(self, monkeypatch: MonkeyPatch) -> None:
        server = make_server()
        create = AsyncMock(
            return_value=make_sandbox(exec_result=SandboxExecResult(return_code=0, stdout="/testbed\n", stderr=""))
        )
        monkeypatch.setattr("resources_servers.swebench.app.SwebenchResourcesServer._create_sandbox", create)
        monkeypatch.setattr(
            "resources_servers.swebench.app.run_instance",
            AsyncMock(return_value={"completed": True, "resolved": True}),
        )
        request = SimpleNamespace(session={SESSION_ID_KEY: "sid"})

        await server.seed_session(request, SWEBenchSeedSessionRequest.model_validate(CHART_ROW))
        await server.verify(request, SWEBenchVerifyRequest.model_validate(CHART_ROW))

        seed_call, verify_call = create.await_args_list
        assert "test_assets" not in seed_call.kwargs
        assert [asset.content for asset in verify_call.kwargs["test_assets"]] == [PNG_A, PNG_B]

    async def test_the_verifier_sandbox_setup_uploads_each_test_asset(self, monkeypatch: MonkeyPatch) -> None:
        sandbox = MagicMock()
        uploaded: dict[str, bytes] = {}

        async def start_with_setup(spec: Any, setup: Any) -> MagicMock:
            await setup(sandbox)
            return sandbox

        async def upload(local_path: Path, remote_path: str) -> None:
            uploaded[remote_path] = Path(local_path).read_bytes()

        sandbox.start_with_setup = AsyncMock(side_effect=start_with_setup)
        sandbox.upload = AsyncMock(side_effect=upload)
        monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_: {})
        monkeypatch.setattr("resources_servers.swebench.app.AsyncSandbox", MagicMock(return_value=sandbox))
        monkeypatch.setattr("resources_servers.swebench.app.patch_swebench_multilingual_sandbox", AsyncMock())
        test_spec = SimpleNamespace(
            instance_image_key="img:key", instance_id="chartjs__Chart.js-10157", repo="chartjs/Chart.js"
        )
        server = make_server()

        await server._create_sandbox(test_spec)
        assert uploaded == {}

        await server._create_sandbox(
            test_spec, test_assets=SWEBenchVerifyRequest.model_validate(CHART_ROW).test_assets()
        )
        assert uploaded == {
            "/tmp/nemo-gym-swebench-test-asset-0": PNG_A,
            "/tmp/nemo-gym-swebench-test-asset-1": PNG_B,
        }


class _FakeEvalSandbox:
    """A verification sandbox in which every command succeeds. The evaluation script prints `tests_printed` on stdout
    and bash's trace, test-output markers included, on stderr; the evaluation command joins the two in order."""

    def __init__(self, tests_printed: str) -> None:
        self.tests_printed = tests_printed

    async def exec(self, command: str, **kwargs: Any) -> SandboxExecResult:
        if command != EVAL_SCRIPT_COMMAND:
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        output = (
            "+ : '>>>>> Start Test Output'\n+ ./node_modules/.bin/karma start\n"
            f"{self.tests_printed}\n+ : '>>>>> End Test Output'\n"
        )
        return SandboxExecResult(stdout=output, stderr="", return_code=0)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        pass

    async def stop(self) -> None:
        pass


class TestFailureOnlyGrading:
    """SWE-bench grades Chart.js, marked and p5.js from failures alone, so their success needs a completed suite."""

    async def _run(self, monkeypatch: MonkeyPatch, tmp_path: Path, tests_printed: str) -> dict:
        monkeypatch.chdir(tmp_path)  # run_instance writes its logs under the working directory
        test_spec = make_server()._make_test_spec(SWEBenchVerifyRequest.model_validate(CHART_ROW))
        container = DockerContainer(id="run-id", instance_id=test_spec.instance_id)
        container._inner_container = _FakeEvalSandbox(tests_printed)
        return await run_instance(
            test_spec=test_spec,
            pred={"instance_id": test_spec.instance_id, "model_patch": "diff --git a/src/a.js b/src/a.js\n"},
            rm_image=False,
            force_rebuild=False,
            client=container,
            run_id="run-id",
        )

    async def test_a_suite_that_never_started_has_no_verdict(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        result = await self._run(
            monkeypatch,
            tmp_path,
            "ERROR [launcher]: Chrome failed 2 times (cannot start). Giving up.\n"
            "Firefox 131.0 (Linux x86_64): Executed 1545 of 1545 SUCCESS (56 secs / 48 secs)",
        )
        assert result["completed"] is False and result["resolved"] is False
        assert "chartjs/Chart.js test suite did not run to completion" in result["error"]

    async def test_a_completed_suite_without_the_failure_is_resolved(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        result = await self._run(
            monkeypatch,
            tmp_path,
            "Chrome 120.0.0.0 (Linux x86_64) Chart.helpers.dom getRelativePosition FAILED\n"
            "Chrome 120.0.0.0 (Linux x86_64): Executed 1545 of 1545 (1 FAILED) (43 secs / 33 secs)",
        )
        assert result["completed"] is True and result["resolved"] is True and result["error"] is None

    async def test_a_failing_required_test_is_unresolved(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        result = await self._run(
            monkeypatch,
            tmp_path,
            f"Chrome 120.0.0.0 (Linux x86_64) {CHART_F2P}FAILED\n"
            "Chrome 120.0.0.0 (Linux x86_64): Executed 1545 of 1545 (1 FAILED) (43 secs / 33 secs)",
        )
        assert result["completed"] is True and result["resolved"] is False


class TestEvalScriptCommand:
    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
    def test_stderr_of_the_tests_stays_between_the_markers_when_only_stdout_is_read(self, tmp_path: Path) -> None:
        """A provider that returns stdout and stderr apart (E2B) still yields one ordered graded region."""
        script = tmp_path / "eval.sh"
        script.write_text(
            f"echo '{START_TEST_OUTPUT}'\n"
            "echo 'PASSED tests/test_a.py::test_one' >&2\n"
            "echo 'FAILED tests/test_a.py::test_two'\n"
            f"echo '{END_TEST_OUTPUT}'\n"
        )
        command = EVAL_SCRIPT_COMMAND.replace("/eval.sh", str(script))

        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True)

        assert result.stderr == ""
        region = result.stdout.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        assert region.splitlines()[1:] == ["PASSED tests/test_a.py::test_one", "FAILED tests/test_a.py::test_two"]
