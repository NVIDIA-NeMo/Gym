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
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxStatus
from nemo_gym.sandbox.manifest import SandboxManifestRecord, append_manifest_record, read_manifest_records
from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.swebench.app import (
    DockerContainer,
    SWEBenchPauseResumeConfig,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
    SWEBenchSeedSessionRequest,
    SWEBenchVerifyRequest,
    SWEBenchVerifyResponse,
)


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


def make_seed_sandbox(sandbox_id: str = "sandbox-123", status: SandboxStatus = SandboxStatus.PAUSED) -> MagicMock:
    sandbox = make_sandbox(exec_result=SandboxExecResult(stdout="/testbed", stderr="", return_code=0))
    sandbox._handle = SandboxHandle(sandbox_id=sandbox_id, provider_name="test-provider", raw=None)
    pty_session = MagicMock()
    pty_session.session_id = "pty-1"
    pty_session.close = AsyncMock()
    sandbox.pty = MagicMock()
    sandbox.pty.create = AsyncMock(return_value=pty_session)
    sandbox.pty.exec = AsyncMock()
    sandbox.status = AsyncMock(return_value=status)
    sandbox.resume = AsyncMock()
    return sandbox


def make_seed_request(session_id: str = "sess-1", **raw_body: Any) -> MagicMock:
    request = MagicMock()
    request.session = {SESSION_ID_KEY: session_id}
    request.json = AsyncMock(return_value=raw_body)
    return request


def instance_payload() -> dict[str, Any]:
    return {
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
        "subset": "my subset",
        "split": "my split",
    }


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
        sandbox.start = AsyncMock()
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
            return sandbox.start.await_args.args[0]

        # Floored to whole cores; explicit sandbox_config.env keys win over the derived caps.
        spec = await created_spec({"resources": {"cpu": 2.7}, "env": {"OMP_NUM_THREADS": "16"}})
        assert spec.env["OMP_NUM_THREADS"] == "16"
        derived = [name for name in CPU_CAP_ENV_VARS if name != "OMP_NUM_THREADS"]
        assert {name: spec.env[name] for name in derived} == {name: "2" for name in derived}

        # Opt-out and no-cpu-limit paths keep env untouched.
        assert (await created_spec({"resources": {"cpu": 2}, "derive_cpu_env": False})).env == {}
        assert (await created_spec({"resources": {"memory_mib": 1024}})).env == {}

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

    def _manifest_config(
        self, tmp_path: Path, resume: bool = False, **overrides: Any
    ) -> SwebenchResourcesServerConfig:
        return SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            pause_resume={"manifest_fpath": str(tmp_path / "manifest.jsonl"), "resume": resume},
            **overrides,
        )

    def test_pause_resume_config_requires_manifest_for_resume(self) -> None:
        with pytest.raises(ValueError, match="requires pause_resume.manifest_fpath"):
            SWEBenchPauseResumeConfig(resume=True)

    async def test_seed_session_records_created_and_verify_records_done(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        config = self._manifest_config(tmp_path, apply_anti_cheating=False)
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        sandbox = make_seed_sandbox()
        monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
        request = make_seed_request(_ng_task_index=3, _ng_rollout_index=1)

        response = await server.seed_session(request, SWEBenchSeedSessionRequest.model_validate(instance_payload()))

        assert response.resumed is False
        assert response.sandbox_handle == "sandbox-123"
        assert response.pty_session_id == "pty-1"
        records = read_manifest_records(tmp_path / "manifest.jsonl")
        assert [(r.rollout_key, r.sandbox_id, r.status, r.rollout_index, r.instance_id) for r in records] == [
            ("3-1", "sandbox-123", "created", 1, "my instance_id")
        ]

        monkeypatch.setattr(
            "resources_servers.swebench.app.run_instance",
            AsyncMock(return_value=dict(resolved=True, completed=True)),
        )
        verify_body = SWEBenchVerifyRequest.model_validate(
            instance_payload()
            | {
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
            }
        )
        verify_response = await server.verify(request, verify_body)

        assert verify_response.resolved is True
        records = read_manifest_records(tmp_path / "manifest.jsonl")
        assert [(r.rollout_key, r.sandbox_id, r.status) for r in records[1:]] == [("3-1", "sandbox-123", "done")]
        assert server._session_id_to_rollout_key == {}

    async def test_seed_session_without_rollout_identity_skips_manifest(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        config = self._manifest_config(tmp_path, apply_anti_cheating=False)
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=make_seed_sandbox()))

        response = await server.seed_session(
            make_seed_request(), SWEBenchSeedSessionRequest.model_validate(instance_payload())
        )

        assert response.resumed is False
        assert read_manifest_records(tmp_path / "manifest.jsonl") == []

    def _paused_manifest(self, tmp_path: Path) -> Path:
        manifest = tmp_path / "manifest.jsonl"
        for status in ("created", "paused"):
            append_manifest_record(
                manifest,
                SandboxManifestRecord(
                    rollout_key="3-1",
                    sandbox_id="sb-paused",
                    status=status,
                    instance_id="my instance_id",
                    rollout_index=1,
                ),
            )
        return manifest

    @pytest.mark.parametrize("already_running", [False, True])
    async def test_seed_session_resumes_paused_sandbox(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, already_running: bool
    ) -> None:
        manifest = self._paused_manifest(tmp_path)
        config = self._manifest_config(tmp_path, resume=True, apply_anti_cheating=True)
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        status = SandboxStatus.RUNNING if already_running else SandboxStatus.PAUSED
        resumed_sandbox = make_seed_sandbox(sandbox_id="sb-paused", status=status)
        monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr("resources_servers.swebench.app.create_provider", lambda *_: MagicMock())
        async_sandbox_cls = MagicMock()
        async_sandbox_cls.connect = AsyncMock(return_value=resumed_sandbox)
        monkeypatch.setattr("resources_servers.swebench.app.AsyncSandbox", async_sandbox_cls)
        create_sandbox = AsyncMock()
        monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
        request = make_seed_request(_ng_task_index=3, _ng_rollout_index=1)

        response = await server.seed_session(request, SWEBenchSeedSessionRequest.model_validate(instance_payload()))

        assert response.resumed is True
        assert response.sandbox_handle == "sb-paused"
        assert async_sandbox_cls.connect.await_args.args[0] == {"sandbox_id": "sb-paused"}
        if already_running:
            resumed_sandbox.resume.assert_not_awaited()
        else:
            resumed_sandbox.resume.assert_awaited_once()
        create_sandbox.assert_not_awaited()
        # Anti-cheating must not run on a resumed sandbox (its git reset --hard
        # would wipe the restored work), but the conda activation still must.
        resumed_sandbox.upload.assert_not_awaited()
        resumed_sandbox.pty.exec.assert_awaited_once()
        assert read_manifest_records(manifest)[-1].status == "resumed"

    async def test_seed_session_falls_back_to_fresh_sandbox_when_resume_fails(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        manifest = self._paused_manifest(tmp_path)
        config = self._manifest_config(tmp_path, resume=True, apply_anti_cheating=True)
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr("resources_servers.swebench.app.create_provider", lambda *_: MagicMock())
        async_sandbox_cls = MagicMock()
        async_sandbox_cls.connect = AsyncMock(side_effect=RuntimeError("sandbox expired"))
        monkeypatch.setattr("resources_servers.swebench.app.AsyncSandbox", async_sandbox_cls)
        fresh_sandbox = make_seed_sandbox()
        monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=fresh_sandbox))
        request = make_seed_request(_ng_task_index=3, _ng_rollout_index=1)

        response = await server.seed_session(request, SWEBenchSeedSessionRequest.model_validate(instance_payload()))

        assert response.resumed is False
        assert response.sandbox_handle == "sandbox-123"
        fresh_sandbox.upload.assert_awaited_once()  # anti-cheating runs on the fresh sandbox
        records = read_manifest_records(manifest)
        assert (records[-2].sandbox_id, records[-2].status) == ("sb-paused", "resume_failed")
        assert (records[-1].sandbox_id, records[-1].status) == ("sandbox-123", "created")
