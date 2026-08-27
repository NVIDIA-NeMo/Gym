# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.apex_agent.app import (
    NG_FAILURE_CLASS_KEY,
    ApexAgent,
    ApexAgentConfig,
    ApexAgentRunRequest,
    instruction_from_input,
    load_runner_source,
)
from responses_api_agents.apex_agent.sandbox_entrypoint import _discover_gateway_url, _patch_code_mcp_cancellation_race


def _body() -> ApexAgentRunRequest:
    return ApexAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Do the work"}]},
            "task_id": "task-1",
            "world_id": "world-1",
            "verifier_metadata": {
                "rubric": [{"criteria": "secret rubric"}],
                "gold_response": "secret gold",
            },
        }
    )


def _agent(
    *,
    image: str = "registry.example/archipelago@sha256:1234",
    auto_build: bool = False,
    supports_vision: bool = True,
) -> ApexAgent:
    config = ApexAgentConfig(
        host="0.0.0.0",
        port=8080,
        name="apex_agent",
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        concurrency=4,
        timeout=3600,
        image=image,
        image_build={
            "enabled": auto_build,
            "source_repo": "https://github.com/Mercor-Intelligence/archipelago.git",
            "source_revision": "0cb5c476c219a9df637e0bd37fb86b2361f4ab89",
            "source_root": None,
            "source_github_token": None,
            "dockerfile": "environment/Dockerfile",
            "docker_tag": "nemo-gym-archipelago:test",
            "timeout": 60,
        },
        sandbox_provider={"apptainer": {}},
        sandbox_spec={},
        edgar_user_agent=None,
        max_turns=200,
        max_output_tokens=32_768,
        supports_vision=supports_vision,
        temperature=1.0,
        top_p=1.0,
        max_snapshot_bytes=None,
        max_world_bytes=None,
        artifact_output_dir=None,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy_model_name": "moonshotai/Kimi-K3"}
    agent = ApexAgent(config=config, server_client=client)
    agent._model_base_url = lambda _body: "http://model/v1"
    return agent


def test_instruction_from_input_only_uses_user_messages() -> None:
    body = _body()
    assert instruction_from_input(body.responses_create_params) == "Do the work"


def test_timeout_failure_is_routed_as_retryable_infra() -> None:
    result = _agent()._failure(
        _body(),
        "sandbox Stirrup rollout exited: direct apptainer command timed out after 12600s",
        return_code=125,
        failure_class="timeout_exceeded",
    )
    payload = result.model_dump()

    assert payload["reward"] == 0.0
    assert payload[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert "_ng_failure_terminal" not in payload
    assert "_ng_no_persist" not in payload


def test_non_timeout_failure_is_not_misclassified_as_timeout() -> None:
    payload = _agent()._failure(_body(), "sandbox Stirrup rollout exited: command failed", return_code=1).model_dump()

    assert NG_FAILURE_CLASS_KEY not in payload


async def test_run_classifies_sandbox_timeout_for_retry(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent()
    agent._ensure_runtime_setup = AsyncMock()
    agent._download_world = AsyncMock()
    agent._stirrup_archive = tmp_path / "stirrup-runtime.tar.gz"
    seed_response = MagicMock(cookies={})
    agent.server_client.post = AsyncMock(return_value=seed_response)
    monkeypatch.setattr("responses_api_agents.apex_agent.app.raise_for_status", AsyncMock())

    class FakeSandbox:
        def __init__(self) -> None:
            self._exec_results = [
                MagicMock(return_code=0),
                MagicMock(return_code=0),
                MagicMock(
                    return_code=125,
                    stderr="direct apptainer command timed out after 12600s",
                    stdout=None,
                    error_type="timeout",
                ),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start(self) -> None:
            return None

        async def upload(self, *_args) -> None:
            return None

        async def exec(self, *_args, **_kwargs):
            return self._exec_results.pop(0)

    monkeypatch.setattr("responses_api_agents.apex_agent.app.AsyncSandbox", MagicMock(return_value=FakeSandbox()))

    result = await agent.run(MagicMock(cookies={}), _body())
    payload = result.model_dump()

    assert payload[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert "_ng_failure_terminal" not in payload
    assert "_ng_no_persist" not in payload


def test_run_request_preserves_task_input_files() -> None:
    payload = _body().model_dump()
    payload["task_input_files"] = "snap_0123456789abcdef0123456789abcdef"

    body = ApexAgentRunRequest.model_validate(payload)

    assert body.task_input_files == "snap_0123456789abcdef0123456789abcdef"


def test_sandbox_config_never_contains_verifier_secrets() -> None:
    body = _body()
    spec = _agent()._sandbox_spec(body, "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])
    serialized = json.dumps(runner)

    assert runner["instruction"] == "Do the work"
    assert runner["policy_model"] == "moonshotai/Kimi-K3"
    assert runner["max_turns"] == 200
    assert runner["max_output_tokens"] == 32_768
    assert runner["supports_vision"] is True
    assert "tokenizer_path" not in runner
    assert "context_window_tokens" not in runner
    assert "max_tool_output_tokens" not in runner
    assert "secret rubric" not in serialized
    assert "secret gold" not in serialized
    assert "CODE_EXEC_RUN_AS_USER" not in spec.env
    assert "/app/apex-gym/stirrup_runtime.py" in spec.files
    assert "FOUNDRY_LOCAL_ROOT" not in spec.env


def test_sandbox_config_propagates_text_only_model_capability() -> None:
    spec = _agent(supports_vision=False)._sandbox_spec(_body(), "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])

    assert runner["supports_vision"] is False


def test_sandbox_runner_uses_archipelago_gateway_and_stirrup() -> None:
    source = load_runner_source()

    assert "_patch_code_mcp_cancellation_race()" in source
    assert "configure_gateway(" in source
    assert '"0"' in source
    assert "_discover_gateway_url(" in source
    assert "run_stirrup_rollout(config, gateway_url)" in source
    assert 'write_snapshot(OUTPUT / "initial.zip")' in source
    assert "overlay_task_files(task_files_zip, scratch)" in source
    assert source.index("overlay_task_files(task_files_zip, scratch)") < source.index(
        'write_snapshot(OUTPUT / "initial.zip")'
    )
    assert "stdout=gateway_log" in source
    assert "stderr=asyncio.subprocess.STDOUT" in source
    assert "stdout=asyncio.subprocess.PIPE" not in source


async def test_gateway_url_uses_uvicorn_dynamic_port(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.write_text("INFO: Uvicorn running on http://127.0.0.1:43127 (Press CTRL+C to quit)\n")
    process = MagicMock(returncode=None)

    assert await _discover_gateway_url(process, log_path) == "http://127.0.0.1:43127"


def test_code_mcp_cancellation_patch_is_idempotent(tmp_path: Path) -> None:
    session_path = tmp_path / "code/.venv/lib/python3.13/site-packages/mcp/shared/session.py"
    session_path.parent.mkdir(parents=True)
    session_path.write_text(
        "async def respond(self, response):\n"
        '        assert not self._completed, "Request already responded to"\n'
        "        await self._send(response)\n",
        encoding="utf-8",
    )

    _patch_code_mcp_cancellation_race(tmp_path)
    patched = session_path.read_text(encoding="utf-8")
    _patch_code_mcp_cancellation_race(tmp_path)

    assert "        if self._completed:\n            return\n" in patched
    assert session_path.read_text(encoding="utf-8") == patched


async def test_runtime_setup_resolves_image_before_stirrup(monkeypatch: MonkeyPatch) -> None:
    agent = _agent()
    events: list[str] = []
    monkeypatch.setattr(
        "responses_api_agents.apex_agent.app.resolve_image",
        MagicMock(side_effect=lambda **_kwargs: events.append("image") or "archipelago.sif"),
    )

    async def _build(_image: str) -> Path:
        events.append("runtime")
        return Path("/tmp/stirrup-runtime.tar.gz")

    agent._build_stirrup_archive = AsyncMock(side_effect=_build)

    async def _inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)

    await agent._ensure_runtime_setup()

    assert events == ["image", "runtime"]


def test_incomplete_rollout_snapshots_are_saved_without_grading(tmp_path: Path) -> None:
    agent = _agent()
    agent.config.artifact_output_dir = str(tmp_path / "saved")
    body = _body()

    output_dir = agent._persist_ungraded_snapshots(
        body,
        {"completion_status": "max_turns"},
        b"initial",
        b"final",
    )

    assert output_dir is not None
    assert (output_dir / "initial_snapshot.zip").read_bytes() == b"initial"
    assert (output_dir / "final_snapshot.zip").read_bytes() == b"final"
    assert json.loads((output_dir / "rollout.json").read_text())["completion_status"] == "max_turns"
