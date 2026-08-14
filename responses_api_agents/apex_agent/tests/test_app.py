# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.apex_agent.app import (
    ApexAgent,
    ApexAgentConfig,
    ApexAgentRunRequest,
    instruction_from_input,
    load_runner_source,
)


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


def _agent(*, image: str = "registry.example/archipelago@sha256:1234", auto_build: bool = False) -> ApexAgent:
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
        harness_repo="https://github.com/Mercor-Intelligence/apex-agent-harness.git",
        harness_root=None,
        harness_github_token=None,
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
        max_turns=100,
        max_output_tokens=4096,
        max_tool_calls_per_turn=3,
        temperature=1.0,
        top_p=1.0,
        max_snapshot_bytes=None,
        max_world_bytes=None,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy_model_name": "moonshotai/Kimi-K3"}
    agent = ApexAgent(config=config, server_client=client)
    agent._model_base_url = lambda _body: "http://model/v1"
    return agent


def test_instruction_from_input_only_uses_user_messages() -> None:
    body = _body()
    assert instruction_from_input(body.responses_create_params) == "Do the work"


def test_sandbox_config_never_contains_verifier_secrets() -> None:
    body = _body()
    spec = _agent()._sandbox_spec(body, "Do the work")
    runner = json.loads(spec.files["/app/apex-gym/runner_config.json"])
    serialized = json.dumps(runner)

    assert runner["instruction"] == "Do the work"
    assert runner["policy_model"] == "moonshotai/Kimi-K3"
    assert "tokenizer_path" not in runner
    assert "context_window_size" not in runner
    assert "max_tool_output_tokens" not in runner
    assert "secret rubric" not in serialized
    assert "secret gold" not in serialized
    assert "CODE_EXEC_RUN_AS_USER" not in spec.env
    assert spec.env["FOUNDRY_LOCAL_ROOT"] == "/app/apex-harness-runtime/.apex"


def test_sandbox_runner_does_not_load_a_tokenizer() -> None:
    source = load_runner_source()

    assert "client.get_tokenizer()" not in source
    assert "agent._tokenizer = None" in source
    assert 'shutil.copy2(snapshot.initial_path, OUTPUT / "initial.zip")' in source


async def test_runtime_setup_checks_harness_before_image(monkeypatch: MonkeyPatch) -> None:
    agent = _agent()
    source_archive = Path("/tmp/apex-harness-source.tar.gz")
    events: list[str] = []
    monkeypatch.setattr(
        "responses_api_agents.apex_agent.app.prepare_harness_source_archive",
        MagicMock(side_effect=lambda **_kwargs: events.append("harness") or source_archive),
    )
    monkeypatch.setattr(
        "responses_api_agents.apex_agent.app.resolve_image",
        MagicMock(side_effect=lambda **_kwargs: events.append("image") or "archipelago.sif"),
    )

    async def _build(_image: str, _source: Path) -> Path:
        events.append("runtime")
        return Path("/tmp/apex-harness-runtime.tar.gz")

    agent._build_harness_archive = AsyncMock(side_effect=_build)

    async def _inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)

    await agent._ensure_runtime_setup()

    assert events == ["harness", "image", "runtime"]


async def test_harness_preflight_failure_stops_startup(monkeypatch: MonkeyPatch) -> None:
    agent = _agent()
    monkeypatch.setattr(
        "responses_api_agents.apex_agent.app.prepare_harness_source_archive",
        MagicMock(side_effect=RuntimeError("could not fetch pinned Apex harness")),
    )

    async def _inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)

    with pytest.raises(RuntimeError, match="could not fetch pinned Apex harness"):
        await agent._preflight_harness_source()

    assert agent._harness_source_archive is None


def test_webserver_registers_harness_startup_preflight() -> None:
    agent = _agent()
    app = agent.setup_webserver()

    assert agent._preflight_harness_source in app.router.on_startup
