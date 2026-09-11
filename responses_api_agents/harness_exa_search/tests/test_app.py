# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import Request
from pytest import mark

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.harness_exa_search.app import HarnessExaSearchAgent, HarnessExaSearchConfig


@mark.parametrize("task_index", [0, 1])
async def test_smoke_rollout_runs_harness_through_sandbox_api(monkeypatch, task_index: int) -> None:
    lines = (Path(__file__).parents[3] / "resources_servers/deepsearchqa/data/example.jsonl").read_text().splitlines()
    task = json.loads(lines[task_index])
    monkeypatch.setenv("MOCK_EXA_SEARCH_RESULT", task["answer"])
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy_model": {"responses_api_models": {"model": {}}}}
    client._build_server_base_url.return_value = "http://model"
    config = HarnessExaSearchConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="test",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="verifier"),
        harness_module="responses_api_agents.harness_exa_search.tests.sample_harness",
        harness_class="SampleHarness",
        harness_config_class="SampleHarnessConfig",
        image="unused-by-local-provider",
        python=sys.executable,
        sandbox_provider={"local": {}},
        sandbox_spec={"env": {"MOCK_EXA_SEARCH_RESULT": task["answer"]}},
        exa_api_key="temporary-test-key",
    )
    agent = HarnessExaSearchAgent(config=config, server_client=client)
    result = await agent.responses(
        Request({"type": "http", "path": "/v1/responses", "path_params": {}, "headers": []}),
        NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=task["problem"])], model="model"
        ),
    )
    assert result.output[0].content[0].text == task["answer"]
    assert result.model_dump()["object"] == "response"
    diagnostics = json.loads(result.metadata["agent_run"])
    assert diagnostics["harness_class"] == "SampleHarness"
    assert diagnostics["runner_status"] == "returned"
    assert diagnostics["runner_duration_ms"] >= 0
    assert "temporary-test-key" not in config.model_dump_json()


async def test_smoke_rollout_uses_prebuilt_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MOCK_EXA_SEARCH_RESULT", "Say hello")
    deps = tmp_path / "deps" / "bin"
    deps.mkdir(parents=True)
    python = deps / "python"
    python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    python.chmod(0o755)
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(deps.parent, arcname=".")

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy_model": {"responses_api_models": {"model": {}}}}
    client._build_server_base_url.return_value = "http://model"
    config = HarnessExaSearchConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="test",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="verifier"),
        harness_module="responses_api_agents.harness_exa_search.tests.sample_harness",
        harness_class="SampleHarness",
        harness_config_class="SampleHarnessConfig",
        image="unused-by-local-provider",
        runtime_archive=archive,
        sandbox_provider={"local": {}},
    )
    agent = HarnessExaSearchAgent(config=config, server_client=client)
    result = await agent.responses(
        Request({"type": "http", "path": "/v1/responses", "path_params": {}, "headers": []}),
        NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content="Say hello")], model="model"
        ),
    )
    assert result.output[0].content[0].text == "Say hello"
