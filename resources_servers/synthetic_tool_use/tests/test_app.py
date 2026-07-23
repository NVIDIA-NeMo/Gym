# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from itertools import count
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymChatCompletion
from resources_servers.synthetic_tool_use.app import (
    PipelineGenerationRequest,
    SyntheticToolUsePipelineConfig,
    SyntheticToolUsePipelineServer,
)
from resources_servers.synthetic_tool_use.common.clients import GymModelGenerator
from resources_servers.synthetic_tool_use.common.models import (
    DomainStageConfig,
    ModelRoleConfig,
    PolicyToolsStageConfig,
    ScenarioStageConfig,
    SeedGenerationConfig,
    StageGenerationResponse,
)
from resources_servers.synthetic_tool_use_domain_generation.app import (
    DomainGenerationResourcesServer,
    DomainGenerationResourcesServerConfig,
)
from resources_servers.synthetic_tool_use_policy_tool_generation.app import (
    PolicyToolGenerationResourcesServer,
    PolicyToolGenerationResourcesServerConfig,
)
from resources_servers.synthetic_tool_use_scenario_generation.app import (
    ScenarioGenerationResourcesServer,
    ScenarioGenerationResourcesServerConfig,
)


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def generation_config(tmp_path: Path) -> SeedGenerationConfig:
    return SeedGenerationConfig(
        run_name="app-test",
        output_dir=tmp_path / "run",
        generation_profile="general",
        source_name="app_test",
        domain_model=ModelRoleConfig(model="domain-model", sampling={"temperature": 0.2}),
        policy_tools_model=ModelRoleConfig(model="policy-model"),
        judge_model=ModelRoleConfig(model="judge-model"),
        scenario_model=ModelRoleConfig(model="scenario-model"),
        domains=DomainStageConfig(request_count=1, semantic_attempts=1),
        policy_tools=PolicyToolsStageConfig(judge_votes=1, golden_comparison_enabled=False),
        scenarios=ScenarioStageConfig(request_count_per_domain=1, scenarios_per_request=1),
    )


def test_generation_output_directory_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="output_dir must be absolute"):
        SeedGenerationConfig(
            run_name="relative",
            output_dir=Path("relative-run"),
            generation_profile="general",
            source_name="relative",
            domain_model=ModelRoleConfig(model="domain"),
            policy_tools_model=ModelRoleConfig(model="policy"),
            scenario_model=ModelRoleConfig(model="scenario"),
        )


class FakeHttpResponse:
    ok = True

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class FakeServerClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def post(self, **kwargs: Any) -> FakeHttpResponse:
        self.calls.append(kwargs)
        return FakeHttpResponse(self.responses.pop(0))


def chat_completion(text: str) -> dict[str, Any]:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {"content": text, "refusal": None, "role": "assistant"},
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_gym_model_generator_uses_chat_completions() -> None:
    client = FakeServerClient([chat_completion("generated text")])
    generator = GymModelGenerator(
        server_client=client,  # type: ignore[arg-type]
        model_server=ModelServerRef(type="responses_api_models", name="generation_model"),
        role=ModelRoleConfig(model="configured-model", sampling={"temperature": 0.2}),
    )

    result = await generator.generate([{"role": "user", "content": "prompt bytes"}])

    assert result.text == "generated text"
    assert result.provider_attempts == 1
    assert client.calls[0]["server_name"] == "generation_model"
    assert client.calls[0]["url_path"] == "/v1/chat/completions"
    request = client.calls[0]["json"]
    assert request.messages == [{"role": "user", "content": "prompt bytes"}]
    assert request.temperature == 0.2
    assert request.model_dump(exclude_unset=True) == {
        "messages": [{"role": "user", "content": "prompt bytes"}],
        "temperature": 0.2,
    }


def test_generation_servers_expose_only_generation_routes(tmp_path: Path) -> None:
    generation = generation_config(tmp_path)
    model_ref = ModelServerRef(type="responses_api_models", name="model")
    judge_ref = ModelServerRef(type="responses_api_models", name="judge")
    common = {"host": "localhost", "port": 8000, "entrypoint": "app.py", "domain": "agent", "name": "test"}
    servers = [
        DomainGenerationResourcesServer.model_construct(
            config=DomainGenerationResourcesServerConfig(**common, generation=generation, model_server=model_ref),
            server_client=None,
        ),
        PolicyToolGenerationResourcesServer.model_construct(
            config=PolicyToolGenerationResourcesServerConfig(
                **common,
                generation=generation,
                model_server=model_ref,
                judge_model_server=judge_ref,
            ),
            server_client=None,
        ),
        ScenarioGenerationResourcesServer.model_construct(
            config=ScenarioGenerationResourcesServerConfig(**common, generation=generation, model_server=model_ref),
            server_client=None,
        ),
    ]

    for server in servers:
        routes = {route.path for route in server.setup_webserver().routes}
        assert "/generate" in routes
        assert "/verify" not in routes
        assert "/seed_session" not in routes


@pytest.mark.asyncio
async def test_pipeline_dispatches_stages_in_dependency_order(tmp_path: Path) -> None:
    responses = [
        StageGenerationResponse(report={"stage": stage}).model_dump()
        for stage in ("domains", "policy_tools", "scenarios")
    ]
    client = FakeServerClient(responses)
    config = SyntheticToolUsePipelineConfig(
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="agent",
        name="synthetic_tool_use_pipeline",
        generation=generation_config(tmp_path),
        domain_generation_server=ResourcesServerRef(
            type="resources_servers", name="synthetic_tool_use_domain_generation"
        ),
        policy_tool_generation_server=ResourcesServerRef(
            type="resources_servers", name="synthetic_tool_use_policy_tool_generation"
        ),
        scenario_generation_server=ResourcesServerRef(
            type="resources_servers", name="synthetic_tool_use_scenario_generation"
        ),
    )
    server = SyntheticToolUsePipelineServer.model_construct(config=config, server_client=client)

    response = await server.generate(
        PipelineGenerationRequest(stages=["scenarios", "domains", "policy_tools"], resume=True)
    )

    assert [call["server_name"] for call in client.calls] == [
        "synthetic_tool_use_domain_generation",
        "synthetic_tool_use_policy_tool_generation",
        "synthetic_tool_use_scenario_generation",
    ]
    assert response.report["domains"] == 0
    routes = {route.path for route in server.setup_webserver().routes}
    assert {"/generate", "/validate", "/materialize"} <= routes
    assert "/verify" not in routes


@pytest.mark.parametrize("profile", ["general", "proactive"])
def test_pipeline_configs_resolve_all_server_apps(profile: str) -> None:
    raw = OmegaConf.load(CONFIGS_DIR / f"{profile}.yaml")
    common = {"host": "localhost", "port": 8000, "name": "test"}
    configs = [
        (
            "synthetic_tool_use_pipeline",
            "synthetic_tool_use",
            SyntheticToolUsePipelineConfig,
        ),
        (
            "synthetic_tool_use_domain_generation",
            "synthetic_tool_use_domain_generation",
            DomainGenerationResourcesServerConfig,
        ),
        (
            "synthetic_tool_use_policy_tool_generation",
            "synthetic_tool_use_policy_tool_generation",
            PolicyToolGenerationResourcesServerConfig,
        ),
        (
            "synthetic_tool_use_scenario_generation",
            "synthetic_tool_use_scenario_generation",
            ScenarioGenerationResourcesServerConfig,
        ),
    ]
    for instance_name, component_name, config_type in configs:
        inner = OmegaConf.to_container(raw[instance_name]["resources_servers"][component_name], resolve=True)
        parsed = config_type.model_validate(inner | common)
        assert parsed.entrypoint == "app.py"
        assert parsed.generation.generation_profile == profile


@pytest.mark.parametrize("profile", ["general", "proactive"])
def test_gym_discovers_complete_server_graph(profile: str) -> None:
    ports = count(15000)
    parser = GlobalConfigDictParser()
    parser_config = GlobalConfigDictParserConfig(
        skip_load_from_cli=True,
        skip_load_from_dotenv=True,
        initial_global_config_dict=OmegaConf.create({"config_paths": [str(CONFIGS_DIR / f"{profile}.yaml")]}),
    )
    with patch(
        "nemo_gym.global_config._find_open_port_using_range",
        side_effect=lambda **_: next(ports),
    ):
        global_config = parser.parse(parser_config)

    assert [server.name for server in parser.filter_for_server_instance_configs(global_config)] == [
        "synthetic_tool_use_pipeline",
        "synthetic_tool_use_domain_generation",
        "synthetic_tool_use_policy_tool_generation",
        "synthetic_tool_use_scenario_generation",
        "synthetic_tool_use_domain_model",
        "synthetic_tool_use_policy_tools_model",
        "synthetic_tool_use_judge_model",
        "synthetic_tool_use_scenario_model",
    ]
