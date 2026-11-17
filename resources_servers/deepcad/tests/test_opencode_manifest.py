# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from omegaconf import OmegaConf
from pydantic import ValidationError

from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import ModelCallSampling, NeMoGymResponseCreateParamsNonStreaming, model_sampling_path_segment
from nemo_gym.rollouts import RolloutCollectionHelper
from nemo_gym.server_runtime import SingleAgentTurnTaskInput
from resources_servers.deepcad.dataset.augment_inputs import view_prompt
from resources_servers.deepcad.opencode_app import WorkspaceSeedRequest
from resources_servers.deepcad.operations import add_operation_hint
from resources_servers.deepcad.prepare_opencode import prepare
from resources_servers.deepcad.tests.test_opencode_sessions import seed
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig, _session_opencode_config_error


CONFIG = Path(__file__).parents[1] / "configs/opencode.yaml"


def prepared(tmp_path, row=None, *, file_input=False, harness="opencode"):
    row = deepcopy(row or seed().task_data)
    row["responses_create_params"]["input"].insert(0, {"role": "system", "content": "Try to use box/fillet."})
    row["responses_create_params"]["input"][-1]["content"].insert(
        0, {"type": "input_text", "text": "Dimensions 1, 2, 3. Output ONLY Python."}
    )
    source, output = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    source.write_text(json.dumps(row) + "\n")
    assert prepare(source, output, file_input=file_input, harness=harness) == 1
    return row, json.loads(output.read_text())


def test_preparation_preserves_images_metadata_hints_and_sampling(tmp_path):
    original = deepcopy(seed().task_data)
    original["responses_create_params"]["temperature"] = 0.8
    original["responses_create_params"]["max_output_tokens"] = 32768
    source, result = prepared(tmp_path, original)
    assert result["verifier_metadata"] == source["verifier_metadata"]
    assert (
        result["responses_create_params"]["input"][-1]["content"][-1]
        == source["responses_create_params"]["input"][-1]["content"][-1]
    )
    assert result["_ng_model_sampling"] == {"temperature": 0.8, "max_output_tokens": 32768}
    assert "temperature" not in result["responses_create_params"]
    assert "max_output_tokens" not in result["responses_create_params"]
    assert result["target_images"] == [
        part["image_url"]
        for message in source["responses_create_params"]["input"]
        if message["role"] == "user"
        for part in message["content"]
        if part["type"] == "input_image"
    ]
    system = result["responses_create_params"]["input"][0]["content"]
    assert "Try to use box/fillet." in system
    assert "native execution tools" in system and "native image capabilities" in system
    assert "with bash" not in system and "read tool" not in system
    assert "python /workspace/.cad/cad_render.py /workspace/model.py --output /workspace/previews" in system
    assert "final model.py is the ONLY artifact graded" in system
    assert "Output ONLY" not in json.dumps(result["responses_create_params"])
    assert "Dimensions 1, 2, 3." in json.dumps(result["responses_create_params"])


def test_prepared_row_is_accepted_by_unmodified_stock_opencode(tmp_path):
    _, row = prepared(tmp_path)
    config = yaml.safe_load(CONFIG.read_text())["deepcad_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    config.update(name="deepcad_opencode_agent", port=8080)
    agent = OpenCodeAgent(config=OpenCodeAgentConfig(**config), server_client=MagicMock(spec=ServerClient))
    text, system, media = agent._session_input(
        NeMoGymResponseCreateParamsNonStreaming(**row["responses_create_params"])
    )
    assert "Dimensions" in text and "model.py" in system
    assert len(media) == 1
    assert _session_opencode_config_error(agent.config.opencode_config) is None


@pytest.mark.parametrize("broken", ["metadata", "images"])
def test_incomplete_inputs_fail_preparation(tmp_path, broken):
    row = deepcopy(seed().task_data)
    if broken == "metadata":
        row["verifier_metadata"] = {"source_step_path": "/missing/reference.step"}
    else:
        row["responses_create_params"]["input"][0]["content"] = "No image"
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises((ValueError, ValidationError)):
        prepare(source, output)


def test_routes_to_resources_owned_session_and_stock_agent(tmp_path):
    _, row = prepared(tmp_path)
    config = OmegaConf.load(CONFIG)
    RolloutCollectionHelper.resolve_task_sources([row], config)
    assert row["agent_ref"]["name"] == "deepcad_opencode_agent"
    servers = config.deepcad_opencode_environment_server.environment_servers
    assert list(servers) == ["single_agent_turn_legacy"]
    assert servers.single_agent_turn_legacy.resources_tool_transports == []
    assert list(config.deepcad_opencode.resources_servers) == ["deepcad"]
    assert config.deepcad_opencode.resources_servers.deepcad.entrypoint == "opencode_app.py"
    assert list(config.deepcad_opencode_agent.responses_api_agents) == ["opencode_agent"]
    assert config.deepcad_opencode_agent.responses_api_agents.opencode_agent.context_window == 131072
    assert config.head_server.host == "127.0.0.1"
    for block in (
        config.deepcad_opencode.resources_servers.deepcad,
        config.deepcad_opencode_agent.responses_api_agents.opencode_agent,
        servers.single_agent_turn_legacy,
    ):
        assert block.host == "127.0.0.1"


def test_cannot_overwrite_source_manifest(tmp_path):
    path = tmp_path / "source.jsonl"
    with pytest.raises(ValueError, match="separate"):
        prepare(path, path)


def test_composed_config_resolves_offline_without_a_legacy_agent_relay(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_BUDGET_DIR", str(tmp_path / "budget"))
    root = CONFIG.parents[3]
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                OmegaConf.load(root / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                OmegaConf.load(CONFIG),
                {
                    "default_host": "192.0.2.10",
                    "policy_base_url": "http://192.0.2.10:8000/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "validation-only",
                    "error_on_agent_without_environment_server": True,
                },
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert config.deepcad_opencode.resources_servers.deepcad.host == "127.0.0.1"
    assert config.policy_model.responses_api_models.vllm_model.host == "192.0.2.10"
    assert list(config.deepcad_opencode_environment_server.environment_servers) == ["single_agent_turn_legacy"]
    assert config.policy_model.responses_api_models.vllm_model.return_token_id_information is True
    assert config.policy_model.responses_api_models.vllm_model.supply_prefix_token_ids is True
    SingleAgentTurnLegacyEnvironmentServerConfig(
        name="deepcad_opencode_environment_server",
        **OmegaConf.to_container(
            config.deepcad_opencode_environment_server.environment_servers.single_agent_turn_legacy, resolve=True
        ),
    )


def test_operation_hint_after_single_turn_rules_survives_conversion(tmp_path):
    row = deepcopy(seed().task_data)
    row["responses_create_params"]["input"].insert(0, {"role": "system", "content": view_prompt("diagonal")})
    add_operation_hint(row, ["box", "fillet"], probability=1.0)
    source, output = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    source.write_text(json.dumps(row) + "\n")
    prepare(source, output)
    result = json.loads(output.read_text())
    system = result["responses_create_params"]["input"][0]["content"]
    assert "Try to use box/fillet operations." in system
    assert "do not read files" not in system
    assert "Output ONLY" not in system
    assert result["verifier_metadata"]["operation_hint_applied"] is True


@pytest.mark.parametrize("file_input", [False, True])
def test_resource_seed_keeps_original_images_after_flat_materialization(tmp_path, file_input):
    source, row = prepared(tmp_path, file_input=file_input)
    task_input = SingleAgentTurnTaskInput.model_validate(
        {key: value for key, value in row.items() if key not in {"task_source", "_ng_model_sampling"}}
    )
    resource_seed = WorkspaceSeedRequest.model_validate(task_input.task_data)
    assert resource_seed.target_images == row["target_images"]
    assert "responses_create_params" not in task_input.task_data
    if file_input:
        prompt = task_input.responses_create_params.input[0].content
        assert isinstance(prompt, str)
        assert "/workspace/target-0.png" in prompt
        assert "Try to use box/fillet." in prompt
        assert "data:image" not in prompt
        assert "reference_step" not in prompt


@pytest.mark.parametrize("field,value", [("temperature", 0.8), ("top_p", 0.9), ("max_output_tokens", 32768)])
def test_sampling_marker_preserves_values_and_rejects_conflicts(tmp_path, field, value):
    row = deepcopy(seed().task_data)
    row["responses_create_params"][field] = value
    row["_ng_model_sampling"] = {field: value}
    _, result = prepared(tmp_path, row, file_input=True)
    assert getattr(ModelCallSampling.model_validate(result["_ng_model_sampling"]), field) == value
    assert field not in result["responses_create_params"]
    row["_ng_model_sampling"][field] = 1 if field == "max_output_tokens" else 0.5
    conflict_dir = tmp_path / "conflict"
    conflict_dir.mkdir()
    with pytest.raises(ValueError, match="Conflicting model sampling"):
        prepared(conflict_dir, row, file_input=True)


@pytest.mark.parametrize("harness", ["pi", "hermes", "codex", "claude_code", "openclaw", "nooa"])
def test_non_opencode_requires_file_input(tmp_path, harness):
    with pytest.raises(ValueError, match="require --file-input"):
        prepared(tmp_path, harness=harness)


@pytest.mark.parametrize("existing_marker", [False, True])
def test_empty_sampling_routes_through_native_episode_request(tmp_path, existing_marker):
    source = deepcopy(seed().task_data)
    for field in ("max_output_tokens", "temperature", "top_p"):
        source["responses_create_params"].pop(field, None)
    source.pop("_ng_model_sampling", None)
    if existing_marker:
        source["_ng_model_sampling"] = {}
    _, row = prepared(tmp_path, source)
    assert row["_ng_model_sampling"] == {}
    row.update(_ng_task_index=0, _ng_rollout_index=0)
    values = yaml.safe_load(CONFIG.read_text())["deepcad_opencode_environment_server"]["environment_servers"][
        "single_agent_turn_legacy"
    ]
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.load(CONFIG)
    environment = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnLegacyEnvironmentServerConfig(
            name="deepcad_opencode_environment_server", port=8080, **values
        ),
        server_client=client,
    )
    request = environment._episode_request_from_row(row)
    assert request.model_sampling is not None
    assert request.model_sampling.model_dump(exclude_none=True) == {}
    segment = environment.model_sampling_segment(request, trained=True)
    assert model_sampling_path_segment(segment) == "/ng-sampling/e30"


def test_lazy_preparation_defers_rendering_and_applies_workspace_prompt_after_materialization(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_BUDGET_DIR", str(tmp_path / "budget"))
    from unittest.mock import patch

    from resources_servers.deepcad.dataset import lazy_inputs
    from resources_servers.deepcad.tests import test_lazy_inputs as fixtures

    fixture = fixtures.LazyInputTests()
    fixture.setUp()
    try:
        row = fixture.example
        row["verifier_metadata"] = {"task_id": fixture.record["task_id"], "reference_step": "0000_00000007.step"}
        row["responses_create_params"]["max_output_tokens"] = 32768
        source, output = tmp_path / "lazy.jsonl", tmp_path / "opencode.jsonl"
        source.write_text(json.dumps(row) + "\n")
        with patch.object(lazy_inputs.subprocess, "run", side_effect=AssertionError("manifest conversion rendered")):
            assert prepare(source, output) == 1
        converted = json.loads(output.read_text())
        assert converted["deepcad_lazy"]["config"]["harness"] == "opencode"
        with patch.object(lazy_inputs.subprocess, "run", side_effect=fixture.worker) as render:
            lazy_inputs.materialize_examples([converted])
        assert render.call_count == 1 and "deepcad_lazy" not in converted
        assert converted["task_source"] == "deepcad_opencode"
        assert converted["responses_create_params"]["max_output_tokens"] == 32768
        assert converted["_ng_model_sampling"] == {"max_output_tokens": 32768}
        converted.update(_ng_task_index=0, _ng_rollout_index=0)
        values = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = values
        environment = SingleAgentTurnLegacyEnvironmentServer(
            config=SingleAgentTurnLegacyEnvironmentServerConfig(
                name="deepcad_opencode_environment_server",
                port=8080,
                **values["deepcad_opencode_environment_server"]["environment_servers"]["single_agent_turn_legacy"],
            ),
            server_client=client,
        )
        request = environment._episode_request_from_row(converted)
        assert request.model_sampling.max_output_tokens == 32768
        assert model_sampling_path_segment(environment.model_sampling_segment(request, trained=True)).startswith(
            "/ng-sampling/"
        )
        system = converted["responses_create_params"]["input"][0]["content"]
        assert "model.py" in system and "native image capabilities" in system and "Output ONLY" not in system
    finally:
        fixture.doCleanups()


def test_real_environment_task_conversion_keeps_resource_seed_images(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD_BUDGET_DIR", str(tmp_path / "budget"))
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
    from resources_servers.deepcad.opencode_app import WorkspaceSeedRequest

    _, row = prepared(tmp_path)
    row.update(_ng_task_index=0, _ng_rollout_index=0)
    config = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = config
    env = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnLegacyEnvironmentServerConfig(
            name="deepcad_opencode_environment_server",
            port=8080,
            **config["deepcad_opencode_environment_server"]["environment_servers"]["single_agent_turn_legacy"],
        ),
        server_client=client,
    )
    episode = env._episode_request_from_row(row)
    task_data = episode.task.task_input.task_data
    assert "responses_create_params" not in task_data
    seeded = WorkspaceSeedRequest.model_validate(task_data)
    assert seeded.target_images == row["target_images"]


@pytest.mark.parametrize("harness", ["opencode", "pi", "hermes", "codex", "claude_code", "openclaw", "nooa"])
def test_lazy_file_input_requires_materialization_first(tmp_path, harness, monkeypatch):
    source, output = tmp_path / "lazy.jsonl", tmp_path / "prepared.jsonl"
    source.write_text(json.dumps({"deepcad_lazy": {}, "responses_create_params": {"input": []}}) + "\n")
    original = source.read_bytes()
    render = MagicMock(side_effect=AssertionError("manifest conversion rendered"))
    monkeypatch.setattr("resources_servers.deepcad.dataset.lazy_inputs.subprocess.run", render)
    with pytest.raises(ValueError, match="Materialize lazy CAD rows"):
        prepare(source, output, file_input=True, harness=harness)
    render.assert_not_called()
    assert source.read_bytes() == original
    assert output.read_text() == ""
