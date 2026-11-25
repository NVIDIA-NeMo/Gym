# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire the existing three bridges without altering tasksets, graders or data weight."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from environments.unified_video.routing import configure_harnesses
from responses_api_agents.codex_agent.app import CodexAgentConfig
from responses_api_agents.hermes_agent.app import HermesAgentConfig
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig


def fixture():
    config = {
        "avb_env": {
            "environment_servers": {
                "harbor": {
                    "agent_server": {"type": "responses_api_agents", "name": "avb_agent"},
                    "tasksets": {"avb_train": {"folder": "/tasks", "tasks": {"t": "digest"}}},
                    "sandbox_provider": "avb_backend",
                    "secrets": {"INDEPENDENT_JUDGE_KEY": "${oc.env:JUDGE_KEY}"},
                    "separate_verifier_for_artifacts": True,
                    "default_episode_timeout_seconds": 21600,
                }
            }
        },
        "avb_agent": {
            "responses_api_agents": {
                "opencode_agent": {
                    "entrypoint": "app.py",
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                    "model": "N4",
                    "token_id_capture": True,
                    "timeout": 1800,
                    "context_window": 262144,
                    "max_output_tokens": 8192,
                    "datasets": [{"name": "original"}],
                    "image_input": True,
                }
            }
        },
        "policy_model": {"responses_api_models": {"vllm_model": {"inline_tool_output_images": False}}},
        "avb_backend": {"docker": {"connection": "unchanged"}},
    }
    rows = [
        {
            "task_source": "avb",
            "task_id": {"taskset": "avb_train", "task_id": str(i)},
            "task_input": {
                "task_data": {"_ng_digest": "digest", "family": "assembly"},
                "responses_create_params": {"input": [{"role": "user", "content": "literal"}]},
            },
        }
        for i in range(3)
    ]
    rows += [{"task_source": "video", "responses_create_params": {"input": ["ordered frames"]}}]
    sources = {
        "avb": {"benchmark": "AVB", "dataset": "DSS", "env": "avb_env", "task_type": "agentic"},
        "video": {"benchmark": "VideoMME", "dataset": "native", "env": "video_env", "task_type": "non_agentic"},
    }
    return config, rows, sources


def test_routes_actual_three_agent_implementations_without_reweighting_or_new_grader():
    config, rows, sources = fixture()
    before, input_before = deepcopy(config), deepcopy(rows)
    output, prepared = configure_harnesses(
        config,
        rows,
        sources=sources,
        routing_seed=0,
        context_tokens=262144,
        output_tokens=8192,
        max_steps=128,
        codex_version="0.144.4",
    )
    assert config == before and rows == input_before and len(prepared) == len(rows)
    env = output["avb_env"]["environment_servers"]["unified_video_harbor"]
    original = before["avb_env"]["environment_servers"]["harbor"]
    assert {key: env[key] for key in original} == original
    assert env["harness_agents"] == {
        "opencode": {"type": "responses_api_agents", "name": "avb_agent"},
        "codex": {"type": "responses_api_agents", "name": "avb_agent_codex"},
        "hermes": {"type": "responses_api_agents", "name": "avb_agent_hermes"},
    }
    assert (
        output["avb_agent"]["responses_api_agents"]["opencode_agent"]
        == before["avb_agent"]["responses_api_agents"]["opencode_agent"]
    )
    assert output["avb_backend"] == before["avb_backend"]
    assert output["avb_agent_codex"]["responses_api_agents"]["codex_agent"]["datasets"] == []
    assert output["avb_agent_hermes"]["responses_api_agents"]["hermes_agent"]["datasets"] == []
    identity = {"host": "127.0.0.1", "port": 0, "name": "validated_agent"}
    codex = CodexAgentConfig.model_validate(
        output["avb_agent_codex"]["responses_api_agents"]["codex_agent"] | identity
    )
    hermes = HermesAgentConfig.model_validate(
        output["avb_agent_hermes"]["responses_api_agents"]["hermes_agent"] | identity
    )
    assert codex.model_context_window == hermes.context_length == 262144
    assert codex.codex_version == "0.144.4"
    assert hermes.max_tokens == 8192 and hermes.max_turns == 128
    assert codex.model == hermes.model == "N4"
    assert [row["harness_routing"]["selected_harness"] for row in prepared[:3]] == ["opencode", "codex", "hermes"]
    assert prepared[-1]["harness_routing"]["selected_harness"] is None
    assert output["policy_model"]["responses_api_models"]["vllm_model"]["inline_tool_output_images"] is True
    for original, routed in zip(rows[:3], prepared[:3], strict=True):
        assert routed["task_input"]["task_data"] == original["task_input"]["task_data"]
        assert routed["task_input"]["responses_create_params"] == original["task_input"]["responses_create_params"]


def test_existing_claw_harness_configuration_is_preserved():
    config, rows, sources = fixture()
    config, _ = configure_harnesses(
        config,
        rows,
        sources=sources,
        routing_seed=42,
        context_tokens=262144,
        output_tokens=8192,
        max_steps=128,
        codex_version="0.144.4",
    )
    config["avb_agent_codex"]["responses_api_agents"]["codex_agent"].update(
        timeout=7200, features={"multi_agent": False}, model_auto_compact_token_limit=180000
    )
    config["avb_agent_hermes"]["responses_api_agents"]["hermes_agent"].update(
        enabled_toolsets=["terminal"], terminal_timeout=360, sandbox_runner_timeout_seconds=7200
    )
    expected = deepcopy(config)
    output, _ = configure_harnesses(
        config,
        rows,
        sources=sources,
        routing_seed=42,
        context_tokens=262144,
        output_tokens=8192,
        max_steps=128,
        codex_version="0.144.4",
    )
    assert output["avb_agent_codex"] == expected["avb_agent_codex"]
    assert output["avb_agent_hermes"] == expected["avb_agent_hermes"]


def test_stock_opencode_yaml_uses_effective_defaults_without_changing_original_budgets():
    config, rows, sources = fixture()
    stock = Path(__file__).resolve().parents[3] / "responses_api_agents/opencode_agent/configs/opencode_agent.yaml"
    config["avb_agent"] = yaml.safe_load(stock.read_text())["opencode_agent"]
    raw = config["avb_agent"]["responses_api_agents"]["opencode_agent"]
    assert "max_output_tokens" not in raw
    original = OpenCodeAgentConfig.model_validate(raw | {"host": "127.0.0.1", "port": 0, "name": "avb_agent"})
    before = deepcopy(config)
    output, _ = configure_harnesses(
        config,
        rows,
        sources=sources,
        routing_seed=0,
        context_tokens=original.context_window,
        output_tokens=original.max_output_tokens,
        max_steps=128,
        codex_version="0.144.4",
    )
    assert output["avb_agent"] == before["avb_agent"] and config == before
    assert (
        output["avb_agent_hermes"]["responses_api_agents"]["hermes_agent"]["max_tokens"] == original.max_output_tokens
    )
    assert output["avb_agent_codex"]["responses_api_agents"]["codex_agent"]["timeout"] == original.timeout
    assert output["avb_agent_codex"]["responses_api_agents"]["codex_agent"]["concurrency"] == original.concurrency


@pytest.mark.parametrize(
    ("harness", "field", "value"),
    [
        ("codex", "model_context_window", 131072),
        ("codex", "codex_version", "0.144.3"),
        ("codex", "model", "another-model"),
        ("codex", "model_server", {"type": "responses_api_models", "name": "another_model_server"}),
        ("codex", "token_id_capture", False),
        ("hermes", "context_length", 131072),
        ("hermes", "max_tokens", 4096),
        ("hermes", "max_turns", 32),
        ("hermes", "model", "another-model"),
        ("hermes", "token_id_capture", False),
        ("hermes", "chat_template_kwargs_enabled", False),
    ],
)
def test_existing_harness_cannot_silently_change_requested_model_or_budgets(harness, field, value):
    config, rows, sources = fixture()
    config, _ = configure_harnesses(
        config,
        rows,
        sources=sources,
        routing_seed=0,
        context_tokens=262144,
        output_tokens=8192,
        max_steps=128,
        codex_version="0.144.4",
    )
    config["avb_agent_" + harness]["responses_api_agents"][harness + "_agent"][field] = value
    with pytest.raises(ValueError, match=field):
        configure_harnesses(
            config,
            rows,
            sources=sources,
            routing_seed=0,
            context_tokens=262144,
            output_tokens=8192,
            max_steps=128,
            codex_version="0.144.4",
        )


def test_existing_codex_must_pass_its_actual_compaction_budget_validator():
    config, rows, sources = fixture()
    config, _ = configure_harnesses(
        config,
        rows,
        sources=sources,
        routing_seed=0,
        context_tokens=262144,
        output_tokens=8192,
        max_steps=128,
        codex_version="0.144.4",
    )
    config["avb_agent_codex"]["responses_api_agents"]["codex_agent"]["model_auto_compact_token_limit"] = 262144
    with pytest.raises(ValidationError, match="90%"):
        configure_harnesses(
            config,
            rows,
            sources=sources,
            routing_seed=0,
            context_tokens=262144,
            output_tokens=8192,
            max_steps=128,
            codex_version="0.144.4",
        )


def test_duplicate_harness_dataset_would_change_sampling_weight_and_is_rejected():
    config, rows, sources = fixture()
    config["avb_agent_codex"] = {
        "responses_api_agents": {
            "codex_agent": {
                "model_server": config["avb_agent"]["responses_api_agents"]["opencode_agent"]["model_server"],
                "datasets": [{"duplicate": True}],
            }
        }
    }
    with pytest.raises(ValueError, match="duplicate dataset"):
        configure_harnesses(
            config,
            rows,
            sources=sources,
            routing_seed=0,
            context_tokens=262144,
            output_tokens=8192,
            max_steps=128,
            codex_version="0.144.4",
        )
