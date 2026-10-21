# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import stat

import pytest
from omegaconf import DictConfig, OmegaConf

from benchmarks.webarena import prepare as webarena_prepare
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


def _write_source(path, count: int) -> str:
    rows = [
        {
            "id": f"webarena-{index}",
            "ques": f"Task {index}",
            "web_name": ["wikipedia"],
            "web": ["__WIKIPEDIA__"],
            "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "fixture"}},
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_validates_denominator_and_writes_model_neutral_rows(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared.jsonl"
    monkeypatch.setattr(webarena_prepare, "SOURCE_SHA256", _write_source(source, 812))

    assert webarena_prepare.prepare(source, output) == output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 812
    assert rows[0]["web_task"]["runtime_profile"] == "visual_browser"
    assert rows[0]["web_task"]["action_profile"] == "computer_use"
    assert rows[0]["responses_create_params"]["input"] == []
    assert "tools" not in rows[0]["responses_create_params"]


def test_prepare_rejects_a_different_task_population(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    monkeypatch.setattr(webarena_prepare, "SOURCE_SHA256", _write_source(source, 1))

    with pytest.raises(ValueError, match="exactly 812 tasks"):
        webarena_prepare.prepare(source, tmp_path / "prepared.jsonl")


def test_nano_omni_profile_composes_without_output_repair() -> None:
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                DictConfig(
                    {
                        "config_paths": [str(webarena_prepare.BENCHMARK_DIR / "configs/nano_omni.yaml")],
                        "policy_base_url": "http://127.0.0.1:8000/v1",
                    }
                ),
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    agent = resolved.webarena_benchmark_agent.responses_api_agents.web_agent
    assert agent.resources_server.name == "webarena_environment"
    assert agent.policy_protocol == "nano_omni_toolcall"
    assert "nano_omni_action_recovery" not in agent
    assert "nano_omni_tool_alias_recovery" not in agent
    assert agent.max_parse_retries == 2
    assert agent.nano_omni_max_tool_calls is None
    assert agent.nano_omni_retry_invalid_tool_calls is False
    # Reference parse retries repeat the same request after one second. They
    # do not inject correction messages or raise the sampling temperature.
    assert agent.nano_omni_parse_retry_feedback is False
    assert agent.nano_omni_parse_retry_temperature is None
    assert agent.nano_omni_parse_retry_delay_secs == 1.0
    assert agent.datasets[0].jsonl_fpath == "benchmarks/webarena/data/webarena.jsonl"
    model = resolved.policy_model.responses_api_models.vllm_model
    assert model.base_url == "http://127.0.0.1:8000/v1"
    assert model.chat_template_kwargs == {"truncate_history_thinking": False}
    resource = resolved.webarena_environment.resources_servers.webarena_browser
    assert resource.terminate_on_action_error is True
    assert resource.max_tool_calls is None
    assert resource.max_computer_actions == agent.nano_omni_max_computer_actions


def test_write_env_is_private_and_rejects_display_sharing(tmp_path) -> None:
    env_path = tmp_path / "env.yaml"
    assert webarena_prepare.write_env(
        env_path,
        input_jsonl=tmp_path / "input.jsonl",
        output_jsonl=tmp_path / "output.jsonl",
    )
    content = env_path.read_text()
    assert "benchmarks/webarena/configs/nano_omni.yaml" in content
    assert "agent_name: webarena_benchmark_agent" in content
    # The reference omits max_tokens from actual requests. Its constructor's
    # 16384 default is not an instruction to impose an output cap here.
    generated = OmegaConf.create(content)
    assert generated.responses_create_params.max_output_tokens is None
    overridden = OmegaConf.merge(generated, {"responses_create_params": {"max_output_tokens": 1024}})
    assert overridden.responses_create_params.max_output_tokens == 1024
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    with pytest.raises(ValueError, match="one DISPLAY"):
        webarena_prepare.write_env(
            tmp_path / "other.yaml",
            input_jsonl=tmp_path / "input.jsonl",
            output_jsonl=tmp_path / "output.jsonl",
            concurrency=2,
        )
