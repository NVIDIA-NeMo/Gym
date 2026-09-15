# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from benchmarks.visualwebarena import prepare as visualwebarena_prepare
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.web.task_images import resolve_local_task_image_path


def _write_source(root: Path, count: int) -> tuple[Path, str]:
    image_path = root / "visualwebarena" / "shopping" / "task_0" / "input_0.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fixture")
    rows = [
        {
            "id": f"visualwebarena-{index}",
            "ques": f"Task {index}",
            "web_name": ["shopping"],
            "web": ["__SHOPPING__"],
            "image": ["visualwebarena/shopping/task_0/input_0.png"] if index == 0 else [],
            "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "fixture"}},
        }
        for index in range(count)
    ]
    source = root / "visualwebarena.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_prepare_validates_images_and_writes_model_neutral_rows(tmp_path, monkeypatch) -> None:
    source, digest = _write_source(tmp_path, 908)
    output = tmp_path / "prepared.jsonl"
    monkeypatch.setattr(visualwebarena_prepare, "SOURCE_SHA256", digest)

    assert visualwebarena_prepare.prepare(source, output, tmp_path) == output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 908
    assert rows[0]["web_task"]["runtime_profile"] == "visual_browser"
    assert rows[0]["web_task"]["action_profile"] == "computer_use"
    assert rows[0]["web_task"]["input_images"] == ["visualwebarena/shopping/task_0/input_0.png"]
    assert rows[0]["responses_create_params"]["input"] == []
    assert "tools" not in rows[0]["responses_create_params"]


def test_prepare_rejects_a_different_task_population(tmp_path, monkeypatch) -> None:
    source, digest = _write_source(tmp_path, 1)
    monkeypatch.setattr(visualwebarena_prepare, "SOURCE_SHA256", digest)

    with pytest.raises(ValueError, match="exactly 908 tasks"):
        visualwebarena_prepare.prepare(source, tmp_path / "prepared.jsonl", tmp_path)


def test_prepare_rejects_missing_reference_images(tmp_path, monkeypatch) -> None:
    source, digest = _write_source(tmp_path, 908)
    monkeypatch.setattr(visualwebarena_prepare, "SOURCE_SHA256", digest)
    (tmp_path / "visualwebarena" / "shopping" / "task_0" / "input_0.png").unlink()

    with pytest.raises(FileNotFoundError, match="missing 1 referenced image"):
        visualwebarena_prepare.prepare(source, tmp_path / "prepared.jsonl", tmp_path)


def test_prepare_honors_explicit_image_root_for_an_external_jsonl(tmp_path, monkeypatch) -> None:
    image_root = tmp_path / "images"
    source, digest = _write_source(image_root, 908)
    external_source = tmp_path / "source.jsonl"
    source.replace(external_source)
    monkeypatch.delenv("VISUALWEBARENA_SOURCE_ROOT", raising=False)
    monkeypatch.setattr(visualwebarena_prepare, "SOURCE_SHA256", digest)

    assert (
        visualwebarena_prepare.prepare(external_source, tmp_path / "prepared.jsonl", image_root)
        == tmp_path / "prepared.jsonl"
    )


def test_cli_env_uses_the_same_image_root_as_explicit_source(tmp_path, monkeypatch) -> None:
    source, digest = _write_source(tmp_path / "source", 908)
    env_path = tmp_path / "env.yaml"
    monkeypatch.delenv("VISUALWEBARENA_SOURCE_ROOT", raising=False)
    monkeypatch.setattr(visualwebarena_prepare, "SOURCE_SHA256", digest)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare.py",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "prepared.jsonl"),
            "--rollout-output",
            str(tmp_path / "rollouts.jsonl"),
            "--env-file",
            str(env_path),
        ],
    )

    visualwebarena_prepare.main()

    assert OmegaConf.load(env_path).visualwebarena_source_root == str(source.parent)


def test_nano_omni_profile_composes_without_output_repair(tmp_path) -> None:
    source_root = str(tmp_path / "source")
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                DictConfig(
                    {
                        "config_paths": [str(visualwebarena_prepare.BENCHMARK_DIR / "configs/nano_omni.yaml")],
                        "policy_base_url": "http://127.0.0.1:8000/v1",
                        "visualwebarena_source_root": source_root,
                    }
                ),
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    agent = resolved.visualwebarena_benchmark_agent.responses_api_agents.web_agent
    assert agent.resources_server.name == "visualwebarena_environment"
    assert agent.policy_protocol == "nano_omni_toolcall"
    assert "nano_omni_action_recovery" not in agent
    assert "nano_omni_tool_alias_recovery" not in agent
    assert agent.max_parse_retries == 2
    assert agent.nano_omni_max_tool_calls is None
    assert agent.nano_omni_retry_invalid_tool_calls is False
    assert agent.nano_omni_parse_retry_feedback is False
    assert agent.nano_omni_parse_retry_temperature is None
    assert agent.nano_omni_parse_retry_delay_secs == 1.0
    assert agent.max_steps == 100
    assert agent.max_image_history == 3
    assert agent.max_task_image_bytes == 32 * 1024 * 1024
    assert agent.datasets[0].jsonl_fpath == "benchmarks/visualwebarena/data/visualwebarena.jsonl"
    assert agent.task_image_root == source_root
    assert resolved.visualwebarena_environment.resources_servers.webarena_browser.task_image_root == source_root
    resource = resolved.visualwebarena_environment.resources_servers.webarena_browser
    assert resource.terminate_on_action_error is True
    assert resource.max_tool_calls is None
    assert resource.max_task_image_bytes == agent.max_task_image_bytes
    model = resolved.policy_model.responses_api_models.vllm_model
    assert model.base_url == "http://127.0.0.1:8000/v1"
    assert model.chat_template_kwargs == {"truncate_history_thinking": False}

    # Task 499 exposed the independent agent/resource byte limits. Exercise
    # both resolved limits with a sparse file larger than the old 25 MiB cap.
    image = tmp_path / "large.png"
    with image.open("wb") as handle:
        handle.truncate(26 * 1024 * 1024)
    for limit in (agent.max_task_image_bytes, resource.max_task_image_bytes):
        path, mime = resolve_local_task_image_path(str(image), image_root=tmp_path, max_bytes=limit)
        assert path == image
        assert mime == "image/png"
    with pytest.raises(ValueError, match="byte limit"):
        resolve_local_task_image_path(str(image), image_root=tmp_path, max_bytes=25 * 1024 * 1024)


def test_write_env_is_private_and_rejects_display_sharing(tmp_path) -> None:
    env_path = tmp_path / "env.yaml"
    assert visualwebarena_prepare.write_env(
        env_path,
        input_jsonl=tmp_path / "input.jsonl",
        output_jsonl=tmp_path / "output.jsonl",
        source_root=tmp_path / "source",
    )
    content = env_path.read_text()
    assert "benchmarks/visualwebarena/configs/nano_omni.yaml" in content
    assert "agent_name: visualwebarena_benchmark_agent" in content
    assert "visualwebarena_source_root:" in content
    sampling = OmegaConf.load(env_path).responses_create_params
    assert sampling.max_output_tokens is None
    assert sampling.temperature == 0.1
    assert sampling.top_p == 0.95
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    with pytest.raises(ValueError, match="one DISPLAY"):
        visualwebarena_prepare.write_env(
            tmp_path / "other.yaml",
            input_jsonl=tmp_path / "input.jsonl",
            output_jsonl=tmp_path / "output.jsonl",
            source_root=tmp_path / "source",
            concurrency=2,
        )
