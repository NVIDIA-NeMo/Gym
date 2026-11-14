# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from benchmarks.longfact.prepare import (
    BENCHMARK_VARIANT,
    TOPICS,
    _parse_rows,
    _source_path,
    build_rows,
    select_rows,
)
from nemo_gym.config.config_types import BenchmarkDatasetConfig, DatasetConfig
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.data.task_data import load_task_data_schema, validate_jsonl_rows
from nemo_gym.data.task_materialization import materialize_task
from nemo_gym.environment.assets import asset_config_path


@pytest.mark.parametrize("hosted_card", [False, True])
@pytest.mark.parametrize("benchmark", [False, True])
@pytest.mark.parametrize("milvus", [False, True])
@pytest.mark.parametrize("judge_temperature", [0.1, 0.7])
def test_published_configs_resolve_example_data_and_judge(
    monkeypatch, hosted_card, benchmark, milvus, judge_temperature
):
    root = Path(__file__).resolve().parents[3]
    for key in tuple(os.environ):
        if key.startswith("LONGFACT_"):
            monkeypatch.delenv(key)
    for key, value in {
        "LONGFACT_EVALUATOR_BASE_URL": "https://offline.invalid/v1",
        "LONGFACT_EVALUATOR_API_KEY": "offline-placeholder",
        "LONGFACT_EVALUATOR_MODEL": "offline-o3",
        "LONGFACT_EVALUATOR_RESPONSE_MODEL": "offline-o3",
        "LONGFACT_QWEN_JUDGE_TEMPERATURE": str(judge_temperature),
        "LONGFACT_SEARCH_BACKEND": "tavily",
        "TAVILY_API_KEY": "offline-placeholder",
    }.items():
        monkeypatch.setenv(key, value)
    selected = "benchmarks/longfact/config.yaml" if benchmark else asset_config_path("resources-server", "longfact")
    config_paths = [str(root / selected), str(root / "benchmarks/longfact/evaluator_local_vllm.yaml")]
    if hosted_card:
        config_paths.extend(
            str(root / path)
            for path in ("benchmarks/longfact/evaluator_openai.yaml", "benchmarks/longfact/gpt5_card.yaml")
        )
    if milvus:
        config_paths.append(str(root / "benchmarks/longfact/milvus.yaml"))
    config_paths.append(str(root / "responses_api_models/openai_model/configs/openai_model.yaml"))
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": config_paths,
                    "policy_base_url": "https://offline.invalid/v1",
                    "policy_api_key": "offline-placeholder",
                    "policy_model_name": "offline-policy",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            skip_exporters=True,
        )
    )
    prefix = "longfact_longfact" if benchmark else "longfact"
    agent_name = f"{prefix}_simple_agent"
    agent = resolved[agent_name]["responses_api_agents"]["simple_agent"]
    [dataset] = agent["datasets"]
    dataset = OmegaConf.to_container(dataset, resolve=True)
    if benchmark:
        benchmark_dataset = BenchmarkDatasetConfig.model_validate(dataset)
        assert benchmark_dataset.prepare_script == Path("benchmarks/longfact/prepare.py")
    else:
        example = DatasetConfig.model_validate(dataset)
        lines = (root / example.jsonl_fpath).read_text().splitlines()
        rows = [json.loads(line) for line in lines]
        assert example.type == "example" and len(rows) == 5
        schema = load_task_data_schema(root / "resources_servers/longfact")
        report = validate_jsonl_rows("longfact", schema, str(example.jsonl_fpath), lines)
        assert report.clean, report.summary()
        assert example.license == "Creative Commons Attribution 4.0 International"
        metrics = json.loads((root / "resources_servers/longfact/data/example_metrics.json").read_text())
        assert DatasetConfig.model_validate(metrics).license == example.license
    assert agent["model_server"]["name"] == "policy_model"
    assert agent["resources_server"]["name"] == (f"{prefix}_resources_server" if benchmark else "longfact")
    resources = resolved[agent["resources_server"]["name"]]["resources_servers"]["longfact"]
    if hosted_card:
        assert resources["evaluator_model_server"]["name"] == "longfact_openai_evaluator_model"
        assert resources["fact_check_mode"] == "gpt5_card_batch"
        assert resources["claim_extraction_prompt"] == "claim_listing_v1"
        assert resources["evaluator_responses_create_params"]["temperature"] is None
        assert resources["evaluator_responses_create_params"]["top_p"] is None
        assert (
            resolved["longfact_openai_evaluator_model"]["responses_api_models"]["openai_model"]["openai_model"]
            == "offline-o3"
        )
    else:
        assert resources["evaluator_model_server"]["name"] == "longfact_evaluator_model"
        assert resources["claim_extraction_prompt"] == "safe_v1"
        judge = resolved["longfact_evaluator_model"]["responses_api_models"]["vllm_model"]
        assert resources["evaluator_responses_create_params"]["temperature"] == judge_temperature
        assert resources["evaluator_responses_create_params"]["top_p"] == 0.8
        assert not judge.get("sampling_overrides")
        assert "presence_penalty" not in judge["extra_body"]
    if milvus:
        assert resources["search_backend"] == "milvus"
        assert resolved["component_extras"]["resources_servers/longfact"] == ["milvus"]
    else:
        assert resources["search_backend"] == "tavily"
        assert not resolved.get("component_extras", {}).get("resources_servers/longfact")
    environment = resolved[f"{prefix}_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert environment["agent_server"]["name"] == agent_name


def test_source_path_uses_official_layout():
    assert _source_path("objects", "longfact-objects_gpt4_01-12-2024_noduplicates", "physics") == (
        "longfact/longfact-objects_gpt4_01-12-2024_noduplicates/longfact-objects_physics.jsonl"
    )


def test_topic_order_matches_case_sensitive_source_file_sort():
    assert TOPICS[:3] == ("20th-century-events", "US-foreign-policy", "accounting")


def test_parse_rows_requires_thirty_valid_prompts():
    source_rows = [{"prompt": f"prompt {index}", "canary": f"fixture canary {index}"} for index in range(30)]
    payload = b"".join(json.dumps(row).encode() + b"\n" for row in source_rows)
    assert _parse_rows(payload, "fixture.jsonl") == source_rows
    with pytest.raises(ValueError, match="Expected 30"):
        _parse_rows(
            b"".join(json.dumps(row).encode() + b"\n" for row in source_rows[:29]),
            "fixture.jsonl",
        )
    source_rows[0].pop("canary")
    with pytest.raises(ValueError, match="upstream canary"):
        _parse_rows(b"".join(json.dumps(row).encode() + b"\n" for row in source_rows), "fixture.jsonl")


def test_build_rows_preserves_subset_topic_and_row_order(tmp_path, monkeypatch):
    import benchmarks.longfact.prepare as module

    monkeypatch.setattr(module, "SUBSETS", (("concepts", "concepts"),))
    monkeypatch.setattr(module, "TOPICS", ("alpha", "beta"))
    for topic in module.TOPICS:
        path = tmp_path / "longfact" / "concepts"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"longfact-concepts_{topic}.jsonl").write_text(
            "".join(
                json.dumps({"prompt": f"{topic} {index}", "canary": f"fixture canary {topic} {index}"}) + "\n"
                for index in range(30)
            ),
            encoding="utf-8",
        )

    rows = build_rows(tmp_path)

    assert len(rows) == 60
    assert [row["id"] for row in rows[:2]] == ["longfact-concepts-alpha-00", "longfact-concepts-alpha-01"]
    assert rows[30]["id"] == "longfact-concepts-beta-00"
    assert all(row["benchmark_variant"] == BENCHMARK_VARIANT for row in rows)
    assert all(row["task_id"] == row["id"] for row in rows)
    assert [row["canary"] for row in rows] == [
        f"fixture canary {topic} {index}" for topic in ("alpha", "beta") for index in range(30)
    ]
    original = [materialize_task(row, taskset="longfact") for row in rows]
    reordered = [materialize_task(row, taskset="longfact") for row in reversed(rows)]
    assert {task["task_id"]["task_id"] for task in original} == {row["id"] for row in rows}
    assert original == list(reversed(reordered))


def test_select_rows_is_reproducible_and_retains_original_rows():
    rows = [{"id": str(index)} for index in range(100)]

    selected = select_rows(rows, sample_size=17, sample_seed=17)

    assert selected == select_rows(rows, sample_size=17, sample_seed=17)
    assert selected != select_rows(rows, sample_size=17, sample_seed=104729)
    assert selected != rows[:17]
    assert len(selected) == 17
    assert len({row["id"] for row in selected}) == 17
    assert all(row in rows for row in selected)


@pytest.mark.parametrize("custom_output", [False, True])
def test_preparation_worker_writes_under_the_callers_directory(tmp_path, monkeypatch, custom_output):
    from benchmarks.longfact import prepare as source
    from nemo_gym.cli._prepare_worker import load_preparer
    from nemo_gym.environment.benchmark_preparation import _multiprocess_benchmark_prepare_fn

    installation = tmp_path / "installation"
    installation.mkdir()
    script = installation / "prepare.py"
    script.write_bytes(Path(source.__file__).read_bytes())
    module = load_preparer(str(script))
    monkeypatch.setattr(module, "SUBSETS", (("concepts", "concepts"),))
    monkeypatch.setattr(module, "TOPICS", ("alpha", "beta"))
    source_dir = tmp_path / "source"
    for topic in module.TOPICS:
        directory = source_dir / "longfact" / "concepts"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"longfact-concepts_{topic}.jsonl").write_text(
            "".join(
                json.dumps({"prompt": f"{topic} {index}", "canary": f"fixture canary {topic} {index}"}) + "\n"
                for index in range(30)
            ),
            encoding="utf-8",
        )
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.setenv("LONGFACT_SAMPLE_SIZE", "1")
    monkeypatch.setenv("LONGFACT_SAMPLE_SEED", "unexpected")
    output = Path("custom/output.jsonl" if custom_output else "benchmarks/longfact/data/longfact_benchmark.jsonl")
    args = {"source_dir": source_dir}
    if custom_output:
        args["output_fpath"] = output
    benchmark = SimpleNamespace(name="longfact", dataset=SimpleNamespace(jsonl_fpath=output))

    try:
        _multiprocess_benchmark_prepare_fn((benchmark, module.__name__, args, None))
    finally:
        sys.modules.pop(module.__name__, None)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 60
    assert [row["id"] for row in rows[:2]] == ["longfact-concepts-alpha-00", "longfact-concepts-alpha-01"]
    assert rows[30]["id"] == "longfact-concepts-beta-00"
    assert all(row["task_id"] == row["id"] for row in rows)
    assert [row["canary"] for row in rows] == [
        f"fixture canary {topic} {index}" for topic in ("alpha", "beta") for index in range(30)
    ]
    assert [row["prompt"] for row in rows] == [
        f"{topic} {index}" for topic in ("alpha", "beta") for index in range(30)
    ]
    assert not (installation / "data").exists()
