# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from benchmarks.indic.swebench import prepare as module
from benchmarks.swebench.verified import prepare as english
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_INSTANCES", 2)
    return [
        {
            **{key: f"original-{key}" for key in module.INSTANCE_FIELDS},
            "instance_id": f"repo__project-{index}",
            "problem_statement": " Fix the {mapping} bug.\r\nKeep code intact. ",
            "hints_text": "HIDDEN_HINT",
            "patch": "HIDDEN_GOLD_PATCH",
            "test_patch": "HIDDEN_TEST_PATCH",
            "FAIL_TO_PASS": '["hidden_failing_test"]',
            "PASS_TO_PASS": '["hidden_passing_test"]',
            **{
                f"problem_statement_{name}_translation": f" {code}: समस्या {{mapping}}\r\nSecond line. "
                for code, name in module.LANGUAGE_NAMES.items()
            },
        }
        for index in (1, 2)
    ]


@pytest.mark.parametrize("language", ["en", "hi"])
def test_exact_english_prompt_and_verifier_fields(records, monkeypatch, tmp_path, language):
    rows = module.build_rows(records, languages=[language])
    source = [
        {**{key: record[key] for key in module.INSTANCE_FIELDS}, "problem_statement": row["problem_statement"]}
        for record, row in zip(records, rows, strict=True)
    ]
    monkeypatch.setattr(english, "load_dataset", lambda *args, **kwargs: Dataset.from_list(source))
    monkeypatch.setattr(english, "get_hf_token", lambda: None)
    monkeypatch.setattr(english, "OUTPUT_FPATH", tmp_path / "english.jsonl")
    native_rows = [json.loads(line) for line in english.prepare().read_text().splitlines()]
    prompt = load_prompt_config("benchmarks/prompts/generic/default.yaml")
    for row, native in zip(rows, native_rows, strict=True):
        materialized = apply_prompt_to_row(row, prompt)
        assert {key: materialized[key] for key in native} == native
        text = materialized["responses_create_params"]["input"][0]["content"]
        assert row["problem_statement"] in text
        for hidden in (
            "HIDDEN_HINT",
            "HIDDEN_GOLD_PATCH",
            "HIDDEN_TEST_PATCH",
            "hidden_failing_test",
            "hidden_passing_test",
        ):
            assert hidden not in text
        assert not any(key.endswith("_translation") for key in row)


def test_all_languages_and_stable_ids(records):
    rows = module.build_rows(records)
    assert len(rows) == 28
    assert {row["language"] for row in rows} == set(module.DEFAULT_LANGUAGES)
    assert len({row["uuid"] for row in rows}) == 28
    selected = module.build_rows(records, languages=["hi"], instance_ids=["repo__project-2"])
    assert selected == [row for row in rows if row["language"] == "hi" and row["instance_id"] == "repo__project-2"]
    assert all(row["subset"] == "verified" and row["split"] == "test" for row in rows)


@pytest.mark.parametrize("value", [None, "", " "])
def test_missing_translation_rejected_even_outside_selected_subset(records, value):
    records[0]["problem_statement_Hindi_translation"] = value
    with pytest.raises(ValueError, match="Missing problem statement: hi/repo__project-1"):
        module.build_rows(records, languages=["hi"], instance_ids=["repo__project-2"])


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"languages": "hi"}, "nonempty sequence"),
        ({"languages": []}, "nonempty sequence"),
        ({"languages": ["xx"]}, "nonempty sequence"),
        ({"languages": ["hi", "hi"]}, "unique"),
        ({"instance_ids": "repo__project-1"}, "unique IDs"),
        ({"instance_ids": []}, "unique IDs"),
        ({"instance_ids": ["repo__project-1", "repo__project-1"]}, "unique IDs"),
        ({"instance_ids": ["missing"]}, "Unknown instance_ids"),
    ],
)
def test_invalid_selection(records, kwargs, error):
    with pytest.raises(ValueError, match=error):
        module.build_rows(records, **kwargs)


def test_missing_duplicate_and_malformed_tasks_rejected(records):
    with pytest.raises(ValueError, match="Expected 2"):
        module.build_rows(records[:1])
    with pytest.raises(ValueError, match="duplicate instance_id"):
        module.build_rows([records[0], records[0]])
    records[0]["test_patch"] = None
    with pytest.raises(ValueError, match="instance fields"):
        module.build_rows(records)


def test_prepare_pins_source_and_validates_before_writing(records, monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return "pinned.parquet"

    def load(*args, **kwargs):
        assert args == ("parquet",)
        assert kwargs == {"data_files": {"test": "pinned.parquet"}, "split": "test"}
        return Dataset.from_list(records)

    monkeypatch.setattr(module, "hf_hub_download", download)
    monkeypatch.setattr(module, "load_dataset", load)
    monkeypatch.setattr(module, "get_hf_token", lambda: None)
    monkeypatch.setattr(module, "get_token", lambda: "test-token")
    output = tmp_path / "nested" / "tasks.jsonl"
    assert module.prepare(languages=["hi"], output_fpath=str(output)) == output
    assert calls == [
        {
            "repo_id": module.SOURCE_ID,
            "filename": "test.parquet",
            "repo_type": "dataset",
            "revision": module.SOURCE_REVISION,
            "token": "test-token",
        }
    ]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == module.build_rows(records, languages=["hi"])
    original = output.read_bytes()
    records[0]["problem_statement_Hindi_translation"] = ""
    with pytest.raises(ValueError, match="Missing problem statement"):
        module.prepare(languages=["hi"], output_fpath=str(output))
    assert output.read_bytes() == original


def test_config_inherits_english_agent_verifier_and_repeats():
    parser = GlobalConfigDictParser()
    configs = []
    for path in ("benchmarks/swebench/verified/opencode.yaml", "benchmarks/indic/swebench/config.yaml"):
        configs.append(
            parser.parse(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=OmegaConf.create(
                        {
                            "config_paths": ["responses_api_models/vllm_model/configs/vllm_model.yaml", path],
                            "policy_base_url": "http://unused/v1",
                            "policy_api_key": "dummy",
                            "policy_model_name": "test",
                        }
                    ),
                    skip_load_from_cli=True,
                    skip_load_from_dotenv=True,
                    offline=True,
                )
            )
        )
    en, indic = configs
    assert indic.swebench_verified_opencode_resources_server == en.swebench_verified_opencode_resources_server
    expected_agent = OmegaConf.to_container(en.swebench_verified_opencode_sandboxed_agent, resolve=True)
    actual_agent = OmegaConf.to_container(indic.swebench_verified_opencode_sandboxed_agent, resolve=True)
    expected_config = expected_agent["responses_api_agents"]["opencode_sandboxed_agent"]
    actual_config = actual_agent["responses_api_agents"]["opencode_sandboxed_agent"]
    assert actual_config["datasets"][0]["num_repeats"] == expected_config["datasets"][0]["num_repeats"] == 3
    expected_config["datasets"] = actual_config["datasets"]
    assert actual_agent == expected_agent
    assert {server.name for server in parser.filter_for_server_instance_configs(indic)} == {
        "policy_model",
        "swebench_verified_opencode_resources_server",
        "swebench_verified_opencode_sandboxed_agent",
    }
    benchmark = BenchmarkConfig.from_config_path(module.BENCHMARK_DIR / "config.yaml")
    assert benchmark.agent_name == "swebench_verified_opencode_sandboxed_agent"
    manifest = load_manifest(module.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.resources_server == "swebench"
    assert manifest.agent_server == "opencode_sandboxed_agent"
    assert manifest.datasets[0].model_dump(exclude_none=True) == {
        key: value for key, value in actual_config["datasets"][0].items() if key != "license"
    }
