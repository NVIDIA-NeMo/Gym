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
    languages = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"}
    assert len(rows) == 28
    assert {row["language"] for row in rows} == languages
    assert len({row["uuid"] for row in rows}) == 28
    selected = module.build_rows(records, languages=["hi"])
    assert selected == [row for row in rows if row["language"] == "hi"]
    assert all(row["subset"] == "verified" and row["split"] == "test" for row in rows)


@pytest.mark.parametrize("value", [None, "", " "])
def test_missing_translation_rejected(records, value):
    records[0]["problem_statement_Hindi_translation"] = value
    with pytest.raises(ValueError, match="Missing problem statement: hi/repo__project-1"):
        module.build_rows(records, languages=["hi"])


@pytest.mark.parametrize("languages", ["hi", [], ["xx"], ["hi", "hi"]])
def test_invalid_languages(records, languages):
    with pytest.raises(ValueError, match="languages must"):
        module.build_rows(records, languages=languages)


def test_incomplete_or_duplicate_source_rejected(records):
    with pytest.raises(ValueError, match="Expected 2"):
        module.build_rows(records[:1])
    with pytest.raises(ValueError, match="Duplicate instance_id"):
        module.build_rows([records[0], records[0]])


def test_prepare_loads_test_split_and_writes_all_languages(records, monkeypatch, tmp_path):
    calls = []

    def load(repo_id, *, split):
        calls.append((repo_id, split))
        return Dataset.from_list(records)

    monkeypatch.setattr(module, "load_dataset", load)
    output = tmp_path / "nested" / "tasks.jsonl"
    assert module.prepare(output_fpath=str(output)) == output
    assert calls == [("ai4bharat/indic-swe-bench", "test")]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == module.build_rows(records)
    original = output.read_bytes()
    records[0]["problem_statement_Hindi_translation"] = ""
    with pytest.raises(ValueError, match="Missing problem statement"):
        module.prepare(output_fpath=str(output))
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
    assert indic.indic_swebench_verified_opencode_resources_server == en.swebench_verified_opencode_resources_server
    expected_agent = OmegaConf.to_container(en.swebench_verified_opencode_sandboxed_agent, resolve=True)
    actual_agent = OmegaConf.to_container(indic.indic_swebench_verified_opencode_sandboxed_agent, resolve=True)
    expected_config = expected_agent["responses_api_agents"]["opencode_sandboxed_agent"]
    actual_config = actual_agent["responses_api_agents"]["opencode_sandboxed_agent"]
    assert actual_config["datasets"][0]["num_repeats"] == expected_config["datasets"][0]["num_repeats"] == 3
    expected_config["datasets"] = actual_config["datasets"]
    assert actual_config["resources_server"]["name"] == "indic_swebench_verified_opencode_resources_server"
    expected_config["resources_server"] = actual_config["resources_server"]
    assert actual_agent == expected_agent
    assert {server.name for server in parser.filter_for_server_instance_configs(indic)} == {
        "policy_model",
        "indic_swebench_verified_opencode_resources_server",
        "indic_swebench_verified_opencode_sandboxed_agent",
    }
    benchmark = BenchmarkConfig.from_config_path(module.BENCHMARK_DIR / "config.yaml")
    assert benchmark.agent_name == "indic_swebench_verified_opencode_sandboxed_agent"
    manifest = load_manifest(module.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.resources_server == "swebench"
    assert manifest.agent_server == "opencode_sandboxed_agent"
    assert manifest.datasets[0].model_dump(exclude_none=True) == {
        key: value for key, value in actual_config["datasets"][0].items() if key != "license"
    }


@pytest.mark.parametrize("indic_first", [False, True])
def test_english_and_indic_can_run_together(indic_first):
    paths = ["benchmarks/swebench/verified/opencode.yaml", "benchmarks/indic/swebench/config.yaml"]
    if indic_first:
        paths.reverse()
    config = GlobalConfigDictParser().parse_no_environment(
        initial_global_config_dict=OmegaConf.create(
            {"config_paths": paths, **GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT}
        )
    )
    for prefix, dataset, path in (
        ("", "swebench_verified", "benchmarks/swebench/data/swebench_verified_benchmark.jsonl"),
        ("indic_", "indic_swebench", "benchmarks/indic/swebench/data/swebench_benchmark.jsonl"),
    ):
        agent = config[
            f"{prefix}swebench_verified_opencode_sandboxed_agent"
        ].responses_api_agents.opencode_sandboxed_agent
        assert agent.datasets[0].name == dataset
        assert agent.datasets[0].jsonl_fpath == path
        assert agent.datasets[0].num_repeats == 3
        assert agent.resources_server.name == f"{prefix}swebench_verified_opencode_resources_server"
        verifier = config[agent.resources_server.name].resources_servers.swebench
        assert verifier.apply_anti_cheating is False
        assert verifier.allowed_agents == ["opencode_sandboxed_agent"]
