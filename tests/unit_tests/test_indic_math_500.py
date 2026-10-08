# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from benchmarks.indic.math_500 import prepare as module
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    return [
        {
            "unique_id": f"test/algebra/{index}.json",
            "problem": " Find $x$ when $2x=1$.\nKeep {braces}. ",
            "answer": r"\frac{1}{2}",
            "solution": "HIDDEN_SOLUTION",
            "subject": "Algebra",
            "level": 1,
            **{
                f"problem_{name}_translation": f" {code}: $2x=1$ में $x$ ज्ञात करें।\n{{braces}} "
                for code, name in module.LANGUAGE_NAMES.items()
            },
        }
        for index in (1, 2)
    ]


def test_english_pipeline_parity_and_no_solution_in_prompt(records, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("english_math_500", "benchmarks/math-500/prepare.py")
    english = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(english)
    monkeypatch.setattr(english, "DATA_DIR", tmp_path)
    monkeypatch.setattr(english, "OUTPUT_FPATH", tmp_path / "english.jsonl")
    originals = [{k: v for k, v in row.items() if not k.endswith("_translation")} for row in records]
    monkeypatch.setattr(
        english.urllib.request,
        "urlretrieve",
        lambda url, path: Path(path).write_text("".join(json.dumps(row) + "\n" for row in originals)),
    )
    original_rows = [json.loads(line) for line in english.prepare().read_text().splitlines()]
    translated = module.build_rows(records, languages=["en", "hi"])
    prompt = load_prompt_config("benchmarks/prompts/generic/math.yaml")
    for original, row in zip(original_rows, translated[:2], strict=True):
        assert row["expected_answer"] == original["expected_answer"]
        assert row["unique_id"] == original["unique_id"]
        assert row["subject"] == original["subject"]
        assert row["level"] == original["level"]
        assert (
            apply_prompt_to_row(row, prompt)["responses_create_params"]
            == apply_prompt_to_row(original, prompt)["responses_create_params"]
        )
    hindi = translated[2]
    text = records[0]["problem_Hindi_translation"]
    assert hindi["question"] == text
    messages = apply_prompt_to_row(hindi, prompt)["responses_create_params"]["input"]
    assert messages == [{"role": "user", "content": prompt.user.format(question=text)}]
    assert "HIDDEN_SOLUTION" not in json.dumps(translated)
    assert hindi["expected_answer"] not in messages[0]["content"]


def test_default_languages_have_distinct_stable_ids_and_filtering(records):
    rows = module.build_rows(records)
    assert len(rows) == 24
    assert len({row["uuid"] for row in rows}) == 24
    assert {row["language"] for row in rows} == set(module.DEFAULT_LANGUAGES)
    assert "as" not in module.DEFAULT_LANGUAGES
    assert "sa" not in module.DEFAULT_LANGUAGES
    assert module.build_rows(records, languages=["hi"]) == [row for row in rows if row["language"] == "hi"]
    assert module.build_rows(records) == rows


def test_assamese_and_sanskrit_remain_available_by_explicit_selection(records):
    rows = module.build_rows(records, languages=["as", "sa"])
    assert len(rows) == 4
    assert {row["language"] for row in rows} == {"as", "sa"}


@pytest.mark.parametrize("value", [None, "", " "])
def test_missing_translation_has_no_english_fallback(records, value):
    records[0]["problem_Hindi_translation"] = value
    with pytest.raises(ValueError, match="Missing problem text: hi/test/algebra/1.json"):
        module.build_rows(records, languages=["hi"])


@pytest.mark.parametrize("languages", ["hi", [], ["xx"], ["hi", "hi"]])
def test_invalid_language_selection(records, languages):
    with pytest.raises(ValueError, match="languages must"):
        module.build_rows(records, languages=languages)


def test_missing_and_duplicate_tasks(records):
    with pytest.raises(ValueError, match="Expected 2"):
        module.build_rows(records[:1])
    with pytest.raises(ValueError, match="Duplicate unique_id"):
        module.build_rows([records[0], records[0]])


@pytest.mark.parametrize(
    "field,value",
    [
        ("unique_id", ""),
        ("answer", None),
        ("answer", ""),
        ("subject", None),
        ("level", True),
        ("level", 0),
        ("level", 6),
    ],
)
def test_invalid_verification_metadata(records, field, value):
    records[0][field] = value
    with pytest.raises(ValueError, match=field):
        module.build_rows(records)


def test_prepare_downloads_data_and_preserves_output_on_validation_failure(records, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(module, "maybe_get_global_config_dict", lambda: None)
    monkeypatch.setattr(module, "get_token", lambda: "test-token")
    monkeypatch.setattr(module, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or "source.parquet")
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: Dataset.from_list(records))
    output = tmp_path / "tasks.jsonl"
    assert module.prepare(languages=["hi"], output_fpath=str(output)) == output
    assert calls[0] == {
        "repo_id": module.SOURCE_ID,
        "filename": "test.parquet",
        "repo_type": "dataset",
        "revision": "29557d8eaa22621b82f3af5557ab60babcf3feb5",
        "token": "test-token",
    }
    original = output.read_bytes()
    assert [json.loads(line) for line in output.read_text().splitlines()] == module.build_rows(
        records, languages=["hi"]
    )
    records[0]["problem_Hindi_translation"] = ""
    with pytest.raises(ValueError):
        module.prepare(languages=["hi"], output_fpath=str(output))
    assert output.read_bytes() == original


def _resolve_benchmarks(*names):
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                        *(f"benchmarks/{name}/config.yaml" for name in names),
                    ],
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
    return OmegaConf.to_container(config, resolve=True)


def test_config_reuses_math_pipeline_and_matches_manifest():
    english = _resolve_benchmarks("math-500")
    indic = _resolve_benchmarks("indic/math_500")
    agent = "indic_math_500_math_with_judge_simple_agent"
    original_agent = english["math_500_math_with_judge_simple_agent"]["responses_api_agents"]["simple_agent"]
    translated_agent = indic[agent]["responses_api_agents"]["simple_agent"]
    assert "math_500_math_with_judge_simple_agent" not in indic
    assert original_agent["datasets"][0]["prompt_config"] == translated_agent["datasets"][0]["prompt_config"]
    assert original_agent["datasets"][0]["num_repeats"] == translated_agent["datasets"][0]["num_repeats"] == 1
    assert translated_agent["resources_server"] == {
        "type": "resources_servers",
        "name": "indic_math_500_math_with_judge_resources_server",
    }
    original_agent["datasets"] = translated_agent["datasets"]
    original_agent["resources_server"] = translated_agent["resources_server"]
    assert original_agent == translated_agent
    original_verifier = english["math_500_math_with_judge_resources_server"]["resources_servers"]["math_with_judge"]
    translated_verifier = indic["indic_math_500_math_with_judge_resources_server"]["resources_servers"][
        "math_with_judge"
    ]
    assert translated_verifier["should_use_judge"] is False
    assert translated_verifier["format_tolerant_answer_extraction"] is True
    assert original_verifier == {
        key: value for key, value in translated_verifier.items() if key != "format_tolerant_answer_extraction"
    }
    benchmark = BenchmarkConfig.from_config_path(module.BENCHMARK_DIR / "config.yaml")
    assert benchmark.agent_name == agent
    manifest = load_manifest(module.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.resources_server == "math_with_judge"
    assert manifest.agent_server == "simple_agent"
    assert manifest.datasets[0].model_dump(exclude_none=True) == {
        key: value for key, value in translated_agent["datasets"][0].items() if key != "license"
    }


@pytest.mark.parametrize("benchmarks", [("math-500", "indic/math_500"), ("indic/math_500", "math-500")])
def test_english_and_indic_benchmarks_compose_in_both_orders(benchmarks):
    combined = _resolve_benchmarks(*benchmarks)
    agents = {
        name: block["responses_api_agents"]["simple_agent"]
        for name, block in combined.items()
        if isinstance(block, dict) and "responses_api_agents" in block
    }
    assert set(agents) == {"math_500_math_with_judge_simple_agent", "indic_math_500_math_with_judge_simple_agent"}
    assert {dataset["name"] for agent in agents.values() for dataset in agent["datasets"]} == {
        "math-500",
        "indic_math_500",
    }
    for benchmark, agent_name, dataset_name in (
        ("math-500", "math_500_math_with_judge_simple_agent", "math-500"),
        ("indic/math_500", "indic_math_500_math_with_judge_simple_agent", "indic_math_500"),
    ):
        standalone = _resolve_benchmarks(benchmark)
        agent = agents[agent_name]
        assert [dataset["name"] for dataset in agent["datasets"]] == [dataset_name]
        assert combined[agent_name] == standalone[agent_name]
        resource_name = agent["resources_server"]["name"]
        assert combined[resource_name] == standalone[resource_name]


def test_local_parquet_preparation_never_downloads(records, monkeypatch, tmp_path):
    source = tmp_path / "test.parquet"
    Dataset.from_list(records).to_parquet(source)

    def reject_download(**kwargs):
        pytest.fail("Local dataset preparation attempted a download")

    monkeypatch.setattr(module, "hf_hub_download", reject_download)
    output = module.prepare(languages=["hi"], source_parquet=str(source), output_fpath=str(tmp_path / "hi.jsonl"))
    assert [json.loads(line) for line in output.read_text().splitlines()] == module.build_rows(
        records, languages=["hi"]
    )
