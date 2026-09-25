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


def test_config_changes_only_dataset_and_matches_manifest():
    parser = GlobalConfigDictParser()
    configs = []
    for path in ("benchmarks/math-500/config.yaml", "benchmarks/indic/math_500/config.yaml"):
        config = parser.parse(
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
        configs.append(OmegaConf.to_container(config, resolve=True))
    english, indic = configs
    agent = "math_500_math_with_judge_simple_agent"
    original_agent = english[agent]["responses_api_agents"]["simple_agent"]
    translated_agent = indic[agent]["responses_api_agents"]["simple_agent"]
    assert original_agent["datasets"][0]["prompt_config"] == translated_agent["datasets"][0]["prompt_config"]
    assert original_agent["datasets"][0]["num_repeats"] == translated_agent["datasets"][0]["num_repeats"] == 1
    original_agent["datasets"] = translated_agent["datasets"]
    english.pop("config_paths")
    indic.pop("config_paths")
    assert english == indic
    benchmark = BenchmarkConfig.from_config_path(module.BENCHMARK_DIR / "config.yaml")
    assert benchmark.agent_name == agent
    manifest = load_manifest(module.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.resources_server == "math_with_judge"
    assert manifest.agent_server == "simple_agent"
    assert manifest.datasets[0].model_dump(exclude_none=True) == {
        key: value for key, value in translated_agent["datasets"][0].items() if key != "license"
    }
