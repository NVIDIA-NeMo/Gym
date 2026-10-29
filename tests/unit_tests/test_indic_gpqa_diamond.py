# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import random

import pytest
import yaml
from omegaconf import OmegaConf

from benchmarks.gpqa import prepare as english_module
from benchmarks.indic.gpqa_diamond import prepare as module
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.data import apply_prompt_to_row, load_prompt_config
from nemo_gym.data.prompt import validate_prompt_compatibility


def record(index=0, language="en", **extra):
    row = {
        "Record ID": f"record-{index}",
        "Question": f"Synthetic question {index}?",
        "Correct Answer": "right ",
        "Incorrect Answer 1": "one",
        "Incorrect Answer 2": "two",
        "Incorrect Answer 3": "three",
        "Explanation": "DO NOT INCLUDE IN MODEL INPUT",
        **extra,
    }
    if language != "en":
        row.update(
            {
                f"{field}_{module.LANGUAGES[language]}_translation": "अनुवाद " + row[field]
                for field in module.TEXT_FIELDS
            }
        )
    return row


@pytest.fixture
def english_rows(tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(english_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(english_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(english_module, "OUTPUT_FPATH", tmp_path / "english.jsonl")

    def prepare(records):
        monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: records)
        with english_module.prepare().open() as handle:
            return [json.loads(line) for line in handle]

    return prepare


def build(records, **kwargs):
    return module.build_rows(
        records,
        language="en",
        **kwargs,
    )


def test_exact_shuffle_prompt_and_subset_stability(monkeypatch, english_rows):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 3)
    records = [record(index) for index in range(3)]
    random_state = random.getstate()
    full = build(records)
    expected_english = english_rows(records)

    assert random.getstate() == random_state
    for index, row in enumerate(full):
        expected = [records[index][field] for field in module.TEXT_FIELDS[1:]]
        random.Random(int(hashlib.md5(records[index]["Question"].encode()).hexdigest(), 16)).shuffle(expected)
        assert row["options"] == [{letter: text} for letter, text in zip("ABCD", expected)]
        assert row["expected_answer"] == "ABCD"[expected.index(records[index]["Correct Answer"])]
        assert row["problem"] == records[index]["Question"] + "\n" + "\n".join(
            f"{letter}: {text}" for letter, text in zip("ABCD", expected)
        )
        english = expected_english[index]
        assert {key: row[key] for key in english if key not in ("uuid", "task_id")} == {
            key: value for key, value in english.items() if key not in ("uuid", "task_id")
        }
        assert row["task_id"] == f"en/{english['task_id']}"
    assert build(records, question_ids=["2"]) == full[2:]

    translated = [record(index, "hi") for index in range(3)]
    hindi = module.build_rows(
        translated,
        language="hi",
    )
    for original, translation in zip(full, hindi):
        assert original["expected_answer"] == translation["expected_answer"]
        assert translation["options"] == [
            {letter: "अनुवाद " + text for letter, text in option.items()} for option in original["options"]
        ]
    assert full[0]["uuid"] != hindi[0]["uuid"]
    assert [row["task_id"] for row in hindi] == ["hi/record-0", "hi/record-1", "hi/record-2"]
    assert len({row["task_id"] for row in full + hindi}) == len(full + hindi)
    assert hindi[0]["metadata"]["subset_for_metrics"] == "hi"
    assert hindi[0]["template_metadata"] == {"allow_localized_answer_labels": True}
    assert "template_metadata" not in full[0]

    prompt = load_prompt_config("benchmarks/prompts/eval/aai/mcq-4choices.yaml")
    validate_prompt_compatibility(full, prompt)
    rendered = apply_prompt_to_row(full[0], prompt)
    assert [message["role"] for message in rendered["responses_create_params"]["input"]] == ["user"]
    assert (
        rendered["responses_create_params"]["input"]
        == apply_prompt_to_row(expected_english[0], prompt)["responses_create_params"]["input"]
    )


def test_duplicate_options_match_english(english_rows):
    records = [record(index) for index in range(198)]
    records[0]["Incorrect Answer 3"] = records[0]["Correct Answer"]
    rows = build(records)

    assert rows[0]["expected_answer"] == english_rows(records)[0]["expected_answer"]
    assert rows[0]["metadata"]["duplicate_choice_text"]
    assert len(rows) == 198


@pytest.mark.parametrize("question_ids", [[], "0", [0], ["0", "0"], ["198"]])
def test_invalid_question_ids(question_ids):
    with pytest.raises(ValueError, match="question_ids"):
        build([record(index) for index in range(198)], question_ids=question_ids)


@pytest.mark.parametrize(
    "extra",
    [{"Question": ""}, {"Question": None}, {"Correct Answer": " "}, {"Incorrect Answer 1": 2}],
)
def test_invalid_source_rows(extra):
    records = [record(index) for index in range(198)]
    records[0].update(extra)
    with pytest.raises(ValueError, match="GPQA"):
        build(records)


def test_prepare_loads_train_split_once_for_all_languages(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    records = [record(index, "hi") for index in range(2)]
    calls = []
    source_path = tmp_path / "train.parquet"
    source_path.write_bytes(b"synthetic parquet download")
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(source_path)

    def load(repo_id, *, split, data_files):
        calls.append((repo_id, split, data_files))
        return records

    monkeypatch.setattr(module, "hf_hub_download", download)
    monkeypatch.setattr(module, "load_dataset", load)
    output = module.prepare(tmp_path / "prepared.jsonl", languages=["en", "hi"], question_ids=["1"])
    with output.open() as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 2
    assert rows[1]["question"] == "अनुवाद Synthetic question 1?"
    assert "DO NOT INCLUDE" not in output.read_text()
    assert rows[1]["metadata"]["source_id"] == "ai4bharat/indic-gpqa"
    assert rows[0]["metadata"]["record_id"] == "record-1"
    assert calls == [("parquet", "train", str(source_path))]
    assert downloads == [
        {
            "repo_id": module.SOURCE_ID,
            "repo_type": "dataset",
            "revision": module.SOURCE_REVISION,
            "filename": "train.parquet",
        }
    ]
    assert rows[0]["metadata"]["source_split"] == "train"
    assert rows[0]["metadata"]["source_revision"] == module.SOURCE_REVISION
    provenance = json.loads(output.with_suffix(".manifest.json").read_text())
    assert provenance["source_sha256"] == module.SOURCE_SHA256
    assert provenance["prepared_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance["rows_per_language"] == {"en": 1, "hi": 1}


def test_config_matches_english_pipeline():
    config = yaml.safe_load((module.DIRECTORY / "config.yaml").read_text(encoding="utf-8"))
    english = yaml.safe_load((english_module.BENCHMARK_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert config["config_paths"] == english["config_paths"]
    assert config["indic_gpqa_diamond_resources_server"] == english["gpqa_mcqa_resources_server"]
    agent = config["indic_gpqa_diamond_agent"]
    english_agent = english["gpqa_mcqa_simple_agent"]
    assert agent["_inherit_from"] == english_agent["_inherit_from"]
    params = agent["responses_api_agents"]["simple_agent"]
    english_params = english_agent["responses_api_agents"]["simple_agent"]
    assert params["max_steps"] == english_params["max_steps"]
    dataset = params["datasets"][0]
    english_dataset = english_params["datasets"][0]
    assert dataset["num_repeats"] == english_dataset["num_repeats"] == 8
    assert dataset["prompt_config"] == english_dataset["prompt_config"]
    environment = config["indic_gpqa_diamond_environment_server"]
    assert environment["_inherit_from"] == english["gpqa_mcqa_environment_server"]["_inherit_from"]
    assert environment["environment_servers"]["legacy_agent"]["agent_server"]["name"] == "indic_gpqa_diamond_agent"
    assert set(config) == {
        "config_paths",
        "indic_gpqa_diamond_resources_server",
        "indic_gpqa_diamond_agent",
        "indic_gpqa_diamond_environment_server",
    }


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "benchmarks",
    [("indic/gpqa_diamond",), ("gpqa", "indic/gpqa_diamond"), ("indic/gpqa_diamond", "gpqa")],
)
def test_environment_routing_alone_and_with_english_gpqa(
    benchmarks: tuple[str, ...], strict: bool, caplog: pytest.LogCaptureFixture
) -> None:
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {
                    "config_paths": [f"benchmarks/{name}/config.yaml" for name in benchmarks],
                    "error_on_agent_without_environment_server": strict,
                },
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    expected = {"indic_gpqa_diamond_environment_server": "indic_gpqa_diamond_agent"}
    if "gpqa" in benchmarks:
        expected["gpqa_mcqa_environment_server"] = "gpqa_mcqa_simple_agent"
    environments = {
        name: block["environment_servers"]["legacy_agent"]
        for name, block in resolved.items()
        if OmegaConf.is_dict(block) and "environment_servers" in block
    }
    assert "agents without an environment server" not in caplog.text
    assert set(environments) == set(expected)
    for name, agent in expected.items():
        assert environments[name]["entrypoint"] == "app.py"
        assert environments[name]["agent_server"] == {"type": "responses_api_agents", "name": agent}
    assert resolved["indic_gpqa_diamond_agent"]["responses_api_agents"]["simple_agent"]["resources_server"] == {
        "type": "resources_servers",
        "name": "indic_gpqa_diamond_resources_server",
    }


@pytest.mark.parametrize("value", [None, "", "record-1"])
def test_prepare_rejects_invalid_source_ids(value, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    source = [record(index, "hi") for index in range(2)]
    source[0]["Record ID"] = value
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(module, "_source_file", lambda _: tmp_path / "synthetic.parquet")
    with pytest.raises(ValueError, match="GPQA Record IDs"):
        module.prepare(tmp_path / "invalid.jsonl", languages=["hi"])
    assert not (tmp_path / "invalid.jsonl").exists()


@pytest.mark.parametrize("value", [None, "", " "])
def test_missing_translation_does_not_fall_back_to_english(value, monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 1)
    source = record(language="hi")
    source["Question_Hindi_translation"] = value
    with pytest.raises(ValueError, match="GPQA hi/0"):
        module.build_rows([source], language="hi")


def test_local_parquet_preparation_is_offline_and_hash_checked(tmp_path, monkeypatch):
    from datasets import Dataset

    source = tmp_path / "train.parquet"
    Dataset.from_list([record(0, "hi")]).to_parquet(source)
    monkeypatch.setattr(module, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())

    def no_download(**kwargs):
        pytest.fail("Local source preparation must not contact the Hub")

    monkeypatch.setattr(module, "hf_hub_download", no_download)
    output = module.prepare(tmp_path / "prepared.jsonl", languages=["hi"], source_path=source)
    with output.open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 1
    assert rows[0]["question"] == "अनुवाद Synthetic question 0?"
    assert rows[0]["template_metadata"]["allow_localized_answer_labels"] is True
    source.write_bytes(b"wrong dataset")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.prepare(tmp_path / "bad.jsonl", languages=["hi"], source_path=source)
    assert not (tmp_path / "bad.jsonl").exists()


def test_dataset_access_failure_has_actionable_setup(monkeypatch):
    from httpx import Request, Response
    from huggingface_hub.errors import RepositoryNotFoundError

    def denied(**kwargs):
        raise RepositoryNotFoundError(
            "not authorized",
            response=Response(404, request=Request("GET", "https://huggingface.co/api/datasets/test")),
        )

    monkeypatch.setattr(module, "hf_hub_download", denied)
    with pytest.raises(RuntimeError, match="authentication alone does not grant access"):
        module.prepare(languages=["hi"])


def test_missing_local_source_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="source_path must be an existing"):
        module.prepare(languages=["hi"], source_path=tmp_path / "missing.parquet")


def test_manifest_matches_resolved_grading_mode(monkeypatch):
    from nemo_gym.environment import validation

    # Exercise real composition validation without requiring benchmark data in unit tests.
    monkeypatch.setattr(validation, "_validate_dataset", lambda *args, **kwargs: None)
    report = validation.validate_environment(module.DIRECTORY / "manifest.yaml")
    assert report.grading_mode == "lenient_answer_colon_md"
    assert report.synchronized_fields == ()
