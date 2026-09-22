# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import random

import pytest
import yaml

from benchmarks.gpqa import prepare as english_module
from benchmarks.indic.gpqa_diamond import prepare as module
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility


def record(index=0, language="en", **extra):
    return {
        "Question": f"Synthetic question {index}?",
        "Correct Answer": "right ",
        "Incorrect Answer 1": "one",
        "Incorrect Answer 2": "two",
        "Incorrect Answer 3": "three",
        "language": module.LANGUAGES[language],
        "language_code": language,
        "judge_pass_stage": "english_source" if language == "en" else "first_judge_pass",
        **extra,
    }


def build(records, **kwargs):
    return module.build_rows(
        records,
        language="en",
        canonical_ids=[f"synthetic-{index}" for index in range(len(records))],
        canonical_questions=[record(index)["Question"] for index in range(len(records))],
        **kwargs,
    )


def test_exact_shuffle_prompt_and_subset_stability(monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 3)
    records = [record(index) for index in range(3)]
    random_state = random.getstate()
    full = build(records)

    assert random.getstate() == random_state
    for index, row in enumerate(full):
        expected = [records[index][field] for field in module.TEXT_FIELDS[1:]]
        random.Random(int(hashlib.md5(records[index]["Question"].encode()).hexdigest(), 16)).shuffle(expected)
        assert row["options"] == [{letter: text} for letter, text in zip("ABCD", expected)]
        assert row["expected_answer"] == "ABCD"[expected.index(records[index]["Correct Answer"])]
        assert row["problem"] == records[index]["Question"] + "\n" + "\n".join(
            f"{letter}: {text}" for letter, text in zip("ABCD", expected)
        )
        english = english_module.build_row(records[index])
        assert {key: row[key] for key in english if key != "uuid"} == {
            key: value for key, value in english.items() if key != "uuid"
        }
    assert build(records, question_ids=["2"]) == full[2:]

    translated = [
        record(index, "hi", **{field: "अनुवाद " + records[index][field] for field in module.TEXT_FIELDS})
        for index in range(3)
    ]
    hindi = module.build_rows(
        translated,
        language="hi",
        canonical_ids=[f"synthetic-{index}" for index in range(3)],
        canonical_questions=[row["Question"] for row in records],
    )
    for original, translation in zip(full, hindi):
        assert original["expected_answer"] == translation["expected_answer"]
        assert translation["options"] == [
            {letter: "अनुवाद " + text for letter, text in option.items()} for option in original["options"]
        ]
    assert full[0]["uuid"] != hindi[0]["uuid"]
    assert hindi[0]["metadata"]["subset_for_metrics"] == "hi"

    prompt = load_prompt_config("benchmarks/prompts/eval/aai/mcq-4choices.yaml")
    validate_prompt_compatibility(full, prompt)
    rendered = apply_prompt_to_row(full[0], prompt)
    assert [message["role"] for message in rendered["responses_create_params"]["input"]] == ["user"]
    assert (
        rendered["responses_create_params"]["input"]
        == apply_prompt_to_row(english_module.build_row(records[0]), prompt)["responses_create_params"]["input"]
    )


def test_duplicate_text_behavior_and_row_69_are_upstream_compatible():
    records = [record(index) for index in range(198)]
    records[0]["Incorrect Answer 3"] = records[0]["Correct Answer"]
    rows = build(records)

    assert rows[0]["expected_answer"] == english_module.build_row(records[0])["expected_answer"]
    assert rows[0]["metadata"]["duplicate_choice_text"]
    assert rows[69]["metadata"]["task_id"] == "69"
    assert len(rows) == 198


@pytest.mark.parametrize("question_ids", [[], "0", [0], ["0", "0"], ["198"]])
def test_invalid_question_ids(question_ids):
    with pytest.raises(ValueError, match="question_ids"):
        build([record(index) for index in range(198)], question_ids=question_ids)


@pytest.mark.parametrize(
    "extra",
    [{"Question": ""}, {"unexpected": 1}, {"language": "Hindi"}, {"judge_pass_stage": "unknown"}],
)
def test_invalid_source_rows(extra):
    records = [record(index) for index in range(198)]
    records[0].update(extra)
    with pytest.raises(ValueError, match="GPQA"):
        build(records)


def test_prepare_uses_pins_and_checks_canonical_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    english = [record(index) for index in range(2)]
    hindi = [record(index, "hi") for index in range(2)]
    canonical = [
        {**{field: row[field] for field in module.TEXT_FIELDS}, "Record ID": f"record-{index}"}
        for index, row in enumerate(english)
    ]
    monkeypatch.setattr(module, "_read_canonical", lambda: canonical)

    calls = []

    def read_parquet(*, repo_id, revision, filename):
        calls.append((repo_id, revision, filename))
        return english if filename == "data/en/train.parquet" else hindi

    monkeypatch.setattr(module, "_read_parquet", read_parquet)
    output = module.prepare(tmp_path / "prepared.jsonl", languages=["hi"], question_ids=["1"])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["metadata"]["canonical_record_id"] == "record-1"
    assert calls == [
        (module.SOURCE_ID, module.SOURCE_REVISION, "data/en/train.parquet"),
        (module.SOURCE_ID, module.SOURCE_REVISION, "data/hi/train.parquet"),
    ]

    english[0]["Question"] = "changed"
    with pytest.raises(ValueError, match="differs from the pinned canonical"):
        module.prepare(tmp_path / "invalid.jsonl", languages=["hi"])


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
    assert set(config) == {"config_paths", "indic_gpqa_diamond_resources_server", "indic_gpqa_diamond_agent"}


def test_english_prepare_retains_row_format(tmp_path, monkeypatch):
    import datasets

    records = [record(index) for index in range(3)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: records)
    monkeypatch.setattr(english_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(english_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(english_module, "OUTPUT_FPATH", tmp_path / "english.jsonl")
    rows = [json.loads(line) for line in english_module.prepare().read_text().splitlines()]
    assert rows == [english_module.build_row(row) for row in records]


@pytest.mark.parametrize("questions", [[], [""], [None]])
def test_invalid_canonical_questions(monkeypatch, questions):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 1)
    with pytest.raises(ValueError, match="Canonical questions"):
        module.build_rows([record()], language="en", canonical_ids=["id"], canonical_questions=questions)
