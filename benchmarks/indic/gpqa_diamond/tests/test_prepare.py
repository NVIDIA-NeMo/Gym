# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random

import pytest
import yaml

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
        **kwargs,
    )


def test_exact_shuffle_prompt_and_subset_stability(monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 3)
    records = [record(index) for index in range(3)]
    random_state = random.getstate()
    full = build(records)

    assert random.getstate() == random_state
    assert full[0]["options"] == [{"A": "three"}, {"B": "one"}, {"C": "two"}, {"D": "right "}]
    assert full[0]["expected_answer"] == "D"
    assert full[0]["prompt"] == (
        "What is the correct answer to this question: Synthetic question 0?\n\nChoices:\n"
        '(A) three\n(B) one\n(C) two\n(D) right \n\nFormat your response as follows: "The correct answer is (insert answer here)"'
    )
    assert build(records, question_ids=["2"]) == full[2:]
    assert build(records, shuffle_seed=7)[0]["options"] != full[0]["options"]

    hindi = module.build_rows(
        [record(index, "hi") for index in range(3)],
        language="hi",
        canonical_ids=[f"synthetic-{index}" for index in range(3)],
    )
    assert [row["expected_answer"] for row in full] == [row["expected_answer"] for row in hindi]
    assert full[0]["uuid"] != hindi[0]["uuid"]
    assert hindi[0]["metadata"]["subset_for_metrics"] == "hi"

    prompt = load_prompt_config(str(module.DIRECTORY / "prompts/default.yaml"))
    validate_prompt_compatibility(full, prompt)
    rendered = apply_prompt_to_row(full[0], prompt)
    assert [message["role"] for message in rendered["responses_create_params"]["input"]] == ["system", "user"]


def test_duplicate_text_behavior_and_row_69_are_upstream_compatible():
    records = [record(index) for index in range(198)]
    records[0]["Incorrect Answer 3"] = records[0]["Correct Answer"]
    rows = build(records)

    assert rows[0]["expected_answer"] == "A"
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


def test_config_uses_official_profile_and_deterministic_chat_defaults():
    config = yaml.safe_load((module.DIRECTORY / "config.yaml").read_text(encoding="utf-8"))
    model = config["policy_model"]["responses_api_models"]["vllm_model"]
    verifier = config["indic_gpqa_diamond_resources_server"]["resources_servers"]["gpqa_diamond"]
    dataset = config["indic_gpqa_diamond_agent"]["responses_api_agents"]["simple_agent"]["datasets"][0]

    assert config["num_repeats"] == dataset["num_repeats"] == 1
    assert config["responses_create_params"]["max_output_tokens"] == 1000
    assert model["chat_template_kwargs"] == {"enable_thinking": False}
    assert model["sampling_overrides"] == {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 0}
    assert verifier["use_official_parser"] is True
