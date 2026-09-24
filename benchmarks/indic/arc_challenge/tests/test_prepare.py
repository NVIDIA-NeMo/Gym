# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.indic.arc_challenge import prepare as module
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility


def source(
    task_id: str,
    question: str = "Which?",
    *,
    labels=None,
    texts=None,
    answer="B",
    language="English",
    language_code="en",
    judge_pass_stage="english_source",
):
    return {
        "id": task_id,
        "question": question,
        "choices": {
            "text": texts or ["one", "two", "three", "four"],
            "label": labels or ["A", "B", "C", "D"],
        },
        "answerKey": answer,
        "language": language,
        "language_code": language_code,
        "judge_pass_stage": judge_pass_stage,
    }


def canonical(row):
    return {key: row[key] for key in ("id", "question", "choices", "answerKey")}


def hindi(row, question=None):
    return row | {
        "question": question or row["question"],
        "language": "Hindi",
        "language_code": "hi",
        "judge_pass_stage": "first_judge_pass",
    }


def training(count=7):
    return [canonical(source(f"train-{index}", f" Demo {index} ", answer="ABCD"[index % 4])) for index in range(count)]


def test_exact_prompt_random_five_shot_and_subset_stability():
    english = [source("test-0", " Which one? "), source("test-1", "Next?")]
    records = [hindi(english[0], " कौन सा? "), hindi(english[1], "अगला?")]
    full = module.build_rows(records, english, training(), language="hi")
    subset = module.build_rows(records, english, training(), language="hi", question_ids=["test-1"])
    assert subset[0] == full[1]
    assert full[0]["metadata"]["fewshot_ids"] == ["train-5", "train-0", "train-6", "train-2", "train-4"]
    assert full[0]["prompt"] == (
        "Question: Demo 5\nA. one\nB. two\nC. three\nD. four\nAnswer: B\n\n"
        "Question: Demo 0\nA. one\nB. two\nC. three\nD. four\nAnswer: A\n\n"
        "Question: Demo 6\nA. one\nB. two\nC. three\nD. four\nAnswer: C\n\n"
        "Question: Demo 2\nA. one\nB. two\nC. three\nD. four\nAnswer: C\n\n"
        "Question: Demo 4\nA. one\nB. two\nC. three\nD. four\nAnswer: A\n\n"
        "Question: कौन सा?\nA. one\nB. two\nC. three\nD. four\nAnswer:"
    )
    assert full[0]["choices"] == [" A", " B", " C", " D"]
    assert full[0]["expected_answer"] == "B"
    prompt = load_prompt_config(str(module.DIRECTORY / "prompts/default.yaml"))
    validate_prompt_compatibility(full, prompt)
    materialized = apply_prompt_to_row(full[0], prompt)
    assert materialized["responses_create_params"]["input"] == [{"role": "user", "content": full[0]["prompt"]}]


@pytest.mark.parametrize(
    "labels,texts,answer,expected",
    [
        (["1", "2", "3"], [" one ", "two", "three"], "2", (["A", "B", "C"], "B")),
        (["A", "B", "C", "D", "E"], ["1", "2", "3", "4", "5"], "E", (list("ABCDE"), "E")),
    ],
)
def test_source_labels_and_variable_choice_counts_are_normalized(labels, texts, answer, expected):
    row = source("x", labels=labels, texts=texts, answer=answer)
    positional, stripped, gold = module._choices(row)
    assert (positional, gold) == expected
    assert stripped[0] == texts[0].strip()
    assert module.render_question(row).endswith(f"{expected[0][-1]}. {texts[-1].strip()}\nAnswer:")


@pytest.mark.parametrize("seed", [True, None, "42"])
def test_invalid_seed(seed):
    row = source("x")
    with pytest.raises(ValueError, match="fewshot_seed"):
        module.build_rows([row], [row], training(), language="en", fewshot_seed=seed)


@pytest.mark.parametrize("ids", [[], ["missing"], ["test-0", "test-0"], "test-0"])
def test_invalid_question_ids(ids):
    row = source("test-0")
    with pytest.raises(ValueError, match="question_ids"):
        module.build_rows([row], [row], training(), language="en", question_ids=ids)


@pytest.mark.parametrize(
    "mutation",
    [
        {"question": " "},
        {"answerKey": "Z"},
        {"choices": {"text": ["one"], "label": ["A"]}},
        {"language_code": "en"},
        {"judge_pass_stage": "unknown"},
    ],
)
def test_invalid_source_rows(mutation):
    english = source("test-0")
    record = hindi(english) | mutation
    with pytest.raises(ValueError, match="ARC"):
        module.build_rows([record], [english], training(), language="hi")


@pytest.fixture
def cached(tmp_path, monkeypatch):
    english = [source("test-0"), source("test-1", answer="C")]
    records = [hindi(english[0], "कौन?"), hindi(english[1], "अगला?")]
    train = training(5)
    files = {
        (module.CANONICAL_SOURCE_ID, module.CANONICAL_TEST_FILE): [canonical(row) for row in english],
        (module.CANONICAL_SOURCE_ID, module.CANONICAL_TRAIN_FILE): train,
        (module.SOURCE_ID, "data/en/test.parquet"): english,
        (module.SOURCE_ID, "data/hi/test.parquet"): records,
    }

    def download(*, repo_id, revision, filename, repo_type):
        assert repo_type == "dataset"
        expected_revision = module.SOURCE_REVISION if repo_id == module.SOURCE_ID else module.CANONICAL_SOURCE_REVISION
        assert revision == expected_revision
        path = tmp_path / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(files[(repo_id, filename)]), path)
        return path

    monkeypatch.setattr(module, "hf_hub_download", download)
    monkeypatch.setattr(module, "TEST_COUNT", 2)
    monkeypatch.setattr(module, "TRAIN_COUNT", 5)
    return english


def test_prepare_is_pinned_and_deterministic(cached, tmp_path):
    output = tmp_path / "arc.jsonl"
    module.prepare(output, languages=["hi"])
    first = output.read_bytes()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["metadata"]["language"] == "hi"
    assert rows[0]["metadata"]["fewshot_language"] == "en"
    assert rows[0]["metadata"]["source_revision"] == module.SOURCE_REVISION
    module.prepare(output, languages=["hi"])
    assert output.read_bytes() == first


@pytest.mark.parametrize("languages", ["hi", [], ["xx"], ["hi", "hi"]])
def test_invalid_languages(languages, tmp_path):
    with pytest.raises(ValueError, match="language"):
        module.prepare(tmp_path / "bad.jsonl", languages=languages)


@pytest.mark.parametrize("render_chat_template", [None, True, False])
def test_native_config_and_server_wiring(monkeypatch, render_chat_template):
    from omegaconf import OmegaConf

    import nemo_gym.global_config as global_config
    from nemo_gym.benchmarks import BenchmarkConfig, _benchmark_config_paths
    from nemo_gym.cli.main import _asset_config_path

    monkeypatch.setattr(global_config, "_find_open_port_using_range", lambda **_: 12345)
    path = module.DIRECTORY / "config.yaml"
    assert _asset_config_path("benchmark", "indic/arc_challenge") == str(path)
    benchmark = BenchmarkConfig.from_config_path(path, strict=False)
    assert benchmark.agent_name == "indic_arc_challenge_multiple_choice_agent"
    assert benchmark.num_repeats == 1
    assert _benchmark_config_paths(module.DIRECTORY) == [path]
    overrides = {
        "config_paths": [str(path)],
        "policy_base_url": "http://unused/v1",
        "policy_api_key": "dummy",
        "policy_model_name": "test",
    }
    if render_chat_template is not None:
        overrides["policy_model"] = {
            "responses_api_models": {"vllm_model": {"render_chat_template": render_chat_template}}
        }
    config = global_config.GlobalConfigDictParser().parse_no_environment(
        initial_global_config_dict=OmegaConf.create(overrides)
    )
    model = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert model["render_chat_template"] is (False if render_chat_template is None else render_chat_template)
    assert model["use_completions_api"] is True
    servers = global_config.GlobalConfigDictParser().filter_for_server_instance_configs(config)
    assert {server.name for server in servers} == {
        "policy_model",
        "indic_arc_challenge_mcqa_resources_server",
        "indic_arc_challenge_multiple_choice_agent",
    }
