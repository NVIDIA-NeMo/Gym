# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.indic.milu import prepare as module
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility


def source(question="कौन?", **extra):
    return {
        "question": question,
        "option1": "एक ",
        "option2": "दो",
        "option3": "तीन",
        "option4": "चार",
        "target": "option2",
        "domain": "Science",
        "subject": "Physics",
        "language": "Hindi",
        "is_translated": False,
        **extra,
    }


def test_exact_prompt_options_and_gold():
    row = module.build_rows([source(" \nकौन?\t ")], [], language="hi", num_fewshot=0)[0]
    assert row["prompt"] == r"कौन?\nA. एक \nB. दो\nC. तीन\nD. चार\nAnswer:"
    assert row["choices"] == [" A", " B", " C", " D"]
    assert row["expected_answer"] == "B"
    validate_prompt_compatibility([row], load_prompt_config(str(module.DIRECTORY / "prompts/default.yaml")))
    assert "input" not in row["responses_create_params"]


def test_synthetic_examples_materialize_once():
    rows = [json.loads(line) for line in (module.DIRECTORY / "data/example.jsonl").read_text().splitlines()]
    prompt = load_prompt_config(str(module.DIRECTORY / "prompts/default.yaml"))
    assert len(rows) == 5
    validate_prompt_compatibility(rows, prompt)
    for row in rows:
        materialized = apply_prompt_to_row(row, prompt)
        assert materialized["responses_create_params"]["input"] == [{"role": "user", "content": row["prompt"]}]
        assert row["metadata"]["synthetic"] is True
        assert row["prompt"].startswith("Synthetic format example:")
        assert r"\nA. " in row["prompt"]
        assert "Choices:" not in row["prompt"]
        assert row["choices"] == [" A", " B", " C", " D"]


def test_random_letter_demonstrations_and_stable_subset():
    tests = [source(f" test {i} ") for i in range(4)]
    validation = [source(f" demo {i} ", target=module.OPTION_FIELDS[i % 4]) for i in range(7)]
    full = module.build_rows(tests, validation, language="hi")
    subset = module.build_rows(tests, validation, language="hi", question_ids=["3"])
    assert subset[0] == full[3]
    assert full[0]["metadata"]["fewshot_ids"] == ["5", "0", "6", "2", "4"]
    assert subset[0]["metadata"]["fewshot_ids"] == ["1", "4", "5", "0", "2"]
    expected = (
        "".join(
            rf"demo {i}\nA. एक \nB. दो\nC. तीन\nD. चार\nAnswer: {answer}" + "\n\n"
            for i, answer in [(5, "B"), (0, "A"), (6, "C"), (2, "C"), (4, "A")]
        )
        + r"test 0\nA. एक \nB. दो\nC. तीन\nD. चार\nAnswer:"
    )
    assert full[0]["prompt"] == expected
    assert full[0]["choices"] == [" A", " B", " C", " D"]
    assert full[0]["metadata"]["num_fewshot"] == 5
    assert full[0]["metadata"]["fewshot_sampler"] == "default"
    assert full[0]["metadata"]["fewshot_seed"] == 42
    other_seed = module.build_rows(tests, validation, language="hi", fewshot_seed=43)
    assert other_seed[0]["metadata"]["fewshot_ids"] == ["0", "2", "1", "3", "4"]
    assert other_seed[0]["metadata"]["fewshot_seed"] == 43
    assert other_seed == module.build_rows(tests, validation, language="hi", fewshot_seed=43)


def test_random_demonstrations_exclude_the_test_document_without_replenishment():
    record = source("test question")
    validation = [record] + [source(f"demo {i}") for i in range(4)]
    row = module.build_rows([record], validation, language="hi")[0]
    assert row["metadata"]["fewshot_ids"] == ["4", "2", "1", "3"]
    assert row["prompt"].count("test question") == 1


@pytest.mark.parametrize("seed", [True, None, "42"])
def test_invalid_fewshot_seed(seed):
    with pytest.raises(ValueError, match="fewshot_seed"):
        module.build_rows([source()], [], language="hi", num_fewshot=0, fewshot_seed=seed)


@pytest.mark.parametrize("ids", [[], ["9"], ["0", "0"], "0", [0]])
def test_bad_question_selection(ids):
    with pytest.raises(ValueError, match="question_ids"):
        module.build_rows([source()], [], language="hi", num_fewshot=0, question_ids=ids)


@pytest.mark.parametrize(
    "field,value",
    [("target", "A"), ("option2", ""), ("language", "English"), ("is_translated", "false"), ("subject", None)],
)
def test_bad_source(field, value):
    with pytest.raises(ValueError, match="MILU"):
        module.build_rows([source(**{field: value})], [], language="hi", num_fewshot=0)


@pytest.mark.parametrize("shots", [-1, 1, True])
def test_bad_fewshot(shots):
    with pytest.raises(ValueError, match="num_fewshot"):
        module.build_rows([source()], [], language="hi", num_fewshot=shots)


@pytest.fixture
def cached(tmp_path, monkeypatch):
    def download(*, repo_id, revision, filename, repo_type):
        assert (repo_id, revision, repo_type) == (module.SOURCE_ID, module.SOURCE_REVISION, "dataset")
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 5 if filename.startswith("Hindi/validation") else 1
        pq.write_table(pa.Table.from_pylist([source(f"{filename} {i}") for i in range(count)]), path)
        return path

    monkeypatch.setattr(module, "hf_hub_download", download)
    monkeypatch.setattr(module, "LANGUAGES", {"hi": "Hindi"})
    monkeypatch.setattr(module, "COUNTS", {"hi": (1, 5)})


def test_native_entrypoint_is_deterministic_and_records_fewshot(cached, tmp_path):
    path = module.prepare(tmp_path / "result.jsonl")
    first = path.read_bytes()
    row = json.loads(first)
    assert row["expected_answer"] == "B"
    assert row["options"][1] == {"B": "दो"}
    assert row["metadata"]["language"] == "hi"
    assert row["metadata"]["source_revision"] == module.SOURCE_REVISION
    assert row["metadata"]["num_fewshot"] == 5
    assert row["metadata"]["fewshot_ids"] == ["0", "4", "2", "1", "3"]
    assert row["metadata"]["fewshot_seed"] == 42
    module.prepare(path)
    assert path.read_bytes() == first
    module.prepare(path, num_fewshot=1, fewshot_seed=7)
    assert json.loads(path.read_text())["metadata"]["fewshot_ids"] == ["2"]
    assert json.loads(path.read_text())["metadata"]["fewshot_seed"] == 7
    module.prepare(path, num_fewshot=0)
    assert json.loads(path.read_text())["metadata"]["fewshot_ids"] == []


@pytest.mark.parametrize("languages", ["hi", [], ["as"], ["hi", "hi"]])
def test_invalid_languages(languages, tmp_path):
    with pytest.raises(ValueError, match="language"):
        module.prepare(tmp_path / "bad.jsonl", languages=languages)


def test_prepare_guards_counts_shots_and_cli(cached, tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError, match="nonnegative"):
        module.prepare(tmp_path / "bad.jsonl", num_fewshot=-1)
    monkeypatch.setattr(module, "COUNTS", {"hi": (2, 1)})
    with pytest.raises(ValueError, match="expected 2"):
        module.prepare(tmp_path / "bad.jsonl")
    monkeypatch.setattr(module, "COUNTS", {"hi": (1, 5)})
    monkeypatch.setattr(
        "sys.argv",
        ["prepare", "--languages", "hi", "--fewshot-seed", "7", "--output-path", str(tmp_path / "cli.jsonl")],
    )
    module.main()
    assert "cli.jsonl" in capsys.readouterr().out
    metadata = json.loads((tmp_path / "cli.jsonl").read_text())["metadata"]
    assert metadata["num_fewshot"] == 5
    assert metadata["fewshot_seed"] == 7


@pytest.mark.parametrize("render_chat_template", [None, True, False])
def test_native_config_and_server_wiring(monkeypatch, render_chat_template):
    from omegaconf import OmegaConf

    import nemo_gym.global_config as global_config
    from nemo_gym.benchmarks import BenchmarkConfig, _benchmark_config_paths
    from nemo_gym.cli.main import _asset_config_path

    monkeypatch.setattr(global_config, "_find_open_port_using_range", lambda **_: 12345)
    path = module.DIRECTORY / "config.yaml"
    assert _asset_config_path("benchmark", "indic/milu") == str(path)
    benchmark = BenchmarkConfig.from_config_path(path, strict=False)
    assert benchmark.agent_name == "milu_multiple_choice_agent"
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
    assert model["chat_template_kwargs"] is None
    servers = global_config.GlobalConfigDictParser().filter_for_server_instance_configs(config)
    assert {server.name for server in servers} == {
        "policy_model",
        "milu_mcqa_resources_server",
        "milu_multiple_choice_agent",
    }
