# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from benchmarks.indic.aime_2026 import prepare as module
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


def canonical(*, index: int = 1) -> dict:
    return {"problem_idx": index, "answer": 472, "problem": " A synthetic sum?\nSecond line. "}


def source(*, language: str = "hi", index: int = 1) -> dict:
    return {
        **canonical(index=index),
        "problem": canonical()["problem"] if language == "en" else " एक काल्पनिक योग?\nदूसरी पंक्ति। ",
        "language": module.LANGUAGE_NAMES[language],
        "language_code": language,
        "judge_pass_stage": "english_source" if language == "en" else "first_judge_pass",
    }


@pytest.fixture
def source_files(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ENGLISH_ROWS", 2)
    files, calls = {}, []

    def install(language, rows):
        path = tmp_path / f"{language or 'canonical'}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        if language is None:
            files[(module.CANONICAL_REPO, module.CANONICAL_REVISION, module.CANONICAL_FILE)] = path
        else:
            files[(module.SOURCE_ID, module.SOURCE_REVISION, f"data/{language}/train.parquet")] = path
        return path

    install(None, [canonical(index=2), canonical()])
    for language in ["en", *module.DEFAULT_LANGUAGES]:
        install(language, [source(language=language, index=2), source(language=language)])

    def download(*args):
        calls.append(args)
        return files[args]

    monkeypatch.setattr(module, "download_hf_file", download)
    return install, files, calls


def test_prompt_and_row_schema_match_english_aime(source_files) -> None:
    _, files, calls = source_files
    records, metadata = module.load_source(languages=["hi", "en"])
    rows = module.build_rows(records)
    assert [row["question_id"] for row in rows] == ["1", "2", "1", "2"]
    assert len({row["uuid"] for row in rows}) == 4
    assert len(calls) == 3
    assert metadata["canonical_file_sha256"] == module.sha256(
        files[(module.CANONICAL_REPO, module.CANONICAL_REVISION, module.CANONICAL_FILE)]
    )
    english_config = yaml.safe_load((module.BENCHMARK_DIR.parents[1] / "aime26/config.yaml").read_text())
    dataset = english_config["aime26_math_with_judge_simple_agent"]["responses_api_agents"]["simple_agent"][
        "datasets"
    ][0]
    prompt = load_prompt_config(str(module.BENCHMARK_DIR.parents[2] / dataset["prompt_config"]))
    for row, record in zip(rows, records, strict=True):
        assert row["question"] == record["problem"]
        assert row["expected_answer"] == "472"
        assert "responses_create_params" not in row
        assert "expected_groups" not in row
        messages = apply_prompt_to_row(row, prompt)["responses_create_params"]["input"]
        assert messages == [
            {
                "role": "user",
                "content": "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"
                + record["problem"],
            }
        ]
        assert "472" not in messages[0]["content"]


def test_default_is_12_indic_configs_english_is_explicit_and_filter_is_stable(source_files) -> None:
    records, metadata = module.load_source()
    assert len(records) == 24
    assert set(metadata["source_configs"]) == set(module.DEFAULT_LANGUAGES)
    assert "as" not in metadata["source_configs"] and "sa" not in metadata["source_configs"]
    selected, filtered_metadata = module.load_source(languages=["en", "hi"], question_ids=[2])
    assert [row["language"] for row in selected] == ["en", "hi"]
    assert all(row["problem_idx"] == 2 for row in selected)
    assert filtered_metadata["question_ids"] == ["2"]
    english, _ = module.load_source(config_name="en", languages=["en"])
    assert len(english) == 2
    hindi, _ = module.load_source(config_name="hi", question_ids=["1"])
    assert len(hindi) == 1 and hindi[0]["question_id"] == "1"


def test_missing_translations_and_review_flags_are_reported_never_replaced(source_files) -> None:
    install, _, _ = source_files
    install("or", [{**source(language="or"), "judge_pass_stage": "failed_after_correction_review_needed"}])
    records, metadata = module.load_source(languages=["hi", "or"])
    assert len(records) == 3
    assert metadata["coverage"]["or"] == {
        "published_rows": 1,
        "selected_rows": 1,
        "missing_english_ids": ["2"],
        "missing_selected_ids": ["2"],
        "judge_pass_stage": {"failed_after_correction_review_needed": 1},
        "human_evaluation_pending": 1,
    }
    assert module.build_rows(records)[-1]["human_evaluation_pending"] is True
    with pytest.raises(ValueError, match="no published translated rows"):
        module.load_source(languages=["or"], question_ids=["2"])


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"languages": "hi"}, "sequence"),
        ({"languages": []}, "unique configuration"),
        ({"languages": ["hi", "hi"]}, "unique configuration"),
        ({"languages": [1]}, "unique configuration"),
        ({"languages": ["xx"]}, "Unsupported"),
        ({"languages": ["as"]}, "Unsupported"),
        ({"languages": ["sa"]}, "Unsupported"),
        ({"languages": ["hi"], "config_name": "bn"}, "conflicts"),
        ({"question_ids": "1"}, "unique positive"),
        ({"question_ids": []}, "unique positive"),
        ({"question_ids": [True]}, "positive decimal"),
        ({"question_ids": [1.0]}, "positive decimal"),
        ({"question_ids": [""]}, "positive decimal"),
        ({"question_ids": [" 1"]}, "positive decimal"),
        ({"question_ids": ["01"]}, "positive decimal"),
        ({"question_ids": ["१"]}, "positive decimal"),
        ({"question_ids": ["0"]}, "positive decimal"),
        ({"question_ids": [0]}, "positive decimal"),
        ({"question_ids": [-1]}, "positive decimal"),
        ({"question_ids": ["id"]}, "positive decimal"),
        ({"question_ids": ["1", 1]}, "unique positive"),
    ],
)
def test_invalid_selection_fails_before_download(monkeypatch, kwargs, error) -> None:
    def unexpected(*args):
        pytest.fail("invalid selection must not download")

    monkeypatch.setattr(module, "download_hf_file", unexpected)
    with pytest.raises(ValueError, match=error):
        module.load_source(**kwargs)


@pytest.mark.parametrize(
    "rows,language,error",
    [
        ([], "hi", "Empty"),
        ([None], "hi", "columns"),
        ([{**source(), "extra": "unknown"}], "hi", "columns"),
        ([{**source(), "problem": None}], "hi", "Nonempty translated problem"),
        ([{**source(), "problem": " "}], "hi", "fallback is forbidden"),
        ([{**source(), "problem_idx": True}], "hi", "positive integer"),
        ([{**source(), "problem_idx": -1}], "hi", "positive integer"),
        ([{**source(), "answer": "472"}], "hi", "integer answers"),
        ([{**source(), "answer": True}], "hi", "integer answers"),
        ([{**source(), "answer": -1}], "hi", "integer answers"),
        ([{**source(), "answer": 1000}], "hi", "integer answers"),
        ([source(), source()], "hi", "Duplicate"),
        ([{**source(), "language_code": "en"}], "hi", "Language metadata"),
        ([{**source(), "language": "English"}], "hi", "Language metadata"),
        ([{**source(), "judge_pass_stage": []}], "hi", "quality stage"),
        ([{**source(), "judge_pass_stage": "new"}], "hi", "quality stage"),
        ([{**source(), "judge_pass_stage": "english_source"}], "hi", "quality stage"),
        ([{**source(language="en"), "judge_pass_stage": "first_judge_pass"}], "en", "quality stage"),
    ],
)
def test_bad_schema_or_identity_is_rejected(rows, language, error) -> None:
    with pytest.raises(ValueError, match=error):
        module._index_rows(rows, language=language)


@pytest.mark.parametrize(
    "language,rows,error",
    [
        (None, [canonical()], "Expected canonical.*1 through 2"),
        (None, [canonical(), canonical(index=3)], "Expected canonical.*1 through 2"),
        ("en", [source(language="en")], "every canonical"),
        ("hi", [source(index=3)], "absent from canonical"),
        ("hi", [{**source(), "answer": 473}], "Canonical field mismatch"),
        ("en", [{**source(language="en"), "problem": "changed"}, source(language="en", index=2)], "field mismatch"),
    ],
)
def test_misaligned_source_fails_closed(source_files, language, rows, error) -> None:
    install, _, _ = source_files
    install(language, rows)
    with pytest.raises(ValueError, match=error):
        module.load_source(languages=["hi"])


def test_unknown_ids_rejected_even_when_excluded_from_translations(source_files) -> None:
    with pytest.raises(ValueError, match="Unknown question IDs"):
        module.load_source(languages=["hi"], question_ids=["3"])


def test_numeric_problem_order_not_lexicographic() -> None:
    selected, identities = module._selection(["hi"], None, [10, "2", 1])
    assert selected == ["hi"] and identities == ["10", "2", "1"]


def test_prepare_manifest_and_no_overwrite_when_validation_fails(source_files, tmp_path, monkeypatch) -> None:
    install, _, _ = source_files
    output = tmp_path / "aime.jsonl"
    result = module.prepare(languages=["hi"], output_fpath=str(output))
    assert result == output
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["prepared_rows"] == 2
    assert manifest["prepared_sha256"] == module.sha256(output)
    assert manifest["evaluation_protocol"] == "gym_aime26"
    assert manifest["protocol_version"] == 2
    assert manifest["max_output_tokens"] == 120000
    assert manifest["thinking_enabled_by_default"] is True
    assert manifest["source_license"] == "CC-BY-NC-SA-4.0"
    assert manifest["prompt_mode"] == "gym_generic_math"
    original = output.read_bytes()
    install("hi", [{**source(), "answer": 473}])
    with pytest.raises(ValueError, match="Canonical field mismatch"):
        module.prepare(languages=["hi"], output_fpath=str(output))
    assert output.read_bytes() == original
    install("hi", [source()])
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "default.jsonl")
    assert module.prepare(config_name="hi").name == "default.jsonl"


def test_cli(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("sys.argv", ["prepare", "--languages", "hi", "--question-ids", "1", "--output-fpath", "x"])
    monkeypatch.setattr(module, "prepare", lambda **kwargs: captured.update(kwargs) or Path("out.jsonl"))
    module.main()
    assert captured["languages"] == ["hi"]
    assert captured["question_ids"] == ["1"]
    assert captured["output_fpath"] == "x"


def test_native_config_reuses_english_components_and_preserves_generation_defaults() -> None:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    manifest = load_manifest(module.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.resources_server == "math_with_judge"
    assert manifest.agent_server == "simple_agent"
    assert manifest.standard_prompt_config == "benchmarks/prompts/generic/math.yaml"
    assert manifest.datasets[0].num_repeats == 1
    configs = []
    parser = GlobalConfigDictParser()
    for path in (module.BENCHMARK_DIR.parents[1] / "aime26/config.yaml", module.BENCHMARK_DIR / "config.yaml"):
        configs.append(
            parser.parse(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=OmegaConf.create(
                        {
                            "config_paths": ["responses_api_models/vllm_model/configs/vllm_model.yaml", str(path)],
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
    english, indic = configs
    resource_key = "indic_aime_2026_math_with_judge_resources_server"
    agent_key = "indic_aime_2026_math_with_judge_simple_agent"
    english_resource = english.aime26_math_with_judge_resources_server.resources_servers.math_with_judge
    indic_resource = indic[resource_key].resources_servers.math_with_judge
    assert indic_resource == english_resource
    assert indic_resource.should_use_judge is False
    assert {server.name for server in parser.filter_for_server_instance_configs(indic)} == {
        "policy_model",
        resource_key,
        agent_key,
    }
    agent = indic[agent_key].responses_api_agents.simple_agent
    assert (
        agent.datasets[0].prompt_config
        == english.aime26_math_with_judge_simple_agent.responses_api_agents.simple_agent.datasets[0].prompt_config
    )
    assert agent.datasets[0].num_repeats == 1
    assert indic.num_repeats == 4 and indic.num_repeats_add_seed is True
    assert indic.responses_create_params.max_output_tokens == 120000
    policy = indic.policy_model.responses_api_models.vllm_model
    assert policy.chat_template_kwargs.enable_thinking is True
    assert policy.sampling_overrides == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
