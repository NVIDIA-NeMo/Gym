# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml

from benchmarks.indic.aime_2026 import prepare as module
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


def problem(*, index: int = 1) -> dict:
    return {"problem_idx": index, "answer": 472, "problem": " A synthetic sum?\nSecond line. "}


def source(*, index: int = 1) -> dict:
    return {
        **problem(index=index),
        **{
            f"problem_{module.LANGUAGE_NAMES[language]}_translation": f" {language} translated problem {index}?\n "
            for language in module.DEFAULT_LANGUAGES
        },
    }


@pytest.fixture
def source_dataset(monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_PROBLEMS", 2)
    rows, calls = [source(), source(index=2)], []

    def install(replacement):
        rows[:] = replacement

    def load(repo_id, *, split, revision):
        calls.append((repo_id, split, revision))
        return rows

    monkeypatch.setattr(module, "load_dataset", load)
    return install, calls


def test_prompt_and_row_schema_match_english_aime(source_dataset) -> None:
    _, calls = source_dataset
    records, metadata = module.load_source(languages=["hi", "en"])
    rows = module.build_rows(records)
    assert [row["question_id"] for row in rows] == ["1", "2", "1", "2"]
    assert len({row["uuid"] for row in rows}) == 4
    assert calls == [(module.SOURCE_ID, "train", module.SOURCE_REVISION)]
    assert metadata["source_rows"] == 2
    english_config = yaml.safe_load((module.BENCHMARK_DIR.parents[1] / "aime26/config.yaml").read_text())
    dataset = english_config["aime26_math_with_judge_simple_agent"]["responses_api_agents"]["simple_agent"][
        "datasets"
    ][0]
    prompt = load_prompt_config(str(module.BENCHMARK_DIR.parents[2] / dataset["prompt_config"]))
    for row, record in zip(rows, records, strict=True):
        assert row["question"] == record["problem"]
        assert row["expected_answer"] == "472"
        assert row["responses_create_params"]["max_output_tokens"] == 120000
        assert "judge_pass_stage" not in row
        messages = apply_prompt_to_row(row, prompt)["responses_create_params"]["input"]
        assert messages == [
            {
                "role": "user",
                "content": "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"
                + record["problem"],
            }
        ]
        assert "472" not in messages[0]["content"]
    assert rows[0]["question"] == source()["problem_Hindi_translation"]
    assert rows[2]["question"] == problem()["problem"]


def test_default_is_14_indic_languages_and_filter_is_stable(source_dataset) -> None:
    records, metadata = module.load_source()
    assert len(records) == 28
    assert set(metadata["source_configs"]) == set(module.DEFAULT_LANGUAGES)
    assert {"as", "sa"} <= set(metadata["source_configs"])
    assert "en" not in metadata["source_configs"]
    selected, filtered_metadata = module.load_source(languages=["en", "hi"], question_ids=[2])
    assert [row["language"] for row in selected] == ["en", "hi"]
    assert all(row["problem_idx"] == 2 for row in selected)
    assert filtered_metadata["question_ids"] == ["2"]
    assert metadata["coverage"]["hi"] == {"published_rows": 2, "selected_rows": 2}
    hindi, _ = module.load_source(config_name="hi", question_ids=["1"])
    assert len(hindi) == 1 and hindi[0]["question_id"] == "1"


@pytest.mark.parametrize("value", [None, "", " "])
def test_missing_translation_never_falls_back_to_english(source_dataset, value) -> None:
    install, _ = source_dataset
    install([{**source(), "problem_Hindi_translation": value}, source(index=2)])
    with pytest.raises(ValueError, match="fallback is forbidden"):
        module.load_source(languages=["hi"], question_ids=[2])


@pytest.mark.parametrize(
    "rows,error",
    [
        ([], "Empty"),
        ([None], "columns"),
        ([{**problem(), "extra": 1}], "columns"),
        ([{**problem(), "problem": None}], "Nonempty"),
        ([{**problem(), "problem_idx": True}], "positive integer"),
        ([{**problem(), "problem_idx": -1}], "positive integer"),
        ([{**problem(), "answer": "472"}], "integer answers"),
        ([{**problem(), "answer": True}], "integer answers"),
        ([{**problem(), "answer": -1}], "integer answers"),
        ([{**problem(), "answer": 1000}], "integer answers"),
        ([problem(), problem()], "Duplicate"),
    ],
)
def test_bad_schema_or_identity_is_rejected(rows, error) -> None:
    with pytest.raises(ValueError, match=error):
        module._index_rows(rows, language="hi")


@pytest.mark.parametrize(
    "rows,error",
    [
        ([], "Empty"),
        ([source()], "Expected AIME 2026 problem IDs"),
        ([source(), source(index=3)], "Expected AIME 2026 problem IDs"),
        ([source(), source()], "Duplicate"),
        ([{**source(), "extra": 1}, source(index=2)], "source columns"),
    ],
)
def test_invalid_source_is_rejected(source_dataset, rows, error) -> None:
    install, _ = source_dataset
    install(rows)
    with pytest.raises(ValueError, match=error):
        module.load_source(languages=["hi"])


def test_unknown_ids_rejected(source_dataset) -> None:
    with pytest.raises(ValueError, match="Unknown question IDs"):
        module.load_source(languages=["hi"], question_ids=["3"])


def test_prepare_manifest_and_no_overwrite_when_validation_fails(source_dataset, tmp_path, monkeypatch) -> None:
    install, _ = source_dataset
    output = tmp_path / "aime.jsonl"
    assert module.prepare(languages=["hi"], output_fpath=str(output)) == output
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["prepared_rows"] == 2
    assert manifest["prepared_sha256"] == module.sha256(output)
    assert manifest["evaluation_protocol"] == "gym_aime26"
    assert manifest["source_id"] == "ai4bharat/indic-aime-2026"
    assert manifest["source_license"] == "Apache-2.0"
    assert manifest["source_revision"] == module.SOURCE_REVISION
    original = output.read_bytes()
    install([{**source(), "answer": 1000}, source(index=2)])
    with pytest.raises(ValueError, match="integer answers"):
        module.prepare(languages=["hi"], output_fpath=str(output))
    assert output.read_bytes() == original
    install([source(), source(index=2)])
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "default.jsonl")
    assert module.prepare(config_name="hi").name == "default.jsonl"


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"languages": "hi"}, "sequence"),
        ({"languages": []}, "unique configuration"),
        ({"languages": ["hi", "hi"]}, "unique configuration"),
        ({"languages": [1]}, "unique configuration"),
        ({"languages": ["xx"]}, "Unsupported"),
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
    def unexpected(*args, **kwargs):
        pytest.fail("invalid selection must not download")

    monkeypatch.setattr(module, "load_dataset", unexpected)
    with pytest.raises(ValueError, match=error):
        module.load_source(**kwargs)


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
    assert manifest.datasets[0].num_repeats == 4
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
    environment_key = "indic_aime_2026_math_with_judge_environment_server"
    english_resource = english.aime26_math_with_judge_resources_server.resources_servers.math_with_judge
    indic_resource = indic[resource_key].resources_servers.math_with_judge
    assert indic_resource == english_resource
    assert indic_resource.should_use_judge is False
    assert {server.name for server in parser.filter_for_server_instance_configs(indic)} == {
        "policy_model",
        resource_key,
        agent_key,
        environment_key,
    }
    assert indic[environment_key].environment_servers.legacy_agent.agent_server.name == agent_key
    agent = indic[agent_key].responses_api_agents.simple_agent
    assert (
        agent.datasets[0].prompt_config
        == english.aime26_math_with_judge_simple_agent.responses_api_agents.simple_agent.datasets[0].prompt_config
    )
    assert agent.datasets[0].num_repeats == 4
    assert indic.get("num_repeats", 1) == 1 and indic.num_repeats_add_seed is True
    assert not indic.get("responses_create_params")
    policy = indic.policy_model.responses_api_models.vllm_model
    assert policy.chat_template_kwargs is None
    assert policy.sampling_overrides is None


def _resolve_cli_config(flags: list[str]) -> dict:
    from omegaconf import OmegaConf

    from nemo_gym.cli.main import _merge_config_paths, build_parser
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    args = build_parser().parse_args(
        [
            "eval",
            "run",
            "--benchmark",
            "indic/aime_2026",
            "--model-type",
            "vllm_model",
            "--model",
            "test",
            "--model-url",
            "http://unused/v1",
            "--model-api-key",
            "dummy",
            *flags,
        ]
    )
    overrides = _merge_config_paths([token for flag in args._command.flags for token in flag.translate_to_hydra(args)])
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.from_dotlist([token.lstrip("+") for token in overrides]),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.to_container(resolved, resolve=True)


def _materialize_dataset(config: dict, agent_name: str, source_rows: list[dict], tmp_path: Path) -> list[dict]:
    from nemo_gym.config_types import BenchmarkDatasetConfig
    from nemo_gym.train_data_utils import TrainDataProcessor

    dataset = dict(config[agent_name]["responses_api_agents"]["simple_agent"]["datasets"][0])
    source_path = tmp_path / f"{agent_name}.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    dataset["jsonl_fpath"] = str(source_path)
    prompt = load_prompt_config(dataset["prompt_config"])
    return [
        apply_prompt_to_row(json.loads(line), prompt) | {"agent_ref": {"name": agent_name}}
        for line in TrainDataProcessor._iter_dataset_lines(None, BenchmarkDatasetConfig.model_validate(dataset))
    ]


def _collect_rows(config: dict, source_rows: list[dict], tmp_path: Path) -> list[dict]:
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    collection = RolloutCollectionConfig.model_validate(
        config | {"input_jsonl_fpath": str(prepared), "output_jsonl_fpath": str(tmp_path / "out.jsonl")}
    )
    return RolloutCollectionHelper._preprocess_rows_from_config(None, collection)


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], (1.0, 0.95, 120000)),
        (["--temperature", "0.4", "--top-p", "0.8", "--max-output-tokens", "4096"], (0.4, 0.8, 4096)),
    ],
)
def test_cli_sampling_materializes_four_distinct_seeded_requests(flags, expected, tmp_path, source_dataset) -> None:
    config = _resolve_cli_config(flags)
    records, _ = module.load_source(languages=["hi"], question_ids=[1])
    source_rows = module.build_rows(records)
    prepared = _materialize_dataset(config, "indic_aime_2026_math_with_judge_simple_agent", source_rows, tmp_path)
    rows = _collect_rows(config, prepared, tmp_path)
    assert len(rows) == 4
    for seed, row in enumerate(rows):
        request = row["responses_create_params"]
        assert (request["temperature"], request["top_p"], request["max_output_tokens"]) == expected
        assert json.loads(request["metadata"]["extra_body"]) == {"top_k": 64, "seed": seed}
        assert json.loads(request["metadata"]["chat_template_kwargs"]) == {"enable_thinking": True}
    assert "seed" not in json.loads(source_rows[0]["responses_create_params"]["metadata"]["extra_body"])


@pytest.mark.parametrize("cli_override", [False, True])
def test_combined_benchmarks_preserve_numb3rs_audio_and_routing(tmp_path, source_dataset, cli_override) -> None:
    from benchmarks.numb3rs.prepare import _format_row

    flags = ["--benchmark", "numb3rs"]
    if cli_override:
        flags += ["--temperature", "0.4", "--top-p", "0.8", "--max-output-tokens", "4096"]
    config = _resolve_cli_config(flags)
    records, _ = module.load_source(languages=["hi"], question_ids=[1])
    indic_agent = "indic_aime_2026_math_with_judge_simple_agent"
    audio_agent = "numb3rs_asr_with_pc_simple_agent"
    indic = _materialize_dataset(config, indic_agent, module.build_rows(records), tmp_path)
    audio = _format_row(
        {"original_text": "forty two", "text": "42", "file_name": "sample.wav", "duration": 5, "category": "NUMBER"},
        audio_prefix=str(tmp_path),
    )
    audio_path = Path(audio["responses_create_params"]["metadata"]["audio_path"])
    prepared = indic + _materialize_dataset(config, audio_agent, [audio], tmp_path)
    rows = _collect_rows(config, prepared, tmp_path)
    assert len(rows) == 5
    assert [row["agent_ref"]["name"] for row in rows] == [indic_agent] * 4 + [audio_agent]
    audio_request = rows[-1]["responses_create_params"]
    assert audio_request["metadata"]["audio_path"] == str(audio_path)
    assert "chat_template_kwargs" not in audio_request["metadata"]
    assert "top_k" not in json.loads(audio_request["metadata"]["extra_body"])
    if cli_override:
        assert (audio_request["temperature"], audio_request["top_p"], audio_request["max_output_tokens"]) == (
            0.4,
            0.8,
            4096,
        )
    else:
        assert not {"temperature", "top_p", "max_output_tokens"} & audio_request.keys()
    assert [json.loads(row["responses_create_params"]["metadata"]["extra_body"])["seed"] for row in rows[:4]] == [
        0,
        1,
        2,
        3,
    ]
