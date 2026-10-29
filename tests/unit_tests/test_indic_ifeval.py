# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from benchmarks.indic.ifeval import prepare as module
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.data.task_data import find_server_dir, load_task_data_schema, validate_jsonl_rows
from nemo_gym.server_runtime import BaseRunRequest


def source_row(key=1, *, tags=None):
    return {
        "key": key,
        "prompt": "नमस्ते लिखिए।",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["नमस्ते"], "num_words": None}],
        "tags": tags if tags is not None else ["correct", "parallel"],
        "resp_lang": "src",
    }


def test_exact_languages_and_aliases():
    assert module.select_languages(None) == ["bn", "gu", "hi", "kn", "mr", "ml", "ne", "or", "pa", "ta", "te", "ur"]
    assert module.select_languages(["ka", "mar", "mal"]) == ["kn", "mr", "ml"]


@pytest.mark.parametrize("languages", [[], ["en"], ["as"], ["sa"], ["ka", "kn"], "hi"])
def test_reject_out_of_scope_or_duplicate_languages(languages):
    with pytest.raises(ValueError, match="language"):
        module.select_languages(languages)


def test_quality_selection_and_unmodified_source_fields():
    records = [source_row(1), source_row(2, tags=["incorrect"]), source_row(3, tags=["correct"])]
    original = copy.deepcopy(records)
    assert [r["id"] for r in module.build_rows(records, language="hi")] == [1, 3]
    assert [r["id"] for r in module.build_rows(records, language="hi", translation_quality="all")] == [1, 2, 3]
    rows = module.build_rows(records, language="hi", translation_quality="parallel")
    assert [r["id"] for r in rows] == [1]
    row = rows[0]
    parsed = BaseRunRequest.model_validate(row)
    assert parsed.responses_create_params.input[0].content == records[0]["prompt"]
    assert row["verifier_metadata"]["kwargs"] == records[0]["kwargs"]
    assert row["responses_create_params"]["input"] == [{"role": "user", "content": records[0]["prompt"]}]
    assert row["metadata"]["source_config"] == "indicifeval-trans"
    assert records == original
    assert row["uuid"] != module.build_rows(records, language="bn")[0]["uuid"]


def test_prepared_rows_match_instruction_following_task_data_schema():
    rows = [module.build_rows([source_row()], language=language)[0] for language in module.LANGUAGES]
    adapter = load_task_data_schema(find_server_dir("instruction_following"))
    report = validate_jsonl_rows("instruction_following", adapter, "prepared.jsonl", [json.dumps(row) for row in rows])

    assert report.rows == 12
    assert report.clean, report.summary()

    misplaced = copy.deepcopy(rows[0])
    misplaced["language"] = misplaced["verifier_metadata"].pop("language")
    report = validate_jsonl_rows("instruction_following", adapter, "misplaced.jsonl", [json.dumps(misplaced)])
    assert report.misplaced_keys == {"language": 1}


@pytest.mark.parametrize(
    "change", [{"tags": None}, {"tags": ["correct", "incorrect"]}, {"key": None}, {"kwargs": []}, {"prompt": ""}]
)
def test_invalid_source_fails_before_writing(change):
    record = source_row()
    record.update(change)
    with pytest.raises(ValueError):
        module.build_rows([record], language="hi")


def test_duplicate_keys_empty_selection_and_invalid_quality():
    with pytest.raises(ValueError, match="duplicate"):
        module.build_rows([source_row(), source_row()], language="hi")
    with pytest.raises(ValueError, match="no rows"):
        module.build_rows([source_row(tags=["incorrect"])], language="hi")
    with pytest.raises(ValueError, match="translation_quality"):
        module.build_rows([source_row()], language="hi", translation_quality="ground")


def test_prepare_downloads_trans_default_revision_and_round_trips(tmp_path, monkeypatch):
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([source_row(key) for key in range(3)]), parquet)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(parquet)

    monkeypatch.setattr(module, "hf_hub_download", download)
    path = module.prepare(tmp_path / "out.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 12 * 3
    assert len({row["uuid"] for row in rows}) == len(rows)
    assert all("source_revision" not in row["metadata"] for row in rows)
    assert {row["subset_for_metrics"] for row in rows} == set(module.LANGUAGES)
    assert calls == [
        {
            "repo_id": "ai4bharat/IndicIFEval",
            "repo_type": "dataset",
            "filename": f"indicifeval-trans/{language}-00000-of-00001.parquet",
        }
        for language in module.LANGUAGES
    ]


def test_config_reuses_english_server_and_simple_agent():
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(module.DIRECTORY / "config.yaml")]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    english = config["instruction_following"]["resources_servers"]["instruction_following"]
    assert english.get("instruction_backend", "english") == "english"
    assert "instruction_following_simple_agent" not in config
    server = config["indic_ifeval_instruction_following"]["resources_servers"]["instruction_following"]
    assert server["instruction_backend"] == "indicifeval_trans"
    assert server["entrypoint"] == "app.py"
    agent = config["indic_ifeval_instruction_following_simple_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["entrypoint"] == "app.py"
    assert agent["resources_server"]["name"] == "indic_ifeval_instruction_following"
    assert agent["model_server"]["name"] == "policy_model"
    assert [dataset["name"] for dataset in agent["datasets"]] == ["indic_ifeval_trans", "example"]
    assert agent["datasets"][0]["prepare_script"] == "benchmarks/indic/ifeval/prepare.py"


def test_manifest_matches_config_and_routes_to_indic_environment():
    from nemo_gym.environment.manifest import load_manifest

    manifest = load_manifest(module.DIRECTORY / "manifest.yaml")
    config = _resolve_cli_config([])
    agent = config["indic_ifeval_instruction_following_simple_agent"]["responses_api_agents"]["simple_agent"]
    dataset = agent["datasets"][0]
    assert manifest.canonical_split == dataset["type"] == "benchmark"
    assert manifest.prompt_source == "prepared"
    for declared, configured in zip(manifest.datasets, agent["datasets"], strict=True):
        for key in ("name", "type", "jsonl_fpath", "prepare_script", "num_repeats"):
            assert getattr(declared, key) == configured.get(key)
    assert dataset["num_repeats"] == 1
    environment = config["indic_ifeval_instruction_following_environment_server"]["environment_servers"][
        "legacy_agent"
    ]
    assert environment["agent_server"]["name"] == "indic_ifeval_instruction_following_simple_agent"
    assert "instruction_following_environment_server" not in config
    assert not config.get("responses_create_params")
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy.get("sampling_overrides") is None
    assert policy.get("chat_template_kwargs") is None


def _resolve_cli_config(flags):
    from nemo_gym.cli.main import _merge_config_paths, build_parser

    args = build_parser().parse_args(
        [
            "eval",
            "run",
            "--benchmark",
            "indic/ifeval",
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
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.from_dotlist([token.lstrip("+") for token in overrides]),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.to_container(config, resolve=True)


@pytest.mark.parametrize("override", [False, True])
def test_cli_defaults_and_overrides_preserve_other_benchmark_metadata(tmp_path, override):
    from benchmarks.numb3rs.prepare import _format_row
    from nemo_gym.data import apply_prompt_to_row, load_prompt_config
    from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper

    flags = ["--benchmark", "numb3rs"]
    if override:
        flags += ["--temperature", "0.6", "--top-p", "0.95", "--max-output-tokens", "32768"]
    config = _resolve_cli_config(flags)
    indic = module.build_rows([source_row()], language="hi")[0]
    indic["agent_ref"] = {"name": "indic_ifeval_instruction_following_simple_agent"}
    audio = _format_row(
        {"original_text": "42", "text": "forty two", "file_name": "sample.wav", "duration": 5, "category": "NUMBER"},
        audio_prefix=str(tmp_path),
    )
    audio = apply_prompt_to_row(audio, load_prompt_config("benchmarks/numb3rs/prompts/default.yaml"))
    audio["agent_ref"] = {"name": "numb3rs_asr_with_pc_simple_agent"}
    source = copy.deepcopy([indic, audio])
    collection = RolloutCollectionConfig.model_validate(
        config | {"input_jsonl_fpath": "unused", "output_jsonl_fpath": str(tmp_path / "out.jsonl")}
    )
    rows = RolloutCollectionHelper._preprocess_raw_rows(
        [(i, json.dumps(row), row.copy()) for i, row in enumerate(source)], collection
    )
    assert len(rows) == 2
    assert [row["agent_ref"] for row in rows] == [indic["agent_ref"], audio["agent_ref"]]
    request, audio_request = [row["responses_create_params"] for row in rows]
    assert audio_request["metadata"] == audio["responses_create_params"]["metadata"]
    assert request["input"] == indic["responses_create_params"]["input"]
    if override:
        for params in (request, audio_request):
            assert (params["temperature"], params["top_p"], params["max_output_tokens"]) == (0.6, 0.95, 32768)
    else:
        assert (request["temperature"], request["max_output_tokens"]) == (0.0, 1280)
        assert "top_p" not in request
        assert not {"temperature", "top_p", "max_output_tokens"} & audio_request.keys()
    assert source == [indic, audio]


def test_repeat_seeds_preserve_indic_sampling_and_source_metadata(tmp_path):
    from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper

    source = module.build_rows([source_row()], language="hi")[0]
    source["agent_ref"] = {"name": "indic_ifeval_instruction_following_simple_agent"}
    source["responses_create_params"]["metadata"] = {
        "extra_body": '{"top_k": 20}',
        "chat_template_kwargs": '{"enable_thinking": true}',
    }
    original = copy.deepcopy(source)
    config = RolloutCollectionConfig(
        input_jsonl_fpath="unused",
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        num_repeats=4,
        num_repeats_add_seed=True,
    )
    rows = RolloutCollectionHelper._preprocess_raw_rows([(0, json.dumps(source), source.copy())], config)
    assert len(rows) == 4
    for seed, row in enumerate(rows):
        params = row["responses_create_params"]
        assert (params["temperature"], params["max_output_tokens"]) == (0.0, 1280)
        assert json.loads(params["metadata"]["extra_body"]) == {"top_k": 20, "seed": seed}
        assert json.loads(params["metadata"]["chat_template_kwargs"]) == {"enable_thinking": True}
    assert source == original


def test_preparation_does_not_import_server_runtime():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.indic.ifeval.prepare; "
            "assert 'resources_servers.instruction_following.app' not in sys.modules; "
            "assert 'resources_servers.instruction_following.indicifeval' not in sys.modules; "
            "assert 'resources_servers.instruction_following.setup_indicifeval' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("english_first", [False, True])
def test_english_and_indic_keep_separate_verifiers(english_first):
    paths = ["benchmarks/indic/ifeval/config.yaml", "benchmarks/ifeval/config.yaml"]
    if english_first:
        paths.reverse()
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, {"config_paths": paths}
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    for prefix, expected_server, backend in (
        ("ifeval", "instruction_following", "english"),
        ("indic_ifeval", "indic_ifeval_instruction_following", "indicifeval_trans"),
    ):
        agent_name = f"{prefix}_instruction_following_simple_agent"
        agent = config[agent_name]["responses_api_agents"]["simple_agent"]
        assert agent["resources_server"]["name"] == expected_server
        verifier = config[expected_server]["resources_servers"]["instruction_following"]
        assert verifier.get("instruction_backend", "english") == backend
        env = config[f"{prefix}_instruction_following_environment_server"]["environment_servers"]["legacy_agent"]
        assert env["agent_server"]["name"] == agent_name
