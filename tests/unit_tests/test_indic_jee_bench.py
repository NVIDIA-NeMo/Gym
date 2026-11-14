# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


prep = importlib.import_module("benchmarks.indic.jee_bench.prepare")


@pytest.fixture
def source(monkeypatch):
    english = [
        {
            "description": f"Paper {index // 50}",
            "index": index % 50 + 1,
            "subject": "math",
            "type": "MCQ",
            "gold": "B",
            "question": "English\n\n(A) 1\n(B) 2",
            "language": "English",
            "language_code": "en",
            "judge_pass_stage": "english_source",
        }
        for index in range(515)
    ]
    configs = {"en": english}
    for lang in prep.DEFAULT_LANGUAGES:
        configs[lang] = [
            {
                **row,
                "question": "अनुवाद\n\n(A) 1\n(B) 2",
                "language": prep.LANGUAGES[lang],
                "language_code": lang,
                "judge_pass_stage": "first_judge_pass",
            }
            for row in english
        ]
    configs["wide"] = [
        {
            **{key: row[key] for key in (*prep.ALIGNMENT_FIELDS, "question")},
            **{
                f"question_{prep.LANGUAGES[lang]}_translation": configs[lang][i]["question"]
                for lang in prep.DEFAULT_LANGUAGES
            },
        }
        for i, row in enumerate(english)
    ]
    calls = []

    def load(repo, **kwargs):
        calls.append((repo, kwargs))
        return deepcopy(configs["wide"])

    monkeypatch.setattr(prep, "load_dataset", load)
    return configs, calls


def test_default_languages_and_ids_are_unique(source, tmp_path):
    configs, calls = source
    configs["wide"][-1]["question_Odia_translation"] = None
    path = prep.prepare(output_fpath=str(tmp_path / "rows.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert len(rows) == 515 * 14 - 1
    assert len({row["uuid"] for row in rows}) == len(rows)
    assert {row["language"] for row in rows} == set(prep.DEFAULT_LANGUAGES)
    assert manifest["coverage"]["or"]["missing_english_ids"] == [prep.question_id(configs["en"][-1])]
    assert all(row["judge_pass_stage"] is None for row in rows)
    assert {row["source_id"] for row in rows} == {"ai4bharat/indic-jee-bench"}
    assert calls == [(prep.SOURCE_ID, {"split": "test", "revision": prep.SOURCE_REVISION})]


def test_english_and_smoke_selection_preserve_prompt_and_gold(source, tmp_path):
    configs, _ = source
    wanted = prep.question_id(configs["en"][0])
    path = prep.prepare(languages=["hi", "en"], question_ids=[wanted], output_fpath=str(tmp_path / "rows.jsonl"))
    hindi, english = [json.loads(line) for line in path.read_text().splitlines()]
    assert hindi["instruction"] == (
        "In this problem, only one option will be correct. Give a detailed solution and end the solution "
        "with the final answer.\n\nProblem: अनुवाद\n(A) 1\n(B) 2\nSolution: Let's think step by step."
    )
    assert hindi["verifier_metadata"]["gold"] == english["verifier_metadata"]["gold"] == "B"
    assert "English" in english["instruction"]
    assert "verifier_metadata" not in hindi["instruction"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"languages": "hi"},
        {"languages": []},
        {"languages": ["hi", "hi"]},
        {"languages": ["xx"]},
        {"question_ids": []},
        {"question_ids": "1"},
        {"question_ids": ["x", "x"]},
        {"question_ids": ["missing/1"]},
    ],
)
def test_invalid_selection(source, tmp_path, kwargs):
    with pytest.raises(ValueError):
        prep.prepare(output_fpath=str(tmp_path / "rows.jsonl"), **kwargs)
    assert not (tmp_path / "rows.jsonl").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("gold", "Z"),
        ("question_Hindi_translation", ""),
        ("question_Hindi_translation", 42),
        ("question", ""),
        ("subject", "invalid"),
        ("type", "unknown"),
        ("index", -1),
        ("gold", "AB"),
    ],
)
def test_bad_translation_or_invalid_shared_metadata_rejected(source, tmp_path, field, value):
    configs, _ = source
    configs["wide"][0][field] = value
    with pytest.raises(ValueError):
        prep.prepare(languages=["hi"], output_fpath=str(tmp_path / "rows.jsonl"))


def test_missing_requested_translation_and_duplicate_rejected(source, tmp_path):
    configs, _ = source
    missing = configs["wide"][-1]
    missing["question_Hindi_translation"] = None
    with pytest.raises(ValueError, match="Requested translations missing"):
        prep.prepare(languages=["hi"], question_ids=[prep.question_id(missing)])
    configs["wide"].append(configs["wide"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        prep.prepare(languages=["hi"])
    configs["wide"].pop()
    configs["wide"].pop()
    with pytest.raises(ValueError, match="515"):
        prep.prepare(languages=["hi"])


def test_local_snapshot_uses_same_conversion(source, monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    configs, _ = source
    source_path = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist(configs["wide"]), source_path)
    remote = prep.prepare(languages=["hi"], output_fpath=str(tmp_path / "remote.jsonl"))
    # Restore a real loader without requiring Hub access.
    monkeypatch.setattr(prep, "load_dataset", load_dataset)
    path = prep.prepare(languages=["hi"], source_dir=str(tmp_path), output_fpath=str(tmp_path / "rows.jsonl"))
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert manifest["prepared_rows"] == 515
    assert path.read_bytes() == remote.read_bytes()
    assert manifest["source_file_sha256"] == {"test.parquet": hashlib.sha256(source_path.read_bytes()).hexdigest()}


def test_missing_translation_column_rejected(source, tmp_path):
    configs, _ = source
    del configs["wide"][0]["question_Hindi_translation"]
    with pytest.raises(ValueError, match="Missing source column: question_Hindi_translation"):
        prep.prepare(languages=["hi"], output_fpath=str(tmp_path / "rows.jsonl"))
    assert not (tmp_path / "rows.jsonl").exists()


def test_other_question_types_and_invalid_gold(source):
    configs, _ = source
    base = configs["en"][0]
    for kind, gold in [("MCQ(multiple)", "AC"), ("Integer", "3"), ("Numeric", "-1.23")]:
        row = {**base, "type": kind, "gold": gold}
        assert prep._index_rows([row], "en")
        prepared = prep.build_rows([row], language="en", revision="fixture")[0]
        assert prepared["verifier_metadata"]["question_type"] == kind
    for kind, gold in [("MCQ", "Z"), ("Numeric", "nan"), ("Integer", "2.5")]:
        with pytest.raises(ValueError):
            prep._index_rows([{**base, "type": kind, "gold": gold}], "en")
    with pytest.raises(ValueError, match="Empty"):
        prep._index_rows([], "en")


def test_native_hydra_list_arguments(source, tmp_path):
    from omegaconf import OmegaConf

    args = OmegaConf.create({"languages": ["hi"], "question_ids": ["Paper 0/1"]})
    path = prep.prepare(**args, output_fpath=str(tmp_path / "rows.jsonl"))
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert manifest["languages"] == ["hi"]
    assert manifest["question_ids"] == ["Paper 0/1"]
    assert manifest["prepared_rows"] == 1


def test_manifest_and_native_selector():
    from nemo_gym.benchmarks import _is_benchmark_config
    from nemo_gym.cli.main import _asset_config_path
    from nemo_gym.environment.manifest import load_manifest

    manifest = load_manifest(prep.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.name == "indic/jee_bench"
    assert manifest.resources_server == "jeebench"
    config = prep.BENCHMARK_DIR / "config.yaml"
    assert _is_benchmark_config(config)
    assert _asset_config_path("benchmark", manifest.name) == str(config.resolve())


def _resolve_cli_config(flags: list[str]) -> dict:
    from omegaconf import OmegaConf

    from nemo_gym.cli.main import _merge_config_paths, build_parser
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    args = build_parser().parse_args(
        [
            "eval",
            "run",
            "--benchmark",
            "indic/jee_bench",
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


def _materialize_dataset(config: dict, agent_name: str, rows: list[dict], tmp_path: Path) -> list[dict]:
    from nemo_gym.config_types import BenchmarkDatasetConfig
    from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config
    from nemo_gym.train_data_utils import TrainDataProcessor

    dataset = dict(config[agent_name]["responses_api_agents"]["simple_agent"]["datasets"][0])
    source_path = tmp_path / f"{agent_name}.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset["jsonl_fpath"] = str(source_path)
    prompt = load_prompt_config(dataset["prompt_config"])
    return [
        apply_prompt_to_row(json.loads(line), prompt) | {"agent_ref": {"name": agent_name}}
        for line in TrainDataProcessor._iter_dataset_lines(None, BenchmarkDatasetConfig.model_validate(dataset))
    ]


def _collect_rows(config: dict, rows: list[dict], tmp_path: Path) -> list[dict]:
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in rows))
    collection = RolloutCollectionConfig.model_validate(
        config | {"input_jsonl_fpath": str(prepared), "output_jsonl_fpath": str(tmp_path / "out.jsonl")}
    )
    return RolloutCollectionHelper._preprocess_rows_from_config(None, collection)


def test_native_config_and_manifest_agree():
    from nemo_gym.environment.manifest import load_manifest

    config = _resolve_cli_config([])
    manifest = load_manifest(prep.BENCHMARK_DIR / "manifest.yaml")
    agent = config["indic_jee_bench_simple_agent"]["responses_api_agents"]["simple_agent"]
    dataset = agent["datasets"][0]
    assert manifest.canonical_split == dataset["type"] == "benchmark"
    assert manifest.datasets[0].num_repeats == dataset["num_repeats"] == 1
    assert manifest.standard_prompt_config == dataset["prompt_config"]
    assert agent["resources_server"]["name"] == "indic_jee_bench_resources_server"
    environment = config["indic_jee_bench_environment_server"]["environment_servers"]["legacy_agent"]
    assert environment["agent_server"] == {"type": "responses_api_agents", "name": "indic_jee_bench_simple_agent"}
    assert not config.get("responses_create_params")


def test_prepared_and_example_rows_match_task_schema(source):
    from nemo_gym.task_data import TaskDataValidator, load_task_data_schema

    configs, _ = source
    server_dir = prep.BENCHMARK_DIR.parents[2] / "resources_servers/jeebench"
    validator = TaskDataValidator(
        server_name="jeebench", adapter=load_task_data_schema(server_dir), dataset_fpath="fixture"
    )
    rows = prep.build_rows(configs["hi"][:1], language="hi", revision="fixture")
    rows += [json.loads(line) for line in (server_dir / "data/example.jsonl").read_text().splitlines()]
    for index, row in enumerate(rows):
        validator.validate_row(index, row)
    assert validator.report.clean, validator.report.summary()
    assert validator.report.rows == 6


@pytest.mark.parametrize("failure", ["missing_gold", "wrong_type", "misplaced_gold"])
def test_task_schema_rejects_missing_malformed_or_misplaced_grading_fields(source, failure):
    from nemo_gym.task_data import TaskDataValidator, load_task_data_schema

    configs, _ = source
    row = prep.build_rows(configs["hi"][:1], language="hi", revision="fixture")[0]
    if failure == "wrong_type":
        row["verifier_metadata"]["question_type"] = "unsupported"
    else:
        gold = row["verifier_metadata"].pop("gold")
        if failure == "misplaced_gold":
            row["gold"] = gold
    server_dir = prep.BENCHMARK_DIR.parents[2] / "resources_servers/jeebench"
    validator = TaskDataValidator(
        server_name="jeebench", adapter=load_task_data_schema(server_dir), dataset_fpath="fixture"
    )
    validator.validate_row(0, row)
    assert not validator.report.clean
    if failure == "misplaced_gold":
        assert validator.report.misplaced_keys == {"gold": 1}
    else:
        assert validator.report.error_rows == 1


@pytest.mark.parametrize(
    "flags,expected",
    [([], (0.0, 2048)), (["--temperature", "1", "--top-p", "0.95", "--max-output-tokens", "50000"], (1.0, 50000))],
)
def test_cli_defaults_and_overrides_reach_collector(source, tmp_path, flags, expected):
    configs, _ = source
    config = _resolve_cli_config(flags)
    prepared = prep.build_rows(configs["hi"][:1], language="hi", revision="fixture")
    rows = _collect_rows(
        config, _materialize_dataset(config, "indic_jee_bench_simple_agent", prepared, tmp_path), tmp_path
    )
    assert len(rows) == 1
    request = rows[0]["responses_create_params"]
    assert (request["temperature"], request["max_output_tokens"]) == expected
    if flags:
        assert request["top_p"] == 0.95
    assert request["input"] == [
        {"role": "system", "content": ""},
        {"role": "user", "content": prepared[0]["instruction"]},
    ]
    assert rows[0]["agent_ref"] == {"name": "indic_jee_bench_simple_agent"}
    assert prepared[0]["responses_create_params"] == {"temperature": 0.0, "max_output_tokens": 2048}


@pytest.mark.parametrize("cli_override", [False, True])
def test_combined_benchmarks_preserve_audio_defaults_and_routing(source, tmp_path, cli_override):
    from benchmarks.numb3rs.prepare import _format_row

    configs, _ = source
    flags = ["--benchmark", "numb3rs"]
    if cli_override:
        flags += ["--temperature", "0.7", "--max-output-tokens", "32000"]
    config = _resolve_cli_config(flags)
    jee_agent, audio_agent = "indic_jee_bench_simple_agent", "numb3rs_asr_with_pc_simple_agent"
    jee = prep.build_rows(configs["hi"][:1], language="hi", revision="fixture")
    audio = _format_row(
        {"original_text": "forty two", "text": "42", "file_name": "sample.wav", "duration": 5, "category": "NUMBER"},
        audio_prefix=str(tmp_path),
    )
    original_audio = deepcopy(audio)
    prepared = _materialize_dataset(config, jee_agent, jee, tmp_path)
    prepared += _materialize_dataset(config, audio_agent, [audio], tmp_path)
    rows = _collect_rows(config, prepared, tmp_path)
    assert [row["agent_ref"]["name"] for row in rows] == [jee_agent, audio_agent]
    audio_request = rows[1]["responses_create_params"]
    assert (
        audio_request["metadata"]["audio_path"] == original_audio["responses_create_params"]["metadata"]["audio_path"]
    )
    assert audio == original_audio
    if cli_override:
        assert all(
            (row["responses_create_params"]["temperature"], row["responses_create_params"]["max_output_tokens"])
            == (0.7, 32000)
            for row in rows
        )
    else:
        assert not {"temperature", "max_output_tokens", "top_p"} & audio_request.keys()


def test_requested_repeats_own_their_seed_metadata(source, tmp_path):
    configs, _ = source
    config = _resolve_cli_config(["--num-repeats", "2"])
    config["num_repeats_add_seed"] = True
    jee = prep.build_rows(configs["hi"][:1], language="hi", revision="fixture")
    jee[0]["responses_create_params"]["metadata"] = {"extra_body": json.dumps({"top_k": -1})}
    rows = _collect_rows(config, _materialize_dataset(config, "indic_jee_bench_simple_agent", jee, tmp_path), tmp_path)
    assert [json.loads(row["responses_create_params"]["metadata"]["extra_body"]) for row in rows] == [
        {"top_k": -1, "seed": 0},
        {"top_k": -1, "seed": 1},
    ]
    assert json.loads(jee[0]["responses_create_params"]["metadata"]["extra_body"]) == {"top_k": -1}
