# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.indic.milu_1_5 import prepare as module
from nemo_gym.data import apply_prompt_to_row, load_prompt_config
from nemo_gym.data.prompt import validate_prompt_compatibility


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


def test_prompt_and_answer_preserve_reference_format():
    row = module.build_rows([source(" \nकौन?\t ")], language="hi")[0]
    prompt = load_prompt_config(str(module.DIRECTORY / "prompts/default.yaml"))
    validate_prompt_compatibility([row], prompt)
    result = apply_prompt_to_row(row, prompt)
    assert result["responses_create_params"]["input"] == [
        {
            "role": "user",
            "content": (
                "Below is a multiple choice question. After explanation finish your answer with statement "
                '"the answer is \\boxed{X}" where X is the correct letter from the below choices.\n'
                "Please read the question carefully and choose the correct answer from the given options.\n"
                "Question: कौन?\nOptions:\nA. एक \nB. दो\nC. तीन\nD. चार"
            ),
        }
    ]
    # Server serialization belongs to the vLLM server suite, alongside its implementation tests.
    assert json.loads(result["responses_create_params"]["metadata"]["extra_body"]) == {
        "stop": ["</s>", "<|im_end|>", "<eos>"]
    }
    assert "input" not in row["responses_create_params"]
    assert row["options"] == [{"A": "एक "}, {"B": "दो"}, {"C": "तीन"}, {"D": "चार"}]
    assert row["expected_answer"] == "B"
    assert row["metadata"]["num_fewshot"] == 0
    assert row["metadata"]["source_dataset"] == "ai4bharat/MILU-1.5"


def test_subset_preserves_order_ids_and_all_gold_letters():
    records = [source(f"question {i}", target=field) for i, field in enumerate(module.OPTION_FIELDS)]
    rows = module.build_rows(records, language="hi")
    assert [row["expected_answer"] for row in rows] == list("ABCD")
    assert module.build_rows(records, language="hi", question_ids=["3", "1"]) == [rows[1], rows[3]]
    assert rows[3]["metadata"]["source_row_index"] == 3
    assert rows[3]["uuid"].endswith("-hi-3")


@pytest.mark.parametrize("ids", ["0", [], [0], ["-1"], ["1"], ["0", "0"]])
def test_invalid_question_ids(ids):
    with pytest.raises(ValueError, match="question_ids"):
        module.build_rows([source()], language="hi", question_ids=ids)


@pytest.mark.parametrize("languages", ["hi", [], ["as"], ["hi", "hi"]])
def test_invalid_languages(languages, tmp_path):
    with pytest.raises(ValueError, match="language"):
        module.prepare(tmp_path / "bad.jsonl", languages=languages)


@pytest.mark.parametrize(
    "extra", [{"question": " "}, {"target": "B"}, {"language": "English"}, {"is_translated": "false"}]
)
def test_invalid_source_rows(extra):
    with pytest.raises(ValueError, match="MILU"):
        module.build_rows([source(**extra)], language="hi")


def test_unknown_build_language():
    with pytest.raises(ValueError, match="language"):
        module.build_rows([], language="as")


@pytest.fixture
def cached(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    path = dataset / "Hindi/test-00000-of-00001.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([source(), source("दूसरा?")]), path)
    checksums = {"Hindi/test-00000-of-00001.parquet": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "checksums.yaml").write_text(json.dumps(checksums))
    monkeypatch.setattr(module, "DIRECTORY", tmp_path)
    monkeypatch.setattr(module, "LANGUAGES", {"hi": "Hindi"})
    monkeypatch.setattr(module, "COUNTS", {"hi": 2})
    return dataset, path


def test_local_preparation_and_default_destination(cached, monkeypatch):
    dataset, _ = cached

    def unexpected_download(**kwargs):
        pytest.fail("Local preparation must not download anything")

    monkeypatch.setattr(module, "hf_hub_download", unexpected_download)
    path = module.prepare(dataset_dir=dataset)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert path == module.DIRECTORY / "data/milu_1_5_benchmark.jsonl"
    assert rows[1]["question"] == "दूसरा?"


def test_pinned_download(cached, tmp_path, monkeypatch):
    _, parquet = cached
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(parquet)

    monkeypatch.setattr(module, "hf_hub_download", download)
    path = module.prepare(tmp_path / "selected.jsonl", languages=["hi"], question_ids=["1"])
    assert len(path.read_text().splitlines()) == 1
    assert calls == [
        {
            "repo_id": "ai4bharat/MILU-1.5",
            "filename": "Hindi/test-00000-of-00001.parquet",
            "repo_type": "dataset",
            "revision": module.SOURCE_REVISION,
        }
    ]


def test_checksum_and_count_guards_preserve_output(cached, tmp_path, monkeypatch):
    dataset, parquet = cached
    output = tmp_path / "existing.jsonl"
    output.write_text("existing data\n")
    monkeypatch.setattr(module, "COUNTS", {"hi": 3})
    with pytest.raises(ValueError, match="expected 3"):
        module.prepare(output, dataset_dir=dataset)
    parquet.write_bytes(parquet.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        module.prepare(output, dataset_dir=dataset)
    assert output.read_text() == "existing data\n"


def test_failure_in_second_language_preserves_output(cached, tmp_path, monkeypatch):
    dataset, _ = cached
    monkeypatch.setattr(module, "LANGUAGES", {"hi": "Hindi", "bn": "Bengali"})
    output = tmp_path / "existing.jsonl"
    output.write_text("previous complete dataset\n")
    # Hindi is serialized first; a failure loading Bengali must not publish a partial file.
    with pytest.raises(FileNotFoundError):
        module.prepare(output, dataset_dir=dataset, languages=["hi", "bn"])
    assert output.read_text() == "previous complete dataset\n"
    assert list(tmp_path.glob(".existing.jsonl.*.tmp")) == []


def test_serialization_failure_preserves_output(cached, tmp_path, monkeypatch):
    dataset, _ = cached
    output = tmp_path / "existing.jsonl"
    output.write_text("previous complete dataset\n")
    original_build = module.build_rows

    def broken_build(*args, **kwargs):
        rows = original_build(*args, **kwargs)
        rows[1]["metadata"]["bad_value"] = object()
        return rows

    monkeypatch.setattr(module, "build_rows", broken_build)
    with pytest.raises(TypeError, match="JSON serializable"):
        module.prepare(output, dataset_dir=dataset)
    assert output.read_text() == "previous complete dataset\n"
    assert list(tmp_path.glob(".existing.jsonl.*.tmp")) == []


def test_cli(cached, tmp_path, monkeypatch, capsys):
    dataset, _ = cached
    output = tmp_path / "cli.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            "--dataset-dir",
            str(dataset),
            "--languages",
            "hi",
            "--question-ids",
            "1",
            "--output-path",
            str(output),
        ],
    )
    module.main()
    assert str(output) in capsys.readouterr().out
    assert json.loads(output.read_text())["metadata"]["source_row_index"] == 1


def test_examples_and_config(monkeypatch):
    from omegaconf import OmegaConf

    import nemo_gym.config.global_config as global_config
    from nemo_gym.benchmarks import BenchmarkConfig, _benchmark_config_paths
    from nemo_gym.cli.main import _asset_config_path
    from nemo_gym.environment.manifest import load_manifest

    prompt = load_prompt_config(str(module.DIRECTORY / "prompts/default.yaml"))
    rows = [json.loads(line) for line in (module.DIRECTORY / "data/example.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    validate_prompt_compatibility(rows, prompt)
    for row in rows:
        assert row["metadata"]["synthetic"] is True
        assert r"\boxed{X}" in apply_prompt_to_row(row, prompt)["responses_create_params"]["input"][0]["content"]

    monkeypatch.setattr(global_config, "_find_open_port_using_range", lambda **_: 12345)
    path = module.DIRECTORY / "config.yaml"
    assert _asset_config_path("benchmark", "indic/milu_1_5") == str(path)
    assert path in _benchmark_config_paths(module.DIRECTORY)
    benchmark = BenchmarkConfig.from_config_path(path, strict=False)
    assert benchmark.agent_name == "milu_1_5_simple_agent"
    assert benchmark.num_repeats == 1
    config = global_config.GlobalConfigDictParser().parse_no_environment(
        initial_global_config_dict=OmegaConf.create(
            {
                "config_paths": [str(path), "responses_api_models/vllm_model/configs/vllm_model.yaml"],
                "policy_base_url": "http://unused/v1",
                "policy_api_key": "dummy",  # pragma: allowlist secret
                "policy_model_name": "test",
            }
        )
    )
    servers = global_config.GlobalConfigDictParser().filter_for_server_instance_configs(config)
    assert {server.name for server in servers} == {
        "policy_model",
        "milu_1_5_mcqa_resources_server",
        "milu_1_5_simple_agent",
        "milu_1_5_mcqa_environment_server",
    }
    assert (
        config["milu_1_5_mcqa_resources_server"]["resources_servers"]["mcqa"]["grading_mode"] == "boxed_and_normalized"
    )
    assert config["milu_1_5_simple_agent"]["responses_api_agents"]["simple_agent"]["max_steps"] == 1
    assert config["policy_model"]["responses_api_models"]["vllm_model"].get("use_completions_api", False) is False
    manifest = load_manifest(module.DIRECTORY / "manifest.yaml")
    assert manifest.name == "indic/milu_1_5"
    assert manifest.resources_server == "mcqa"
    assert manifest.agent_server == "simple_agent"
    assert manifest.grading_mode == "boxed_and_normalized"
    assert manifest.datasets[0].jsonl_fpath == str(benchmark.dataset.jsonl_fpath)
    assert manifest.datasets[0].prompt_config == str(benchmark.dataset.prompt_config)


def test_run_overrides_preserve_row_scoped_vllm_metadata(tmp_path):
    from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper

    row = module.build_rows([source()], language="hi")[0]
    row["agent_ref"] = {"name": "milu_1_5_simple_agent"}
    config = RolloutCollectionConfig(
        input_jsonl_fpath="unused",
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        num_repeats=1,
        responses_create_params={"temperature": 0.4, "top_p": 0.8, "max_output_tokens": 4096},
    )
    prepared = RolloutCollectionHelper._preprocess_raw_rows([(0, json.dumps(row), row.copy())], config)

    assert len(prepared) == 1
    request = prepared[0]["responses_create_params"]
    assert (request["temperature"], request["top_p"], request["max_output_tokens"]) == (0.4, 0.8, 4096)
    assert json.loads(request["metadata"]["extra_body"]) == {"stop": ["</s>", "<|im_end|>", "<eos>"]}
