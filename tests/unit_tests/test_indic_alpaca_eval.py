# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.indic_alpaca_eval import prepare as module
from benchmarks.indic_alpaca_eval.prepare import prepare
from resources_servers.alpaca_pointwise_judge.task_data import TaskData


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "source"
    for code in ["en", "hi", "kn"]:
        rows = [
            dict(
                row_id=f"id-{i}",
                row_index=i,
                instruction=f"{code} instruction {i}",
                language_code=code,
                judge_pass_stage="failed_after_correction_review_needed",
                human_evaluation_pending=True,
                human_review_reason="failed_final_judge",
            )
            for i in range(805)
        ]
        path = root / "data" / code / "eval.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
    return root


def test_prepare_alignment_aliases_and_coverage(dataset, tmp_path):
    output = tmp_path / "out"
    path = prepare(dataset, output_dir=output, languages=["hi", "ka"], limit_per_language=2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert len({row["_ng_task_index"] for row in rows}) == 4
    assert {row["id"] for row in rows} == {"hi/id-0", "hi/id-1", "kn/id-0", "kn/id-1"}
    for row in rows:
        task = TaskData.model_validate(row).verifier_metadata
        assert row["responses_create_params"]["input"][0]["content"] == task.question
        assert task.human_evaluation_pending
        assert task.expected_counts == {"hi": 2, "kn": 2}
    hi = json.loads((output / "hi.jsonl").read_text().splitlines()[0])
    kn = json.loads((output / "kn.jsonl").read_text().splitlines()[0])
    assert hi["_ng_task_index"] != kn["_ng_task_index"]
    assert hi["_ng_task_index"] == rows[0]["_ng_task_index"]
    assert hi["verifier_metadata"]["evaluation_set_id"] == rows[0]["verifier_metadata"]["evaluation_set_id"]
    smoke = [json.loads(line) for line in (output / "smoke.jsonl").read_text().splitlines()]
    assert len(smoke) == 2
    assert smoke[0]["verifier_metadata"]["expected_counts"] == {"hi": 1, "kn": 1}
    assert smoke[0]["verifier_metadata"]["evaluation_set_id"] != rows[0]["verifier_metadata"]["evaluation_set_id"]
    assert json.loads((output / "manifest.json").read_text())["source_revision"] is None
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["prepared_sha256"] == module.sha256(path)
    assert manifest["adapter_sha256"] == module.sha256(Path(module.__file__))
    assert manifest["rubric_sha256"] == module.sha256(module.RUBRIC_PATH)


@pytest.mark.parametrize("mutation", ["reorder", "missing", "duplicate", "language", "blank"])
def test_bad_source_rejected(dataset, tmp_path, mutation):
    path = dataset / "data/hi/eval.parquet"
    rows = pq.read_table(path).to_pylist()
    if mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = rows[0]
    elif mutation == "language":
        rows[0]["language_code"] = "kn"
    else:
        rows[0]["instruction"] = " "
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError):
        prepare(dataset, output_dir=tmp_path / "out", languages=["hi"])
    assert not (tmp_path / "out/indic_alpaca_eval_benchmark.jsonl").exists()


@pytest.mark.parametrize(
    "languages,limit", [([], None), (["ka", "kn"], None), (["sa"], None), (["hi"], 0), (["hi"], 806)]
)
def test_bad_selection_rejected(dataset, tmp_path, languages, limit):
    with pytest.raises(ValueError):
        prepare(dataset, output_dir=tmp_path / "out", languages=languages, limit_per_language=limit)


@pytest.fixture
def wide_dataset(tmp_path):
    root = tmp_path / "wide-source"
    root.mkdir()
    rows = [
        {
            "id": f"source-{i}",
            "position": i,
            "instruction": f"English instruction {i}",
            "instruction_Hindi_translation": f"Hindi instruction {i}",
            "instruction_Kannada_translation": f"Kannada instruction {i}",
        }
        for i in range(805)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "eval.parquet")
    return root


def test_prepare_wide_source(wide_dataset, tmp_path):
    output = tmp_path / "out"
    path = prepare(wide_dataset, output_dir=output, languages=["hi", "ka"], limit_per_language=2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["hi/source-0", "hi/source-1", "kn/source-0", "kn/source-1"]
    assert [row["_ng_task_index"] for row in rows] == [1610, 1611, 2415, 2416]
    for row in rows:
        metadata = TaskData.model_validate(row).verifier_metadata
        assert metadata.question == f"{metadata.language_name} instruction {metadata.row_index}"
        assert row["responses_create_params"]["input"][0]["content"] == metadata.question
        assert metadata.human_evaluation_pending is None
        assert metadata.judge_pass_stage == "not_provided"
        assert metadata.expected_counts == {"hi": 2, "kn": 2}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_repo"] == "ai4bharat/indic-alpaca-eval"
    assert manifest["source_revision"] is None
    assert manifest["dataset_sha256"] == dict.fromkeys(["hi", "kn"], module.sha256(wide_dataset / "eval.parquet"))


@pytest.mark.parametrize("mutation", ["duplicate_id", "missing_row", "position", "blank", "missing_column"])
def test_bad_wide_source_rejected(wide_dataset, tmp_path, mutation):
    path = wide_dataset / "eval.parquet"
    rows = pq.read_table(path).to_pylist()
    if mutation == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif mutation == "missing_row":
        rows.pop()
    elif mutation == "position":
        rows[1]["position"] = 0
    elif mutation == "blank":
        rows[0]["instruction_Hindi_translation"] = " "
    else:
        for row in rows:
            del row["instruction_Hindi_translation"]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError):
        prepare(wide_dataset, output_dir=tmp_path / "out", languages=["hi"])
    assert not (tmp_path / "out/indic_alpaca_eval_benchmark.jsonl").exists()


def test_download_is_pinned(wide_dataset, tmp_path, monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(wide_dataset)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    prepare(output_dir=tmp_path / "out", languages=["hi"])
    assert calls == [
        dict(
            repo_id=module.SOURCE_REPO,
            repo_type="dataset",
            revision=module.SOURCE_REVISION,
            allow_patterns=["eval.parquet"],
        )
    ]
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["source_revision"] == module.SOURCE_REVISION


def resolve_config(flags):
    from omegaconf import OmegaConf

    from nemo_gym.cli.main import _merge_config_paths, build_parser
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    args = build_parser().parse_args(
        [
            "eval",
            "run",
            "--benchmark",
            "indic_alpaca_eval",
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


@pytest.mark.parametrize("cli_override", [False, True])
def test_native_cli_defaults_preserve_other_tasks_and_metadata(dataset, tmp_path, monkeypatch, cli_override):
    from nemo_gym.environment.manifest import load_manifest
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    monkeypatch.setenv("INDIC_ALPACA_JUDGE_API_KEY", "test-only")
    flags = ["--temperature", "0.4", "--top-p", "0.8", "--max-output-tokens", "4096"] if cli_override else []
    config = resolve_config(["--benchmark", "numb3rs", *flags])
    manifest = load_manifest(Path(module.__file__).parent / "manifest.yaml")
    assert manifest.canonical_split == "benchmark"
    assert manifest.datasets[0].num_repeats == 1
    agent = config["indic_alpaca_eval_simple_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["datasets"][0]["num_repeats"] == 1
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert not policy.get("sampling_overrides") and not policy.get("chat_template_kwargs")
    if not cli_override:
        assert not config.get("responses_create_params")
    prepared = prepare(dataset, output_dir=tmp_path / "out", languages=["hi"], limit_per_language=2)
    original = [json.loads(line) for line in prepared.read_text().splitlines()]
    for row in original:
        row["agent_ref"] = {"name": "indic_alpaca_eval_simple_agent"}
    audio = {
        "agent_ref": {"name": "numb3rs_asr_with_pc_simple_agent"},
        "responses_create_params": {
            "input": [{"role": "user", "content": "Transcribe"}],
            "metadata": {"audio_path": "/synthetic/audio.wav"},
        },
    }
    combined = tmp_path / "combined.jsonl"
    combined.write_text("".join(json.dumps(row) + "\n" for row in [*original, audio]))
    collection = RolloutCollectionConfig.model_validate(
        config
        | {
            "input_jsonl_fpath": str(combined),
            "output_jsonl_fpath": str(tmp_path / "rollouts.jsonl"),
            "num_repeats": 2,
            "num_repeats_add_seed": True,
        }
    )
    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, collection)
    indic_rows = [row for row in rows if row["agent_ref"]["name"] == "indic_alpaca_eval_simple_agent"]
    assert len(indic_rows) == 4
    for row in indic_rows:
        request = row["responses_create_params"]
        assert (request["temperature"], request["top_p"], request["max_output_tokens"]) == (
            (0.4, 0.8, 4096) if cli_override else (1.0, 0.95, 120000)
        )
        assert json.loads(request["metadata"]["chat_template_kwargs"]) == {"enable_thinking": True}
    assert [json.loads(row["responses_create_params"]["metadata"]["extra_body"])["seed"] for row in indic_rows] == [
        0,
        1,
        0,
        1,
    ]
    indic_rows[0]["responses_create_params"]["metadata"]["extra_body"] = "{}"
    assert json.loads(indic_rows[1]["responses_create_params"]["metadata"]["extra_body"])["seed"] == 1
    assert json.loads(original[0]["responses_create_params"]["metadata"]["extra_body"])["seed"] == 42
    audio_rows = [row for row in rows if row["agent_ref"]["name"] == "numb3rs_asr_with_pc_simple_agent"]
    for row in audio_rows:
        request = row["responses_create_params"]
        assert request["metadata"]["audio_path"] == "/synthetic/audio.wav"
        assert "chat_template_kwargs" not in request["metadata"]
        if not cli_override:
            assert not {"temperature", "top_p", "max_output_tokens"} & request.keys()
