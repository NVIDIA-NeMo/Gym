# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from benchmarks.indic.frontier_math import prepare as prep
from benchmarks.indic.frontier_math.summarize import summarize
from nemo_gym.cli.main import _asset_config_path
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility
from nemo_gym.registry import _manifest_entry
from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows


@pytest.mark.parametrize("rendered", [False, True], ids=["raw", "rendered"])
def test_task_data_schema_validates_examples_without_dropping_metadata(rendered: bool) -> None:
    server_dir = prep.BENCHMARK_DIR.parents[2] / "resources_servers/frontiermath"
    adapter = load_task_data_schema(server_dir)
    assert adapter is not None
    rows = [json.loads(line) for line in (server_dir / "data/example.jsonl").read_text().splitlines()]
    if not rendered:
        for row in rows:
            row.pop("responses_create_params")
    report = validate_jsonl_rows("frontiermath", adapter, "examples", [json.dumps(row) for row in rows])
    assert report.rows == 5
    assert report.clean, report.summary()
    for row in rows:
        task = {key: value for key, value in row.items() if key != "responses_create_params"}
        assert adapter.validate_python(task).model_dump() == task


@pytest.mark.parametrize(
    "field,value",
    [("answer_type", "approximate"), ("tier", "not-an-integer"), ("human_evaluation_pending", "unknown")],
)
def test_task_data_schema_rejects_invalid_grading_metadata(field: str, value: str) -> None:
    server_dir = prep.BENCHMARK_DIR.parents[2] / "resources_servers/frontiermath"
    adapter = load_task_data_schema(server_dir)
    assert adapter is not None
    row = json.loads((server_dir / "data/example.jsonl").read_text().splitlines()[0])
    row[field] = value
    report = validate_jsonl_rows("frontiermath", adapter, "invalid", [json.dumps(row)])
    assert report.error_rows == 1
    assert field in report.summary()


def test_manifest_and_committed_requests() -> None:
    benchmark = prep.BENCHMARK_DIR
    manifest = load_manifest(benchmark / "manifest.yaml")
    assert manifest.name == "indic/frontier_math"
    entry = _manifest_entry(benchmark.parents[1], benchmark / "manifest.yaml", "benchmark")
    config_path = Path(_asset_config_path("benchmark", entry.name))
    assert config_path == benchmark / "config.yaml"
    assert manifest.canonical_split == "benchmark"
    assert manifest.datasets[0].num_repeats == 1
    assert Path(manifest.datasets[0].prepare_script).resolve() == benchmark / "prepare.py"
    assert all(dataset.jsonl_fpath.endswith("indic_frontiermath_raw.jsonl") for dataset in manifest.datasets)
    template = yaml.safe_load(prep.PROMPT_PATH.read_text())["user"]
    examples = benchmark.parents[2] / "resources_servers/frontiermath/data/example.jsonl"
    for line in examples.read_text().splitlines():
        row = json.loads(line)
        assert row["responses_create_params"]["input"][0]["content"] == template.format(
            language=row["language"], question=row["question"]
        )


def test_native_config_routes_to_frontiermath_without_global_sampling_defaults() -> None:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                        str(prep.BENCHMARK_DIR / "config.yaml"),
                    ],
                    "policy_base_url": "http://unused/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "test",
                    "error_on_agent_without_environment_server": True,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert {server.name for server in parser.filter_for_server_instance_configs(config)} == {
        "policy_model",
        "frontiermath",
        "indic_frontiermath_simple_agent",
        "indic_frontiermath_environment_server",
    }
    environment = config.indic_frontiermath_environment_server.environment_servers.legacy_agent
    assert environment.agent_server.name == "indic_frontiermath_simple_agent"
    agent = config.indic_frontiermath_simple_agent.responses_api_agents.simple_agent
    assert agent.resources_server.name == "frontiermath"
    assert agent.datasets[0].num_repeats == 1
    assert agent.datasets[0].type == "benchmark"
    assert agent.datasets[0].jsonl_fpath == "benchmarks/indic/frontier_math/data/indic_frontiermath_raw.jsonl"
    assert not config.get("responses_create_params")
    policy = config.policy_model.responses_api_models.vllm_model
    assert policy.chat_template_kwargs is None
    assert policy.sampling_overrides is None


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # A tiny independent fixture exercises joins and version checking without requiring private local paths.
    english = [
        {
            "row_id": "p1",
            "row_index": 0,
            "problem": "Compute 1+1.",
            "language": "English",
            "language_code": "en",
            "judge_pass_stage": "english_source",
            "human_evaluation_pending": False,
            "human_review_reason": "",
        }
    ]
    import hashlib

    manifest = {
        "version": "test",
        "answers": [
            {
                "row_id": "p1",
                "tier": 1,
                "answer_type": "integer",
                "expected_answer": "2",
                "english_problem_sha256": hashlib.sha256(english[0]["problem"].encode()).hexdigest(),
            }
        ],
    }
    (tmp_path / "answers.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(prep, "BENCHMARK_DIR", tmp_path)
    rows = [
        {
            "id": "p1",
            "tier": 1,
            "problem": english[0]["problem"],
            "answer": r"\(2\)",
            "solution": "This solution must not be included in prepared rows.",
            "problem_Hindi_translation": "१+१ की गणना करें।",
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "sample.parquet")
    monkeypatch.setattr(prep, "hf_hub_download", lambda **kwargs: pytest.fail("Unexpected network access"))
    return tmp_path


def test_join_and_no_reference_in_prompt(snapshot: Path) -> None:
    output = prep.prepare(dataset_dir=str(snapshot), languages=["hi", "en"], output_dir=str(snapshot / "output"))
    assert output == snapshot / "output/indic_frontiermath_raw.jsonl"
    rendered = output.with_name("indic_frontiermath_benchmark.jsonl")
    rows = [json.loads(line) for line in rendered.read_text().splitlines()]
    assert [row["language_code"] for row in rows] == ["hi", "en"]
    assert all(row["expected_answer"] == "2" for row in rows)
    assert rows[0]["human_evaluation_pending"]
    assert rows[0]["task_id"] != rows[1]["task_id"]
    assert rows[0]["responses_create_params"]["input"][0]["content"].endswith("१+१ की गणना करें।")
    assert "expected_answer" not in json.dumps(rows[0]["responses_create_params"])
    assert rows[0]["judge_pass_stage"] == "not_provided"
    assert rows[1]["judge_pass_stage"] == "english_source"
    assert all("solution" not in row and "answer" not in row for row in rows)
    audit = json.loads((snapshot / "output/preparation.json").read_text())
    assert audit["failed_translation_review"] == 0
    assert audit["source_id"] == "ai4bharat/indic-frontier-math"
    assert audit["source_revision"] is None
    assert audit["local_override"] is True
    raw_rows = [
        json.loads(line) for line in (snapshot / "output/indic_frontiermath_raw.jsonl").read_text().splitlines()
    ]
    prompt = load_prompt_config(str(prep.PROMPT_PATH))
    validate_prompt_compatibility(raw_rows, prompt)
    assert [apply_prompt_to_row(row, prompt) for row in raw_rows] == rows


def test_changed_english_rejected(snapshot: Path) -> None:
    path = snapshot / "sample.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["problem"] = "Changed problem"
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="statement changed"):
        prep.prepare(dataset_dir=str(snapshot), languages=["hi"])


@pytest.mark.parametrize("languages", [[], ["xx"], ["hi", "hi"]])
def test_invalid_language_selection(snapshot: Path, languages: list[str]) -> None:
    with pytest.raises(ValueError, match="unique subset"):
        prep.prepare(dataset_dir=str(snapshot), languages=languages)


def test_missing_translation_rejected(snapshot: Path) -> None:
    path = snapshot / "sample.parquet"
    table = pq.read_table(path).drop_columns(["problem_Hindi_translation"])
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="Missing source columns.*problem_Hindi_translation"):
        prep.prepare(dataset_dir=str(snapshot), languages=["hi"])


def test_download_uses_ai4bharat_pinned_revision(snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloads = []

    def download(**kwargs: str) -> str:
        downloads.append(kwargs)
        return str(snapshot / "sample.parquet")

    monkeypatch.setattr(prep, "hf_hub_download", download)
    output = prep.prepare(languages=["hi"], output_dir=str(snapshot / "output"))
    assert downloads == [
        {
            "repo_id": "ai4bharat/indic-frontier-math",
            "filename": "sample.parquet",
            "repo_type": "dataset",
            "revision": prep.SOURCE_REVISION,
        }
    ]
    assert json.loads(output.read_text())["question"] == "१+१ की गणना करें।"
    audit = json.loads(output.with_name("preparation.json").read_text())
    assert audit["source_revision"] == prep.SOURCE_REVISION
    assert audit["local_override"] is False


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("problem_Hindi_translation", "", "Empty problem for hi"),
        ("problem_Hindi_translation", "  ", "Empty problem for hi"),
        ("problem_Hindi_translation", None, "Empty problem for hi"),
        ("answer", "3", "Source answer changed"),
        ("tier", 2, "Tier mismatch"),
        ("id", "wrong-id", "Source problem IDs"),
    ],
)
def test_source_mismatch_rejected_before_output(snapshot: Path, field: str, value: object, error: str) -> None:
    path = snapshot / "sample.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows), path)
    output = snapshot / "output"
    with pytest.raises(ValueError, match=error):
        prep.prepare(dataset_dir=str(snapshot), languages=["hi"], output_dir=str(output))
    assert not output.exists()


def test_duplicate_source_ids_rejected(snapshot: Path) -> None:
    path = snapshot / "sample.parquet"
    rows = pq.read_table(path).to_pylist()
    pq.write_table(pa.Table.from_pylist(rows * 2), path)
    with pytest.raises(ValueError, match="Source problem IDs"):
        prep.prepare(dataset_dir=str(snapshot), languages=["hi"])


def test_summary_weights_problems_and_reports_coverage(tmp_path: Path) -> None:
    rows = [
        {
            "row_id": problem,
            "language_code": lang,
            "reward": reward,
            "grading_status": "correct" if reward else "incorrect",
            "judge_pass_stage": "english_source" if lang == "en" else "failed_after_correction_review_needed",
            "human_evaluation_pending": lang != "en",
        }
        for lang, problem, reward in [("en", "a", 1), ("en", "a", 1), ("en", "b", 0), ("hi", "a", 0)]
    ]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    summary = summarize(path)["languages"]
    assert summary["en"]["pass_at_1"] == 0.5
    assert summary["en"]["normalized_pass_at_1"] == 0.5
    assert summary["en"]["normalized_recovered_attempts"] == 0
    assert not summary["en"]["complete_12_problem_coverage"]
    assert summary["hi"]["paired_delta_vs_english"] == -1
    assert summary["hi"]["pass_at_1_translation_review_passed"] is None
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="missing"):
        summarize(path)


def _summary_row(problem: str, **updates) -> dict:
    return {
        "row_id": problem,
        "language_code": "hi",
        "reward": 0.0,
        "grading_status": "incorrect",
        "judge_pass_stage": "first_judge_pass",
        "human_evaluation_pending": False,
        **updates,
    }


@pytest.mark.parametrize("failure_status", ["verifier_error", "timeout"])
def test_summary_excludes_failed_and_masked_attempts(tmp_path: Path, failure_status: str) -> None:
    rows = [
        _summary_row("a", reward=1.0, grading_status="correct"),
        _summary_row("a", reward=1.0, grading_status="correct"),
        _summary_row("a", grading_status=failure_status),
        _summary_row(
            "b",
            grading_status="invalid_expression",
            extracted_answer="n=214",
            answer_type="integer",
            expected_answer="214",
        ),
        _summary_row("b", reward=1.0, grading_status="correct", mask_sample=True),
        _summary_row("c", grading_status=failure_status, mask_sample=True),
        _summary_row("c", language_code="en", reward=1.0, grading_status="correct"),
    ]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    report = summarize(path)["languages"]["hi"]
    assert report["pass_at_1"] == 0.5
    assert report["normalized_pass_at_1"] == 1.0
    assert report["normalized_recovered_attempts"] == 1
    assert report["problems"] == 3 and report["attempts"] == 6
    assert report["scored_problems"] == 2 and report["scored_attempts"] == 3
    assert report["excluded_attempts"] == 3
    assert report["masked_attempts"] == 2
    assert report["verifier_failure_attempts"] == 2
    assert report["grading_statuses"][failure_status] == 2
    assert report["pass_at_1_translation_review_passed"] == 0.5
    assert report["paired_english_problems"] == 0
    assert report["paired_delta_vs_english"] is None


def test_summary_retains_language_with_no_successful_grades(tmp_path: Path) -> None:
    rows = [_summary_row(str(i), grading_status="verifier_error", mask_sample=True) for i in range(12)]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    report = summarize(path)["languages"]["hi"]
    assert report["pass_at_1"] is None and report["normalized_pass_at_1"] is None
    assert report["problems"] == 12 and report["attempts"] == 12
    assert report["scored_problems"] == 0 and report["scored_attempts"] == 0
    assert report["excluded_attempts"] == 12 and report["verifier_failure_attempts"] == 12
    assert report["observed_12_problem_coverage"]
    assert not report["complete_12_problem_coverage"]
    assert report["grading_statuses"] == {"verifier_error": 12}


def test_summary_excludes_normalization_worker_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_to_start(*args, **kwargs):
        raise OSError("Unavailable")

    monkeypatch.setattr("benchmarks.indic.frontier_math.normalization.subprocess.run", fail_to_start)
    rows = [
        _summary_row("a", reward=1.0, grading_status="correct"),
        _summary_row(
            "b",
            grading_status="invalid_expression",
            extracted_answer=r"x=\frac{1}{2}",
            answer_type="expression",
            expected_answer=r"\frac{1}{2}",
        ),
    ]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    report = summarize(path)["languages"]["hi"]
    assert report["pass_at_1"] == report["normalized_pass_at_1"] == 1.0
    assert report["scored_attempts"] == report["scored_problems"] == 1
    assert report["excluded_attempts"] == report["normalization_failure_attempts"] == 1
    assert report["verifier_failure_attempts"] == 1
    assert sum(report["normalization_failures"].values()) == 1
