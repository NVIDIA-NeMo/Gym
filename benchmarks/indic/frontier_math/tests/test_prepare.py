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
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.cli.main import _asset_config_path
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility
from nemo_gym.registry import _manifest_entry
from resources_servers.frontiermath.app import FrontierMathRunRequest


def test_manifest_and_committed_requests() -> None:
    benchmark = Path(__file__).resolve().parents[1]
    manifest = load_manifest(benchmark / "manifest.yaml")
    assert manifest.name == "indic/frontier_math"
    entry = _manifest_entry(benchmark.parents[1], benchmark / "manifest.yaml", "benchmark")
    config_path = Path(_asset_config_path("benchmark", entry.name))
    assert config_path == benchmark / "config.yaml"
    config = BenchmarkConfig.from_config_path(config_path)
    assert config.dataset.prepare_script.resolve() == benchmark / "prepare.py"
    assert config.dataset.jsonl_fpath.resolve() == benchmark / "data/indic_frontiermath_raw.jsonl"
    assert all(dataset.jsonl_fpath.endswith("indic_frontiermath_raw.jsonl") for dataset in manifest.datasets)
    template = yaml.safe_load(prep.PROMPT_PATH.read_text())["user"]
    examples = benchmark.parents[2] / "resources_servers/frontiermath/data/example.jsonl"
    for line in examples.read_text().splitlines():
        row = json.loads(line)
        request = FrontierMathRunRequest.model_validate(row)
        assert request.responses_create_params.input[0].content == template.format(
            language=row["language"], question=row["question"]
        )


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
    for code in ("en", "hi"):
        rows = (
            english
            if code == "en"
            else [
                {
                    **english[0],
                    "problem": "१+१ की गणना करें।",
                    "language": "Hindi",
                    "language_code": "hi",
                    "judge_pass_stage": "failed_after_correction_review_needed",
                    "human_evaluation_pending": True,
                }
            ]
        )
        directory = tmp_path / "data" / code
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), directory / "sample.parquet")
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
    assert json.loads((snapshot / "output/preparation.json").read_text())["failed_translation_review"] == 1
    raw_rows = [
        json.loads(line) for line in (snapshot / "output/indic_frontiermath_raw.jsonl").read_text().splitlines()
    ]
    prompt = load_prompt_config(str(prep.PROMPT_PATH))
    validate_prompt_compatibility(raw_rows, prompt)
    assert [apply_prompt_to_row(row, prompt) for row in raw_rows] == rows


def test_changed_english_rejected(snapshot: Path) -> None:
    path = snapshot / "data/en/sample.parquet"
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
    path = snapshot / "data/hi/sample.parquet"
    pq.write_table(pq.read_table(path).slice(0, 0), path)
    with pytest.raises(ValueError, match="Problem IDs for hi"):
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
