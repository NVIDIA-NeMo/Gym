# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark discovery, prompt preservation, and safe dataset materialization."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.chembench import prepare
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


GYM_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def benchmark():
    return "chembench", prepare


def example(name):
    path = GYM_ROOT / "resources_servers" / name / "data" / "example.jsonl"
    return json.loads(path.read_text().splitlines()[0])


def test_discovery_and_judge_wiring(benchmark, monkeypatch):
    name, module = benchmark
    monkeypatch.chdir(GYM_ROOT)
    path = Path("benchmarks") / name / "config.yaml"
    config = BenchmarkConfig.from_config_path(path)
    assert config.name == name
    assert config.agent_name == f"{name}_simple_agent"
    assert config.dataset.jsonl_fpath.resolve() == module.OUTPUT_FPATH
    assert config.dataset.prepare_script == Path("benchmarks") / name / "prepare.py"
    assert config.num_repeats == 1
    assert config.dataset.prompt_config is None  # Already rendered source prompts must not be rewritten.
    initial = OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, OmegaConf.load(path))
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)
    verifier = resolved[name]["resources_servers"][name]
    assert not verifier["verified"]
    assert not verifier.get("judge_model_server")


def test_preserves_prepared_messages_and_metadata(benchmark, tmp_path):
    name, module = benchmark
    source, target = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    payload = (json.dumps(example(name), ensure_ascii=False) + "\n").encode()
    source.write_bytes(payload)
    assert module.prepare(source, target) == target
    assert target.read_bytes() == payload
    # Repeating preparation and materializing in place are safe and deterministic.
    module.prepare(source, target)
    module.prepare(target, target)
    assert target.read_bytes() == payload


@pytest.mark.parametrize("problem", ["empty", "malformed", "duplicate", "metadata", "input", "identity"])
def test_invalid_export_does_not_replace_existing_output(benchmark, tmp_path, problem):
    name, module = benchmark
    row = example(name)
    if problem == "metadata":
        row["verifier_metadata"] = {}
    elif problem == "input":
        row["responses_create_params"]["input"] = []
    elif problem == "identity":
        row["verifier_metadata"].pop("uuid", None)
    text = json.dumps(row) + "\n"
    if problem == "empty":
        text = ""
    elif problem == "malformed":
        text += "not json\n"
    elif problem == "duplicate":
        text += text
    source, target = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    source.write_text(text)
    target.write_text("previous valid output\n")
    with pytest.raises(ValueError):
        module.prepare(source, target)
    assert target.read_text() == "previous valid output\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output.jsonl", "source.jsonl"]


def test_explicit_missing_input_reports_the_path(benchmark, tmp_path):
    name, module = benchmark
    with pytest.raises(FileNotFoundError, match=f"Explicit converted {name} input does not exist"):
        module.prepare(tmp_path / "missing.jsonl", tmp_path / "out.jsonl")
    assert not (tmp_path / "out.jsonl").exists()


def test_default_preparation_builds_data_without_a_local_export(benchmark, tmp_path, monkeypatch):
    name, module = benchmark
    calls = []

    def build():
        calls.append(True)
        yield example(name)

    monkeypatch.setattr(module, "build_rows", build)
    output = tmp_path / "prepared.jsonl"
    module.prepare(output_path=output)
    assert calls == [True]
    assert json.loads(output.read_text()) == example(name)


def test_failed_download_preserves_existing_output(benchmark, tmp_path, monkeypatch):
    name, module = benchmark

    def interrupted():
        yield example(name)
        raise OSError("download interrupted")

    monkeypatch.setattr(module, "build_rows", interrupted)
    output = tmp_path / "prepared.jsonl"
    output.write_text("existing output\n")
    with pytest.raises(OSError, match="download interrupted"):
        module.prepare(output_path=output)
    assert output.read_text() == "existing output\n"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("tolerance", [None, 0.0, 0.5])
def test_chembench_preserves_source_tolerance(tolerance, tmp_path, monkeypatch):
    from datasets import Dataset

    from benchmarks.chembench import prepare
    from resources_servers.chembench.task_data import TaskData

    source = {
        "name": "numeric",
        "uuid": "numeric-1",
        "subfield": "test",
        "keywords": [],
        "in_humansubset_wo_tool": False,
        "examples": [{"input": "A question", "target": "100", "target_scores": None}],
    }
    monkeypatch.setattr("benchmarks.chembench.conversion.TOPICS", ["general_chemistry"])
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: Dataset.from_list([source]))
    baseline = list(prepare.build_rows())[0]
    source["relative_tolerance"] = tolerance
    output = prepare.prepare(output_path=tmp_path / "prepared.jsonl")
    row = json.loads(output.read_text())
    assert row["responses_create_params"] == baseline["responses_create_params"]
    assert row["verifier_metadata"] == baseline["verifier_metadata"] | {"relative_tolerance": tolerance}
    metadata = TaskData.model_validate_json(json.dumps(row["verifier_metadata"]))
    assert metadata.relative_tolerance == tolerance


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"The ion $\ce{SO4^{2-}}$", "The ion SO4^{2-}"),
        (r"\pu{10 mol L^{-1}}", "10 mol L^{-1}"),
        (r"$x^{2} + y$", "x^{2} + y"),
        (r"$x $", "x "),
        (r"\ce{H2} + \ce{O2} with \pu{25 C}", "H2 + O2 with 25 C"),
        ("[START_SMILES]CCO[END_SMILES]", "CCO"),
        ("  ordinary text  ", "ordinary text"),
        (r"unclosed \ce{H2O and $x", r"unclosed \ce{H2O and $x"),
    ],
)
def test_upstream_latex_cleanup(text, expected):
    from benchmarks.chembench.conversion import clean_text

    assert clean_text(text) == expected


def test_cleanup_applies_to_questions_and_options_without_changing_labels():
    from benchmarks.chembench.conversion import format_entry

    source = {
        "name": "markup",
        "uuid": "markup-1",
        "subfield": "test",
        "keywords": [],
        "in_humansubset_wo_tool": False,
        "examples": [
            {
                "input": r"Identify $\ce{H2O}$ at \pu{25 C}.",
                "target_scores": json.dumps({r"\ce{H2O}": 1, r"\ce{CO2}": 0}),
            }
        ],
    }
    row = format_entry(source, "general_chemistry", use_cot=False)
    assert "Question: Identify H2O at 25 C." in row["problem"]
    assert "Options:\nA. H2O\nB. CO2" in row["problem"]
    assert row["options"] == "A. H2O\nB. CO2"
    assert row["expected_answer"] == "A"
