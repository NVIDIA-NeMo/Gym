# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark discovery, prompt preservation, and safe dataset materialization."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.sciknoweval import prepare
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


GYM_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def benchmark():
    return "sciknoweval", prepare


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
    assert verifier["judge_model_server"]["name"] == "policy_model"
    assert verifier["judge_responses_create_params"]["max_output_tokens"] == 16384


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
        row["verifier_metadata"].pop("id", None)
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


def test_rubric_download_is_pinned_cached_and_checked(monkeypatch, tmp_path):
    import hashlib
    import io

    import yaml

    from benchmarks.sciknoweval import rubrics as module

    source = yaml.safe_dump(
        {
            "formula_derivation": {"user": "Question: {prompt}"},
            "problem_solving": {"user": "Question: {prompt}"},
            "extract_doping": {
                "type": "MCQ",
                "system": "Compare",
                "user": "The question involves hosts. Your task: {response}",
            },
        }
    ).encode()
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(source).hexdigest())
    calls = []

    def download(url, *, timeout):
        assert module.REVISION in url
        assert timeout == 120
        calls.append(url)
        return io.BytesIO(source)

    monkeypatch.setattr(module, "urlopen", download)
    first = module.load_rubrics(tmp_path)
    assert module.load_rubrics(tmp_path) == first
    assert len(calls) == 1
    assert first["formula_derivation"]["user"] == "Question: {question}"
    assert "drug-drug" in first["drug_drug_relation_extraction"]["user"]
    cache = next(tmp_path.iterdir())
    cache.write_bytes(b"corrupted cache")
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: io.BytesIO(b"HTTP error page"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_rubrics(tmp_path)
    assert cache.read_bytes() == b"corrupted cache"
