# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Litmus-Bench release preparation and verifier compatibility."""

from __future__ import annotations

import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.config_types import DatasetConfig
from nemo_gym.environment.manifest import EnvironmentManifest
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.litmus_agent.app import LitmusAgentConfig, LitmusAgentResourcesServer, LitmusAgentVerifyRequest


prepare_module = importlib.import_module("benchmarks.litmus-bench.prepare")
BENCHMARK_DIR = Path(__file__).resolve().parents[1]
CONFIG_FPATH = BENCHMARK_DIR / "config.yaml"


def _source_row(*, uuid: str = "litmus-1", answer: float | str = 1.9231974) -> dict:
    return {
        "dataset": "litmus_bench",
        "question_family": "tier5_admet_direct",
        "question_license": "OpenMDW-1.1",
        "responses_create_params": {"input": [{"role": "user", "content": "Predict the property. Answer: number"}]},
        "expected_answer": answer,
        "answer_type": "float",
        "output_regex": r"Answer:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "match": {"rule": "abs_window", "abs_tol": 0.3},
        "property": "LOG HLM_CLint (mL/min/kg)",
        "method": "direct",
        "metadata": {"uuid": uuid, "sources": [{"dataset": "biogen", "row_id": "Mol2465"}]},
        "agent_ref": {"name": "litmus_agent_agent"},
    }


def _patch_release(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rows: list[dict]) -> list[dict]:
    source = tmp_path / "source.jsonl"
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    source.write_bytes(payload)
    monkeypatch.setattr(
        prepare_module,
        "RELEASE_FILE",
        ("validation.jsonl", len(rows), hashlib.sha256(payload).hexdigest()),
    )
    monkeypatch.setattr(prepare_module, "SUBSET_FAMILIES", {"direct": {"tier5_admet_direct": len(rows)}})
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path / "prepared")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(prepare_module, "hf_hub_download", fake_download)
    return calls


def test_prepare_downloads_release_and_preserves_scoring_and_provenance(monkeypatch, tmp_path) -> None:
    rows = [
        _source_row(uuid="float"),
        _source_row(uuid="bool", answer=1),
        _source_row(uuid="string", answer=" Paris "),
    ]
    rows[1]["answer_type"] = "bool"
    rows[1].pop("match")
    rows[2].update(answer_type="string", output_regex=r"Answer:\s*(.+)", match={"rule": "string_eq"})
    calls = _patch_release(monkeypatch, tmp_path, rows)

    output_path = prepare_module.prepare()
    first_content = output_path.read_bytes()
    assert prepare_module.prepare() == output_path
    assert output_path.read_bytes() == first_content
    assert (
        calls
        == [
            {
                "repo_id": prepare_module.HF_DATASET,
                "filename": "validation.jsonl",
                "repo_type": "dataset",
            }
        ]
        * 2
    )
    prepared = [json.loads(line) for line in first_content.splitlines()]
    for source, result in zip(rows, prepared):
        expected = deepcopy(source)
        expected.pop("agent_ref")
        assert result == expected
    assert prepared[0]["expected_answer"] == 1.9231974
    assert "match" not in prepared[1]
    assert prepare_module.is_prepared_data_current(output_path)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda row: row.pop("output_regex"), "missing required fields"),
        (lambda row: row.update(method="mcp-python"), "method must be 'direct'"),
        (lambda row: row["responses_create_params"].update(tools=[{"type": "function"}]), "must not expose tools"),
        (lambda row: row.update(expected_answer=float("nan")), "finite and numeric"),
        (lambda row: row.update(expected_answer=True), "finite and numeric"),
        (lambda row: row.update(answer_type="unsupported"), "unsupported answer_type"),
        (lambda row: row.update(answer_type="string"), "expected_answer must be a string"),
        (lambda row: row.update(expected_answer="1.5"), "finite and numeric"),
        (lambda row: row.update(answer_type="bool", expected_answer=2), "must be 0 or 1"),
        (lambda row: row.update(output_regex="("), "invalid output_regex"),
        (lambda row: row.update(output_regex="(one)(two)"), "exactly one capture group"),
        (lambda row: row["metadata"].pop("uuid"), "metadata.uuid"),
    ],
)
def test_prepare_rejects_schema_drift(monkeypatch, tmp_path, mutate, error) -> None:
    row = _source_row()
    mutate(row)
    _patch_release(monkeypatch, tmp_path, [row])
    with pytest.raises(ValueError, match=error):
        prepare_module.prepare()
    assert not (tmp_path / "prepared/litmus-bench_benchmark.jsonl").exists()


def test_prepare_rejects_duplicate_metadata_uuids(monkeypatch, tmp_path) -> None:
    _patch_release(monkeypatch, tmp_path, [_source_row(), _source_row()])
    with pytest.raises(ValueError, match="duplicate uuid"):
        prepare_module.prepare()


@pytest.mark.parametrize("failure", ["checksum", "count"])
def test_invalid_release_does_not_replace_existing_output(monkeypatch, tmp_path, failure) -> None:
    _patch_release(monkeypatch, tmp_path, [_source_row()])
    output_path = prepare_module.prepare()
    original = output_path.read_bytes()
    if failure == "checksum":
        (tmp_path / "source.jsonl").write_text("corrupt data\n")
        error = "checksum mismatch"
    else:
        path, count, sha256 = prepare_module.RELEASE_FILE
        monkeypatch.setattr(prepare_module, "RELEASE_FILE", (path, count + 1, sha256))
        error = "has 1 rows; expected 2"
    with pytest.raises(ValueError, match=error):
        prepare_module.prepare()
    assert output_path.read_bytes() == original


def test_cache_rejects_old_modified_and_changed_release_data(monkeypatch, tmp_path) -> None:
    _patch_release(monkeypatch, tmp_path, [_source_row()])
    output_path = tmp_path / "prepared/litmus-bench_benchmark.jsonl"
    output_path.parent.mkdir()
    output_path.write_text("old benchmark\n")
    assert not prepare_module.is_prepared_data_current(output_path)
    prepare_module.prepare()
    assert prepare_module.is_prepared_data_current(output_path)
    output_path.write_text("modified benchmark\n")
    assert not prepare_module.is_prepared_data_current(output_path)
    prepare_module.prepare()
    monkeypatch.setattr(prepare_module, "HF_DATASET", "nvidia/another-release")
    assert not prepare_module.is_prepared_data_current(output_path)


def test_single_subset_and_local_export_use_same_validation(monkeypatch, tmp_path) -> None:
    calls = _patch_release(monkeypatch, tmp_path, [_source_row()])
    source_path = tmp_path / "validation.jsonl"
    source_path.write_bytes((tmp_path / "source.jsonl").read_bytes())
    output_path = prepare_module.prepare(subset="direct", source_dir=str(tmp_path))
    assert output_path.name == "adme-tier5-direct_benchmark.jsonl"
    assert not calls
    assert prepare_module.is_prepared_data_current(output_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer_type", "expected", "match", "correct", "incorrect"),
    [
        ("float", 11, {"rule": "exact"}, "11", "12"),
        ("float", 1.9231974, {"rule": "abs_window", "abs_tol": 0.3}, "2.1", "3"),
        ("float", 100, {"rule": "rel_window", "rel_tol": 0.1}, "105", "120"),
        ("bool", 1, None, "1", "0"),
        ("string", "Paris", None, "  pArIs  ", "Berlin"),
    ],
)
async def test_prepared_rows_keep_verifier_semantics(answer_type, expected, match, correct, incorrect) -> None:
    row = _source_row(answer=expected)
    row["answer_type"] = answer_type
    if answer_type == "string":
        row["output_regex"] = r"Answer:\s*(.+)"
    if match is None:
        row.pop("match")
    else:
        row["match"] = match
    prepared = json.loads(prepare_module._render_rows([row], expected_rows=1))
    server = LitmusAgentResourcesServer(
        config=LitmusAgentConfig(host="127.0.0.1", port=8080, entrypoint=""),
        server_client=MagicMock(spec=ServerClient),
    )
    for answer, expected_reward in [(correct, 1.0), (incorrect, 0.0)]:
        response = NeMoGymResponse(
            id="test-response",
            created_at=0,
            model="test",
            object="response",
            output=[
                {
                    "id": "test-message",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"Answer: {answer}", "annotations": []}],
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        request = LitmusAgentVerifyRequest(response=response, **prepared)
        assert (await server.verify(request)).reward == expected_reward


def test_benchmark_config_is_discoverable_and_isolated() -> None:
    benchmark = BenchmarkConfig.from_config_path(CONFIG_FPATH, strict=False)
    assert benchmark is not None
    assert benchmark.name == "litmus-bench"
    assert benchmark.agent_name == "litmus_bench_benchmark_agent"
    assert benchmark.num_repeats == 5
    assert benchmark.dataset.prompt_config is None
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/litmus-bench/data/litmus-bench_benchmark.jsonl")
    assert benchmark.dataset.prepare_script == Path("benchmarks/litmus-bench/prepare.py")

    initial_config = OmegaConf.merge(
        OmegaConf.load(CONFIG_FPATH), GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial_config)
    assert "litmus_agent" not in resolved
    assert "litmus_agent_agent" not in resolved
    assert resolved.responses_create_params.max_output_tokens == 131072
    resource = resolved.litmus_bench_benchmark_resources_server.resources_servers.litmus_agent
    agent = resolved.litmus_bench_benchmark_agent.responses_api_agents.simple_agent
    assert resource.verified is False
    assert agent.resources_server.name == "litmus_bench_benchmark_resources_server"
    assert len(agent.datasets) == 1
    assert agent.datasets[0].license == "CC-BY-SA-4.0 AND LicenseRef-OpenMDW-1.1"
    manifest = EnvironmentManifest.model_validate(
        OmegaConf.to_container(OmegaConf.load(BENCHMARK_DIR / "manifest.yaml"))
    )
    assert manifest.licensing == agent.datasets[0].license


@pytest.mark.parametrize("config_name", ["litmus_agent", "litmus_agent_apptainer"])
def test_training_sources_use_release_layout_and_combined_license(config_name: str) -> None:
    config = OmegaConf.load(BENCHMARK_DIR.parents[1] / f"resources_servers/litmus_agent/configs/{config_name}.yaml")
    dataset = DatasetConfig.model_validate(
        OmegaConf.to_container(config.litmus_agent_agent.responses_api_agents.simple_agent.datasets[0])
    )
    assert dataset.source.repo_id == prepare_module.HF_DATASET
    assert dataset.source.artifact_fpath == "train.jsonl"
    assert dataset.jsonl_fpath == "resources_servers/litmus_agent/data/litmus-bench/train.jsonl"
    assert dataset.license == "CC-BY-SA-4.0 AND LicenseRef-OpenMDW-1.1"


@pytest.mark.parametrize("subset", ["tier12", "direct", "analogue", "comparison"])
def test_subset_config_uses_shared_preparation(subset: str) -> None:
    name = "config_tier12" if subset == "tier12" else f"config_adme_{subset}"
    config = OmegaConf.load(BENCHMARK_DIR / f"{name}.yaml")
    benchmark = BenchmarkConfig.from_config_path(BENCHMARK_DIR / f"{name}.yaml", strict=False)
    assert config.prepare_script_args.subset == subset
    assert benchmark is not None
    assert benchmark.dataset.prepare_script == Path("benchmarks/litmus-bench/prepare.py")
    assert benchmark.dataset.jsonl_fpath.name == prepare_module._output_filename(subset)
    assert benchmark.num_repeats == 5
    manifest = EnvironmentManifest.model_validate(
        OmegaConf.to_container(OmegaConf.load(BENCHMARK_DIR / name / "manifest.yaml"))
    )
    assert manifest.licensing == config[benchmark.agent_name].responses_api_agents.simple_agent.datasets[0].license


def test_subset_filters_by_family_and_preserves_rows_without_match(monkeypatch, tmp_path) -> None:
    direct = _source_row(uuid="direct")
    direct.pop("match")
    comparison = _source_row(uuid="comparison", answer=1)
    comparison.update(question_family="tier5_admet_comparison", answer_type="bool")
    comparison.pop("match")
    _patch_release(monkeypatch, tmp_path, [direct, comparison])
    monkeypatch.setattr(
        prepare_module,
        "SUBSET_FAMILIES",
        {
            "direct": {"tier5_admet_direct": 1},
            "comparison": {"tier5_admet_comparison": 1},
        },
    )
    for subset, source in [("direct", direct), ("comparison", comparison)]:
        output = prepare_module.prepare(subset=subset)
        expected = deepcopy(source)
        expected.pop("agent_ref")
        assert json.loads(output.read_text()) == expected
        assert prepare_module.is_prepared_data_current(output)


def test_unexpected_question_family_cannot_silently_disappear(monkeypatch, tmp_path) -> None:
    row = _source_row()
    row["question_family"] = "unknown"
    _patch_release(monkeypatch, tmp_path, [row])
    with pytest.raises(ValueError, match="question family counts"):
        prepare_module.prepare(subset="direct")


@pytest.mark.parametrize("subset", ["tier12", "direct", "analogue", "comparison"])
def test_examples_follow_release_schema_and_subset(subset: str) -> None:
    name = "litmus-tier12" if subset == "tier12" else f"adme-tier5-{subset}"
    path = BENCHMARK_DIR / "data/examples" / f"{name}_example.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rendered = prepare_module._render_rows(rows, expected_rows=10)
    assert len(rendered.splitlines()) == 10
    assert all(row["question_family"] in prepare_module.SUBSET_FAMILIES[subset] for row in rows)
    assert all("agent_ref" not in row for row in rows)
    if subset == "tier12":
        assert sum(row["question_family"] == "tier1_rdkit" for row in rows) == 5


def test_unknown_subset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown Litmus-Bench subset"):
        prepare_module.prepare(subset="unknown")
