# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from responses_api_agents.deep_swe import prepare as prepare_module


def _checkout(root: Path, task_ids: tuple[str, ...]) -> Path:
    for task_id in task_ids:
        task = root / "tasks" / task_id
        task.mkdir(parents=True)
        (task / "task.toml").write_text('schema_version = "1.1"\n')
    return root


def test_prepare_writes_shared_deterministic_113_task_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = tuple(f"task-{index:03d}" for index in range(prepare_module.EXPECTED_TASK_COUNT))
    checkouts = {
        commit: _checkout(tmp_path / profile, task_ids) for profile, commit in prepare_module.PROFILE_COMMITS.items()
    }
    calls: list[dict[str, Any]] = []

    async def fake_checkout(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return checkouts[kwargs["commit"]]

    monkeypatch.setattr(prepare_module, "ensure_checkout", fake_checkout)
    monkeypatch.setattr(
        prepare_module,
        "task_tree_digest",
        lambda checkout: f"{next(commit for commit, path in checkouts.items() if path == checkout):0<64}"[:64],
    )
    output = tmp_path / "data" / "deep_swe_benchmark_validation.jsonl"
    monkeypatch.setattr(prepare_module, "OUTPUT_PATH", output)
    monkeypatch.setattr(prepare_module, "CACHE_DIR", tmp_path / "cache")

    assert prepare_module.prepare() == output
    first_bytes = output.read_bytes()
    assert prepare_module.prepare() == output
    assert output.read_bytes() == first_bytes

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == prepare_module.EXPECTED_TASK_COUNT
    assert [row["verifier_metadata"]["task_id"] for row in rows] == list(task_ids)
    assert set(rows[0]["verifier_metadata"]["benchmark_profiles"]) == set(prepare_module.PROFILE_COMMITS)
    assert all(call["expected_task_count"] == prepare_module.EXPECTED_TASK_COUNT for call in calls)
    assert not list(output.parent.glob("*.tmp"))


def test_prepare_rejects_profile_task_id_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_ids = tuple(f"task-{index:03d}" for index in range(prepare_module.EXPECTED_TASK_COUNT))
    commits = tuple(prepare_module.PROFILE_COMMITS.values())
    checkouts = {
        commits[0]: _checkout(tmp_path / "first", base_ids),
        commits[1]: _checkout(tmp_path / "second", (*base_ids[:-1], "different-task")),
    }

    async def fake_checkout(**kwargs: Any) -> Path:
        return checkouts[kwargs["commit"]]

    monkeypatch.setattr(prepare_module, "ensure_checkout", fake_checkout)
    monkeypatch.setattr(prepare_module, "task_tree_digest", lambda _: "0" * 64)
    monkeypatch.setattr(prepare_module, "OUTPUT_PATH", tmp_path / "output.jsonl")
    with pytest.raises(ValueError, match="task IDs differ"):
        prepare_module.prepare()


def test_prepare_rejects_wrong_task_count_and_empty_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = _checkout(tmp_path / "short", ("only-task",))
    with pytest.raises(ValueError, match="Expected 113"):
        prepare_module._task_ids(checkout)

    monkeypatch.setattr(prepare_module, "PROFILE_COMMITS", {})
    with pytest.raises(RuntimeError, match="No DeepSWE benchmark profiles"):
        prepare_module.prepare()


def test_write_jsonl_removes_partial_output_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output.jsonl"

    def fail_record(*_: object) -> dict[str, object]:
        raise RuntimeError("record failure")

    monkeypatch.setattr(prepare_module, "_record", fail_record)
    with pytest.raises(RuntimeError, match="record failure"):
        prepare_module._write_jsonl(output, ("task",), {})
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_profiles_are_distinct_benchmarks_with_three_repeats() -> None:
    root = Path(__file__).resolve().parents[1]
    current = OmegaConf.load(root / "configs" / "deep_swe.yaml")
    aa = OmegaConf.load(root / "configs" / "deep_swe_aa_v1.yaml")

    current_agent = current.deep_swe.responses_api_agents.deep_swe
    aa_agent = aa.deep_swe_aa_v1.responses_api_agents.deep_swe
    current_benchmark = next(dataset for dataset in current_agent.datasets if dataset.type == "benchmark")
    aa_benchmark = next(dataset for dataset in aa_agent.datasets if dataset.type == "benchmark")

    assert current_benchmark.name == "deep_swe_v1_1"
    assert aa_benchmark.name == "deep_swe_aa_v1"
    assert current_benchmark.num_repeats == aa_benchmark.num_repeats == 3
    assert current_benchmark.jsonl_fpath == aa_benchmark.jsonl_fpath
    assert current_agent.benchmark_git_commit == prepare_module.CURRENT_V1_1_COMMIT
    assert aa_agent.benchmark_git_commit == prepare_module.AA_V1_COMMIT
    assert aa_agent.claude_code_env.ANTHROPIC_CUSTOM_HEADERS == (
        "X-NMP-Principal-Id: service:evaluator\nX-Inference-Priority: batch"
    )
    assert aa_agent.claude_code_env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC == "1"
