# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from pytest import MonkeyPatch, raises

from benchmarks.toolalignbench import prepare as prepare_module
from benchmarks.toolalignbench import reconcile_baselines


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(task: int, repeat: int, text: str = "No action required.") -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "document"}]},
        "id": f"task-{task}",
        "domain": "financial",
        "scenario_type": "safe",
        "prompt_condition": "tamely-act-system-prompt--without-reasoning",
        "tool_names": ["writeInternalLog", "email"],
        "remaining_documents": ["d2", "d3", "d4"],
        "agent_ref": {"name": "toolalignbench_benchmark_agent"},
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "response": {
            "model": "fixture-model",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        },
        "reward": 1.0,
        "mask_sample": False,
        "failure_kind": None,
        "failure_reason": None,
        "episode_timed_out": False,
        "model_incomplete": False,
        "num_documents": 4,
        "num_documents_completed": 4,
        "num_unparsed_tool_call_replies": 0,
        "num_unknown_tool_calls": 0,
    }


def test_documents_digest_is_path_order_stable(tmp_path: Path) -> None:
    for subdir, values in {"prompts": ["a", "b"], "prompts-safe": ["c", "d"]}.items():
        domain = tmp_path / "benchmark" / subdir / "domain"
        domain.mkdir(parents=True)
        for index, value in enumerate(values, 1):
            (domain / f"{index}.md").write_text(value, encoding="utf-8")
        (domain / "_SOURCE.md").write_text("ignored", encoding="utf-8")

    count, first = prepare_module._documents_digest(tmp_path)
    _, second = prepare_module._documents_digest(tmp_path)

    assert count == 4
    assert first == second


def test_reconcile_replaces_every_affected_identity(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(reconcile_baselines, "EXPECTED_TASKS", 2)
    monkeypatch.setattr(reconcile_baselines, "EXPECTED_REPEATS", 2)
    monkeypatch.setattr(reconcile_baselines, "EXPECTED_ROWS", 4)
    malformed = "<tool_call><email><arguments>{}</arguments></tool_call>"
    originals = [_row(0, 0), _row(0, 1, malformed), _row(1, 0), _row(1, 1)]
    original_path = tmp_path / "original.jsonl"
    _write_jsonl(original_path, originals)

    replacement = _row(9, 0)
    replacement.update(
        rerun_target_task_index=0,
        rerun_target_rollout_index=1,
        ng_model_call_capture={"rollout_id": "9-0", "calls": [], "metrics": {}},
        ng_trajectory={
            "rollout_id": "9-0",
            "task_id": "9",
            "turns": [{"task_id": "9", "rollout_id": "9-0", "turn_no": 1}],
        },
    )
    replacement_path = tmp_path / "replacement.jsonl"
    _write_jsonl(replacement_path, [replacement])

    summary = reconcile_baselines.reconcile(original_path, replacement_path, tmp_path / "out", "fixture")
    merged = reconcile_baselines._read_jsonl(tmp_path / "out/fixture_toolalignbench_rollouts.jsonl")

    assert summary["replaced_rows"] == 1
    assert summary["reused_rows"] == 3
    assert [reconcile_baselines._key(row) for row in merged] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    repaired = merged[1]
    assert repaired["ng_model_call_capture"]["rollout_id"] == "0-1"
    assert repaired["ng_trajectory"] == {
        "rollout_id": "0-1",
        "task_id": "0",
        "turns": [{"task_id": "0", "rollout_id": "0-1", "turn_no": 1}],
    }


def test_reconcile_rejects_masked_replacement(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(reconcile_baselines, "EXPECTED_TASKS", 1)
    monkeypatch.setattr(reconcile_baselines, "EXPECTED_REPEATS", 1)
    monkeypatch.setattr(reconcile_baselines, "EXPECTED_ROWS", 1)
    original = _row(0, 0, "<tool_call><email><arguments>{}</arguments></tool_call>")
    original_path = tmp_path / "original.jsonl"
    _write_jsonl(original_path, [original])
    replacement = _row(4, 0)
    replacement.update(rerun_target_task_index=0, rerun_target_rollout_index=0, mask_sample=True)
    replacement_path = tmp_path / "replacement.jsonl"
    _write_jsonl(replacement_path, [replacement])

    with raises(ValueError, match="not scoreable"):
        reconcile_baselines.reconcile(original_path, replacement_path, tmp_path / "out", "fixture")


def test_rerun_reasons_uses_persisted_parser_diagnostics() -> None:
    row = _row(0, 0)
    row.update(
        mask_sample=True,
        failure_kind="toolalignbench:unparsed_tool_call",
        num_unparsed_tool_call_replies=2,
    )

    assert reconcile_baselines.rerun_reasons(row) == {
        "masked_sample",
        "prior_failure_reason",
        "unparsed_tool_call",
    }
