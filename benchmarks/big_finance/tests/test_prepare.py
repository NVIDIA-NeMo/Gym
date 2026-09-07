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

import pytest

from benchmarks.big_finance import prepare


def _row(index: int = 1) -> dict:
    return {
        "id": f"bf-{index}",
        "query": "What was revenue?",
        "reference_answer": "$10 million",
        "rubric": [{"text": "Finds revenue.", "points": 2}],
        "evaluation_only": True,
        "do_not_train": True,
        "benchmark_canary": "test canary",
        "sources": ["https://www.sec.gov/example"],
    }


def test_convert_preserves_labels_and_bakes_upstream_surface(tmp_path) -> None:
    source = tmp_path / "fixture.jsonl"
    output = tmp_path / "gym.jsonl"
    source.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    assert prepare.convert_file(source, output, expected_count=1) == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    for key in (
        "id",
        "query",
        "reference_answer",
        "rubric",
        "evaluation_only",
        "do_not_train",
        "benchmark_canary",
        "sources",
    ):
        assert row[key] == _row()[key]
    spec = prepare.load_spec()
    assert row["license"] == spec["dataset_license"] == "CC BY 4.0"
    assert row["provenance"] == {
        "repository": spec["repository"],
        "upstream_commit_id": spec["upstream_commit_id"],
        "dataset_url": spec["dataset_url"],
        "attribution": (
            "Big Finance benchmark, public release subset (n = 50), "
            "Rogo Technologies (2026), licensed under CC BY 4.0."
        ),
    }
    assert row["responses_create_params"]["input"] == [
        {"role": "system", "content": spec["system_prompt"], "type": "message"},
        {"role": "user", "content": _row()["query"], "type": "message"},
    ]
    assert [tool["name"] for tool in row["responses_create_params"]["tools"]] == [
        "web_search",
        "edgar_search",
        "fetch_url",
        "python_exec",
        "final_answer",
    ]
    assert row["responses_create_params"]["parallel_tool_calls"] is True


def test_public_count_guard_avoids_silent_partial_download(tmp_path) -> None:
    source = tmp_path / "fixture.jsonl"
    source.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 50"):
        prepare.convert_file(source, tmp_path / "out.jsonl", expected_count=50)


def test_evaluation_only_guards_cannot_be_disabled() -> None:
    row = _row()
    row["do_not_train"] = False
    with pytest.raises(ValueError, match="evaluation_only and do_not_train"):
        prepare.convert_rows([row])


def test_dataset_url_and_snapshot_are_commit_pinned() -> None:
    spec = prepare.load_spec()
    assert len(spec["upstream_commit_id"]) == 40
    assert f"/{spec['upstream_commit_id']}/" in spec["dataset_url"]
    assert spec["upstream_commit_id"] == prepare._UPSTREAM_SHA


def test_stale_snapshot_fails_loudly(monkeypatch) -> None:
    monkeypatch.setattr(prepare, "_UPSTREAM_SHA", "0" * 40)
    with pytest.raises(ValueError, match="update the snapshot and pin together"):
        prepare.load_spec()


def test_public_fixture_preserves_provenance_prompt_and_license() -> None:
    fixture = prepare.ENV_DIR.parents[1] / "resources_servers/big_finance/data/example.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    spec = prepare.load_spec()

    assert len(rows) == 5
    assert len({row["id"] for row in rows}) == 5
    for row in rows:
        assert row["license"] == spec["dataset_license"]
        assert row["provenance"]["repository"] == spec["repository"]
        assert row["provenance"]["upstream_commit_id"] == prepare._UPSTREAM_SHA
        assert row["provenance"]["dataset_url"] == spec["dataset_url"]
        assert row["responses_create_params"]["input"][0] == {
            "role": "system",
            "content": spec["system_prompt"],
            "type": "message",
        }
