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
"""Unit tests for sweep manifests."""

from __future__ import annotations

import collections
import json

import pytest
import yaml

from nemo_gym.sweep.build import build_sweep
from nemo_gym.sweep.manifest import SweepValidationError, load_manifest, validate_manifest


def _write_config(tmp_path, name, agent):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({agent: {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}}}))
    return path


def _write_data(tmp_path, name, agent, rows=3, missing_ref=False):
    path = tmp_path / name
    lines = []
    for i in range(rows):
        row = {"prompt": f"row-{i}"}
        if not missing_ref:
            row["agent_ref"] = {"type": "responses_api_agents", "name": agent}
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def _manifest(tmp_path, entries, defaults=None):
    doc = {"nickname": "testrun", "defaults": defaults or {"num_repeats": 8}, "entries": entries}
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_manifest(path)


def test_config_paths_dedupe_preserving_order(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_config(tmp_path, "b.yaml", "agent_b")
    manifest = _manifest(
        tmp_path,
        [
            {"label": "one", "agent": "agent_a", "configs": ["a.yaml", "b.yaml"], "data": "x.jsonl"},
            {"label": "two", "agent": "agent_b", "configs": ["b.yaml"], "data": "y.jsonl"},
        ],
    )
    assert manifest.config_paths() == ["a.yaml", "b.yaml"]


def test_num_repeats_global_default_with_local_override(tmp_path):
    manifest = _manifest(
        tmp_path,
        [
            {"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": "x.jsonl"},
            {"label": "two", "agent": "agent_b", "configs": ["b.yaml"], "data": "y.jsonl", "num_repeats": 4},
        ],
    )
    assert manifest.num_repeats() == {"_default": 8, "agent_b": 4}


def test_duplicate_labels_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate entry labels"):
        _manifest(
            tmp_path,
            [
                {"label": "dup", "agent": "agent_a", "configs": ["a.yaml"], "data": "x.jsonl"},
                {"label": "dup", "agent": "agent_b", "configs": ["b.yaml"], "data": "y.jsonl"},
            ],
        )


def test_validate_accepts_matching_agent_ref(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a")
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    assert validate_manifest(manifest, repo_root=tmp_path) == []


def test_validate_rejects_agent_ref_mismatch(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_WRONG")
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    with pytest.raises(SweepValidationError, match="data declares agent_ref"):
        validate_manifest(manifest, repo_root=tmp_path)


def test_validate_rejects_agent_not_declared_by_config(tmp_path):
    _write_config(tmp_path, "a.yaml", "some_other_agent")
    _write_data(tmp_path, "x.jsonl", "agent_a")
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    with pytest.raises(SweepValidationError, match="is not declared by"):
        validate_manifest(manifest, repo_root=tmp_path)


def test_validate_rejects_rows_without_agent_ref(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", missing_ref=True)
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    with pytest.raises(SweepValidationError, match="no agent_ref"):
        validate_manifest(manifest, repo_root=tmp_path)


def test_validate_rejects_conflicting_repeats_on_shared_agent(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a")
    _write_data(tmp_path, "y.jsonl", "agent_a")
    manifest = _manifest(
        tmp_path,
        [
            {
                "label": "one",
                "agent": "agent_a",
                "configs": ["a.yaml"],
                "data": str(tmp_path / "x.jsonl"),
                "num_repeats": 4,
            },
            {
                "label": "two",
                "agent": "agent_a",
                "configs": ["a.yaml"],
                "data": str(tmp_path / "y.jsonl"),
                "num_repeats": 8,
            },
        ],
    )
    with pytest.raises(SweepValidationError, match="different num_repeats"):
        validate_manifest(manifest, repo_root=tmp_path)


def test_validate_warns_when_entries_share_an_agent(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a")
    _write_data(tmp_path, "y.jsonl", "agent_a")
    manifest = _manifest(
        tmp_path,
        [
            {"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")},
            {"label": "two", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "y.jsonl")},
        ],
    )
    warnings = validate_manifest(manifest, repo_root=tmp_path)
    assert len(warnings) == 1 and "share agent 'agent_a'" in warnings[0]


def test_build_concatenates_and_limits(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_config(tmp_path, "b.yaml", "agent_b")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=5)
    _write_data(tmp_path, "y.jsonl", "agent_b", rows=5)
    manifest = _manifest(
        tmp_path,
        [
            {"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")},
            {"label": "two", "agent": "agent_b", "configs": ["b.yaml"], "data": str(tmp_path / "y.jsonl")},
        ],
    )
    report = build_sweep(manifest, tmp_path / "out", limit_per_entry=2)
    assert report.rows_per_entry == {"one": 2, "two": 2}
    rows = [json.loads(line) for line in report.input_jsonl.read_text().splitlines()]
    assert [r["agent_ref"]["name"] for r in rows] == ["agent_a"] * 2 + ["agent_b"] * 2
    assert yaml.safe_load(report.config_yaml.read_text())["config_paths"] == ["a.yaml", "b.yaml"]


def test_build_applies_agent_ref_override(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=2)
    manifest = _manifest(
        tmp_path,
        [
            {
                "label": "one",
                "agent": "agent_a",
                "agent_ref_override": "agent_override",
                "configs": ["a.yaml"],
                "data": str(tmp_path / "x.jsonl"),
            }
        ],
    )
    report = build_sweep(manifest, tmp_path / "out")
    rows = [json.loads(line) for line in report.input_jsonl.read_text().splitlines()]
    assert {r["agent_ref"]["name"] for r in rows} == {"agent_override"}
    assert report.overrides_applied == {"one": "agent_override"}


def test_build_refuses_to_clobber_existing_input(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=1)
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    build_sweep(manifest, tmp_path / "out")
    with pytest.raises(SweepValidationError, match="already exists"):
        build_sweep(manifest, tmp_path / "out")


def test_nickname_is_required(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"entries": [{"label": "a", "agent": "x", "configs": ["a.yaml"], "data": "d"}]}))
    with pytest.raises(Exception, match="nickname"):
        load_manifest(path)


def test_build_scopes_outputs_under_nickname(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=4)
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    report = build_sweep(manifest, tmp_path / "out")
    assert report.input_jsonl.parent.name == "testrun"
    assert report.num_shards == 1


def test_build_shards_round_robin_across_entries(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_config(tmp_path, "b.yaml", "agent_b")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=4)
    _write_data(tmp_path, "y.jsonl", "agent_b", rows=4)
    doc = {
        "nickname": "sharded",
        "num_shards": 2,
        "entries": [
            {"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")},
            {"label": "two", "agent": "agent_b", "configs": ["b.yaml"], "data": str(tmp_path / "y.jsonl")},
        ],
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(doc))
    report = build_sweep(load_manifest(path), tmp_path / "out")
    assert report.num_shards == 2
    out = report.input_jsonl.parent
    shards = sorted(out.glob("input_*.jsonl"))
    assert len(shards) == 2
    # every shard sees both agents, so each exercises the full fan-out
    for s in shards:
        names = {json.loads(line)["agent_ref"]["name"] for line in s.read_text().splitlines()}
        assert names == {"agent_a", "agent_b"}


def test_materialize_expands_repeats_with_stable_identity(tmp_path):
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_config(tmp_path, "b.yaml", "agent_b")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=3)
    _write_data(tmp_path, "y.jsonl", "agent_b", rows=2)
    manifest = _manifest(
        tmp_path,
        [
            {"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")},
            {
                "label": "two",
                "agent": "agent_b",
                "configs": ["b.yaml"],
                "data": str(tmp_path / "y.jsonl"),
                "num_repeats": 2,
            },
        ],
        defaults={"num_repeats": 4},
    )
    report = materialize(manifest, tmp_path / "out", jobs=2)

    rows = [json.loads(line) for line in report.materialized_fpath.read_text().splitlines()]
    assert report.total_source_rows == 5
    assert report.total_materialized_rows == 3 * 4 + 2 * 2 == len(rows)

    # task indices are contiguous per-entry ranges in manifest order
    by_agent = {}
    for r in rows:
        by_agent.setdefault(r["agent_ref"]["name"], set()).add(r["_ng_task_index"])
    assert by_agent["agent_a"] == {0, 1, 2}
    assert by_agent["agent_b"] == {3, 4}

    # each task carries exactly its entry's repeat count, numbered from zero
    per_task = collections.Counter(r["_ng_task_index"] for r in rows)
    assert [per_task[i] for i in range(5)] == [4, 4, 4, 2, 2]
    for task in range(5):
        idxs = sorted(r["_ng_rollout_index"] for r in rows if r["_ng_task_index"] == task)
        assert idxs == list(range(per_task[task]))


def test_materialize_is_deterministic_across_runs(tmp_path):
    """Regenerating on another node must reproduce the same resume keys."""
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=4)
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
        defaults={"num_repeats": 3},
    )
    first = materialize(manifest, tmp_path / "a", jobs=2).materialized_fpath.read_bytes()
    second = materialize(manifest, tmp_path / "b", jobs=1).materialized_fpath.read_bytes()
    assert first == second


def test_materialize_completes_the_resume_gate(tmp_path):
    """Gym resumes only when BOTH the materialized file and the output file exist."""
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=2)
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    report = materialize(manifest, tmp_path / "out")
    assert report.materialized_fpath.exists()
    assert report.output_fpath.exists() and report.output_fpath.stat().st_size == 0
    # the name Gym derives from the output path
    assert report.materialized_fpath.name == "rollouts_materialized_inputs.jsonl"


def test_materialize_refuses_to_clobber(tmp_path):
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=1)
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    materialize(manifest, tmp_path / "out")
    with pytest.raises(SweepValidationError, match="already exists"):
        materialize(manifest, tmp_path / "out")


def test_materialize_writes_observed_counts_report(tmp_path):
    """Row counts are a result, not a declaration: the report records what the data held."""
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=5)
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
        defaults={"num_repeats": 3},
    )
    report = materialize(manifest, tmp_path / "out")
    assert report.report_fpath.name == "sweep_report.json"

    doc = json.loads(report.report_fpath.read_text())
    assert doc["total_source_rows"] == 5
    assert doc["total_materialized_rows"] == 15
    assert doc["entries"]["one"] == {"source_rows": 5, "materialized_rows": 15, "num_repeats": 3}
    assert doc["materialized_bytes"] == report.materialized_fpath.stat().st_size
