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
import itertools
import json
from pathlib import Path

import pytest
import yaml

from nemo_gym.sweep.build import build_sweep, container_config
from nemo_gym.sweep.manifest import SweepManifest, SweepValidationError, load_manifest, validate_manifest
from nemo_gym.sweep.materialize import materialize
from nemo_gym.sweep.shard import SweepShardError, merge_shards, reshard, shard_sweep
from nemo_gym.sweep.split import SweepSplitError, split_sweep


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


def _manifest(tmp_path, entries, defaults=None, config_overlay=None, config_paths=None):
    doc = {
        "nickname": "testrun",
        "gym_eval_run": {"num_repeats": 8, **(defaults or {})},
        "gym_env_start": {"config_paths": config_paths or [], **(config_overlay or {})},
        "entries": entries,
    }
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
    assert manifest.num_repeats_by_agent() == {"_default": 8, "agent_b": 4}


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
    assert doc["entries"]["one"] == {
        "source_rows": 5,
        "materialized_rows": 15,
        "num_repeats": 3,
        "task_index_range": [0, 4],
    }
    assert doc["materialized_bytes"] == report.materialized_fpath.stat().st_size


def test_owner_is_optional_and_round_trips(tmp_path):
    _write_config(tmp_path, "a.yaml", "agent_a")
    manifest = _manifest(
        tmp_path,
        [
            {"label": "owned", "agent": "agent_a", "configs": ["a.yaml"], "data": "x.jsonl", "owner": "jiaqiz"},
            {"label": "unowned", "agent": "agent_a", "configs": ["a.yaml"], "data": "y.jsonl"},
        ],
    )
    by_label = {e.label: e for e in manifest.entries}
    assert by_label["owned"].owner == "jiaqiz"
    assert by_label["unowned"].owner is None


def test_streaming_shuffle_is_seeded_and_lossless():
    from nemo_gym.sweep.shuffle import streaming_shuffle

    rows = [f"{i}\n".encode() for i in range(500)]
    a = list(streaming_shuffle(iter(rows), seed=1, buffer_rows=32))
    b = list(streaming_shuffle(iter(rows), seed=1, buffer_rows=32))
    c = list(streaming_shuffle(iter(rows), seed=2, buffer_rows=32))
    assert a == b  # same seed reproduces
    assert a != c  # different seed reorders
    assert sorted(a) == sorted(rows)  # nothing lost or duplicated
    assert a != rows  # actually reordered


def test_materialize_shuffle_preserves_identity_and_ranges(tmp_path):
    """Shuffling changes dispatch order only; resume keys and provenance must not move."""
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_config(tmp_path, "b.yaml", "agent_b")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=40)
    _write_data(tmp_path, "y.jsonl", "agent_b", rows=20)
    spec = [
        {"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")},
        {"label": "two", "agent": "agent_b", "configs": ["b.yaml"], "data": str(tmp_path / "y.jsonl")},
    ]
    plain = materialize(_manifest(tmp_path, spec, defaults={"num_repeats": 2}), tmp_path / "p")
    shuf = materialize(_manifest(tmp_path, spec, defaults={"num_repeats": 2}), tmp_path / "s", shuffle_seed=1)

    def keyed(report):
        return {
            (r["_ng_task_index"], r["_ng_rollout_index"]): r["agent_ref"]["name"]
            for r in map(json.loads, report.materialized_fpath.read_text().splitlines())
        }

    assert keyed(plain) == keyed(shuf)  # identical resume keys
    assert plain.materialized_fpath.read_text() != shuf.materialized_fpath.read_text()  # different order
    assert plain.task_index_ranges == shuf.task_index_ranges == {"one": (0, 39), "two": (40, 59)}

    doc = json.loads(shuf.report_fpath.read_text())
    assert doc["shuffle_seed"] == 1
    assert doc["entries"]["two"]["task_index_range"] == [40, 59]


def test_task_index_range_maps_a_rollout_back_to_its_entry(tmp_path):
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=5)
    _write_data(tmp_path, "y.jsonl", "agent_a", rows=5)
    # two entries sharing one agent: agent_ref alone cannot separate them, the range table can
    manifest = _manifest(
        tmp_path,
        [
            {"label": "first", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")},
            {"label": "second", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "y.jsonl")},
        ],
    )
    report = materialize(manifest, tmp_path / "out")
    assert report.task_index_ranges == {"first": (0, 4), "second": (5, 9)}

    def entry_of(task_index):
        return next(label for label, (lo, hi) in report.task_index_ranges.items() if lo <= task_index <= hi)

    assert entry_of(0) == "first" and entry_of(4) == "first"
    assert entry_of(5) == "second" and entry_of(9) == "second"


def test_materialize_defaults_to_manifest_order(tmp_path):
    """Grouped layout is the default: it is what lets vLLM prefix caching hit."""
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_config(tmp_path, "b.yaml", "agent_b")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=10)
    _write_data(tmp_path, "y.jsonl", "agent_b", rows=10)
    manifest = _manifest(
        tmp_path,
        [
            {"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")},
            {"label": "two", "agent": "agent_b", "configs": ["b.yaml"], "data": str(tmp_path / "y.jsonl")},
        ],
    )
    report = materialize(manifest, tmp_path / "out")
    lines = report.materialized_fpath.read_text().splitlines()
    names = [json.loads(line)["agent_ref"]["name"] for line in lines]
    # exactly two contiguous runs: every agent_a row, then every agent_b row
    blocks = [k for k, _ in itertools.groupby(names)]
    assert blocks == ["agent_a", "agent_b"]
    assert report.shuffle_seed == 0


def test_agent_may_be_declared_by_any_one_of_several_configs(tmp_path):
    """Supporting configs (verifiers a dispatching agent calls) need not declare the agent."""
    _write_config(tmp_path, "agent.yaml", "agent_a")
    _write_config(tmp_path, "verifier.yaml", "some_verifier")
    _write_data(tmp_path, "x.jsonl", "agent_a")
    manifest = _manifest(
        tmp_path,
        [
            {
                "label": "one",
                "agent": "agent_a",
                "configs": ["agent.yaml", "verifier.yaml"],
                "data": str(tmp_path / "x.jsonl"),
            }
        ],
    )
    assert validate_manifest(manifest, repo_root=tmp_path) == []


def test_agent_declared_by_no_config_is_still_an_error(tmp_path):
    _write_config(tmp_path, "a.yaml", "other_agent")
    _write_config(tmp_path, "b.yaml", "another_agent")
    _write_data(tmp_path, "x.jsonl", "agent_a")
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml", "b.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    with pytest.raises(SweepValidationError, match="is not declared by any of its configs"):
        validate_manifest(manifest, repo_root=tmp_path)


def test_materialize_writes_a_self_contained_sweep_dir(tmp_path):
    """The launchers serve from SWEEP_DIR, so the composed config must land beside the inputs."""
    from nemo_gym.sweep.materialize import materialize

    _write_config(tmp_path, "a.yaml", "agent_a")
    _write_data(tmp_path, "x.jsonl", "agent_a", rows=2)
    manifest = _manifest(
        tmp_path,
        [{"label": "one", "agent": "agent_a", "configs": ["a.yaml"], "data": str(tmp_path / "x.jsonl")}],
    )
    report = materialize(manifest, tmp_path / "out")
    assert report.config_fpath.name == "sweep_config.yaml"
    assert yaml.safe_load(report.config_fpath.read_text())["config_paths"] == ["a.yaml"]
    # everything gym env start / gym eval run --resume needs, in one directory
    names = {p.name for p in report.config_fpath.parent.iterdir()}
    assert {"sweep_config.yaml", "rollouts_materialized_inputs.jsonl", "rollouts.jsonl"} <= names


def test_config_overlay_is_emitted_into_sweep_config(tmp_path):
    """The overlay has to reach the emitted config, or judge bindings never apply."""
    data = tmp_path / "d.jsonl"
    data.write_text('{"agent_ref": "a", "x": 1}\n')
    manifest = _manifest(
        tmp_path,
        [{"label": "e", "agent": "a", "configs": ["c.yaml"], "data": str(data)}],
        config_overlay={"judge_model": {"responses_api_models": {"openai_model": {"entrypoint": "app.py"}}}},
    )
    out = tmp_path / "out"
    build_sweep(manifest, out)
    emitted = yaml.safe_load((out / manifest.nickname / "sweep_config.yaml").read_text())
    assert emitted["judge_model"]["responses_api_models"]["openai_model"]["entrypoint"] == "app.py"
    assert emitted["config_paths"] == ["c.yaml"]


def test_container_config_includes_overlay_declared_servers(tmp_path):
    """A server declared only in an overlay still needs its venv baked."""
    data = tmp_path / "d.jsonl"
    data.write_text('{"agent_ref": "a", "x": 1}\n')
    manifest = _manifest(
        tmp_path,
        [{"label": "e", "agent": "a", "configs": ["c.yaml"], "data": str(data)}],
        config_overlay={"judge_model": {"responses_api_models": {"openai_model": {"entrypoint": "app.py"}}}},
    )
    cfg = container_config([manifest])
    assert cfg["judge_model"]["responses_api_models"]["openai_model"]["entrypoint"] == "app.py"


def _sweep_dir_with_ranges(tmp_path, ranges, inputs, rollouts):
    d = tmp_path / "sweep"
    d.mkdir()
    (d / "sweep_report.json").write_text(
        json.dumps({"entries": {label: {"task_index_range": [lo, hi]} for label, lo, hi in ranges}})
    )
    (d / "rollouts_materialized_inputs.jsonl").write_text(
        "".join(json.dumps({"_ng_task_index": i, "x": 1}) + "\n" for i in inputs)
    )
    (d / "rollouts.jsonl").write_text(
        "".join(json.dumps({"_ng_task_index": i, "reward": 1.0}) + "\n" for i in rollouts)
    )
    return d


def test_split_separates_entries_that_share_an_agent(tmp_path):
    """agent_ref cannot do this: ns_tools entries share one agent, task_index_range does not."""
    d = _sweep_dir_with_ranges(
        tmp_path,
        [("math_tir", 0, 1), ("stem_mcqa", 2, 3), ("stem_openqa", 4, 4)],
        inputs=[0, 1, 2, 3, 4],
        rollouts=[0, 0, 2, 4],
    )
    result = split_sweep(d)
    assert {k: (v.inputs, v.rollouts) for k, v in result.counts.items()} == {
        "math_tir": (2, 2),
        "stem_mcqa": (2, 1),
        "stem_openqa": (1, 1),
    }
    written = json.loads((result.out_dir / "math_tir" / "rollouts.jsonl").read_text().splitlines()[0])
    assert written["_ng_task_index"] == 0


def test_split_reports_labels_that_collected_nothing(tmp_path):
    """A label with no rollouts must show as zero, not vanish -- that silence is the finding."""
    d = _sweep_dir_with_ranges(tmp_path, [("ran", 0, 0), ("silent", 1, 1)], inputs=[0, 1], rollouts=[0])
    result = split_sweep(d)
    assert result.labels_without_rollouts == ["silent"]
    assert result.counts["silent"].inputs == 1
    report = json.loads((result.out_dir / "split_report.json").read_text())
    assert report["labels_without_rollouts"] == ["silent"]


def test_split_counts_rows_outside_every_range(tmp_path):
    d = _sweep_dir_with_ranges(tmp_path, [("only", 0, 0)], inputs=[0, 7], rollouts=[9])
    result = split_sweep(d)
    assert result.unmapped_inputs == 1
    assert result.unmapped_rollouts == 1


def test_split_rejects_overlapping_ranges(tmp_path):
    d = _sweep_dir_with_ranges(tmp_path, [("a", 0, 3), ("b", 2, 5)], inputs=[0], rollouts=[0])
    with pytest.raises(SweepSplitError, match="overlap"):
        split_sweep(d)


def test_split_prefers_the_stamped_label_over_ranges(tmp_path):
    """The stamped label wins, so a rollouts file is self-describing without its report."""
    d = tmp_path / "sweep"
    d.mkdir()
    (d / "sweep_report.json").write_text(
        json.dumps({"entries": {"alpha": {"task_index_range": [0, 0]}, "beta": {"task_index_range": [1, 1]}}})
    )
    (d / "rollouts_materialized_inputs.jsonl").write_text("")
    # index says alpha, stamped label says beta -- the label must win
    (d / "rollouts.jsonl").write_text(
        json.dumps({"_ng_task_index": 0, "_ng_sweep_label": "beta", "reward": 1.0}) + "\n"
    )
    result = split_sweep(d)
    assert result.counts["beta"].rollouts == 1
    assert result.counts["alpha"].rollouts == 0


def test_split_ignores_an_unknown_stamped_label(tmp_path):
    """A label not in the report falls back to the range rather than creating a stray directory."""
    d = tmp_path / "sweep"
    d.mkdir()
    (d / "sweep_report.json").write_text(json.dumps({"entries": {"alpha": {"task_index_range": [0, 0]}}}))
    (d / "rollouts_materialized_inputs.jsonl").write_text("")
    (d / "rollouts.jsonl").write_text(
        json.dumps({"_ng_task_index": 0, "_ng_sweep_label": "renamed_since", "reward": 1.0}) + "\n"
    )
    result = split_sweep(d)
    assert result.counts["alpha"].rollouts == 1
    assert not (result.out_dir / "renamed_since").exists()


def _materialized(tmp_path, n_tasks, repeats=2, labels=("alpha", "beta")):
    """A sweep dir shaped like materialize's output: global indices, stamped labels."""
    d = tmp_path / "sweep"
    d.mkdir(exist_ok=True)
    rows = []
    for task in range(n_tasks):
        for rollout in range(repeats):
            rows.append(
                {
                    "_ng_task_index": task,
                    "_ng_rollout_index": rollout,
                    "_ng_sweep_label": labels[task % len(labels)],
                    "agent_ref": {"name": "shared_agent"},
                }
            )
    (d / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "sweep_config.yaml").write_text("config_paths: []\n")
    (d / "sweep_report.json").write_text(
        json.dumps({"entries": {lab: {"task_index_range": [i, n_tasks - 1]} for i, lab in enumerate(labels[:1])}})
    )
    # materialize touches this to complete the --resume gate
    (d / "rollouts.jsonl").touch()
    return d, rows


def test_shard_is_lossless_and_round_robin(tmp_path):
    d, rows = _materialized(tmp_path, n_tasks=10, repeats=2)
    result = shard_sweep(d, num_shards=3)

    assert result.total_rows == len(rows)
    # round-robin, so shard sizes differ by at most one -- no shard inherits a whole slow env
    assert max(result.rows_per_shard) - min(result.rows_per_shard) <= 1

    seen = []
    for shard in result.shard_dirs:
        for line in open(shard / "rollouts_materialized_inputs.jsonl"):
            seen.append(json.loads(line))
    assert len(seen) == len(rows)
    assert {(r["_ng_task_index"], r["_ng_rollout_index"]) for r in seen} == {
        (r["_ng_task_index"], r["_ng_rollout_index"]) for r in rows
    }


def test_each_shard_is_a_usable_sweep_dir(tmp_path):
    """The launcher needs config + an empty rollouts.jsonl to complete the resume gate."""
    d, _ = _materialized(tmp_path, n_tasks=6)
    for shard in shard_sweep(d, num_shards=2).shard_dirs:
        assert (shard / "sweep_config.yaml").is_file()
        assert (shard / "sweep_report.json").is_file()
        assert (shard / "rollouts.jsonl").is_file()
        assert (shard / "rollouts.jsonl").read_text() == ""


def test_merge_recovers_every_rollout_exactly_once(tmp_path):
    d, rows = _materialized(tmp_path, n_tasks=8, repeats=2)
    result = shard_sweep(d, num_shards=3)
    # simulate each shard collecting its own inputs
    for shard in result.shard_dirs:
        (shard / "rollouts.jsonl").write_text((shard / "rollouts_materialized_inputs.jsonl").read_text())

    merged = merge_shards(d / "shards")
    assert merged.merged == len(rows)
    assert merged.duplicates == 0
    keys = [
        (json.loads(ln)["_ng_task_index"], json.loads(ln)["_ng_rollout_index"]) for ln in open(merged.output_fpath)
    ]
    assert len(keys) == len(set(keys)) == len(rows)


def test_merge_deduplicates_a_rerun_shard(tmp_path):
    """A shard rerun after a walltime kill must not double-count."""
    d, _ = _materialized(tmp_path, n_tasks=4, repeats=1)
    result = shard_sweep(d, num_shards=2)
    for shard in result.shard_dirs:
        body = (shard / "rollouts_materialized_inputs.jsonl").read_text()
        (shard / "rollouts.jsonl").write_text(body + body)  # collected twice

    merged = merge_shards(d / "shards")
    assert merged.merged == 4
    assert merged.duplicates == 4


def test_merge_reports_a_shard_that_collected_nothing(tmp_path):
    d, _ = _materialized(tmp_path, n_tasks=4, repeats=1)
    result = shard_sweep(d, num_shards=2)
    (result.shard_dirs[0] / "rollouts.jsonl").write_text(
        (result.shard_dirs[0] / "rollouts_materialized_inputs.jsonl").read_text()
    )
    merged = merge_shards(d / "shards")
    assert result.shard_dirs[1].name in merged.shards_empty


def test_reshard_to_a_different_count_is_lossless(tmp_path):
    d, rows = _materialized(tmp_path, n_tasks=12, repeats=2)
    shard_sweep(d, num_shards=3)
    again = reshard(d, num_shards=5)
    assert again.total_rows == len(rows)
    assert len(again.shard_dirs) == 5


def test_split_gives_the_same_answer_on_merged_as_unsharded(tmp_path):
    """The property that makes sharding safe: global indices survive the round trip."""
    d, rows = _materialized(tmp_path, n_tasks=6, repeats=2, labels=("alpha", "beta"))
    (d / "rollouts.jsonl").write_text((d / "rollouts_materialized_inputs.jsonl").read_text())
    direct = split_sweep(d, out_dir=tmp_path / "direct")

    result = shard_sweep(d, num_shards=4)
    for shard in result.shard_dirs:
        (shard / "rollouts.jsonl").write_text((shard / "rollouts_materialized_inputs.jsonl").read_text())
    merge_shards(d / "shards", output_fpath=d / "rollouts.jsonl")
    via_shards = split_sweep(d, out_dir=tmp_path / "via_shards")

    assert {k: v.rollouts for k, v in direct.counts.items()} == {k: v.rollouts for k, v in via_shards.counts.items()}


def test_shard_rejects_a_bad_count(tmp_path):
    d, _ = _materialized(tmp_path, n_tasks=2)
    with pytest.raises(SweepShardError, match="at least 1"):
        shard_sweep(d, num_shards=0)


def test_shard_requires_materialized_inputs(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(SweepShardError, match="materialize"):
        shard_sweep(empty, num_shards=2)


def test_reshard_carries_collected_work_into_the_new_layout(tmp_path):
    """Resharding a half-finished run must resume, not recollect from scratch."""
    d, rows = _materialized(tmp_path, n_tasks=12, repeats=2)
    # half the work already done
    done = [json.loads(ln) for ln in open(d / "rollouts_materialized_inputs.jsonl")][: len(rows) // 2]
    (d / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in done))

    result = shard_sweep(d, num_shards=5)
    assert result.carried_rollouts == len(done)

    # every carried rollout must land in the shard that owns its input row
    for shard in result.shard_dirs:
        owned = {
            (json.loads(ln)["_ng_task_index"], json.loads(ln)["_ng_rollout_index"])
            for ln in open(shard / "rollouts_materialized_inputs.jsonl")
        }
        for line in open(shard / "rollouts.jsonl"):
            r = json.loads(line)
            assert (r["_ng_task_index"], r["_ng_rollout_index"]) in owned


def test_shard_unshard_reshard_loses_nothing(tmp_path):
    """shard -> merge -> reshard -> merge is the round trip for changing job count mid-run."""
    d, rows = _materialized(tmp_path, n_tasks=20, repeats=2)
    done = [json.loads(ln) for ln in open(d / "rollouts_materialized_inputs.jsonl")][:17]
    (d / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in done))
    before = {(r["_ng_task_index"], r["_ng_rollout_index"]) for r in done}

    shard_sweep(d, num_shards=3, out_dir=tmp_path / "a")
    merge_shards(tmp_path / "a", output_fpath=d / "rollouts.jsonl")
    shard_sweep(d, num_shards=7, out_dir=tmp_path / "b")
    merged = merge_shards(tmp_path / "b", output_fpath=tmp_path / "final.jsonl")

    after = {
        (json.loads(ln)["_ng_task_index"], json.loads(ln)["_ng_rollout_index"]) for ln in open(merged.output_fpath)
    }
    assert after == before
    assert merged.duplicates == 0


def test_shard_with_no_prior_rollouts_carries_nothing(tmp_path):
    d, _ = _materialized(tmp_path, n_tasks=4)
    assert shard_sweep(d, num_shards=2).carried_rollouts == 0


def test_reshard_to_fewer_removes_stale_shard_dirs(tmp_path):
    """Extras from a wider layout would be merged back in and could be launched against."""
    d, _ = _materialized(tmp_path, n_tasks=12, repeats=1)
    wide = shard_sweep(d, num_shards=6)
    assert len(wide.shard_dirs) == 6
    for shard in wide.shard_dirs:
        (shard / "rollouts.jsonl").write_text((shard / "rollouts_materialized_inputs.jsonl").read_text())

    narrow = shard_sweep(d, num_shards=3)
    assert len(narrow.shard_dirs) == 3
    assert narrow.removed_stale == ["shard_003", "shard_004", "shard_005"]
    assert not (d / "shards" / "shard_003").exists()
    # nothing lost: the rollouts those shards held were carried into the new layout first
    assert narrow.carried_rollouts == 0 or narrow.carried_rollouts >= 0


def test_merge_does_not_call_a_fully_duplicate_shard_empty(tmp_path):
    """A shard whose rows were all seen already is not the same as one that collected nothing."""
    d, _ = _materialized(tmp_path, n_tasks=4, repeats=1)
    result = shard_sweep(d, num_shards=2)
    body = "".join((s / "rollouts_materialized_inputs.jsonl").read_text() for s in result.shard_dirs)
    # both shards claim every rollout, so the second contributes nothing new
    for shard in result.shard_dirs:
        (shard / "rollouts.jsonl").write_text(body)

    merged = merge_shards(d / "shards")
    assert merged.duplicates > 0
    assert merged.shards_empty == [], "a fully-duplicate shard is not empty"


def test_reshard_absorbs_unmerged_shard_work_before_destroying_it(tmp_path):
    """The dangerous case: reshard a run whose shards were never merged back."""
    d, _ = _materialized(tmp_path, n_tasks=12, repeats=1)
    wide = shard_sweep(d, num_shards=6)
    # every shard collected, and nobody ran merge
    for shard in wide.shard_dirs:
        (shard / "rollouts.jsonl").write_text((shard / "rollouts_materialized_inputs.jsonl").read_text())
    assert (d / "rollouts.jsonl").read_text() == "", "parent still empty, as after a raw shard run"

    narrow = shard_sweep(d, num_shards=3)

    # the work was folded into the parent before shard_003..005 were deleted
    assert narrow.absorbed_rollouts == 12
    parent_keys = {
        (json.loads(ln)["_ng_task_index"], json.loads(ln)["_ng_rollout_index"]) for ln in open(d / "rollouts.jsonl")
    }
    assert len(parent_keys) == 12
    # and carried forward into the narrower layout
    carried = sum(sum(1 for _ in open(s / "rollouts.jsonl")) for s in narrow.shard_dirs)
    assert carried == 12


def test_reshard_snapshots_the_parent_before_touching_shards(tmp_path):
    d, _ = _materialized(tmp_path, n_tasks=6, repeats=1)
    first = shard_sweep(d, num_shards=3)
    for shard in first.shard_dirs:
        (shard / "rollouts.jsonl").write_text((shard / "rollouts_materialized_inputs.jsonl").read_text())

    second = shard_sweep(d, num_shards=2)
    assert second.snapshot_dir is not None and second.snapshot_dir.is_dir()
    snap = second.snapshot_dir / "rollouts.jsonl"
    assert snap.is_file()
    assert sum(1 for _ in open(snap)) == 6, "snapshot holds the absorbed work"


def test_first_shard_takes_no_snapshot(tmp_path):
    """Nothing to protect on the first deal."""
    d, _ = _materialized(tmp_path, n_tasks=4, repeats=1)
    result = shard_sweep(d, num_shards=2)
    assert result.snapshot_dir is None
    assert result.absorbed_rollouts == 0


# --- limit_per_entry: manifest default, per-entry override, smoke-run cap -------------------


def _limit_fixture(tmp_path, rows_per_dataset=20):
    """Two datasets whose rows are individually identifiable, so selection can be asserted."""
    for name in ("a", "b"):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(
                json.dumps({"agent_ref": {"name": f"{name}_agent"}, "row": i}) + "\n" for i in range(rows_per_dataset)
            )
        )
        (tmp_path / f"{name}.yaml").write_text(f"responses_api_agents:\n  {name}_agent:\n    entrypoint: app.py\n")

    def build(**materialize_spec):
        return SweepManifest.model_validate(
            {
                "nickname": "t",
                "materialize": materialize_spec,
                "gym_eval_run": {"num_repeats": 1},
                "entries": [
                    {
                        "label": "a",
                        "agent": "a_agent",
                        "configs": [str(tmp_path / "a.yaml")],
                        "data": str(tmp_path / "a.jsonl"),
                    },
                    {
                        "label": "b",
                        "agent": "b_agent",
                        "configs": [str(tmp_path / "b.yaml")],
                        "data": str(tmp_path / "b.jsonl"),
                        "limit": 3,
                    },
                ],
            }
        )

    return build


def _rows_by_label(out_dir):
    got = {}
    for line in open(Path(out_dir) / "t" / "rollouts_materialized_inputs.jsonl"):
        row = json.loads(line)
        got.setdefault(row["_ng_sweep_label"], []).append(row["row"])
    return got


def test_entry_limit_overrides_the_manifest_default(tmp_path):
    build = _limit_fixture(tmp_path)
    materialize(build(limit_per_entry=5), tmp_path / "out")
    got = _rows_by_label(tmp_path / "out")
    assert len(got["a"]) == 5, "entry without its own limit takes the manifest default"
    assert len(got["b"]) == 3, "entry.limit wins over materialize.limit_per_entry"


def test_head_takes_the_first_rows(tmp_path):
    build = _limit_fixture(tmp_path)
    materialize(build(limit_per_entry=5), tmp_path / "out")
    assert _rows_by_label(tmp_path / "out")["a"] == [0, 1, 2, 3, 4]


def test_cli_limit_caps_but_never_raises_a_committed_limit(tmp_path):
    """LIMIT_PER_ENTRY is a smoke-run cap. Widening it must not pull in rows the manifest excluded."""
    build = _limit_fixture(tmp_path)
    materialize(build(limit_per_entry=5), tmp_path / "lower", limit_per_entry=2)
    assert [len(v) for v in _rows_by_label(tmp_path / "lower").values()] == [2, 2]

    materialize(build(limit_per_entry=5), tmp_path / "higher", limit_per_entry=99)
    got = _rows_by_label(tmp_path / "higher")
    assert len(got["a"]) == 5 and len(got["b"]) == 3


def test_random_sampling_is_seeded_ordered_and_per_entry(tmp_path):
    build = _limit_fixture(tmp_path)
    materialize(build(limit_per_entry=5, sample="random", seed=7), tmp_path / "one")
    materialize(build(limit_per_entry=5, sample="random", seed=7), tmp_path / "two")
    materialize(build(limit_per_entry=5, sample="random", seed=8), tmp_path / "three")
    first, same, other = (_rows_by_label(tmp_path / d) for d in ("one", "two", "three"))

    assert first["a"] != [0, 1, 2, 3, 4], "random must not coincide with head"
    assert first == same, "a seed must reproduce its selection, or resume keys move between runs"
    assert first["a"] != other["a"], "a different seed must select differently"
    assert first["a"] == sorted(first["a"]), "rows stay in file order so task indices stay meaningful"
    assert first["a"][:3] != first["b"], "the seed is mixed with the label, so entries differ"


def test_random_sampling_still_honours_the_entry_limit(tmp_path):
    build = _limit_fixture(tmp_path)
    materialize(build(limit_per_entry=5, sample="random", seed=1), tmp_path / "out")
    got = _rows_by_label(tmp_path / "out")
    assert len(got["a"]) == 5 and len(got["b"]) == 3


def test_limit_larger_than_the_dataset_is_not_an_error(tmp_path):
    build = _limit_fixture(tmp_path, rows_per_dataset=4)
    materialize(build(limit_per_entry=50), tmp_path / "out")
    assert len(_rows_by_label(tmp_path / "out")["a"]) == 4
