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

import pytest
import yaml

from nemo_gym.sweep.build import build_sweep, container_config
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


def _manifest(tmp_path, entries, defaults=None, config_overlay=None):
    doc = {"nickname": "testrun", "defaults": defaults or {"num_repeats": 8}, "entries": entries}
    if config_overlay is not None:
        doc["config_overlay"] = config_overlay
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


def _write_manifest_file(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return path


def test_includes_merges_entries_configs_and_overlay(tmp_path):
    """One manifest composes the lane manifests; the design is one deployment, one input."""
    data = tmp_path / "d.jsonl"
    data.write_text('{"agent_ref": "a", "x": 1}\n')
    _write_manifest_file(tmp_path, "child_a.yaml", {
        "nickname": "a", "defaults": {"num_repeats": 4},
        "extra_configs": ["shared.yaml"],
        "config_overlay": {"srv_a": {"k": 1}},
        "entries": [{"label": "ea", "agent": "a", "configs": ["ca.yaml"], "data": str(data)}],
    })
    _write_manifest_file(tmp_path, "child_b.yaml", {
        "nickname": "b", "defaults": {"num_repeats": 8},
        "extra_configs": ["shared.yaml", "b_only.yaml"],
        "config_overlay": {"srv_b": {"k": 2}},
        "entries": [{"label": "eb", "agent": "b", "configs": ["cb.yaml"], "data": str(data)}],
    })
    parent = _write_manifest_file(tmp_path, "all.yaml", {
        "nickname": "combined", "includes": ["child_a.yaml", "child_b.yaml"],
    })
    m = load_manifest(parent)

    assert [e.label for e in m.entries] == ["ea", "eb"]
    assert m.nickname == "combined"
    assert m.config_overlay == {"srv_a": {"k": 1}, "srv_b": {"k": 2}}
    # shared.yaml appears in both children but must not be duplicated
    assert m.config_paths().count("shared.yaml") == 1
    # each child's own default travels with its entries rather than being inherited
    assert {e.label: e.num_repeats for e in m.entries} == {"ea": 4, "eb": 8}


def test_includes_rejects_self_reference(tmp_path):
    parent = _write_manifest_file(tmp_path, "loop.yaml", {
        "nickname": "loop", "includes": ["loop.yaml"], "entries": [],
    })
    with pytest.raises(SweepValidationError, match="includes itself"):
        load_manifest(parent)


def test_own_overlay_wins_over_included(tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text('{"agent_ref": "a", "x": 1}\n')
    _write_manifest_file(tmp_path, "child.yaml", {
        "nickname": "c", "config_overlay": {"srv": {"cap": 64}},
        "entries": [{"label": "e", "agent": "a", "configs": ["c.yaml"], "data": str(data)}],
    })
    parent = _write_manifest_file(tmp_path, "all.yaml", {
        "nickname": "combined", "includes": ["child.yaml"],
        "config_overlay": {"srv": {"cap": 8192}},
    })
    assert load_manifest(parent).config_overlay == {"srv": {"cap": 8192}}
