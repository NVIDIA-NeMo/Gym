# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from pydantic import ValidationError

from resources_servers.sol_execbench.problem_store import NATIVE_REVISION, ProblemManifest, problem_digest
from resources_servers.sol_execbench.scoring import WorkloadAnchor, load_anchor_manifest, score_workload


def fixture_manifest() -> tuple[ProblemManifest, dict]:
    definition = {"name": "synthetic_identity", "inputs": {}, "outputs": {}}
    workloads = [{"uuid": "synthetic-a", "axes": {}, "inputs": {}}, {"uuid": "synthetic-b", "axes": {}, "inputs": {}}]
    task_id = "synthetic/identity"
    digest = problem_digest(task_id, definition, workloads, [])
    manifest = ProblemManifest(
        schema_version=1,
        source={"repository": "synthetic", "revision": "fixture-v1"},
        native_revision=NATIVE_REVISION,
        problems=[
            {
                "task_id": task_id,
                "problem_digest": digest,
                "definition": definition,
                "workloads": workloads,
                "assets": [],
            }
        ],
    )
    anchors = {
        "schema_version": 1,
        "problem_manifest_sha256": "a" * 64,
        "target_hardware": "B200",
        "provenance": {"source": "synthetic test fixture", "revision": "fixture-v1"},
        "tasks": {
            task_id: {
                "problem_digest": digest,
                "workloads": {uuid: {"baseline_ms": 10.0, "sol_ms": 2.0} for uuid in ("synthetic-a", "synthetic-b")},
            }
        },
    }
    return manifest, anchors


def load_fixture(tmp_path, *, mutate=None, data=None, sha256=None, target_hardware="B200"):
    manifest, anchors = fixture_manifest()
    if mutate is not None:
        mutate(anchors)
    payload = data if data is not None else json.dumps(anchors).encode()
    path = tmp_path / "anchors.json"
    path.write_bytes(payload)
    return load_anchor_manifest(
        path,
        sha256 if sha256 is not None else hashlib.sha256(payload).hexdigest(),
        problem_manifest=manifest,
        problem_manifest_sha256="a" * 64,
        target_hardware=target_hardware,
    )


def test_score_anchors_and_monotonicity():
    anchor = WorkloadAnchor(baseline_ms=10.0, sol_ms=2.0)
    scores = [score_workload(latency, anchor) for latency in (2.0, 6.0, 10.0, 26.0)]
    assert scores == pytest.approx([1.0, 2.0 / 3.0, 0.5, 0.25])
    assert all(faster > slower for faster, slower in zip(scores, scores[1:]))


@pytest.mark.parametrize("candidate", [0.0, -1.0, float("nan"), float("inf"), -float("inf"), True, "3", 1.99])
def test_invalid_or_subsol_latency_is_an_audit_not_a_clipped_reward(candidate):
    with pytest.raises(ValueError):
        score_workload(candidate, WorkloadAnchor(baseline_ms=10.0, sol_ms=2.0))


@pytest.mark.parametrize(
    "baseline,sol",
    [
        (0.0, 0.0),
        (-1.0, 0.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (True, 0.0),
        ("10", 2.0),
    ],
)
def test_invalid_anchor_rejected(baseline, sol):
    with pytest.raises(ValidationError):
        WorkloadAnchor(baseline_ms=baseline, sol_ms=sol)


def test_zero_sol_bound_is_supported():
    assert score_workload(10.0, WorkloadAnchor(baseline_ms=10.0, sol_ms=0.0)) == 0.5


def test_loads_hash_bound_complete_manifest(tmp_path):
    result = load_fixture(tmp_path)
    assert result.provenance.source == "synthetic test fixture"
    assert set(result.tasks["synthetic/identity"].workloads) == {"synthetic-a", "synthetic-b"}
    assert score_workload(6.0, result.tasks["synthetic/identity"].workloads["synthetic-a"]) == pytest.approx(2 / 3)


@pytest.mark.parametrize("sha256", ["0" * 64, "invalid", "A" * 64])
def test_rejects_bad_file_hash(tmp_path, sha256):
    with pytest.raises(ValueError, match="Anchor manifest SHA256 mismatch"):
        load_fixture(tmp_path, sha256=sha256)


def test_rejects_wrong_manifest_binding_and_hardware(tmp_path):
    with pytest.raises(ValueError, match="problem manifest SHA256 mismatch"):
        load_fixture(tmp_path, mutate=lambda d: d.update(problem_manifest_sha256="b" * 64))
    with pytest.raises(ValueError, match="target hardware mismatch"):
        load_fixture(tmp_path, target_hardware="LOCAL")
    with pytest.raises(ValueError, match="problem digest mismatch"):
        load_fixture(tmp_path, mutate=lambda d: d["tasks"]["synthetic/identity"].update(problem_digest="b" * 64))


@pytest.mark.parametrize("extra", [False, True])
def test_rejects_missing_or_extra_tasks(tmp_path, extra):
    def mutate(data):
        if extra:
            data["tasks"]["unexpected/task"] = data["tasks"]["synthetic/identity"]
        else:
            data["tasks"]["replacement/task"] = data["tasks"].pop("synthetic/identity")

    with pytest.raises(ValueError, match="Anchor tasks must exactly match"):
        load_fixture(tmp_path, mutate=mutate)


@pytest.mark.parametrize("extra", [False, True])
def test_rejects_missing_or_extra_workload_uuids(tmp_path, extra):
    def mutate(data):
        workloads = data["tasks"]["synthetic/identity"]["workloads"]
        if extra:
            workloads["unexpected"] = {"baseline_ms": 10.0, "sol_ms": 2.0}
        else:
            del workloads["synthetic-b"]

    with pytest.raises(ValueError, match="Anchor workload UUIDs must exactly match"):
        load_fixture(tmp_path, mutate=mutate)


@pytest.mark.parametrize("field", ["schema_version", "synthetic/identity", "synthetic-a", "baseline_ms"])
def test_rejects_duplicate_json_keys_at_any_depth(tmp_path, field):
    _, payload = fixture_manifest()
    data = json.dumps(payload)
    replacements = {
        "schema_version": '"schema_version": 1, "schema_version": 1',
        "synthetic/identity": '"synthetic/identity": {}, "synthetic/identity":',
        "synthetic-a": '"synthetic-a": {}, "synthetic-a":',
        "baseline_ms": '"baseline_ms": 10.0, "baseline_ms":',
    }
    needle = '"schema_version": 1' if field == "schema_version" else json.dumps(field) + ":"
    data = data.replace(needle, replacements[field], 1)
    with pytest.raises(ValueError, match="Duplicate anchor JSON key"):
        load_fixture(tmp_path, data=data.encode())


@pytest.mark.parametrize("location", ["root", "provenance", "task", "workload"])
def test_rejects_unknown_fields(tmp_path, location):
    def mutate(data):
        target = {
            "root": data,
            "provenance": data["provenance"],
            "task": data["tasks"]["synthetic/identity"],
            "workload": data["tasks"]["synthetic/identity"]["workloads"]["synthetic-a"],
        }[location]
        target["unexpected"] = "ignored?"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_fixture(tmp_path, mutate=mutate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("problem_manifest_sha256", "bad"),
        ("target_hardware", "H100"),
        ("provenance", {"source": " ", "revision": "v1"}),
    ],
)
def test_rejects_invalid_schema_values(tmp_path, field, value):
    with pytest.raises(ValidationError):
        load_fixture(tmp_path, mutate=lambda d: d.update({field: value}))
