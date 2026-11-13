# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay contracts keep asset identity and modeled paths explicit before execution."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from responses_api_agents.ngc_recorded_context.runtime_contract import RuntimeContract, read_runtime_contract


@pytest.fixture
def runtime_data() -> dict:
    return {
        "schema_version": "recorded-context-runtime-v1",
        "input_hashes": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in ("case.json", "bundle.json", "actor-params.json")
        },
        "source_root": "/opt/repos/example",
        "source_schemas": {"coverage": "coverage-v1", "inventory": "inventory-v1", "extra": "extra-v1"},
        "probe_script": "/opt/agent/probe.py",
        "path_aliases": {"/workspace/agent/": "/opt/agent/"},
        "answer_paths": ["/workspace/logs/answer.txt"],
        "workspace_prefix": "/workspace/logs/",
        "status_script": "/opt/agent/status.py",
        "session_id_path": "/workspace/session-id.txt",
        "transcript_template": "/workspace/transcripts/{session_id}.jsonl",
        "optional_runtime_paths": ["/workspace/logs/optional-review.json"],
        "get_wrappers": {"/opt/agent/read.sh": "GET:api.example.invalid"},
        "ordering": {
            "task_id": "synthetic-ordering",
            "task_names": ["synthetic-ordering", "example-ordering"],
            "rubric_version": "ordering-v1",
            "criterion_id": "probe-before-answer",
            "probe_namespace": "sibling",
        },
    }


@pytest.mark.parametrize("pinned", [False, True])
def test_read_retains_asset_identities_and_declared_replay_paths(
    tmp_path: Path, runtime_data: dict, pinned: bool
) -> None:
    path = tmp_path / "runtime.json"
    data = json.dumps(runtime_data).encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest() if pinned else None

    result = read_runtime_contract(path, expected_sha256=digest)

    assert result.model_dump() == runtime_data
    assert path.read_bytes() == data


def test_changed_runtime_bytes_are_rejected_before_they_can_be_parsed(tmp_path: Path, runtime_data: dict) -> None:
    digest = hashlib.sha256(json.dumps(runtime_data).encode()).hexdigest()
    path = tmp_path / "runtime.json"
    corrupted = b'{"schema_version":'
    path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="Replay runtime contract checksum mismatch"):
        read_runtime_contract(path, expected_sha256=digest)

    assert path.read_bytes() == corrupted


def test_missing_runtime_explains_how_to_supply_a_pinned_contract(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Replay assets require runtime.json") as error:
        read_runtime_contract(tmp_path / "runtime.json")

    assert "runtime_contract_path with runtime_contract_sha256" in str(error.value)
    assert "pinned version-2 candidate" in str(error.value)
    assert isinstance(error.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("name", ["../escape.json", "nested/case.json", ".", "..", ""])
def test_asset_manifest_cannot_select_paths_outside_its_package(runtime_data: dict, name: str) -> None:
    runtime_data["input_hashes"][name] = hashlib.sha256(b"extra asset").hexdigest()

    with pytest.raises(ValidationError, match="simple filenames"):
        RuntimeContract.model_validate(runtime_data)


@pytest.mark.parametrize("digest", ["a" * 63, "g" * 64, "A" * 64])
def test_asset_manifest_requires_complete_lowercase_sha256_identities(runtime_data: dict, digest: str) -> None:
    runtime_data["input_hashes"]["case.json"] = digest

    with pytest.raises(ValidationError, match="SHA-256 identities"):
        RuntimeContract.model_validate(runtime_data)


@pytest.mark.parametrize("name", ["case.json", "bundle.json", "actor-params.json"])
def test_each_frozen_actor_asset_must_have_an_identity(runtime_data: dict, name: str) -> None:
    del runtime_data["input_hashes"][name]

    with pytest.raises(ValidationError, match="omits required actor assets"):
        RuntimeContract.model_validate(runtime_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_root", "relative/repository"),
        ("probe_script", "/opt/agent/../outside.py"),
        ("workspace_prefix", "relative/workspace/"),
        ("status_script", "/opt/agent/../outside.py"),
        ("session_id_path", "relative/session-id.txt"),
        ("transcript_template", "/workspace/../transcripts/{session_id}.jsonl"),
        ("answer_paths", ["/workspace/logs/../outside.txt"]),
        ("optional_runtime_paths", ["relative/optional.json"]),
        ("get_wrappers", {"../outside.sh": "GET:api.example.invalid"}),
    ],
)
def test_modeled_paths_cannot_be_relative_or_traverse_parents(runtime_data: dict, field: str, value: object) -> None:
    runtime_data[field] = value

    with pytest.raises(ValidationError, match="absolute and cannot traverse parents"):
        RuntimeContract.model_validate(runtime_data)


@pytest.mark.parametrize("paths", [[], ["/outside/answer.txt"]])
def test_answer_artifacts_must_be_declared_inside_the_modeled_workspace(runtime_data: dict, paths: list[str]) -> None:
    runtime_data["answer_paths"] = paths

    with pytest.raises(ValidationError, match="inside the modeled workspace"):
        RuntimeContract.model_validate(runtime_data)


@pytest.mark.parametrize(
    "template", ["/workspace/transcripts/static.jsonl", "/workspace/transcripts/{session_id}-{session_id}.jsonl"]
)
def test_transcript_path_identifies_exactly_one_session(runtime_data: dict, template: str) -> None:
    runtime_data["transcript_template"] = template

    with pytest.raises(ValidationError, match="exactly one session_id placeholder"):
        RuntimeContract.model_validate(runtime_data)


@pytest.mark.parametrize("fault", ["missing", "unexpected"])
def test_capture_source_schema_declarations_are_complete_and_explicit(runtime_data: dict, fault: str) -> None:
    if fault == "missing":
        del runtime_data["source_schemas"]["coverage"]
    else:
        runtime_data["source_schemas"]["invented"] = "unrecognized-v1"

    with pytest.raises(ValidationError, match="source schemas must be declared explicitly"):
        RuntimeContract.model_validate(runtime_data)
