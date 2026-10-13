# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts where two pieces of checkpoint code must agree, checked over many inputs rather than one example.

- What the record check accepts, the writer writes and the reader returns unchanged,
  or the commit fails with a typed error naming the episode.
- Every capture key Gym builds decodes back to the same rollout and attempt.
- Every control route answers an invalid request, or invalid state on disk, with a typed error, never a server error.
"""

import enum
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from nemo_gym._checkpoint.control import CheckpointRecord, JsonPayload
from nemo_gym._checkpoint.errors import CheckpointStateError
from nemo_gym._checkpoint.store import participant_dir, read_participant_state, write_participant_state
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from tests.unit_tests.test_checkpoint_control import FakeParticipant, body, make_client


class PayloadRecord(CheckpointRecord):
    state: JsonPayload


class Color(enum.Enum):
    RED = "red"


@dataclass
class Point:
    x: int


def _circular() -> list:
    value: list = []
    value.append(value)
    return value


PAYLOADS: dict[str, Any] = {
    "nested": {"a": [1, 2.5, None, True, "x"], "b": {"c": [], "d": {}}},
    "key_order": {"z": 1, "a": 2, "m": 3},
    "big_int": 2**70,
    "negative_big_int": -(2**70),
    "big_int_inside": {"hashes": [2**64, 2**100]},
    "nan_and_infinities": [math.nan, math.inf, -math.inf],
    "unicode": {"text": "café ☃ \U0001f600"},
    "int_keys": {1: "a"},
    "int_keys_beside_a_big_int": {1: 2**70},
    "tuple_keys": {(1, 2): "a"},
    "tuple": [1, (2, 3)],
    "datetime": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "uuid": UUID(int=1),
    "enum": Color.RED,
    "dataclass": Point(x=1),
    "set": {1},
    "bytes": b"x",
    "decimal": Decimal("1.5"),
    "object": object(),
    "circular": _circular(),
    # Deeper than orjson supports, so it takes the slower check.
    "deep": json.loads("[" * 400 + "]" * 400),
}


def _as_json(value: Any) -> Any:
    """``value`` as JSON carries it: arrays come back as lists."""
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _as_json(item) for key, item in value.items()}
    return value


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return list(left) == list(right) and all(_same(left[key], right[key]) for key in left)
    return left == right


@pytest.mark.parametrize("name", PAYLOADS)
def test_a_payload_the_check_accepts_is_restored_unchanged_or_fails_the_commit_by_name(
    name: str, tmp_path: Path
) -> None:
    value = PAYLOADS[name]
    try:
        record = PayloadRecord(episode_id=EpisodeId(rollout_id=name.replace("_", "-")), state=value)
    except ValidationError:
        return
    try:
        write_participant_state(
            tmp_path, kind="fake", instance="f", checkpoint_id="c", records=[record.to_json_record()]
        )
    except CheckpointStateError as error:
        assert name.replace("_", "-") in str(error)
        return

    [restored] = read_participant_state(tmp_path, kind="fake", instance="f")[1]
    assert _same(restored["state"], _as_json(value)), f"{name} was accepted but restored as {restored['state']!r}"


@pytest.mark.parametrize("name", ["nested", "key_order", "big_int", "big_int_inside", "nan_and_infinities", "unicode"])
def test_every_payload_json_carries_unchanged_is_accepted(name: str) -> None:
    PayloadRecord(episode_id=EpisodeId(rollout_id="r"), state=PAYLOADS[name])


EXPLICIT_IDS = ["job", "job-a2", "job-a", "job-2", "a-a1-b", "x.y_z-1", "r-a10", "r-a0", "r-a01"]
ATTEMPTS = [None, 0, 1, 3]


@pytest.mark.parametrize("explicit", EXPLICIT_IDS)
@pytest.mark.parametrize("attempt", ATTEMPTS)
def test_every_capture_key_built_from_a_run_body_decodes_back_to_itself(explicit: str, attempt: Any) -> None:
    run_body: dict[str, Any] = {"_ng_rollout_id": explicit}
    if attempt is not None:
        run_body["_ng_attempt_index"] = attempt
    try:
        key = maybe_rollout_id_from_run_body(run_body)
    except ValueError:
        return
    assert EpisodeId.from_capture_key(key).capture_key == key


@pytest.mark.parametrize("task", [0, 7])
@pytest.mark.parametrize("rollout", [0, 3])
@pytest.mark.parametrize("attempt", ATTEMPTS)
def test_every_derived_capture_key_decodes_to_its_attempt(task: int, rollout: int, attempt: Any) -> None:
    run_body: dict[str, Any] = {"_ng_task_index": task, "_ng_rollout_index": rollout}
    if attempt is not None:
        run_body["_ng_attempt_index"] = attempt

    decoded = EpisodeId.from_capture_key(maybe_rollout_id_from_run_body(run_body))

    assert decoded == EpisodeId(rollout_id=f"{task}-{rollout}", attempt=attempt or 0)


INVALID_BODIES: dict[str, list[dict[str, Any]]] = {
    "prepare": [{}, {"checkpoint_id": "", "deadline_ts": 1}, {"checkpoint_id": "../x", "deadline_ts": 1}],
    "renew": [{"checkpoint_id": "c"}, {"checkpoint_id": "c", "deadline_ts": "soon"}],
    "retire": [body(), body(episode_ids=[]), body(episode_ids=[{"rollout_id": "r-a1"}]), body(episode_ids="r")],
    "forget": [body(), body(rollout_ids=[])],
    "commit": [body(), body(checkpoint_dir=""), body(checkpoint_dir="d", extra=1)],
    "restore": [body(), body(checkpoint_dir="d"), body(checkpoint_dir="d", episode_ids=[{"attempt": 1}])],
    "resume": [{"deadline_ts": 1}, body(checkpoint_id=7)],
}


@pytest.mark.parametrize(
    ("operation", "invalid"),
    [(operation, invalid) for operation, bodies in INVALID_BODIES.items() for invalid in bodies],
)
async def test_every_control_route_refuses_an_invalid_request_with_a_client_error(
    operation: str, invalid: dict[str, Any]
) -> None:
    async with make_client(FakeParticipant()) as client:
        response = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=invalid)

    assert 400 <= response.status_code < 500


def _manifest(directory: Path) -> Path:
    return directory / "manifest.json"


def _corrupt_manifest(directory: Path, change: Any) -> None:
    manifest = json.loads(_manifest(directory).read_text())
    _manifest(directory).write_text(change(manifest) if callable(change) else change)


CORRUPTIONS: dict[str, Any] = {
    "not_json": "{not json",
    "a_list": "[]",
    "no_records_file": lambda manifest: json.dumps({k: v for k, v in manifest.items() if k != "records_file"}),
    "no_digest": lambda manifest: json.dumps({k: v for k, v in manifest.items() if k != "records_sha256"}),
    "records_file_outside": lambda manifest: json.dumps({**manifest, "records_file": "../../../outside.jsonl"}),
    "records_file_missing": lambda manifest: json.dumps({**manifest, "records_file": "records-missing.jsonl"}),
    "record_count_a_string": lambda manifest: json.dumps({**manifest, "record_count": "one"}),
    "no_checkpoint_id": lambda manifest: json.dumps({k: v for k, v in manifest.items() if k != "checkpoint_id"}),
}


@pytest.mark.parametrize("corruption", CORRUPTIONS)
async def test_restore_refuses_a_corrupt_manifest_with_a_typed_error(corruption: str, tmp_path: Path) -> None:
    write_participant_state(
        tmp_path,
        kind="fake",
        instance="fake-1",
        checkpoint_id="c",
        records=[{"episode_id": {"rollout_id": "r"}, "value": 1}],
    )
    (tmp_path / "outside.jsonl").write_text(json.dumps({"episode_id": {"rollout_id": "r"}, "value": 1}) + "\n")
    _corrupt_manifest(participant_dir(tmp_path, kind="fake", instance="fake-1"), CORRUPTIONS[corruption])
    participant = FakeParticipant()

    async with make_client(participant) as client:
        response = await client.post(
            "/ng-control/v1/checkpoint/restore",
            json=body("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
        )

    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_checkpoint_state"
    assert participant.executions == {} and participant.accepting is True


async def test_commit_into_a_directory_that_cannot_hold_state_is_a_typed_error(tmp_path: Path) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory")
    participant = FakeParticipant()
    participant.executions["r"] = {"parked": True, "value": 1}

    async with make_client(participant) as client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=body())
        response = await client.post("/ng-control/v1/checkpoint/commit", json=body(checkpoint_dir=str(blocked)))

    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_checkpoint_state"


def test_a_payload_nested_deeper_than_python_recurses_is_refused_not_crashing() -> None:
    with pytest.raises(ValidationError, match="not JSON"):
        PayloadRecord(episode_id=EpisodeId(rollout_id="r"), state=json.loads("[" * 5000 + "]" * 5000))
