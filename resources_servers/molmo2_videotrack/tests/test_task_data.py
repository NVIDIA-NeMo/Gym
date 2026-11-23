# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.data import load_task_data_schema, validate_jsonl_rows
from resources_servers.molmo2_videotrack.task_data import Molmo2VerifierMetadata, TaskData

from ..app import Molmo2VideoTrackVerifyRequest


SERVER_DIR = Path(__file__).parents[1]
EXAMPLE_ROWS = [json.loads(line) for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()]


def _validate(*rows: dict[str, Any]) -> Any:
    adapter = load_task_data_schema(SERVER_DIR)
    assert adapter is not None
    return validate_jsonl_rows("molmo2_videotrack", adapter, "rows.jsonl", [json.dumps(row) for row in rows])


def _row(**metadata_overrides: Any) -> dict[str, Any]:
    row = json.loads(json.dumps(EXAMPLE_ROWS[1]))
    row["verifier_metadata"].update(metadata_overrides)
    return row


def test_the_schema_is_the_model_the_verify_request_validates_with() -> None:
    # One declaration: a change to what the verifier requires changes what dataset validation requires.
    assert Molmo2VideoTrackVerifyRequest.model_fields["verifier_metadata"].annotation is Molmo2VerifierMetadata
    assert issubclass(TaskData, Molmo2VerifierMetadata)
    for name, field in Molmo2VerifierMetadata.model_fields.items():
        assert TaskData.model_fields[name].is_required() == field.is_required()
        assert TaskData.model_fields[name].annotation == field.annotation
        assert field.json_schema_extra["legacy_location"] == "verifier_metadata"


def test_committed_example_rows_validate_clean() -> None:
    report = _validate(*EXAMPLE_ROWS)

    assert report.rows == 5
    assert report.clean, report.summary()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_width", None),
        ("image_width", 0),
        ("image_height", -1),
        ("frames", []),
        ("frames", [{"frame": 0, "objects": []}]),
        ("frames", [{"frame": 1, "objects": [{"xy": [1.0], "mask": {}}]}]),
        ("frames", [{"frame": 1, "objects": [{"xy": [1.0, 2.0]}]}]),
    ],
    ids=["no_width", "zero_width", "negative_height", "no_frames", "frame_zero", "one_coordinate", "no_mask"],
)
def test_rows_the_verifier_would_reject_fail_dataset_validation(field: str, value: Any) -> None:
    row = _row(**{field: value})
    if value is None:
        del row["verifier_metadata"][field]

    assert _validate(row).error_rows == 1
    with pytest.raises(ValueError):
        Molmo2VerifierMetadata.model_validate(row["verifier_metadata"])


def test_rows_without_the_optional_fields_validate() -> None:
    row = _row()
    metadata = row["verifier_metadata"]
    for name in ("category", "id", "video", "clip", "expression", "fps"):
        del metadata[name]
    del row["task_id"]
    # A frame with no visible object, and no source frame index.
    metadata["frames"] = [{"frame": 1}, *metadata["frames"][1:]]
    assert len(metadata["frames"][1]["objects"]) == 2

    report = _validate(row)

    assert report.clean, report.summary()


def test_a_verifier_field_placed_beside_verifier_metadata_is_reported() -> None:
    row = _row()
    row["image_width"] = row["verifier_metadata"].pop("image_width")

    assert _validate(row).misplaced_keys == {"image_width": 1}


def test_undeclared_fields_are_reported_but_accepted_by_the_verifier() -> None:
    row = _row(note="extra")

    assert _validate(row).unknown_keys == {"note": 1}
    assert Molmo2VerifierMetadata.model_validate(row["verifier_metadata"]).note == "extra"
