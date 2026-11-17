# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.data import load_task_data_schema, normalize_task_fields
from resources_servers.video_caption.task_data import TaskData


SERVER_DIR = Path(__file__).resolve().parents[1]


def row(**overrides):
    base = {
        "atomic_events": {"visual_events": ["A person walks.", "A person sits."], "pure_audio_events": 3},
        "reference_caption": "A person walks in and sits down.",
    }
    return base | overrides


def test_schema_loads_through_the_data_tooling_and_accepts_the_example_row():
    adapter = load_task_data_schema(SERVER_DIR)
    assert adapter is not None
    example = json.loads((SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()[0])
    task_fields, _ = normalize_task_fields(example)
    data = adapter.validate_python(task_fields)
    assert len(data.atomic_events.visual_events) == 3
    assert data.reference_caption


def test_reference_caption_is_optional_and_audio_fields_ride_along():
    data = TaskData.model_validate(row(reference_caption=None))
    assert data.reference_caption is None
    assert data.atomic_events.model_extra == {"pure_audio_events": 3}


@pytest.mark.parametrize(
    "bad",
    [
        {"atomic_events": {"visual_events": []}},
        {"atomic_events": {"visual_events": [" "]}},
        {"atomic_events": {"visual_events": ["Walking", ""]}},
        {"atomic_events": {}},
        {"reference_caption": "   "},
    ],
)
def test_rows_the_verifier_would_mask_fail_validation(bad):
    with pytest.raises(ValidationError):
        TaskData.model_validate(row(**bad))


def test_missing_events_fail_validation():
    with pytest.raises(ValidationError):
        TaskData.model_validate({"reference_caption": "text"})


def test_request_model_and_task_data_declare_the_same_row() -> None:
    # The wire model (app.py) and the data schema (task_data.py) are declared twice on purpose: the schema rejects
    # blank text at validation time, the wire model lets it through so verify() can mask the row. Same fields though.
    from resources_servers.video_caption import app, task_data

    def shape(model):
        return {name: field.is_required() for name, field in model.model_fields.items()}

    in_metadata = {
        name: field.is_required()
        for name, field in task_data.TaskData.model_fields.items()
        if (field.json_schema_extra or {}).get("legacy_location") == "verifier_metadata"
    }
    assert shape(app.CaptionVerifierMetadata) == in_metadata
    assert shape(app.VisualAtomicEvents) == shape(task_data.VisualAtomicEvents)
    top_level = set(task_data.TaskData.model_fields) - set(in_metadata)
    assert top_level == {"source_index"}
    assert "source_index" in app.VideoCaptionVerifyRequest.model_fields
