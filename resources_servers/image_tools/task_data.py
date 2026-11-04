# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the image_tools server (single-step image-tool-selection pivot).

Mirrors ``ImageToolsPivotRunRequest`` (app.py, extra='allow'): every field is wire-Optional.
There is no verifier_metadata. Committed evidence: ``data/example.jsonl`` (5 rows spanning the
bbox, color, and point families) carries ``expected_action`` + ``metadata`` on every row, with
``uuid`` and ``expected_answer`` absent — matching the Optional typing. Production rows come from
the pivot-dataset pipeline; this schema is otherwise derived from the wire model and verify()
code paths. Note count_objects_tool rows carry ``{tolerance, min_size}`` arguments with no color
key — color-family argument scoring compares only ``img_idx`` and ``color``.

The expected action is deliberately kept open: ``extract_expected_action`` (app.py) sources it
three ways in priority order — top-level ``expected_action``, ``metadata['expected_action']``,
then ``json.loads(expected_answer)`` — and its ``arguments`` dict has six per-tool-family shapes
(bbox: bbox_2d/img_idx[/label/factor]; point: point_2d; color: color; pair: img_idx_a/img_idx_b;
multi: img_indices; scalar: degrees/axis/alpha/factor). ``arguments`` may itself arrive as a
JSON-encoded string (verify json.loads it). Typing only the stable {name, arguments} envelope as
an open dict; do not chase closure on the per-family payloads.

Quirk: ``responses_create_params.metadata.extra_body`` is a JSON-encoded string in pivot rows
('{"stop": ["</tool_call>"], ...}'), but that lives under a framework key and is not typed here.
"""

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description=(
            "Stable task identifier; logging and verify-response passthrough only. Typed on the "
            "wire model but absent from pivot rows today (None at runtime)."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_action: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Gold tool call {name: str, arguments: dict|JSON-str}. Primary source for "
            "extract_expected_action; arguments shape is per-tool-family (six families), e.g. "
            "bbox rows carry {bbox_2d: list[int], img_idx: int, label: str, factor?: int} where "
            "label/labels/factor are ignored by argument scoring."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "JSON-encoded fallback encoding of the same {name, arguments} expected-action object "
            "(third-priority source, json.loads'd). Stays a str on the wire; absent from pivot "
            "rows today."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Free-form dict passed through to the verify response. verify() only probes "
            "metadata['expected_action'] as the second-priority expected-action source; observed "
            "pivot subkeys (source_id: str, turn_index: int, tool_name: str, "
            "num_images_in_prefix: int) are provenance."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
