# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Round-trip tests for CoercingFinishParams.

Covers the 4 known broken `paths` shapes observed in the DSv4-Pro GDPVal r5
client log (vLLM 0.20.0 + --tool-call-parser deepseek_v4, pre-PR #41801) plus
the canonical correct shapes that should round-trip unchanged.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from responses_api_agents.stirrup_agent.finish_tool_coercing import (
    COERCING_FINISH_TOOL,
    CoercingFinishParams,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Empty list as JSON-encoded string — the most common bad shape.
        ('{"reason":"ok","paths":"[]"}', []),
        # Single-element JSON-encoded array.
        ('{"reason":"ok","paths":"[\\"a.txt\\"]"}', ["a.txt"]),
        # Multi-element JSON-encoded array, with whitespace.
        (
            '{"reason":"ok","paths":"[\\"a.txt\\", \\"b.pdf\\"]"}',
            ["a.txt", "b.pdf"],
        ),
        # Bare filename string — wrap into single-element list.
        ('{"reason":"ok","paths":"single.docx"}', ["single.docx"]),
        # Correct list shape — passthrough.
        ('{"reason":"ok","paths":[]}', []),
        ('{"reason":"ok","paths":["a.txt"]}', ["a.txt"]),
        ('{"reason":"ok","paths":["a.txt","b.pdf"]}', ["a.txt", "b.pdf"]),
    ],
)
def test_paths_coercion(raw: str, expected: list[str]) -> None:
    """Each known-bad and known-good shape round-trips to a clean list[str]."""
    p = CoercingFinishParams.model_validate_json(raw)
    assert p.paths == expected
    assert all(isinstance(x, str) for x in p.paths)


def test_reason_required() -> None:
    """`reason` is still required — coercion only changes `paths` semantics."""
    with pytest.raises(ValidationError):
        CoercingFinishParams.model_validate_json('{"paths":[]}')


def test_paths_required() -> None:
    """`paths` is still required — coercion has no default."""
    with pytest.raises(ValidationError):
        CoercingFinishParams.model_validate_json('{"reason":"ok"}')


def test_paths_dict_rejected_loudly() -> None:
    """An object for `paths` (not list / str) raises a clean pydantic error
    instead of silently coercing."""
    with pytest.raises(ValidationError):
        CoercingFinishParams.model_validate_json('{"reason":"ok","paths":{"a":1}}')


def test_paths_with_non_string_items_stringified() -> None:
    """A real list with non-string items gets stringified (defensive)."""
    p = CoercingFinishParams.model_validate({"reason": "ok", "paths": [1, "b.txt"]})
    assert p.paths == ["1", "b.txt"]


def test_tool_carries_coercing_params() -> None:
    """The exported tool references CoercingFinishParams as its parameters
    schema — sanity check that the wire-up doesn't fall back to upstream."""
    assert COERCING_FINISH_TOOL.parameters is CoercingFinishParams
    assert COERCING_FINISH_TOOL.name == "finish"
