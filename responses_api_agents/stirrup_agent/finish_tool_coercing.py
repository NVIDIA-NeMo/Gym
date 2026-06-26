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
"""Gym-side replacement for stirrup's SIMPLE_FINISH_TOOL with `paths` coercion.

Upstream stirrup's FinishParams expects ``paths: list[str]``. DSv4-Pro served
by vLLM 0.20.0 (the wedu image ``vllm-deepseekv4-v0200-cu130-ray-arm64.sqsh``)
emits non-string-typed tool-call args as JSON-encoded strings: the model emits
``<｜DSML｜parameter ... string="false">[...]</｜DSML｜parameter>`` per its
chat-template, but vLLM's ``--tool-call-parser deepseek_v4`` in 0.20.0 doesn't
honor the ``string="false"`` flag and forwards the inner JSON as a literal
string. The Gym client therefore receives ``"paths": "[\\"foo.txt\\"]"``
(string) instead of ``"paths": ["foo.txt"]`` (list). Stirrup's pydantic
validator rejects with ``Input should be a valid list (type=list_type)`` and
the agent loops forever on the same broken shape.

Upstream vLLM PR #41801 (merged 2026-05-06) fixes this, but the wedu image
predates the merge. This module bypasses the parser bug at the Gym schema
layer with a ``@field_validator(mode="before")`` that:

1. Passes through real lists unchanged.
2. JSON-decodes a string that looks like a JSON array (``"[...]"``).
3. Wraps a bare filename string into a single-element list.
4. Leaves other types alone so pydantic raises a clean structural error.

Observed broken shapes from the r5 GDPVal client log (verbatim):

  "paths": "[]"
  "paths": "[\\"article.txt\\"]"
  "paths": "[\\"article.txt\\", \\"chart.jpg\\"]"
  "paths": "Case Feedback.docx"            # bare filename
  "paths": {...}                            # rare; pydantic will reject

Once the wedu image is rebuilt against vLLM main ≥ #41801, this coercion is
a no-op (the first ``isinstance(v, list)`` branch will always take) and the
module can be removed.
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from stirrup.constants import FINISH_TOOL_NAME
from stirrup.core.models import Tool, ToolUseCountMetadata
from stirrup.tools.finish import _validating_finish_executor


class CoercingFinishParams(BaseModel):
    """Same shape as stirrup.tools.finish.FinishParams, with ``paths`` coercion."""

    reason: Annotated[str, Field(description="Reason for finishing.")]
    paths: Annotated[
        list[str],
        Field(description=("List of file paths created or modified. Do not include directories, only files.")),
    ]

    @field_validator("paths", mode="before")
    @classmethod
    def _coerce_paths(cls, v):
        if isinstance(v, list):
            return [str(p) for p in v]
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return [str(p) for p in parsed]
            if stripped:
                return [stripped]
            return []
        return v


COERCING_FINISH_TOOL: Tool[CoercingFinishParams, ToolUseCountMetadata] = Tool[
    CoercingFinishParams, ToolUseCountMetadata
](
    name=FINISH_TOOL_NAME,
    description=(
        "Signal task completion with a reason. Use when the task is finished "
        "or cannot proceed further. Note that you will need a separate turn "
        "to finish."
    ),
    parameters=CoercingFinishParams,
    executor=_validating_finish_executor,
)
