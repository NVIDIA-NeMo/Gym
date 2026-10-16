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

Qwen3.8-Flash-Next served with ``--tool-call-parser qwen3_xml`` renders the
array one level deeper: ``paths`` arrives as a real list whose single element
is the *rendered* list, so the pass-through branch above accepted it verbatim
and the whole rendering was existence-checked as one filename. Observed in
GDPVal invocation f5dccfaddbdd5722 (verbatim, from the finish tool results):

  "paths": ["['/root/output/Award_Spending_Rate_Analysis.xlsx']"]
  "paths": ["[/root/output/Daily Shipment Manifest.xlsx, /root/output/Daily Shipment Manifest.pdf]"]

Neither survives ``json.loads``: the first is a Python repr (single quotes),
the second has no quoting at all. Elements are therefore parsed with
``json.loads``, then ``ast.literal_eval``, then a comma split of the bracket
body, and any element that yields a list is spliced into the result.

Once every serving parser forwards arrays faithfully this coercion is a no-op
(no element starts with ``[``) and the module can be removed.
"""

from __future__ import annotations

import ast
import json
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.finish import _validating_finish_executor


def _parse_rendered_list(value: str) -> list[str] | None:
    """The paths inside a string holding a rendered list, or None if it is not one.

    Serving-side tool-call parsers render an array argument with whatever
    quoting they please, so try the strict readers first and fall back to
    splitting the bracket body. A path containing a comma would split wrongly,
    but the executor existence-checks every path, so that surfaces as a normal
    "file does not exist" rejection rather than a silent bad submission.
    """
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    for load in (json.loads, ast.literal_eval):
        try:
            parsed = load(stripped)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(p) for p in parsed]
    body = stripped[1:-1].strip()
    if not body:
        return []
    return [part.strip().strip("\"'") for part in body.split(",") if part.strip()]


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
        if isinstance(v, str):
            rendered = _parse_rendered_list(v)
            if rendered is not None:
                return rendered
            stripped = v.strip()
            return [stripped] if stripped else []
        if isinstance(v, list):
            paths: list[str] = []
            for item in v:
                rendered = _parse_rendered_list(item) if isinstance(item, str) else None
                if rendered is None:
                    paths.append(str(item))
                else:
                    paths.extend(rendered)
            return paths
        return v


COERCING_FINISH_TOOL: Tool[CoercingFinishParams, ToolUseCountMetadata] = Tool[
    CoercingFinishParams, ToolUseCountMetadata
](
    name=DEFAULT_FINISH_TOOL_NAME,
    description=(
        "Signal task completion with a reason. Use when the task is finished "
        "or cannot proceed further. Note that you will need a separate turn "
        "to finish."
    ),
    parameters=CoercingFinishParams,
    executor=_validating_finish_executor,
)


ABANDON_FINISH_TOOL_NAME = "abandon_task_finish"


class AbandonFinishParams(BaseModel):
    """Early-exit finish tool: abandon the task without submitting deliverables."""

    reason: Annotated[str, Field(description="Brief reason the task cannot be completed.")]


async def _abandon_finish_executor(params: AbandonFinishParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)


# Second finish tool (GDPval-AA v2 early-exit). Stirrup >= 0.1.9 accepts a list
# of finish tools (PR #49); the agent may call this instead of ``finish`` when
# it genuinely cannot complete the task, submitting no files.
ABANDON_FINISH_TOOL: Tool[AbandonFinishParams, ToolUseCountMetadata] = Tool[AbandonFinishParams, ToolUseCountMetadata](
    name=ABANDON_FINISH_TOOL_NAME,
    description=(
        "Signal that you do not believe the task can be completed, with a "
        "brief reason, instead of submitting files. Use only when required "
        "inputs are missing, a hard dependency is unavailable, or the request "
        "is incoherent. Do not use it to escape difficulty. Note that you will "
        "need a separate turn to finish."
    ),
    parameters=AbandonFinishParams,
    executor=_abandon_finish_executor,
)
