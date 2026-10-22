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
"""Tool specs: one input dataclass yields the CLI subcommand, the LLM tool schema and the dispatch entry.

Only the annotation types the tools use are supported. Anything else raises rather than give a wrong schema.
"""

from __future__ import annotations

import dataclasses
import enum
import typing
from collections.abc import Callable
from typing import Any


_JSON_TYPES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


class Cost(enum.Enum):
    """What a call costs, so the budget and the Tier-2 gate need no tool names."""

    CHEAP = "cheap"  # CPU, milliseconds, run on every candidate
    GPU_BATCH = "gpu_batch"  # GPU, batched, budgeted, never per-candidate
    EXPENSIVE = "expensive"  # GPU, minutes, independent check only


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner type, is_optional) for `T` or `Optional[T]`."""
    if typing.get_origin(annotation) is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _field_schema(annotation: Any) -> dict[str, Any]:
    inner, _ = _unwrap_optional(annotation)
    if inner in _JSON_TYPES:
        return {"type": _JSON_TYPES[inner]}
    if typing.get_origin(inner) is list and typing.get_args(inner) == (str,):
        return {"type": "array", "items": {"type": "string"}}
    # Structured inputs render as open objects. The tool validates the shape itself.
    if typing.get_origin(inner) is list and typing.get_args(inner) == (dict,):
        return {"type": "array", "items": {"type": "object"}}
    if inner is dict or typing.get_origin(inner) is dict:
        return {"type": "object"}
    raise TypeError(f"unsupported tool-input annotation: {annotation!r}")


def json_schema(model: type, for_model: bool = False) -> dict[str, Any]:
    """JSON Schema for a tool-input dataclass. Fields without a default are required.

    `for_model` drops the operator fields, which is what a controller is shown.
    """
    # field.type is a string under `from __future__ import annotations`, so resolve it.
    hints = typing.get_type_hints(model)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in dataclasses.fields(model):
        if for_model and field.metadata.get("operator"):
            continue
        schema = _field_schema(hints[field.name])
        if field.metadata.get("help"):
            schema["description"] = field.metadata["help"]
        properties[field.name] = schema
        has_default = (
            field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        if not has_default:
            required.append(field.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def field(help: str, operator: bool = False, **kwargs: Any) -> Any:
    """`dataclasses.field` carrying the one-line help both projections show.

    `operator=True` marks a deployment knob (a path, a timeout) that stays on the CLI and out of the model's schema.
    """
    metadata = {"help": help}
    if operator:
        metadata["operator"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


def _coerce(model: type, name: str, value: Any) -> Any:
    """Cast a scalar the model sent as a string (`"10"`) into the field's type.

    A value that will not convert is passed through for the tool's own validation to report.
    """
    types = {f.name: f.type for f in dataclasses.fields(model)}
    declared = types.get(name)
    if not isinstance(value, str) or declared not in ("int", "float", "bool", int, float, bool):
        return value
    try:
        if declared in ("bool", bool):
            return value.strip().lower() in ("1", "true", "yes")
        return (int if declared in ("int", int) else float)(value)
    except ValueError:
        return value


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """One tool, and everything the three projections need to render it."""

    name: str  # "fold.eternafold": group.verb, the registry name and, with "_" for ".", the LLM tool name
    summary: str
    cost: Cost
    model: type  # input dataclass
    fn: Callable[[Any], Any]
    #: What the task must supply for this tool to be offered, so no tool uses withheld information.
    needs: frozenset[str] = frozenset()
    #: The steps a composite performs. It is offered instead of its steps, never alongside them.
    steps: tuple[str, ...] = ()

    @property
    def group(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def verb(self) -> str:
        return self.name.split(".", 1)[1]

    def input_schema(self) -> dict[str, Any]:
        return json_schema(self.model)

    @property
    def llm_name(self) -> str:
        """The tool name as a model sees it. Providers forbid dots in function names."""
        return self.name.replace(".", "_")

    def llm_tool(self) -> dict[str, Any]:
        """Anthropic tool definition. The OpenAI shape wraps this unchanged."""
        # The model sees its spend, so it should also see which calls cost.
        charged = {
            "CHEAP": "cheap",
            "GPU_BATCH": "charges gpu_seconds",
            "EXPENSIVE": "charges gpu_seconds, minutes per call",
        }
        return {
            "name": self.llm_name,
            "description": f"{self.summary} [{charged.get(self.cost.name, self.cost.name)}]",
            "input_schema": json_schema(self.model, for_model=True),
        }

    def call(self, **kwargs: Any) -> Any:
        """Run the tool. An empty value ("", null, []) for a defaulted field is dropped, so the default applies."""
        optional = {
            f.name
            for f in dataclasses.fields(self.model)
            if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        }  # type: ignore[misc]
        cleaned = {
            k: _coerce(self.model, k, v) for k, v in kwargs.items() if not (k in optional and v in ("", None, []))
        }
        known = {f.name for f in dataclasses.fields(self.model)}
        unknown = sorted(set(cleaned) - known)
        if unknown:
            # Name every unknown argument and what the tool takes, not just the first offender.
            raise TypeError(
                f"{self.name}: unknown argument(s) {', '.join(unknown)}; this tool takes: {', '.join(sorted(known))}"
            )
        return self.fn(self.model(**cleaned))
