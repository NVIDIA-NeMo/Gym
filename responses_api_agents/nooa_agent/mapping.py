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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from responses_api_agents.nooa_agent.config import NOOAArgumentBinding


Transform = Callable[[Any], Any]

_ALLOWED_SOURCE_PATHS = ("agent_inputs", "responses_create_params.input")


def validate_source_path(source: str) -> None:
    """Reject unsafe or malformed paths before a rollout starts."""

    parts = source.split(".")
    if not source or any(not part or not (part.isidentifier() or part.isdigit()) for part in parts):
        raise ValueError("source must be a non-empty dotted path of identifiers or sequence indexes")

    if any(part.startswith("_") for part in parts) or not any(
        source == allowed or source.startswith(f"{allowed}.") for allowed in _ALLOWED_SOURCE_PATHS
    ):
        raise ValueError(
            f"source {source!r} is not allowed; map only from responses_create_params.input or agent_inputs"
        )


def resolve_source(row: Any, source: str) -> Any:
    """Resolve a dotted path against a complete Gym run row."""

    validate_source_path(source)
    value = row
    for part in source.split("."):
        if isinstance(value, BaseModel):
            if not hasattr(value, part):
                raise ValueError(f"source {source!r} does not exist at {part!r}")
            value = getattr(value, part)
        elif isinstance(value, Mapping):
            if part not in value:
                raise ValueError(f"source {source!r} does not exist at {part!r}")
            value = value[part]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and part.isdigit():
            index = int(part)
            try:
                value = value[index]
            except IndexError as error:
                raise ValueError(f"source {source!r} has no index {index}") from error
        else:
            raise ValueError(f"source {source!r} cannot traverse {part!r}")
    return value


def identity(value: Any) -> Any:
    """Return a mapped value unchanged."""

    return value


def normalize_responses_input(value: Any) -> str | list[Any]:
    """Convert Pydantic Responses input items into plain Python values."""

    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError("Responses input must be a string or a sequence of input items")
    return [item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else item for item in value]


def latest_user_text(value: Any) -> str:
    """Extract text from the most recent user message in Responses input."""

    normalized = normalize_responses_input(value)
    if isinstance(normalized, str):
        return normalized

    for item in reversed(normalized):
        if not isinstance(item, Mapping) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, Mapping)
                and part.get("type") in {"input_text", "output_text"}
                and isinstance(part.get("text"), str)
            ]
            if text_parts:
                return "\n".join(text_parts)
        raise ValueError("latest user message does not contain text content")

    raise ValueError("Responses input does not contain a user message")


TRANSFORMS: Mapping[str, Transform] = {
    "identity": identity,
    "latest_user_text": latest_user_text,
    "normalize_responses_input": normalize_responses_input,
}


def get_transform(name: str) -> Transform:
    """Look up a built-in transform by its configuration name."""

    try:
        return TRANSFORMS[name]
    except KeyError as error:
        raise ValueError(f"unknown transform {name!r}; available transforms: {sorted(TRANSFORMS)}") from error


def materialize_arguments(row: Any, bindings: Mapping[str, NOOAArgumentBinding]) -> dict[str, Any]:
    """Build keyword arguments for a configured NOOA entrypoint."""

    return {
        argument_name: get_transform(binding.transform)(resolve_source(row, binding.source))
        for argument_name, binding in bindings.items()
    }
