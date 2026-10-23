# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, lossless addresses for context and candidate evidence."""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from resources_servers.partner_if.evidence import MAX_EVIDENCE_CHARS, EvidenceBundle


Origin = Literal["context", "candidate"]


@dataclass(frozen=True)
class EvidenceSource:
    origin: Origin
    text: str
    value_type: str = "string"
    container_types: tuple[str, ...] = ()
    supports_candidate: bool = False

    @property
    def lines(self) -> list[str]:
        # Keep line terminators: concatenating an inclusive range recovers exact
        # source text, including CRLF, trailing spaces and Unicode whitespace.
        return self.text.splitlines(keepends=True)

    def payload(self) -> dict:
        return {
            "origin": self.origin,
            "value_type": self.value_type,
            "container_types": list(self.container_types),
            "supports_candidate": self.supports_candidate,
            "lines": [{"number": i, "text": line} for i, line in enumerate(self.lines, 1)],
        }


def evidence_sources(evidence: EvidenceBundle) -> dict[str, EvidenceSource]:
    """Address all fields without promoting frozen input history to new behavior.

    Structured values use JSON-pointer field paths; string leaves remain literal
    text rather than JSON-escaped copies. Numeric/boolean/null and empty-container
    leaves use their JSON representation. Source IDs depend only on evidence.
    """
    sources: dict[str, EvidenceSource] = {}

    def escape(name: str) -> str:
        return name.replace("~", "~0").replace("/", "~1")

    def add(
        identifier: str,
        text: str,
        origin: Origin,
        *,
        value_type: str = "string",
        container_types: tuple[str, ...] = (),
        supports_candidate: bool = True,
    ) -> None:
        if identifier in sources:
            raise ValueError(f"Duplicate evidence source: {identifier}")
        sources[identifier] = EvidenceSource(
            origin=origin,
            text=text,
            value_type=value_type,
            container_types=container_types,
            supports_candidate=supports_candidate and origin == "candidate",
        )

    def walk(
        identifier: str,
        value: JsonValue,
        origin: Origin,
        container_types: tuple[str, ...] = (),
        *,
        supports_candidate: bool = True,
    ) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                walk(
                    f"{identifier}/{escape(key)}",
                    child,
                    origin,
                    (*container_types, "object"),
                    supports_candidate=supports_candidate,
                )
        elif isinstance(value, list) and value:
            for index, child in enumerate(value):
                walk(
                    f"{identifier}/{index}",
                    child,
                    origin,
                    (*container_types, "array"),
                    supports_candidate=supports_candidate,
                )
        else:
            value_type = (
                "string"
                if isinstance(value, str)
                else "object"
                if isinstance(value, dict)
                else "array"
                if isinstance(value, list)
                else "null"
                if value is None
                else "boolean"
                if isinstance(value, bool)
                else "number"
            )
            add(
                identifier,
                value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                origin,
                value_type=value_type,
                container_types=container_types,
                supports_candidate=supports_candidate,
            )

    add("answer", evidence.answer, "candidate")
    for index, message in enumerate(evidence.context):
        walk(f"context:{index}", message, "context")
    if evidence.instructions is not None:
        add("instructions", evidence.instructions, "context")
    for index, tool in enumerate(evidence.tools):
        walk(f"tool:{index}", tool, "context")
    for name, text in evidence.documents.items():
        add(f"document:{escape(name)}", text, "context")
    for name, artifact in evidence.artifacts.items():
        # An oversized artifact is a capture error for the items that need it, not evidence for the rest.
        if artifact.status == "present" and len(artifact.text) <= MAX_EVIDENCE_CHARS:
            add(f"artifact:{escape(name)}", artifact.text, "candidate")
    seen = set()
    for event in evidence.trajectory:
        if event.source_index in seen:
            raise ValueError(f"Duplicate event source_index: {event.source_index}")
        seen.add(event.source_index)
        origin: Origin = "context" if event.kind == "context" else "candidate"
        for field, value in event.model_dump(mode="json", exclude_none=True).items():
            walk(
                f"event:{event.source_index}/{field}",
                value,
                origin,
                ("object",),
                supports_candidate=field in {"text", "arguments", "tool_name"},
            )
    return sources
