# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-facing OSWorld protocol contracts, independent of execution and history selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from responses_api_agents.osworld_agent.trajectory import stable_id


@dataclass(frozen=True)
class ProtocolMessageContext:
    """Selected observations and serialized assistant turns for a protocol renderer.

    Turn indices include the current observation at ``len(observations)``.
    The renderer owns message roles and media ordering; the shared history
    policy owns which of these turns may retain an image.
    """

    instruction: str
    system_prompt: str
    observations: Sequence[Mapping[str, Any]]
    assistant_history: Sequence[str]
    current_observation: Mapping[str, Any]
    image_turns: frozenset[int]
    guidance: str = ""


@dataclass(frozen=True)
class OSWorldModelProtocol:
    """Frozen model wire format with a parser and an optional message renderer.

    A new model can reuse the rollout loop by registering one of these
    contracts. Parser and renderer identifiers must change when their wire
    behavior changes. Existing Markdown protocols retain their contract IDs.
    """

    protocol_id: str
    parser_id: str
    instruction_template: str
    step_template: str
    text_history_template: str
    system_prompt_thinking: str
    system_prompt_non_thinking: str
    assistant_history_template_thinking: str
    assistant_history_template_non_thinking: str
    parser: Callable[..., tuple[str, list[str], dict[str, Any]]]
    schema_version: int = 1
    message_renderer: Callable[[ProtocolMessageContext], list[dict[str, Any]]] | None = None
    message_renderer_id: str | None = None
    chat_template_kwargs_thinking: Mapping[str, Any] | None = None
    chat_template_kwargs_non_thinking: Mapping[str, Any] | None = None
    parse_retry_template: str | None = None

    def __post_init__(self) -> None:
        if (self.message_renderer is None) != (self.message_renderer_id is None):
            raise ValueError("A protocol message_renderer requires a message_renderer_id, and vice versa")
        if self.message_renderer is not None and not self.message_renderer_id:
            raise ValueError("message_renderer_id must be nonempty")

    def system_prompt(self, *, thinking: bool, password: str) -> str:
        template = self.system_prompt_thinking if thinking else self.system_prompt_non_thinking
        return template.replace("{password}", password)

    def assistant_history_template(self, *, thinking: bool) -> str:
        return self.assistant_history_template_thinking if thinking else self.assistant_history_template_non_thinking

    def chat_template_kwargs(self, *, thinking: bool) -> dict[str, Any] | None:
        values = self.chat_template_kwargs_thinking if thinking else self.chat_template_kwargs_non_thinking
        return dict(values) if values is not None else None

    def to_contract(self, *, thinking: bool) -> dict[str, Any]:
        wire_material = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "parser_id": self.parser_id,
            "thinking": thinking,
            "instruction_template": self.instruction_template,
            "step_template": self.step_template,
            "text_history_template": self.text_history_template,
            "system_prompt_template": self.system_prompt_thinking if thinking else self.system_prompt_non_thinking,
            "assistant_history_template": self.assistant_history_template(thinking=thinking),
        }
        extensions: dict[str, Any] = {}
        if self.message_renderer_id is not None:
            extensions["message_renderer_id"] = self.message_renderer_id
        template_kwargs = self.chat_template_kwargs(thinking=thinking)
        if template_kwargs is not None:
            extensions["chat_template_kwargs"] = template_kwargs
        if self.parse_retry_template is not None:
            extensions["parse_retry_template"] = self.parse_retry_template
        wire_material.update(extensions)
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "parser_id": self.parser_id,
            "thinking": thinking,
            "wire_contract_id": stable_id("osworld-model-wire", wire_material),
            **extensions,
        }
