# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate complete Codex catalog records without changing their serialized content."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")


class _Truncation(_Record):
    mode: Literal["bytes", "tokens"]
    limit: int


class _Reasoning(_Record):
    effort: str = Field(min_length=1)
    description: str


class _ServiceTier(_Record):
    id: str
    name: str
    description: str


class _Upgrade(_Record):
    model: str
    migration_markdown: str


class _Availability(_Record):
    message: str


class _Messages(_Record):
    instructions_template: str | None = None
    instructions_variables: dict[str, str | None] | None = None
    approvals: dict[str, str | None] | None = None
    auto_review: dict[str, str | None] | None = None


class _Model(_Record):
    # Require the complete serialized record, including fields for which the CLI can supply defaults.
    # This is a wire-format check, not a replacement implementation of Codex model resolution.
    slug: str = Field(min_length=1)
    display_name: str
    description: str | None
    supported_reasoning_levels: list[_Reasoning]
    shell_type: Literal["default", "local", "unified_exec", "disabled", "shell_command"]
    visibility: Literal["list", "hide", "none"]
    supported_in_api: bool
    priority: int
    additional_speed_tiers: list[str]
    service_tiers: list[_ServiceTier]
    availability_nux: _Availability | None
    upgrade: _Upgrade | None
    base_instructions: str
    include_skills_usage_instructions: bool
    supports_reasoning_summaries: bool
    default_reasoning_summary: Literal["auto", "concise", "detailed", "none"]
    support_verbosity: bool
    default_verbosity: Literal["low", "medium", "high"] | None
    apply_patch_tool_type: Literal["freeform"] | None
    web_search_tool_type: Literal["text", "text_and_image"]
    truncation_policy: _Truncation
    supports_parallel_tool_calls: bool
    supports_image_detail_original: bool
    effective_context_window_percent: int
    experimental_supported_tools: list[str]
    input_modalities: list[Literal["text", "image"]]
    supports_search_tool: bool
    use_responses_lite: bool
    model_messages: _Messages | None = None
    default_reasoning_level: str | None = None
    default_service_tier: str | None = None
    context_window: int | None = None
    max_context_window: int | None = None
    auto_compact_token_limit: int | None = None
    comp_hash: str | None = None
    auto_review_model_override: str | None = None
    tool_mode: str | None = None
    multi_agent_version: str | None = None


class _Catalog(_Record):
    models: list[_Model] = Field(min_length=1)


def load_model_catalog(path: Path, *, model: str | None) -> tuple[str, bool]:
    """Return unchanged UTF-8 JSON and whether the exact model has no context metadata."""
    text = path.read_bytes().decode("utf-8")
    catalog = _Catalog.model_validate_json(text)
    slugs = [entry.slug for entry in catalog.models]
    if len(set(slugs)) != len(slugs):
        raise ValueError("model_catalog_json must not contain duplicate model slugs")
    if model is None or model not in slugs:
        raise ValueError(f"model_catalog_json requires an exact entry for the effective CLI model {model!r}")
    entry = catalog.models[slugs.index(model)]
    unknown_context = entry.context_window is None and entry.max_context_window is None
    if unknown_context and entry.auto_compact_token_limit is not None:
        raise ValueError("model_catalog_json with unknown context must also omit auto_compact_token_limit")
    return text, unknown_context
