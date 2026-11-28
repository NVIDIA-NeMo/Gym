# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge selection shared by the Environment Server and the dependency-light task verifier."""

from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict, cast


STANDARD_JUDGE_MODELS = ("claude-sonnet-4-6", "gpt-5.5")


class LabJudgeConfig(TypedDict):
    """One judge's API route; omitted transport settings inherit the shared endpoint."""

    model_name: str
    base_url: NotRequired[str]
    api_key: NotRequired[str]
    reasoning_effort: NotRequired[str]


def validate_judges(value: object) -> list[LabJudgeConfig]:
    """Require one explicit judge or the official pair without exposing credentials in errors."""
    error = (
        "Configure lab_judges with one or two entries, each containing a nonempty model_name "
        "and optional base_url, api_key, and reasoning_effort."
    )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not 1 <= len(value) <= 2:
        raise ValueError(error)
    judges: list[LabJudgeConfig] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) - {"model_name", "base_url", "api_key", "reasoning_effort"}:
            raise ValueError(error)
        model = entry.get("model_name")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(error)
        judge: dict[str, str] = {"model_name": model.strip()}
        for field in ("base_url", "api_key", "reasoning_effort"):
            setting = entry.get(field)
            if setting is not None:
                if not isinstance(setting, str):
                    raise ValueError(error)
                judge[field] = setting
        judges.append(cast(LabJudgeConfig, judge))
    if len(judges) == 2 and {judge["model_name"].split("/")[-1] for judge in judges} != set(STANDARD_JUDGE_MODELS):
        raise ValueError("Two lab_judges must be the official pair: claude-sonnet-4-6 and gpt-5.5")
    return judges
