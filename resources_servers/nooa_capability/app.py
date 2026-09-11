# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify NOOA capability results with eval_pipeline ExactMatch semantics."""

from __future__ import annotations

import ast
import json
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


class NOOACapabilityResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class NOOACapabilityRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    expected_result: Any


class NOOACapabilityVerifyRequest(NOOACapabilityRunRequest, BaseVerifyRequest):
    pass


class NOOACapabilityVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_result: Any
    actual_result: Any = None
    output_correct: bool


def _parse_value(value: Any) -> Any:
    """Match NOOA v0.0.9 eval_pipeline's ExactMatch value normalization."""

    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _parse_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_parse_value(item) for item in value]
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    try:
        return _parse_value(json.loads(stripped))
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return _parse_value(ast.literal_eval(stripped))
    except (ValueError, SyntaxError):
        return value


def _values_equal(expected: Any, actual: Any) -> bool:
    """Match NOOA v0.0.9 eval_pipeline's recursive equality rules."""

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return expected == actual
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.lower().strip() == actual.lower().strip()
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _values_equal(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual, strict=True)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(expected).issubset(actual) and all(_values_equal(expected[key], actual[key]) for key in expected)
    if type(expected) is type(actual):
        return expected == actual
    return str(expected).lower().strip() == str(actual).lower().strip()


def _last_assistant_text(body: BaseVerifyRequest) -> str | None:
    for output in reversed(body.response.output):
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        texts = [
            content.text
            for content in getattr(output, "content", []) or []
            if isinstance(getattr(content, "text", None), str)
        ]
        if texts:
            return "\n".join(texts).strip()
    return None


class NOOACapabilityResourcesServer(SimpleResourcesServer):
    config: NOOACapabilityResourcesServerConfig

    async def verify(self, body: NOOACapabilityVerifyRequest) -> NOOACapabilityVerifyResponse:
        expected = _parse_value(body.expected_result)
        actual = _parse_value(_last_assistant_text(body))
        if isinstance(actual, dict) and not isinstance(expected, dict):
            for key in ("answer", "response", "result", "output"):
                if key in actual:
                    actual = _parse_value(actual[key])
                    break
        correct = _values_equal(expected, actual)
        return NOOACapabilityVerifyResponse(
            **body.model_dump(exclude={"expected_result"}),
            reward=1.0 if correct else 0.0,
            expected_result=expected,
            actual_result=actual,
            output_correct=correct,
        )


if __name__ == "__main__":
    NOOACapabilityResourcesServer.run_webserver()
