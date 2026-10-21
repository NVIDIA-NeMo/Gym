# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-model-call sampling carried by a harness's immutable Model Server URL.

The prefix lets unmodified harnesses inherit the same controls for ordinary turns,
summaries and delegated calls. It adds no request registry or network hop. These
controls are caller settings; a Model Server's configured sampling overrides still
take precedence when it builds the provider request.
"""

import base64
import binascii
import json
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT
from nemo_gym.rollout_correlation import scope_with_routed_path


MODEL_SAMPLING_PATH_PREFIX = "ng-sampling"
_MAX_ENCODED_LENGTH = 1024
_ENCODED_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_GENERATION_PATHS = frozenset({"/v1/chat/completions", "/v1/responses"})


class ModelCallSampling(BaseModel):
    """Per-call controls; the token cap preserves smaller harness-selected limits."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    max_output_tokens: int | None = Field(default=None, gt=0)

    @classmethod
    def from_request(cls, body: Any) -> "ModelCallSampling":
        """Extract and validate supported controls at an agent's request boundary."""
        try:
            return cls.model_validate({name: getattr(body, name, None) for name in cls.model_fields})
        except ValidationError as error:
            raise HTTPException(422, detail=error.errors(include_url=False, include_input=False)) from error


def model_sampling_base_url(base_url: str, sampling: ModelCallSampling | None) -> str:
    """Prefix a Gym endpoint while retaining its rollout, capture and API path."""
    if sampling is None or not (controls := sampling.model_dump(exclude_none=True)):
        return base_url
    encoded = base64.urlsafe_b64encode(json.dumps(controls, separators=(",", ":")).encode()).decode().rstrip("=")
    if len(encoded) > _MAX_ENCODED_LENGTH:
        raise ValueError("Model-call sampling exceeds the supported URL length")
    parts = urlsplit(base_url)
    return urlunsplit(parts._replace(path=f"/{MODEL_SAMPLING_PATH_PREFIX}/{encoded}{parts.path}"))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate model-call sampling field")
        result[name] = value
    return result


def _decode_sampling(encoded: str) -> ModelCallSampling:
    if not encoded or len(encoded) > _MAX_ENCODED_LENGTH or not _ENCODED_PATTERN.fullmatch(encoded):
        raise ValueError("Invalid model-call sampling URL")
    raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
    return ModelCallSampling.model_validate(json.loads(raw, object_pairs_hook=_unique_object))


def _generation_path(path: str) -> str:
    """Recognize generation without consuming the rollout prefix needed by capture."""
    if path.startswith(f"/{ROLLOUT_PATH_PREFIX}/"):
        parts = path.split("/", 3)
        path = f"/{parts[3]}" if len(parts) == 4 and parts[2] else ""
        if path.startswith(f"/{TOKEN_CAPTURE_PATH_SEGMENT}/"):
            path = path[len(TOKEN_CAPTURE_PATH_SEGMENT) + 1 :]
    return path


class ModelSamplingMiddleware:
    """Normalize sampling before policy admission and capture observe a model call.

    Other routes retain their body and method. Streaming response bytes are never
    inspected or changed; only the outbound model request is adapted.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        prefix = f"/{MODEL_SAMPLING_PATH_PREFIX}"
        if scope["type"] != "http" or not (path == prefix or path.startswith(f"{prefix}/")):
            await self.app(scope, receive, send)
            return

        try:
            _, _, encoded, remainder = path.split("/", 3)
            sampling = _decode_sampling(encoded)
        except (ValueError, UnicodeDecodeError, binascii.Error):
            await JSONResponse({"detail": "Invalid model-call sampling URL"}, status_code=422)(scope, receive, send)
            return

        scope = scope_with_routed_path(scope, f"/{remainder}")
        controls = sampling.model_dump(exclude_none=True)
        generation_path = _generation_path(scope["path"])
        if scope["method"] != "POST" or generation_path not in _GENERATION_PATHS or not controls:
            await self.app(scope, receive, send)
            return

        chunks = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            body = json.loads(chunks)
            if not isinstance(body, dict):
                raise ValueError("Model request must be a JSON object")
        except (ValueError, UnicodeDecodeError):
            await JSONResponse({"detail": "Model request must be a JSON object"}, status_code=422)(
                scope, receive, send
            )
            return
        if "max_output_tokens" in controls:
            # Harnesses can reserve less output near the context limit or for a
            # summary. The caller's cap must never raise those smaller limits.
            aliases = (
                ("max_tokens", "max_completion_tokens")
                if generation_path == "/v1/chat/completions"
                else ("max_output_tokens",)
            )
            limit = controls.pop("max_output_tokens")
            for name in aliases:
                harness_limit = body.pop(name, None)
                if type(harness_limit) is int and harness_limit > 0:
                    limit = min(limit, harness_limit)
            controls[aliases[-1]] = limit
        body.update(controls)
        rewritten = json.dumps(body, separators=(",", ":")).encode()
        scope = {
            **scope,
            "headers": [(key, value) for key, value in scope.get("headers", []) if key.lower() != b"content-length"]
            + [(b"content-length", str(len(rewritten)).encode())],
        }
        replayed = False

        async def rewritten_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": rewritten, "more_body": False}
            return await receive()

        await self.app(scope, rewritten_receive, send)
