# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrying typed JSON Responses API calls used by LongFact."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import TypeVar

from aiohttp import ClientResponseError
from pydantic import BaseModel, ValidationError

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
)
from nemo_gym.verification import JudgeError, call_judge
from resources_servers.gymnasium import extract_text


LOG = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)
INTERNAL_MODEL_TRANSPORT_MAX_ATTEMPTS = 1


class EvaluatorResponseError(JudgeError, RuntimeError):
    """Raised after an evaluator request exhausts its parse/request retries."""


class EvaluatorParseError(EvaluatorResponseError):
    """Raised when repeated evaluator responses cannot be parsed as JSON."""


class EvaluatorIdentityError(EvaluatorResponseError):
    """Raised when a response came from a different model than the pinned one."""


class _SingleAttemptJudgeClient:
    def __init__(self, client) -> None:
        self._client = client

    async def post(self, **kwargs):
        return await self._client.post(**kwargs, _max_num_tries=INTERNAL_MODEL_TRANSPORT_MAX_ATTEMPTS)


class JsonEvaluator:
    def __init__(
        self,
        *,
        server_client,
        model_server: ModelServerRef,
        create_params: NeMoGymResponseCreateParamsNonStreaming,
        max_concurrency: int,
        max_attempts: int,
        timeout_seconds: float,
        expected_model_identity: str | None = None,
    ) -> None:
        self._server_client = _SingleAttemptJudgeClient(server_client)
        self._model_server = model_server
        self._create_params = create_params
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._expected_model_identity = (expected_model_identity or "").strip()

    @property
    def response_model_validation(self) -> str:
        expected = self._expected_model_identity
        if expected and expected.upper() not in {"MISSING", "NONE", "NULL", "UNKNOWN"}:
            return "exact"
        return "not_enforced"

    async def generate_json(
        self,
        schema: type[SchemaT],
        prompt: str,
        *,
        constrain_json_schema: bool = False,
    ) -> SchemaT:
        params = self._create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        # Hosted overlays clear inherited sampling with null; omit those fields
        # from the request so reasoning judges use their provider defaults.
        for field in ("temperature", "top_p"):
            if getattr(params, field) is None:
                params.model_fields_set.discard(field)
        if constrain_json_schema:
            # Local vLLM supports guided JSON decoding; hosted judges can leave it disabled.
            params.text = {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                }
            }
            params.model_fields_set.add("text")
        else:
            # Upstream asks for JSON in the prompt and parses ordinary text.
            # Clear any inherited constraint when guided decoding is disabled.
            params.text = None
            params.model_fields_set.discard("text")

        last_error: Exception | None = None
        response_text: str | None = None
        async with self._semaphore:
            attempt = 0
            while response_text is None and attempt < self._max_attempts:
                attempt += 1
                try:
                    await asyncio.sleep(random.uniform(0, 0.2))
                    parsed = await call_judge(
                        self._server_client,
                        server_name=self._model_server.name,
                        url_path="/v1/responses",
                        json=params,
                        response_model=NeMoGymResponse,
                        timeout=self._timeout_seconds,
                        max_attempts=1,
                    )
                    self._validate_model_identity(parsed)
                    if parsed.status != "completed" or parsed.incomplete_details is not None:
                        reason = parsed.incomplete_details.reason if parsed.incomplete_details is not None else None
                        raise EvaluatorResponseError(
                            f"Evaluator response {parsed.id} has status={parsed.status}, incomplete_reason={reason}"
                        )
                    response_text = extract_text(parsed)
                except EvaluatorResponseError:
                    # Unexpected model identity and incomplete judge output cannot supply a usable grade.
                    raise
                except Exception as exc:
                    if self._is_bad_request(exc):
                        raise EvaluatorResponseError("Evaluator request rejected with HTTP 400") from exc
                    last_error = exc
                    LOG.warning(
                        "Free-form JSON evaluator request failed (%d/%d, schema=%s): %s",
                        attempt,
                        self._max_attempts,
                        schema.__name__,
                        f"{type(exc).__name__}: {exc}",
                    )
                    if attempt < self._max_attempts:
                        base_wait = 2 ** (attempt - 1)
                        await asyncio.sleep(base_wait + random.uniform(0, base_wait * 0.5))

        if response_text is None:
            raise EvaluatorResponseError(
                f"Evaluator failed for {schema.__name__} after {self._max_attempts} attempts"
            ) from last_error

        # The released workers parse one completed sampler response exactly
        # once. A malformed answer is an evaluation error, not a reason to ask
        # the model for a different semantic answer.
        return self._parse_json(schema, response_text, salvage_embedded_json=True)

    @staticmethod
    def _is_bad_request(exc: Exception) -> bool:
        """Match upstream's immediate ``openai.BadRequestError`` handling."""

        if isinstance(exc, NgFailureEvent) and isinstance(exc.__cause__, Exception):
            exc = exc.__cause__
        return isinstance(exc, ClientResponseError) and exc.status == 400

    def _validate_model_identity(self, response: NeMoGymResponse) -> None:
        expected = self._expected_model_identity
        if self.response_model_validation != "exact":
            return
        actual = str(response.model or "").strip()
        if actual != expected:
            raise EvaluatorIdentityError(
                f"Response model identity mismatch: expected {expected!r}, received {actual!r}"
            )

    @classmethod
    def _parse_json(
        cls,
        schema: type[SchemaT],
        text: str,
        *,
        sanitize_json: bool = True,
        salvage_embedded_json: bool = False,
    ) -> SchemaT:
        """Parse ordinary model text, then validate it against the internal schema."""

        payload = cls._load_json(
            text,
            sanitize_json=sanitize_json,
            salvage_embedded_json=salvage_embedded_json,
        )
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Evaluator JSON did not match {schema.__name__}: {exc}") from exc

    @classmethod
    def _load_json(
        cls,
        text: str,
        *,
        sanitize_json: bool = True,
        salvage_embedded_json: bool = False,
    ):
        candidate = text.strip()
        if sanitize_json:
            candidate = cls._sanitize_json_string(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            # The first JSON opener owns recovery; backticks inside a claim cannot replace its outer payload.
            payload_start = re.search(r'```|[\[{"]', candidate)
            if payload_start is not None:
                fenced = payload_start.group() == "```"
                if fenced or salvage_embedded_json:
                    embedded = candidate[payload_start.end() if fenced else payload_start.start() :]
                    if fenced and embedded.startswith("json"):
                        embedded = embedded[4:]
                    embedded = embedded.lstrip()
                    try:
                        payload, end = json.JSONDecoder().raw_decode(embedded)
                    except json.JSONDecodeError as embedded_exc:
                        raise ValueError("Evaluator did not return parseable JSON") from embedded_exc
                    suffix = embedded[end:].lstrip()
                    if fenced and suffix and not suffix.startswith("```"):
                        raise ValueError("Evaluator did not return parseable JSON") from exc
                    return payload
            raise ValueError("Evaluator did not return parseable JSON") from exc

    @staticmethod
    def _sanitize_json_string(text: str) -> str:
        """Repair LaTeX control words while preserving escaped backslashes and JSON character escapes."""

        ambiguous_latex_commands = {
            "begin",
            "beta",
            "bf",
            "bar",
            "boxed",
            "binom",
            "bullet",
            "forall",
            "frac",
            "nu",
            "nabla",
            "ne",
            "neg",
            "neq",
            "not",
            "notin",
            "rightarrow",
            "rho",
            "right",
            "rangle",
            "rm",
            "theta",
            "tau",
            "tan",
            "text",
            "textbf",
            "times",
            "tfrac",
            "tilde",
            "to",
            "top",
        }

        def fix_escape(match: re.Match[str]) -> str:
            token = match.group(1)
            if token in ambiguous_latex_commands:
                return "\\\\" + token
            if token.startswith("u"):
                return match.group(0) if re.fullmatch(r"u[0-9a-fA-F]{4}", token) else "\\\\" + token
            if token[0] in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                return match.group(0)
            return "\\\\" + token

        return re.sub(r"\\(u[0-9a-fA-F]{4}|[A-Za-z]+|.)", fix_escape, text)
