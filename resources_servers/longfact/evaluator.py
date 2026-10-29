# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrying JSON and text Responses API calls used by LongFact."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import TypeVar

from aiohttp import ClientResponseError
from pydantic import BaseModel, ValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.gymnasium import extract_text


LOG = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)
INTERNAL_MODEL_TRANSPORT_MAX_ATTEMPTS = 1


class EvaluatorResponseError(RuntimeError):
    """Raised after an evaluator request exhausts its parse/request retries."""


class EvaluatorParseError(EvaluatorResponseError):
    """Raised when repeated evaluator responses cannot be parsed as JSON."""


class EvaluatorIdentityError(EvaluatorResponseError):
    """Raised when a response came from a different model than the pinned one."""


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
        self._server_client = server_client
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

    @property
    def response_reasoning_validation(self) -> str:
        return "exact"

    async def generate_json(
        self,
        schema: type[SchemaT],
        system: str | None,
        user: str,
        *,
        reasoning_effort: str | None = None,
        include_schema_instruction: bool = False,
        sanitize_json: bool = True,
        salvage_embedded_json: bool = False,
        constrain_json_schema: bool = False,
        validate_schema: bool = True,
    ) -> SchemaT | object:
        params = self._create_params.model_copy(deep=True)
        messages = []
        if system:
            system_text = system.rstrip()
            if include_schema_instruction:
                schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
                system_text = (
                    f"{system_text}\n\n"
                    "OUTPUT FORMAT:\n"
                    "Return only valid JSON matching the schema below. Do not use Markdown fences or add commentary.\n"
                    f"{schema_text}"
                )
            messages.append(NeMoGymEasyInputMessage(role="system", content=system_text))
        messages.append(NeMoGymEasyInputMessage(role="user", content=user))
        params.input = messages
        if constrain_json_schema:
            # The local vLLM Responses server supports OpenAI's structured
            # output shape.  LongFact opts into guided decoding because one
            # malformed answer among thousands of per-claim calls otherwise
            # invalidates an entire prompt row.  Keep this opt-in so existing
            # Keep evaluator sampling behavior stable for LongFact.
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
        if reasoning_effort is not None:
            params.reasoning = {"effort": reasoning_effort}

        last_error: Exception | None = None
        response_text: str | None = None
        async with self._semaphore:
            for attempt in range(1, self._max_attempts + 1):
                try:
                    await asyncio.sleep(random.uniform(0, 0.2))
                    response = await self._server_client.post(
                        server_name=self._model_server.name,
                        url_path="/v1/responses",
                        json=params,
                        timeout=self._timeout_seconds,
                        _max_num_tries=INTERNAL_MODEL_TRANSPORT_MAX_ATTEMPTS,
                    )
                    await raise_for_status(response)
                    parsed = NeMoGymResponse.model_validate(await get_response_json(response))
                    self._validate_model_identity(parsed)
                    self._validate_reasoning_identity(parsed, reasoning_effort)
                    response_text = extract_text(parsed)
                    break
                except EvaluatorIdentityError:
                    # A successful response from an unexpected model is a
                    # provenance failure, not a transient transport failure.
                    raise
                except Exception as exc:
                    if self._is_bad_request(exc):
                        raise EvaluatorResponseError(f"Evaluator request rejected with HTTP {exc.status}") from exc
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
        if not validate_schema:
            return self._load_json(
                response_text,
                sanitize_json=sanitize_json,
                salvage_embedded_json=salvage_embedded_json,
            )
        return self._parse_json(
            schema,
            response_text,
            sanitize_json=sanitize_json,
            salvage_embedded_json=salvage_embedded_json,
        )

    async def generate_text(
        self,
        user: str,
        *,
        reasoning_effort: str | None = None,
    ) -> str:
        """Make the unstructured Responses call used by upstream follow-up generation."""

        params = self._create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=user)]
        params.text = None
        params.model_fields_set.discard("text")
        if reasoning_effort is not None:
            params.reasoning = {"effort": reasoning_effort}

        last_error: Exception | None = None
        async with self._semaphore:
            for attempt in range(1, self._max_attempts + 1):
                try:
                    await asyncio.sleep(random.uniform(0, 0.2))
                    response = await self._server_client.post(
                        server_name=self._model_server.name,
                        url_path="/v1/responses",
                        json=params,
                        timeout=self._timeout_seconds,
                        _max_num_tries=INTERNAL_MODEL_TRANSPORT_MAX_ATTEMPTS,
                    )
                    await raise_for_status(response)
                    parsed = NeMoGymResponse.model_validate(await get_response_json(response))
                    self._validate_model_identity(parsed)
                    self._validate_reasoning_identity(parsed, reasoning_effort)
                    # Upstream's sampler can return an empty completed response.
                    # Callers decide whether that means "stop", an error, or an
                    # empty planner selection; do not resample it here.
                    return extract_text(parsed).strip()
                except EvaluatorIdentityError:
                    raise
                except Exception as exc:
                    if self._is_bad_request(exc):
                        raise EvaluatorResponseError(f"Evaluator request rejected with HTTP {exc.status}") from exc
                    last_error = exc
                    LOG.warning(
                        "Evaluator text call failed (%d/%d): %s",
                        attempt,
                        self._max_attempts,
                        f"{type(exc).__name__}: {exc}",
                    )
                    if attempt < self._max_attempts:
                        base_wait = 2 ** (attempt - 1)
                        await asyncio.sleep(base_wait + random.uniform(0, base_wait * 0.5))

        raise EvaluatorResponseError(
            f"Evaluator failed for unstructured text after {self._max_attempts} attempts"
        ) from last_error

    @staticmethod
    def _is_bad_request(exc: Exception) -> bool:
        """Match upstream's immediate ``openai.BadRequestError`` handling."""

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

    @staticmethod
    def _validate_reasoning_identity(
        response: NeMoGymResponse,
        expected_effort: str | None,
    ) -> None:
        if expected_effort is None:
            return
        if isinstance(response, dict):
            reasoning = response.get("reasoning")
        else:
            reasoning = response.reasoning
        if isinstance(reasoning, dict):
            actual_effort = reasoning.get("effort")
        else:
            actual_effort = reasoning.effort if reasoning is not None else None
        if actual_effort != expected_effort:
            raise EvaluatorIdentityError(
                f"Response reasoning identity mismatch: expected {expected_effort!r}, received {actual_effort!r}"
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
        # Match upstream extract_json_from_response exactly: strip the whole
        # response, prefer the first fenced payload when present, and otherwise
        # require the complete response to be JSON. Do not salvage an unfenced
        # object embedded in commentary.
        candidate = text.strip()
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1].split("```", 1)[0].strip()
        if sanitize_json:
            candidate = cls._sanitize_json_string(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            if salvage_embedded_json:
                decoder = json.JSONDecoder()
                for index, character in enumerate(candidate):
                    if character not in "[{":
                        continue
                    try:
                        payload, _ = decoder.raw_decode(candidate[index:])
                        return payload
                    except json.JSONDecodeError:
                        continue
            raise ValueError("Evaluator did not return parseable JSON") from exc

    @staticmethod
    def _sanitize_json_string(text: str) -> str:
        """Escape invalid backslashes commonly produced in LaTeX-rich claims."""

        def fix_escape(match: re.Match[str]) -> str:
            character = match.group(1)
            if character in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                return match.group(0)
            return "\\\\" + character

        return re.sub(r"\\(.)", fix_escape, text)
