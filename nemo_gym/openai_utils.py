# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The OpenAI-compatible client that Gym servers use to call model endpoints, with retries.

The Responses API and Chat Completions types it sends and receives live in `nemo_gym.core.responses_api_types`.
"""

import json
import random
import uuid
from asyncio import sleep
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
)

from aiohttp import ClientTimeout
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_validator,
)

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.client.http_client import (
    MAX_NUM_TRIES,
    ClientResponse,
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    raise_for_status,
    request,
)
from nemo_gym.core.episode import current_rollout_id
from nemo_gym.server_utils import ClientResponseError


# See https://platform.openai.com/docs/guides/error-codes/api-errors
# 404 can be a transient model-routing failure; retries remain bounded.
# 408 is a request timeout.
# 500 is internal server error, which may sporadically occur
# 502 is Bad gateway (when the endpoint is overloaded)
# 504 is Gateway timeout (when the endpoint config has too low of a gateway timeout setting for the model to finish generating)
RATE_LIMIT_ERROR_CODES = [429, 502, 503, 504, 520]
RETRY_ERROR_CODES = RATE_LIMIT_ERROR_CODES + [404, 408, 500]
# 429 is usually a transient rate limit. Match only these spent-key codes/types;
# generic "quota exceeded" wording is used by per-minute limits that recover.
PERMANENT_QUOTA_CODES = ("budget_exceeded", "insufficient_quota")
PERMANENT_AUTH_CODES = ("invalid_api_key", "invalid_api_token", "authentication_error")
# A provider that says it is busy is waited out on a time budget, not a count of tries.
WAIT_OUT_ERROR_CODES = (429, 503)
RETRY_BACKOFF_BASE_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 60.0
# 2**6 = 64 s already passes the cap, so larger exponents only risk a float overflow on very long retry runs.
RETRY_BACKOFF_MAX_EXPONENT = 6


def _retry_after_seconds(response: ClientResponse) -> Optional[float]:
    """Seconds a Retry-After header asks for, as a number or an HTTP date; None without a usable one."""
    value = (getattr(response, "headers", None) or {}).get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except ValueError:
        return None
    return (retry_at.replace(tzinfo=retry_at.tzinfo or timezone.utc) - datetime.now(timezone.utc)).total_seconds()


def _retry_wait_seconds(tries: int, response: ClientResponse) -> float:
    """Full-jitter exponential backoff, or the provider's Retry-After, both capped."""
    retry_after = _retry_after_seconds(response)
    # A zero wait would spend none of max_rate_limit_wait_seconds, so it falls back to backoff.
    if retry_after is not None and retry_after > 0:
        return min(retry_after, RETRY_BACKOFF_MAX_SECONDS)
    exponent = min(tries - 1, RETRY_BACKOFF_MAX_EXPONENT)
    return random.uniform(0, min(RETRY_BACKOFF_MAX_SECONDS, RETRY_BACKOFF_BASE_SECONDS * 2**exponent))


def _decode_error_text(content: bytes | str) -> str:
    return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content


def _parsed_error_codes(content: bytes | str) -> list[str]:
    """error.code / error.type tokens from an OpenAI-style error body."""
    text = _decode_error_text(content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    if "error" not in payload and isinstance(payload.get("detail"), dict):
        # A NeMo Gym model server returns a propagated provider error body as FastAPI's `detail`.
        payload = payload["detail"]
    error = payload.get("error")
    if isinstance(error, str):
        return [error]
    if not isinstance(error, dict):
        return []
    return [str(error[key]) for key in ("code", "type") if error.get(key) is not None]


def _error_body_is_permanent_quota(content: bytes | str) -> bool:
    """True when a 429 body is a spent key, not a transient rate limit."""
    codes = {token.lower() for token in _parsed_error_codes(content)}
    return any(code in codes for code in PERMANENT_QUOTA_CODES)


def _error_body_is_permanent_auth(content: bytes | str) -> bool:
    """True when a 401/403 body is a revoked or invalid key, not a one-off denial."""
    codes = {token.lower() for token in _parsed_error_codes(content)}
    if any(code in codes for code in PERMANENT_AUTH_CODES):
        return True
    lowered = _decode_error_text(content).lower()
    return "invalid api key" in lowered or "incorrect api key" in lowered


class PermanentEndpointError(ClientResponseError):
    """The upstream key is spent or unauthorized; further calls on this client skip the wire."""


REQUEST_ID_HEADER = "X-Request-Id"


def _rollout_request_id(headers: Mapping[str, str]) -> Optional[str]:
    """A request id that names the rollout, for calls made inside a rollout context.

    vLLM takes its request id from ``X-Request-Id`` and records it as ``gen_ai.request.id`` on
    the request's span and in its request log, so an engine span or log line can be matched to a
    rollout without walking the trace. The suffix keeps every call distinct, including across
    server workers that cannot share a counter. A caller's own header wins.
    """
    if any(key.lower() == REQUEST_ID_HEADER.lower() for key in headers):
        return None
    rollout_id = current_rollout_id()
    if not rollout_id:
        return None
    return f"{rollout_id}-{uuid.uuid4().hex[:8]}"


class NeMoGymAsyncOpenAI(BaseModel):  # pragma: no cover
    """This is just a stub class that wraps around aiohttp"""

    base_url: str
    api_key: str

    internal: bool = Field(
        default=False,
        description="Set this to true for internal NeMo Gym servers, which may retry indefinitely.",
    )

    max_connection_retries: Optional[int] = Field(
        default=None,
        description=(
            "How many connection-error retries per request; None retries forever. "
            "Allows callers that can resolve a moved endpoint to avoid stalling forever."
        ),
    )

    max_http_attempts: int = Field(default=MAX_NUM_TRIES, ge=1)

    max_rate_limit_wait_seconds: float = Field(
        default=3600.0,
        ge=0,
        description=(
            "Total time to wait out 429 and 503 answers before failing the request, instead of "
            "max_http_attempts tries; the request fails once the next wait would pass it. 0 keeps 429 and 503 "
            "on the max_http_attempts count. Internal NeMo Gym servers keep retrying them without limit."
        ),
    )

    default_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra headers to include in every request.",
    )

    max_num_tries: Optional[Literal[1]] = Field(
        default=None,
        description=(
            "Set to 1 to disable both inner transport and HTTP-status retry "
            "layers when a caller owns the complete retry schedule; this overrides "
            "max_http_attempts and the transport's generic-error retry limit. None "
            "preserves NeMo Gym's default behavior."
        ),
    )

    request_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Total time limit for each HTTP attempt (aiohttp ClientTimeout.total), "
            "not a deadline for the whole retry schedule. None means no limit."
        ),
    )
    connect_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Time limit for each HTTP attempt to obtain a connection, including "
            "waiting for a free aiohttp connection-pool slot (ClientTimeout.connect). "
            "Requires request_timeout_seconds."
        ),
    )

    @model_validator(mode="after")
    def validate_timeout_pair(self) -> "NeMoGymAsyncOpenAI":
        if self.connect_timeout_seconds is not None and self.request_timeout_seconds is None:
            raise ValueError("connect_timeout_seconds requires request_timeout_seconds")
        return self

    # Spent-key/auth trip: (status, body, url). Later calls raise a fresh exception.
    _permanent_trip: Optional[tuple[int, bytes, str]] = PrivateAttr(default=None)

    def _raise_permanent_error(self) -> None:
        assert self._permanent_trip is not None
        status, body, url = self._permanent_trip
        snippet = body.decode("utf-8", errors="replace")[:200]
        error = PermanentEndpointError(
            request_info=None,
            history=(),
            status=status,
            message=(
                f"Skipping further requests to {self.base_url}: HTTP {status} is permanent "
                f"(url={url} error_msg={snippet})"
            ),
            headers=None,
        )
        error.response_content = body
        raise error

    def _trip_permanent_error(self, status: int, content: bytes | str, url: Any) -> None:
        body = content if isinstance(content, bytes) else content.encode("utf-8", errors="replace")
        self._permanent_trip = (status, body, str(url))
        print(
            f"[model_retry_stop url={url} status={status} error_msg={body.decode('utf-8', errors='replace')[:200]}]",
            flush=True,
        )
        self._raise_permanent_error()

    async def _request(self, **request_kwargs: Dict) -> ClientResponse:
        if self._permanent_trip is not None:
            self._raise_permanent_error()
        request_headers = request_kwargs.pop("headers", {})
        connection_retries = self.max_connection_retries
        if connection_retries is None and request_kwargs.get("_resend_after_disconnect") is False:
            connection_retries = MAX_NUM_TRIES
        headers = self.default_headers | request_headers | {"Authorization": f"Bearer {self.api_key}"}
        request_id = _rollout_request_id(headers)
        if request_id is not None:
            headers["X-Request-Id"] = request_id
        request_kwargs = request_kwargs | {
            "headers": headers,
            "_internal": self.internal,
            "_max_num_tries": self.max_num_tries,
            "_max_connection_retries": connection_retries,
        }
        if self.request_timeout_seconds is not None:
            request_kwargs["timeout"] = ClientTimeout(
                total=self.request_timeout_seconds,
                connect=self.connect_timeout_seconds,
            )
        return await self._request_with_retry(**request_kwargs)

    async def _request_with_retry(self, **request_kwargs: Dict) -> ClientResponse:
        if self._permanent_trip is not None:
            self._raise_permanent_error()
        resend = request_kwargs.get("_resend_after_disconnect", True)
        # A status response may follow generation too. Keep permanent-error detection,
        # but do not resend it; safe connection setup retries remain in the transport.
        max_num_tries = (self.max_num_tries or self.max_http_attempts) if resend else 1
        tries = 0
        waited = 0.0
        while tries < max_num_tries:
            if self._permanent_trip is not None:
                self._raise_permanent_error()
            tries += 1
            response = await request(**request_kwargs)

            if response.status in (401, 403):
                content = await response.content.read()
                if _error_body_is_permanent_auth(content):
                    self._trip_permanent_error(response.status, content, request_kwargs.get("url"))
                return response

            if response.status not in RETRY_ERROR_CODES:
                return response

            content = await response.content.read()
            if response.status == 429 and _error_body_is_permanent_quota(content):
                self._trip_permanent_error(response.status, content, request_kwargs.get("url"))

            wait = _retry_wait_seconds(tries, response)
            # Internal NeMo Gym servers extend max tries for retryable errors unless
            # the caller owns the complete retry schedule. A busy provider is waited
            # out until max_rate_limit_wait_seconds is spent: a positive budget ends the
            # request even when max_http_attempts allows more tries, and 0 leaves 429
            # and 503 on the max_http_attempts count.
            if resend and self.max_num_tries is None and response.status in RATE_LIMIT_ERROR_CODES and self.internal:
                max_num_tries += 1
            elif (
                resend
                and self.max_num_tries is None
                and response.status in WAIT_OUT_ERROR_CODES
                and self.max_rate_limit_wait_seconds > 0
            ):
                if waited + wait > self.max_rate_limit_wait_seconds:
                    await raise_for_status(response, content)
                max_num_tries = max(max_num_tries, tries + 1)

            # Preserve the final error body for raise_for_status and avoid sleeping
            # after the last attempt. Reading intermediate bodies releases sockets.
            if tries >= max_num_tries:
                await raise_for_status(response, content)

            kind = "rate_limit" if response.status in RATE_LIMIT_ERROR_CODES else "http_error"
            print(
                f"[model_retry url={request_kwargs.get('url')} status={response.status} kind={kind} try={tries} max_tries={max_num_tries} wait_s={wait:.1f} error_msg={content.decode('utf-8', errors='replace')[:200]}]",
                flush=True,
            )
            waited += wait
            await sleep(wait)

        await raise_for_status(response)

    async def _raise_for_status(self, response: ClientResponse, request_kwargs: Dict[str, Any]) -> None:
        if not response.ok and is_global_aiohttp_client_request_debug_enabled():
            print(f"Request kwargs: {json.dumps(request_kwargs)}")

        await raise_for_status(response)

    async def create_models(self):
        request_kwargs = dict(url=f"{self.base_url}/models")
        response = await self._request(method="GET", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_chat_completion(self, *, _resend_after_disconnect: bool = True, **kwargs) -> Dict[str, Any]:
        request_kwargs = dict(
            url=f"{self.base_url}/chat/completions",
            json=kwargs,
        )
        if not _resend_after_disconnect:
            request_kwargs["_resend_after_disconnect"] = False
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_completion(self, **kwargs):
        request_kwargs = dict(
            url=f"{self.base_url}/completions",
            json=kwargs,
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_response(self, **kwargs):
        request_kwargs = dict(
            url=f"{self.base_url}/responses",
            json=kwargs,
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_tokenize(self, **kwargs):
        base_url = self.base_url.removesuffix("/v1")
        request_kwargs = dict(
            url=f"{base_url}/tokenize",
            json=kwargs,
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_embeddings(self, **kwargs):
        """POST ``{base_url}/embeddings`` with ``kwargs`` as the JSON body and return the parsed response dict."""
        request_kwargs = dict(
            url=f"{self.base_url}/embeddings",
            json=kwargs,
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
