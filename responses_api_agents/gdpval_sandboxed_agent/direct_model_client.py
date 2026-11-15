# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use Gym's rollout-specific OpenAI endpoint with the pinned OpenHands fork.

The pinned client's message serialization, token metadata and completion logs
remain unchanged. Model timing is written separately from runtime metrics.
"""

import asyncio
import json as json_module
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from aiohttp import ClientTimeout

from nemo_gym.client import raise_for_status, request


logger = logging.getLogger(__name__)


@dataclass
class BufferedModelResponse:
    """Successful response consumable after its event-loop-local session closes."""

    body: bytes
    cookies: SimpleCookie
    ok: bool = True

    async def read(self) -> bytes:
        return self.body


# Errors that mean the request was lost or cut, not rejected: safe to re-issue once.
_RETRYABLE_ERRORS = (
    asyncio.TimeoutError,
    aiohttp.ServerTimeoutError,
    aiohttp.ClientConnectorError,
    aiohttp.ClientOSError,
    aiohttp.ClientPayloadError,
    aiohttp.ServerDisconnectedError,
)


class DirectModelTransport:
    def __init__(
        self,
        base_url: str,
        *,
        read_timeout_s: float = 1800,
        timing_path: Path | None = None,
        failed_request_dir: Path | None = None,
        read_retries: int = 0,
    ) -> None:
        if not math.isfinite(read_timeout_s) or read_timeout_s <= 0:
            raise ValueError("Model read timeout must be positive and finite")
        if read_retries < 0:
            raise ValueError("Model read retries must be non-negative")
        self.read_timeout_s = read_timeout_s
        # A bounded retry can recover a lost stateless completion request.
        # Each attempt is logged separately in model-requests.jsonl.
        self.read_retries = read_retries
        self.timing_path = timing_path
        self.failed_request_dir = failed_request_dir
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("A complete HTTP(S) policy endpoint is required")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("Policy endpoint must not contain credentials, query, or fragment")
        # Keep the entire rollout/token-capture prefix supplied by the Gym agent.
        self.completions_url = base_url.rstrip("/") + "/chat/completions"

    async def post(
        self, *, server_name: str, url_path: str, json: dict[str, Any], cookies: Mapping[str, Any] | None
    ) -> BufferedModelResponse:
        if url_path != "/v1/chat/completions":
            raise ValueError(f"Unsupported OpenHands model route: {url_path}")
        # The pinned LiteLLM setup adds this Bedrock-only option even for an
        # OpenAI-compatible endpoint. Gym rejects it, including when it is null.
        payload = dict(json)
        if payload.pop("aws_region_name", None) is not None:
            raise ValueError("AWS provider options are not supported by the direct policy endpoint")
        # OpenHands labels tool results with a legacy `name`; OpenAI associates
        # them with assistant calls via tool_call_id instead. Keep that ID and
        # all content/token fields, and do not mutate the completion log inputs.
        payload["messages"] = [
            {k: v for k, v in message.items() if k != "name"} if message.get("role") == "tool" else dict(message)
            for message in payload["messages"]
        ]
        # Gym replaces its shared client when the controller changes event loops.
        # This adapter owns retries, so each shared-client call makes one attempt.
        last_error = None
        for attempt in range(self.read_retries + 1):
            if attempt:
                await asyncio.sleep(2.0)
            try:
                return await self._post_once(payload, cookies, attempt)
            except _RETRYABLE_ERRORS as error:
                last_error = error
                if attempt >= self.read_retries:
                    raise
        raise last_error  # pragma: no cover

    async def _post_once(
        self, payload: dict[str, Any], cookies: Mapping[str, Any] | None, attempt: int
    ) -> BufferedModelResponse:
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        timing = {
            "request_id": request_id,
            "started_unix_s": time.time(),
            "read_timeout_s": self.read_timeout_s,
            "attempt": attempt,
        }
        request_path = None
        # Opt-in: retain the exact outgoing body for failed calls. Write BEFORE
        # dispatch so controller termination still leaves a usable fixture.
        # Headers, cookies, endpoint URLs and credentials are never included.
        if self.failed_request_dir is not None:
            try:
                self.failed_request_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                request_path = self.failed_request_dir / (request_id + ".json")
                with os.fdopen(os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                    json_module.dump(payload, stream)
            except OSError:
                timing["request_fixture_error"] = True
        try:
            response = await request(
                "POST",
                self.completions_url,
                json=payload,
                cookies=cookies,
                headers={"Authorization": "Bearer EMPTY", "X-GDPVal-Request-ID": request_id},
                timeout=ClientTimeout(total=None, connect=60, sock_read=self.read_timeout_s),
                _max_num_tries=1,
            )
            async with response:
                timing["headers_elapsed_s"] = time.monotonic() - started
                timing["http_status"] = response.status
                await raise_for_status(response)
                body = await response.read()
                # Log only identifiers/counts, never prompts, URLs or credentials.
                try:
                    decoded = json_module.loads(body)
                    timing["response_id"] = decoded.get("id")
                    usage = decoded.get("usage") or {}
                    timing["usage"] = {
                        k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage
                    }
                except (ValueError, AttributeError, TypeError):
                    pass
                if request_path is not None:
                    try:
                        request_path.unlink(missing_ok=True)
                    except OSError:
                        timing["request_fixture_cleanup_error"] = True
                return BufferedModelResponse(body, response.cookies)
        except BaseException as error:
            timing["error_type"] = type(error).__name__
            raise
        finally:
            timing["elapsed_s"] = time.monotonic() - started
            if self.timing_path is not None:
                try:
                    with self.timing_path.open("a") as stream:
                        stream.write(json_module.dumps(timing) + "\n")
                except OSError:
                    # Diagnostic I/O must not replace the model result/error.
                    pass


@asynccontextmanager
async def use_direct_model_client(
    *,
    read_timeout_s: float = 1800,
    timing_path: Path | None = None,
    failed_request_dir: Path | None = None,
    read_retries: int = 0,
) -> AsyncIterator[None]:
    from openhands.agenthub import nemo_gym_client

    original_client = nemo_gym_client.NemoGymClient
    runtime_metrics = os.environ.get("NEMO_GYM_METRICS_FPATH")
    model_metrics_path = Path(runtime_metrics).with_name("model-metrics.json") if runtime_metrics else None
    # Policy and condenser clients share the total. The condenser can run in a worker thread.
    model_call_time = 0.0
    metrics_lock = threading.Lock()

    class DirectModelClient(original_client):
        def __init__(self, llm):
            self.llm = llm
            self.model_server_cookies = None
            self.ng_server_client = DirectModelTransport(
                llm.config.base_url,
                read_timeout_s=read_timeout_s,
                timing_path=timing_path,
                failed_request_dir=failed_request_dir,
                read_retries=read_retries,
            )

        @staticmethod
        def _update_model_call_time(start_time: float) -> None:
            # The pinned client and runtime both rewrite metrics.json. Reading it between
            # truncate and write can fail a successful model call, so never read that file.
            nonlocal model_call_time
            with metrics_lock:
                model_call_time += max(0.0, time.time() - start_time)
                if model_metrics_path is None:
                    return
                temporary = model_metrics_path.with_name(f".{model_metrics_path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    temporary.write_text(json_module.dumps({"total_model_call_time": model_call_time}) + "\n")
                    os.replace(temporary, model_metrics_path)
                except OSError:
                    # Timing diagnostics must not turn a successful completion into a failed rollout.
                    logger.warning("Could not save OpenHands model timing", exc_info=True)
                finally:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass

    # The pinned CodeAct constructor imports this symbol when creating its client.
    nemo_gym_client.NemoGymClient = DirectModelClient
    try:
        yield
    finally:
        nemo_gym_client.NemoGymClient = original_client
