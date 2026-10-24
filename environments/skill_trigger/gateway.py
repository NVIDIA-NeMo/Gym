# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Trial-scoped inference transport with host-only evidence and shared limits.

The adapter owns lifecycle and session attribution. This server only forwards
to its configured provider, never a destination supplied by the evaluated agent.
It records request/response bytes outside Harbor's agent-visible log mount.
"""

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from aiohttp import ClientError, ClientTimeout, web

from environments.skill_trigger.budget import BudgetReached, TrialBudget
from environments.skill_trigger.scoring import EvidenceError


Outcome = Literal["output_limit", "provider_error", "trace_error", "timeout", "routing_horizon", "cancelled"]


def response_outcome(raw: bytes, *, streaming: bool) -> Outcome | None:
    """Require a complete provider response; a truncated stream is not a miss."""
    try:
        text = raw.decode("utf-8")
        if streaming:
            events = []
            for block in text.replace("\r\n", "\n").split("\n\n"):
                data = "\n".join(line[5:].lstrip() for line in block.splitlines() if line.startswith("data:"))
                if data and data != "[DONE]":
                    events.append(json.loads(data))
        else:
            events = [json.loads(text)]
    except (ValueError, UnicodeError):
        return "trace_error"
    if not events or any(not isinstance(event, dict) for event in events):
        return "trace_error"
    terminal = False
    limited = False
    for event in events:
        response = event.get("response", event)
        if not isinstance(response, dict):
            return "trace_error"
        if (
            event.get("type") in {"error", "response.failed"}
            or event.get("error")
            or response.get("status") == "failed"
        ):
            return "provider_error"
        if event.get("type") == "response.incomplete" or response.get("status") == "incomplete":
            details = response.get("incomplete_details") or {}
            if not isinstance(details, dict) or details.get("reason") != "max_output_tokens":
                return "provider_error"
            limited = True
        terminal |= event.get("type") == "response.completed" or response.get("status") == "completed"
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            return "trace_error"
        for choice in choices:
            if not isinstance(choice, dict):
                return "trace_error"
            limited |= choice.get("finish_reason") == "length"
            terminal |= choice.get("finish_reason") is not None
    return "output_limit" if limited else None if terminal else "trace_error"


class InferenceGateway:
    """Serve one trial; the provider key never enters the agent environment.

    ``resolve_session`` is supplied by the host adapter and must establish the
    session from captured harness evidence before returning a registered ID.
    Agent-provided metadata alone is insufficient proof. A root-only harness
    can supply a callback returning its already established root session ID.
    """

    def __init__(
        self,
        *,
        budget: TrialBudget,
        upstream_url: str,
        api_key: str,
        evidence_dir: Path,
        resolve_session: Callable[[dict[str, object]], Awaitable[str]],
    ):
        self.budget = budget
        self._upstream_url = upstream_url.rstrip("/")
        self._api_key = api_key
        self._resolve_session = resolve_session
        self.token = secrets.token_urlsafe(32)
        evidence_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.evidence_dir = evidence_dir
        self.outcome: Outcome | None = None
        self.stopped = asyncio.Event()
        self._received = 0
        self._runner: web.AppRunner | None = None
        self.port: int | None = None

    async def start(self, *, bind_host: str, port: int = 0) -> int:
        """Listen on an explicitly chosen host; return the allocated port."""
        if self._runner is not None:
            raise EvidenceError("Gateway already started")
        app = web.Application(client_max_size=30_000_000)
        app.router.add_get("/health", self._health)
        app.router.add_post("/v1/{endpoint:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None, shutdown_timeout=2)
        await self._runner.setup()
        site = web.TCPSite(self._runner, bind_host, port)
        try:
            await site.start()
            addresses = self._runner.addresses
            if len(addresses) != 1:
                raise EvidenceError("Expected one gateway listening address")
            self.port = addresses[0][1]
            return self.port
        except BaseException:
            await self._runner.cleanup()
            self._runner = None
            raise

    async def _health(self, incoming: web.Request) -> web.Response:
        """Check container connectivity before release without consuming a model request."""
        if not secrets.compare_digest(incoming.headers.get("Authorization", ""), "Bearer " + self.token):
            raise web.HTTPForbidden()
        return web.Response(text="ready")

    async def close(self) -> None:
        """Stop admission, close connections, and wait for evidence writes."""
        self.budget.close()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        await self.budget.wait_settled()

    def _stop(self, outcome: Outcome) -> None:
        # Incomplete evidence must never be hidden by a competing budget event.
        if self.outcome is None or outcome in {"provider_error", "trace_error", "cancelled"}:
            self.outcome = outcome
        self.stopped.set()
        self.budget.close()

    @staticmethod
    def _write(path: Path, value: object) -> None:
        with path.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")

    async def _handle(self, incoming: web.Request) -> web.StreamResponse:
        if not secrets.compare_digest(incoming.headers.get("Authorization", ""), "Bearer " + self.token):
            raise web.HTTPForbidden()
        endpoint = "/" + incoming.match_info["endpoint"]
        if endpoint not in {"/responses", "/chat/completions"}:
            raise web.HTTPNotFound()
        # No await between number allocation and mkdir: concurrent requests
        # get distinct, non-overwriting attempt directories, including blocked ones.
        self._received += 1
        folder = self.evidence_dir / f"{self._received:03d}"
        folder.mkdir(mode=0o700)
        downstream: web.StreamResponse | None = None
        metadata: dict[str, object] = {"endpoint": endpoint, "forwarded": False}
        try:
            raw = await incoming.read()
            (folder / "request-original.json").write_bytes(raw)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise EvidenceError("Inference payload must be an object")
            if self.stopped.is_set():
                return web.json_response({"error": "trial_stopped"}, status=409)
            remaining = 120 - self.budget.elapsed()
            if remaining <= 0:
                raise BudgetReached("timeout")
            attribution_deadline = asyncio.timeout(remaining)
            try:
                async with attribution_deadline:
                    session_id = await self._resolve_session(payload)
            except TimeoutError:
                if attribution_deadline.expired():
                    raise BudgetReached("timeout") from None
                raise
            async with self.budget.forward(session_id=session_id, endpoint=endpoint, payload=payload) as sent:
                metadata.update(session_id=session_id, forwarded=True, sent_at=self.budget.elapsed())
                self._write(folder / "request-sent.json", sent)
                # Import lazily so offline scoring needs no Gym server stack.
                from nemo_gym.server_utils import request

                headers = {"Authorization": "Bearer " + self._api_key, "Accept-Encoding": "identity"}
                if incoming.headers.get("OpenAI-Beta"):
                    headers["OpenAI-Beta"] = incoming.headers["OpenAI-Beta"]
                response = await request(
                    "POST",
                    self._upstream_url + endpoint,
                    json=sent,
                    headers=headers,
                    _max_num_tries=1,
                    _max_connection_retries=0,
                    _resend_after_disconnect=False,
                    allow_redirects=False,
                    timeout=ClientTimeout(total=None),
                    auto_decompress=False,
                )
                async with response:
                    metadata["http_status"] = response.status
                    metadata["provider_metadata"] = {
                        key: value
                        for key, value in response.headers.items()
                        if key.lower() in {"retry-after", "date", "x-request-id", "openai-processing-ms"}
                        or key.lower().startswith(("x-ratelimit-", "ratelimit-"))
                    }
                    downstream = web.StreamResponse(
                        status=response.status,
                        headers={"Content-Type": response.headers.get("Content-Type", "application/json")},
                    )
                    await downstream.prepare(incoming)
                    chunks = []
                    with (folder / "response.body").open("xb") as record:
                        async for chunk in response.content.iter_any():
                            record.write(chunk)
                            record.flush()
                            chunks.append(chunk)
                            await downstream.write(chunk)
                    metadata["ended_at"] = self.budget.elapsed()
                    outcome = (
                        "provider_error"
                        if response.status != 200
                        else response_outcome(b"".join(chunks), streaming=sent.get("stream") is True)
                    )
                    if outcome:
                        self._stop(outcome)
                    await downstream.write_eof()
                    return downstream
        except BudgetReached as error:
            self._stop(error.outcome)
            metadata["outcome"] = error.outcome
            if downstream is None:
                return web.json_response({"error": error.outcome}, status=409)
            if incoming.transport:
                incoming.transport.close()
            return downstream
        except asyncio.CancelledError:
            self._stop("cancelled")
            metadata["outcome"] = "cancelled"
            raise
        except (ClientError, ConnectionError, TimeoutError) as error:
            self._stop("provider_error")
            metadata.update(outcome="provider_error", error_type=type(error).__name__)
            if downstream is not None:
                if incoming.transport:
                    incoming.transport.close()
                return downstream
            return web.json_response({"error": "provider_error"}, status=502)
        except Exception as error:
            # This is the trial transport boundary: preserve malformed input,
            # evidence-write failures, and unexpected collection errors as
            # unscored outcomes without leaking credentials in exception text.
            self._stop("trace_error")
            metadata.update(outcome="trace_error", error_type=type(error).__name__)
            if downstream is not None:
                if incoming.transport:
                    incoming.transport.close()
                return downstream
            return web.json_response({"error": "trace_error"}, status=500)
        finally:
            metadata["trial_outcome"] = self.outcome
            self._write(folder / "metadata.json", metadata)
