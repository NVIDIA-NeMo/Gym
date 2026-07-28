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
"""Thin agent server that brokers rollouts to a user-hosted remote agent service.

The remote service implements ONE endpoint: ``POST {agent_base_url}/v1/responses``.
It receives the row's ``responses_create_params`` (never ``verifier_metadata`` — the
answer key stays inside Gym), runs its own agent loop with its own model and tools,
and returns a single finished Responses API trajectory. This server owns the Gym
side of the rollout: it seeds the session, holds the session cookies, verifies on
the resources server, and returns the verify response from ``/run``.

Failures never raise out of ``/run``: every failure (remote endpoint down, timeout,
malformed reply, seed/verify errors) becomes a reward-0 verify response carrying the
``_ng_failure_class`` sentinel, which rollout collection routes to the failures
sidecar and retries on resume.
"""

import asyncio
from typing import Any, Dict, Literal, Optional, Tuple
from urllib.parse import urlparse

import orjson
from aiohttp import ClientOSError, ClientTimeout, ServerDisconnectedError
from fastapi import Body, Request
from pydantic import ConfigDict, PrivateAttr, field_validator
from pydantic import ValidationError as PydanticValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.global_config import SKILLS_REF_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY
from nemo_gym.server_utils import get_global_aiohttp_client, get_response_json, get_server_url, raise_for_status


REMOTE_AGENT_FAILURE_CLASS = "remote_agent_error"

# Header names of the session-forwarding contract (tools_mode="forward"): the remote
# service echoes the cookie on every resources-server tool call it makes. The URL is
# re-sent per request (rather than configured remote-side) because Gym assigns servers
# random ports on every `gym env start` — a statically configured address goes stale on
# every restart; the cookie is minted per rollout and has no static equivalent at all.
RESOURCES_URL_HEADER = "X-NeMo-Gym-Resources-Server-Url"
SESSION_COOKIE_HEADER = "X-NeMo-Gym-Session-Cookie"

_REMOTE_MAX_TRIES = 3
_REMOTE_RETRY_SLEEP_SECS = 0.5
_FAILURE_PRINT_HEAD = 5
_FAILURE_PRINT_INTERVAL = 100
_AGGREGATE_PROXY_TIMEOUT_SECS = 600.0

# Result/routing keys this server itself produces. Input rows may carry stale copies
# (e.g. a rollouts or failures JSONL re-fed as a dataset); they must never collide with
# the fresh values or leak through the verify echo into the dispatcher's routing.
_RESERVED_RESULT_KEYS = ("reward", "response", "error", NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY)


def normalize_remote_url(url: str) -> str:
    """Validate the remote service URL and strip any trailing slash."""
    normalized = url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"agent_base_url must be an absolute http:// or https:// URL, got {url!r}")
    # "/v1/responses" is string-appended; anything after "?" or "#" would swallow it (bare
    # delimiters parse as an empty query/fragment, so check the string itself).
    if "?" in normalized or "#" in normalized or parsed.params:
        raise ValueError(
            f"agent_base_url must not carry a query string or fragment, got {url!r}. "
            "Pass auth material via your service's own configuration instead."
        )
    # Credentials would be stamped into logged configs and error messages; never echo the URL.
    if parsed.username or parsed.password:
        raise ValueError(
            "agent_base_url must not embed credentials (user:pass@host). "
            "Pass auth material via your service's own configuration instead."
        )
    return normalized


def cookie_header_value(cookies: Any) -> Optional[str]:
    """Serialize seed-session cookies (SimpleCookie morsels or a plain dict) into a Cookie header."""
    if not cookies:
        return None
    pairs = []
    for key, value in cookies.items():
        pairs.append(f"{key}={getattr(value, 'value', value)}")
    return "; ".join(pairs) if pairs else None


class RemoteAgentConfig(BaseResponsesAPIAgentConfig):
    agent_base_url: str
    resources_server: ResourcesServerRef
    concurrency: int = 32
    remote_responses_timeout_secs: float = 1800.0
    # Bound on the whole /run body (seed + remote call + verify), applied after the
    # semaphore is acquired so queue wait does not count against it. The collector's
    # named-agent hop carries no timeout of its own; this is the only wallclock bound.
    run_timeout_secs: float = 2100.0
    # Who serves the tools a dataset declares:
    #   "refuse"  — nobody can: reject tool-declaring tasks up front (terminal failure row)
    #               instead of letting verify() score silent zeros against untouched state.
    #   "forward" — Gym does: send the resources-server URL and session cookie as headers on
    #               every remote request; the service echoes the cookie on each tool call.
    #   "remote"  — the service does: it implements the declared tools itself; nothing is
    #               forwarded and the guard stands down.
    tools_mode: Literal["refuse", "forward", "remote"] = "refuse"
    # The resources-server URL advertised to the remote service with tools_mode="forward".
    # The default (resolved from the global config) is the BIND address — typically
    # 127.0.0.1, unreachable from another machine. Set this to the externally reachable URL
    # when the remote service runs off-host (bind vs. advertise can genuinely differ: NAT,
    # tunnels, load balancers). This only changes the header string; making the address
    # actually route to the resources server is the operator's job.
    advertised_resources_url: Optional[str] = None

    @field_validator("agent_base_url")
    @classmethod
    def _normalize_agent_base_url(cls, value: str) -> str:
        return normalize_remote_url(value)

    @field_validator("advertised_resources_url")
    @classmethod
    def _normalize_advertised_resources_url(cls, value: Optional[str]) -> Optional[str]:
        return normalize_remote_url(value) if value else value


class RemoteAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class RemoteAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class RemoteAgent(SimpleResponsesAPIAgent):
    config: RemoteAgentConfig
    sem: Optional[asyncio.Semaphore] = None
    _num_failures: int = PrivateAttr(default=0)
    _warn_counts: Dict[str, int] = PrivateAttr(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    async def responses(self, body=Body()) -> NeMoGymResponse:
        raise NotImplementedError(
            "RemoteAgent brokers a remote service; drive it through /run. The remote service's "
            "own /v1/responses is called by run(), not exposed here."
        )

    async def run(self, request: Request, body: RemoteAgentRunRequest = Body()) -> RemoteAgentVerifyResponse:
        record = self._sanitized_record(body)
        async with self.sem:
            try:
                return await asyncio.wait_for(
                    self._run_once(request, body, record), timeout=self.config.run_timeout_secs
                )
            except asyncio.TimeoutError:
                return self._failure_response(
                    record,
                    f"/run exceeded run_timeout_secs={self.config.run_timeout_secs}s "
                    "(seed + remote /v1/responses + verify)",
                )
            except Exception as e:  # noqa: BLE001 -- never 500; one task must not abort the whole collection
                return self._failure_response(record, f"unexpected error: {type(e).__name__}: {e}")

    def _sanitized_record(self, body: RemoteAgentRunRequest) -> Dict[str, Any]:
        record = body.model_dump()
        for key in _RESERVED_RESULT_KEYS:
            record.pop(key, None)
        return record

    async def _run_once(
        self, request: Request, body: RemoteAgentRunRequest, record: Dict[str, Any]
    ) -> RemoteAgentVerifyResponse:
        # body and record are two views of the same row: `record` (sanitized dict, computed
        # before run()'s try so failure rows can be built in ANY error state) feeds the Gym
        # hops; `body` (typed model) is kept solely because exclude_unset information — which
        # fields the dataset actually set — exists only on the model, and the remote wire
        # payload must not carry materialized None defaults.
        guard_error = self._tools_guard_error(record)
        if guard_error:
            return self._failure_response(record, guard_error, terminal=True)

        if record.get(SKILLS_REF_KEY_NAME):
            self._throttled_warn(
                "skills_ref",
                "WARNING: this run carries a skills_ref, but RemoteAgent cannot stage skills into a "
                "remote service; the skills config is ignored.",
            )

        # 1. Seed the session; the cookies key all per-session state on the resources server.
        cookies = request.cookies
        try:
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=record,
                cookies=cookies,
            )
            await raise_for_status(seed_response)
            cookies = seed_response.cookies
        except Exception as e:
            return self._failure_response(
                record, f"/seed_session on the resources server failed: {type(e).__name__}: {e}"
            )

        # 2. One POST to the remote service: create-params in, finished trajectory out.
        # exclude_unset keeps the wire payload to exactly what the dataset row carried.
        remote_params = body.responses_create_params.model_dump(exclude_unset=True)
        remote_result, remote_error, terminal = await self._post_remote_responses(remote_params, cookies)
        if remote_error is not None:
            return self._failure_response(record, remote_error, terminal=terminal)

        try:
            response = NeMoGymResponse.model_validate(remote_result)
        except PydanticValidationError as e:
            # A shape error will not fix itself on retry.
            return self._failure_response(
                record,
                f"remote service returned an invalid Responses API object: {str(e)[:500]}",
                terminal=True,
            )
        self._warn_on_response_quality(response)

        # 3. Verify on the SAME session; the verify response (reward included) is /run's result.
        try:
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=record | {"response": response.model_dump(mode="json")},
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            verify_json = await get_response_json(verify_response)
        except Exception as e:
            return self._failure_response(record, f"/verify on the resources server failed: {type(e).__name__}: {e}")

        return RemoteAgentVerifyResponse.model_validate(verify_json)

    def _tools_guard_error(self, record: Dict[str, Any]) -> Optional[str]:
        """Refuse tool-declaring tasks the remote service cannot serve, instead of scoring silent zeros.

        A dataset that declares tools expects them to be called during the rollout. Without
        forward_session the remote service has no session cookie, so any per-session state
        those tools mutate stays untouched and verify() scores 0 on every row.
        """
        declared_tools = (record.get("responses_create_params") or {}).get("tools")
        if declared_tools and self.config.tools_mode == "refuse":
            return (
                'the task declares tools but tools_mode="refuse" (the default). Set '
                'tools_mode="forward" so the remote service can call Gym-hosted tools with the '
                'session cookie, or tools_mode="remote" if the service implements the declared '
                "tools itself."
            )
        return None

    async def _post_remote_responses(
        self, remote_params: Dict[str, Any], cookies: Any
    ) -> Tuple[Optional[Dict], Optional[str], bool]:
        """POST create-params to the remote /v1/responses. Returns (result, error, terminal)."""
        remote_url = f"{self.config.agent_base_url}/v1/responses"
        client = get_global_aiohttp_client()
        data = orjson.dumps(remote_params)
        headers = {"Content-Type": "application/json"}
        if self.config.tools_mode == "forward":
            resources_url = self.config.advertised_resources_url or get_server_url(self.config.resources_server.name)
            advertised_host = urlparse(resources_url).hostname or ""
            remote_host = urlparse(self.config.agent_base_url).hostname or ""
            if advertised_host in ("127.0.0.1", "localhost") and remote_host not in ("127.0.0.1", "localhost"):
                self._throttled_warn(
                    "loopback_resources_url",
                    f"WARNING: forwarding resources-server URL {resources_url} (a loopback address) to the "
                    f"off-host remote service at {self.config.agent_base_url}. Its tool calls will not reach "
                    "Gym; set advertised_resources_url to an externally reachable URL.",
                )
            headers[RESOURCES_URL_HEADER] = resources_url
            session_cookie = cookie_header_value(cookies)
            if session_cookie:
                headers[SESSION_COOKIE_HEADER] = session_cookie
        timeout = ClientTimeout(total=self.config.remote_responses_timeout_secs)

        response = None
        last_connect_error: Optional[BaseException] = None
        for num_try in range(1, _REMOTE_MAX_TRIES + 1):
            try:
                # aiohttp follows a redirected POST as a body-less GET; fail with the 3xx instead.
                response = await client.request(
                    "POST", remote_url, data=data, headers=headers, timeout=timeout, allow_redirects=False
                )
                break
            except (ClientOSError, ServerDisconnectedError) as e:
                # Refused/reset (ClientOSError) and keepalive races (ServerDisconnectedError)
                # are transient connection noise; everything else fails fast.
                last_connect_error = e
                if num_try < _REMOTE_MAX_TRIES:
                    await asyncio.sleep(_REMOTE_RETRY_SLEEP_SECS)
            except asyncio.TimeoutError:
                return (
                    None,
                    f"remote /v1/responses timed out after {self.config.remote_responses_timeout_secs}s "
                    "(remote_responses_timeout_secs; raise it if rollouts legitimately run longer)",
                    False,
                )
            except Exception as e:
                return None, f"{type(e).__name__}: {e}", False
        if response is None:
            return (
                None,
                f"could not reach the remote service after {_REMOTE_MAX_TRIES} tries "
                f"({type(last_connect_error).__name__}: {last_connect_error}). "
                f"Is your service running at {self.config.agent_base_url}?",
                False,
            )

        # client.request() returns once headers arrive; the body read can still raise
        # (mid-body disconnect, deadline) and must honor the same never-raise contract.
        try:
            content = await response.read()
        except Exception as e:
            return None, f"reading the response body failed: {type(e).__name__}: {e}", False
        # response.ok is `status < 400`; reject 3xx explicitly (redirects are not followed).
        if not response.ok or response.status >= 300:
            location = response.headers.get("Location", "")
            return (
                None,
                f"HTTP {response.status}"
                + (f" (redirect to {location}; fix agent_base_url to point at the final address)" if location else "")
                + f": {content[:500].decode(errors='replace')}",
                False,
            )
        try:
            result = orjson.loads(content)
        except orjson.JSONDecodeError as e:
            return None, f"response is not valid JSON: {e}", False
        if not isinstance(result, dict):
            return None, f"expected a JSON object from /v1/responses, got {type(result).__name__}", False
        return result, None, False

    def _throttled_warn(self, key: str, message: str) -> None:
        """Per-key sampled warning: the first few occurrences, then every 100th. At production
        concurrency an unthrottled per-rollout print garbles the collector's progress bar."""
        n = self._warn_counts.get(key, 0) + 1
        self._warn_counts[key] = n
        if n <= _FAILURE_PRINT_HEAD or n % _FAILURE_PRINT_INTERVAL == 0:
            print(f"{message} (occurrence #{n})", flush=True)

    def _warn_on_response_quality(self, response: NeMoGymResponse) -> None:
        if response.usage is None:
            self._throttled_warn(
                "missing_usage",
                "WARNING: the remote response carries no usage; token metrics for this agent will be "
                "empty. Have your service report usage {input_tokens, output_tokens, total_tokens}.",
            )

    def _failure_response(
        self, record: Dict[str, Any], error: str, terminal: bool = False
    ) -> RemoteAgentVerifyResponse:
        self._num_failures += 1
        n = self._num_failures
        if n <= _FAILURE_PRINT_HEAD or n % _FAILURE_PRINT_INTERVAL == 0:
            print(f"[remote_agent] rollout failed (failure #{n}): {error}", flush=True)
        routing: Dict[str, Any] = {NG_FAILURE_CLASS_KEY: REMOTE_AGENT_FAILURE_CLASS, "error": error}
        if terminal:
            routing[NG_TERMINAL_KEY] = True
        # Dict-merge with later keys winning: `record` is sanitized of reserved keys, but merge
        # order still guarantees fresh reward/response/routing even if a caller passes a raw dump.
        return RemoteAgentVerifyResponse.model_validate(
            record | {"reward": 0.0, "response": self._empty_response().model_dump(mode="json")} | routing
        )

    def _empty_response(self) -> NeMoGymResponse:
        """Minimal valid response for the failure path, so /run can return 200 with reward 0
        (never 500) even when the remote service produced nothing."""
        return NeMoGymResponse(
            id="remote_agent_failure",
            created_at=0.0,
            model="remote_agent",
            object="response",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "id": "msg_0",
                    "content": [{"type": "output_text", "text": "", "annotations": []}],
                }
            ],
            parallel_tool_calls=False,
            tools=[],
            tool_choice="auto",
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server.

        Bounded: the ServerClient hop otherwise retries connection errors forever, and a dead
        resources server at end-of-run would hang the collector after all rollouts are on disk.
        """

        async def _proxy() -> AggregateMetrics:
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/aggregate_metrics",
                json=body,
            )
            await raise_for_status(response)
            return AggregateMetrics.model_validate(await get_response_json(response))

        return await asyncio.wait_for(_proxy(), timeout=_AGGREGATE_PROXY_TIMEOUT_SECS)


if __name__ == "__main__":
    RemoteAgent.run_webserver()
