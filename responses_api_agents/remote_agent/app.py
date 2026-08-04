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
"""Agent server that drives a user-hosted remote agent service through Gym's tool loop.

The remote service implements ONE endpoint, ``POST {agent_base_url}/v1/responses``, composing
with this server as OpenAI Responses-compliant agents: each call it receives the conversation
so far (the row's
``responses_create_params`` with the accumulated output and tool results appended to
``input``) and returns a Responses API object. To have Gym execute a tool from the
environment, it returns a ``function_call`` item WITHOUT a matching
``function_call_output``; tool calls it already answered itself (its own internal tools)
ride along as paired call+output items and are passed through untouched. Gym runs the
loop: it executes unpaired calls against the resources server and re-posts until the
service returns a final assistant message.

The resources server is never exposed to the service: tool execution, session cookies,
and ``verifier_metadata`` all stay inside Gym.

Failures never raise out of ``/run``: every failure (remote endpoint down, timeout,
malformed reply, seed/verify errors) becomes a reward-0 verify response carrying the
``_ng_failure_class`` sentinel, which rollout collection routes to the failures
sidecar and retries on resume.
"""

import asyncio
import json
from traceback import print_exc
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import orjson
from aiohttp import ClientOSError, ClientTimeout, ServerDisconnectedError
from fastapi import Body, Request, Response
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
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    accumulate_response_usage,
)
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY
from nemo_gym.server_utils import (
    get_global_aiohttp_client,
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    raise_for_status,
)


REMOTE_AGENT_FAILURE_CLASS = "remote_agent_error"

_REMOTE_MAX_TRIES = 3
_REMOTE_RETRY_SLEEP_SECS = 0.5
_FAILURE_PRINT_HEAD = 5
_FAILURE_PRINT_INTERVAL = 100
_AGGREGATE_PROXY_TIMEOUT_SECS = 600.0

# Result/routing keys this server itself produces. Input rows may carry stale copies
# (e.g. a rollouts or failures JSONL re-fed as a dataset); they must never collide with
# the fresh values or leak through the verify echo into the dispatcher's routing.
_RESERVED_RESULT_KEYS = ("reward", "response", "error", NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY)


class RemoteAgentError(RuntimeError):
    """A rollout-level failure from the remote hop; retryable on resume."""


class RemoteAgentTerminalError(RemoteAgentError):
    """A failure that will not fix itself on retry (e.g. an invalid response shape).

    run() reaches responses() over an HTTP self-post, so this class's NAME is the wire
    contract: the exception middleware serializes it into the 500 body and run() matches
    the name string to set the terminal routing flag.
    """


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


class RemoteAgentConfig(BaseResponsesAPIAgentConfig):
    agent_base_url: str
    resources_server: ResourcesServerRef
    concurrency: int = 32
    # Per-call bound on one POST to the remote service; a rollout makes one call per loop step.
    remote_responses_timeout_secs: float = 1800.0
    # Bound on the whole /run body (seed + the full agent/tool loop + verify), applied after
    # the semaphore is acquired so queue wait does not count against it. The collector's
    # named-agent hop carries no timeout of its own; this is the only whole-rollout bound.
    run_timeout_secs: float = 2100.0
    # Maximum loop steps (remote calls) per rollout; None leaves run_timeout_secs as the only bound.
    max_steps: Optional[int] = None

    @field_validator("agent_base_url")
    @classmethod
    def _normalize_agent_base_url(cls, value: str) -> str:
        return normalize_remote_url(value)


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

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        agent_server_cookies = None  # the service's own cookies, round-tripped so it can keep per-rollout state
        resources_server_cookies = request.cookies

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            agent_response, agent_server_cookies = await self._post_agent_responses(new_body, agent_server_cookies)

            output = agent_response.output
            new_outputs.extend(output)

            if not usage:
                usage = agent_response.usage
                agent_response.usage = None

            if usage and agent_response.usage:
                accumulate_response_usage(usage, agent_response.usage)

            if agent_response.incomplete_details:
                break

            # Execute only unpaired calls: a call the service already answered itself (matching
            # function_call_output in the same response) is its own internal-tool record and
            # passes through into the trajectory untouched.
            answered_call_ids = {o.call_id for o in output if o.type == "function_call_output"}
            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in output if o.type == "function_call" and o.call_id not in answered_call_ids
            ]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    # Malformed arguments go back to the service as a tool error output
                    # instead of crashing the rollout; repr(e) keeps the exception type
                    # even when str(e) is empty.
                    tool_response = NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=json.dumps({"error": f"Invalid tool call arguments: {e!r}"}),
                    )
                    new_outputs.append(tool_response)
                    continue

                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=parsed_arguments,
                    cookies=resources_server_cookies,
                )
                # No raise_for_status: a tool error (unknown tool, invalid call) is a valid
                # result the service should see and react to.
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

            if self.config.max_steps and step >= self.config.max_steps:
                break

        # Resources-server cookies propagate for downstream verification; the service's own
        # cookies are its private session and deliberately stay out.
        for k, v in resources_server_cookies.items():
            response.set_cookie(k, v)

        agent_response.output = new_outputs
        agent_response.usage = usage
        return agent_response

    async def _post_agent_responses(
        self, new_body: NeMoGymResponseCreateParamsNonStreaming, cookies: Optional[Dict[str, str]]
    ) -> Tuple[NeMoGymResponse, Dict[str, str]]:
        """One hardened POST to the remote service. Returns (validated response, its cookies)."""
        remote_url = f"{self.config.agent_base_url}/v1/responses"
        client = get_global_aiohttp_client()
        # exclude_unset keeps the wire payload to the fields the dataset row (and the loop)
        # actually set, never materialized None defaults.
        data = orjson.dumps(new_body.model_dump(exclude_unset=True))
        headers = {"Content-Type": "application/json"}
        timeout = ClientTimeout(total=self.config.remote_responses_timeout_secs)

        response = None
        last_connect_error: Optional[BaseException] = None
        for num_try in range(1, _REMOTE_MAX_TRIES + 1):
            try:
                # Never follow redirects: aiohttp re-issues 301/302/303 as a body-less GET and
                # re-sends 307/308 to an address the user never configured; fail with the 3xx.
                response = await client.request(
                    "POST",
                    remote_url,
                    data=data,
                    headers=headers,
                    cookies=cookies or {},
                    timeout=timeout,
                    allow_redirects=False,
                )
                break
            except (ClientOSError, ServerDisconnectedError) as e:
                # Refused/reset (ClientOSError) and keepalive races (ServerDisconnectedError)
                # are transient connection noise; everything else fails fast.
                last_connect_error = e
                if num_try < _REMOTE_MAX_TRIES:
                    await asyncio.sleep(_REMOTE_RETRY_SLEEP_SECS)
            except asyncio.TimeoutError:
                raise RemoteAgentError(
                    f"remote /v1/responses timed out after {self.config.remote_responses_timeout_secs}s "
                    "(remote_responses_timeout_secs; raise it if agent calls legitimately run longer)"
                ) from None
            except Exception as e:
                if is_global_aiohttp_client_request_debug_enabled():
                    print_exc()
                raise RemoteAgentError(f"{type(e).__name__}: {e}") from e
        if response is None:
            raise RemoteAgentError(
                f"could not reach the remote service after {_REMOTE_MAX_TRIES} tries "
                f"({type(last_connect_error).__name__}: {last_connect_error}). "
                f"Is your service running at {self.config.agent_base_url}?"
            )

        # client.request() returns once headers arrive; the body read can still fail
        # (mid-body disconnect, deadline).
        try:
            content = await response.read()
        except Exception as e:
            if is_global_aiohttp_client_request_debug_enabled():
                print_exc()
            raise RemoteAgentError(f"reading the response body failed: {type(e).__name__}: {e}") from e
        # response.ok is `status < 400`; reject 3xx explicitly (redirects are not followed).
        if not response.ok or response.status >= 300:
            if is_global_aiohttp_client_request_debug_enabled():
                print(
                    f"[remote_agent] full HTTP {response.status} body: {content.decode(errors='replace')}", flush=True
                )
            location = response.headers.get("Location", "")
            raise RemoteAgentError(
                f"HTTP {response.status}"
                + (f" (redirect to {location}; fix agent_base_url to point at the final address)" if location else "")
                + f": {content[:500].decode(errors='replace')}"
            )
        try:
            result = orjson.loads(content)
        except orjson.JSONDecodeError as e:
            raise RemoteAgentError(f"response is not valid JSON: {e}") from e
        if not isinstance(result, dict):
            raise RemoteAgentError(f"expected a JSON object from /v1/responses, got {type(result).__name__}")

        try:
            validated = NeMoGymResponse.model_validate(result)
        except PydanticValidationError as e:
            if is_global_aiohttp_client_request_debug_enabled():
                print(f"[remote_agent] full validation error: {e}", flush=True)
            # A shape error will not fix itself on retry.
            raise RemoteAgentTerminalError(
                f"remote service returned an invalid Responses API object: {str(e)[:500]}"
            ) from e

        merged_cookies = dict(cookies or {})
        merged_cookies.update({k: morsel.value for k, morsel in response.cookies.items()})
        return validated, merged_cookies

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
                    "(seed + agent/tool loop + verify)",
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
        # fields the dataset actually set — exists only on the model.
        if record.get(SKILLS_REF_KEY_NAME):
            self._throttled_warn(
                "skills_ref",
                "WARNING: this run carries a skills_ref, but RemoteAgent cannot stage skills into a "
                "remote service; the skills config is ignored.",
            )

        # Seed the session; the cookies key all per-session state on the resources server.
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

        try:
            loop_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(loop_response)
            response_json = await get_response_json(loop_response)
            cookies = loop_response.cookies
        except Exception as e:
            content = getattr(e, "response_content", b"")
            text = content.decode(errors="replace") if isinstance(content, (bytes, bytearray)) else str(content)
            # Terminal classification crosses the HTTP self-post boundary by exception NAME:
            # the middleware serialized the raised RemoteAgentTerminalError into the 500 body.
            terminal = "RemoteAgentTerminalError" in text
            detail = text or f"{type(e).__name__}: {e}"
            return self._failure_response(record, f"agent loop failed: {detail[:500]}", terminal=terminal)

        self._warn_on_response_quality(response_json)

        # Verify on the SAME session; the verify response (reward included) is /run's result.
        try:
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=record | {"response": response_json},
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            verify_json = await get_response_json(verify_response)
        except Exception as e:
            return self._failure_response(record, f"/verify on the resources server failed: {type(e).__name__}: {e}")

        return RemoteAgentVerifyResponse.model_validate(verify_json)

    def _throttled_warn(self, key: str, message: str) -> None:
        """Per-key sampled warning: the first few occurrences, then every 100th. At production
        concurrency an unthrottled per-rollout print garbles the collector's progress bar."""
        n = self._warn_counts.get(key, 0) + 1
        self._warn_counts[key] = n
        if n <= _FAILURE_PRINT_HEAD or n % _FAILURE_PRINT_INTERVAL == 0:
            print(f"{message} (occurrence #{n})", flush=True)

    def _warn_on_response_quality(self, response_json: Dict[str, Any]) -> None:
        if not response_json.get("usage"):
            self._throttled_warn(
                "missing_usage",
                "WARNING: the remote response carries no usage; token metrics for this agent will be "
                "empty. Have your service report the full usage object: {input_tokens, output_tokens, "
                "total_tokens, input_tokens_details: {cached_tokens}, output_tokens_details: "
                "{reasoning_tokens}}.",
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
