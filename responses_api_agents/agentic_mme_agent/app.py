# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native multi-turn atomic Agentic-MME harness with append-only model history."""

import asyncio
import itertools
import json
from dataclasses import dataclass
from typing import Any

import cv2
from aiohttp import ClientError, ClientResponseError
from fastapi import HTTPException, Request, Response
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import BaseRunRequest
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.agentic_mme.app import TOOL_TRACE_METRICS_KEY, AgenticMMEVerifyResponse
from resources_servers.agentic_mme.image_tools import IMAGE_TOOLS, ImageWorkspace, function_schema
from responses_api_agents.agentic_mme_agent.retrieval import RETRIEVAL_TOOLS, Retrieval, RetrievalConfig


# Refusals that measure the policy: its own tool images and history outgrew the context or the
# request limits. The episode ends unanswered and scores 0. vLLM's "At most N image(s) may be
# provided in one prompt" is not one of them: it is a serving limit (limit_mm_per_prompt), so it
# raises like any other configuration error; max_images_per_request keeps episodes within it.
_CONTEXT_LIMIT_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "request_too_large",
)


def is_context_limit(status: int, body: str) -> bool:
    """A 400/413 that names a context or request-size limit."""
    text = body.lower()
    return status in (400, 413) and any(marker in text for marker in _CONTEXT_LIMIT_MARKERS)


class PolicyRequestError(RuntimeError):
    """A policy call failed for a reason outside the model (auth, rate limit, server error, timeout,
    lost connection, unreadable reply). Raised out of /run so rollout collection records the task in
    the failures sidecar and retries it, instead of scoring the model 0."""


class AgenticMMEAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # None removes the limit: the episode then ends when the model answers without a tool call,
    # or when the model's context or request limit refuses the call, recorded as policy_error. Under an agent
    # session a failed call fails the response with its cause, and the Environment Server grades or retries.
    max_rounds: int | None = Field(default=15, ge=1, le=100)
    max_tool_calls: int | None = Field(default=15, ge=0, le=100)
    # Per worker: with num_workers > 1, each worker runs this many rollouts at once.
    max_concurrent_rollouts: int = Field(default=8, ge=1)
    max_image_pixels: int = Field(default=16_000_000, ge=1)
    max_total_image_pixels: int = Field(default=64_000_000, ge=1)
    # Images per model request (input images plus tool images; every image stays in every later
    # request). None (the default) sets no limit. Set it to the serving engine's per-prompt image
    # limit (vLLM limit_mm_per_prompt.image): an image tool call past it returns a tool error, so the
    # model answers with the images it has instead of the request being refused.
    max_images_per_request: int | None = Field(default=None, ge=1)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class AgenticMMERunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_replay: list[dict[str, Any]] = Field(default_factory=list)


@dataclass(frozen=True)
class PolicyCallError:
    """A policy-model call that failed and ended the episode.

    ``status`` is the HTTP status when the model server refused the call for a reason other than a context or
    request-size limit; that refusal is ``model_context_limit`` instead, so an outcome policy can tell the two apart.
    A call that failed before a status (timeout, lost connection) is a transport error, and a reply that is not a
    Response has an unknown cause.
    """

    message: str
    failure_kind: str
    status: int | None = None
    # False only for a context or request-size refusal, which ends the episode unanswered (the model's own
    # result). Every other failure is infrastructure: /run raises it into the failures sidecar.
    infrastructure: bool = True

    def failed_response(self, response: NeMoGymResponse) -> NeMoGymResponse:
        """Mark the episode's response failed with this cause, as Gym's shared agent outcome.

        The Environment Server then applies its outcome policy. ``configs/agentic_mme.yaml`` sets it to grade a
        context-limit refusal as the model's own result and to fail any other rejected request (400, 413, 422)
        ungraded, as ``/run`` does; a transport failure or a retryable status such as 429 or 503 is retried.
        """
        outcome = AgentOutcome(failure_kind=self.failure_kind, status_code=self.status)
        return NeMoGymResponse.model_validate(
            response.model_dump()
            | {
                "status": "failed",
                "incomplete_details": None,
                "error": {"code": "server_error", "message": self.message},
                "metadata": {
                    **(response.metadata or {}),
                    AGENT_OUTCOME_METADATA_KEY: outcome.model_dump_json(exclude_none=True),
                },
            }
        )


@dataclass
class AgenticMMESessionState(AgentSessionState):
    """One agent session runs one episode.

    An identical retry of the activation joins the running episode, a different request is refused, and close
    cancels it. The image tools run in this agent, so the session holds no tool grant and no sandbox.
    """

    activation_key: tuple[NeMoGymResponseCreateParamsNonStreaming, str] | None = None
    execution: asyncio.Task[tuple[NeMoGymResponse, dict[str, str]]] | None = None


class AgenticMMEAgent(SimpleResponsesAPIAgent):
    config: AgenticMMEAgentConfig
    _rollouts: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._rollouts = asyncio.Semaphore(self.config.max_concurrent_rollouts)
        cv2.setNumThreads(1)

    def tool_schemas(self) -> list[dict[str, Any]]:
        tools = dict(IMAGE_TOOLS)
        if self.config.retrieval.mode != "disabled":
            # Text search always comes with reverse image search (Google Lens), in live and replay mode alike,
            # so a replay offers exactly the tools of the run it replays.
            tools.update(RETRIEVAL_TOOLS)
        return [function_schema(name, model, description) for name, (model, description) in tools.items()]

    async def _episode(
        self,
        params: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
        replay: list[dict[str, Any]] | None = None,
        model_url_path: str = "/v1/responses",
    ) -> tuple[NeMoGymResponse, Any, PolicyCallError | None]:
        async with self._rollouts:
            workspace = ImageWorkspace(
                max_pixels=self.config.max_image_pixels,
                max_total_pixels=self.config.max_total_image_pixels,
                max_images=self.config.max_images_per_request,
            )
            return await self._episode_with_workspace(params, cookies, replay, workspace, model_url_path)

    async def _episode_with_workspace(
        self,
        params: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
        replay: list[dict[str, Any]] | None,
        workspace: ImageWorkspace,
        model_url_path: str = "/v1/responses",
    ) -> tuple[NeMoGymResponse, Any, PolicyCallError | None]:
        params = params.model_copy(deep=True)
        if isinstance(params.input, str):
            params.input = [NeMoGymEasyInputMessage(role="user", content=params.input)]
        for item in params.input:
            data = item if isinstance(item, dict) else item.model_dump()
            content = data.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "input_image":
                        await asyncio.to_thread(workspace.load, part["image_url"])
        if not workspace.images:
            raise ValueError("Agentic-MME requires at least one input image")
        tools = self.tool_schemas()
        # Inputs contain task prompts only; verifier metadata never enters model context.
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            params.model_dump() | {"tools": tools, "parallel_tool_calls": False}
        )
        retrieval = Retrieval(self.config.retrieval, replay)
        outputs = []
        trace = []
        usage = None
        attempts = 0
        finished = False
        policy_error = None
        generated = None
        # No forced final-answer call: every model call sees the same tools and tool_choice, so each
        # call's prompt extends the previous one (NeMo RL token capture links a rollout's calls by
        # that prefix; a tools-stripped answer-only call broke the chain). max_rounds tool rounds
        # plus one more call to answer; the episode ends unanswered when the rounds run out or the
        # model calls a tool beyond max_tool_calls.
        max_rounds, max_tool_calls = self.config.max_rounds, self.config.max_tool_calls
        rounds = itertools.count() if max_rounds is None else range(max_rounds + 1)
        for turn in rounds:
            new_params = params.model_copy(update={"input": [*params.input, *outputs], "tool_choice": "auto"})
            try:
                response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    # Under token capture this is /ng-rollout/<id>/training-token-capture/v1/responses, so the
                    # policy model server records every call of this rollout; otherwise plain /v1/responses.
                    url_path=model_url_path,
                    json=new_params,
                    cookies=cookies,
                )
                if response.ok:
                    reply = NeMoGymResponse.model_validate(await get_response_json(response))
                else:
                    error_body = (await response.read()).decode(errors="replace")
            except (TimeoutError, ClientError, ValueError) as exc:
                # The call failed without an HTTP status (connection lost, timeout after the client's
                # retries, a body that is not JSON or not a Response): infrastructure, not the model.
                message = f"{type(exc).__name__}: {str(exc)[:1000]}"
                if isinstance(exc, ClientResponseError):
                    policy_error = PolicyCallError(message, MODEL_REQUEST_ERROR, exc.status)
                elif isinstance(exc, ValueError):
                    policy_error = PolicyCallError(message, "unknown")
                else:
                    policy_error = PolicyCallError(message, MODEL_TRANSPORT_ERROR)
                break
            if not response.ok:
                # A context or request-size refusal (many tool images accumulated) ends the episode
                # unanswered; any other status (a serving limit such as vLLM's per-prompt image limit, an
                # unsupported parameter, 401, 429, 5xx) is infrastructure.
                context_limit = is_context_limit(response.status, error_body)
                policy_error = PolicyCallError(
                    f"HTTP {response.status}: {error_body[:1000]}",
                    MODEL_CONTEXT_LIMIT if context_limit else MODEL_REQUEST_ERROR,
                    None if context_limit else response.status,
                    infrastructure=not context_limit,
                )
                break
            cookies = response.cookies
            generated = reply
            usage = accumulate_response_usage(usage, generated.usage)
            # Preserve typed output items, including token IDs/log-probs for training.
            outputs.extend(generated.output)
            calls = [item for item in generated.output if item.type == "function_call"]
            if generated.incomplete_details:
                break
            if not calls and any(item.type == "message" and item.role == "assistant" for item in generated.output):
                finished = True
                break
            image_observations = []
            budget_exceeded = False
            for call in calls:
                arguments = None
                try:
                    if max_tool_calls is not None and attempts >= max_tool_calls:
                        budget_exceeded = True
                        raise ValueError("tool budget exhausted")
                    attempts += 1  # Invalid calls consume budget too.
                    arguments = json.loads(call.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    if call.name in IMAGE_TOOLS:
                        result = await asyncio.to_thread(workspace.apply, call.name, arguments)
                    elif call.name in RETRIEVAL_TOOLS:
                        result = await retrieval.call(call.name, arguments, workspace)
                    else:
                        raise ValueError("unknown tool")
                except (TimeoutError, ClientError):
                    result = {"ok": False, "error": "retrieval provider unavailable or timed out"}
                except (ValueError, TypeError, KeyError, OSError, cv2.error) as exc:
                    result = {"ok": False, "error": str(exc)}
                trace.append(
                    {
                        "index": len(trace) + 1,
                        "turn": turn + 1,
                        "tool_name": call.name,
                        "arguments": arguments,
                        # The new image is already in the next user message; keeping its data URL here
                        # would store every image twice in the rollout row.
                        "output": {key: value for key, value in result.items() if key != "image_url"},
                    }
                )
                image_url = result.get("image_url")
                # The trace keeps the provider's answer ("raw"), the leak guard's count ("contaminated") and
                # the cache outcome ("cache": a cold and a warm run must see the same prompt); the model sees
                # none of them. A failed call shows only its error, so a page the guard filtered
                # reads like any other failed fetch.
                if result.get("ok"):
                    hidden = ("image_url", "raw", "contaminated", "cache")
                    observation = {key: value for key, value in result.items() if key not in hidden}
                else:
                    observation = {"ok": False, "error": result.get("error")}
                # Set type explicitly: requests are serialized with exclude_unset, and the model
                # server's converter cannot route an item that has neither type nor role.
                outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output", call_id=call.call_id, output=json.dumps(observation)
                    )
                )
                if image_url:
                    image_observations.append(
                        NeMoGymEasyInputMessage(
                            type="message",
                            role="user",
                            content=[
                                {"type": "input_text", "text": f"Image {result['new_image_index']}"},
                                {"type": "input_image", "image_url": image_url, "detail": "auto"},
                            ],
                        )
                    )
            # Complete all function-call outputs before inserting user images.
            # Some backends emit multiple calls even when parallel_tool_calls=False.
            outputs.extend(image_observations)
            if budget_exceeded:
                break
        if generated is None:
            generated = NeMoGymResponse.model_validate(
                {
                    "id": "policy_error",
                    "created_at": 0,
                    "model": "",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [],
                }
            )
        # The trace goes in metrics, which the model never sees, so that verify receives it on /run and under an
        # agent session alike: a session's Environment Server sends verify only the task data and this response.
        metrics = {**(generated.metrics or {}), TOOL_TRACE_METRICS_KEY: trace}
        result = generated.model_copy(update={"output": outputs, "usage": usage, "metrics": metrics})
        if not finished and result.incomplete_details is None:
            # An episode that ends without an answer (tool or round budget used up, or a context-limit
            # refusal) is incomplete, but not a length cut: NeMo RL's overlong filtering and Gym's
            # truncated_answer label read incomplete_details.reason == "max_output_tokens", and would
            # drop the loss for running out of budget. The verifier scores it 0 as missing_answer, since
            # no assistant message follows the last tool call.
            result = NeMoGymResponse.model_validate(result.model_dump() | {"status": "incomplete"})
        return result, cookies, policy_error

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgenticMMESessionState:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("Agentic-MME runs its image tools in the agent and does not use a sandbox")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError("Agentic-MME does not use granted tools")
        if self.config.retrieval.mode == "replay":
            # The replay list is a field of the task row. An agent session receives only the model request.
            raise AgentSeedRejectedError(
                "retrieval.mode=replay reads retrieval_replay from the task row, which agent sessions do not "
                "receive; use retrieval.mode=disabled or live under an environment server"
            )
        return AgenticMMESessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, AgenticMMESessionState):
            raise TypeError("Expected Agentic-MME session state")
        execution = state.execution
        if execution is not None:
            if not execution.done():
                execution.cancel()
                done, _ = await asyncio.wait({execution}, timeout=self.config.session_close_timeout_seconds)
                if not done:
                    raise TimeoutError(
                        "Agentic-MME episode did not stop within the session close timeout; close must retry"
                    )
            await asyncio.gather(execution, return_exceptions=True)
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)

    async def _session_episode(
        self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict[str, str], model_url_path: str
    ) -> tuple[NeMoGymResponse, dict[str, str]]:
        result, model_cookies, policy_error = await self._episode(body, cookies, model_url_path=model_url_path)
        if policy_error is not None:
            # Report the cause, so the Environment Server's outcome policy decides between grading and retry.
            result = policy_error.failed_response(result)
        return result, {name: getattr(value, "value", value) for name, value in (model_cookies or {}).items()}

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Run one episode: the image-tool loop, with every model call through the rollout's model route.

        Under an agent session, the session runs one activation. An identical retry joins it, and a different
        request, or one routed under another rollout, is refused with 409.
        """
        route = self.url_path_for_request("/v1/responses", request)
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            result, model_cookies, _ = await self._episode(body, request.cookies, model_url_path=route)
            for name, value in (model_cookies or {}).items():
                response.set_cookie(name, value)
            return result
        state = self._require_agent_session(session_id)
        if not isinstance(state, AgenticMMESessionState):
            raise TypeError("Expected Agentic-MME session state")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Agent session does not match the rollout route")
        key = (body.model_copy(deep=True), route)
        if state.activation_key is not None and state.activation_key != key:
            raise HTTPException(409, "Session already has another activation")
        if state.execution is None:
            state.activation_key = key
            state.execution = asyncio.create_task(self._session_episode(body, dict(request.cookies), route))
        try:
            result, model_cookies = await asyncio.shield(state.execution)
        except asyncio.CancelledError:
            request_task = asyncio.current_task()
            if state.execution.cancelled() and request_task is not None and not request_task.cancelling():
                raise HTTPException(409, "Agent session is closing") from None
            raise
        for name, value in model_cookies.items():
            response.set_cookie(name, value)
        return result

    async def run(self, request: Request, body: AgenticMMERunRequest) -> AgenticMMEVerifyResponse:
        result, _, policy_error = await self._episode(
            body.responses_create_params,
            request.cookies,
            body.retrieval_replay,
            model_url_path=self.url_path_for_run("/v1/responses", body),
        )
        if policy_error is not None and policy_error.infrastructure:
            raise PolicyRequestError(policy_error.message)
        verification = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json={
                "responses_create_params": body.responses_create_params.model_dump(),
                "response": result.model_dump(),
                "verifier_metadata": body.verifier_metadata,
            },
            cookies=request.cookies,
        )
        await raise_for_status(verification)
        # Verify returns the tool trace and its counts.
        verified = AgenticMMEVerifyResponse.model_validate(await get_response_json(verification))
        return verified.model_copy(
            update={
                "policy_error": policy_error.message if policy_error else None,
                **({"failure_reason": "policy_error"} if policy_error else {}),
            }
        )


if __name__ == "__main__":
    AgenticMMEAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = AgenticMMEAgent.run_webserver()  # noqa: F401
