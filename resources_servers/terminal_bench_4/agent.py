# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an agent harness in a sandbox owned by a resources server.

Use this lifecycle when task setup and grading need to control the same environment,
while a separate agent worker runs the harness and makes model calls. For example,
a benchmark can provision task-specific images and sidecars once, then hand the main
sandbox to different harnesses without duplicating provisioning or verifier logic.

The resources server implements the seed/start/verify/cancel contract and owns the
execution deadline and sandbox teardown. The agent supplies setup and execution
callbacks; this module attaches a borrowed sandbox handle, propagates session
cookies, requests grading or cancellation, and releases the handle without deleting
the environment needed by the verifier.
"""

import asyncio
from pathlib import Path
from time import time
from uuid import uuid4

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config
from nemo_gym.server_utils import SESSION_ID_KEY, get_global_config_dict, get_response_json, raise_for_status
from resources_servers.terminal_bench_4.handoff import AgentTermination, SandboxedSeedResponse


def empty_response(body, model):
    return NeMoGymResponse(
        id="resp_" + uuid4().hex,
        created_at=int(time()),
        model=model,
        object="response",
        output=[],
        tool_choice=body.tool_choice,
        tools=body.tools,
        parallel_tool_calls=body.parallel_tool_calls,
    )


async def run_borrowed(agent, request, body, *, setup, execute):
    body = body.model_copy(deep=True)
    cookies = dict(request.cookies)

    async def resources(path, payload):
        nonlocal cookies
        response = await agent.server_client.post(
            server_name=agent.config.resources_server.name,
            url_path=path,
            json=payload,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies |= response.cookies
        return await get_response_json(response)

    payload = body.model_dump()
    payload["rollout_id"] = (
        agent.rollout_id_from_run(body) or body.capture_rollout_id or payload.get("rollout_id") or uuid4().hex
    )
    payload["client_session_id"] = request.session[SESSION_ID_KEY]
    # A resources HTTP retry reuses this nonce; another /run invocation cannot
    # start a second harness against the same active rollout.
    payload["execution_id"] = uuid4().hex
    seed = SandboxedSeedResponse.model_validate(await resources("/seed_session", payload))
    sandbox = None
    started = False
    response = empty_response(body.responses_create_params, agent.config.model_server.name)
    termination = AgentTermination(reason="infrastructure_error", detail="Agent setup did not complete")
    extra = {}
    try:
        providers = agent.config.sandbox_providers
        if seed.sandbox.provider not in providers:
            raise ValueError(f"Unconfigured sandbox provider alias: {seed.sandbox.provider}")
        config = resolve_provider_config(providers[seed.sandbox.provider], get_global_config_dict())
        provider = create_provider(config)
        try:
            sandbox = await AsyncSandbox.connect(
                seed.sandbox.model_dump(exclude={"provider"}),
                provider=provider,
                owns_sandbox=False,
            )
        except BaseException:
            await provider.aclose()
            raise
        await asyncio.wait_for(setup(sandbox, seed), timeout=seed.setup_timeout_sec)
        budget = await resources("/start_session", {"session_id": seed.session_id})
        started = True
        body.responses_create_params.input = [NeMoGymEasyInputMessage(role="user", content=seed.instruction)]
        try:
            response, termination, extra = await execute(sandbox, seed, budget["agent_timeout_sec"])
        except TimeoutError:
            termination = AgentTermination(reason="timeout")
        except asyncio.CancelledError:
            termination = AgentTermination(reason="cancelled")
            raise
        except Exception as exc:
            termination = AgentTermination(reason="infrastructure_error", detail=f"{type(exc).__name__}: {exc}")
    finally:
        directory = Path("results") / agent.config.name / seed.session_id
        if directory.exists() and str(directory) not in termination.artifacts:
            termination.artifacts.append(str(directory))
        try:
            if started:
                # Verification is shielded from a disconnected collector: resources
                # persist the result and always retain ownership of teardown.
                verify = asyncio.create_task(
                    resources(
                        "/verify",
                        {
                            **body.model_dump(),
                            "session_id": seed.session_id,
                            "response": response.model_dump(mode="json"),
                            "termination": termination.model_dump(),
                        },
                    )
                )
                result = await asyncio.shield(verify)
            else:
                await asyncio.shield(resources("/cancel_session", {"session_id": seed.session_id}))
        finally:
            if sandbox is not None:
                await sandbox.release()
    return result | extra


def artifact_directory(agent_name, session_id):
    path = Path("results") / agent_name / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path
