# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the unchanged BrowseComp loop against scoped Gym model and tool endpoints."""

import asyncio
import json
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit


# The source bundle is shipped by the adapter; task PYTHONPATH and user site packages are ignored (-I).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiohttp import ClientResponse
from aiohttp.client import _RequestOptions
from fastapi import Request, Response
from omegaconf import OmegaConf
from typing_extensions import Unpack

from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import ServerClient, get_global_aiohttp_client, get_response_json, raise_for_status, request
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import DirectHTTPToolAccess
from responses_api_agents.browsecomp_agent.app import (
    _SANDBOX_LOOP_CONNECTION_RETRIES,
    BrowsecompAgent,
    BrowsecompAgentConfig,
    BrowsecompAgentSessionState,
)


class ScopedModelClient(ServerClient):
    """Only the rollout-qualified Model Server is visible, never its upstream credentials."""

    model_name: str
    model_url: str

    async def post(self, server_name: str, url_path: str, **kwargs: Unpack[_RequestOptions]) -> ClientResponse:
        if server_name != self.model_name or not url_path.endswith("/v1/responses"):
            raise ValueError("Sandbox runner attempted an ungranted model route")
        body = kwargs.get("json")
        if isinstance(body, NeMoGymResponseCreateParamsNonStreaming):
            kwargs["json"] = body.model_dump(mode="json", exclude_unset=True)
        return await request(
            "POST",
            f"{self.model_url}/responses",
            _internal=True,
            _max_connection_retries=_SANDBOX_LOOP_CONNECTION_RETRIES,
            **kwargs,
        )


class SandboxBrowsecompAgent(BrowsecompAgent):
    """Token counting stays behind the adapter; it never grants upstream model credentials."""

    token_count_url: str | None = None
    agent_cookies: dict[str, str]

    async def _count_prompt_tokens(self, body: NeMoGymResponseCreateParamsNonStreaming) -> int:
        if self.token_count_url is None:
            raise ValueError("BrowseComp token counting was not granted")
        reply = await request(
            "POST",
            self.token_count_url,
            json=body.model_dump(mode="json"),
            cookies=self.agent_cookies,
            _internal=True,
            _max_connection_retries=_SANDBOX_LOOP_CONNECTION_RETRIES,
        )
        await raise_for_status(reply)
        return int((await get_response_json(reply))["count"])


async def check_reachable(urls: list[str | None], *, attempts: int = 3, timeout_s: float = 10.0) -> None:
    """Fail the activation when the sandbox cannot open a connection to a granted endpoint.

    The loop's calls resend a refused connection for about 90 seconds before they fail the activation, so checking
    first fails an endpoint the sandbox cannot reach at all without spending the rollout's budget.
    """
    for url in filter(None, urls):
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        for attempt in range(attempts):
            try:
                _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_s)
            except (OSError, TimeoutError) as error:
                if attempt == attempts - 1:
                    raise ConnectionError(f"The sandbox cannot connect to {host}:{port}: {error!r}") from error
                await asyncio.sleep(1)
            else:
                writer.close()
                break


async def run(input_path: Path, output_path: Path) -> None:
    """Persist a receipt on success, error and supervisor interruption."""
    # The supervisor's deadline counts from launch, so the loop's deadline includes the endpoint checks below.
    started_at = time.monotonic()
    payload = json.loads(input_path.read_text())
    input_path.unlink()  # The runner retains scoped cookies in memory, outside model-readable staged files.
    # Deliberately empty: the sandbox never receives the server's secret-bearing global config.
    config = OmegaConf.create({})
    os.environ["NEMO_GYM_CONFIG_DICT"] = "{}"
    seed = AgentSeedSessionRequest.model_validate(payload["seed"])
    grant = DirectHTTPToolAccess.model_validate(payload["tool_access"]) if payload["tool_access"] else None
    state = BrowsecompAgentSessionState(
        request=seed, tool_access=grant, resources_cookies=dict(grant.cookies) if grant else {}
    )
    agent_config = BrowsecompAgentConfig.model_validate(payload["config"])
    client = ScopedModelClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
        global_config_dict=config,
        model_name=agent_config.model_server.name,
        model_url=payload["model_url"],
    )
    agent = SandboxBrowsecompAgent(
        config=agent_config,
        server_client=client,
        token_count_url=payload.get("token_count_url"),
        agent_cookies=payload.get("agent_cookies", {}),
    )
    receipt = {"runtime": {"hostname": socket.gethostname(), "pid": os.getpid(), "python": sys.executable}}
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await check_reachable(
            [payload["model_url"], str(grant.base_url) if grant else None, payload.get("token_count_url")]
        )
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(payload["body"])
        http_request = Request({"type": "http", "headers": [], "path": "/v1/responses"})
        result = await agent._run_session_loop(http_request, Response(), body, state, started_at=started_at)
        receipt["response"] = result.model_dump(mode="json")
    except asyncio.CancelledError:
        receipt["error"] = "BrowseComp harness was interrupted by its supervisor"
        receipt["interrupted"] = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["traceback"] = traceback.format_exc()
    finally:
        receipt["resources_cookies"] = state.resources_cookies
        partial = output_path.with_suffix(".partial")
        partial.write_text(json.dumps(receipt))
        partial.replace(output_path)
        await get_global_aiohttp_client().close()


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1]), Path(sys.argv[2])))
