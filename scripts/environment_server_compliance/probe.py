# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed and close an agent session on a sandbox the probe lends it, and record what the agent left behind.

This runs in the component's own venv. The lent sandbox is the local provider made connectable in this process, so
the probe needs no model, container or network of its own.
"""

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
from omegaconf import DictConfig

from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig, GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    LocalProvider,
    SandboxAccess,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
    register_provider,
)

from .component import default_agent_config
from .engine import Target, Unavailable


PROVIDER_NAME = "borrowable_local"
PROVIDER_CONFIG_REF = "probe_sandbox"
PROBE_TIMEOUT_SECONDS = 600
# A non-loopback host: some agents refuse a model URL that a sandbox cannot reach.
POLICY_MODEL = {
    "responses_api_models": {
        "probe": {"name": "policy_model", "host": "probe.invalid", "port": 1, "entrypoint": "app.py", "num_workers": 1}
    }
}
# Config the agent's default one leaves unset, which its seed requires; values as benchmarks pin them.
CONFIG_OVERRIDES = {"claude_code_agent": {"claude_code_version": "2.1.261"}}
SANDBOX_REFUSAL = "does not support sandbox access"
SEED_PATH = "/v1/agent_sessions"
CLOSE_PATH = "/v1/agent_sessions/close"


@dataclass(frozen=True)
class SessionObservation:
    """How an agent answered a seed that lends a sandbox, and then the close; None where that did not happen.

    `sandbox_killed` is whether the lent sandbox was stopped once the close was accepted, or None if unknown.
    """

    sandbox_access: Literal["accepted", "refused", "failed"]
    session_close: Literal["accepted", "refused", "failed"] | None
    sandbox_killed: bool | None


class BorrowableLocalProvider(LocalProvider):
    """The local provider, connectable in this process so one sandbox can be lent to an agent.

    A sandbox that was closed is gone, as with a real provider, whatever its working directory.
    """

    name = PROVIDER_NAME
    handles: ClassVar[dict[str, SandboxHandle]] = {}
    closed: ClassVar[set[str]] = set()

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        self.handles[handle.sandbox_id] = handle
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Any) -> SandboxHandle:
        return self.handles[descriptor["sandbox_id"]]

    async def exec(self, handle: SandboxHandle, *args: Any, **kwargs: Any) -> Any:
        if handle.sandbox_id in self.closed:
            raise RuntimeError(f"sandbox {handle.sandbox_id} was closed")
        return await super().exec(handle, *args, **kwargs)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.STOPPED if handle.sandbox_id in self.closed else await super().status(handle)

    async def close(self, handle: SandboxHandle) -> None:
        self.closed.add(handle.sandbox_id)
        await super().close(handle)


def agent_config(component: Path) -> dict[str, Any]:
    """The component's default agent config, completed so that the agent can be built and seeded."""
    config = default_agent_config(component)
    return {
        "host": "127.0.0.1",
        "port": 1,
        "entrypoint": "app.py",
        **config,
        **CONFIG_OVERRIDES.get(component.name, {}),
    }


def _build_agent(agent_cls: type, config: dict[str, Any]) -> Any:
    global_config = get_global_config_dict(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=DictConfig(
                {PROVIDER_CONFIG_REF: {PROVIDER_NAME: {"concurrency": 1}}, "policy_model": POLICY_MODEL}
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
    )
    try:
        return agent_cls(
            config=agent_cls.model_fields["config"].annotation.model_validate(config),
            server_client=ServerClient(
                head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=global_config
            ),
        )
    except Exception as error:
        raise Unavailable(f"cannot build the agent from its config: {type(error).__name__}: {error}") from error


def _refuses_sandbox(reply: httpx.Response) -> bool:
    try:
        detail = reply.json().get("detail")
    except (ValueError, AttributeError):
        return False
    return reply.status_code == 422 and isinstance(detail, str) and SANDBOX_REFUSAL in detail


async def _probe(agent_cls: type, config: dict[str, Any], workdir: Path) -> SessionObservation:
    BorrowableLocalProvider.closed.clear()
    register_provider(PROVIDER_NAME, BorrowableLocalProvider, override=True)
    agent = _build_agent(agent_cls, config)
    owner = AsyncSandbox(BorrowableLocalProvider(concurrency=1))
    await owner.start(SandboxSpec(workdir=str(workdir)))
    try:
        access = SandboxAccess(
            connection=DirectSandboxConnection(
                provider_config_ref=PROVIDER_CONFIG_REF, descriptor=await owner.serialize()
            ),
            workdir=str(workdir),
        )
        episode = EpisodeId(rollout_id="compliance-probe")
        seed = AgentSeedSessionRequest(
            agent_session_id="compliance-probe",
            episode_id=episode,
            task_id=TaskId(taskset="compliance", task_id="probe"),
            sandbox_access=access,
            agent_timeout_sec=PROBE_TIMEOUT_SECONDS,
        )
        close = AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=episode)
        transport = httpx.ASGITransport(app=agent.setup_webserver(), raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
            reply = await client.post(SEED_PATH, json=seed.model_dump(mode="json"))
            if _refuses_sandbox(reply):
                return SessionObservation("refused", None, None)
            if reply.status_code != 200:
                return SessionObservation("failed", None, None)
            reply = await client.post(CLOSE_PATH, json=close.model_dump(mode="json"))
            if reply.status_code != 200:
                return SessionObservation("accepted", "failed", None)
        try:
            result = await owner.exec("true", timeout_s=30)
            killed = None if result.error_type is not None else result.return_code != 0
        except RuntimeError:
            killed = True
        return SessionObservation("accepted", "accepted", killed)
    finally:
        await owner.stop()


def probe_session(
    agent_cls: type, config: dict[str, Any], *, timeout: float = PROBE_TIMEOUT_SECONDS
) -> SessionObservation:
    """Seed a session of `agent_cls` on a lent sandbox, close it, and report what became of the sandbox."""
    with tempfile.TemporaryDirectory() as workdir:
        try:
            return asyncio.run(asyncio.wait_for(_probe(agent_cls, config, Path(workdir)), timeout))
        except TimeoutError as error:
            raise Unavailable(f"the session did not seed and close within {timeout:g}s") from error


def observe_session(target: Target) -> SessionObservation:
    """The session observation of the target's agent, probed once for all rules that need it."""

    def produce() -> SessionObservation:
        if target.component is None:
            raise Unavailable("there is no component directory to run the agent from")
        return probe_session(target.server, agent_config(target.component))

    return target.once("session", produce)
