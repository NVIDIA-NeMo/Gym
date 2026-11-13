# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stopped sandbox's measured usage reaches the record of whatever stopped it, once, with no server code."""

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import PrivateAttr

from nemo_gym.agents.agent_activation_types import AgentContinuationCapabilities, AgentContinuationRequirements
from nemo_gym.agents.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.agents.utils.sandbox_observation import lifecycle_observation
from nemo_gym.artifacts.rollout_rows import _rollout_for_export
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_context import rollout_context
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxUsageSample
from nemo_gym.server_runtime.base_environment_server import BaseEnvironmentServerConfig
from nemo_gym.server_runtime.base_resources_server import BaseRunRequest
from nemo_gym.server_runtime.sandbox_usage import fill_reply_sandbox_usage, fill_sandbox_usage
from nemo_gym.server_runtime.verifier_observability import (
    bind_stopped_sandbox,
    observe_verifier,
    observe_verifier_sandbox,
)
from nemo_gym.trajectory.trajectory_record import (
    NG_AGENT_OBSERVATIONS_KEY,
    AgentObservationBundle,
    SandboxObservation,
    resource_usage_fields,
)
from tests.unit_tests.sandbox.test_sandbox_connect import FakeConnectableProvider
from tests.unit_tests.server_runtime.test_environment_server import _EnvironmentServer, _request, _Response


def _usage(source: str, scope: str, kind: str, *, cpu: int, peak: int) -> SandboxUsageSample:
    return SandboxUsageSample(
        source=source,
        scope=scope,
        sampled_at=1.0,
        cpu_usage_usec_total=cpu,
        cpu_kind=kind,
        memory_peak_bytes=peak,
        peak_kind=kind,
    )


CONTAINER = _usage("cgroup_v2", "container", "counter", cpu=3_000_000, peak=2**30)


class MeasuredProvider(FakeConnectableProvider):
    """A connectable in-memory provider whose sandboxes report the given usage samples."""

    def __init__(self, *samples: SandboxUsageSample) -> None:
        self.samples = samples or (CONTAINER,)

    async def usage(self, handle) -> tuple[SandboxUsageSample, ...]:
        return self.samples


async def _started(*samples: SandboxUsageSample) -> AsyncSandbox:
    return await AsyncSandbox(MeasuredProvider(*samples)).start(SandboxSpec(image="task"))


def test_only_a_container_counter_fills_the_scalar_fields() -> None:
    container = resource_usage_fields([{**_summary(CONTAINER)}])
    assert (container["cpu_time_s"], container["peak_memory_mib"], container["resource_usage_source"]) == (
        3.0,
        1024.0,
        "cgroup_v2:container",
    )
    lower_bounds = resource_usage_fields(
        [
            _summary(_usage("procfs_mntns", "process_tree", "sampled_lower_bound", cpu=1, peak=1)),
            _summary(_usage("cgroup_v2", "shared_step", "counter", cpu=9, peak=9)),
        ]
    )
    assert set(lower_bounds) == {"resource_usage"} and resource_usage_fields([]) == {}


def _summary(sample: SandboxUsageSample) -> dict:
    return {
        "source": sample.source,
        "scope": sample.scope,
        "clock": sample.clock,
        "first_sampled_at": sample.sampled_at,
        "last_sampled_at": sample.sampled_at,
        "sample_count": 1,
        "cpu_usage_usec_total": sample.cpu_usage_usec_total,
        "cpu_kind": sample.cpu_kind,
        "memory_peak_bytes": sample.memory_peak_bytes,
        "peak_kind": sample.peak_kind,
    }


async def test_a_verification_records_each_sandbox_it_stops_once_and_it_reaches_the_exported_trajectory() -> None:
    lent = await _started()
    await lent.serialize()  # handed to an agent, which connect()s and measures nothing
    grading = await _started(_usage("procfs_mntns", "process_tree", "sampled_lower_bound", cpu=500_000, peak=7))
    unmeasured = await AsyncSandbox(FakeConnectableProvider()).start(SandboxSpec(image="task"))
    outside = await _started()

    async def verify() -> dict:
        # No usage code here: stopping is enough. An explicit record of the same sandbox replaces the automatic
        # one and keeps its usage, so the lent sandbox still appears once.
        await lent.stop()
        observe_verifier_sandbox(lifecycle_observation(lent, role="agent"))
        await grading.stop()
        await unmeasured.stop()
        return {"response": {"id": "r", "output": []}, "reward": 1.0}

    with rollout_context("0-0"):
        result = await observe_verifier(verify, enabled=True)()
    await outside.stop()  # stopped outside any verification: recorded nowhere

    trajectory = _build_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, result)
    exported = _rollout_for_export({"ng_trajectory": trajectory.model_dump(mode="json")})["ng_trajectory"]
    observations = exported["verifier_invocations"][0]["sandbox_observations"]
    assert [(item["sandbox_id"], item["role"]) for item in observations] == [
        (lent.sandbox_id, "agent"),
        (grading.sandbox_id, "verifier"),
    ]
    agent, grader = observations
    assert (agent["cpu_time_s"], agent["teardown_confirmed"]) == (3.0, True)
    assert [item["scope"] for item in agent["resource_usage"]] == ["container"]
    # A process-tree lower bound never fills the sandbox's own scalars.
    assert grader["cpu_time_s"] is None and grader["resource_usage"][0]["peak_kind"] == "sampled_lower_bound"


class Agent(SimpleResponsesAPIAgent):
    """Creates, observes and stops its own sandbox the way a sandboxed harness does, with no usage code."""

    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body: BaseRunRequest) -> dict:
        sandbox = await _started()
        # Built before the stop, as a harness reports where its tools ran.
        bundle = AgentObservationBundle(
            source="test", records=[SandboxObservation(role="agent", sandbox_id=sandbox.sandbox_id)]
        )
        await sandbox.stop()
        return {"reward": 1.0, NG_AGENT_OBSERVATIONS_KEY: bundle.model_dump(mode="json")}

    async def _seed_agent_session_state(self, body):
        return SessionState(request=body, sandbox=await _started())

    async def _close_agent_session_state(self, state):
        bundle = AgentObservationBundle(
            source="test",
            records=[
                SandboxObservation(role="agent", sandbox_id=state.sandbox.sandbox_id),
                SandboxObservation(role="agent", sandbox_id="borrowed-elsewhere"),
            ],
        )
        await state.sandbox.stop()
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id, agent_observations=bundle)


@dataclass
class SessionState(AgentSessionState):
    sandbox: AsyncSandbox = field(default=None)


@pytest.fixture
def agent() -> Agent:
    return Agent(
        config=BaseResponsesAPIAgentConfig(host="localhost", port=1, entrypoint="app.py", name="test"),
        server_client=MagicMock(spec=ServerClient),
    )


def test_a_run_reports_the_usage_of_the_sandbox_it_stopped(agent: Agent) -> None:
    with TestClient(agent.setup_webserver()) as client:
        reply = client.post("/run", json={"responses_create_params": {"input": "hi"}})

    assert reply.status_code == 200, reply.text
    (record,) = reply.json()[NG_AGENT_OBSERVATIONS_KEY]["records"]
    assert record["cpu_time_s"] == 3.0 and record["resource_usage"][0]["scope"] == "container"


async def test_a_session_close_reports_the_usage_of_the_sandboxes_it_stopped(agent: Agent) -> None:
    body = AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id={"rollout_id": "session"},
        task_id={"taskset": "test", "task_id": "task"},
    )
    request = SimpleNamespace(session={}, headers={}, path_params={})
    await agent.seed_agent_session(request, body)

    receipt = await agent.close_agent_session(
        request, AgentCloseSessionRequest(agent_session_id="session", episode_id=body.episode_id)
    )

    own, borrowed = receipt.agent_observations.records
    assert own.cpu_time_s == 3.0 and len(own.resource_usage) == 1
    # A sandbox this agent only borrowed is measured by its creator, never here.
    assert borrowed.resource_usage is None and borrowed.cpu_time_s is None


def test_usage_already_on_an_observation_is_kept() -> None:
    recorded = SandboxObservation(role="agent", sandbox_id="box", **resource_usage_fields([_summary(CONTAINER)]))
    bundle = AgentObservationBundle(source="test", records=[recorded])
    other = resource_usage_fields([_summary(_usage("cgroup_v2", "container", "counter", cpu=9_000_000, peak=1))])

    fill_sandbox_usage(bundle, {"box": other})
    fill_sandbox_usage(None, {"box": other})

    assert bundle.records[0].cpu_time_s == 3.0


async def test_usage_fills_a_record_the_server_made_before_the_stop() -> None:
    sandbox = await _started()
    unmeasured = await AsyncSandbox(FakeConnectableProvider()).start(SandboxSpec(image="task"))

    async def verify() -> dict:
        observe_verifier_sandbox(SandboxObservation(role="environment"))  # names no sandbox: kept as given
        observe_verifier_sandbox(lifecycle_observation(sandbox, role="environment"))
        await sandbox.stop()
        await unmeasured.stop()
        bind_stopped_sandbox(unmeasured)  # nothing measured: nothing recorded
        return {"reward": 1.0}

    with rollout_context("0-0"):
        result = await observe_verifier(verify, enabled=True)()

    unnamed, recorded = result["ng_verifier_invocations"][0]["sandbox_observations"]
    assert unnamed["sandbox_id"] is None and "resource_usage" not in unnamed
    assert (recorded["role"], recorded["sandbox_id"], recorded["cpu_time_s"]) == (
        "environment",
        sandbox.sandbox_id,
        3.0,
    )


class ContinuationAgent(Agent):
    """A multi-turn agent whose close waits on ``_release`` before stopping its sandbox."""

    _release: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _entered: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _closes: int = PrivateAttr(default=0)

    def _agent_continuation_capabilities(self):
        return AgentContinuationCapabilities()

    def _continuation_close_timeout(self):
        return 1.0

    async def _close_agent_session_state(self, state):
        self._closes += 1
        self._entered.set()
        await self._release.wait()
        return await super()._close_agent_session_state(state)


async def test_a_retried_continuation_close_still_reports_the_usage() -> None:
    agent = ContinuationAgent(
        config=BaseResponsesAPIAgentConfig(host="localhost", port=1, entrypoint="app.py", name="test"),
        server_client=MagicMock(spec=ServerClient),
    )
    body = AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id={"rollout_id": "session"},
        task_id={"taskset": "test", "task_id": "task"},
        continuation=AgentContinuationRequirements(),
    )
    request = SimpleNamespace(session={}, headers={}, path_params={})
    await agent.seed_agent_session(request, body)
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=body.episode_id)

    # The first waiter is cancelled while the shielded close is still stopping the sandbox; a retry joins it.
    waiter = asyncio.create_task(agent.close_agent_session(request, close))
    await agent._entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    agent._release.set()
    receipt = await agent.close_agent_session(request, close)
    again = await agent.close_agent_session(request, close)

    own = receipt.agent_observations.records[0]
    assert agent._closes == 1 and own.cpu_time_s == 3.0 and len(own.resource_usage) == 1
    assert again.agent_observations.records[0].cpu_time_s == 3.0


class _ObservedResponse(_Response):
    ng_agent_observations: AgentObservationBundle | None = None


class _LendingEnvironment(_EnvironmentServer):
    """Lends a measured sandbox to its agent and stops it in the episode's cleanup, with no usage code."""

    response_model = _ObservedResponse

    async def run(self, request, cleanup):
        sandbox = await _started()
        await sandbox.serialize()  # lent to the agent, which reports where its tools ran and measures nothing
        cleanup.register_cleanup("sandbox", sandbox.stop)
        bundle = AgentObservationBundle(
            source="agent", records=[SandboxObservation(role="agent", sandbox_id=sandbox.sandbox_id)]
        )
        return _ObservedResponse(
            episode_id=request.episode_id, task_id=request.task.task_id, result="ok", ng_agent_observations=bundle
        )


async def test_an_episode_reports_the_usage_of_the_sandbox_it_lent_and_stopped_in_cleanup() -> None:
    server = _LendingEnvironment(
        config=BaseEnvironmentServerConfig(
            name="environment", host="127.0.0.1", port=1234, entrypoint="app.py", cleanup_timeout_seconds=5
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    response = await server.run_request(_request())

    (record,) = response.ng_agent_observations.records
    assert record.cpu_time_s == 3.0 and record.resource_usage[0].scope == "container"


def test_bundles_nested_in_a_reply_are_filled() -> None:
    usage = {"box": resource_usage_fields([_summary(CONTAINER)])}
    record = SandboxObservation(role="agent", sandbox_id="box").model_dump(mode="json")
    # An Environment Server's episode result, with a per-turn bundle, as JSON.
    reply = {
        "result": {
            "ng_agent_observations": {"source": "a", "records": [dict(record)]},
            "turns": [{"ng_agent_observations": {"source": "a", "records": [dict(record)]}}],
        },
        "response": {"output": [{"text": "unrelated"}]},
    }

    fill_reply_sandbox_usage(reply, usage)

    filled = [reply["result"]["ng_agent_observations"], reply["result"]["turns"][0]["ng_agent_observations"]]
    assert [bundle["records"][0]["cpu_time_s"] for bundle in filled] == [3.0, 3.0]
