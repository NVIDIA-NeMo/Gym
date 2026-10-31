# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment Server sessions: TB4 lends its task sandbox through SandboxAccess and grades on /verify."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.terminal_bench_4 import app as app_module
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.app import TerminalBench4Config, TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import (
    AgentTermination,
    ResourcesSessionVerifyRequest,
    SeedSessionResponse,
)
from resources_servers.terminal_bench_4.tests.test_environment import make_environment


BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "terminal_bench_4"
TASK = "atrx-vep-crispr"
OTHER_TASK = "batched-eval-parity"


class FakeEnvironment:
    def __init__(self, workdir: str = "/app") -> None:
        self.workdir = workdir
        self.main = None
        self.uses_compose = False
        self.closed = False
        self.stops = 0
        self.stop_error: Exception | None = None
        # Fields the server persists for every environment.
        self.session_id = "tb4-environment"
        self.resources = [{"service": "main", "sandbox_id": "task-sandbox"}]
        self.cleanup_errors = []

    async def agent_workdir(self) -> str:
        return self.workdir

    async def stop(self) -> None:
        self.stops += 1
        if self.stop_error is not None:
            raise self.stop_error
        self.closed = True


class Harness:
    """A TB4 server whose provisioning and grading are faked at the lifecycle boundary."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config_overrides) -> None:
        monkeypatch.chdir(tmp_path)
        resources = OmegaConf.load(BENCHMARK_DIR / "resources.yaml").terminal_bench_4
        merged = OmegaConf.merge(resources, {"resources_servers": {"terminal_bench_4": config_overrides}})
        config = OmegaConf.to_container(merged.resources_servers.terminal_bench_4, resolve=True)
        config |= {"host": "", "port": 0, "name": "terminal_bench_4", "artifacts_dir": str(tmp_path / "jobs")}
        self.server = TerminalBench4ResourcesServer(
            config=TerminalBench4Config.model_validate(config), server_client=MagicMock(spec=ServerClient)
        )
        manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text())
        self.row = {
            "task_id": f"terminal-bench/{TASK}",
            "task_name": f"terminal-bench/{TASK}",
            "task_ref": next(task["ref"] for task in manifest["tasks"] if task["name"] == TASK),
            "dataset_ref": manifest["ref"],
            "category": "cpu",
        }
        self.agent_user = None
        self.prepare_error: Exception | None = None
        self.prepared = 0
        self.graded: list[bool] = []
        self.environments: list[FakeEnvironment] = []
        self.release_prepare = asyncio.Event()
        self.release_prepare.set()
        monkeypatch.setattr(
            TerminalBench4ResourcesServer, "_prepare_session", lambda server, session: self._prepare(session)
        )
        monkeypatch.setattr(lifecycle, "finalize_session", self._finalize)

    async def _prepare(self, session) -> None:
        self.prepared += 1
        await self.release_prepare.wait()
        session.task = SimpleNamespace(config=SimpleNamespace(agent=SimpleNamespace(user=self.agent_user)))
        session.environment, session.verifier_environment = FakeEnvironment(), FakeEnvironment()
        self.environments += [session.environment, session.verifier_environment]
        session.result["environment_setup"] = {"started_at": "t0", "finished_at": "t1"}
        if self.prepare_error is not None:
            # The real _prepare_session records the failure, cleans up, and keeps a terminated seed.
            session.termination = AgentTermination(reason="infrastructure_error", detail=str(self.prepare_error))
            await lifecycle.cleanup(session)
            session.seed_response = SeedSessionResponse(session_id=session.session_id, termination=session.termination)
            return
        session.seed_response = SeedSessionResponse(
            session_id=session.session_id, sandbox_descriptor={"sandbox_id": "task-sandbox"}
        )
        session.phase = "ready"

    async def _finalize(self, session, *, grade) -> None:
        self.graded.append(grade)
        if grade:
            session.result["verifier_result"] = {
                "rewards": {"reward": 0},
                "num_tests_passed": 3,
                "num_tests_total": 4,
                "num_tests_failed": 1,
                "num_tests_skipped": 0,
            }
        await lifecycle.cleanup(session)

    def seed(self, session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
        return ResourcesSeedSessionRequest(
            resources_session_id=session_id,
            episode_id={"rollout_id": "rollout", "attempt": 0},
            task_id={"taskset": "terminal_bench_4", "task_id": self.row["task_id"]},
            task_data=self.row,
        )

    @staticmethod
    def verify_request(body: dict) -> ResourcesSessionVerifyRequest:
        return ResourcesSessionVerifyRequest.model_validate(body)

    def verify_body(self, status: str = "completed", **response_fields) -> dict:
        response = {
            "id": "resp",
            "created_at": 0,
            "model": "test-model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": status,
            **response_fields,
        }
        # The Environment Server's verify body: the task row plus the prompt and the agent's response.
        return self.row | {
            "responses_create_params": {"input": [{"role": "user", "content": "Solve the task."}]},
            "response": response,
        }


@pytest.fixture
def harness(tmp_path, monkeypatch) -> Harness:
    return Harness(tmp_path, monkeypatch, sandbox_provider_ref="task_sandbox")


def close_body(seed: ResourcesSeedSessionRequest) -> dict:
    return ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id=seed.episode_id
    ).model_dump(mode="json")


def masked(result) -> tuple:
    """The lost-session fields of a verify result, from HTTP JSON or a direct call's response."""
    payload = result if isinstance(result, dict) else json.loads(result.body)
    return payload["reward"], payload["mask_sample"], payload["failure_kind"]


def test_lends_the_task_sandbox_grades_once_and_releases_on_close(harness) -> None:
    seed = harness.seed()
    with TestClient(harness.server.setup_webserver()) as client:
        seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {
            "resources_session_id": "resources-session",
            "resources_tools": None,
            "sandbox_access": {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "task_sandbox",
                    "descriptor": {"sandbox_id": "task-sandbox"},
                },
                "workdir": "/app",
            },
        }
        assert client.cookies
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).json() == seeded.json()
        assert harness.prepared == 1

        verified = client.post("/verify", json=harness.verify_body())
        assert verified.status_code == 200, verified.text
        result = verified.json()
        assert (result["reward"], result["partial_credit"], result["evaluation_completed"]) == (0.0, 0.75, True)
        assert result["termination"]["reason"] == "completed"
        assert result["task_id"] == harness.row["task_name"]
        assert result["response"]["model"] == "test-model"
        assert harness.graded == [True]
        # A lost verdict replays without grading again; a different request cannot reuse it.
        assert client.post("/verify", json=harness.verify_body()).json() == result
        assert client.post("/verify", json=harness.verify_body(id="other")).status_code == 409
        assert harness.graded == [True]

        assert client.post("/close_session", json=close_body(seed)).status_code == 200
        assert client.post("/close_session", json=close_body(seed)).status_code == 200
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 409
        lost = client.post("/verify", json=harness.verify_body())
        assert lost.status_code == 200, lost.text
        assert masked(lost.json()) == (0.0, True, "session_lost")
        assert lost.json()["response"]["model"] == "test-model"
    assert all(environment.closed for environment in harness.environments)
    record = json.loads(next((Path("jobs")).glob("*.json")).read_text())
    assert record["resources_session_id"] == "resources-session"


def test_shared_resources_session_contract(harness) -> None:
    check_resources_session_contract(
        harness.server.setup_webserver(), harness.seed(), keeps_state=True, verify_body=harness.verify_body()
    )
    assert harness.graded == []
    assert all(environment.closed for environment in harness.environments)


async def test_concurrent_seeds_share_one_provisioning(harness) -> None:
    harness.release_prepare.clear()
    requests = [SimpleNamespace(session={SESSION_ID_KEY: f"cookie-{index}"}) for index in range(4)]
    seeds = [asyncio.create_task(harness.server.seed_session(request, harness.seed())) for request in requests]
    await asyncio.sleep(0)
    harness.release_prepare.set()
    results = await asyncio.gather(*seeds)
    assert all(result == results[0] for result in results)
    assert harness.prepared == 1


@pytest.mark.parametrize("field", ["episode_id", "task_id", "task_data"])
async def test_seed_rejects_a_changed_request(harness, field) -> None:
    request = SimpleNamespace(session={})
    await harness.server.seed_session(request, harness.seed())
    changed = harness.seed()
    if field == "episode_id":
        changed.episode_id = changed.episode_id.model_copy(update={"attempt": 1})
    elif field == "task_id":
        changed.task_id = changed.task_id.model_copy(update={"task_id": "other"})
    else:
        changed.task_data["category"] = "gpu"
    with pytest.raises(HTTPException) as error:
        await harness.server.seed_session(request, changed)
    assert error.value.status_code == 409
    assert harness.prepared == 1


async def test_close_before_seed_fences_a_late_seed_and_checks_the_episode(harness) -> None:
    seed = harness.seed()
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    with pytest.raises(HTTPException, match="already closed"):
        await harness.server.seed_session(SimpleNamespace(session={}), seed)
    other = ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id={"rollout_id": "other", "attempt": 0}
    )
    with pytest.raises(HTTPException) as error:
        await harness.server.close_resources_session(other)
    assert error.value.status_code == 409
    assert harness.prepared == 0


async def test_close_without_verification_releases_without_grading(harness) -> None:
    seed = harness.seed()
    await harness.server.seed_session(SimpleNamespace(session={}), seed)
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    session = next(iter(harness.server._sessions.values()))
    assert harness.graded == []
    assert session.phase == "closed"
    assert session.termination.reason == "cancelled"
    assert all(environment.closed for environment in harness.environments)


async def test_close_fences_inflight_provisioning_and_waits_for_its_cleanup(harness) -> None:
    harness.release_prepare.clear()
    seed = harness.seed()
    seeding = asyncio.create_task(harness.server.seed_session(SimpleNamespace(session={}), seed))
    await asyncio.sleep(0)
    closing = asyncio.create_task(
        harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    )
    await asyncio.sleep(0)
    assert not closing.done()
    harness.release_prepare.set()
    seeded, closed = await asyncio.gather(seeding, closing, return_exceptions=True)
    # The fenced seed releases its sandboxes instead of lending them.
    assert isinstance(seeded, HTTPException) and seeded.status_code == 409
    assert "closed during setup" in seeded.detail
    assert closed.resources_session_id == seed.resources_session_id
    assert harness.graded == []
    assert all(environment.closed for environment in harness.environments)
    session = next(iter(harness.server._sessions.values()))
    assert session.termination.reason == "cancelled"


async def test_cancelled_close_fences_slow_provisioning_until_its_cleanup(harness, monkeypatch) -> None:
    now = 0.0
    monkeypatch.setattr(app_module, "monotonic", lambda: now)
    harness.release_prepare.clear()
    seed = harness.seed()
    seeding = asyncio.create_task(harness.server.seed_session(SimpleNamespace(session={}), seed))
    await asyncio.sleep(0)
    wrong = close_body(seed) | {"episode_id": {"rollout_id": "rollout", "attempt": 1}}
    with pytest.raises(HTTPException, match="episode_id"):
        await harness.server.close_resources_session(ResourcesCloseSessionRequest(**wrong))
    assert seed.resources_session_id not in harness.server._closed_resources
    closing = asyncio.create_task(
        harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    )
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    # A pending allocation keeps its fence past the normal retry window.
    now = 1000.0
    other = close_body(seed) | {"resources_session_id": "other"}
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**other))
    assert seed.resources_session_id in harness.server._closed_resources
    assert seed.resources_session_id not in harness.server._close_expiry
    harness.release_prepare.set()
    with pytest.raises(HTTPException, match="closed during setup"):
        await seeding
    # The late seed's teardown completes the close and starts its retention window.
    assert harness.server._close_expiry[seed.resources_session_id] == 1300
    assert harness.graded == []
    assert all(environment.closed for environment in harness.environments)


async def test_cancelled_close_still_releases_every_sandbox_and_the_slot(tmp_path, monkeypatch) -> None:
    harness = Harness(tmp_path, monkeypatch, sandbox_provider_ref="task_sandbox", max_concurrent_sessions=1)
    monkeypatch.setattr(app_module, "monotonic", lambda: 0.0)
    seed = harness.seed()
    await harness.server.seed_session(SimpleNamespace(session={}), seed)
    session = next(iter(harness.server._sessions.values()))
    # The real provisioning holds an admission slot until cleanup.
    await session.slots.acquire()
    session.owns_slot = True
    task_sandbox, deletion, deleting = harness.environments[0], asyncio.Event(), []
    stop = task_sandbox.stop

    async def delete():
        await deletion.wait()
        await stop()

    async def gated_stop():
        # Like Environment.stop: the provider deletion is shielded from its caller's cancellation.
        if not deleting:
            deleting.append(asyncio.create_task(delete()))
        await asyncio.shield(deleting[0])

    task_sandbox.stop = gated_stop
    closing = asyncio.create_task(
        harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    )
    while not deleting:
        await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    deletion.set()

    async def retention_recorded():
        while seed.resources_session_id not in harness.server._close_expiry:
            await asyncio.sleep(0)

    await asyncio.wait_for(retention_recorded(), 5)
    assert all(environment.closed for environment in harness.environments)
    assert not session.owns_slot and not harness.server._slots.locked()
    assert harness.server._close_expiry[seed.resources_session_id] == 300
    assert session.termination.reason == "cancelled"
    assert harness.graded == []


async def test_failed_close_keeps_its_fence_past_the_retry_window(harness, monkeypatch) -> None:
    now = 0.0
    monkeypatch.setattr(app_module, "monotonic", lambda: now)
    request = SimpleNamespace(session={})
    seed = harness.seed()
    await harness.server.seed_session(request, seed)
    harness.environments[0].stop_error = RuntimeError("provider unavailable")
    with pytest.raises(HTTPException) as error:
        await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    assert error.value.status_code == 503
    now = 1000.0
    other = close_body(seed) | {"resources_session_id": "other"}
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**other))
    assert seed.resources_session_id in harness.server._closed_resources
    assert seed.resources_session_id not in harness.server._close_expiry
    with pytest.raises(HTTPException, match="already closed"):
        await harness.server.seed_session(request, seed)
    lost = await harness.server.verify(request, harness.verify_request(harness.verify_body()))
    assert masked(lost) == (0.0, True, "session_lost")
    harness.environments[0].stop_error = None
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    assert harness.server._close_expiry[seed.resources_session_id] == 1300
    assert all(environment.closed for environment in harness.environments)


async def test_close_retries_a_failed_task_sandbox_deletion(harness, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "monotonic", lambda: 0.0)
    seed = harness.seed()
    await harness.server.seed_session(SimpleNamespace(session={}), seed)
    session = next(iter(harness.server._sessions.values()))
    # The real wrapper, whose callers share one teardown task.
    session.environment, box, _, _ = make_environment(tmp_path, monkeypatch)
    await session.environment.start()
    box.stop.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(HTTPException) as error:
        await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    assert error.value.status_code == 503
    assert seed.resources_session_id not in harness.server._close_expiry
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    assert box.stop.await_count == 2 and lifecycle.released(session)
    assert harness.server._close_expiry[seed.resources_session_id] == 300


async def test_completed_close_records_expire_without_retry_renewal(harness, monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(app_module, "monotonic", lambda: now)
    request = SimpleNamespace(session={})
    seed = harness.seed()
    key = seed.resources_session_id
    await harness.server.seed_session(request, seed)
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    assert harness.server._close_expiry[key] == 400
    now = 200.0
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    assert harness.server._close_expiry[key] == 400
    with pytest.raises(HTTPException, match="already closed"):
        await harness.server.seed_session(request, seed)
    now = 401.0
    other = close_body(seed) | {"resources_session_id": "other"}
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**other))
    assert key not in harness.server._closed_resources
    assert key not in harness.server._close_expiry
    assert key not in harness.server._resources_sessions
    assert key not in harness.server._resources_locks
    assert not harness.server._sessions and not harness.server._by_identity
    lost = await harness.server.verify(request, harness.verify_request(harness.verify_body()))
    assert masked(lost) == (0.0, True, "session_lost")
    assert "No resources session" in json.loads(lost.body)["failure_reason"]
    assert harness.prepared == 1
    assert harness.graded == []


async def test_a_waiter_on_an_expired_lock_joins_the_replacement(harness, monkeypatch) -> None:
    now = 0.0
    monkeypatch.setattr(app_module, "monotonic", lambda: now)
    seed = harness.seed()
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    old = harness.server._resources_locks[seed.resources_session_id]
    await old.acquire()
    waiting = asyncio.create_task(harness.server.seed_session(SimpleNamespace(session={}), seed))
    try:
        try:
            await asyncio.sleep(0)
            assert not waiting.done()
            now = 301.0
            current = await asyncio.wait_for(harness.server.seed_session(SimpleNamespace(session={}), seed), 5)
        finally:
            old.release()
        assert await asyncio.wait_for(waiting, 5) == current
        assert harness.server._resources_locks[seed.resources_session_id] is not old
        assert harness.prepared == 1
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))


async def test_failed_provisioning_is_retryable_and_replayed(harness) -> None:
    harness.prepare_error = RuntimeError("image pull failed")
    seed = harness.seed()
    for _ in range(2):
        with pytest.raises(HTTPException) as error:
            await harness.server.seed_session(SimpleNamespace(session={}), seed)
        assert error.value.status_code == 503
        assert "image pull failed" in error.value.detail
    assert harness.prepared == 1
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    assert all(environment.closed for environment in harness.environments)


async def test_tasks_with_a_dedicated_agent_user_are_rejected(harness) -> None:
    harness.agent_user = "agent"
    with pytest.raises(HTTPException) as error:
        await harness.server.seed_session(SimpleNamespace(session={}), harness.seed())
    assert error.value.status_code == 422
    assert "default user" in error.value.detail
    assert all(environment.closed for environment in harness.environments)


async def test_relative_workdir_fails_provisioning(harness, monkeypatch) -> None:
    async def prepare_relative(server, session):
        await harness._prepare(session)
        session.environment.workdir = "relative"

    monkeypatch.setattr(TerminalBench4ResourcesServer, "_prepare_session", prepare_relative)
    with pytest.raises(HTTPException) as error:
        await harness.server.seed_session(SimpleNamespace(session={}), harness.seed())
    assert error.value.status_code == 503
    assert all(environment.closed for environment in harness.environments)


@pytest.mark.parametrize(
    "status,details,reason",
    [
        ("completed", None, "completed"),
        ("incomplete", None, "timeout"),
        ("incomplete", {"reason": "max_output_tokens"}, "nonzero_exit"),
        # An agent that died while its sandbox lives (codex_agent verify_after_agent_error).
        ("failed", None, "nonzero_exit"),
    ],
)
async def test_termination_follows_the_agent_response_and_grading_still_runs(harness, status, details, reason) -> None:
    request = SimpleNamespace(session={})
    await harness.server.seed_session(request, harness.seed())
    body = harness.verify_body(status, incomplete_details=details)
    verified = await harness.server.verify(request, harness.verify_request(body))
    assert verified.termination.reason == reason
    assert verified.evaluation_completed
    assert harness.graded == [True]


async def test_verify_of_an_expired_session_is_masked_and_not_graded(harness) -> None:
    request = SimpleNamespace(session={})
    seed = harness.seed()
    await harness.server.seed_session(request, seed)
    session = next(iter(harness.server._sessions.values()))
    session.agent_deadline = asyncio.get_running_loop().time()
    expired = await harness.server.verify(request, harness.verify_request(harness.verify_body()))
    assert masked(expired) == (0.0, True, "session_expired")
    await harness.server.close_resources_session(ResourcesCloseSessionRequest(**close_body(seed)))
    expired = await harness.server.verify(request, harness.verify_request(harness.verify_body()))
    assert masked(expired) == (0.0, True, "session_expired")
    assert harness.graded == [False]
    assert all(environment.closed for environment in harness.environments)


async def test_verify_rejects_another_task_without_consuming_the_session(harness) -> None:
    request = SimpleNamespace(session={})
    await harness.server.seed_session(request, harness.seed())
    other = harness.verify_body() | {"task_name": f"terminal-bench/{OTHER_TASK}"}
    with pytest.raises(HTTPException) as error:
        await harness.server.verify(request, harness.verify_request(other))
    assert error.value.status_code == 409
    verified = await harness.server.verify(request, harness.verify_request(harness.verify_body()))
    assert (verified.reward, verified.partial_credit) == (0.0, 0.75)


async def test_seed_requires_a_provider_reference(tmp_path, monkeypatch) -> None:
    harness = Harness(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as error:
        await harness.server.seed_session(SimpleNamespace(session={}), harness.seed())
    assert error.value.status_code == 422
    assert harness.prepared == 0


async def test_seed_rejects_an_unpinned_task(harness) -> None:
    seed = harness.seed()
    seed.task_data["task_ref"] = "sha256:" + "0" * 64
    with pytest.raises(HTTPException) as error:
        await harness.server.seed_session(SimpleNamespace(session={}), seed)
    assert error.value.status_code == 422


def test_legacy_seed_and_verify_shapes_still_route_to_the_tb4_session_path(harness, monkeypatch) -> None:
    async def tb4_prepare(server, session):
        session.seed_response = SeedSessionResponse(session_id=session.session_id, task_id=session.request.task_name)

    monkeypatch.setattr(TerminalBench4ResourcesServer, "_prepare_session", tb4_prepare)
    legacy = harness.row | {"responses_create_params": {"input": []}, "rollout_id": "rollout-1"}
    with TestClient(harness.server.setup_webserver()) as client:
        response = client.post("/seed_session", json=legacy)
        assert response.status_code == 200, response.text
        assert response.json()["session_id"].startswith("tb4-")
        assert "sandbox_access" not in response.json()
        unknown = client.post(
            "/verify",
            json={
                "session_id": "tb4-" + "0" * 32,
                "termination": {"reason": "completed"},
                **harness.verify_body(),
            },
        )
        assert unknown.status_code == 404


def test_registered_native_verify_binds_owned_sandbox_lifecycle(tmp_path, monkeypatch):
    # Actual native session route and AsyncSandbox lifecycle; provisioning and grade are controlled.
    from nemo_gym.sandbox import AsyncSandbox, SandboxHandle, SandboxSpec

    original_prepare = Harness._prepare
    original_stop = FakeEnvironment.stop

    class Provider:
        def __init__(self, identity):
            self.identity = identity

        async def create(self, spec):
            return SandboxHandle(self.identity, "scripted-sdk", None)

        async def close(self, handle):
            return None

        async def aclose(self):
            return None

    async def prepare(self, session):
        await original_prepare(self, session)
        for role, environment in (("agent", session.environment), ("verifier", session.verifier_environment)):
            environment.main = AsyncSandbox(Provider(role), SandboxSpec(image="scripted"))
            await environment.main.start()

    async def stop(environment):
        await environment.main.stop()
        await original_stop(environment)

    monkeypatch.setattr(Harness, "_prepare", prepare)
    monkeypatch.setattr(FakeEnvironment, "stop", stop)
    harness = Harness(tmp_path, monkeypatch, sandbox_provider_ref="task_sandbox")
    harness.server.server_client.global_config_dict = {"observability_enabled": True}
    with TestClient(harness.server.setup_webserver()) as client:
        seed = client.post("/ng-rollout/native-lifecycle/seed_session", json=harness.seed().model_dump(mode="json"))
        assert seed.status_code == 200
        verified = client.post("/ng-rollout/native-lifecycle/verify", json=harness.verify_body())
        assert verified.status_code == 200, verified.text
        result = verified.json()
        invocation = result["ng_verifier_invocations"][0]
        assert invocation["status"] == "completed"
        assert invocation.get("grading_operations") is None
        facts = invocation["sandbox_observations"]
        assert [(fact["role"], fact["sandbox_id"]) for fact in facts] == [("agent", "agent"), ("verifier", "verifier")]
        assert all(fact["teardown_confirmed"] is True and fact["observed_client_interval_s"] >= 0 for fact in facts)
        assert all(fact["outcome"] == "unknown" for fact in facts)
        assert result["reward"] == 0 and result["partial_credit"] == 0.75
        assert harness.graded == [True]
