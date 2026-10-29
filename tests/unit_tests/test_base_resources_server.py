# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    ResourcesSessionLostError,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.core import (
    JUDGE_FAILED,
    SESSION_EXPIRED,
    SESSION_LOST,
    EpisodeId,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    TaskId,
)
from nemo_gym.reward_profile import select_measured
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.session_routing import ResourcesSessionLifetimes
from nemo_gym.single_agent_turn_types import SingleAgentTurnResult
from nemo_gym.testing.session_conformance import check_resources_session_contract


def _resources_server() -> SimpleResourcesServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")

    class TestSimpleResourcesServer(SimpleResourcesServer):
        async def verify(self, body):
            pass

    return TestSimpleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestBaseVerifyResponse:
    def test_failure_reason_defaults_none_and_round_trips(self) -> None:
        response = BaseVerifyResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=0.0,
        )
        assert response.failure_reason is None
        assert response.model_dump()["failure_reason"] is None

        rescued = response.model_copy(update={"failure_reason": "judge response unparseable after 3 attempts"})
        assert rescued.model_dump()["failure_reason"] == "judge response unparseable after 3 attempts"
        assert rescued.reward == 0.0


class TestBaseMultiRewardVerifyResponse:
    def test_reward_components_round_trip(self) -> None:
        response = BaseMultiRewardVerifyResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=2.0,
            reward_components={"correctness": 1.0, "format": 1.0},
        )
        dumped = response.model_dump()
        assert dumped["reward_components"] == {"correctness": 1.0, "format": 1.0}
        assert dumped["reward"] == 2.0


class TestBaseResourcesServer:
    def test_run_request_does_not_emit_environment_specific_control_fields(self) -> None:
        request = BaseRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
        )

        assert set(request.model_dump()) == {"responses_create_params"}

    def test_sanity(self) -> None:
        _resources_server().setup_webserver()

    def test_reverify_mode(self) -> None:
        assert asyncio.run(_resources_server().get_reverify_mode()) == ReverifyMode.UNKNOWN

    def test_stateless_server_answers_typed_and_legacy_session_calls(self) -> None:
        """A server with no per-rollout state serves an Environment Server's seed and close without overrides."""
        client = TestClient(_resources_server().setup_webserver())
        identity = {"episode_id": {"rollout_id": "rollout", "attempt": 0}}

        typed_seed = client.post(
            "/seed_session",
            json={
                "resources_session_id": "resources-session",
                "task_id": {"taskset": "tasks", "task_id": "task"},
                "task_data": {"question": "2+2"},
            }
            | identity,
        )
        # An Agent's /run seeds with its legacy row, which keeps the empty response.
        legacy_seed = client.post("/seed_session", json={"responses_create_params": {"input": "2+2"}})
        typed_close = client.post("/close_session", json={"resources_session_id": "resources-session"} | identity)

        assert typed_seed.status_code == 200
        assert typed_seed.json()["resources_session_id"] == "resources-session"
        assert (legacy_seed.status_code, legacy_seed.json()) == (200, {})
        assert (typed_close.status_code, typed_close.json()) == (200, {"resources_session_id": "resources-session"})

    def test_stateless_server_follows_the_session_contract(self) -> None:
        check_resources_session_contract(
            _resources_server().setup_webserver(),
            ResourcesSeedSessionRequest(
                resources_session_id="resources-session",
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="tasks", task_id="task"),
                task_data={},
            ),
            keeps_state=False,
        )


class TestVerifyResponseFailureReporting:
    """`mask_sample` / `failure_reason` on the contract, so every environment can use them."""

    def _params(self) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(input="hi")

    def _response(self) -> NeMoGymResponse:
        return NeMoGymResponse.model_construct(id="resp-1", output=[])

    def test_defaults_keep_existing_environments_unchanged(self) -> None:
        response = BaseVerifyResponse(responses_create_params=self._params(), response=self._response(), reward=0.0)
        assert response.mask_sample is False
        assert response.failure_reason is None

    def test_round_trip_preserves_the_flag_and_reason(self) -> None:
        response = BaseVerifyResponse(
            responses_create_params=self._params(),
            response=self._response(),
            reward=0.0,
            mask_sample=True,
            failure_reason="judge_unavailable",
        )
        dumped = response.model_dump()
        assert dumped["mask_sample"] is True
        assert dumped["failure_reason"] == "judge_unavailable"

    def test_zero_reward_is_not_implicitly_masked(self) -> None:
        """A policy that scores zero must stay a valid sample."""
        response = BaseVerifyResponse(responses_create_params=self._params(), response=self._response(), reward=0.0)
        assert response.mask_sample is False

    def test_a_degraded_but_legitimately_scored_rollout_is_not_masked(self) -> None:
        """The diagnosis is independent of the decision to exclude the sample."""
        response = BaseVerifyResponse(
            responses_create_params=self._params(),
            response=self._response(),
            reward=0.4,
            failure_reason="one retry was needed to reach the judge",
        )
        assert response.mask_sample is False
        assert response.failure_reason is not None


class TestFailureKindOnTheContract:
    """The groupable half of the diagnosis, drawn from the shared vocabulary."""

    def _params(self) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(input="hi")

    def _response(self) -> NeMoGymResponse:
        return NeMoGymResponse.model_construct(id="resp-1", output=[])

    def _verify(self, **kwargs) -> BaseVerifyResponse:
        return BaseVerifyResponse(
            responses_create_params=self._params(), response=self._response(), reward=0.0, **kwargs
        )

    def test_absent_by_default(self) -> None:
        assert self._verify().failure_kind is None

    def test_a_registered_kind_round_trips(self) -> None:
        assert self._verify(failure_kind=SESSION_LOST).model_dump()["failure_kind"] == SESSION_LOST

    def test_naming_a_kind_does_not_decide_usability(self) -> None:
        """A degraded but validly measured sample names a kind and stays unmasked."""
        degraded = self._verify(failure_kind=JUDGE_FAILED, failure_reason="one retry was needed")

        assert degraded.mask_sample is False
        assert degraded.failure_kind == JUDGE_FAILED

    def test_an_unregistered_kind_warns_but_the_response_survives(self, caplog) -> None:
        """Dropping the response would replace a wrong label with a lost failure."""
        import logging

        from nemo_gym.core import failure_kinds

        failure_kinds._WARNED_UNKNOWN.discard("invented_kind")
        with caplog.at_level(logging.WARNING):
            response = self._verify(failure_kind="invented_kind")

        assert response.failure_kind == "invented_kind"
        assert any("invented_kind" in record.getMessage() for record in caplog.records)

    def test_an_environment_can_namespace_its_own(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            response = self._verify(failure_kind="lexmount_browser:quota_exhausted")

        assert response.failure_kind == "lexmount_browser:quota_exhausted"
        assert caplog.records == []


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)


class TestResourcesSessionFence:
    def test_a_closed_session_is_refused_until_its_record_expires(self, monkeypatch) -> None:
        clock = [100.0]
        monkeypatch.setattr("nemo_gym.base_resources_server.monotonic", lambda: clock[0])
        fence = ResourcesSessionFence(window_seconds=10.0)
        fence.mark_closed("closed", EPISODE)

        assert fence.closed_episode("closed") == EPISODE
        with pytest.raises(HTTPException) as refused:
            fence.raise_if_closed("closed")
        assert refused.value.status_code == 409
        fence.raise_if_closed("open")

        clock[0] += 10.0
        assert fence.closed_episode("closed") is None
        fence.raise_if_closed("closed")

    def test_records_stay_bounded_by_the_window(self, monkeypatch) -> None:
        clock = [0.0]
        monkeypatch.setattr("nemo_gym.base_resources_server.monotonic", lambda: clock[0])
        fence = ResourcesSessionFence(window_seconds=5.0)
        for index in range(1000):
            fence.mark_closed(f"session-{index}", EPISODE)
            clock[0] += 1.0

        # Only the sessions closed within the last five seconds are still remembered.
        assert fence.closed_episode("session-995") is None
        assert fence.closed_episode("session-996") == EPISODE
        assert len(fence._closed) == 4

    def test_a_retried_close_keeps_the_first_episode_and_expiry(self, monkeypatch) -> None:
        clock = [0.0]
        monkeypatch.setattr("nemo_gym.base_resources_server.monotonic", lambda: clock[0])
        fence = ResourcesSessionFence(window_seconds=10.0)
        fence.mark_closed("session", EPISODE)
        clock[0] = 8.0
        fence.mark_closed("session", EpisodeId(rollout_id="another-rollout", attempt=0))

        assert fence.closed_episode("session") == EPISODE
        clock[0] = 10.0
        assert fence.closed_episode("session") is None

    def test_the_lock_serializes_one_session_and_is_dropped_when_unused(self) -> None:
        fence = ResourcesSessionFence(window_seconds=300.0)
        order: list[str] = []

        async def hold(name: str) -> None:
            async with fence.locked("session"):
                order.append(f"{name} in")
                await asyncio.sleep(0)
                order.append(f"{name} out")

        async def main() -> None:
            await asyncio.gather(hold("seed"), hold("close"))

        asyncio.run(main())

        assert order == ["seed in", "seed out", "close in", "close out"]
        assert fence._locks == {}

    def test_the_lock_does_not_block_another_session(self) -> None:
        fence = ResourcesSessionFence(window_seconds=300.0)

        async def main() -> None:
            release = asyncio.Event()
            entered: list[str] = []

            async def hold() -> None:
                async with fence.locked("slow-seed"):
                    entered.append("slow-seed")
                    await release.wait()

            holder = asyncio.create_task(hold())
            await asyncio.sleep(0)
            # Another session's close must not wait for the slow seed.
            async with asyncio.timeout(1):
                async with fence.locked("other"):
                    entered.append("other")
            release.set()
            await holder
            assert entered == ["slow-seed", "other"]

        asyncio.run(main())

        assert fence._locks == {}

    def test_clear_forgets_closed_sessions(self) -> None:
        fence = ResourcesSessionFence(window_seconds=300.0)
        fence.mark_closed("closed", EPISODE)

        fence.clear()

        assert fence.closed_episode("closed") is None
        fence.raise_if_closed("closed")

    def test_a_cancelled_waiter_releases_its_lock_entry(self) -> None:
        fence = ResourcesSessionFence(window_seconds=300.0)

        async def main() -> None:
            async with fence.locked("session"):
                waiter = asyncio.create_task(fence.locked("session").__aenter__())
                await asyncio.sleep(0)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter

        asyncio.run(main())

        assert fence._locks == {}


def _cookie_keyed_server(
    *,
    releases_by_resources_session_id: bool = True,
    conflict_is_final: bool = True,
    masks_lost_session: bool = True,
    fences_close: bool = True,
) -> SimpleResourcesServer:
    """A server that keys per-rollout state by the session cookie, as several servers do."""
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")

    class CookieKeyedResourcesServer(SimpleResourcesServer):
        _fence: ResourcesSessionFence = PrivateAttr(
            default_factory=lambda: ResourcesSessionFence(window_seconds=300.0)
        )
        _held: dict[str, str] = PrivateAttr(default_factory=dict)
        _episodes: dict[str, EpisodeId] = PrivateAttr(default_factory=dict)

        async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
            if request.session[SESSION_ID_KEY] in self._held:
                return BaseVerifyResponse(**body.model_dump(), reward=1.0)
            if masks_lost_session:
                return BaseVerifyResponse(**body.model_dump(), **lost_session_verification("no session at verify"))
            return BaseVerifyResponse(**body.model_dump(), reward=0.0)

        async def seed_session(
            self, request: Request, body: ResourcesSeedSessionRequest
        ) -> ResourcesSeedSessionResponse:
            async with self._fence.locked(body.resources_session_id):
                self._fence.raise_if_closed(body.resources_session_id)
                bound = self._episodes.setdefault(body.resources_session_id, body.episode_id)
                if bound != body.episode_id:
                    if conflict_is_final:
                        raise ResourcesSessionConflictError("resources_session_id is bound to another episode")
                    raise ValueError("resources_session_id is bound to another episode")
                self._held[request.session[SESSION_ID_KEY]] = body.resources_session_id
            return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)

        async def close_resources_session(
            self, request: Request, body: ResourcesCloseSessionRequest
        ) -> ResourcesCloseSessionResponse:
            async with self._fence.locked(body.resources_session_id):
                closed_episode = self._fence.closed_episode(body.resources_session_id)
                if fences_close and closed_episode not in (None, body.episode_id):
                    raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
                if releases_by_resources_session_id:
                    for key in [key for key, held in self._held.items() if held == body.resources_session_id]:
                        del self._held[key]
                else:
                    self._held.pop(request.session[SESSION_ID_KEY], None)
                self._episodes.pop(body.resources_session_id, None)
                self._fence.mark_closed(body.resources_session_id, body.episode_id)
            return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    return CookieKeyedResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


VERIFY_BODY = BaseVerifyRequest(
    responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
    response=NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    ),
).model_dump(mode="json")


class TestLostSeedReplyConformance:
    SEED = ResourcesSeedSessionRequest(
        resources_session_id="contract-session",
        episode_id=EPISODE,
        task_id=TaskId(taskset="tasks", task_id="task"),
        task_data={},
    )

    def test_a_close_without_the_cookie_must_release_the_session(self) -> None:
        server = _cookie_keyed_server(releases_by_resources_session_id=False)

        with pytest.raises(AssertionError, match="close left state for contract-session-lost-seed-reply"):
            check_resources_session_contract(
                server.setup_webserver(),
                self.SEED,
                keeps_state=True,
                holds_state=lambda resources_session_id: resources_session_id in server._held.values(),
            )

    def test_a_server_that_releases_by_resources_session_id_passes(self) -> None:
        server = _cookie_keyed_server(releases_by_resources_session_id=True)

        check_resources_session_contract(
            server.setup_webserver(),
            self.SEED,
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server._held.values(),
        )

    def test_a_conflicting_seed_must_be_refused_as_final(self) -> None:
        server = _cookie_keyed_server(conflict_is_final=False)

        with pytest.raises(AssertionError, match="to another episode must get 409, got 500"):
            check_resources_session_contract(server.setup_webserver(), self.SEED, keeps_state=True)

    def test_a_late_seed_must_be_refused_with_409_not_500(self) -> None:
        class ValueErrorFence(ResourcesSessionFence):
            def raise_if_closed(self, resources_session_id: str) -> None:
                if self.closed_episode(resources_session_id) is not None:
                    raise ValueError(f"Resources session is already closed: {resources_session_id}")

        server = _cookie_keyed_server()
        server._fence = ValueErrorFence(window_seconds=300.0)

        with pytest.raises(AssertionError, match="seed after close must get 409 for contract-session, got 500"):
            check_resources_session_contract(server.setup_webserver(), self.SEED, keeps_state=True)

    def test_a_close_naming_another_episode_after_close_must_be_refused(self) -> None:
        server = _cookie_keyed_server(fences_close=False)

        with pytest.raises(AssertionError, match="close of contract-session that names another episode must get 409"):
            check_resources_session_contract(server.setup_webserver(), self.SEED, keeps_state=True)

    def test_a_lost_session_must_verify_as_a_masked_sample(self) -> None:
        server = _cookie_keyed_server(masks_lost_session=False)

        with pytest.raises(AssertionError, match="must be masked as session_lost or session_expired"):
            check_resources_session_contract(
                server.setup_webserver(), self.SEED, keeps_state=True, verify_body=VERIFY_BODY
            )

    def test_a_server_that_follows_every_rule_passes_with_verify(self) -> None:
        server = _cookie_keyed_server()

        check_resources_session_contract(
            server.setup_webserver(),
            self.SEED,
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server._held.values(),
            verify_body=VERIFY_BODY,
        )

    def test_the_verify_check_needs_a_stateful_server(self) -> None:
        with pytest.raises(ValueError, match="requires keeps_state"):
            check_resources_session_contract(
                _resources_server().setup_webserver(), self.SEED, keeps_state=False, verify_body=VERIFY_BODY
            )


class TestRefusalsAndLostSessions:
    @pytest.mark.parametrize(
        ("error", "status"),
        [(ResourcesSeedRejectedError("bad task_data"), 422), (ResourcesSessionConflictError("bound"), 409)],
    )
    def test_refusals_are_final_4xx_and_still_value_errors(self, error: HTTPException, status: int) -> None:
        assert error.status_code == status
        assert isinstance(error, ValueError)

    def test_a_closed_session_is_refused_as_a_conflict(self) -> None:
        fence = ResourcesSessionFence(window_seconds=300.0)
        fence.mark_closed("closed", EPISODE)

        with pytest.raises(ResourcesSessionConflictError, match="already closed: closed"):
            fence.raise_if_closed("closed")

    def test_a_lost_session_tool_call_names_its_failure_kind(self) -> None:
        lost = ResourcesSessionLostError("browser is gone")
        expired = ResourcesSessionLostError("lifetime ran out", failure_kind=SESSION_EXPIRED)

        assert lost.status_code == 404
        assert lost.detail == {"failure_kind": SESSION_LOST, "message": "browser is gone"}
        assert expired.detail["failure_kind"] == SESSION_EXPIRED

    def test_a_lost_session_tool_call_warns_on_an_unregistered_failure_kind(self, caplog) -> None:
        import logging

        from nemo_gym.core import failure_kinds

        failure_kinds._WARNED_UNKNOWN.discard("invented_lost_kind")
        with caplog.at_level(logging.WARNING):
            lost = ResourcesSessionLostError("gone", failure_kind="invented_lost_kind")

        assert lost.detail["failure_kind"] == "invented_lost_kind"
        assert any("invented_lost_kind" in record.getMessage() for record in caplog.records)

    def test_a_lost_session_verifies_as_a_masked_sample(self, caplog) -> None:
        import logging

        class ExtraVerifyRequest(BaseVerifyRequest):
            model_config = ConfigDict(extra="allow")

        # A request model that allows extra fields can already carry verify fields, such as on a reverify.
        body = ExtraVerifyRequest.model_validate(VERIFY_BODY | {"reward": 1.0, "mask_sample": False})
        with caplog.at_level(logging.WARNING, logger="nemo_gym.base_resources_server"):
            verified = BaseVerifyResponse(**(body.model_dump() | lost_session_verification("sandbox gone")))

        assert verified.reward == 0.0
        assert verified.mask_sample is True
        assert verified.failure_kind == SESSION_LOST
        assert verified.failure_reason == "sandbox gone"
        # The warning makes a systematic loss, such as a cookie that never reaches /verify, visible in the logs.
        assert [record.getMessage() for record in caplog.records] == [
            "Masking the verify of a session this server no longer holds (session_lost): sandbox gone"
        ]

    def test_a_lost_session_sample_is_kept_for_coverage_and_dropped_from_quality(self) -> None:
        verified = BaseVerifyResponse(
            **(BaseVerifyRequest.model_validate(VERIFY_BODY).model_dump() | lost_session_verification("sandbox gone"))
        )
        # The Environment Server's result keeps the verify fields at the top level.
        lost = SingleAgentTurnResult.model_validate(verified.model_dump(mode="json")).model_dump(mode="json")
        scored = {"reward": 1.0, "mask_sample": False}
        rows = [{"_ng_task_index": 0, "_ng_rollout_index": index} for index in range(2)]
        results = [scored | rows[0], lost | rows[1]]

        measured_rows, measured, masked, coverage = select_measured(rows, results)

        assert [row["_ng_rollout_index"] for row in measured_rows] == [0]
        assert [result["reward"] for result in measured] == [1.0]
        assert [(result["failure_kind"], result["reward"]) for result in masked] == [(SESSION_LOST, 0.0)]
        assert coverage

    def test_refusals_reach_the_wire_as_final_4xx_through_the_server_middleware(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")

        class RefusingResourcesServer(SimpleResourcesServer):
            def setup_webserver(self) -> FastAPI:
                app = super().setup_webserver()
                app.post("/tool")(self.tool)
                return app

            async def tool(self) -> str:
                raise ResourcesSessionLostError("sandbox is gone")

            async def seed_session(self, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
                raise ResourcesSeedRejectedError("task_data is not a task")

            async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
                raise AssertionError("not called")

        server = RefusingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
        seed = TestLostSeedReplyConformance.SEED

        rejected = client.post("/seed_session", json=seed.model_dump(mode="json"))
        lost = client.post("/tool", json={})

        assert (rejected.status_code, rejected.json()) == (422, {"detail": "task_data is not a task"})
        assert (lost.status_code, lost.json()) == (
            404,
            {"detail": {"failure_kind": SESSION_LOST, "message": "sandbox is gone"}},
        )


class TestClosedResourcesSessionRecords:
    def test_a_record_lasts_the_window_after_the_first_close(self, monkeypatch) -> None:
        clock = [100.0]
        monkeypatch.setattr("nemo_gym.base_resources_server.monotonic", lambda: clock[0])
        records = ClosedResourcesSessionRecords[str](window_seconds=10)

        records.remember("a", "first close")
        clock[0] = 105.0
        records.remember("a", "retried close")
        records.remember("b", "close")
        assert records.get("a") == "first close"
        assert "a" in records and len(records) == 2

        # A retried close does not extend the record, so it ends 10 seconds after the first close.
        clock[0] = 110.0
        assert records.get("a") is None and "a" not in records
        assert records.get("b") == "close"
        clock[0] = 115.0
        assert len(records) == 0
        assert records.get("never closed") is None

    def test_a_record_never_outlives_the_routing_claim(self, monkeypatch) -> None:
        clock = [0.0]
        monkeypatch.setattr("nemo_gym.base_resources_server.monotonic", lambda: clock[0])
        monkeypatch.setattr("nemo_gym.session_routing.monotonic", lambda: clock[0])
        window = 300.0
        records = ClosedResourcesSessionRecords[bool](window_seconds=window)
        claims = ResourcesSessionLifetimes(window, starts=("/seed_session",), ends=("/close_session",))

        claims.served("id", {"path": "/seed_session", "headers": []}, 200)
        # The server records the close before the router sees its reply, and the caller retries it later.
        records.remember("id", True)
        clock[0] = 0.5
        claims.served("id", {"path": "/close_session", "headers": []}, 200)
        clock[0] = 120.0
        records.remember("id", True)
        clock[0] = 120.25
        claims.served("id", {"path": "/close_session", "headers": []}, 200)

        for step in range(0, 4400):
            clock[0] = step / 10
            if "id" in records:
                assert claims.holds("id"), f"the record outlived the claim at {clock[0]}"
        assert "id" not in records and not claims.holds("id")

    def test_a_window_that_is_not_positive_is_refused(self) -> None:
        for window in (0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="window_seconds must be positive"):
                ClosedResourcesSessionRecords[str](window_seconds=window)
