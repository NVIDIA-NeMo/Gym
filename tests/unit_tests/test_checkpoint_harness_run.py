# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SandboxedHarnessAgent`` with a fake adapter, through the real agent checkpoint participant.

The resources server is faked at the HTTP seam; the adapter's launches do what each test scripts: finish, run
until a checkpoint asks the run to park, fail, or block until cancelled.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig
from pydantic import BaseModel, JsonValue

from nemo_gym._checkpoint.agent import AgentSessionRecord
from nemo_gym._checkpoint.control import CheckpointRequest
from nemo_gym._checkpoint.harness import HarnessInterrupted, ResumeDecision
from nemo_gym._checkpoint.harness_run import (
    BookmarkError,
    LaunchContext,
    LentSandbox,
    SandboxedHarnessAgent,
    SeedBookmark,
    parse_bookmark,
)
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


CLOSE = CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 3600)


class Reply:
    def __init__(self, payload: dict, *, status: int = 200, headers: Optional[dict] = None) -> None:
        self._payload = payload
        self.status = status
        self.ok = status < 400
        self.headers = headers or {}
        self.cookies = {"ng_session": "s"}
        self.content = SimpleNamespace(read=self.read)

    async def json(self) -> dict:
        return self._payload

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    async def text(self) -> str:
        return json.dumps(self._payload)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status}")


class Resources:
    """The lender at the HTTP seam. ``restart_only`` sends the restart header; ``lends=False`` lends nothing."""

    def __init__(self, *, restart_only: bool = False, lends: bool = True, access_status: int = 200) -> None:
        self.calls: list[str] = []
        self.restart_only = restart_only
        self.lends = lends
        self.access_status = access_status
        self.current = "sb-1"

    async def post(self, *, server_name: str = "", url_path: str, json: Any = None, cookies: Any = None, **_: Any):
        self.calls.append(url_path)
        if url_path == "/seed_session":
            headers = {"x-ng-checkpoint-restart": "1"} if self.restart_only else {}
            return Reply({"sandbox_handle": "sb-1"} if self.lends else {}, headers=headers)
        if url_path == "/sandbox_access":
            if not self.lends or self.access_status == 404:
                return Reply({"detail": "no sandbox"}, status=404)
            if self.access_status != 200:
                return Reply({"detail": "broken"}, status=self.access_status)
            descriptor = {"sandbox_id": self.current}
            connection = {"kind": "direct", "provider_config_ref": "s", "descriptor": descriptor}
            return Reply({"connection": connection, "workdir": "/repo"})
        if url_path == "/close_session":
            return Reply({"closed": True})
        raise AssertionError(url_path)


class Box:
    def __init__(self) -> None:
        self.stopped = False
        self.disconnected = 0

    async def stop(self) -> None:
        self.stopped = True

    async def disconnect(self) -> None:
        self.disconnected += 1


class Outcome(BaseModel):
    text: str


class FakeAdapter:
    """Each launch does the next of ``script``: "finish", "until_park" (then interrupted), "fail", or "block"."""

    outcome_model = Outcome

    def __init__(self, *script: str, resume_ok: bool = True) -> None:
        self.script = list(script)
        self.resume_ok = resume_ok
        self.box = Box()
        self.connected: list[Optional[LentSandbox]] = []
        self.launched: list[tuple[dict[str, JsonValue], bool]] = []
        self.started = asyncio.Event()
        self.verified: list[Outcome] = []
        self.states = 0
        # How many times a pausing launch has waited for a park request.
        self.waits = 0

    def new_state(self) -> dict[str, JsonValue]:
        self.states += 1
        return {"folder": f"f{self.states}"}

    async def connect(self, lent: Optional[LentSandbox]) -> Any:
        self.connected.append(lent)
        return self.box

    async def resume_decision(self, sandbox: Any, state: dict[str, JsonValue]) -> ResumeDecision:
        return ResumeDecision(resume=self.resume_ok, reason="ok" if self.resume_ok else "no saved session")

    async def launch(self, launch: LaunchContext) -> Outcome:
        self.launched.append((dict(launch.state), launch.resume))
        self.started.set()
        action = self.script.pop(0)
        if action == "until_park":
            await launch.wait_park_requested()
            raise HarnessInterrupted("stopped")
        if action.startswith("pause"):
            # "pause" or "pause<n>": paused at each of the next n checkpoints, continuing in place, then done.
            for _ in range(int(action[len("pause") :] or 1)):
                self.waits += 1
                await launch.wait_park_requested()
                assert launch.park is not None, "a run that pauses gets a park callback"
                await launch.park()
            return Outcome(text=f"launch {len(self.launched)}")
        if action == "fail":
            raise RuntimeError("the harness crashed")
        if action == "block":
            await asyncio.Event().wait()
        return Outcome(text=f"launch {len(self.launched)}")

    async def verify(self, cookies: dict[str, str], outcome: Outcome) -> dict[str, Any]:
        self.verified.append(outcome)
        return {"reward": 1.0, "text": outcome.text}


class PreparingAdapter(FakeAdapter):
    """Verifies in two steps: prepare captures a patch (and the owner ends the session), then grading replays."""

    def __init__(self, *script: str) -> None:
        super().__init__(*script)
        self.prepared: list[Outcome] = []
        self.graded: list[dict[str, JsonValue]] = []
        self.grading = asyncio.Event()
        self.release_grading: Optional[asyncio.Event] = None

    def prepares_verification(self) -> bool:
        return True

    async def prepare_verification(self, cookies: dict[str, str], outcome: Outcome) -> dict[str, JsonValue]:
        self.prepared.append(outcome)
        return {"model_patch": f"patch of {outcome.text}"}

    async def verify_prepared(
        self, cookies: dict[str, str], outcome: Outcome, prepared: dict[str, JsonValue]
    ) -> dict[str, Any]:
        self.graded.append(prepared)
        self.grading.set()
        if self.release_grading is not None:
            await self.release_grading.wait()
        return {"reward": 1.0, "text": outcome.text, "graded": prepared["model_patch"]}


class PausingAdapter(FakeAdapter):
    """Pauses its harness at checkpoints instead of having it stopped."""

    def pauses_at_checkpoints(self) -> bool:
        return True


class DriverConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef


class Driver(SandboxedHarnessAgent):
    config: DriverConfig

    async def responses(self, body: Any = None) -> Any:
        raise NotImplementedError

    async def run(self, body: Any = None) -> Any:
        raise NotImplementedError


def make_driver(resources: Resources, *, checkpointing: bool = True) -> Driver:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "t"}} if checkpointing else {}
    )
    client.post = resources.post
    config = DriverConfig(
        host="",
        port=0,
        entrypoint="",
        name="driver",
        resources_server=ResourcesServerRef(type="resources_servers", name="r"),
    )
    driver = Driver(config=config, server_client=client)
    driver.setup_webserver()
    return driver


def request() -> Any:
    return SimpleNamespace(cookies={}, session={SESSION_ID_KEY: "agent-session"}, state=SimpleNamespace())


def body(attempt: int = 0) -> BaseRunRequest:
    key = EpisodeId(rollout_id="r", attempt=attempt).capture_key
    return BaseRunRequest.model_validate({"responses_create_params": {"input": []}, "_ng_rollout_id": key})


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not met"
        await asyncio.sleep(0.01)


async def install(driver: Driver, bookmark: dict[str, Any]) -> None:
    participant = driver.checkpoint_participant
    record = AgentSessionRecord(
        session_key="run:r", episode_id=EpisodeId(rollout_id="r"), session={}, boundary=None, episode=bookmark
    )
    await participant.install([record], [EpisodeId(rollout_id="r")])
    await participant.open_admission()


def harness_bookmark(**overrides: Any) -> dict[str, Any]:
    return {
        "v": 1,
        "next": "harness",
        "cookies": {"ng_session": "s"},
        "verify_mode": "wait",
        "sandbox": {"descriptor": {"sandbox_id": "sb-1"}, "workdir": None},
        "state": {"folder": "saved"},
        "resume": True,
        "launches": [{"resumed": False, "reason": None, "seconds": 2.0, "ended": "interrupted"}],
    } | overrides


# -- the stages ------------------------------------------------------------------------------------------------


async def test_without_checkpointing_it_seeds_launches_verifies_and_stops_the_sandbox() -> None:
    resources, adapter = Resources(), FakeAdapter("finish")
    driver = make_driver(resources, checkpointing=False)

    done = await driver.run_harness(request(), body(), adapter)

    assert done.result == {"reward": 1.0, "text": "launch 1"} and done.outcome == Outcome(text="launch 1")
    assert [launch.ended for launch in done.launches] == ["finished"]
    assert resources.calls == ["/seed_session"], "the adapter verifies; the driver only seeds"
    assert adapter.connected == [LentSandbox(descriptor={"sandbox_id": "sb-1"})], "no /sandbox_access needed"
    assert adapter.box.stopped, "nothing checkpoints the sandbox: the run stops it, as before"


async def test_an_interrupted_launch_parks_and_the_resume_continues_it_in_the_same_state() -> None:
    resources, adapter = Resources(), FakeAdapter("until_park", "finish")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant

    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.started.wait(), 5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    [record] = await participant.export(None, checkpoint_id="c")
    await participant.open_admission()
    done = await asyncio.wait_for(run, 5)

    bookmark = record.episode
    assert (bookmark["v"], bookmark["next"], bookmark["resume"], bookmark["state"]) == (
        1,
        "harness",
        True,
        {"folder": "f1"},
    )
    assert [launch["ended"] for launch in bookmark["launches"]] == ["interrupted"]
    assert adapter.launched == [({"folder": "f1"}, False), ({"folder": "f1"}, True)], "same state, continued"
    assert [(launch.resumed, launch.reason, launch.ended) for launch in done.launches] == [
        (False, None, "interrupted"),
        (True, "ok", "finished"),
    ]
    assert resources.calls == ["/seed_session", "/sandbox_access", "/sandbox_access"]
    assert not adapter.box.stopped, "the owner checkpoints the sandbox and ends it at verify"


async def test_a_run_queued_for_a_slot_is_exported_at_its_seed_bookmark_and_runs_after_resume() -> None:
    # Parked runs keep their slots, so a checkpoint cannot wait for the queue: the queued run must be exported.
    resources = Resources()
    first, queued = FakeAdapter("until_park", "finish"), FakeAdapter("finish")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant
    slot = asyncio.Semaphore(1)

    def run_body(rollout_id: str) -> BaseRunRequest:
        key = EpisodeId(rollout_id=rollout_id).capture_key
        return BaseRunRequest.model_validate({"responses_create_params": {"input": []}, "_ng_rollout_id": key})

    holder = asyncio.create_task(driver.run_harness(request(), run_body("r"), first, slot=slot))
    await asyncio.wait_for(first.started.wait(), 5)
    waiter = asyncio.create_task(driver.run_harness(request(), run_body("q"), queued, slot=slot))
    await wait_until(lambda: participant.has_session("run:q"))
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    report = participant.readiness()
    records = {record.session_key: record for record in await participant.export(None, checkpoint_id="c")}
    await participant.open_admission()
    await asyncio.wait_for(asyncio.gather(holder, waiter), 5)

    assert report.blockers == [] and report.restarts == [], "waiting for a slot neither blocks nor restarts"
    assert set(records) == {"run:r", "run:q"}, "the queued run is in the checkpoint"
    assert records["run:r"].episode["next"] == "harness"
    assert records["run:q"].episode == SeedBookmark().model_dump(mode="json")
    assert queued.launched == [({"folder": "f1"}, False)], "the queued run starts once it gets the slot"
    assert queued.verified == [Outcome(text="launch 1")]


async def test_a_slot_that_comes_free_while_admission_is_closed_parks_the_run_before_it_seeds() -> None:
    resources, adapter = Resources(), FakeAdapter("finish")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant
    slot = asyncio.Semaphore(0)

    run = asyncio.create_task(driver.run_harness(request(), body(), adapter, slot=slot))
    await wait_until(lambda: participant.has_session("run:r"))
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    slot.release()
    await asyncio.sleep(0.05)
    assert resources.calls == [], "no task work while a checkpoint is open"
    [record] = await participant.export(None, checkpoint_id="c")
    await participant.open_admission()
    await asyncio.wait_for(run, 5)

    assert record.episode == SeedBookmark().model_dump(mode="json")
    assert resources.calls[0] == "/seed_session"


@pytest.mark.parametrize("lender", ["restart_header", "no_sandbox"])
async def test_a_lender_that_cannot_capture_the_session_makes_the_run_a_restart_that_stops_its_sandbox(
    lender: str,
) -> None:
    resources = Resources(restart_only=lender == "restart_header", lends=lender != "no_sandbox")
    adapter = FakeAdapter("block")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant

    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.started.wait(), 5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    report = participant.readiness()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert report.restarts == ["r"] and report.blockers == [], "a restart never holds up a checkpoint"
    assert adapter.box.stopped and "/close_session" not in resources.calls


async def test_a_sandbox_access_404_keeps_the_hint() -> None:
    resources, adapter = Resources(access_status=404), FakeAdapter("finish")
    driver = make_driver(resources)

    await driver.run_harness(request(), body(), adapter)

    assert adapter.connected == [LentSandbox(descriptor={"sandbox_id": "sb-1"})]


async def test_a_failing_sandbox_access_fails_the_attempt_and_closes_the_session() -> None:
    resources, adapter = Resources(access_status=500), FakeAdapter("finish")
    driver = make_driver(resources)
    await install(driver, harness_bookmark())

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await driver.run_harness(request(), body(attempt=1), adapter)

    assert adapter.connected == [] and resources.calls == ["/sandbox_access", "/close_session"]


async def test_a_replacement_connects_to_the_sandbox_the_owner_names_not_the_hint() -> None:
    resources, adapter = Resources(), FakeAdapter("finish")
    resources.current = "sb-fork"
    driver = make_driver(resources)
    await install(driver, harness_bookmark())

    done = await driver.run_harness(request(), body(attempt=1), adapter)

    assert adapter.connected == [LentSandbox(descriptor={"sandbox_id": "sb-fork"}, workdir="/repo")]
    assert adapter.launched == [({"folder": "saved"}, True)]
    assert [launch.ended for launch in done.launches] == ["interrupted", "finished"], "the bookmark's launch log"


async def test_a_park_requested_before_the_launch_launches_nothing_until_the_resume() -> None:
    resources, adapter = Resources(), FakeAdapter("finish")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant
    connect = adapter.connect

    async def connect_during_a_checkpoint(lent: Optional[LentSandbox]) -> Any:
        if not adapter.connected:
            await participant.close_admission(CLOSE)
        return await connect(lent)

    adapter.connect = connect_during_a_checkpoint  # type: ignore[method-assign]
    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await wait_until(lambda: adapter.connected and participant.readiness().ready)
    [record] = await participant.export(None, checkpoint_id="c")
    assert adapter.launched == [], "nothing was launched only to be stopped"
    await participant.open_admission()
    done = await asyncio.wait_for(run, 5)

    assert record.episode["resume"] is False and record.episode["launches"] == []
    assert adapter.launched == [({"folder": "f1"}, False)] and len(done.launches) == 1


async def test_a_saved_session_that_cannot_be_continued_starts_over_in_fresh_state() -> None:
    resources, adapter = Resources(), FakeAdapter("finish", resume_ok=False)
    driver = make_driver(resources)
    await install(driver, harness_bookmark())

    done = await driver.run_harness(request(), body(attempt=1), adapter)

    assert adapter.launched == [({"folder": "f1"}, False)], "a new state, and the task from the start"
    assert done.launches[-1].reason == "no saved session" and done.launches[-1].resumed is False


async def test_a_continuation_at_verify_grades_again_without_launching() -> None:
    resources, adapter = Resources(), FakeAdapter()
    driver = make_driver(resources)
    await install(
        driver,
        {"v": 1, "next": "verify", "cookies": {}, "verify_mode": "wait", "outcome": {"text": "kept"}, "launches": []},
    )

    done = await driver.run_harness(request(), body(attempt=1), adapter)

    assert adapter.launched == [] and adapter.verified == [Outcome(text="kept")]
    assert done.result == {"reward": 1.0, "text": "kept"}


async def test_a_continuation_at_return_returns_the_stored_result() -> None:
    resources, adapter = Resources(), FakeAdapter()
    driver = make_driver(resources)
    await install(
        driver, {"v": 1, "next": "return", "result": {"reward": 0.5}, "outcome": {"text": "kept"}, "launches": []}
    )

    done = await driver.run_harness(request(), body(attempt=1), adapter)

    assert done.result == {"reward": 0.5} and done.outcome == Outcome(text="kept")
    assert adapter.verified == [] and resources.calls == []


async def test_prepared_verification_does_not_hold_up_a_checkpoint_for_grading() -> None:
    resources, adapter = Resources(), PreparingAdapter("finish")
    adapter.release_grading = asyncio.Event()
    driver = make_driver(resources)
    participant = driver.checkpoint_participant

    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.grading.wait(), 5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    report = participant.readiness()
    [record] = await participant.export(None, checkpoint_id="c")
    adapter.release_grading.set()
    await participant.open_admission()
    done = await asyncio.wait_for(run, 5)

    assert report.blockers == [], "grading is a replay step"
    assert record.episode["next"] == "verify" and record.episode["prepared"] == {"model_patch": "patch of launch 1"}
    assert adapter.verified == [] and adapter.prepared == [Outcome(text="launch 1")]
    assert done.result == {"reward": 1.0, "text": "launch 1", "graded": "patch of launch 1"}
    assert "/close_session" not in resources.calls, "the prepare ended the session"
    assert not adapter.box.stopped


async def test_a_continuation_at_a_prepared_verify_grades_the_saved_capture_without_preparing_again() -> None:
    resources, adapter = Resources(), PreparingAdapter()
    driver = make_driver(resources)
    await install(
        driver,
        {
            "v": 1,
            "next": "verify",
            "cookies": {},
            "verify_mode": "wait",
            "outcome": {"text": "kept"},
            "launches": [],
            "prepared": {"model_patch": "saved"},
        },
    )

    done = await driver.run_harness(request(), body(attempt=1), adapter)

    assert adapter.prepared == [] and adapter.graded == [{"model_patch": "saved"}]
    assert done.result["graded"] == "saved" and resources.calls == []


async def test_a_run_that_fails_after_preparing_does_not_close_the_ended_session() -> None:
    resources, adapter = Resources(), PreparingAdapter("finish")

    async def failing(*_: Any) -> dict[str, Any]:
        raise RuntimeError("grading failed")

    adapter.verify_prepared = failing  # type: ignore[method-assign]
    driver = make_driver(resources)

    with pytest.raises(RuntimeError, match="grading failed"):
        await driver.run_harness(request(), body(), adapter)

    assert adapter.prepared and "/close_session" not in resources.calls


def test_a_bookmark_of_another_version_or_shape_is_refused() -> None:
    assert parse_bookmark(None) == SeedBookmark()
    with pytest.raises(BookmarkError, match="version 2 is not supported"):
        parse_bookmark({"v": 2, "next": "harness"})
    with pytest.raises(BookmarkError, match="version None"):
        parse_bookmark({"next": "opencode", "sandbox_handle": "sb-1"})
    with pytest.raises(BookmarkError, match="malformed"):
        parse_bookmark({"v": 1, "next": "harness", "cookies": {}})


# -- leaving the sandbox ---------------------------------------------------------------------------------------


async def test_a_checkpointed_run_cancelled_while_parked_leaves_the_sandbox_to_the_checkpoint() -> None:
    resources, adapter = Resources(), FakeAdapter("until_park")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant

    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.started.wait(), 5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert not adapter.box.stopped and "/close_session" not in resources.calls
    assert adapter.box.disconnected == 1, "only the client is released"


@pytest.mark.parametrize("ending", ["fail", "cancel"])
async def test_a_checkpointed_run_that_leaves_without_a_verdict_closes_its_session_and_never_stops_the_sandbox(
    ending: str,
) -> None:
    resources, adapter = Resources(), FakeAdapter("fail" if ending == "fail" else "block")
    driver = make_driver(resources)

    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.started.wait(), 5)
    if ending == "cancel":
        run.cancel()
    with pytest.raises(asyncio.CancelledError if ending == "cancel" else RuntimeError):
        await run

    assert resources.calls[-1] == "/close_session"
    assert not adapter.box.stopped and adapter.box.disconnected == 1


async def test_a_checkpointing_off_run_that_fails_stops_its_sandbox() -> None:
    resources, adapter = Resources(), FakeAdapter("fail")
    driver = make_driver(resources, checkpointing=False)

    with pytest.raises(RuntimeError, match="crashed"):
        await driver.run_harness(request(), body(), adapter)

    assert adapter.box.stopped and "/close_session" not in resources.calls


# -- pausing the harness at checkpoints ------------------------------------------------------------------------


async def test_a_paused_launch_is_exported_at_its_bookmark_and_continues_in_place_after_each_checkpoint() -> None:
    resources, adapter = Resources(), PausingAdapter("pause2")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant

    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.started.wait(), 5)
    records = []
    for checkpoint in ("c1", "c2"):
        await participant.close_admission(CheckpointRequest(checkpoint_id=checkpoint, deadline_ts=time.time() + 3600))
        await wait_until(lambda: participant.readiness().ready)
        [record] = await participant.export(None, checkpoint_id=checkpoint)
        records.append(record.episode)
        assert not run.done(), "the launch is paused, not ended"
        await participant.open_admission()
        # The launch continues and waits for the next checkpoint.
        await wait_until(lambda: adapter.waits == len(records) + 1 or run.done())
    done = await asyncio.wait_for(run, 5)

    for pauses, bookmark in enumerate(records, start=1):
        assert (bookmark["next"], bookmark["resume"], bookmark["state"]) == ("harness", True, {"folder": "f1"})
        assert [(launch["ended"], launch["pauses"]) for launch in bookmark["launches"]] == [("interrupted", pauses)]
    assert adapter.launched == [({"folder": "f1"}, False)], "launched once: never relaunched"
    assert [(launch.ended, launch.pauses) for launch in done.launches] == [("finished", 2)]
    assert done.result == {"reward": 1.0, "text": "launch 1"}
    assert resources.calls == ["/seed_session", "/sandbox_access"]
    assert not adapter.box.stopped


async def test_a_checkpoint_that_closes_before_a_paused_launch_continues_finds_it_still_paused() -> None:
    resources, adapter = Resources(), PausingAdapter("pause")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant

    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.started.wait(), 5)
    records = []
    for checkpoint in ("c1", "c2"):
        # c2 closes at once after c1 opens, before the run wakes: it stays parked, and paused, for c2 too.
        await participant.close_admission(CheckpointRequest(checkpoint_id=checkpoint, deadline_ts=time.time() + 3600))
        await wait_until(lambda: participant.readiness().ready)
        [record] = await participant.export(None, checkpoint_id=checkpoint)
        records.append(record.episode)
        await participant.open_admission()
    done = await asyncio.wait_for(run, 5)

    assert records[0] == records[1], "both checkpoints export the same paused bookmark"
    assert [(launch.ended, launch.pauses) for launch in done.launches] == [("finished", 1)]
    assert adapter.launched == [({"folder": "f1"}, False)]


async def test_the_bookmark_of_a_paused_launch_relaunches_on_the_saved_session_after_a_crash() -> None:
    resources, adapter = Resources(), PausingAdapter("pause")
    driver = make_driver(resources)
    participant = driver.checkpoint_participant
    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.started.wait(), 5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    [record] = await participant.export(None, checkpoint_id="c")
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)

    replacement, relaunched = make_driver(Resources()), PausingAdapter("finish")
    await install(replacement, record.episode)
    done = await asyncio.wait_for(replacement.run_harness(request(), body(attempt=1), relaunched), 5)

    assert relaunched.launched == [({"folder": "f1"}, True)], "the crash relaunch continues the saved session"
    assert [(launch.resumed, launch.ended, launch.pauses) for launch in done.launches] == [
        (False, "interrupted", 1),
        (True, "finished", 0),
    ]


async def test_an_adapter_that_does_not_pause_gets_no_park_callback() -> None:
    resources, adapter = Resources(), FakeAdapter("until_park", "finish")
    driver = make_driver(resources)
    seen: list[object] = []
    launch = adapter.launch

    async def spy(context: LaunchContext) -> Outcome:
        seen.append(context.park)
        return await launch(context)

    adapter.launch = spy  # type: ignore[method-assign]
    participant = driver.checkpoint_participant
    run = asyncio.create_task(driver.run_harness(request(), body(), adapter))
    await asyncio.wait_for(adapter.started.wait(), 5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    await participant.export(None, checkpoint_id="c")
    await participant.open_admission()
    await asyncio.wait_for(run, 5)

    assert seen == [None, None], "it is stopped and relaunched, as before"
