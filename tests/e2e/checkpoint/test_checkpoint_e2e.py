# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-rollout checkpointing across real Gym processes, driven by Gym's coordination functions.

Every scenario checkpoints a rollout mid-episode, usually kills every Gym process, restarts them, restores,
and runs the replacement attempt. The fake backend records every model call, so each test can tell
exactly which work was continued and which was redone.

These tests start real server processes and take up to a minute each. They are skipped unless
``NEMO_GYM_CHECKPOINT_E2E=1``::

    NEMO_GYM_CHECKPOINT_E2E=1 pytest tests/e2e/checkpoint -x

The real-model smoke test also needs an OpenAI-compatible vLLM server with tool calling and a reasoning
parser, for example::

    vllm serve Qwen/Qwen3-0.6B --port 8000 --enable-auto-tool-choice --tool-call-parser hermes \\
        --reasoning-parser qwen3
    NEMO_GYM_CHECKPOINT_E2E=1 NEMO_GYM_CHECKPOINT_VLLM_URL=http://127.0.0.1:8000/v1 \\
        pytest tests/e2e/checkpoint -k real_model
"""

import asyncio
import collections
import json
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from checkpoint_deployment import (
    CAPTURE_CONTROL_TOKEN,
    COUNTER_SCRIPT,
    TOKEN,
    Deployment,
    counter_row,
    weather_episode,
    weather_row,
)

import nemo_gym.server_utils as server_utils
from nemo_gym._checkpoint import coordination
from nemo_gym.episode_types import EpisodeId


# The policy model server runs in one process, or as a coordinator plus uvicorn workers.
POLICY_WORKERS = pytest.mark.parametrize("policy_workers", [1, 2])

pytestmark = pytest.mark.skipif(
    os.environ.get("NEMO_GYM_CHECKPOINT_E2E") != "1",
    reason="starts real server processes; set NEMO_GYM_CHECKPOINT_E2E=1 to run",
)


@pytest.fixture(autouse=True)
async def gym_http_client() -> AsyncIterator[None]:
    """Coordination calls go through Gym's global aiohttp client, which is bound to this test's loop."""
    server_utils.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())
    yield
    await server_utils._GLOBAL_AIOHTTP_CLIENT.close()
    server_utils._GLOBAL_AIOHTTP_CLIENT = None
    # Coordination uses the reserved control pool, created on first use in this test's loop.
    if server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT is not None:
        await server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT.close()
        server_utils._GLOBAL_AIOHTTP_CONTROL_CLIENT = None


@pytest.fixture
def deploy(tmp_path: Path) -> Iterator[Callable[..., Deployment]]:
    deployments: list[Deployment] = []

    def start(topology: str, **options: Any) -> Deployment:
        deployment = Deployment(topology, tmp_path, **options)
        deployments.append(deployment)
        deployment.start_backend()
        deployment.start_gym()
        return deployment

    yield start
    for deployment in deployments:
        deployment.stop()


async def wait_until(predicate: Callable[[], bool], timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.1)


def deadline(seconds: float = 20) -> float:
    return time.time() + seconds


def environment_phase(deployment: Deployment) -> str:
    status = httpx.get(
        f"{deployment.url('environment')}/ng-control/v1/checkpoint/status",
        headers={"authorization": f"Bearer {TOKEN}"},
        timeout=10,
    )
    return status.json()["phase"]


async def checkpoint(
    deployment: Deployment,
    checkpoint_dir: Path,
    rollout_ids: list[str],
    *,
    attempt: int = 0,
    checkpoint_id: str = "c1",
) -> None:
    """Prepare and commit, continuing ``rollout_ids``; the caller resumes or crashes afterwards."""
    participants = await deployment.participants()
    prepared = await coordination.prepare(participants, checkpoint_id, deadline_ts=deadline())
    assert prepared.prepared, prepared.blockers()
    episode_ids = [EpisodeId(rollout_id=rollout_id, attempt=attempt) for rollout_id in rollout_ids]
    await coordination.commit(participants, checkpoint_id, str(checkpoint_dir), episode_ids, deadline_ts=deadline())


async def crash_and_restore(
    deployment: Deployment,
    checkpoint_dir: Path,
    rollout_ids: list[str],
    *,
    skip_kinds: tuple[str, ...] = (),
    attempt: int = 0,
    restore_id: str = "r1",
) -> None:
    deployment.crash_gym()
    if not deployment.external_inference:
        deployment.backend("/_ctl/release", {})
    deployment.slow_verify_flag.unlink(missing_ok=True)
    deployment.slow_verify_log.write_text("")
    deployment.start_gym()
    participants = await deployment.participants()
    restoring = coordination.Participants(
        client=participants.client,
        auth_token=participants.auth_token,
        members=tuple(member for member in participants.members if member.kind not in skip_kinds),
    )
    episode_ids = [EpisodeId(rollout_id=rollout_id, attempt=attempt) for rollout_id in rollout_ids]
    await coordination.restore(restoring, restore_id, str(checkpoint_dir), episode_ids, deadline_ts=deadline())
    await coordination.resume(participants, restore_id, deadline_ts=deadline())


def records_file(participant_dir: Path) -> Path:
    """A participant's records file, which is named after its digest."""
    [path] = participant_dir.glob("records-*.jsonl")
    return path


def environment_record(checkpoint_dir: Path) -> dict:
    [line] = records_file(checkpoint_dir / "gym/environment/environment").read_text().splitlines()
    return json.loads(line)


@POLICY_WORKERS
async def test_native_episode_continues_from_its_last_boundary_after_a_crash(
    deploy, tmp_path: Path, policy_workers: int
) -> None:
    deployment = deploy("native", policy_workers=policy_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_episode("crash-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["crash-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt", ["crash-1"])
        first.cancel()

        replacement = await http.post("/run", json=weather_episode("crash-1", attempt=1))

    calls = deployment.backend_calls()
    # Call 0 was delivered and its tool ran before the checkpoint; call 1 was never delivered.
    # The replacement continues after the tool result, so it makes one call and never repeats call 0.
    assert [call["n_messages"] for call in calls] == [2, 4, 4]
    assert replacement.json()["result"]["reward"] == 1.0


@POLICY_WORKERS
async def test_a_second_crash_before_the_replacement_starts_still_continues(
    deploy, tmp_path: Path, policy_workers: int
) -> None:
    deployment = deploy("native", policy_workers=policy_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_episode("twice-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        await checkpoint(deployment, tmp_path / "ckpt1", ["twice-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt1", ["twice-1"])
        first.cancel()
        # The next checkpoint lands before the framework re-dispatches attempt 1, and Gym crashes again.
        await checkpoint(deployment, tmp_path / "ckpt2", ["twice-1"], attempt=1, checkpoint_id="c2")
        await crash_and_restore(deployment, tmp_path / "ckpt2", ["twice-1"], attempt=1, restore_id="r2")

        replacement = await http.post("/run", json=weather_episode("twice-1", attempt=2))

    # Attempt 2 still continues after the tool result: it makes one call and never repeats call 0.
    assert [call["n_messages"] for call in deployment.backend_calls()] == [2, 4, 4]
    assert replacement.json()["result"]["reward"] == 1.0


@POLICY_WORKERS
async def test_a_checkpoint_restores_again_after_its_replacement_made_calls(
    deploy, tmp_path: Path, policy_workers: int
) -> None:
    deployment = deploy("counter", token_capture=True, policy_workers=policy_workers)
    deployment.backend("/_ctl/script", COUNTER_SCRIPT)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=counter_row("again-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["again-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt", ["again-1"])
        first.cancel()
        # Attempt 1 commits model calls to its capture ledger, then Gym crashes before the next checkpoint.
        await http.post("/run", json=counter_row("again-1", attempt=1))
        dead_calls = len(deployment.backend_calls())
        await crash_and_restore(deployment, tmp_path / "ckpt", ["again-1"], restore_id="r2")

        replacement = await http.post("/run", json=counter_row("again-1", attempt=1))

    manifest = httpx.get(
        f"{deployment.url('policy_model')}/training-token-capture/control/rollouts/again-1-a1/manifest",
        headers={"authorization": f"Bearer {CAPTURE_CONTROL_TOKEN}"},
    ).json()
    admissions = [call["admission"] for call in deployment.backend_calls()]
    chain = [record["model_call_id"] for record in manifest["records"]]
    dead = {admission["model_call_id"] for admission in admissions[2:dead_calls]}

    # The second run of attempt 1 continues from the checkpoint's boundary, as the first did,
    # and none of the dead run's calls are part of its lineage.
    assert replacement.json()["reward"] == 1.0
    assert admissions[dead_calls]["parent_call_id"] == admissions[0]["model_call_id"]
    assert chain[0] == admissions[0]["model_call_id"]
    assert dead and not dead & set(chain)
    assert not manifest.get("failures")


async def test_legacy_run_continues_from_its_last_boundary_after_a_crash(deploy, tmp_path: Path) -> None:
    deployment = deploy("legacy")
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_row("legacy-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["legacy-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt", ["legacy-1"])
        first.cancel()

        replacement = await http.post("/run", json=weather_row("legacy-1", attempt=1))

    assert [call["n_messages"] for call in deployment.backend_calls()] == [1, 3, 3]
    assert replacement.json()["reward"] == 1.0


@pytest.mark.parametrize("restore_resources", [True, False])
async def test_resources_state_is_restored_with_the_episode(deploy, tmp_path: Path, restore_resources: bool) -> None:
    deployment = deploy("counter")
    deployment.backend("/_ctl/script", COUNTER_SCRIPT)
    # The first increment runs before the checkpoint; the model call for the second is held.
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=counter_row("counter-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["counter-1"])
        skip = () if restore_resources else ("resources",)
        await crash_and_restore(deployment, tmp_path / "ckpt", ["counter-1"], skip_kinds=skip)
        first.cancel()

        replacement = await http.post("/run", json=counter_row("counter-1", attempt=1))

    # Only the restored counter (3 + 1) reaches the expected 6 after the second increment.
    # Without it, the continued episode increments a fresh counter: the negative control shows the restore matters.
    if restore_resources:
        assert replacement.json()["reward"] == 1.0
    else:
        assert replacement.status_code != 200 or replacement.json()["reward"] != 1.0


@POLICY_WORKERS
async def test_token_lineage_continues_across_a_crash(deploy, tmp_path: Path, policy_workers: int) -> None:
    deployment = deploy("counter", token_capture=True, policy_workers=policy_workers)
    deployment.backend("/_ctl/script", COUNTER_SCRIPT)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=counter_row("counter-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["counter-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt", ["counter-1"])
        first.cancel()
        replacement = await http.post("/run", json=counter_row("counter-1", attempt=1))

    manifest = httpx.get(
        f"{deployment.url('policy_model')}/training-token-capture/control/rollouts/counter-1-a1/manifest",
        headers={"authorization": f"Bearer {CAPTURE_CONTROL_TOKEN}"},
    ).json()
    admissions = [call["admission"] for call in deployment.backend_calls()]
    chain = [record["model_call_id"] for record in manifest["records"]]

    # Call 1 was never delivered,
    # so its replacement (call 2) continues from call 0 and call 1 is not part of the continued lineage.
    assert replacement.json()["reward"] == 1.0
    assert admissions[2]["rollout_id"] == "counter-1-a1"
    assert admissions[2]["parent_call_id"] == admissions[0]["model_call_id"]
    assert chain[0] == admissions[0]["model_call_id"]
    assert admissions[1]["model_call_id"] not in chain
    assert not manifest.get("failures")


@POLICY_WORKERS
async def test_an_in_flight_generation_is_cut_and_its_prefix_continued(
    deploy, tmp_path: Path, policy_workers: int
) -> None:
    deployment = deploy("native", token_capture=True, generation_cuts=True, policy_workers=policy_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_episode("cut-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["cut-1"])
        cut_call = deployment.backend_calls()[1]["admission"]["model_call_id"]
        cut_rounds = deployment.backend("/_ctl/cuts")
        await crash_and_restore(deployment, tmp_path / "ckpt", ["cut-1"])
        first.cancel()
        replacement = await http.post("/run", json=weather_episode("cut-1", attempt=1))

    after = [call for call in deployment.backend_calls() if call["admission"]["rollout_id"] == "cut-1-a1"]
    assert cut_rounds == [[cut_call]]
    assert [call.get("continued_from") for call in after] == [cut_call]
    assert replacement.json()["result"]["reward"] == 1.0


@POLICY_WORKERS
async def test_a_checkpoint_without_a_crash_parks_and_then_resumes_the_episode(
    deploy, tmp_path: Path, policy_workers: int
) -> None:
    deployment = deploy("native", policy_workers=policy_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 0})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        run = asyncio.create_task(http.post("/run", json=weather_episode("park-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 1)
        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline())
        # The generation finishes while Gym is prepared: the model server holds its response.
        deployment.backend("/_ctl/release", {})
        # The backend has answered: the policy model server now holds the response, and nothing else moves.
        await wait_until(lambda: deployment.backend_calls()[0].get("returned", False))
        await asyncio.sleep(1.0)
        parked = not run.done() and len(deployment.backend_calls()) == 1
        await coordination.resume(participants, "c1", deadline_ts=deadline())
        response = await run

    assert prepared.prepared
    assert parked
    assert response.json()["result"]["reward"] == 1.0
    assert len(deployment.backend_calls()) == 2


@pytest.mark.parametrize("mode", ["replay", "wait"])
async def test_verification_in_flight_is_replayed_or_waited_for(deploy, tmp_path: Path, mode: str) -> None:
    deployment = deploy("slow", checkpoint_verify=mode)
    deployment.slow_verify_flag.touch()
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_episode("verify-1")))
        await wait_until(lambda: deployment.verifications() == 1)
        participants = await deployment.participants()
        early = await coordination.prepare(participants, "c1", deadline_ts=deadline(1))
        if mode == "wait":
            deployment.slow_verify_flag.unlink()
        await checkpoint(deployment, tmp_path / "ckpt", ["verify-1"])
        record = environment_record(tmp_path / "ckpt")
        await crash_and_restore(deployment, tmp_path / "ckpt", ["verify-1"])
        first.cancel()
        replacement = await http.post("/run", json=weather_episode("verify-1", attempt=1))

    assert replacement.json()["result"]["reward"] == 1.0
    assert len(deployment.backend_calls()) == 2
    if mode == "replay":
        # A replay step does not hold the checkpoint; the restored episode runs verification again.
        assert early.prepared
        assert record["boundary"]["next"] == "verify"
        assert deployment.verifications() == 1
    else:
        # A wait step holds the checkpoint until it finishes; its result is restored, never re-run.
        assert early.blockers() == {"environment": ["verify-1"]}
        assert record["boundary"]["next"] == "return"
        assert deployment.verifications() == 0


@pytest.mark.parametrize("crash", [False, True])
async def test_an_episode_that_finishes_during_prepare_is_neither_lost_nor_redone(
    deploy, tmp_path: Path, crash: bool
) -> None:
    deployment = deploy("slow")
    deployment.slow_verify_flag.touch()
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        run = asyncio.create_task(http.post("/run", json=weather_episode("done-1")))
        await wait_until(lambda: deployment.verifications() == 1)
        participants = await deployment.participants()
        preparing = asyncio.create_task(coordination.prepare(participants, "c1", deadline_ts=deadline()))
        # The environment server has closed admission before verification finishes.
        await wait_until(lambda: environment_phase(deployment) == "preparing")
        deployment.slow_verify_flag.unlink()
        prepared = await preparing
        await asyncio.sleep(0.5)
        # No episode finishes while Gym is prepared: the result waits at the episode's last boundary.
        delivered_during_checkpoint = run.done()
        await coordination.commit(
            participants, "c1", str(tmp_path / "ckpt"), [EpisodeId(rollout_id="done-1")], deadline_ts=deadline()
        )
        record = environment_record(tmp_path / "ckpt")
        if crash:
            await crash_and_restore(deployment, tmp_path / "ckpt", ["done-1"])
            run.cancel()
            response = await http.post("/run", json=weather_episode("done-1", attempt=1))
        else:
            await coordination.resume(participants, "c1", deadline_ts=deadline())
            response = await run

    assert prepared.prepared
    assert not delivered_during_checkpoint
    assert record["boundary"]["next"] == "return"
    assert response.json()["result"]["reward"] == 1.0
    assert len(deployment.backend_calls()) == 2
    assert deployment.verifications() == (0 if crash else 1)


@pytest.mark.skipif(
    not os.environ.get("NEMO_GYM_CHECKPOINT_VLLM_URL"), reason="set NEMO_GYM_CHECKPOINT_VLLM_URL to a vLLM server"
)
async def test_real_model_rollouts_are_either_finished_or_continued_after_a_crash(deploy, tmp_path: Path) -> None:
    deployment = deploy(
        "native",
        inference_url=os.environ["NEMO_GYM_CHECKPOINT_VLLM_URL"],
        model_name=os.environ.get("NEMO_GYM_CHECKPOINT_VLLM_MODEL", "Qwen/Qwen3-0.6B"),
    )
    rollout_ids = [f"real-{index}" for index in range(64)]
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=300) as http:
        # Identical rollouts sent together also finish together on a fast server,
        # so the checkpoint would find none in flight.
        # Spaced out, and no longer sent once the first finishes, most are mid-episode.
        runs = {}
        for rollout_id in rollout_ids:
            if any(run.done() for run in runs.values()):
                break
            runs[rollout_id] = asyncio.create_task(http.post("/run", json=weather_episode(rollout_id)))
            await asyncio.sleep(0.02)
        rollout_ids = list(runs)
        # Checkpoint while the model is mid-generation for most rollouts.
        await wait_until(lambda: any(run.done() for run in runs.values()), timeout=120)
        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline(120))
        finished = {rollout_id for rollout_id, run in runs.items() if run.done()}
        await asyncio.sleep(2.0)
        finished_during_checkpoint = {rollout_id for rollout_id, run in runs.items() if run.done()} - finished
        # The controller continues every rollout that has not replied.
        continued = [rollout_id for rollout_id in rollout_ids if rollout_id not in finished]
        replies = await coordination.commit(
            participants,
            "c1",
            str(tmp_path / "ckpt"),
            [EpisodeId(rollout_id=rollout_id) for rollout_id in continued],
            deadline_ts=deadline(),
        )
        records = records_file(tmp_path / "ckpt/gym/environment/environment").read_text().splitlines()
        next_steps = sorted(json.loads(line)["boundary"]["next"] for line in records)
        agent_sessions = records_file(tmp_path / "ckpt/gym/agent/agent").read_text().splitlines()
        await crash_and_restore(deployment, tmp_path / "ckpt", continued)
        for rollout_id in continued:
            runs[rollout_id].cancel()
        replacements = await asyncio.gather(
            *(http.post("/run", json=weather_episode(rollout_id, attempt=1)) for rollout_id in continued)
        )

    first_results = [runs[rollout_id].result().json() for rollout_id in sorted(finished)]
    continued_results = [response.json() for response in replacements]
    print(f"finished before the checkpoint: {len(finished)}; continued: {len(continued)}; next steps: {next_steps}")
    print("rewards before:", [result["result"]["reward"] for result in first_results])
    print("rewards continued:", [result.get("result", {}).get("reward") for result in continued_results])

    assert prepared.prepared
    assert continued, "the checkpoint caught no rollout mid-episode"
    assert not finished_during_checkpoint
    # Every rollout that has not replied is exported by its environment server.
    assert replies["environment"]["episode_ids"] == sorted(continued)
    # An episode checkpointed inside its agent step continues the agent's own session, not the step.
    assert len(agent_sessions) == next_steps.count("invoke_agent")
    assert all(response.status_code == 200 and response.json().get("failure") is None for response in replacements)


@pytest.mark.skipif(
    not os.environ.get("NEMO_GYM_CHECKPOINT_VLLM_URL"), reason="set NEMO_GYM_CHECKPOINT_VLLM_URL to a vLLM server"
)
async def test_real_model_restarts_keep_running_through_a_checkpoint_and_start_over_after_a_crash(
    deploy, tmp_path: Path
) -> None:
    deployment = deploy(
        "mixed",
        inference_url=os.environ["NEMO_GYM_CHECKPOINT_VLLM_URL"],
        model_name=os.environ.get("NEMO_GYM_CHECKPOINT_VLLM_MODEL", "Qwen/Qwen3-0.6B"),
    )
    environment = httpx.AsyncClient(base_url=deployment.url("environment"), timeout=300)
    restarting = httpx.AsyncClient(base_url=deployment.url("restart_environment"), timeout=300)
    async with environment, restarting:
        # Alternate between the saved environment and the restart-only one, spaced out as in the test above.
        runs: dict[str, asyncio.Task] = {}
        for index in range(64):
            if any(run.done() for run in runs.values()):
                break
            client, rollout_id = (
                (environment, f"saved-{index}") if index % 2 == 0 else (restarting, f"restart-{index}")
            )
            runs[rollout_id] = asyncio.create_task(client.post("/run", json=weather_episode(rollout_id)))
            await asyncio.sleep(0.02)
        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline(120))
        in_flight = [EpisodeId(rollout_id=rollout_id) for rollout_id, run in runs.items() if not run.done()]
        continued = prepared.continued(in_flight)
        restarts = prepared.restarts()
        await coordination.commit(participants, "c1", str(tmp_path / "ckpt"), continued, deadline_ts=deadline())
        saved = [episode.rollout_id for episode in continued]
        await crash_and_restore(deployment, tmp_path / "ckpt", saved)
        for run in runs.values():
            run.cancel()
        # The restarts' first attempts are refused everywhere before they start over from their input.
        participants = await deployment.participants()
        await coordination.retire(
            participants,
            "retire-restarts",
            sorted(restarts, key=lambda episode: episode.capture_key),
            deadline_ts=deadline(),
        )
        replacements = await asyncio.gather(
            *(environment.post("/run", json=weather_episode(rollout_id, attempt=1)) for rollout_id in saved),
            *(
                restarting.post("/run", json=weather_episode(episode.rollout_id, attempt=1))
                for episode in sorted(restarts, key=lambda episode: episode.capture_key)
            ),
        )

    print(f"continued: {len(saved)}; restarted: {len(restarts)}")
    print("rewards:", [response.json().get("result", {}).get("reward") for response in replacements])
    assert prepared.prepared, prepared.blockers()
    assert saved and restarts, "the checkpoint caught no saved or no restart rollout mid-episode"
    # Every restart-only rollout still in flight is a restart, and nothing else is.
    # A restart keeps running through the checkpoint,
    # so one prepare reported may have finished by the time the test lists what is in flight.
    reported = {episode.rollout_id for episode in restarts}
    running = {episode.rollout_id for episode in in_flight if episode.rollout_id.startswith("restart-")}
    assert running <= reported <= {rollout_id for rollout_id in runs if rollout_id.startswith("restart-")}
    assert all(rollout_id.startswith("saved-") for rollout_id in saved)
    assert all(response.status_code == 200 and response.json().get("failure") is None for response in replacements)


@pytest.mark.skipif(
    not os.environ.get("NEMO_GYM_CHECKPOINT_SCALE"), reason="set NEMO_GYM_CHECKPOINT_SCALE to a rollout count"
)
@POLICY_WORKERS
async def test_checkpoint_at_scale(deploy, tmp_path: Path, policy_workers: int) -> None:
    """Checkpoint thousands of in-flight rollouts, crash, restore, and finish every one of them."""
    count = int(os.environ["NEMO_GYM_CHECKPOINT_SCALE"])
    # The training framework raises Gym's per-host connection limit the same way.
    deployment = deploy(
        "native",
        policy_workers=policy_workers,
        token_capture=True,
        generation_cuts=True,
        extra_config={"global_aiohttp_connector_limit_per_host": 16384},
    )
    # Every rollout's first model call returns; later calls are held, so the checkpoint finds them in flight.
    deployment.backend("/_ctl/hold", {"after_calls": count})
    rollout_ids = [f"scale-{index}" for index in range(count)]
    timings: dict[str, float] = {}
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=900, limits=limits) as http:
        started = time.monotonic()
        runs = [asyncio.create_task(http.post("/run", json=weather_episode(rollout_id))) for rollout_id in rollout_ids]
        await wait_until(lambda: len(deployment.backend_calls()) >= count + count // 2, timeout=600)
        timings["dispatch_to_half_held"] = time.monotonic() - started

        participants = await deployment.participants()
        clock = time.monotonic()
        # Prepare stage by stage, in coordination's order, to see where the time goes.
        for kind in coordination.PREPARE_ORDER:
            stage = coordination.Participants(
                client=participants.client,
                auth_token=participants.auth_token,
                members=tuple(member for member in participants.members if member.kind == kind),
            )
            stage_clock = time.monotonic()
            await coordination.prepare(stage, "scale", deadline_ts=deadline(300))
            timings[f"prepare_{kind}"] = time.monotonic() - stage_clock
        prepared = await coordination.prepare(participants, "scale", deadline_ts=deadline(300))
        timings["prepare"] = time.monotonic() - clock
        assert prepared.prepared, prepared.blockers()
        finished = {rollout_id for rollout_id, run in zip(rollout_ids, runs) if run.done()}
        clock = time.monotonic()
        replies = await coordination.commit(
            participants,
            "scale",
            str(tmp_path / "ckpt"),
            [EpisodeId(rollout_id=rollout_id) for rollout_id in rollout_ids],
            deadline_ts=deadline(300),
        )
        timings["commit"] = time.monotonic() - clock
        exported = replies["environment"]["episode_ids"]
        retained = replies["policy_model"]["staging_keys"]

        clock = time.monotonic()
        deployment.crash_gym()
        deployment.backend("/_ctl/release", {})
        deployment.start_gym()
        timings["restart"] = time.monotonic() - clock
        for run in runs:
            run.cancel()
        participants = await deployment.participants()
        clock = time.monotonic()
        await coordination.restore(
            participants,
            "scale-restore",
            str(tmp_path / "ckpt"),
            [EpisodeId.from_capture_key(key) for key in exported],
            deadline_ts=deadline(300),
        )
        timings["restore"] = time.monotonic() - clock
        clock = time.monotonic()
        await coordination.resume(participants, "scale-restore", deadline_ts=deadline(120))
        timings["resume"] = time.monotonic() - clock

        clock = time.monotonic()
        replacements = await asyncio.gather(
            *(http.post("/run", json=weather_episode(key, attempt=1)) for key in exported)
        )
        timings["replacements"] = time.monotonic() - clock
        cuts = deployment.backend("/_ctl/cuts")
        continued = sum(1 for call in deployment.backend_calls() if call.get("continued_from"))

    sizes = {
        kind: sum(f.stat().st_size for f in (tmp_path / "ckpt" / "gym" / kind).rglob("records-*.jsonl"))
        for kind in ("environment", "agent", "model", "resources")
    }
    results = [response.json() for response in replacements]
    rewards = [(result.get("result") or {}).get("reward") for result in results]
    failures = collections.Counter(
        str(result.get("failure") or result)[:160] for result, reward in zip(results, rewards) if reward != 1.0
    )
    print(
        f"\nscale={count} policy_workers={policy_workers} finished_before={len(finished)} exported={len(exported)}"
        f"\ntimings_s={ {name: round(value, 2) for name, value in timings.items()} }"
        f"\nrecords_bytes={sizes} report={prepared.replies['policy_model']['report']['counts']}"
        f"\ncut_rounds={len(cuts)} prefixes_cut={sum(len(round_) for round_ in cuts)} calls_continued={continued}"
        f"\nstaging_keys_retained={len(retained)}"
        f"\nreplacement_failures={sum(failures.values())} {failures.most_common(5)}"
    )

    # The controller's contract: every rollout either finished before the checkpoint or was exported.
    assert finished.isdisjoint(exported)
    assert finished | set(exported) == set(rollout_ids)
    assert rewards == [1.0] * len(exported)


async def test_a_retired_episode_stops_everywhere_and_leaves_nothing_behind(deploy, tmp_path: Path) -> None:
    deployment = deploy("native", token_capture=True)
    # The episode's first model call is answered; the second is held, so the episode is mid-flight.
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        run = asyncio.create_task(http.post("/run", json=weather_episode("gone-1")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        participants = await deployment.participants()

        await coordination.retire(participants, "retire", [EpisodeId(rollout_id="gone-1")], deadline_ts=deadline())

        calls_at_retire = len(deployment.backend_calls())
        await wait_until(run.done, timeout=10)
        # Nothing of the attempt is left to answer the held call, or to make another one.
        deployment.backend("/_ctl/release", {})
        await asyncio.sleep(1)
        headers = {"authorization": f"Bearer {TOKEN}"}
        servers = ("environment", "agent", "resources", "policy_model")

        def statuses() -> dict[str, dict]:
            return {
                name: httpx.get(f"{deployment.url(name)}/ng-control/v1/checkpoint/status", headers=headers).json()
                for name in servers
            }

        after_retire = statuses()
        # A duplicate of the retired attempt's /run, such as a client retry, is refused until the rollout is forgotten.
        late = await http.post("/run", json=weather_episode("gone-1"))
        await coordination.forget(participants, "forget", ["gone-1"], deadline_ts=deadline())
        after_forget = statuses()
        manifest = httpx.get(
            f"{deployment.url('policy_model')}/training-token-capture/control/rollouts/gone-1/manifest",
            headers={"authorization": f"Bearer {CAPTURE_CONTROL_TOKEN}"},
        )

    assert len(deployment.backend_calls()) == calls_at_retire
    assert after_retire["environment"]["report"]["counts"]["episodes"] == 0
    assert after_retire["agent"]["report"]["counts"]["sessions"] == 0
    assert after_retire["resources"]["report"]["counts"]["sessions"] == 0
    assert after_retire["policy_model"]["report"]["counts"]["inflight"] == 0
    # Only the rollout's refusal remains, one entry per server, until the controller forgets it.
    assert {name: status["retired_rollouts"] for name, status in after_retire.items()} == dict.fromkeys(servers, 1)
    assert "stale_attempt" in late.text
    assert {name: status["retired_rollouts"] for name, status in after_forget.items()} == dict.fromkeys(servers, 0)
    assert after_forget["resources"]["retired_sessions"] == 0
    # The checkpoint retire also retired the attempt's capture ledger.
    assert manifest.status_code == 410 and "is retired" in manifest.json()["detail"]


async def _mixed_checkpoint(deployment: Deployment, checkpoint_dir: Path) -> coordination.PrepareResult:
    """With one continued and one restart episode in flight, checkpoint everything that can be continued."""
    participants = await deployment.participants()
    prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline())
    # The restart episode's resources server cannot capture it, yet it does not hold the checkpoint up.
    assert prepared.prepared, prepared.blockers()
    in_flight = [EpisodeId(rollout_id="continued"), EpisodeId(rollout_id="restart")]
    await coordination.commit(
        participants, "c1", str(checkpoint_dir), prepared.continued(in_flight), deadline_ts=deadline()
    )
    return prepared


async def test_a_restart_episode_never_holds_up_a_checkpoint_and_starts_over_after_a_crash(
    deploy, tmp_path: Path
) -> None:
    deployment = deploy("mixed")
    # Every model call waits, so both episodes are in flight at the checkpoint.
    deployment.backend("/_ctl/hold", {"after_calls": 0})
    environment = httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120)
    restarting = httpx.AsyncClient(base_url=deployment.url("restart_environment"), timeout=120)
    async with environment, restarting:
        first = asyncio.create_task(environment.post("/run", json=weather_episode("continued")))
        second = asyncio.create_task(restarting.post("/run", json=weather_episode("restart")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        prepared = await _mixed_checkpoint(deployment, tmp_path / "ckpt")
        await crash_and_restore(deployment, tmp_path / "ckpt", ["continued"])
        first.cancel()
        second.cancel()
        # The controller refuses the restart's crashed attempt everywhere, then starts it over as its next attempt.
        participants = await deployment.participants()
        await coordination.retire(
            participants, "retire-restarts", [EpisodeId(rollout_id="restart")], deadline_ts=deadline()
        )
        continued = await environment.post("/run", json=weather_episode("continued", attempt=1))
        restarted = await restarting.post("/run", json=weather_episode("restart", attempt=1))

    assert prepared.restarts() == {EpisodeId(rollout_id="restart")}
    # Only the continued episode is in the checkpoint.
    assert environment_record(tmp_path / "ckpt")["episode_id"]["rollout_id"] == "continued"
    assert records_file(tmp_path / "ckpt/gym/environment/restart_environment").read_text() == ""
    assert continued.json()["result"]["reward"] == 1.0
    assert restarted.json()["result"]["reward"] == 1.0


async def test_a_restart_episode_keeps_running_through_a_checkpoint_without_a_crash(deploy, tmp_path: Path) -> None:
    deployment = deploy("mixed")
    deployment.backend("/_ctl/hold", {"after_calls": 0})
    environment = httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120)
    restarting = httpx.AsyncClient(base_url=deployment.url("restart_environment"), timeout=120)
    async with environment, restarting:
        first = asyncio.create_task(environment.post("/run", json=weather_episode("continued")))
        second = asyncio.create_task(restarting.post("/run", json=weather_episode("restart")))
        await wait_until(lambda: len(deployment.backend_calls()) == 2)
        await _mixed_checkpoint(deployment, tmp_path / "ckpt")
        await coordination.resume(await deployment.participants(), "c1", deadline_ts=deadline())
        deployment.backend("/_ctl/release", {})
        continued, restarted = await asyncio.gather(first, second)

    # Nothing crashed, so neither episode lost any work: each finishes as its first attempt.
    assert continued.json()["result"]["reward"] == 1.0
    assert restarted.json()["result"]["reward"] == 1.0
    assert restarted.json()["episode_id"]["attempt"] == 0
