# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint must not lose a finished cohort that only some of its members have recorded."""

import asyncio
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from omegaconf import DictConfig

import resources_servers.genrm_compare.app as genrm
from nemo_gym._checkpoint.control import CheckpointRequest, next_attempt
from nemo_gym._checkpoint.resources import ResourcesParticipant
from nemo_gym._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.verification import JudgeError
from resources_servers.genrm_compare.tests.support import request_with_session


def comparison_result(score_1=4.0, score_2=2.0, ranking=1.0):
    scores = (score_1, score_2, ranking)
    return (*scores, *scores, -1.0, -1.0, -1.0, 0.0, 0.0)


def member_0_wins(*_args, pair_idx, **_kwargs):
    """Member 0 scores higher in either order, so the members finish with different rewards and a mix-up shows."""
    return comparison_result(5.0, 1.0) if pair_idx[0] == 0 else comparison_result(1.0, 5.0)


def member(index, *, group="group", attempt=0, answer=None):
    return genrm.GenRMCompareVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "2+2?"}]),
        response=NeMoGymResponse(
            id=answer or f"answer-{index}",
            created_at=0.0,
            model="test",
            tools=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            output=[],
            object="response",
        ),
        group_id=group,
        group_attempt=attempt,
        rollout_index=index,
    )


EPISODES = [EpisodeId(rollout_id=f"group_g{index}") for index in range(2)]
SESSIONS = [f"session-{index}" for index in range(2)]
CHECKPOINT = CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 3600)


def participant_server(config, *, judge=None) -> tuple[genrm.GenRMCompareResourcesServer, ResourcesParticipant]:
    """A server behind the real resources participant, so sessions bind and restore as they do in Gym."""
    server = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    participant = ResourcesParticipant(server, "exported", verify_step_type="replay")
    server._checkpoint = participant
    server._run_single_comparison = judge or AsyncMock(return_value=comparison_result())
    return server, participant


def seeded(config, *, judge=None) -> tuple[genrm.GenRMCompareResourcesServer, ResourcesParticipant]:
    server, participant = participant_server(config, judge=judge)
    for session_id, episode_id in zip(SESSIONS, EPISODES):
        participant.seeded(session_id, episode_id)
    return server, participant


async def verify_as(server, session_id: str, body):
    return await server.verify(request_with_session(session_id), body)


async def until(condition, timeout: float = 2) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0)


def count_wait_open(participant) -> dict[str, int]:
    """Count the waits for admission to reopen, so a test sees a held outcome actually reach the hold."""
    counts = {"entered": 0}
    wait_open = participant.wait_open

    async def counting_wait_open() -> None:
        counts["entered"] += 1
        await wait_open()

    participant.wait_open = counting_wait_open
    return counts


async def finish_cohort(server) -> list:
    """Verify both members; member 0 then records its outcome and closes, member 1 has not recorded its own."""
    outcomes = await asyncio.gather(
        *(verify_as(server, SESSIONS[index], member(index)) for index in range(2)), return_exceptions=True
    )
    server._checkpoint.ended(SESSIONS[0])
    return outcomes


async def restore(config, records, *, judge=None):
    server, participant = participant_server(config, judge=judge)
    await participant.install(records, [record.episode_id for record in records])
    return server, participant


async def test_a_member_that_has_not_recorded_its_reward_gets_it_back_after_a_restore(config):
    server, participant = seeded(config, judge=AsyncMock(side_effect=member_0_wins))
    outcomes = await finish_cohort(server)
    assert outcomes[0].reward != outcomes[1].reward
    records = await participant.export(None)
    assert [record.session_id for record in records] == [SESSIONS[1]]
    await server.aclose()

    restored, _ = await restore(config, records)
    replayed = await asyncio.wait_for(verify_as(restored, SESSIONS[1], member(1)), timeout=0.5)

    assert replayed.reward == outcomes[1].reward
    # Answered from the restored cohort: no sibling had to verify again, and the judge did not run.
    restored._run_single_comparison.assert_not_awaited()
    await restored.aclose()


@pytest.mark.parametrize("recorded_before_checkpoint", [[0], []], ids=["member_1_open", "both_open"])
async def test_a_restored_cohort_survives_another_checkpoint_before_its_members_replay(
    config, recorded_before_checkpoint
):
    """Each member gets back its own outcome, not a sibling's, after two checkpoint-crash-restore cycles."""
    server, participant = seeded(config, judge=AsyncMock(side_effect=member_0_wins))
    outcomes = await asyncio.gather(*(verify_as(server, SESSIONS[index], member(index)) for index in range(2)))
    assert outcomes[0].reward != outcomes[1].reward
    for index in recorded_before_checkpoint:
        participant.ended(SESSIONS[index])
    checkpoint_a = await participant.export(None)
    await server.aclose()

    restored, restored_participant = await restore(config, checkpoint_a)
    # Checkpoint B lands before the open members replay their verification.
    checkpoint_b = await restored_participant.export(None)
    assert [record.state for record in checkpoint_b] == [record.state for record in checkpoint_a]
    await restored.aclose()

    again, _ = await restore(config, checkpoint_b)
    open_members = [index for index in range(2) if index not in recorded_before_checkpoint]
    for index in open_members:
        replayed = await asyncio.wait_for(verify_as(again, SESSIONS[index], member(index)), timeout=0.5)
        assert replayed.reward == outcomes[index].reward
    again._run_single_comparison.assert_not_awaited()
    await again.aclose()


async def test_a_member_that_has_not_recorded_its_judge_failure_gets_it_back_after_a_restore(config):
    """A judge failure reaches callers as a masked row they record, so it must survive a crash like a reward."""
    server, participant = seeded(config, judge=AsyncMock(side_effect=JudgeError("judge is down")))
    outcomes = await finish_cohort(server)
    assert all(isinstance(outcome, JudgeError) for outcome in outcomes)
    records = await participant.export(None)
    assert records[0].state["judge_failure"] == str(outcomes[1])
    await server.aclose()

    restored, _ = await restore(config, records)
    with pytest.raises(JudgeError, match="judge is down"):
        await asyncio.wait_for(verify_as(restored, SESSIONS[1], member(1)), timeout=0.5)
    restored._run_single_comparison.assert_not_awaited()
    await restored.aclose()


@pytest.mark.parametrize("judge_fails", [False, True], ids=["reward", "judge_failure"])
async def test_outcomes_are_held_while_a_checkpoint_is_open(config, judge_fails):
    """An outcome published after this server exported would let one member record it while its siblings could not."""
    judge = AsyncMock(side_effect=JudgeError("judge is down")) if judge_fails else None
    server, participant = seeded(config, judge=judge)
    counts = count_wait_open(participant)
    await participant.close_admission(CHECKPOINT)
    members = [asyncio.create_task(verify_as(server, SESSIONS[index], member(index))) for index in range(2)]
    await until(lambda: counts["entered"] == 1)

    # The judge ran, but no member has its outcome, and the export holds none.
    assert not any(task.done() for task in members)
    assert [record.state for record in await participant.export(None)] == [{}, {}]

    await participant.open_admission()
    outcomes = await asyncio.wait_for(asyncio.gather(*members, return_exceptions=True), timeout=0.5)
    if judge_fails:
        assert all(isinstance(outcome, JudgeError) for outcome in outcomes)
    else:
        assert all(outcome.reward == 3.0 for outcome in outcomes)
    await server.aclose()


async def test_a_session_whose_cohort_has_no_outcome_yet_exports_nothing(config):
    server, participant = seeded(config)
    waiting = asyncio.create_task(verify_as(server, SESSIONS[0], member(0)))
    await until(lambda: server._session_members)

    assert [record.state for record in await participant.export(None)] == [{}, {}]
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    await server.aclose()


@pytest.mark.parametrize("how", ["failed_restore", "left_out_of_commit"])
async def test_a_restored_cohort_nothing_continues_from_is_dropped(config, how):
    """The rollout starts over from its input under the same group attempt; a leftover cohort would refuse it."""
    server, participant = seeded(config)
    await finish_cohort(server)
    records = await participant.export(None)
    await server.aclose()

    restored, restored_participant = await restore(config, records)
    replacement = next_attempt(EPISODES[1])
    if how == "failed_restore":
        await restored_participant.mark_retired([replacement])
        await restored_participant.retire(replacement)
    else:
        await restored_participant.delete_restored(replacement)
    assert restored._verify_cohorts == {}
    assert restored._latest_group_attempts == {}
    assert restored._session_members == {}

    # Both members regenerate and re-form the group, as they would on a process with no restored cohort.
    rewards = await asyncio.wait_for(
        asyncio.gather(
            *(verify_as(restored, f"fresh-{index}", member(index, answer=f"new-{index}")) for index in range(2))
        ),
        timeout=0.5,
    )
    assert all(reward.reward == 3.0 for reward in rewards)
    await restored.aclose()


async def test_a_restored_cohort_stays_once_a_member_has_replayed(config):
    sessions = ["s0", "s1", "s2"]
    config3 = config.model_copy(update={"num_rollouts_per_prompt": 3})
    server, participant = participant_server(config3, judge=AsyncMock(return_value=comparison_result()))
    episodes = [EpisodeId(rollout_id=f"group_g{index}") for index in range(3)]
    for session_id, episode_id in zip(sessions, episodes):
        participant.seeded(session_id, episode_id)
    await asyncio.gather(*(verify_as(server, sessions[index], member(index)) for index in range(3)))
    participant.ended("s0")
    records = await participant.export(None)
    await server.aclose()

    restored, restored_participant = await restore(config3, records)
    # Member 1 replays and continues; member 2's restored session is then left out of a commit.
    await verify_as(restored, "s1", member(1))
    await restored_participant.delete_restored(next_attempt(episodes[2]))
    [cohort] = restored._verify_cohorts.values()
    assert set(cohort.members) == {1}

    # Retiring the member that replayed keeps it: it continued from the cohort.
    await restored_participant.mark_retired([next_attempt(episodes[1])])
    await restored_participant.retire(next_attempt(episodes[1]))
    assert list(restored._verify_cohorts.values()) == [cohort]
    assert set(cohort.members) == {1}
    await restored.aclose()


async def test_a_restore_that_names_a_cohort_inconsistently_installs_nothing(config):
    server, participant = seeded(config)
    await asyncio.gather(*(verify_as(server, SESSIONS[index], member(index)) for index in range(2)))
    records = await participant.export(None)
    await server.aclose()
    records[1] = records[1].model_copy(update={"state": {**records[1].state, "group_attempt": 1}})

    restored, restored_participant = participant_server(config)
    with pytest.raises(ValueError, match="inconsistently"):
        await restored_participant.install(records, [record.episode_id for record in records])
    assert restored._verify_cohorts == {}
    assert restored._latest_group_attempts == {}
    assert restored._session_members == {}
    await restored.aclose()


async def test_a_restore_whose_sessions_disagree_on_a_member_installs_nothing(config):
    server, participant = seeded(config)
    await asyncio.gather(*(verify_as(server, SESSIONS[index], member(index)) for index in range(2)))
    records = await participant.export(None)
    await server.aclose()
    # A second session claims member 1 with another response.
    impostor = records[1].model_copy(
        update={"session_id": "impostor", "state": {**records[1].state, "response_digest": "0" * 64}}
    )

    restored, restored_participant = participant_server(config)
    with pytest.raises(ValueError, match="disagree"):
        await restored_participant.install([*records, impostor], [record.episode_id for record in records])
    assert restored._verify_cohorts == {}
    assert restored._session_members == {}
    await restored.aclose()


async def test_a_restore_refuses_digests_saved_under_another_scheme(config):
    """A digest from another serialization would refuse every exact replay; refuse the saved state instead."""
    server, participant = seeded(config)
    await finish_cohort(server)
    records = await participant.export(None)
    assert records[0].state["response_digest_scheme"] == genrm.RESPONSE_DIGEST_SCHEME
    await server.aclose()
    records[0] = records[0].model_copy(
        update={"state": {**records[0].state, "response_digest_scheme": "orjson-sha256"}}
    )

    restored, restored_participant = participant_server(config)
    with pytest.raises(ValueError, match="response_digest_scheme"):
        await restored_participant.install(records, [record.episode_id for record in records])
    assert restored._verify_cohorts == {}
    await restored.aclose()


async def test_a_restored_cohort_refuses_an_older_attempt_of_its_group(config):
    server, participant = seeded(config)
    await asyncio.gather(*(verify_as(server, SESSIONS[index], member(index, attempt=1)) for index in range(2)))
    participant.ended(SESSIONS[0])
    records = await participant.export(None)
    await server.aclose()

    restored, _ = await restore(config, records)
    with pytest.raises(genrm.HTTPException, match="superseded"):
        await verify_as(restored, "other", member(0, attempt=0))
    await restored.aclose()


async def test_member_records_end_with_their_sessions_and_the_server(config):
    server, participant = seeded(config)
    await asyncio.gather(*(verify_as(server, SESSIONS[index], member(index)) for index in range(2)))
    assert set(server._session_members) == set(SESSIONS)

    await participant.mark_retired([EPISODES[0]])
    await participant.retire(EPISODES[0])
    assert set(server._session_members) == {SESSIONS[1]}

    await server.aclose()
    assert server._session_members == {}


async def test_a_cohort_without_a_group_id_records_no_members(config):
    server, participant = seeded(config)
    await asyncio.gather(*(verify_as(server, SESSIONS[index], member(index, group=None)) for index in range(2)))
    assert server._session_members == {}
    await server.aclose()


async def test_member_records_end_with_their_pruned_cohort(config):
    config = config.model_copy(update={"cohort_result_ttl_s": 0.01})
    server, participant = seeded(config)
    await asyncio.gather(*(verify_as(server, SESSIONS[index], member(index)) for index in range(2)))
    assert set(server._session_members) == set(SESSIONS)
    # Both members recorded their outcomes and closed their sessions, so nothing needs the cohort any more.
    for session_id in SESSIONS:
        participant.ended(session_id)

    # The next cohort's verification prunes the expired one, and its member records go with it.
    await asyncio.sleep(0.02)
    pending = asyncio.create_task(verify_as(server, "later", member(0, group="later")))
    await until(lambda: "later" in server._session_members)
    assert set(server._session_members) == {"later"}
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    await server.aclose()


@pytest.mark.parametrize("judge_fails", [False, True], ids=["reward", "judge_failure"])
@pytest.mark.parametrize("pruned_by", ["count", "age"])
async def test_pruning_keeps_an_outcome_an_open_session_still_needs(config, pruned_by, judge_fails):
    """A member that has not recorded its outcome replays after a restore; an evicted cohort would leave it waiting
    for a sibling that already finished, until the collection timeout."""
    limit = {"max_terminal_cohorts": 1} if pruned_by == "count" else {"cohort_result_ttl_s": 0.01}
    config = config.model_copy(update=limit)
    judge = AsyncMock(side_effect=JudgeError("judge is down") if judge_fails else member_0_wins)
    groups = ["a", "b"]
    server, participant = participant_server(config, judge=judge)
    for group in groups:
        for index in range(2):
            participant.seeded(f"{group}{index}", EpisodeId(rollout_id=f"{group}_g{index}"))
    outcomes = {}
    for group in groups:
        results = await asyncio.gather(
            *(verify_as(server, f"{group}{index}", member(index, group=group)) for index in range(2)),
            return_exceptions=True,
        )
        outcomes[group] = results[1]
        # Member 0 recorded its outcome; member 1 has not.
        participant.ended(f"{group}0")
        if pruned_by == "age":
            await asyncio.sleep(0.02)
    records = await participant.export(None)
    assert sorted(record.session_id for record in records) == ["a1", "b1"]
    assert all(record.state for record in records)
    await server.aclose()

    restored, restored_participant = await restore(config, records)
    if pruned_by == "age":
        await asyncio.sleep(0.02)
    for group in groups:
        # Each replay prunes first; the other group's cohort, and its own, are past the limit.
        replay = asyncio.wait_for(verify_as(restored, f"{group}1", member(1, group=group)), timeout=0.5)
        if judge_fails:
            with pytest.raises(JudgeError, match="judge is down"):
                await replay
        else:
            assert (await replay).reward == outcomes[group].reward
    restored._run_single_comparison.assert_not_awaited()
    # A kept cohort keeps its group's attempt watermark, so the group still refuses older attempts.
    restored._prune_terminal_cohorts()
    assert set(restored._latest_group_attempts) == set(groups)

    # Once the members close their sessions, the cohorts are ordinary tombstones again.
    for group in groups:
        restored_participant.ended(f"{group}1")
    if pruned_by == "age":
        await asyncio.sleep(0.02)
    restored._prune_terminal_cohorts()
    assert len(restored._verify_cohorts) == (1 if pruned_by == "count" else 0)
    await restored.aclose()


@pytest.mark.parametrize("judge_fails", [False, True], ids=["reward", "judge_failure"])
async def test_an_outcome_stays_held_when_a_checkpoint_closes_admission_again_before_publication(config, judge_fails):
    """Admission can reopen and close again while the outcome waits for the cohort lock; the check under it holds."""
    judge = AsyncMock(side_effect=JudgeError("judge is down")) if judge_fails else None
    server, participant = seeded(config, judge=judge)
    counts = count_wait_open(participant)
    await participant.close_admission(CHECKPOINT)
    members = [asyncio.create_task(verify_as(server, SESSIONS[index], member(index))) for index in range(2)]
    await until(lambda: counts["entered"] == 1)
    [cohort] = server._verify_cohorts.values()

    async with cohort.lock:
        await participant.open_admission()
        # The held outcome wakes and waits for the lock; a new checkpoint closes admission meanwhile.
        for _ in range(5):
            await asyncio.sleep(0)
        await participant.close_admission(CHECKPOINT)
    await until(lambda: counts["entered"] == 2)
    assert not any(task.done() for task in members)

    await participant.open_admission()
    outcomes = await asyncio.wait_for(asyncio.gather(*members, return_exceptions=True), timeout=0.5)
    assert all(isinstance(outcome, JudgeError) for outcome in outcomes) == judge_fails
    await server.aclose()


AUTH = {"authorization": "Bearer t"}


def http_server(config, *, judge_fails: bool):
    """A server behind its real app, with checkpointing on, so requests pass the checkpoint middleware."""
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    server = genrm.GenRMCompareResourcesServer(config=config, server_client=server_client)
    if judge_fails:
        server._run_single_comparison = AsyncMock(side_effect=JudgeError("judge is down"))
    else:
        server._run_single_comparison = AsyncMock(side_effect=member_0_wins)
    return server, httpx.ASGITransport(app=server.setup_webserver())


def control(checkpoint_id: str, **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 10, **extra}


def body_json(index: int) -> dict:
    return member(index).model_dump(mode="json", by_alias=True)


def outcome(response: httpx.Response) -> object:
    assert response.status_code == 200, response.text
    row = response.json()
    return ("masked", row["failure_kind"]) if row.get("mask_sample") else row["reward"]


async def commit(client: httpx.AsyncClient, checkpoint_id: str) -> str:
    directory = tempfile.mkdtemp()
    for route, body in (
        ("prepare", control(checkpoint_id)),
        ("commit", control(checkpoint_id, checkpoint_dir=directory)),
    ):
        (await client.post(f"/ng-control/v1/checkpoint/{route}", json=body, headers=AUTH)).raise_for_status()
    return directory


async def restore_and_resume(client: httpx.AsyncClient, directory: str, scope: list[dict]) -> None:
    restored = await client.post(
        "/ng-control/v1/checkpoint/restore",
        json=control("restore", checkpoint_dir=directory, episode_ids=scope),
        headers=AUTH,
    )
    restored.raise_for_status()
    (await client.post("/ng-control/v1/checkpoint/resume", json=control("restore"), headers=AUTH)).raise_for_status()


@pytest.mark.parametrize("checkpoints", [1, 2])
@pytest.mark.parametrize("judge_fails", [False, True], ids=["reward", "judge_failure"])
async def test_a_replay_through_the_checkpoint_control_plane_gets_the_recorded_outcome(
    config, judge_fails, checkpoints
):
    """Member 0 recorded its outcome; member 1, whose caller closes its session later, replays after the crash."""
    server, transport = http_server(config, judge_fails=judge_fails)
    clients = [httpx.AsyncClient(transport=transport, base_url="http://genrm") for _ in range(3)]
    async with clients[0], clients[1], clients[2]:
        for index in range(2):
            seed = await clients[index].post(
                f"/ng-rollout/g{index}/seed_session", json={"responses_create_params": {"input": "hi"}}
            )
            seed.raise_for_status()
        replies = await asyncio.gather(
            clients[0].post("/ng-rollout/g0/verify", json=body_json(0)),
            clients[1].post(
                "/ng-rollout/g1/verify", json=body_json(1), headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
            ),
        )
        recorded = outcome(replies[1])
        assert recorded != outcome(replies[0]) or judge_fails
        directory = await commit(clients[2], "save-a")
        cookies = dict(clients[1].cookies)
    await server.aclose()

    attempt = 0
    for _ in range(checkpoints):
        server, transport = http_server(config, judge_fails=judge_fails)
        control_client = httpx.AsyncClient(transport=transport, base_url="http://genrm")
        async with control_client:
            await restore_and_resume(control_client, directory, [{"rollout_id": "g1", "attempt": attempt}])
            attempt += 1
            if attempt < checkpoints:
                # Another checkpoint and crash before member 1 replays.
                directory = await commit(control_client, f"save-{attempt}")
                await server.aclose()

    replay = httpx.AsyncClient(transport=transport, base_url="http://genrm", cookies=cookies)
    async with replay:
        # Without the saved outcome, this would wait for member 0 until the collection timeout.
        response = await asyncio.wait_for(
            replay.post(f"/ng-rollout/g1-a{attempt}/verify", json=body_json(1)), timeout=0.5
        )
    assert outcome(response) == recorded
    server._run_single_comparison.assert_not_awaited()
    await server.aclose()
