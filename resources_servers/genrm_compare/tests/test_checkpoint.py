# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint must not lose a finished cohort that only some of its members have recorded."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.genrm_compare.app as genrm
from nemo_gym._checkpoint.control import CheckpointRequest, next_attempt
from nemo_gym._checkpoint.resources import ResourcesParticipant
from nemo_gym.episode_types import EpisodeId
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.genrm_compare.tests.support import request_with_session


def comparison_result(score_1=4.0, score_2=2.0, ranking=1.0):
    scores = (score_1, score_2, ranking)
    return (*scores, *scores, -1.0, -1.0, -1.0, 0.0, 0.0)


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
    participant = ResourcesParticipant(server, "exported", verify_mode="replay")
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
    server, participant = seeded(config)
    outcomes = await finish_cohort(server)
    records = await participant.export(None)
    assert [record.session_id for record in records] == [SESSIONS[1]]
    await server.aclose()

    restored, _ = await restore(config, records)
    replayed = await asyncio.wait_for(verify_as(restored, SESSIONS[1], member(1)), timeout=0.5)

    assert replayed.reward == outcomes[1].reward
    # Answered from the restored cohort: no sibling had to verify again, and the judge did not run.
    restored._run_single_comparison.assert_not_awaited()
    await restored.aclose()


async def test_a_restored_cohort_survives_another_checkpoint_before_its_members_replay(config):
    server, participant = seeded(config)
    outcomes = await finish_cohort(server)
    checkpoint_a = await participant.export(None)
    await server.aclose()

    restored, restored_participant = await restore(config, checkpoint_a)
    # Checkpoint B lands before member 1 replays its verification.
    checkpoint_b = await restored_participant.export(None)
    assert [record.state for record in checkpoint_b] == [record.state for record in checkpoint_a]
    await restored.aclose()

    again, _ = await restore(config, checkpoint_b)
    replayed = await asyncio.wait_for(verify_as(again, SESSIONS[1], member(1)), timeout=0.5)

    assert replayed.reward == outcomes[1].reward
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
