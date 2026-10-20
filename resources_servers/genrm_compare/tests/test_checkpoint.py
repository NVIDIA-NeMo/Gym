# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint must not lose a completed cohort that only some of its members have recorded."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.genrm_compare.app as genrm
from nemo_gym._checkpoint.control import next_attempt
from nemo_gym._checkpoint.resources import ResourcesParticipant
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import rollout_context


def comparison_result(score_1=4.0, score_2=2.0, ranking=1.0):
    scores = (score_1, score_2, ranking)
    return (*scores, *scores, -1.0, -1.0, -1.0, 0.0, 0.0)


def member(index, *, group="group", attempt=0):
    return genrm.GenRMCompareVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "2+2?"}]),
        response=NeMoGymResponse(
            id=f"answer-{index}",
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


class FakeParticipant:
    """The parts of ``ResourcesParticipant`` GenRM uses: admission, and which attempt owns each session."""

    def __init__(self) -> None:
        self.accepting = True
        self._open = asyncio.Event()
        self._open.set()
        self.sessions: dict[str, EpisodeId] = {}

    def close(self) -> None:
        self.accepting = False
        self._open.clear()

    def open(self) -> None:
        self.accepting = True
        self._open.set()

    async def wait_open(self) -> None:
        while not self._open.is_set():
            await self._open.wait()

    def session_episode(self, session_id: str):
        return self.sessions.get(session_id)


def checkpointing_server(config) -> tuple[genrm.GenRMCompareResourcesServer, FakeParticipant]:
    server = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    participant = FakeParticipant()
    server._checkpoint = participant
    server._run_single_comparison = AsyncMock(return_value=comparison_result())
    return server, participant


async def verify_as(server, episode: EpisodeId, body):
    with rollout_context(episode.capture_key):
        return await server.verify(body)


EPISODES = [EpisodeId(rollout_id=f"group_g{index}_a0") for index in range(2)]


async def test_a_member_that_has_not_recorded_its_reward_gets_it_back_after_a_restore(config):
    """Member 0 recorded its reward before the checkpoint; member 1 had not, so it replays verification."""
    server, participant = checkpointing_server(config)
    participant.sessions = {"session-0": EPISODES[0], "session-1": EPISODES[1]}
    first = asyncio.create_task(verify_as(server, EPISODES[0], member(0)))
    await asyncio.sleep(0)
    rewards = [(await verify_as(server, EPISODES[1], member(1))).reward, (await first).reward]

    # Member 0 closed its session, so only member 1's session is checkpointed.
    states = await server.export_session_states(["session-1"])
    await server.aclose()

    restored, restored_participant = checkpointing_server(config)
    await restored.restore_session_states(states)
    replacement = EpisodeId(rollout_id=EPISODES[1].rollout_id, attempt=1)
    replayed = await asyncio.wait_for(verify_as(restored, replacement, member(1)), timeout=0.5)

    assert replayed.reward == rewards[0]
    # Answered from the restored cohort: no sibling had to verify again, and the judge did not run.
    restored._run_single_comparison.assert_not_awaited()
    await restored.aclose()


async def test_a_session_whose_cohort_has_no_reward_yet_exports_nothing(config):
    server, participant = checkpointing_server(config)
    participant.sessions = {"session-0": EPISODES[0], "unknown": EpisodeId(rollout_id="other")}
    waiting = asyncio.create_task(verify_as(server, EPISODES[0], member(0)))
    await asyncio.sleep(0)

    assert await server.export_session_states(["session-0", "unknown"]) == {"session-0": {}, "unknown": {}}

    # An empty state restores nothing, so replayed members rebuild the cohort as before.
    restored, _ = checkpointing_server(config)
    await restored.restore_session_states({"session-0": {}})
    assert restored._verify_cohorts == {}
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    await server.aclose()
    await restored.aclose()


async def test_rewards_are_held_while_a_checkpoint_is_open(config):
    """A reward published after this server exported would let one member record it while its siblings could not."""
    server, participant = checkpointing_server(config)
    participant.sessions = {"session-0": EPISODES[0], "session-1": EPISODES[1]}
    participant.close()
    members = [asyncio.create_task(verify_as(server, EPISODES[index], member(index))) for index in range(2)]
    for _ in range(20):
        await asyncio.sleep(0)

    # The judge ran, but no member has its reward, and the export holds none.
    assert server._run_single_comparison.await_count > 0
    assert not any(task.done() for task in members)
    assert await server.export_session_states(["session-0", "session-1"]) == {"session-0": {}, "session-1": {}}

    participant.open()
    rewards = await asyncio.wait_for(asyncio.gather(*members), timeout=0.5)
    assert all(reward.reward == 3.0 for reward in rewards)
    await server.aclose()


async def test_a_restore_that_names_a_cohort_inconsistently_installs_nothing(config):
    server, participant = checkpointing_server(config)
    participant.sessions = {"session-0": EPISODES[0], "session-1": EPISODES[1]}
    await asyncio.gather(*(verify_as(server, EPISODES[index], member(index)) for index in range(2)))
    states = await server.export_session_states(["session-0", "session-1"])
    await server.aclose()
    states["session-1"] = {**states["session-1"], "group_attempt": 1}

    restored, _ = checkpointing_server(config)
    with pytest.raises(ValueError, match="inconsistently"):
        await restored.restore_session_states(states)
    assert restored._verify_cohorts == {}
    assert restored._latest_group_attempts == {}
    await restored.aclose()


async def test_a_restored_cohort_refuses_an_older_attempt_of_its_group(config):
    server, participant = checkpointing_server(config)
    participant.sessions = {"session-0": EPISODES[0]}
    attempt_one = [member(index, attempt=1) for index in range(2)]
    await asyncio.gather(*(verify_as(server, EPISODES[index], attempt_one[index]) for index in range(2)))
    states = await server.export_session_states(["session-0"])
    await server.aclose()

    restored, _ = checkpointing_server(config)
    await restored.restore_session_states(states)
    with pytest.raises(genrm.HTTPException, match="superseded"):
        await verify_as(restored, EPISODES[1], member(1, attempt=0))
    await restored.aclose()


def participant_server(config) -> tuple[genrm.GenRMCompareResourcesServer, ResourcesParticipant]:
    """A server behind the real resources participant, so restores bind sessions as Gym does."""
    server = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    participant = ResourcesParticipant(server, "exported", verify_mode="replay")
    server._checkpoint = participant
    server._run_single_comparison = AsyncMock(return_value=comparison_result())
    return server, participant


async def test_a_restored_cohort_survives_another_checkpoint_before_its_members_replay(config):
    """A restore installs state before it binds sessions to their attempts, so the export must not need a replay."""
    server, participant = participant_server(config)
    for index, episode in enumerate(EPISODES):
        participant.seeded(f"session-{index}", episode)
    first = asyncio.create_task(verify_as(server, EPISODES[0], member(0)))
    await asyncio.sleep(0)
    reward = (await verify_as(server, EPISODES[1], member(1))).reward
    await first
    # Member 0 recorded its reward and closed its session; member 1 has not.
    participant.ended("session-0")
    checkpoint_a = await participant.export(None)
    assert [record.session_id for record in checkpoint_a] == ["session-1"]
    await server.aclose()

    restored, restored_participant = participant_server(config)
    await restored_participant.install(checkpoint_a, [record.episode_id for record in checkpoint_a])
    # Checkpoint B lands before member 1 replays its verification.
    checkpoint_b = await restored_participant.export(None)
    assert [record.state for record in checkpoint_b] == [record.state for record in checkpoint_a]
    await restored.aclose()

    again, again_participant = participant_server(config)
    await again_participant.install(checkpoint_b, [record.episode_id for record in checkpoint_b])
    replacement = next_attempt(next_attempt(EPISODES[1]))
    replayed = await asyncio.wait_for(verify_as(again, replacement, member(1)), timeout=0.5)

    assert replayed.reward == reward
    again._run_single_comparison.assert_not_awaited()
    await again.aclose()
