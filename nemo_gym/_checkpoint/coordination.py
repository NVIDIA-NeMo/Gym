# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint coordination: the order in which Gym's participants are driven.

A training controller (for example NeMo RL's Gym actor) calls these functions instead of sequencing
participants itself. They hold no state between calls; every call is safe to retry with the same
checkpoint ID.

The participant order is a Gym invariant:

- ``prepare`` runs stage by stage: environment servers, then the policy model, then agents, then
  resources servers. Environment servers park their episodes first. The policy model then closes
  admission and cuts in-flight generations, so agents waiting on it reach a boundary. Resources servers
  go last because agents and environments call them until they park.
- ``resume`` runs the stages in reverse, so no server is released before the servers it calls.
- ``commit`` and ``restore`` fan out to every participant at once.
- ``retire`` runs callers before callees: environment servers, then agents, then policy model and
  resources servers together. Each stops the attempts' work before it replies.

What the controller still owns:

- Which rollouts to continue. A prepare that misses its deadline returns its blockers; the controller
  calls ``resume`` to abort, then retires them and calls ``prepare`` again.
- Rollouts that finish before the checkpoint. No episode finishes while Gym is prepared, but a ``/run``
  reply can already be on the wire. Before it builds the checkpoint's rollout set, the controller waits
  for every outstanding ``/run`` whose episode is not in the committed ``episode_ids``.
- Publishing. ``commit`` writes each participant's state; the controller publishes the checkpoint only
  after every commit succeeded, and then calls ``resume``.

Restore is all-or-nothing: if any participant fails, every participant is resumed, which discards the
restored state, and the controller restarts those rollouts from their inputs.
"""

import asyncio
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

import orjson

from nemo_gym._checkpoint.control import CHECKPOINT_ROUTE_PREFIX, next_attempt
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import ServerClient


LOGGER = logging.getLogger(__name__)

ParticipantKind = Literal["environment", "model", "agent", "resources"]
PREPARE_ORDER: tuple[ParticipantKind, ...] = ("environment", "model", "agent", "resources")
# Retire stops callers before the servers they call: environment servers call agents and resources servers,
# agents call the policy model and resources servers.
RETIRE_ORDER: tuple[tuple[ParticipantKind, ...], ...] = (("environment",), ("agent",), ("model", "resources"))
_SERVER_TYPES = ("environment_servers", "responses_api_models", "responses_api_agents", "resources_servers")


class CoordinationError(RuntimeError):
    """One or more participants failed a coordination call."""

    def __init__(self, operation: str, failures: Mapping[str, str]) -> None:
        self.operation = operation
        self.failures = dict(failures)
        detail = "; ".join(f"{server}: {error}" for server, error in sorted(self.failures.items()))
        super().__init__(f"checkpoint {operation} failed on {len(self.failures)} participant(s): {detail}")


@dataclass(frozen=True)
class Participant:
    server_name: str
    kind: ParticipantKind


@dataclass(frozen=True)
class Participants:
    """The checkpoint participants of one Gym deployment and how to reach them."""

    client: ServerClient
    auth_token: str
    members: tuple[Participant, ...]

    def of_kind(self, kind: ParticipantKind) -> list[Participant]:
        return [member for member in self.members if member.kind == kind]


@dataclass(frozen=True)
class PrepareResult:
    """Whether every participant is prepared; if not, what each unprepared participant waits for."""

    prepared: bool
    replies: dict[str, dict[str, Any]]

    def blockers(self) -> dict[str, list[str]]:
        return {
            server: reply["report"]["blockers"]
            for server, reply in self.replies.items()
            if reply.get("phase") != "prepared"
        }


async def discover(client: ServerClient, *, auth_token: str) -> Participants:
    """Find every server in the deployment that takes part in checkpoints.

    A server without the control routes, such as an auxiliary model or the legacy agent relay, is not a
    participant. Every other failure is raised: a participant the controller cannot reach would make
    every later checkpoint unsafe.
    """
    names = [
        name
        for name, entry in client.global_config_dict.items()
        if isinstance(entry, Mapping) and any(server_type in entry for server_type in _SERVER_TYPES)
    ]
    members: list[Participant] = []
    failures: dict[str, str] = {}

    async def probe(name: str) -> None:
        try:
            response = await client.request(
                server_name=name,
                url_path=f"{CHECKPOINT_ROUTE_PREFIX}/status",
                method="GET",
                headers=_auth(auth_token),
                _control=True,
            )
            if response.status == 404:
                return
            members.append(Participant(server_name=name, kind=(await _payload(response))["kind"]))
        except Exception as error:
            failures[name] = _describe(error)

    await asyncio.gather(*(probe(name) for name in names))
    if failures:
        raise CoordinationError("discover", failures)
    if not any(member.kind == "model" for member in members):
        raise CoordinationError("discover", {"deployment": "no policy model takes part in checkpoints"})
    members.sort(key=lambda member: (PREPARE_ORDER.index(member.kind), member.server_name))
    return Participants(client=client, auth_token=auth_token, members=tuple(members))


async def prepare(participants: Participants, checkpoint_id: str, *, deadline_ts: float) -> PrepareResult:
    """Prepare every participant in order, stopping at the first stage that is not ready by the deadline."""
    replies: dict[str, dict[str, Any]] = {}
    body = {"checkpoint_id": checkpoint_id, "deadline_ts": deadline_ts}
    for kind in PREPARE_ORDER:
        stage = await _fan_out(participants, participants.of_kind(kind), "prepare", body, deadline_ts)
        replies.update(stage)
        if any(reply["phase"] != "prepared" for reply in stage.values()):
            return PrepareResult(prepared=False, replies=replies)
    return PrepareResult(prepared=True, replies=replies)


async def renew(participants: Participants, checkpoint_id: str, *, deadline_ts: float) -> None:
    """Extend every participant's lease to ``deadline_ts`` plus its grace period.

    Call it when publishing a checkpoint takes longer than the lease of the last control call.
    """
    body = {"checkpoint_id": checkpoint_id, "deadline_ts": deadline_ts}
    await _fan_out(participants, participants.members, "renew", body, deadline_ts)


async def retire(
    participants: Participants, checkpoint_id: str, episode_ids: Iterable[EpisodeId], *, deadline_ts: float
) -> None:
    """Stop these attempts everywhere and free their state; their rollouts restart from input.

    Refused while a checkpoint is open: resume first.
    """
    body = {
        "checkpoint_id": checkpoint_id,
        "deadline_ts": deadline_ts,
        "episode_ids": [episode_id.model_dump(mode="json") for episode_id in episode_ids],
    }
    if body["episode_ids"]:
        # Callers before callees: once a server stops the attempts, nothing upstream can still call it for them.
        for kinds in RETIRE_ORDER:
            members = [member for member in participants.members if member.kind in kinds]
            if members:
                await _fan_out(participants, members, "retire", body, deadline_ts)


async def commit(
    participants: Participants,
    checkpoint_id: str,
    checkpoint_dir: str,
    episode_ids: Iterable[EpisodeId],
    *,
    deadline_ts: float,
) -> dict[str, dict[str, Any]]:
    """Write every participant's state for the episodes the controller will continue.

    ``episode_ids`` must name every episode the controller continues: each episode in flight, and each episode
    restored earlier whose replacement has not started yet, named by that replacement attempt. Restored state of
    an episode the scope leaves out is retired after the write.
    """
    body = {
        "checkpoint_id": checkpoint_id,
        "deadline_ts": deadline_ts,
        "checkpoint_dir": checkpoint_dir,
        "episode_ids": [episode_id.model_dump(mode="json") for episode_id in episode_ids],
    }
    return await _fan_out(participants, participants.members, "commit", body, deadline_ts)


async def restore(
    participants: Participants,
    checkpoint_id: str,
    checkpoint_dir: str,
    episode_ids: Iterable[EpisodeId],
    *,
    deadline_ts: float,
) -> dict[str, dict[str, Any]]:
    """Install the checkpointed state of these episodes in every participant, or in none.

    Each restored episode continues as its next attempt. If any participant fails, the next attempts are
    retired everywhere and the error is raised; the controller then restarts those rollouts from their
    inputs as later attempts.
    """
    episode_ids = list(episode_ids)
    body = {
        "checkpoint_id": checkpoint_id,
        "deadline_ts": deadline_ts,
        "checkpoint_dir": checkpoint_dir,
        "episode_ids": [episode_id.model_dump(mode="json") for episode_id in episode_ids],
    }
    try:
        return await _fan_out(participants, participants.members, "restore", body, deadline_ts)
    except CoordinationError:
        LOGGER.warning("checkpoint %s restore failed; discarding the restored state everywhere", checkpoint_id)
        cleanup_deadline = max(deadline_ts, time.time() + 30)
        # Retiring the replacement attempts frees their restored state, so nothing can continue from a
        # partial restore.
        await retire(participants, checkpoint_id, [next_attempt(e) for e in episode_ids], deadline_ts=cleanup_deadline)
        await resume(participants, checkpoint_id, deadline_ts=cleanup_deadline)
        raise


async def resume(participants: Participants, checkpoint_id: str, *, deadline_ts: float) -> None:
    """Release every participant in reverse order. This also aborts a checkpoint that was not published."""
    body = {"checkpoint_id": checkpoint_id, "deadline_ts": deadline_ts}
    failures: dict[str, str] = {}
    for kind in reversed(PREPARE_ORDER):
        try:
            await _fan_out(participants, participants.of_kind(kind), "resume", body, deadline_ts)
        except CoordinationError as error:
            # Keep releasing the other stages; a participant that cannot be reached resumes on its own
            # when its lease expires.
            failures.update(error.failures)
    if failures:
        raise CoordinationError("resume", failures)


async def _fan_out(
    participants: Participants,
    members: list[Participant] | tuple[Participant, ...],
    operation: str,
    body: dict[str, Any],
    deadline_ts: float,
) -> dict[str, dict[str, Any]]:
    replies: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    async def call(member: Participant) -> None:
        try:
            replies[member.server_name] = await _post(participants, member, operation, body, deadline_ts)
        except Exception as error:
            failures[member.server_name] = _describe(error)

    await asyncio.gather(*(call(member) for member in members))
    if failures:
        raise CoordinationError(operation, failures)
    return replies


async def _post(
    participants: Participants, member: Participant, operation: str, body: dict[str, Any], deadline_ts: float
) -> dict[str, Any]:
    # The participant bounds its own work by deadline_ts; the extra seconds cover the reply in flight.
    async with asyncio.timeout(max(deadline_ts - time.time(), 0) + 5):
        response = await participants.client.request(
            server_name=member.server_name,
            url_path=f"{CHECKPOINT_ROUTE_PREFIX}/{operation}",
            method="POST",
            json=body,
            headers=_auth(participants.auth_token),
            _control=True,
        )
        return await _payload(response)


async def _payload(response: Any) -> dict[str, Any]:
    content = await response.read()
    if response.status != 200:
        raise _ParticipantError(response.status, content)
    return orjson.loads(content)


class _ParticipantError(Exception):
    def __init__(self, status: int, content: bytes) -> None:
        try:
            error = orjson.loads(content).get("error") or {}
        except (orjson.JSONDecodeError, AttributeError):
            error = {}
        self.code: Optional[str] = error.get("code") if isinstance(error, dict) else None
        detail = error.get("detail") if isinstance(error, dict) else None
        super().__init__(f"HTTP {status} {self.code or ''} {detail or content[:200]!r}".strip())


def _describe(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
