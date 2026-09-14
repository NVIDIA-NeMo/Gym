# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GenRM's complete-group contract at the file-based evaluation boundary.

External callers (including RL) own their identities and failure protocol. This
module is only used by Gym's evaluation collector and reverification command.
"""

import asyncio
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from uuid import uuid4

from omegaconf import DictConfig

from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    COHORT_FAILURE_MODE_KEY_NAME,
    GROUP_ATTEMPT_KEY_NAME,
    GROUP_ID_KEY_NAME,
    GROUP_MEMBER_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)


def genrm_group_sizes(global_config: dict) -> dict[str, int]:
    """Return instance names and group sizes for configured cohort verifiers."""
    sizes = {}
    for name, block in global_config.items():
        if not isinstance(block, (dict, DictConfig)):
            continue
        settings = (block.get("resources_servers") or {}).get("genrm_compare")
        if settings is not None and int(settings.get("num_rollouts_per_prompt", 1)) > 1:
            sizes[name] = int(settings["num_rollouts_per_prompt"])
    return sizes


def prepare_genrm_collection(
    rows: list[dict], global_config: dict, concurrency: int | None, *, resume: bool = False
) -> bool:
    """Stamp complete evaluation groups while preserving global rollout indices.

    On resume only whole pending groups are supported. Refuse a partial group
    before dispatch rather than silently changing its already-persisted rewards.
    The caller must persist these updated identities before starting requests.
    Returns whether any GenRM input rows were found.
    """
    sizes = genrm_group_sizes(global_config)
    if not sizes:
        return False
    agent_sizes = {}
    for name, block in global_config.items():
        if not isinstance(block, (dict, DictConfig)):
            continue
        for settings in (block.get("responses_api_agents") or {}).values():
            resource = (settings.get("resources_server") or {}).get("name")
            if resource in sizes:
                agent_sizes[name] = sizes[resource]

    groups = defaultdict(list)
    for row in rows:
        agent = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
        if agent in agent_sizes:
            groups[(agent, row[TASK_INDEX_KEY_NAME], row.get(GROUP_ID_KEY_NAME))].append(row)
    updates = []
    identity_owners = {}
    for (agent, task, group_id), members in groups.items():
        size = agent_sizes[agent]
        if concurrency and concurrency < size:
            raise ConfigError(f"GenRM group size {size} requires concurrency >= {size}; got {concurrency}")
        members.sort(key=lambda row: row[ROLLOUT_INDEX_KEY_NAME])
        if len(members) % size or (group_id is not None and len(members) != size):
            raise ConfigError(
                f"GenRM task {task}, agent {agent} has {len(members)} pending members, expected complete groups of {size}. "
                "Partial-group resume is unsupported; rerun the complete group into a fresh output file."
            )
        if resume and group_id is None:
            raise ConfigError(
                "GenRM cache lacks group identities. Start a fresh output file to migrate legacy results safely."
            )
        for start in range(0, len(members), size):
            cohort = members[start : start + size]
            attempts = [row.get(GROUP_ATTEMPT_KEY_NAME, 0) for row in cohort]
            if any(type(attempt) is not int or attempt < 0 for attempt in attempts) or len(set(attempts)) != 1:
                raise ConfigError("GenRM group members must share one nonnegative integer group attempt")
            if resume and {row.get(GROUP_MEMBER_INDEX_KEY_NAME) for row in cohort} != set(range(size)):
                raise ConfigError("GenRM resume requires every persisted member slot exactly once")
            identity = group_id if group_id is not None else uuid4().hex
            if not isinstance(identity, str) or not 1 <= len(identity) <= 256:
                raise ConfigError("GenRM group IDs must be nonempty strings of at most 256 characters")
            if identity in identity_owners and identity_owners[identity] != (agent, task):
                raise ConfigError("GenRM group IDs must not be reused across tasks or agents")
            identity_owners[identity] = (agent, task)
            for index, row in enumerate(cohort):
                updates.append(
                    (
                        row,
                        {
                            GROUP_ID_KEY_NAME: identity,
                            GROUP_ATTEMPT_KEY_NAME: attempts[0] + int(resume),
                            GROUP_MEMBER_INDEX_KEY_NAME: row[GROUP_MEMBER_INDEX_KEY_NAME] if resume else index,
                            # Only this collector understands the judge-failed sidecar contract.
                            COHORT_FAILURE_MODE_KEY_NAME: "row",
                        },
                    )
                )
    for row, update in updates:
        row.update(update)
    return bool(groups)


def reject_genrm_reverification(global_config: dict) -> None:
    """Individual reverification cannot reconstruct or replace a GenRM cohort."""
    if genrm_group_sizes(global_config):
        raise ConfigError(
            "GenRM cohort reverification is unsupported, including force and judge_failed_only. "
            "It requires complete group membership and a coordinated fresh group attempt; "
            "rerun complete groups into a fresh output file."
        )


def collection_admission(rows: list[dict], semaphore):
    """Reserve enough permits for an entire group before admitting its members.

    Independent per-row permits can deadlock even at concurrency=N when slots
    are occupied by members of different groups. Ordinary rows retain their
    existing semaphore behavior. Only the file collector's prepared groups opt in.
    """

    def key(row):
        return row.get(GROUP_ID_KEY_NAME) if row.get(COHORT_FAILURE_MODE_KEY_NAME) == "row" else None

    sizes = Counter(key(row) for row in rows if key(row) is not None)
    admitted = {}
    reservation_lock = asyncio.Lock()

    @asynccontextmanager
    async def admission(row):
        group = key(row)
        if group is None or not isinstance(semaphore, asyncio.Semaphore):
            async with semaphore:
                yield
            return
        future = admitted.get(group)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            admitted[group] = future
            acquired = 0
            try:
                async with reservation_lock:
                    for _ in range(sizes[group]):
                        await semaphore.acquire()
                        acquired += 1
                future.set_result(None)
            except BaseException as error:
                for _ in range(acquired):
                    semaphore.release()
                future.set_exception(error)

        def release_if_admitted(done):
            if not done.cancelled() and done.exception() is None:
                semaphore.release()

        try:
            await asyncio.shield(future)
        except BaseException:
            # A member cancelled before admission still owns one reservation
            # if its group subsequently enters. Release it when that happens.
            future.add_done_callback(release_if_admitted)
            raise
        try:
            yield
        finally:
            semaphore.release()

    return admission
