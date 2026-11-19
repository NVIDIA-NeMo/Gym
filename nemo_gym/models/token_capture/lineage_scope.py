# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scope an episode's captured calls by agent instance and model server, and attribute failures to them.

A **lineage scope** is the calls of one agent instance on one model server. Parent search, retry grouping and
terminal selection never cross a lineage scope, so a judge whose prompt embeds the assistant's transcript never
continues the assistant's chain, and two agent instances with identical first prompts are never retries
of each other. The calls of a single-agent episode form one lineage scope.

Outcomes are per agent instance: a failure or a pending call belongs to the agent instance that made the
call, and a failure with no model call, or whose call's identity is unknown, belongs to the whole episode.
Delivery and the manifest both use these helpers, so they agree on every attribution.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

from nemo_gym.models.token_capture.records import CallIdentity, ModelRef, TokenEntry


# (agent instance, model server name). ``None`` parts mean an unnamed agent or an unnamed model server.
LineageScopeKey = tuple[str | None, str | None]
# The agent instance an unnamed agent's outcomes are keyed by.
UNNAMED_AGENT_INSTANCE: str | None = None
# The agent instance whose build is the episode's result when the episode names agent instances.
PRIMARY_AGENT_INSTANCE = "assistant"

_T = TypeVar("_T")


def lineage_scope_key(identity: CallIdentity) -> LineageScopeKey:
    """The lineage scope of a call made with ``identity``."""
    return (identity.agent_instance, identity.model_ref.name if identity.model_ref is not None else None)


def entry_lineage_scope(entry: TokenEntry) -> LineageScopeKey:
    return (entry.agent_instance, entry.model_ref.name if entry.model_ref is not None else None)


def row_lineage_scope(row: Mapping[str, Any] | object) -> LineageScopeKey:
    """The lineage scope of a ledger row, or of a wire record such as ``CallRecord`` or ``ManifestFailure``."""
    if not isinstance(row, Mapping):
        row = {"agent_instance": getattr(row, "agent_instance", None), "model_ref": getattr(row, "model_ref", None)}
    model_ref = row.get("model_ref")
    if isinstance(model_ref, ModelRef):
        name = model_ref.name
    elif isinstance(model_ref, Mapping):
        name = model_ref.get("name")
    else:
        name = None
    return (row.get("agent_instance"), name)


def same_agent_other_server(key: LineageScopeKey, other: LineageScopeKey) -> bool:
    """Whether ``other`` is the same agent instance served by a different model server."""
    return other[0] == key[0] and other[1] != key[1]


def entries_by_lineage_scope(entries: Iterable[TokenEntry]) -> dict[LineageScopeKey, list[TokenEntry]]:
    """Partition entries by lineage scope, in first-appearance order."""
    scopes: dict[LineageScopeKey, list[TokenEntry]] = {}
    for entry in entries:
        scopes.setdefault(entry_lineage_scope(entry), []).append(entry)
    return scopes


def by_agent_instance(items: Iterable[_T], key: Callable[[_T], LineageScopeKey]) -> dict[str | None, list[_T]]:
    """Partition items by ``key(item)[0]``, the agent instance of their lineage scope, in first-appearance order."""
    partitions: dict[str | None, list[_T]] = {}
    for item in items:
        partitions.setdefault(key(item)[0], []).append(item)
    return partitions


def primary_agent_instance(agent_instances: Iterable[str | None]) -> str | None:
    """The agent instance whose result is the episode's top-level result.

    ``assistant`` when the episode has one; otherwise the unnamed agent; otherwise the only agent instance.
    ``assistant`` comes first, so an unnamed caller beside it, such as a helper or a judge that does not
    name itself, never takes the scored response. Raise ``LookupError`` when none of these exists.
    """
    names = list(dict.fromkeys(agent_instances))
    if PRIMARY_AGENT_INSTANCE in names:
        return PRIMARY_AGENT_INSTANCE
    if UNNAMED_AGENT_INSTANCE in names:
        return UNNAMED_AGENT_INSTANCE
    if len(names) == 1:
        return names[0]
    raise LookupError(f"no primary agent instance among {names}")


def inherited_identity_columns(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return each ledger row's identity columns, inheriting them from its call's intent row when it has none.

    A row whose call has no named agent instance, on the row or on its intent, keeps whatever identity it
    has, which may be none: such a row applies to the whole episode.
    """
    intents = {
        row["model_call_id"]: CallIdentity.from_columns(row).columns()
        for row in rows
        if row.get("intent") and "model_call_id" in row
    }
    columns = []
    for row in rows:
        own = CallIdentity.from_columns(row).columns()
        if "agent_instance" not in own:
            own = {**intents.get(row.get("model_call_id"), {}), **own}
        columns.append(own)
    return columns


def is_episode_wide(identity_columns: Mapping[str, Any], *, named_episode: bool) -> bool:
    """Whether a failed or lost call applies to the whole episode rather than to one agent instance.

    It does when nothing about its identity is known, or when the episode names its agent instances but this
    call's agent instance is unknown. A mark that names no model call, which the model server writes for a
    request it could not classify, belongs to the agent instance that made the request when that is known.
    """
    if not identity_columns:
        return True
    return named_episode and "agent_instance" not in identity_columns


class OutcomeAttribution:
    """Which agent instances an episode's failures and pending calls belong to.

    ``agent_instances`` holds every agent instance with its own failure or pending call; the unnamed agent
    is ``None``. ``episode_wide`` is set by a failure with no known identity, or with no agent instance in an
    episode that names them: it masks every agent instance.
    """

    def __init__(self) -> None:
        self.agent_instances: set[str | None] = set()
        self.episode_wide = False

    def add(self, identity_columns: Mapping[str, Any], *, named_episode: bool) -> None:
        """Attribute one failed or pending call.

        In an episode that names its agent instances, a call without a name cannot be placed and is
        episode-wide. In a single-agent episode it belongs to the unnamed agent.
        """
        if is_episode_wide(identity_columns, named_episode=named_episode):
            self.episode_wide = True
            return
        self.agent_instances.add(identity_columns.get("agent_instance"))
