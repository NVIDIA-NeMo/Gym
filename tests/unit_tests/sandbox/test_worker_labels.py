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
from collections.abc import Mapping

import pytest

from nemo_gym.core import worker_resources
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.worker_labels import (
    WORKER_LABEL,
    CloseByMetadataResult,
    close_by_metadata,
    worker_labels,
)


class _Provider:
    """A provider whose listing and current labels are set by the test."""

    name = "fake"

    def __init__(self, listed: Mapping[str, Mapping[str, str]], *, failing: frozenset[str] = frozenset()) -> None:
        self.listed = {sandbox_id: dict(labels) for sandbox_id, labels in listed.items()}
        self.current = {sandbox_id: dict(labels) for sandbox_id, labels in listed.items()}
        self.failing = failing
        self.closed: list[str] = []

    async def list_by_metadata(self, labels: Mapping[str, str]) -> list[SandboxHandle]:
        # Advisory: returns everything it lists, matching or not, as a lax server-side filter could.
        return [SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None) for sandbox_id in self.listed]

    async def metadata_of(self, handle: SandboxHandle) -> dict[str, str] | None:
        return self.current.get(handle.sandbox_id)

    async def close(self, handle: SandboxHandle) -> None:
        if handle.sandbox_id in self.failing:
            raise RuntimeError("provider refused")
        self.closed.append(handle.sandbox_id)


def test_the_worker_label_names_this_worker_under_a_fixed_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_resources, "_routing_id", None)
    worker_resources.set_worker_routing_id("d" * 32)

    # The key does not follow attribution.key_prefix: a reap must find the label whatever a provider block sets.
    assert WORKER_LABEL == "nemo-gym.nvidia.com/worker"
    assert worker_labels() == {WORKER_LABEL: "d" * 32}


async def test_close_by_metadata_rechecks_every_listed_sandbox_before_closing_it() -> None:
    provider = _Provider(
        {
            "match": {WORKER_LABEL: "dead", "other": "x"},
            "never-matched": {WORKER_LABEL: "live"},
            "relabeled": {WORKER_LABEL: "dead"},
            "gone": {WORKER_LABEL: "dead"},
            "broken": {WORKER_LABEL: "dead"},
        },
        failing=frozenset({"broken"}),
    )
    provider.current["relabeled"] = {WORKER_LABEL: "live"}
    del provider.current["gone"]

    result = await close_by_metadata(provider, {WORKER_LABEL: "dead"})

    assert result == CloseByMetadataResult(
        closed=("match",), refused=("never-matched", "relabeled"), failed=("broken",), gone=("gone",)
    )
    assert provider.closed == ["match"]


@pytest.mark.parametrize("labels", [{}, {WORKER_LABEL: ""}])
async def test_close_by_metadata_refuses_labels_that_select_nobody_in_particular(labels: dict[str, str]) -> None:
    provider = _Provider({"any": {}})

    with pytest.raises(ValueError, match="at least one label"):
        await close_by_metadata(provider, labels)
    assert provider.closed == []


async def test_close_by_metadata_needs_a_provider_that_lists_by_label() -> None:
    class _NoListing:
        name = "plain"

    with pytest.raises(NotImplementedError, match="'plain' cannot close sandboxes by label"):
        await close_by_metadata(_NoListing(), {WORKER_LABEL: "dead"})


async def test_close_by_metadata_refuses_a_value_providers_would_rewrite() -> None:
    provider = _Provider({"any": {WORKER_LABEL: "a/b"}})

    with pytest.raises(ValueError, match="valid Kubernetes label values"):
        await close_by_metadata(provider, {WORKER_LABEL: "a/b"})
    assert provider.closed == []
