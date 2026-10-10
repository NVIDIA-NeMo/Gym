# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Pairwise reference resolution and where each reference's submission is read from."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
    AABriefcaseReference,
)


@pytest.fixture(autouse=True)
def _no_dataset_on_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)


def _server(**overrides: Any) -> AABriefcaseLiteResourcesServer:
    return AABriefcaseLiteResourcesServer.model_construct(
        config=AABriefcaseLiteResourcesServerConfig.model_construct(
            dataset_dir="unused",
            pairwise_references=dict(overrides.pop("pairwise_references", {})),
            pairwise_reference_ids=list(overrides.pop("pairwise_reference_ids", ["ref_a"])),
        )
    )


class TestReferenceSource:
    def test_a_shipped_reference_reads_from_the_dataset_submissions(self) -> None:
        server = _server()
        server._aa_dataset_root = Path("/data/briefcase")

        source = server._reference_source("gpt-5-5", AABriefcaseReference(), "w1_t1")

        assert source == Path("/data/briefcase/submissions/gpt-5-5/w1_t1/submission")

    def test_a_local_run_reads_the_repeat_layout(self) -> None:
        reference = AABriefcaseReference(deliverables_dir="/runs/glm/deliverables_cache", repeat=2)

        source = _server()._reference_source("glm-5.3-flash", reference, "w1_t4")

        assert source == Path("/runs/glm/deliverables_cache/task_w1_t4/repeat_2")

    def test_the_repeat_defaults_to_zero(self) -> None:
        reference = AABriefcaseReference(deliverables_dir="/runs/glm/deliverables_cache")

        source = _server()._reference_source("glm-5.3-flash", reference, "w1_t1")

        assert source == Path("/runs/glm/deliverables_cache/task_w1_t1/repeat_0")


class TestResolution:
    def test_pairwise_reference_ids_still_resolve(self) -> None:
        server = _server(pairwise_reference_ids=["ref_a", "ref_b"])

        resolved = server._resolved_references()

        assert list(resolved) == ["ref_a", "ref_b"]
        assert resolved["ref_a"].deliverables_dir is None
        assert resolved["ref_a"].exclude_judges == []

    def test_pairwise_references_win_when_set(self) -> None:
        server = _server(
            pairwise_reference_ids=["stale"],
            pairwise_references={
                "glm-5.3-flash": AABriefcaseReference(deliverables_dir="/runs/glm/deliverables_cache", repeat=1)
            },
        )

        resolved = server._resolved_references()

        assert list(resolved) == ["glm-5.3-flash"]
        assert resolved["glm-5.3-flash"].repeat == 1

    def test_an_empty_id_list_with_pairwise_references_is_not_empty(self) -> None:
        configured = _server(
            pairwise_reference_ids=[],
            pairwise_references={"glm-5.3-flash": AABriefcaseReference(deliverables_dir="/runs/glm")},
        )
        empty = _server(pairwise_reference_ids=[], pairwise_references={})

        assert list(configured._resolved_references()) == ["glm-5.3-flash"]
        assert empty._resolved_references() == {}
