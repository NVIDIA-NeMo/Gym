# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run in the repo-root venv, like `gym eval prepare`. Regenerating a frozen instance through the ported tracks must
# reproduce SwarmBench's recorded hash byte for byte, which pins the port to the upstream generators.

import json

import pytest

from benchmarks.swarmbench import prepare
from resources_servers.swarmbench.app import CANARY


def _instances(fpath) -> dict:
    return {i["id"]: i for i in json.loads(fpath.read_text())["instances"]}


V2 = _instances(prepare.MANIFEST_FPATH)
V1 = _instances(prepare.BENCHMARK_DIR / "instances.json")


@pytest.mark.parametrize("instance_id", ["search-hard-dev-000", "rootcause-easy-dev-000", "code-easy-dev-000"])
def test_v2_instance_regenerates_byte_for_byte(instance_id: str) -> None:
    row = prepare.build_row(V2[instance_id], canary=True)
    assert row["verifier_metadata"] == dict(V2[instance_id], canary=True)
    assert "input" not in row["responses_create_params"]  # the prompt_config builds the input
    names = {t["name"] for t in row["responses_create_params"]["tools"]}
    assert names and all(name.startswith(V2[instance_id]["track"] + "_") for name in names)
    assert row["prompt"].startswith(CANARY + "\n\n")
    budget = V2[instance_id]["budget_s"]
    assert row["prompt"].endswith(prepare.BUDGET_LINE.format(b=budget, g=V2[instance_id]["grace_s"]))


@pytest.mark.parametrize("instance_id", ["corpus-dev-n64-000", "search-dev-n64-000"])
def test_v1_instance_still_regenerates(instance_id: str) -> None:
    row = prepare.build_row(V1[instance_id])
    assert not row["prompt"].startswith(CANARY) and "Time budget" not in row["prompt"]


def test_manifest_covers_every_track_and_tier() -> None:
    cells = {(i["split"], i["tier"], i["track"]) for i in V2.values()}
    assert len(cells) == 2 * 3 * 8
    assert all({"tier", "track", "n", "seed", "budget_s", "sha256"} <= set(i) for i in V2.values())


def test_hash_mismatch_is_rejected() -> None:
    instance = dict(V2["finance-easy-dev-000"], sha256="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        prepare.build_row(instance, canary=True)


TYPES = _instances(prepare.TYPES_MANIFEST_FPATH)


@pytest.mark.parametrize("instance_id", ["a01_evidence_matrix-easy-dev-000", "f60_dag-easy-dev-000"])
def test_type_instance_regenerates_byte_for_byte(instance_id: str) -> None:
    row = prepare.build_row(TYPES[instance_id])
    assert row["prompt"].startswith(CANARY + "\n\n")  # the types embed the canary themselves
    assert row["verifier_metadata"]["canary"] is False and "Time budget" in row["prompt"]


def test_types_manifest_covers_every_type_and_tier() -> None:
    cells = {(i["split"], i["track"], i["tier"]) for i in TYPES.values()}
    assert len(cells) == 2 * 20 * 3
