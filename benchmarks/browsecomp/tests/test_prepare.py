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
"""Unit tests for the BrowseComp subset selection (network-free).

By default prepare() writes a reproducible 400-sample subset (seed 42); BROWSECOMP_RUN_FULL=1
keeps the full 1266. These exercise the pure _select_samples helper without downloading the CSV.
"""

import base64
import random

import pandas

from benchmarks.browsecomp.prepare import (
    BROWSECOMP_SUBSET_N,
    BROWSECOMP_SUBSET_SEED,
    _select_samples,
    derive_key,
    map_browsecomp_sample_to_rl_sample,
)
from resources_servers.browsecomp_advanced_harness.tool_definitions import WORKSPACE_TOOLS


def _df(n: int = 1266) -> pandas.DataFrame:
    return pandas.DataFrame({"row": range(n)})


def test_default_selects_400():
    out = _select_samples(_df(), run_full=False)
    assert len(out) == BROWSECOMP_SUBSET_N == 400


def test_run_full_keeps_1266_identity():
    df = _df()
    out = _select_samples(df, run_full=True)
    assert len(out) == 1266
    assert list(out["row"]) == list(range(1266))  # full set, unchanged order


def test_subset_is_deterministic_and_matches_seeded_sample():
    df = _df()
    a = _select_samples(df, run_full=False)
    b = _select_samples(df, run_full=False)
    assert list(a["row"]) == list(b["row"])  # same rows every call
    expected = random.Random(BROWSECOMP_SUBSET_SEED).sample(range(1266), BROWSECOMP_SUBSET_N)
    assert list(a["row"]) == expected


def test_subset_is_strict_subset_no_dupes():
    out = _select_samples(_df(), run_full=False)
    rows = set(out["row"])
    assert rows.issubset(set(range(1266)))
    assert len(rows) == 400  # no fabricated or duplicated rows


def _encrypted_row(problem: str, answer: str, canary: str = "canary") -> dict:
    def encrypt(text: str) -> str:
        data = text.encode()
        return base64.b64encode(bytes(a ^ b for a, b in zip(data, derive_key(canary, len(data))))).decode()

    return {"problem": encrypt(problem), "answer": encrypt(answer), "canary": canary}


def test_rows_carry_the_workspace_tools_by_default():
    sample = map_browsecomp_sample_to_rl_sample(_encrypted_row("Who?", "Ada"))
    assert sample["responses_create_params"]["tools"] == WORKSPACE_TOOLS
    assert (sample["question"], sample["ground_truth"]) == ("Who?", "Ada")


def test_harness_neutral_rows_keep_the_prompt_without_tools():
    row = _encrypted_row("Who?", "Ada")
    with_tools = map_browsecomp_sample_to_rl_sample(row)
    without_tools = map_browsecomp_sample_to_rl_sample(row, tools=None)
    assert "tools" not in without_tools["responses_create_params"]
    assert without_tools["responses_create_params"]["input"] == with_tools["responses_create_params"]["input"]
    assert (without_tools["question"], without_tools["ground_truth"]) == ("Who?", "Ada")
