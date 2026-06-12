# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the dataset-name-driven agent runtime resolver.

Mirrors OpenHands' set_dataset_type + _get_workspace_path + per-dataset env activation,
but resolves to the IN-PLACE repo dir (no /workspace copy) and pins the agent's python via
the env bin on openclaw exec.pathPrepend (the only lever that reaches the agent — openclaw's
exec rebuilds PATH from a sanitized base). Covers every dataset OpenHands does, by name.
"""

import pytest

from responses_api_agents.swe_agents.openclaw.dataset_env import (
    resolve_agent_env_bin,
    resolve_dataset_type,
    resolve_workspace_path,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("princeton-nlp/SWE-bench_Verified", "SWE-bench"),
        ("SWE-bench_Lite", "SWE-bench"),
        ("SWE-bench_Multilingual", "SWE-bench_Multilingual"),
        ("r2e-gym-subset", "R2E-Gym"),
        ("nv-internal-1", "nv-internal-1"),
        ("swe-bench-ext", "swe-bench-ext"),
        ("SWE-rebench", "SWE-rebench"),
        ("SWE-rebench-V2", "SWE-rebench-V2"),
        ("SWE-Gym", "SWE-Gym"),
        ("SWE-bench-Live", "SWE-bench-Live"),
        # precedence: V2 must win over the bare 'swe-rebench' substring; the underscore alias too
        ("my-org/SWE-rebench-V2-fork", "SWE-rebench-V2"),
        ("swe-rebench_v2", "SWE-rebench-V2"),
        # matching is case-insensitive; 'multilingual' must not be swallowed by 'multimodal'
        ("R2E-GYM", "R2E-Gym"),
        ("some-multilingual-set", "SWE-bench_Multilingual"),
        ("SWE-bench_Multimodal", "Multimodal"),
        # the multimodal branch is checked before multilingual: a name with both substrings
        # resolves to Multimodal (multimodal precedes multilingual in the impl)
        ("multimodal-multilingual-set", "Multimodal"),
    ],
)
def test_resolve_dataset_type(name, expected):
    assert resolve_dataset_type(name) == expected


def test_resolve_dataset_type_empty_defaults_to_swe_bench():
    assert resolve_dataset_type("") == "SWE-bench"
    assert resolve_dataset_type(None) == "SWE-bench"


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("SWE-bench", "/testbed"),  # the default branch (every non-special dataset maps here)
        ("nv-internal-1", "/app"),
        ("swe-bench-ext", "/workspace/repo"),
    ],
)
def test_resolve_workspace_path_in_place(dtype, expected):
    assert resolve_workspace_path(dtype, {}) == expected


def test_resolve_workspace_path_rebench_v2_uses_repo_name():
    assert resolve_workspace_path("SWE-rebench-V2", {"repo": "psf/requests"}) == "/requests"
    assert resolve_workspace_path("SWE-rebench-V2", {"repo": "requests"}) == "/requests"


def test_resolve_workspace_path_rebench_v2_missing_repo_fails_loud():
    with pytest.raises(ValueError, match="repo"):
        resolve_workspace_path("SWE-rebench-V2", {})


# --- agent env bin: the dir prepended to openclaw exec.pathPrepend so the agent's `python`
# resolves to the repo interpreter. conda activate in run_openclaw.sh can't reach the agent
# (openclaw's exec rebuilds PATH from a sanitized base; pathPrepend is applied after, so it
# is the only reliable lever). Verified by wet-run diagnostic.
def test_resolve_agent_env_bin_conda():
    assert resolve_agent_env_bin("princeton-nlp/SWE-bench_Verified") == "/opt/miniconda3/envs/testbed/bin"
    # Multimodal is in the conda 'testbed' set, so it resolves to the same conda bin.
    assert resolve_agent_env_bin("SWE-bench_Multimodal") == "/opt/miniconda3/envs/testbed/bin"


def test_resolve_agent_env_bin_venv():
    assert resolve_agent_env_bin("r2e-gym") == "/testbed/.venv/bin"


@pytest.mark.parametrize("name", ["nv-internal-1", "swe-bench-ext", "SWE-bench-Live", "SWE-rebench-V2"])
def test_resolve_agent_env_bin_none(name):
    assert resolve_agent_env_bin(name) is None
