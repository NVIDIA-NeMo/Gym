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
"""The TB4 Harbor-path overlay restores the poll cadence of the benchmark's own server."""

from pathlib import Path

import yaml
from omegaconf import OmegaConf

from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxOperationConfig


ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"
OVERLAY = ROOT / "benchmarks/terminal_bench_4/harbor_sandbox.yaml"


def test_overlay_only_sets_the_poll_cadence():
    overlay = yaml.safe_load(OVERLAY.read_text())
    assert overlay == {
        "sandbox": {
            "opensandbox": {"operations": {"background_poll_initial_s": 0.25, "background_poll_interval_s": 2.0}}
        }
    }


def test_overlay_merged_over_the_shipped_block_matches_the_provider_defaults():
    merged = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(SHIPPED), OmegaConf.load(OVERLAY)), resolve=False)
    operations = OpenSandboxOperationConfig(**merged["sandbox"]["opensandbox"]["operations"])
    defaults = OpenSandboxOperationConfig()
    # The cadence the old TB4 server ran at (it set neither key, so the provider defaults applied).
    assert (operations.background_poll_initial_s, operations.background_poll_interval_s) == (
        defaults.background_poll_initial_s,
        defaults.background_poll_interval_s,
    )
    shipped = yaml.safe_load(SHIPPED.read_text())["sandbox"]["opensandbox"]["operations"]
    assert shipped["background_poll_interval_s"] > operations.background_poll_interval_s
    # Everything else in the shipped block is untouched.
    assert operations.background_exec is shipped["background_exec"]
    assert operations.status_poll_timeout_s == shipped["status_poll_timeout_s"]


def test_readme_documents_the_overlay():
    readme = (ROOT / "benchmarks/terminal_bench_4/README.md").read_text()
    assert readme.count("--config benchmarks/terminal_bench_4/harbor_sandbox.yaml") == 2
