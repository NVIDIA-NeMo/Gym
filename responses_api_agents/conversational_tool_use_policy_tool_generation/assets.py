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

"""Package-local prompt and golden-reference loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from responses_api_agents.conversational_tool_use_policy_tool_generation.models import PolicyToolProfile


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
GOLDENS_DIR = PACKAGE_DIR / "references" / "golden_policies"


@dataclass(frozen=True)
class GoldenPair:
    index: int
    policy: str
    tools: str


@dataclass(frozen=True)
class PolicyToolAssets:
    profile: PolicyToolProfile
    policy_prompt: str
    tools_prompt: str
    policy_refine_prompt: str
    tools_refine_prompt: str
    cohesion_judge_prompt: str
    golden_judge_prompt: str
    golden_pairs: tuple[GoldenPair, ...]


def _read_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def load_assets(profile: PolicyToolProfile) -> PolicyToolAssets:
    profile_prefix = "general" if profile == "general" else "proactive"
    golden_pairs = []
    for index in range(1, 9):
        golden_pairs.append(
            GoldenPair(
                index=index,
                policy=(GOLDENS_DIR / f"policy-{index}.md").read_text(encoding="utf-8").strip(),
                tools=(GOLDENS_DIR / f"tools_{index}.jsonl").read_text(encoding="utf-8").strip(),
            )
        )
    return PolicyToolAssets(
        profile=profile,
        policy_prompt=_read_prompt(f"{profile_prefix}_policy.txt"),
        tools_prompt=_read_prompt(f"{profile_prefix}_tools.txt"),
        policy_refine_prompt=_read_prompt(f"{profile_prefix}_policy_refine.txt"),
        tools_refine_prompt=_read_prompt("tools_refine.txt"),
        cohesion_judge_prompt=_read_prompt("cohesion_judge.txt"),
        golden_judge_prompt=_read_prompt("golden_judge.txt"),
        golden_pairs=tuple(golden_pairs),
    )
