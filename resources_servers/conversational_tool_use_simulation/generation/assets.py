# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the independently owned prompt assets used by the generation workflow."""

from dataclasses import dataclass

from resources_servers.conversational_tool_use_simulation.generation.domain.assets import (
    domain_asset_hashes,
    load_domain_prompt,
)
from resources_servers.conversational_tool_use_simulation.generation.policy_tools.profiles import (
    PolicyToolsProfile,
    load_profile,
    profile_asset_hashes,
)
from resources_servers.conversational_tool_use_simulation.generation.scenarios.assets import (
    ScenarioPrompts,
    load_scenario_prompts,
    scenario_asset_hashes,
)


@dataclass(frozen=True)
class GenerationAssets:
    domain_prompt: str
    policy_tools: PolicyToolsProfile
    scenarios: ScenarioPrompts


def load_generation_assets(profile_name: str) -> GenerationAssets:
    return GenerationAssets(
        domain_prompt=load_domain_prompt(),
        policy_tools=load_profile(profile_name),
        scenarios=load_scenario_prompts(),
    )


def generation_asset_hashes(profile_name: str) -> dict[str, str]:
    hashes = {
        **domain_asset_hashes(),
        **profile_asset_hashes(profile_name),
        **scenario_asset_hashes(),
    }
    return dict(sorted(hashes.items()))
