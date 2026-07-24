# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package-local scenario-generation prompt loading."""

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "scenario_system.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "scenario_user.txt"
SCHEMA_PATH = PROMPTS_DIR / "customer_scenario_collection_schema.json"


@dataclass(frozen=True)
class ScenarioAssets:
    system_prompt: str
    user_prompt: str
    schema: str


def load_assets() -> ScenarioAssets:
    return ScenarioAssets(
        system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        user_prompt=USER_PROMPT_PATH.read_text(encoding="utf-8"),
        schema=SCHEMA_PATH.read_text(encoding="utf-8"),
    )
