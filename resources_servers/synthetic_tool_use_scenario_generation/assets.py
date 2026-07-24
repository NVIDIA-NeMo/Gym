# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Customer-scenario prompt assets."""

import hashlib
from pathlib import Path

from pydantic import BaseModel


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "scenario_system.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "scenario_user.txt"
SCHEMA_PATH = PROMPTS_DIR / "customer_scenario_collection_schema.json"


class ScenarioPrompts(BaseModel):
    system: str
    user: str


def load_scenario_prompts() -> ScenarioPrompts:
    return ScenarioPrompts(
        system=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip(),
        user=USER_PROMPT_PATH.read_text(encoding="utf-8").strip(),
    )


def scenario_asset_hashes() -> dict[str, str]:
    return {
        "scenario_schema": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        "scenario_system_prompt": hashlib.sha256(SYSTEM_PROMPT_PATH.read_bytes()).hexdigest(),
        "scenario_user_prompt": hashlib.sha256(USER_PROMPT_PATH.read_bytes()).hexdigest(),
    }
