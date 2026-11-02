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

"""Package-local scenario-generation prompt loading."""

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
PROMPT_FILENAMES = ("scenario_system.txt", "scenario_user.txt")
SCHEMA_PATH = PROMPTS_DIR / "customer_scenario_collection_schema.json"
PREPARE_COMMAND = "python -m resources_servers.conversational_tool_use_simulation.prepare"


@dataclass(frozen=True)
class ScenarioAssets:
    system_prompt: str
    user_prompt: str
    schema: str


def load_assets() -> ScenarioAssets:
    missing = [filename for filename in PROMPT_FILENAMES if not (PROMPTS_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Conversational tool-use prompts are not prepared. "
            f"Run `{PREPARE_COMMAND}`; missing {PROMPTS_DIR / missing[0]}."
        )
    return ScenarioAssets(
        system_prompt=(PROMPTS_DIR / "scenario_system.txt").read_text(encoding="utf-8"),
        user_prompt=(PROMPTS_DIR / "scenario_user.txt").read_text(encoding="utf-8"),
        schema=SCHEMA_PATH.read_text(encoding="utf-8"),
    )
