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

"""Prompt assets owned by the domain-generation agent."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
PROMPT_FILENAMES = ("domain_generation.txt",)
PREPARE_COMMAND = "python -m resources_servers.conversational_tool_use_simulation.prepare"


def _read_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Conversational tool-use prompts are not prepared. Run `{PREPARE_COMMAND}`; missing {path}."
        )
    return path.read_text(encoding="utf-8").strip()


def load_domain_prompt() -> str:
    return _read_prompt("domain_generation.txt")
