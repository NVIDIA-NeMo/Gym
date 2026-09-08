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
"""System prompt for the role-swapped LLM user simulator.

The guidelines text (prompts/user-sim-guidelines.md) is adapted from the
tau2-bench user simulator (MIT); the scenario block is serialized in the same
layout as tau2-bench's ``StructuredUserInstructions`` so tasks written in that
format drive the same persona here.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional


# Termination tokens the simulator emits (same convention as tau2-bench).
STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")

GUIDELINES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "user-sim-guidelines.md")
_SIM_SYSTEM_TEMPLATE = "{guidelines}\n\n<scenario>\n{instructions}\n</scenario>"

_guidelines_cache: Optional[str] = None


def load_guidelines() -> str:
    global _guidelines_cache
    if _guidelines_cache is None:
        with open(GUIDELINES_PATH, encoding="utf-8") as f:
            text = f.read()
        # Strip the attribution HTML comment; it is not part of the prompt.
        text = re.sub(r"\A\s*<!--.*?-->\s*", "", text, flags=re.DOTALL)
        _guidelines_cache = text.replace("<PERSONA_GUIDELINES>", "").strip()
    return _guidelines_cache


def scenario_to_text(scenario: dict[str, Any]) -> str:
    """Serialize a task's user_scenario in the tau2-bench StructuredUserInstructions layout."""
    persona = scenario.get("persona")
    instr = scenario.get("instructions") or {}
    lines: list[str] = []
    if persona:
        lines.append("Persona:")
        lines.append("\t" + str(persona).replace("\n", "\n\t"))
    if isinstance(instr, str):
        lines.append(instr)
        return "\n".join(lines)
    lines.append("Domain: banking")
    if instr.get("reason_for_call"):
        lines.append("Reason for call:\n\t" + instr["reason_for_call"].replace("\n", "\n\t"))
    if instr.get("known_info"):
        lines.append("Known info:\n\t" + instr["known_info"].replace("\n", "\n\t"))
    if instr.get("unknown_info"):
        lines.append("Unknown info:\n\t" + instr["unknown_info"].replace("\n", "\n\t"))
    if instr.get("task_instructions"):
        lines.append("Task instructions:\n\t" + instr["task_instructions"].replace("\n", "\n\t"))
    return "\n".join(lines)


def user_sim_system_prompt(scenario: dict[str, Any]) -> str:
    return _SIM_SYSTEM_TEMPLATE.format(guidelines=load_guidelines(), instructions=scenario_to_text(scenario))


def derive_opening_message(scenario: dict[str, Any]) -> str:
    """Fallback opening line when a task row carries no ``opening_message``
    (the shipped datasets always include one)."""
    instr = (scenario or {}).get("instructions") or {}
    if isinstance(instr, dict):
        reason = instr.get("reason_for_call")
        if reason:
            return str(reason).strip()
    elif isinstance(instr, str) and instr.strip():
        return instr.strip()
    return "Hi, I need some help with my account."
