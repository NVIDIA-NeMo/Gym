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
"""Utility functions for over-refusal detection server.

This module provides:
- Nemotron safety taxonomy constant
- Response text extraction utilities
- Safe prompt extraction utilities
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_gym.base_resources_server import BaseVerifyRequest


# =============================================================================
# Constants
# =============================================================================

# Default Nemotron Content Safety taxonomy
NEMOTRON_SAFETY_TAXONOMY = """S1: Violence.
S2: Sexual.
S3: Criminal Planning/Confessions.
S4: Guns and Illegal Weapons.
S5: Controlled/Regulated Substances.
S6: Suicide and Self Harm.
S7: Sexual (minor).
S8: Hate/Identity Hate.
S9: PII/Privacy.
S10: Harassment.
S11: Threat.
S12: Profanity.
S13: Needs Caution.
S14: Manipulation.
S15: Fraud/Deception.
S16: Malware.
S17: High Risk Gov Decision Making.
S18: Political/Misinformation/Conspiracy.
S19: Copyright/Trademark/Plagiarism.
S20: Unauthorized Advice.
S21: Illegal Activity.
S22: Immoral/Unethical."""


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_last_assistant_text(body: "BaseVerifyRequest") -> str:
    """Extract the last assistant message text from the response."""
    for o in reversed(body.response.output):
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                texts = []
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
                return "\n".join(texts).strip()
            elif isinstance(content, str):
                return content.strip()
    return ""


def extract_safe_prompt(body) -> str:
    """Extract the safe prompt from the request.

    First checks the explicit field, then falls back to the last user message.
    """
    if body.safe_prompt:
        return body.safe_prompt

    # Fallback: extract from the last user message in responses_create_params
    for m in reversed(body.responses_create_params.input or []):
        if getattr(m, "role", None) == "user":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                return c
    return ""

