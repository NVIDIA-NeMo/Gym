# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt rendering for domain generation."""


def domain_followup_prompt(prompt: str, known_domain_names: list[str]) -> str:
    return (
        prompt + f"\n\nPreviously brainstormed domains: {known_domain_names}.\n"
        "Do not repeat these domains. Try looking for other domains or find specific sub-domains."
    )
