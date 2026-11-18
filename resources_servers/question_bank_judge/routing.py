# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map the question bank's domain labels to their configured graders."""

from typing import Literal


_STEM_DOMAINS = {
    "biology",
    "chemistry / materials",
    "engineering / computer science",
    "medicine / health",
    "physics",
}
_DOMAIN_ALIASES = {"phys": "physics", "bio": "biology", "chem": "chemistry / materials", "math": "mathematics"}


def canonical_domain(value: str) -> str:
    label = value.strip().casefold()
    return _DOMAIN_ALIASES.get(label, label)


def grading_route(domain: object) -> Literal["math", "semantic"] | None:
    """Return the configured route for an eligible domain, including recorded aliases."""
    label = canonical_domain(domain) if isinstance(domain, str) else ""
    if label == "mathematics":
        return "math"
    if label in _STEM_DOMAINS:
        return "semantic"
    return None
