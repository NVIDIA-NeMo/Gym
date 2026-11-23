# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable artifact contracts for harness observability.

Readers supply native Gym JSON;
artifact conformance is deliberately separate from behavioral qualification.

This package is the harness checker's public API.
Code outside ``nemo_gym/`` imports the checker's names from ``nemo_gym.rollouts.harness_capabilities``.
They are not part of ``nemo_gym.rollouts``, whose names are the rollout collector's.
"""

__version__ = "0.2.0"

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.rollouts.harness_capabilities.behavior import inspect_behavior
    from nemo_gym.rollouts.harness_capabilities.checker import NAMES, EvidenceScope, inspect_record
    from nemo_gym.rollouts.harness_capabilities.checks import PriorityTier
    from nemo_gym.rollouts.harness_capabilities.cli import digest_file, inspect_bundle, json_rows, main
    from nemo_gym.rollouts.harness_capabilities.compaction import inspect_compaction
    from nemo_gym.rollouts.harness_capabilities.health import inspect_health
    from nemo_gym.rollouts.harness_capabilities.results import Results, gate_passes, render_matrices


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "EvidenceScope": "checker",
    "NAMES": "checker",
    "PriorityTier": "checks",
    "Results": "results",
    "digest_file": "cli",
    "gate_passes": "results",
    "inspect_behavior": "behavior",
    "inspect_bundle": "cli",
    "inspect_compaction": "compaction",
    "inspect_health": "health",
    "inspect_record": "checker",
    "json_rows": "cli",
    "main": "cli",
    "render_matrices": "results",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
