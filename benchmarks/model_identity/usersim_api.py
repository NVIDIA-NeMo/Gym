# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every name the benchmark takes from UserSim, so a change of the pinned revision touches one file.

Import it only where UserSim is installed: the scoring and server code, and preparation's worker.
"""

from usersim.engine.external import construct_probe_episode
from usersim.engine.probes.identity_disclosure.scorer import score_identity_disclosure_trajectory
from usersim.engine.probes.identity_disclosure.spec import load_identity_spec, reset_identity_spec_cache
from usersim.extension import ConversationSimulatorConfig
from usersim.taxonomy.eval_cell import scorer_state


__all__ = [
    "ConversationSimulatorConfig",
    "construct_probe_episode",
    "load_identity_spec",
    "reset_identity_spec_cache",
    "score_identity_disclosure_trajectory",
    "scorer_state",
]
