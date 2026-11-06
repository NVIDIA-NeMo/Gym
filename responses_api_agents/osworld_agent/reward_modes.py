# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map an OSWorld evaluator score and the agent's terminal action to a training reward.

binary     1 when the score reaches the success threshold, else 0 (the default).
raw        the evaluator score itself (partial credit included).
signed_v1  RFC 0039: 1 on success; -1 when the agent declared DONE but the task is not done; 0 otherwise (gave up
           with FAIL, ran out of steps, or stopped without a terminal action). Infeasible tasks need no special
           case: OSWorld's "infeasible" evaluator scores 1 exactly when the agent ends with FAIL, so FAIL earns 1 and
           a DONE on an impossible task earns -1.
Infrastructure failures are not rewards at all: runtime admission masks those samples.
"""

from __future__ import annotations

from typing import Optional


REWARD_MODES = ("binary", "raw", "signed_v1")


def reward_from_score(
    mode: str, score: float, terminal_action: Optional[str], success_threshold: float = 1.0
) -> float:
    if mode == "raw":
        return float(score)
    success = float(score) >= success_threshold
    if mode == "binary":
        return 1.0 if success else 0.0
    if mode == "signed_v1":
        if success:
            return 1.0
        return -1.0 if (terminal_action or "").upper() == "DONE" else 0.0
    raise ValueError(f"Unsupported reward_mode: {mode!r}")
