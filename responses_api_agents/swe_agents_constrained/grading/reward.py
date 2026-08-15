"""Reward computation for GRPO training.

Two training modes are kept separate because they target different learning objectives
and can interfere if mixed:

  FORMAT    — teach the model to structure output correctly (thinking tags, numbered
               plans, JSON format, TLDR prefix, etc.) while solving tasks correctly.
               Datasets: SWE-bench, HumanEval+, GSM8K, MBPP+, EnvFactory-RL.

  TOOL      — teach the model which tool to call and when, with correct arguments.
               (No active dataset source; reserved for future tool-calling benchmarks.)

Reward formula (shaped multiplicative):

  reward = task_reward × (1 + α × constraint_reward)

  task_reward ∈ [0, 1]      — continuous where verifier supports partial credit
  constraint_reward ∈ [0, 1] — fraction of required constraint elements present
  α ≥ 0                      — boost strength; higher = stronger constraint pressure

Why shaped (not simple task × constraint):
  - Tasks in the calibrated pool have pass rates as low as 5–10 %, so simple
    multiplication would make reward near-zero for hard domains (SWE-bench, MBPP).
  - Shaped reward provides gradient signal from task_reward even when constraint
    is not yet satisfied, giving the model a learning foothold early in training.
  - task_reward = 0 always gives reward = 0, preventing constraint reward hacking.

Default α values:
  FORMAT mode: α = 1.0  (constraint bonus doubles reward when both pass)
  TOOL mode:   α = 0.5  (softer boost — tool selection is already the primary signal)
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class TrainingMode(str, Enum):
    FORMAT = "format"
    TOOL   = "tool"


# Default α per training mode
_DEFAULT_ALPHA: dict[TrainingMode, float] = {
    TrainingMode.FORMAT: 1.0,
    TrainingMode.TOOL:   0.5,
}

# Which task verifiers belong to each mode
VERIFIERS_BY_MODE: dict[TrainingMode, set[str]] = {
    TrainingMode.FORMAT: {"humaneval_runner", "mbpp_runner", "gsm8k_runner", "swebench_runner",
                          "envfactory_runner"},
    TrainingMode.TOOL:   set(),
}


class RewardComponents(NamedTuple):
    task_reward:       float   # raw task score ∈ [0, 1]
    constraint_reward: float   # raw constraint score ∈ [0, 1]
    alpha:             float   # boost coefficient used
    total:             float   # final shaped reward


def compute_reward(
    task_reward: float,
    constraint_reward: float,
    *,
    mode: TrainingMode | None = None,
    alpha: float | None = None,
) -> RewardComponents:
    """Compute shaped reward for one (completion, constraint) pair.

    Args:
        task_reward:       Score from the task verifier ∈ [0, 1].
        constraint_reward: Score from the constraint verifier ∈ [0, 1].
        mode:              Training mode — determines default α when alpha is None.
        alpha:             Override boost coefficient. If None, uses mode default.

    Returns:
        RewardComponents with individual scores and the final total.
    """
    if alpha is None:
        alpha = _DEFAULT_ALPHA.get(mode or TrainingMode.FORMAT, 1.0)

    total = task_reward * (1.0 + alpha * constraint_reward)
    return RewardComponents(
        task_reward=task_reward,
        constraint_reward=constraint_reward,
        alpha=alpha,
        total=total,
    )


def mode_for_verifier(task_verifier: str) -> TrainingMode:
    """Return the training mode that owns a given task verifier."""
    for mode, verifiers in VERIFIERS_BY_MODE.items():
        if task_verifier in verifiers:
            return mode
    return TrainingMode.FORMAT


# ── Partial-credit task scorers ───────────────────────────────────────────────
# Used when the verifier can provide continuous [0,1] signal instead of binary.

def mbpp_partial_score(test_list: list[str], code: str) -> float:
    """Fraction of MBPP test assertions that pass. Requires sandbox execution."""
    import re
    import subprocess
    import sys
    import tempfile
    import textwrap
    from pathlib import Path

    if not test_list:
        return 0.0

    passed = 0
    for assertion in test_list:
        snippet = code + "\n" + assertion
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(textwrap.dedent(snippet))
            fpath = f.name
        try:
            result = subprocess.run(
                [sys.executable, fpath],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                passed += 1
        except Exception:
            pass
        finally:
            Path(fpath).unlink(missing_ok=True)

    return passed / len(test_list)
