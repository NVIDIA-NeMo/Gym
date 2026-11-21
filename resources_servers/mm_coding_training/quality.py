# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The execution-gated quality reward used by the Super visual-coding experiments."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Quality(BaseModel):
    """A complete, finite and internally consistent four-field judge verdict."""

    model_config = ConfigDict(extra="forbid", strict=True)
    quality: float = Field(ge=0, le=1, allow_inf_nan=False)
    task_fidelity: float = Field(ge=0, le=1, allow_inf_nan=False)
    visual_fidelity: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def check_average(self) -> "Quality":
        # Decimal rounding at the inclusive boundary must survive binary float representation.
        if abs(self.quality - (self.task_fidelity + self.visual_fidelity) / 2) > 0.005 + 1e-12:
            raise ValueError("quality must equal the mean of task and visual fidelity")
        return self


class ScoringInput(BaseModel):
    """Facts produced by execution and the trusted judge, never accepted from policy output."""

    model_config = ConfigDict(extra="forbid", strict=True)
    executed: bool
    verdict: Quality | None = None

    @model_validator(mode="after")
    def require_measurement(self) -> "ScoringInput":
        if self.executed != (self.verdict is not None):
            raise ValueError("Successful execution requires a verdict; failed execution must skip the judge")
        return self


def final_reward(result: ScoringInput) -> float:
    """Compute E * (0.2 + 0.8 * mean(task_fidelity, visual_fidelity))."""
    if not result.executed:
        return 0.0
    assert result.verdict is not None
    return 0.2 + 0.8 * (result.verdict.task_fidelity + result.verdict.visual_fidelity) / 2


class QualityVerifier:
    """The deterministic reward calculation, shared with the workload fixture."""

    def verify(self, body: ScoringInput) -> dict[str, float]:
        return {"reward": final_reward(body)}
