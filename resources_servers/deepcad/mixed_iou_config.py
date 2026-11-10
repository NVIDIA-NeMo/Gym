# SPDX-License-Identifier: Apache-2.0
"""Lightweight schema; importing the Gym server does not load geometry libraries."""

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MixedIoUWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brep_iou: float = Field(default=0.30, ge=0, le=1, allow_inf_nan=False)
    octree_iou: float = Field(default=0.30, ge=0, le=1, allow_inf_nan=False)
    boundary_iou: float = Field(default=0.40, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def unit_sum(self):
        if not math.isclose(sum(self.model_dump().values()), 1.0, abs_tol=1e-9):
            raise ValueError("Mixed IoU weights must sum to one")
        return self


class MixedIoUConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weights: MixedIoUWeights = Field(default_factory=MixedIoUWeights)
    octree_depth: int = Field(default=5, ge=0, le=8)
    boundary_band_width: float = Field(default=1 / 64, gt=0, le=1, allow_inf_nan=False)

    def validate_resolution(self, resolution: int) -> None:
        if not 8 <= resolution <= 256:
            raise ValueError("Voxel resolution must be between 8 and 256")
        if 2**self.octree_depth > resolution:
            raise ValueError("Octree depth exceeds the voxel resolution")
        if self.boundary_band_width * resolution < 1:
            raise ValueError("Boundary band must span at least one voxel")
