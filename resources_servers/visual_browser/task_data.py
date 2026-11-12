# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light task schema for the visual-browser resource server."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class WebTaskData(BaseModel):
    """Dataset representation of the model-independent web task envelope."""

    model_config = ConfigDict(extra="allow")

    benchmark: Literal["webarena", "visualwebarena", "webvoyager"]
    task_id: Union[str, int]
    intent: str = ""
    start_urls: List[str] = Field(default_factory=list)
    sites: List[str] = Field(default_factory=list)
    input_images: List[str] = Field(default_factory=list)
    runtime_profile: Literal["visual_browser"] = "visual_browser"
    observation_profile: Optional[Literal["a11y", "screenshot", "som"]] = None
    action_profile: Literal["computer_use"] = "computer_use"
    verifier_profile: Optional[str] = None
    auth_profile: Optional[str] = None
    seed: int = 0
    task_kwargs: Dict[str, Any] = Field(default_factory=dict)
    original_metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskData(BaseModel):
    """Task-owned fields consumed by the visual browser or judge."""

    model_config = ConfigDict(extra="allow")

    web_task: WebTaskData = Field(
        description="Normalized benchmark task used to seed the browser and construct the agent prompt.",
        json_schema_extra={"consumed_by": ["prompt", "verify", "provenance"]},
    )


__all__ = ["TaskData", "WebTaskData"]
