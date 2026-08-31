# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema shared by Gym web-browser environments.

The dataset row carries one normalized ``web_task`` envelope. Framework-owned
``responses_create_params`` is intentionally absent from this schema. The
nested model mirrors ``nemo_gym.web.models.WebTask`` without importing the web
runtime, keeping dataset discovery and validation dependency-light.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class WebTaskData(BaseModel):
    """Dependency-light representation of the normalized web task envelope."""

    model_config = ConfigDict(extra="allow")

    benchmark: Literal["webarena", "visualwebarena", "webvoyager"]
    task_id: Union[str, int]
    intent: str = ""
    start_urls: List[str] = Field(default_factory=list)
    sites: List[str] = Field(default_factory=list)
    input_images: List[str] = Field(default_factory=list)
    runtime_profile: Literal["browsergym", "native_visual", "selenium"] = "browsergym"
    observation_profile: Optional[Literal["a11y", "screenshot", "som"]] = None
    action_profile: Literal["browsergym_highlevel", "native_toolcall", "webvoyager_legacy"] = "browsergym_highlevel"
    verifier_profile: Optional[str] = None
    auth_profile: Optional[str] = None
    seed: int = 0
    task_kwargs: Dict[str, Any] = Field(default_factory=dict)
    original_metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskData(BaseModel):
    """Task-owned fields consumed by a Gym web runtime or judge."""

    model_config = ConfigDict(extra="allow")

    web_task: WebTaskData = Field(
        description="Normalized benchmark task used to seed the browser and construct the agent prompt.",
        json_schema_extra={"consumed_by": ["prompt", "verify", "provenance"]},
    )
