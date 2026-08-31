# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for WebVoyager screenshot-trajectory judging."""

from typing import Any, Dict, List, Optional

from pydantic import Field

from resources_servers.browsergym_web.task_data import TaskData as WebTaskRow


class TaskData(WebTaskRow):
    final_answer: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    screenshots: List[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    page_urls: List[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional stored rollout response used to recover judge evidence during reverification.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
