# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the SWE-bench Pro v2 resources server."""

from pydantic import Field

from resources_servers.swebench_pro.task_data import TaskData as SWEBenchProTaskData


class TaskData(SWEBenchProTaskData):
    """A v1 SWE-bench Pro row plus the v2 image and per-task verifier files."""

    docker_image: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    test_sh: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    test_config: str = Field(json_schema_extra={"consumed_by": ["verify"]})
