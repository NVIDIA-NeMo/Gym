# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flat fields emitted by the AA-Briefcase-Lite prepare script.

The verifier requires only task_id. The remaining fields feed the Stirrup task
strategy during generation; deliverables_dir is injected at verify time.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    week: Optional[int] = None
    dataset_dir: Optional[str] = None
    dataset_revision: Optional[str] = None
    task_md_path: Optional[str] = None
    deliverable_filenames: Optional[List[str]] = None
    shared_files: Optional[List[str]] = None
    week_files: Optional[List[str]] = None
    scenario_overview_path: Optional[str] = None
    week_overview_path: Optional[str] = None
