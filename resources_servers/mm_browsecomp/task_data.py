# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the mm_browsecomp server: one MM-BrowseComp question per row.

Prompt images are already inside ``responses_create_params.input`` as data URLs. The grading fields are
plaintext after preparation, which is why prepared data is gitignored.
"""

from typing import List

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Upstream row id.
    id: int
    question: str
    ground_truth: str
    checklist: List[str]
    category: str
    subtask: str
    level: int
    # The row's id is in data/MMBrowseComp.jsonl, the 224 questions the paper scores.
    in_paper_set: bool
    # A checklist item is tagged video (checklist_property code 2).
    needs_video: bool
