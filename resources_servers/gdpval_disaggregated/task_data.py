# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the gdpval_disaggregated server.

Dataset rows carry no task fields of their own: each row is only ``responses_create_params``, and the
server looks the task up by ``responses_create_params.metadata.task_id``. The prompt, reference files and
rubric for that ID live in the server-side task file named by ``dataset_path`` (``data/tasks.jsonl``),
which the seed stages into the sandbox and the judge reads after the sandbox stops, so the rubric never
reaches the policy.
"""

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="forbid")
