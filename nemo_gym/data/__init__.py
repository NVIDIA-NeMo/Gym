# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Task data: datasets and their preparation, prompts, task rows and their materialization, and task formats such as Harbor."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.data.dataset_preparation import TrainDataProcessor
    from nemo_gym.data.gitlab_utils import download_jsonl_dataset
    from nemo_gym.data.prompt import PromptConfig, apply_prompt_to_row, fill_prompt, load_prompt_config
    from nemo_gym.data.task_data import (
        TaskDataValidator,
        load_task_data_schema,
        normalize_task_fields,
        validate_jsonl_rows,
    )
    from nemo_gym.data.task_materialization import materialize_task
    from nemo_gym.data.vision_input import build_image_input


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "PromptConfig": "prompt",
    "TaskDataValidator": "task_data",
    "TrainDataProcessor": "dataset_preparation",
    "apply_prompt_to_row": "prompt",
    "build_image_input": "vision_input",
    "download_jsonl_dataset": "gitlab_utils",
    "fill_prompt": "prompt",
    "load_prompt_config": "prompt",
    "load_task_data_schema": "task_data",
    "materialize_task": "task_materialization",
    "normalize_task_fields": "task_data",
    "validate_jsonl_rows": "task_data",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
