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
    from nemo_gym.data.gitlab_utils import download_jsonl_dataset
    from nemo_gym.data.prompt import PromptConfig, apply_prompt_to_row, fill_prompt, load_prompt_config
    from nemo_gym.data.task_data import normalize_task_fields
    from nemo_gym.data.tasks.harbor import DIGEST_KEY
    from nemo_gym.data.tasks.harbor.compose_overlay import (
        ComposeOverlayError,
        HarborComposeOverlay,
        merged_mcp_servers,
        overlay_files,
        wrap_first_instruction,
    )
    from nemo_gym.data.tasks.harbor.dataset_config import DATASET_FILE
    from nemo_gym.data.tasks.harbor.digest import is_ignored
    from nemo_gym.data.tasks.harbor.dockerfile import DEFAULT_SHELL, RECORD_VERSION, OverlayRun
    from nemo_gym.data.tasks.harbor.errors import HarborTaskError
    from nemo_gym.data.tasks.harbor.image_configs import (
        check_task_compose_fields,
        check_task_compose_services,
        check_task_sidecar_extra_hosts,
        load_compose_document,
        plan_task_compose_volumes,
        reusable_record,
        task_declares_gpus,
        without_restated_main_build,
    )
    from nemo_gym.data.tasks.harbor.materialize import AGENT_TIMEOUT_METADATA_KEY, STEP_METADATA_KEY, HarborTaskInput
    from nemo_gym.data.tasks.harbor.models import (
        HarborArtifact,
        HarborEnvironment,
        HarborHealthcheck,
        HarborMCPServer,
        HarborTaskStep,
        HarborVerifier,
    )
    from nemo_gym.data.tasks.harbor.prepare import ORACLE_AGENT
    from nemo_gym.data.tasks.harbor.python_verifier import VERIFIER_FILE, VerifierShim, write_shim
    from nemo_gym.data.tasks.harbor.records import BUILT_IMAGES_FILE, MAIN_SERVICE
    from nemo_gym.data.tasks.harbor.registry import ImageConfigError
    from nemo_gym.data.tasks.harbor.task import HarborTask, HarborTurn, load_task
    from nemo_gym.data.tasks.harbor.task_data import TaskDataError, TaskDataSchema, check_row_matches_folder
    from nemo_gym.data.tasks.harbor.verifier_env import TESTS_DOCKERFILE
    from nemo_gym.data.vision_input import build_image_input


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "AGENT_TIMEOUT_METADATA_KEY": "tasks.harbor.materialize",
    "BUILT_IMAGES_FILE": "tasks.harbor.records",
    "ComposeOverlayError": "tasks.harbor.compose_overlay",
    "DATASET_FILE": "tasks.harbor.dataset_config",
    "DEFAULT_SHELL": "tasks.harbor.dockerfile",
    "DIGEST_KEY": "tasks.harbor",
    "HarborArtifact": "tasks.harbor.models",
    "HarborComposeOverlay": "tasks.harbor.compose_overlay",
    "HarborEnvironment": "tasks.harbor.models",
    "HarborHealthcheck": "tasks.harbor.models",
    "HarborMCPServer": "tasks.harbor.models",
    "HarborTask": "tasks.harbor.task",
    "HarborTaskError": "tasks.harbor.errors",
    "HarborTaskInput": "tasks.harbor.materialize",
    "HarborTaskStep": "tasks.harbor.models",
    "HarborTurn": "tasks.harbor.task",
    "HarborVerifier": "tasks.harbor.models",
    "ImageConfigError": "tasks.harbor.registry",
    "MAIN_SERVICE": "tasks.harbor.records",
    "ORACLE_AGENT": "tasks.harbor.prepare",
    "OverlayRun": "tasks.harbor.dockerfile",
    "PromptConfig": "prompt",
    "RECORD_VERSION": "tasks.harbor.dockerfile",
    "STEP_METADATA_KEY": "tasks.harbor.materialize",
    "TESTS_DOCKERFILE": "tasks.harbor.verifier_env",
    "TaskDataError": "tasks.harbor.task_data",
    "TaskDataSchema": "tasks.harbor.task_data",
    "VERIFIER_FILE": "tasks.harbor.python_verifier",
    "VerifierShim": "tasks.harbor.python_verifier",
    "apply_prompt_to_row": "prompt",
    "build_image_input": "vision_input",
    "check_row_matches_folder": "tasks.harbor.task_data",
    "check_task_compose_fields": "tasks.harbor.image_configs",
    "check_task_compose_services": "tasks.harbor.image_configs",
    "check_task_sidecar_extra_hosts": "tasks.harbor.image_configs",
    "download_jsonl_dataset": "gitlab_utils",
    "fill_prompt": "prompt",
    "is_ignored": "tasks.harbor.digest",
    "load_compose_document": "tasks.harbor.image_configs",
    "load_prompt_config": "prompt",
    "load_task": "tasks.harbor.task",
    "merged_mcp_servers": "tasks.harbor.compose_overlay",
    "normalize_task_fields": "task_data",
    "overlay_files": "tasks.harbor.compose_overlay",
    "plan_task_compose_volumes": "tasks.harbor.image_configs",
    "reusable_record": "tasks.harbor.image_configs",
    "task_declares_gpus": "tasks.harbor.image_configs",
    "without_restated_main_build": "tasks.harbor.image_configs",
    "wrap_first_instruction": "tasks.harbor.compose_overlay",
    "write_shim": "tasks.harbor.python_verifier",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
