# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Harbor task folders and hub references into NeMo Gym materialized tasks.

Harbor is the input format; Gym owns execution. A task is a folder holding
``task.toml``, ``instruction.md``, ``environment/`` and ``tests/``. A dataset is a
folder of task folders. ``harbor:<dataset>[@<version>]`` names a dataset in the
Harbor registry, which this package fetches into ``environments/harbor/`` in the Gym checkout, or
under the working directory when Gym is installed as a package (``gym dataset pull``;
:func:`nemo_gym.data.tasks.harbor.hub.datasets_dir`); a committed entry there declares its dataset as
``[gym] source`` and is pulled into its own folder (:mod:`nemo_gym.data.tasks.harbor.entries`).

This package is the Harbor format's public API.
Code outside ``nemo_gym/`` imports Harbor names from ``nemo_gym.data.tasks.harbor``, not from ``nemo_gym.data``.
"""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.data.tasks.harbor.compose_overlay import (
        ComposeOverlayError,
        HarborComposeOverlay,
        merged_mcp_servers,
        overlay_files,
        wrap_first_instruction,
    )
    from nemo_gym.data.tasks.harbor.dataset_config import DATASET_FILE
    from nemo_gym.data.tasks.harbor.digest import (
        DATASET_DIGEST_KEY,
        DIGEST_KEY,
        content_hash,
        context_digest,
        is_ignored,
    )
    from nemo_gym.data.tasks.harbor.dockerfile import DEFAULT_SHELL, RECORD_VERSION, OverlayRun, OverlayWorkdir
    from nemo_gym.data.tasks.harbor.errors import HarborTaskError
    from nemo_gym.data.tasks.harbor.image_configs import (
        check_task_compose_fields,
        check_task_compose_services,
        check_task_sidecar_extra_hosts,
        load_compose_document,
        plan_task_compose_volumes,
        record_compose_images,
        reusable_record,
        task_declares_gpus,
        without_restated_main_build,
    )
    from nemo_gym.data.tasks.harbor.materialize import (
        AGENT_TIMEOUT_METADATA_KEY,
        STEP_METADATA_KEY,
        HarborTaskInput,
        materialize_task,
    )
    from nemo_gym.data.tasks.harbor.missing_reward import (
        MISSING_REWARD_MODULE,
        MISSING_REWARD_OUTCOMES,
        MissingRewardClassifier,
        MissingRewardClassifierError,
        MissingRewardEvidence,
        MissingRewardVerdict,
        load_missing_reward_classifier,
    )
    from nemo_gym.data.tasks.harbor.models import (
        HarborArtifact,
        HarborEnvironment,
        HarborHealthcheck,
        HarborMCPServer,
        HarborTaskConfig,
        HarborTaskStep,
        HarborVerifier,
    )
    from nemo_gym.data.tasks.harbor.prepare import ORACLE_AGENT
    from nemo_gym.data.tasks.harbor.publish import answer_write_command
    from nemo_gym.data.tasks.harbor.python_verifier import VERIFIER_FILE, VerifierShim, write_shim
    from nemo_gym.data.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE, MAIN_SERVICE
    from nemo_gym.data.tasks.harbor.registry import ImageConfigError
    from nemo_gym.data.tasks.harbor.task import HarborTask, HarborTurn, built_service_images, discover_tasks, load_task
    from nemo_gym.data.tasks.harbor.task_data import (
        TASK_DATA_FILE,
        TASK_DATA_MODULE,
        TaskDataError,
        TaskDataSchema,
        check_row_matches_folder,
        load_task_data_schema,
    )
    from nemo_gym.data.tasks.harbor.verifier_env import TESTS_DOCKERFILE


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "AGENT_TIMEOUT_METADATA_KEY": "materialize",
    "BUILT_IMAGES_FILE": "records",
    "ComposeOverlayError": "compose_overlay",
    "DATASET_DIGEST_KEY": "digest",
    "DATASET_FILE": "dataset_config",
    "DEFAULT_SHELL": "dockerfile",
    "DIGEST_KEY": "digest",
    "HarborArtifact": "models",
    "HarborComposeOverlay": "compose_overlay",
    "HarborEnvironment": "models",
    "HarborHealthcheck": "models",
    "HarborMCPServer": "models",
    "HarborTask": "task",
    "HarborTaskConfig": "models",
    "HarborTaskError": "errors",
    "HarborTaskInput": "materialize",
    "HarborTaskStep": "models",
    "HarborTurn": "task",
    "HarborVerifier": "models",
    "IMAGE_CONFIGS_FILE": "records",
    "ImageConfigError": "registry",
    "MAIN_SERVICE": "records",
    "MISSING_REWARD_MODULE": "missing_reward",
    "MISSING_REWARD_OUTCOMES": "missing_reward",
    "MissingRewardClassifier": "missing_reward",
    "MissingRewardClassifierError": "missing_reward",
    "MissingRewardEvidence": "missing_reward",
    "MissingRewardVerdict": "missing_reward",
    "ORACLE_AGENT": "prepare",
    "OverlayRun": "dockerfile",
    "OverlayWorkdir": "dockerfile",
    "RECORD_VERSION": "dockerfile",
    "STEP_METADATA_KEY": "materialize",
    "TASK_DATA_FILE": "task_data",
    "TASK_DATA_MODULE": "task_data",
    "TESTS_DOCKERFILE": "verifier_env",
    "TaskDataError": "task_data",
    "TaskDataSchema": "task_data",
    "VERIFIER_FILE": "python_verifier",
    "VerifierShim": "python_verifier",
    "answer_write_command": "publish",
    "built_service_images": "task",
    "check_row_matches_folder": "task_data",
    "check_task_compose_fields": "image_configs",
    "check_task_compose_services": "image_configs",
    "check_task_sidecar_extra_hosts": "image_configs",
    "content_hash": "digest",
    "context_digest": "digest",
    "discover_tasks": "task",
    "is_ignored": "digest",
    "load_compose_document": "image_configs",
    "load_missing_reward_classifier": "missing_reward",
    "load_task": "task",
    "load_task_data_schema": "task_data",
    "materialize_task": "materialize",
    "merged_mcp_servers": "compose_overlay",
    "overlay_files": "compose_overlay",
    "plan_task_compose_volumes": "image_configs",
    "record_compose_images": "image_configs",
    "reusable_record": "image_configs",
    "task_declares_gpus": "image_configs",
    "without_restated_main_build": "image_configs",
    "wrap_first_instruction": "compose_overlay",
    "write_shim": "python_verifier",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
