# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor task rules shared by the harbor_tasks server and Harbor benchmark provisioning.

The server applies them when it seeds a task; benchmarks/harbor/prepare_utils applies them when it writes rows and
builds images. Keeping them here keeps the two from drifting. This module imports only the standard library and
Harbor, so provisioning can run it in an isolated Harbor environment without Gym's server dependencies.
"""

from pathlib import Path

from harbor.environments.definition import environment_content_hash
from harbor.models.job.config import DatasetConfig
from harbor.models.task.config import NetworkMode, TaskOS
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode
from harbor.tasks.client import TaskClient


COMPOSE_FILE_NAMES = ("docker-compose.yaml", "docker-compose.yml")


async def download_dataset_tasks(dataset: DatasetConfig, *, disable_verification: bool = False) -> list[Path]:
    """Resolve a Harbor dataset to local task directories, downloading registry tasks into Harbor's cache."""
    task_configs = await dataset.get_task_configs(disable_verification=disable_verification)
    downloaded = await TaskClient().download_tasks(
        [config.get_task_id() for config in task_configs],
        overwrite=dataset.overwrite,
        output_dir=dataset.download_dir,
    )
    return downloaded.paths


def builds_image(task: Task) -> bool:
    """Whether the task's sandbox image is built from its environment/Dockerfile rather than prebuilt."""
    return not task.config.environment.docker_image and (task.paths.environment_dir / "Dockerfile").is_file()


def image_reference(task: Task, image_template: str | None) -> str | None:
    """The image a task's sandbox starts from: its own prebuilt image, else the tag of its built environment.

    A built environment is tagged by Harbor's content hash of environment/, so identical environments share one
    image. Returns None when the task has neither, or builds an image but no template is configured.
    """
    if task.config.environment.docker_image:
        return task.config.environment.docker_image
    if image_template is None or not builds_image(task):
        return None
    return image_template.format(environment_hash=environment_content_hash(task.paths.environment_dir))


def unsupported_features(
    task: Task, *, image_template: str | None = None, allow_unenforced_network_policy: bool = False
) -> list[str]:
    """List the task features harbor_tasks cannot reproduce faithfully; empty when the task is supported."""
    unsupported = []
    if task.has_steps:
        unsupported.append("multi-step tasks ([[steps]])")
    if resolve_task_verifier_mode(task.config) == VerifierEnvironmentMode.SEPARATE:
        unsupported.append("separate verifier environments")
    if task.config.environment.os != TaskOS.LINUX:
        unsupported.append(f"{task.config.environment.os.value} tasks")
    if any((task.paths.environment_dir / name).exists() for name in COMPOSE_FILE_NAMES):
        unsupported.append("docker-compose environments")
    elif image_reference(task, image_template) is None:
        if builds_image(task):
            unsupported.append("Dockerfile environments without an image_template")
        else:
            unsupported.append("environments with neither a docker_image nor a Dockerfile")
    if task.config.environment.mcp_servers:
        # Harbor hands these to the agent, which never reads the task, so the task would silently run without them.
        unsupported.append("task-declared MCP servers")
    network_modes = {
        task.config.environment.network_mode,
        task.config.agent.network_mode,
        task.config.verifier.network_mode,
    }
    if network_modes - {None, NetworkMode.PUBLIC} and not allow_unenforced_network_policy:
        unsupported.append("restricted network policies (set allow_unenforced_network_policy to run unenforced)")
    return unsupported
