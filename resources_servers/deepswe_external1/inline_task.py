# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.deepswe.task_schema import (
    AgentConfig,
    EnvironmentConfig,
    NetworkMode,
    Task,
    TaskConfig,
    VerifierCollectConfig,
    VerifierConfig,
    VerifierEnvironmentMode,
)
from resources_servers.deepswe_external1.task_data import TaskData


class InlineTask(Task):
    """In-memory view for DeepSWE's collector and grader, without task-store paths or I/O."""

    def __init__(self, data: TaskData) -> None:
        self.data = data
        agent, verifier = data.agent, data.verifier
        self.config = TaskConfig(
            metadata={"task_id": data.task_id, "base_commit_hash": data.base_commit},
            environment=EnvironmentConfig(
                docker_image=data.image,
                cpus=agent.cpus,
                memory_mb=agent.memory_mb,
                storage_mb=agent.storage_mb,
                env=agent.env,
            ),
            agent=AgentConfig(timeout_sec=agent.timeout_sec, network_mode=NetworkMode.NO_NETWORK),
            verifier=VerifierConfig(
                timeout_sec=verifier.timeout_sec,
                network_mode=NetworkMode.NO_NETWORK,
                environment_mode=VerifierEnvironmentMode.SEPARATE,
                environment=EnvironmentConfig(
                    docker_image=data.verifier_image,
                    cpus=verifier.cpus,
                    memory_mb=verifier.memory_mb,
                    storage_mb=verifier.storage_mb,
                    env=verifier.env,
                ),
                collect=[
                    VerifierCollectConfig(
                        command=(
                            "set -eu; cd /app; mkdir -p /logs/artifacts; "
                            "git config --global --add safe.directory /app; "
                            "git diff --binary --no-ext-diff --no-textconv --no-color "
                            f"{data.base_commit} HEAD -- . > /logs/artifacts/model.patch"
                        ),
                        timeout_sec=data.collect_timeout_sec,
                    )
                ],
            ),
        )
