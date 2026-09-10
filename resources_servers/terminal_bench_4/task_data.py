# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the terminal_bench_4 server.

Rows carry the four sandbox coordinates top-level (no verifier_metadata), mirroring
``TerminalBench4VerifyRequest`` in app.py: seed_session() starts the AGENT sandbox from
``docker_image``; verify() collects the artifacts declared in ``<task_folder>/task.toml``, tears
the agent sandbox down, starts the VERIFIER sandbox from ``verifier_docker_image`` (a prebuilt
image of ``<task_folder>/tests/Dockerfile`` with ``/tests`` baked in), re-materializes the artifacts
there and runs ``/tests/test.sh``.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_name: str = Field(
        description=(
            "Terminal-Bench 4.0 task id, e.g. 'terminal-bench/cad-model'; keys sandbox metadata "
            "(instance_id) and is echoed in the verify response."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    docker_image: str = Field(
        description=(
            "Agent (environment) image the agent sandbox is started from, e.g. "
            "'harborframework/terminal-bench:cad-model-environment-v4.0.0'."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    verifier_docker_image: str = Field(
        description=(
            "Verifier image built from the task's tests/Dockerfile with /tests baked in, e.g. "
            "'harborframework/terminal-bench:cad-model-verifier-v4.0.0'; the verifier sandbox is started from it."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_folder: str = Field(
        description=(
            "Path to the task directory holding task.toml (artifacts, verifier timeout and resources) and, "
            "for golden-patch mode, solution/. Absolute, or relative to the Gym checkout."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    docker_image_digest: Optional[str] = Field(
        default=None,
        description="Registry digest the agent image tag resolved to when the row was built (provenance only).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    verifier_docker_image_digest: Optional[str] = Field(
        default=None,
        description="Registry digest the verifier image tag resolved to when the row was built (provenance only).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
