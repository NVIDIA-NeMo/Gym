# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the terminal_bench_2_1 server.

Rows carry four sandbox coordinates top-level (no verifier_metadata), the fourth optional:
``TerminalBench21VerifyRequest`` (app.py) requires ``task_name``, ``docker_image``, and
``task_folder`` as ``str``, so this schema does too, and accepts an optional ``agent_user``.
seed_session() starts the evaluation sandbox from ``docker_image`` (tagging it with
``task_name``) and echoes ``agent_user`` to the agent harness; verify() uploads the task's
``tests/`` (and, when config.is_verifying_golden_patch=true, ``solution/``) from
``task_folder`` into the sandbox and runs the test script there as the image default, running
the golden ``solve.sh`` as ``agent_user``.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_name: str = Field(
        description=(
            "Terminal-Bench 2.1 task id, e.g. 'terminal-bench/path-tracing'; also keys sandbox "
            "metadata (instance_id) and is echoed in the verify response."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    docker_image: str = Field(
        description="Docker image the evaluation sandbox is started from.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_folder: str = Field(
        description=(
            "Repo-relative path to the task directory (under benchmarks/terminal_bench_2_1/); "
            "verify() uploads its tests/ (and solution/ in golden-patch mode) into the sandbox."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    agent_user: str | int | None = Field(
        default=None,
        description=(
            "Identity the agent harness runs as inside the task sandbox (an account name for `su`, or a uid; "
            "mirrors Harbor task.toml [agent] user). None keeps the image default; 'root' or 0 is the explicit "
            "per-row image default and beats the agent lane's default. The verifier always runs as the image "
            "default (root on the supported images), so a non-root agent_user needs a root-default image with "
            "that account. Digit-only strings are normalized to int by the server."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
