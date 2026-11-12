# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend task identity: the backend fields every IF receipt binds."""

from resources_servers.agentic_if.task_variants.builder import digest


def backend_task_digest(row: dict) -> str:
    """Bind execution and grading inputs independently of actor prompt overrides.

    Collector routing, timestamps and rollout IDs are deliberately not task
    identity. This is an integrity checksum, never an authentication signature.
    """
    keys = (
        "instance_id",
        "task_id",
        "repo",
        "base_commit",
        "image_name",
        "image_url",
        "pre_commands",
        "f2p_patch",
        "f2p_script",
        "sandbox_spec",
        "docker_image",
        "verifier_image",
        "oracle_image",
        "workdir",
        "workspace_root",
        "environment",
        "agent",
        "agent_user",
        "verifier",
        "patch",
        "test_patch",
        "install_config",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "task_name",
        "task_folder",
        # Rows with package references are rejected, but sealed receipts hash these keys as null.
        "task_ref",
        "dataset_ref",
    )
    return digest({key: row.get(key) for key in keys})
