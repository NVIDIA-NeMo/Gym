# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# full_ci.py substitutes this template into a temporary root BUILD.pants-full-ci.
experimental_workspace_environment(name="workspace")

package_shell_command(
    name="job",
    environment="full_ci",
    command=__COMMAND__,
    workdir="/",
    # Existing CI has mutable package indexes, system tools, and mounted fixtures.
    # Never replay a successful result across invocations without pinning those.
    cache_scope="session",
    # CI hooks may rewrite tracked files with identical bytes. Watching their
    # mtimes can restart this very command. The adapter supplies the complete
    # GYM_PANTS_INPUT_DIGEST and rejects source changes after execution.
    workspace_invalidation_sources=[],
    extra_env_vars=__ENVIRONMENT__,
    path_env_modify="off",
    output_files=["command-status.txt"],
    output_path="full-ci",
    outputs_match_mode="all",
    log_output=True,
    # The matching GitHub job deadline governs both engines, including setup.
    # Pants defaults to 30 seconds, so keep its ceiling above every job budget.
    timeout=21600,
)
