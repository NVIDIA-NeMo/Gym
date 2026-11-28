# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The environment variables that carry the secrets servers sign their sessions with for one run.

A server that signs its sessions with a secret made for the run keeps it in its process environment
(``run_session_secret`` in ``nemo_gym.server_runtime.base_server``), so its uvicorn workers inherit it, and so does
every other process the server starts. A sandbox provider that passes its own environment on to the code it runs
drops these variables first.
"""

from collections.abc import Mapping


# Each server's variable is this prefix and a hash of the server's name.
RUN_SESSION_SECRET_ENV_PREFIX = "NEMO_GYM_RUN_SESSION_SECRET_"


def without_run_session_secrets(environ: Mapping[str, str]) -> dict[str, str]:
    """A copy of ``environ`` without the variables that carry run session secrets."""
    return {name: value for name, value in environ.items() if not name.startswith(RUN_SESSION_SECRET_ENV_PREFIX)}
