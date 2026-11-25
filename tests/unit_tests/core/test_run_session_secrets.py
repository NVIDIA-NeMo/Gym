# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.core.run_session_secrets import RUN_SESSION_SECRET_ENV_PREFIX, without_run_session_secrets
from nemo_gym.core.run_token import RUN_TOKEN_ENV


def test_without_run_session_secrets_drops_only_their_variables() -> None:
    environ = {
        "PATH": "/usr/bin",
        RUN_SESSION_SECRET_ENV_PREFIX + "0123456789ABCDEF": "secret",
        "NEMO_GYM_RUN_SESSION": "not a secret variable",
    }

    assert without_run_session_secrets(environ) == {
        "PATH": "/usr/bin",
        "NEMO_GYM_RUN_SESSION": "not a secret variable",
    }
    # A copy: the mapping it reads keeps them.
    assert environ[RUN_SESSION_SECRET_ENV_PREFIX + "0123456789ABCDEF"] == "secret"


def test_without_run_session_secrets_drops_the_runs_token() -> None:
    environ = {"PATH": "/usr/bin", RUN_TOKEN_ENV: "the-token", RUN_TOKEN_ENV + "_NOT": "kept"}

    assert without_run_session_secrets(environ) == {"PATH": "/usr/bin", RUN_TOKEN_ENV + "_NOT": "kept"}
