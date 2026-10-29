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
"""Whether this process is a Gym server worker, and the environment variables that tell it.

The main server process sets these variables before uvicorn starts its workers, so each worker knows its role and the entrypoint it serves.
"""

from os import environ, getenv

from nemo_gym.config.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME


def _has_injected_global_config_env() -> bool:
    return getenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME) is not None


# These environment variables are the ONLY environment variables that Gym uses. Please do not set these, they are only used here to pass information
# from main proc to child procs under FastAPI/uvicorn parallelism
IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME = "IS_NEMO_GYM_FASTAPI_WORKER"
IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME = "IS_NEMO_GYM_FASTAPI_ENTRYPOINT"
NEMO_GYM_FASTAPI_NUM_WORKERS = "NEMO_GYM_FASTAPI_NUM_WORKERS"


def is_nemo_gym_fastapi_worker() -> bool:  # pragma: no cover
    return getenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME) == "1"


def set_is_nemo_gym_fastapi_worker() -> None:  # pragma: no cover
    environ[IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME] = "1"


def is_nemo_gym_fastapi_entrypoint(file: str) -> bool:  # pragma: no cover
    return is_nemo_gym_fastapi_worker() and file.endswith(getenv(IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME))


def set_is_nemo_gym_fastapi_entrypoint(file: str) -> None:  # pragma: no cover
    environ[IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME] = file


def get_nemo_gym_fastapi_num_workers() -> int:  # pragma: no cover
    return int(getenv(NEMO_GYM_FASTAPI_NUM_WORKERS, "1"))


def set_nemo_gym_fastapi_num_workers(num_workers: int) -> None:  # pragma: no cover
    environ[NEMO_GYM_FASTAPI_NUM_WORKERS] = str(num_workers)
