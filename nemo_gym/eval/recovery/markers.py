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
"""The strings the auto-resume chain scripts and the commands they run use to leave a result for the next segment.

They are the contract between the chain (``nemo_gym.orchestration.executors``, which renders the scripts) and the
code the scripts run, so they live below both: the chain imports them from here.
"""

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)

# The driver step's environment names the file, <job dir>/.gym_driver_failure.<allocation>, where a driver failure
# that the next segment would repeat leaves its registered kind (see the guard in resume_script.py).
DRIVER_FAILURE_ENV = "GYM_DRIVER_FAILURE_FILE"


def record_failure_kind(error: BaseException) -> None:
    """Leave the registered kind ``error`` carries in the failure file the driver step names, for the next guard.

    The driver step of a resumable job names the file in ``DRIVER_FAILURE_ENV``; elsewhere this does nothing.
    """
    kind = getattr(error, "failure_kind", None)
    path = os.environ.get(DRIVER_FAILURE_ENV)
    if not isinstance(kind, str) or not path:
        return
    try:
        Path(path).write_text(kind + "\n")
    except OSError as write_error:
        # The next segment then reads the saved state as it would without the kind.
        logger.warning("Cannot record failure kind %s in %s: %s", kind, path, write_error)
