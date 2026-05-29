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
from unittest.mock import patch

import pytest

import nemo_gym.openai_utils as openai_utils
import nemo_gym.server_utils as server_utils


@pytest.fixture(autouse=True)
def _manage_global_aiohttp_client():
    """Prevent Hydra CLI parsing, disable retries, and reset the global aiohttp client between tests.

    get_global_aiohttp_client() calls get_global_config_dict() which parses Hydra CLI args.
    Under pytest, those args cause argparse.SystemExit. Mocking returns default config values.
    MAX_NUM_TRIES=1 disables retries so tests fail fast on rate limits or server errors.
    """
    with (
        patch.object(server_utils, "get_global_config_dict", return_value={}),
        patch.object(openai_utils, "MAX_NUM_TRIES", 1),
        patch.object(openai_utils, "RETRY_ERROR_CODES", []),
    ):
        yield
    server_utils._GLOBAL_AIOHTTP_CLIENT = None
