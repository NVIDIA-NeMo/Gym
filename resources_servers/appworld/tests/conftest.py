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
"""Install AppWorld before collection so `skipif` markers see it.

`pytest.mark.skipif` is evaluated at module import time, i.e. during collection
and before any fixture runs — so the integration test's "is AppWorld available?"
predicate has to be answered here. Failures are swallowed: an environment
without network access simply skips the integration test instead of erroring the
whole suite.
"""

import os

import pytest

from nemo_gym import server_utils
from resources_servers.appworld.setup_appworld import ensure_appworld


@pytest.fixture(autouse=True)
def fresh_global_aiohttp_client():
    """Rebind gym's aiohttp singleton to each test's event loop.

    ``get_global_aiohttp_client`` caches one session process-wide, but
    pytest-asyncio gives every test a fresh event loop — so a session created in
    one test raises "Event loop is closed" in the next. Tests that talk to real
    worker processes need a client bound to the loop they are running on.
    """
    server_utils._GLOBAL_AIOHTTP_CLIENT = None
    yield
    server_utils._GLOBAL_AIOHTTP_CLIENT = None


def pytest_configure(config):  # noqa: ARG001 — pytest hook signature
    # The end-to-end test talks to its worker over gym's global aiohttp client,
    # which otherwise resolves its config through hydra and would try to parse
    # pytest's argv. `gym env test` already sets this; keep bare `pytest` working.
    os.environ.setdefault("NEMO_GYM_CONFIG_DICT", "{}")
    try:
        ensure_appworld()
    except Exception:  # noqa: BLE001 — offline or sandboxed: the marker will skip
        pass
