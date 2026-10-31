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
import os
import shutil

import pytest


def pytest_configure(config):
    """Install the browser once, so the episode tests run on a fresh runner instead of skipping."""
    from resources_servers.interactive_browser.browser._chromium import ensure_chromium
    from resources_servers.vab_css.chromium_host import ensure_chromium_host_libraries

    ensure_chromium()
    # Elsewhere the chromium_host fixture skips the browser tests with its diagnostic.
    if os.geteuid() == 0 and shutil.which("apt-get") is not None:
        ensure_chromium_host_libraries()


@pytest.fixture(scope="session")
def chromium_host() -> None:
    """Skip only when Chromium lacks system libraries and this runner may not install them."""
    from resources_servers.vab_css.chromium_host import missing_host_dependencies

    if missing_host_dependencies() is not None:
        pytest.skip("Chromium lacks system libraries; run `python -m playwright install-deps chromium` as root")
