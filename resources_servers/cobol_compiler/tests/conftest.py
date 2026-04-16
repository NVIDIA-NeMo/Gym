# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import shutil

import pytest
from setup_cobc import ensure_cobc


_cobc_install_attempted = False
_cobc_available = False


def pytest_configure(config):
    """Run ensure_cobc() once during pytest startup, before collection."""
    global _cobc_install_attempted, _cobc_available  # noqa: PLW0603
    if not _cobc_install_attempted:
        _cobc_install_attempted = True
        try:
            ensure_cobc()
        except Exception:
            pass
        _cobc_available = shutil.which("cobc") is not None


@pytest.fixture(scope="session")
def require_cobc():
    """Skip the test if cobc is not available after install attempt."""
    if not _cobc_available:
        pytest.skip("GnuCOBOL (cobc) not available")
