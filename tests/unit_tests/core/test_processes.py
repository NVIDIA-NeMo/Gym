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
import subprocess
import sys

import pytest

from nemo_gym.core.processes import process_may_be_running


def test_this_process_is_running() -> None:
    assert process_may_be_running(os.getpid()) is True


def test_a_finished_process_is_not_running() -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()

    assert process_may_be_running(process.pid) is False


@pytest.mark.parametrize("pid", [10**30, -(10**30)])
def test_a_number_no_process_can_have_counts_as_running(pid: int) -> None:
    # os.kill raises OverflowError for these; doubt keeps whatever the caller would remove.
    assert process_may_be_running(pid) is True


def test_a_process_this_user_cannot_signal_counts_as_running() -> None:
    if os.getuid() == 0:
        pytest.skip("root may signal every process")

    assert process_may_be_running(1) is True
