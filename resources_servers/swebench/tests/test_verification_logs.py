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
from pathlib import Path

from resources_servers.swebench.verification_logs import verification_log_path


def test_no_folder_means_no_logs() -> None:
    assert verification_log_path(None, "owner/repo-1") is None


def test_each_verification_gets_its_own_folder_inside_the_root(tmp_path: Path) -> None:
    first = verification_log_path(tmp_path, "../owner/repo-1")
    second = verification_log_path(tmp_path, "../owner/repo-1")

    # The id cannot climb out of the root or split into nested folders.
    assert first.parent == tmp_path / "instance-..%2Fowner%2Frepo-1"
    assert first != second and first.parent == second.parent
    # Building the path creates nothing; the verifier creates the folder when it writes.
    assert not any(tmp_path.iterdir())
