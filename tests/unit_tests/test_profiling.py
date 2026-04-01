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
from unittest.mock import MagicMock

from pydantic import ValidationError
from pytest import MonkeyPatch, raises

import nemo_gym.profiling
from nemo_gym.profiling import Profiler


class TestProfiling:
    def test_sanity(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(nemo_gym.profiling, "graph_from_dot_file", MagicMock(return_value=(MagicMock(),)))

        monkeypatch.setattr(Profiler, "_check_for_dot_installation", MagicMock())

        profiler = Profiler(name="test_name", base_profile_dir=tmp_path / "profile")
        profiler.start()
        profiler.stop()

    def test_profiler_errors_on_invalid_name(self, tmp_path: Path) -> None:
        with raises(ValidationError, match="Spaces are not allowed"):
            Profiler(name="test name", base_profile_dir=tmp_path / "profile")
