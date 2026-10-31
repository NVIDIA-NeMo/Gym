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


import pytest

from resources_servers.lean_proof.toolchain import (
    normalize_version,
    parse_lean_version,
)


class TestParsing:
    @pytest.mark.parametrize(
        "output,expected",
        [
            ({"stdout": '"4.19.0"', "stderr": ""}, "4.19.0"),
            ({"stdout": "", "stderr": "4.12.0"}, "4.12.0"),
            ({"stdout": "info: 4.19.0", "stderr": ""}, "4.19.0"),
            # A failed `import Mathlib` must not be read as a version.
            ({"stdout": "", "stderr": "error: unknown package 'Mathlib'"}, None),
            ({"stdout": "", "stderr": ""}, None),
        ],
    )
    def test_parse_lean_version(self, output, expected):
        assert parse_lean_version(output) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("leanprover/lean4:v4.19.0", "4.19.0"),
            ("v4.19.0", "4.19.0"),
            ("4.19.0", "4.19.0"),
            (None, ""),
        ],
    )
    def test_normalize_version(self, raw, expected):
        assert normalize_version(raw) == expected
