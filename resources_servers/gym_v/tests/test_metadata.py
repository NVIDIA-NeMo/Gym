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

import pytest
from PIL import Image

from resources_servers.gym_v._metadata import sanitize_metadata


def test_sanitize_metadata_handles_nested_json_types() -> None:
    sanitized = sanitize_metadata(
        {
            "tuple": (1, 2),
            "set": {"b", "a"},
            "path": Path("/tmp/example"),
            "nested": {"items": [("x", "y")]},
        }
    )

    assert sanitized == {
        "tuple": [1, 2],
        "set": ["a", "b"],
        "path": "/tmp/example",
        "nested": {"items": [["x", "y"]]},
    }


def test_sanitize_metadata_handles_numpy_arrays() -> None:
    np = pytest.importorskip("numpy")

    sanitized = sanitize_metadata({"array": np.array([[1, 2], [3, 4]])})

    assert sanitized == {"array": [[1, 2], [3, 4]]}


def test_sanitize_metadata_drops_pil_images() -> None:
    sanitized = sanitize_metadata(
        {
            "keep": "value",
            "image": Image.new("RGB", (2, 2), (255, 0, 0)),
        }
    )

    assert sanitized == {"keep": "value"}


def test_sanitize_metadata_falls_back_to_repr_for_unknown_types() -> None:
    class Unknown:
        pass

    value = Unknown()
    sanitized = sanitize_metadata({"unknown": value})

    assert sanitized == {"unknown": repr(value)}
