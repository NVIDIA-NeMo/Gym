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

from glob import glob
from pathlib import Path


paths_to_fix = [
    "pyproject.toml",
    ".python-version",
    "uv.lock",
    *glob("*/*/pyproject.toml", recursive=True),
]
print("Downgrading the Python from 3.13.14 to 3.12.13 in the following paths:\n", "\n".join(paths_to_fix))
for path_to_fix in paths_to_fix:
    path_to_fix = Path(path_to_fix)
    content = path_to_fix.read_text()
    content = content.replace("3.13.14", "3.12.13")
    path_to_fix.write_text(content)

# Need to also downgrade Ray from 2.56.1 (no Python 3.12 support) to 2.55.1
path = Path("pyproject.toml")
content = path.read_text()
content = content.replace("ray[default]>=2.56.1", "ray[default]==2.55.1")
path.write_text(content)
