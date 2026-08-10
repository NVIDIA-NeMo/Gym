# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
from pathlib import Path

# opencv-python-headless is excluded from the shipped container via
# sys_platform == 'never'. Install it before tests via the codec install
# script so tests that import cv2 don't fail.
_INSTALL_SCRIPT = Path(__file__).resolve().parents[4] / "docker" / "install_codec_deps.sh"

if _INSTALL_SCRIPT.exists():
    subprocess.run(["bash", str(_INSTALL_SCRIPT)], check=True)
else:
    subprocess.run(
        [sys.executable, "-m", "uv", "pip", "install", "--no-config", "opencv-python-headless==5.0.0.93"],
        check=True,
    )
