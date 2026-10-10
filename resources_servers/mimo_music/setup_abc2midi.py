# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import subprocess
import sys
from pathlib import Path


PREFIX = Path(__file__).parent / ".abcmidi"
SOURCE = "https://github.com/sshlien/abcmidi/archive/refs/tags/2025.02.16.tar.gz"


def ensure_abc2midi() -> str:
    found = os.environ.get("ABC2MIDI_BIN") or shutil.which("abc2midi") or str(PREFIX / "abc2midi")
    if not Path(found).exists() and sys.platform == "darwin" and shutil.which("brew"):
        subprocess.run(["brew", "install", "abcmidi"], check=True)
        found = shutil.which("abc2midi") or found
    if not Path(found).exists():
        PREFIX.mkdir(exist_ok=True)
        subprocess.run(
            f"curl -fsSL {SOURCE} | tar xz --strip-components=1 -C {PREFIX} && make -C {PREFIX} abc2midi",
            shell=True,
            check=True,
        )
    subprocess.run([found, "-ver"], check=True, capture_output=True)
    os.environ["ABC2MIDI_BIN"] = found
    return found
