# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


LOG = logging.getLogger(__name__)

_INSTALL_URL = "https://cli.kiro.dev/install"


def ensure_kiro_cli() -> None:
    """Ensure ``kiro-cli`` is available using Kiro's official installer."""
    if shutil.which("kiro-cli"):
        return

    local_bin = Path.home() / ".local" / "bin"
    local_cli = local_bin / "kiro-cli"
    if local_cli.is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
        return

    with tempfile.TemporaryDirectory(prefix="kiro-install-") as temp_dir:
        script = Path(temp_dir) / "install.sh"
        urllib.request.urlretrieve(_INSTALL_URL, script)  # noqa: S310
        subprocess.run(["bash", str(script)], check=True)

    if local_cli.is_file():
        os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")

    if not shutil.which("kiro-cli"):
        raise RuntimeError("Kiro CLI installation completed without a kiro-cli executable")

    LOG.info("Kiro CLI is ready at %s", shutil.which("kiro-cli"))
