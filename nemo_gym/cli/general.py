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

import json
import os
import platform
import sys
from importlib.metadata import version as md_version
from subprocess import Popen

import psutil
from pydantic import Field

from nemo_gym import PARENT_DIR, __version__
from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import JSON_OUTPUT_KEY_NAME, get_global_config_dict


class VersionConfig(BaseNeMoGymCLIConfig):
    """
    Display gym version and system information.

    Examples:

    ```bash
    # Display version information
    gym --version

    # Output as JSON
    gym --version --json
    ```
    """

    json_format: bool = Field(default=False, alias="json", description="Output in JSON format for programmatic use.")


def version():  # pragma: no cover
    """Display gym version and system information."""
    global_config_dict = get_global_config_dict()
    # Just here for help.
    VersionConfig.model_validate(global_config_dict)

    version_info = {
        "nemo_gym": __version__,
        "python": platform.python_version(),
        "python_path": sys.executable,
        "installation_path": str(PARENT_DIR),
    }

    key_deps = [
        "openai",
        "ray",
    ]

    dependencies = {dep: md_version(dep) for dep in key_deps}

    version_info["dependencies"] = dependencies

    # System info
    version_info["system"] = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpus": os.cpu_count(),
    }

    # Memory info
    mem = psutil.virtual_memory()
    version_info["system"]["memory_gb"] = round(mem.total / (1024**3), 2)

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps(version_info))
    else:
        output = f"""\
NeMo Gym v{version_info["nemo_gym"]}
Python {version_info["python"]} ({version_info["python_path"]})
Installation: {version_info["installation_path"]}"""

        if "dependencies" in version_info:
            deps_lines = "\n".join(f"  {dep}: {ver}" for dep, ver in version_info["dependencies"].items())
            sys_info = version_info["system"]
            output += f"""

Key Dependencies:
{deps_lines}

System:
  OS: {sys_info["os"]}
  Platform: {sys_info["platform"]}
  Architecture: {sys_info["architecture"]}
  Processor: {sys_info["processor"]}
  CPUs: {sys_info["cpus"]}
  Memory: {sys_info["memory_gb"]} GB"""

        print(output)


def reinstall():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    Popen("uv sync --extra dev --group docs", shell=True).communicate()
