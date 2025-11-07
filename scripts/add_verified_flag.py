# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import sys
from pathlib import Path

import yaml


def ensure_verified_flag(yaml_path: Path) -> bool:
    """
    Adds verified: false flag to config if it doesn't exist.
    Returns whether the config was modified.
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    if not data:
        return False

    modified = False
    for v in data.values():
        if isinstance(v, dict) and "resources_servers" in v:
            resources_servers_dict = v["resources_servers"]
            if isinstance(resources_servers_dict, dict):
                for server_config in resources_servers_dict.values():
                    if isinstance(server_config, dict) and "verified" not in server_config:
                        server_config["verified"] = False
                        modified = True

    if modified:
        with yaml_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return True

    return False


def main():
    # idx 0 is the name of the script, idx 1+ are the changed files
    changed_files = sys.argv[1:]

    if not changed_files:
        return 0

    modified_count = 0
    for filepath in changed_files:
        yaml_file = Path(filepath)
        # Add verified: false flag to config if it doesn't exist.
        if ensure_verified_flag(yaml_file):
            modified_count += 1

    if modified_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
