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
from pathlib import Path

import yaml


def visit_resource_server(data, level=1):  # pragma: no cover
    domain = None
    description = None
    verified = None
    verified_url = None
    if level == 4:
        domain = data.get("domain")
        description = data.get("description")
        verified = data.get("verified", False)
        verified_url = data.get("verified_url")
        return domain, description, verified, verified_url
    else:
        for k, v in data.items():
            if level == 2 and k != "resources_servers":
                continue
            return visit_resource_server(v, level + 1)


def visit_agent_datasets(data):  # pragma: no cover
    license = None
    types = []
    for k1, v1 in data.items():
        if k1.endswith("_simple_agent") and isinstance(v1, dict):
            v2 = v1.get("responses_api_agents")
            if isinstance(v2, dict):
                # Look for any agent key
                for agent_key, v3 in v2.items():
                    if isinstance(v3, dict):
                        datasets = v3.get("datasets")
                        if isinstance(datasets, list):
                            for entry in datasets:
                                if isinstance(entry, dict):
                                    types.append(entry.get("type"))
                                    if entry.get("type") == "train":
                                        license = entry.get("license")
                            return license, types


def extract_config_metadata(yaml_path: Path) -> tuple[str, str, str, list[str], bool, str]:  # pragma: no cover
    """
    Domain:
        {name}_resources_server:
            resources_servers:
                {name}:
                    domain: {example_domain}
                    verified: {true/false}
                    ...
        {something}_simple_agent:
            responses_api_agents:
                simple_agent:
                    datasets:
                        - name: train
                          type: {example_type_1}
                          license: {example_license_1}
                        - name: validation
                          type: {example_type_2}
                          license: {example_license_2}
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    domain, description, verified, verified_url = visit_resource_server(data)
    license, types = visit_agent_datasets(data)

    return domain, description, license, types, verified, verified_url
