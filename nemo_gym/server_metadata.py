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
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResourcesServerMetadata:
    """Metadata extracted from resources server YAML config."""

    domain: Optional[str] = None
    description: Optional[str] = None
    verified: bool = False
    verified_url: Optional[str] = None
    value: Optional[str] = None

    def to_dict(self) -> dict[str, str | bool | None]:  # pragma: no cover
        """Convert to dict for backward compatibility with hf_utils.py"""
        return {
            "domain": self.domain,
            "description": self.description,
            "verified": self.verified,
            "verified_url": self.verified_url,
            "value": self.value,
        }


def visit_resources_server(data: dict, level: int = 1) -> ResourcesServerMetadata:  # pragma: no cover
    """Extract resources server metadata from YAML data."""
    resource = ResourcesServerMetadata()
    if level == 4:
        resource.domain = data.get("domain")
        resource.description = data.get("description")
        resource.verified = data.get("verified", False)
        resource.verified_url = data.get("verified_url")
        resource.value = data.get("value")
        return resource
    elif isinstance(data, dict):
        for k, v in data.items():
            if level == 2 and k != "resources_servers":
                continue
            return visit_resources_server(v, level + 1)
    return resource
