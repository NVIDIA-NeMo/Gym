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
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


README_PATH = Path("docs/environments/index.md")

# Directories to scan for environment configs. Agents can also define environments.
SCAN_FOLDERS = [Path("resources_servers"), Path("responses_api_agents")]

# TODO: see if we still need target folder
TARGET_FOLDER = SCAN_FOLDERS[0]


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


@dataclass
class AgentDatasetsMetadata:
    """Metadata extracted from agent datasets configuration."""

    license: str | None = None
    types: list[str] = field(default_factory=list)
    huggingface_repo_id: Optional[str] = None

    def to_dict(self) -> dict[str, str | list[str] | None]:  # pragma: no cover
        """Convert to dict for backward compatibility."""
        return {
            "huggingface_repo_id": self.huggingface_repo_id,
            "license": self.license,
            "types": self.types,
        }


@dataclass
class ConfigMetadata:
    """Combined metadata from YAML configuration file."""

    huggingface_repo_id: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
    verified: bool = False
    verified_url: Optional[str] = None
    value: Optional[str] = None
    license: Optional[str] = None
    types: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml_data(
        cls, resource: ResourcesServerMetadata, agent: AgentDatasetsMetadata
    ) -> "ConfigMetadata":  # pragma: no cover
        """Combine resources server and agent datasets metadata."""
        return cls(
            domain=resource.domain,
            description=resource.description,
            verified=resource.verified,
            verified_url=resource.verified_url,
            value=resource.value,
            huggingface_repo_id=agent.huggingface_repo_id,
            license=agent.license,
            types=agent.types,
        )


@dataclass
class ServerInfo:
    """Information about a resources server for table generation."""

    name: str
    display_name: str
    config_metadata: ConfigMetadata
    config_path: str
    config_filename: str
    readme_path: str
    yaml_file: Path

    @property
    def huggingface_repo_id(self) -> str | None:  # pragma: no cover
        return self.config_metadata.huggingface_repo_id

    @property
    def domain(self) -> str | None:  # pragma: no cover
        return self.config_metadata.domain

    @property
    def types(self) -> list[str]:  # pragma: no cover
        return self.config_metadata.types

    def get_description_for_example_table(self) -> str:  # pragma: no cover
        if self.config_metadata.description:
            return self.config_metadata.description
        elif self.config_metadata.domain:
            return f"{self.config_metadata.domain.title()} example"
        else:
            return "Example resources server"

    def get_domain_or_empty(self) -> str:  # pragma: no cover
        return self.config_metadata.domain or ""

    def get_description_or_dash(self) -> str:  # pragma: no cover
        return self.config_metadata.description or "-"

    def get_value_or_dash(self) -> str:  # pragma: no cover
        return self.config_metadata.value or "-"

    def get_license_or_dash(self) -> str:  # pragma: no cover
        return self.config_metadata.license or "-"

    def get_verified_mark(self) -> str:  # pragma: no cover
        if self.config_metadata.verified and self.config_metadata.verified_url:
            return f"<a href='{self.config_metadata.verified_url}'>✓</a>"
        elif self.config_metadata.verified:
            return "✓"
        else:
            return "-"

    def get_train_mark(self) -> str:  # pragma: no cover
        return "✓" if "train" in set(self.config_metadata.types) else "-"

    def get_validation_mark(self) -> str:  # pragma: no cover
        return "✓" if "validation" in set(self.config_metadata.types) else "-"

    def get_dataset_link(self) -> str:  # pragma: no cover
        if not self.config_metadata.huggingface_repo_id:
            return "-"
        repo_id = self.config_metadata.huggingface_repo_id
        dataset_name = repo_id.split("/")[-1]
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        return f"<a href='{dataset_url}'>{dataset_name}</a>"

    def get_config_link(self, use_filename: bool = True) -> str:  # pragma: no cover
        return f"<a href='{self.config_path}'>{self.config_filename if use_filename else 'config'}</a>"

    def get_readme_link(self) -> str:  # pragma: no cover
        return f"<a href='{self.readme_path}'>README</a>"


_KNOWN_SERVER_TYPES = {"resources_servers", "responses_api_agents"}


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
            if level == 2 and k not in _KNOWN_SERVER_TYPES:
                continue
            return visit_resources_server(v, level + 1)
    return resource


def visit_agent_datasets(data: dict) -> AgentDatasetsMetadata:  # pragma: no cover
    agent = AgentDatasetsMetadata()
    for k1, v1 in data.items():
        if not isinstance(v1, dict):
            continue
        # Look under any server type
        for server_type in _KNOWN_SERVER_TYPES:
            v2 = v1.get(server_type)
            if not isinstance(v2, dict):
                continue
            for v3 in v2.values():
                if not isinstance(v3, dict):
                    continue
                datasets = v3.get("datasets")
                if isinstance(datasets, list):
                    for entry in datasets:
                        if isinstance(entry, dict):
                            agent.types.append(entry.get("type"))
                            if entry.get("type") == "train":
                                agent.license = entry.get("license")
                                hf_id = entry.get("huggingface_identifier")
                                if hf_id and isinstance(hf_id, dict):
                                    agent.huggingface_repo_id = hf_id.get("repo_id")
    return agent


def extract_config_metadata(yaml_path: Path) -> ConfigMetadata:  # pragma: no cover
    """
    Domain:
        {name}_resources_server:
            resources_servers:
                {name}:
                    domain: {example_domain}
                    verified: {true/false}
                    description: {example_description}
                    value: {example_value}
                    ...
        {something}_simple_agent:
            responses_api_agents:
                simple_agent:
                    datasets:
                        - name: train
                          type: {example_type_1}
                          license: {example_license_1}
                          huggingface_identifier:
                            repo_id: {example_repo_id_1}
                            artifact_fpath: {example_artifact_fpath_1}
                        - name: validation
                          type: {example_type_2}
                          license: {example_license_2}
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return ConfigMetadata()

    resource_data = visit_resources_server(data)
    agent_data = visit_agent_datasets(data)

    return ConfigMetadata.from_yaml_data(resource_data, agent_data)


def get_example_and_training_server_info() -> tuple[list[ServerInfo], list[ServerInfo]]:  # pragma: no cover
    """Categorize servers into example-only and training-ready with metadata."""
    example_only_servers = []
    training_servers = []

    for scan_folder in SCAN_FOLDERS:
        if not scan_folder.exists():
            continue

        for subdir in scan_folder.iterdir():
            if not subdir.is_dir():
                continue

            configs_folder = subdir / "configs"
            if not (configs_folder.exists() and configs_folder.is_dir()):
                continue

            yaml_files = list(configs_folder.glob("*.yaml"))
            if not yaml_files:
                continue

            for yaml_file in yaml_files:
                yaml_data = extract_config_metadata(yaml_file)
                if not yaml_data.types:
                    continue
                if scan_folder.name == "responses_api_agents" and not yaml_data.domain:
                    continue

                server_name = subdir.name
                is_example_only = server_name.startswith("example_")

                display_name = (
                    (server_name[len("example_") :] if is_example_only else server_name).replace("_", " ").title()
                )

                config_path = f"{scan_folder.name}/{server_name}/configs/{yaml_file.name}"
                readme_path = f"{scan_folder.name}/{server_name}/README.md"

                server_info = ServerInfo(
                    name=server_name,
                    display_name=display_name,
                    config_metadata=yaml_data,
                    config_path=config_path,
                    config_filename=yaml_file.name,
                    readme_path=readme_path,
                    yaml_file=yaml_file,
                )

                if is_example_only:
                    example_only_servers.append(server_info)
                else:
                    training_servers.append(server_info)

    return example_only_servers, training_servers


def generate_example_cards(servers: list[ServerInfo]) -> str:  # pragma: no cover
    """Generate a small card grid for example-only servers."""
    if not servers:
        return "<p><em>No example environments found.</em></p>"

    servers = sorted(servers, key=lambda s: normalize_str(s.display_name))
    lines = ['<div class="env-grid env-grid-examples">']
    for s in servers:
        desc = s.get_description_for_example_table()
        lines.append(
            f'<div class="env-card">'
            f'<div class="env-card-top"><strong class="env-card-name">{s.display_name}</strong></div>'
            f'<p class="env-card-desc">{desc}</p>'
            f'<div class="env-card-footer">'
            f"{s.get_config_link()} &nbsp; {s.get_readme_link()}"
            f"</div></div>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def generate_training_cards(servers: list[ServerInfo]) -> str:  # pragma: no cover
    """Generate filterable card grid for training/eval servers."""
    if not servers:
        return "<p><em>No training environments found.</em></p>"

    servers = sorted(
        servers,
        key=lambda s: (
            normalize_str(s.get_domain_or_empty()),
            normalize_str(s.display_name),
            normalize_str(s.config_filename),
        ),
    )
    lines = [
        '<div id="env-filter-bar"></div>',
        '<p id="env-count" class="env-count-label"></p>',
        '<div id="env-cards-grid" class="env-grid">',
    ]
    for s in servers:
        domain = s.get_domain_or_empty()
        badge_cls = f"env-badge env-badge-{domain}" if domain else "env-badge"
        train = "1" if s.get_train_mark() == "✓" else "0"
        val = "1" if s.get_validation_mark() == "✓" else "0"
        train_label = f'<span class="env-avail {"yes" if train == "1" else "no"}">train</span>'
        val_label = f'<span class="env-avail {"yes" if val == "1" else "no"}">val</span>'
        license_html = f'<span class="env-license">{s.get_license_or_dash()}</span>'
        dataset_link = s.get_dataset_link()
        dataset_html = f" &nbsp; {dataset_link}" if dataset_link != "-" else ""
        lines.append(
            f'<div class="env-card" data-domain="{domain}" data-train="{train}" data-val="{val}" data-name="{s.display_name}">'
            f'<div class="env-card-top">'
            f'<strong class="env-card-name">{s.display_name}</strong>'
            f'<span class="{badge_cls}">{domain}</span>'
            f"</div>"
            f'<p class="env-card-desc">{s.get_description_or_dash()}</p>'
            f'<div class="env-card-footer">'
            f"{train_label} {val_label} &nbsp; "
            f"{s.get_config_link(use_filename=False)}{dataset_html} &nbsp; "
            f"{license_html}"
            f"</div></div>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def normalize_str(s: str) -> str:  # pragma: no cover
    """
    Rows with identical domain values may get reordered differently
    between local and CI runs. We normalize text and
    use all columns as tie-breakers to ensure deterministic sorting.
    """
    if not s or not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).casefold().strip()


def main():  # pragma: no cover
    text = README_PATH.read_text()

    example_servers, training_servers = get_example_and_training_server_info()

    example_table_str = generate_example_cards(example_servers)
    training_table_str = generate_training_cards(training_servers)

    example_pattern = re.compile(
        r"(<!-- START_EXAMPLE_ONLY_SERVERS_TABLE -->)(.*?)(<!-- END_EXAMPLE_ONLY_SERVERS_TABLE -->)",
        flags=re.DOTALL,
    )

    if not example_pattern.search(text):
        sys.stderr.write(
            f"Error: {README_PATH} does not contain <!-- START_EXAMPLE_ONLY_SERVERS_TABLE --> and <!-- END_EXAMPLE_ONLY_SERVERS_TABLE --> markers.\n"
        )
        sys.exit(1)

    text = example_pattern.sub(lambda m: f"{m.group(1)}\n{example_table_str}\n{m.group(3)}", text)

    training_pattern = re.compile(
        r"(<!-- START_TRAINING_SERVERS_TABLE -->)(.*?)(<!-- END_TRAINING_SERVERS_TABLE -->)",
        flags=re.DOTALL,
    )

    if not training_pattern.search(text):
        sys.stderr.write(
            f"Error: {README_PATH} does not contain <!-- START_TRAINING_SERVERS_TABLE --> and <!-- END_TRAINING_SERVERS_TABLE --> markers.\n"
        )
        sys.exit(1)

    text = training_pattern.sub(lambda m: f"{m.group(1)}\n{training_table_str}\n{m.group(3)}", text)

    README_PATH.write_text(text)


if __name__ == "__main__":
    main()
