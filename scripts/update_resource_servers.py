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


README_PATH = Path("README.md")

TARGET_FOLDER = Path("resources_servers")


@dataclass
class ResourceServerMetadata:
    """Metadata extracted from resource server YAML config."""

    dataset_name: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
    verified: bool = False
    verified_url: Optional[str] = None
    value: Optional[str] = None

    def to_dict(self) -> dict[str, str | bool | None]:  # pragma: no cover
        """Convert to dict for backward compatibility with hf_utils.py"""
        return {
            "dataset_name": self.dataset_name,
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

    def to_dict(self) -> dict[str, str | list[str] | None]:  # pragma: no cover
        """Convert to dict for backward compatibility."""
        return {
            "license": self.license,
            "types": self.types,
        }


@dataclass
class ConfigMetadata:
    """Combined metadata from YAML configuration file."""

    dataset_name: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
    verified: bool = False
    verified_url: Optional[str] = None
    value: Optional[str] = None
    license: Optional[str] = None
    types: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml_data(
        cls, resource: ResourceServerMetadata, agent: AgentDatasetsMetadata
    ) -> "ConfigMetadata":  # pragma: no cover
        """Combine resource server and agent datasets metadata."""
        return cls(
            dataset_name=resource.dataset_name,
            domain=resource.domain,
            description=resource.description,
            verified=resource.verified,
            verified_url=resource.verified_url,
            value=resource.value,
            license=agent.license,
            types=agent.types,
        )


@dataclass
class ServerInfo:
    """Information about a resource server for table generation."""

    name: str
    display_name: str
    config_metadata: ConfigMetadata
    config_path: str
    config_filename: str
    readme_path: str
    yaml_file: Path

    @property
    def dataset_name(self) -> str | None:  # pragma: no cover
        return self.config_metadata.dataset_name

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
            return "Example resource server"

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
        return f"<a href='{self.config_path}'>{self.config_metadata.dataset_name}</a>"

    def get_config_link(self) -> str:  # pragma: no cover
        return f"<a href='{self.config_path}'>{self.config_filename}</a>"

    def get_readme_link(self) -> str:  # pragma: no cover
        return f"<a href='{self.readme_path}'>README</a>"


def visit_resource_server(data: dict, level: int = 1) -> ResourceServerMetadata:  # pragma: no cover
    """Extract resource server metadata from YAML data."""
    resource = ResourceServerMetadata()
    if level == 4:
        resource.dataset_name = data.get("dataset_name")
        resource.domain = data.get("domain")
        resource.description = data.get("description")
        resource.verified = data.get("verified", False)
        resource.verified_url = data.get("verified_url")
        resource.value = data.get("value")
        return resource
    else:
        for k, v in data.items():
            if level == 2 and k != "resources_servers":
                continue
            return visit_resource_server(v, level + 1)
    return resource


def visit_agent_datasets(data: dict) -> AgentDatasetsMetadata:  # pragma: no cover
    agent = AgentDatasetsMetadata()
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
                                    agent.types.append(entry.get("type"))
                                    if entry.get("type") == "train":
                                        agent.license = entry.get("license")
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
                    dataset_name: {example_dataset_name}
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

    resource_data = visit_resource_server(data)
    agent_data = visit_agent_datasets(data)

    return ConfigMetadata.from_yaml_data(resource_data, agent_data)


def get_example_and_training_server_info() -> tuple[list[ServerInfo], list[ServerInfo]]:  # pragma: no cover
    """Categorize servers into example-only and training-ready with metadata."""
    example_only_servers = []
    training_servers = []

    for subdir in TARGET_FOLDER.iterdir():
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

            server_name = subdir.name
            is_example_only = server_name.startswith("example_")

            if not is_example_only and not yaml_data.dataset_name:
                continue

            display_name = (
                (server_name[len("example_") :] if is_example_only else server_name).replace("_", " ").title()
            )

            config_path = f"{TARGET_FOLDER.name}/{server_name}/configs/{yaml_file.name}"
            readme_path = f"{TARGET_FOLDER.name}/{server_name}/README.md"

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


def generate_example_only_table(servers: list[ServerInfo]) -> str:  # pragma: no cover
    """Generate table for example-only resource servers."""
    if not servers:
        return "| Name | Demonstrates | Config | README |\n| ---- | ------------------- | ----------- | ------ |\n"

    col_names = ["Name", "Demonstrates", "Config", "README"]
    rows = []

    for server in servers:
        rows.append(
            [
                server.display_name,
                server.get_description_for_example_table(),
                server.get_config_link(),
                server.get_readme_link(),
            ]
        )

    rows.sort(key=lambda r: tuple(normalize_str(cell) for cell in r))

    table = [col_names, ["-" for _ in col_names]] + rows
    return format_table(table)


def generate_training_table(servers: list[ServerInfo]) -> str:  # pragma: no cover
    """Generate table for training resource servers."""
    if not servers:
        # TODO: Add back in when we can verify resource servers
        # return "| Dataset | Domain | Resource Server | Description | Value | Train | Validation | Verified | License |\n| ------- | ------ | --------------- | ----------- | ----- | ----- | ---------- | -------- | ------- |\n"
        return "| Dataset | Domain | Resource Server | Description | Value | Train | Validation | License |\n| ------- | ------ | --------------- | ----------- | ----- | ----- | ---------- | ------- |\n"

    col_names = [
        "Dataset",
        "Domain",
        "Resource Server",
        "Description",
        "Value",
        "Train",
        "Validation",
        # "Verified",
        "License",
    ]
    rows = []

    for server in servers:
        # TODO: Add back in when we can verify resource servers
        # verified_mark = server.get_verified_mark()

        rows.append(
            [
                server.get_dataset_link(),
                server.get_domain_or_empty(),
                server.display_name,
                server.get_description_or_dash(),
                server.get_value_or_dash(),
                server.get_train_mark(),
                server.get_validation_mark(),
                # TODO: Add back in when we can verify resource servers
                # verified_mark,
                server.get_license_or_dash(),
            ]
        )

    rows.sort(
        key=lambda r: (
            normalize_str(r[1]),  # domain
            0 if "✓" in r[7] else 1,  # verified first (reverse order for checkmarks...hyphens)
            tuple(normalize_str(cell) for cell in r),
        )
    )

    table = [col_names, ["-" for _ in col_names]] + rows
    return format_table(table)


def normalize_str(s: str) -> str:  # pragma: no cover
    """
    Rows with identical domain values may get reordered differently
    between local and CI runs. We normalize text and
    use all columns as tie-breakers to ensure deterministic sorting.
    """
    if not s or not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).casefold().strip()


def format_table(table: list[list[str]]) -> str:  # pragma: no cover
    """Format grid of data into markdown table."""
    col_widths = []
    num_cols = len(table[0])

    for i in range(num_cols):
        max_len = 0
        for row in table:
            cell_len = len(str(row[i]))
            if cell_len > max_len:
                max_len = cell_len
        col_widths.append(max_len)

    # Pretty print cells for raw markdown readability
    formatted_rows = []
    for i, row in enumerate(table):
        formatted_cells = []
        for j, cell in enumerate(row):
            cell = str(cell)
            col_width = col_widths[j]
            pad_total = col_width - len(cell)
            if i == 1:  # header separater
                formatted_cells.append(cell * col_width)
            else:
                formatted_cells.append(cell + " " * pad_total)
        formatted_rows.append("| " + (" | ".join(formatted_cells)) + " |")

    return "\n".join(formatted_rows)


def main():  # pragma: no cover
    text = README_PATH.read_text()

    example_servers, training_servers = get_example_and_training_server_info()

    example_table_str = generate_example_only_table(example_servers)
    training_table_str = generate_training_table(training_servers)

    example_pattern = re.compile(
        r"(<!-- START_EXAMPLE_ONLY_SERVERS_TABLE -->)(.*?)(<!-- END_EXAMPLE_ONLY_SERVERS_TABLE -->)",
        flags=re.DOTALL,
    )

    if not example_pattern.search(text):
        sys.stderr.write(
            "Error: README.md does not contain <!-- START_EXAMPLE_ONLY_SERVERS_TABLE --> and <!-- END_EXAMPLE_ONLY_SERVERS_TABLE --> markers.\n"
        )
        sys.exit(1)

    text = example_pattern.sub(lambda m: f"{m.group(1)}\n{example_table_str}\n{m.group(3)}", text)

    training_pattern = re.compile(
        r"(<!-- START_TRAINING_SERVERS_TABLE -->)(.*?)(<!-- END_TRAINING_SERVERS_TABLE -->)",
        flags=re.DOTALL,
    )

    if not training_pattern.search(text):
        sys.stderr.write(
            "Error: README.md does not contain <!-- START_TRAINING_SERVERS_TABLE --> and <!-- END_TRAINING_SERVERS_TABLE --> markers.\n"
        )
        sys.exit(1)

    text = training_pattern.sub(lambda m: f"{m.group(1)}\n{training_table_str}\n{m.group(3)}", text)

    README_PATH.write_text(text)


if __name__ == "__main__":
    main()
