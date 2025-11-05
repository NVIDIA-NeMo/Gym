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
import re
import sys
import unicodedata
from pathlib import Path

import yaml


README_PATH = Path("README.md")
TARGET_FOLDER = Path("resources_servers")


def extract_config_metadata(yaml_path: Path) -> tuple[str, str, list[str]]:
    """
    Domain, License, Types:
        {name}_resources_server:
            resources_servers:
                {name}:
                    domain: {example_domain}
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

    domain = None
    example_description = None
    license = None
    types = []

    def visit_domain_and_example_description(data, level=1):
        nonlocal domain, example_description
        if level == 4:
            domain = data.get("domain")
            example_description = data.get("example_description")
            return
        else:
            for k, v in data.items():
                if level == 2 and k != "resources_servers":
                    continue
                visit_domain_and_example_description(v, level + 1)

    def visit_license_and_types(data):
        nonlocal license
        for k1, v1 in data.items():
            if k1.endswith("_simple_agent") and isinstance(v1, dict):
                v2 = v1.get("responses_api_agents")
                if isinstance(v2, dict):
                    v3 = v2.get("simple_agent")
                    if isinstance(v3, dict):
                        datasets = v3.get("datasets")
                        if isinstance(datasets, list):
                            for entry in datasets:
                                if isinstance(entry, dict):
                                    types.append(entry.get("type"))
                                    if entry.get("type") == "train":
                                        license = entry.get("license")
                            return

    visit_domain_and_example_description(data)
    visit_license_and_types(data)

    return domain, example_description, license, types


def get_example_and_training_server_info() -> tuple[list[dict], list[dict]]:
    """Categorize servers into example-only and training-ready with metadata."""
    example_only_servers = []
    training_servers = []

    # Search subdirectories and nested examples folder
    search_paths = [TARGET_FOLDER]
    examples_folder = TARGET_FOLDER / "examples"
    if examples_folder.exists():
        search_paths.append(examples_folder)

    for search_path in search_paths:
        for subdir in search_path.iterdir():
            if not subdir.is_dir() or subdir.name == "examples":
                continue

            configs_folder = subdir / "configs"
            if not (configs_folder.exists() and configs_folder.is_dir()):
                continue

            yaml_files = list(configs_folder.glob("*.yaml"))
            if not yaml_files:
                continue

            for yaml_file in yaml_files:
                domain, example_description, license, types = extract_config_metadata(yaml_file)

                server_name = subdir.name
                display_name = server_name.replace("_", " ").title()
                # Handles nested examples folder structure
                relative_path = subdir.relative_to(TARGET_FOLDER)
                config_path = f"{TARGET_FOLDER.name}/{relative_path}/configs/{yaml_file.name}"
                readme_path = f"{TARGET_FOLDER.name}/{relative_path}/README.md"

                server_info = {
                    "name": server_name,
                    "display_name": display_name,
                    "domain": domain,
                    "example_description": example_description,
                    "config_path": config_path,
                    "readme_path": readme_path,
                    "types": types,
                    "license": license,
                    "yaml_file": yaml_file,
                }

                # TODO: verify this logic
                # Example-only: only has "example" type
                # Training-ready: has "train" or "validation" type
                # types_set = set(types) if types else set()

                # has_train_or_val = "train" in types_set or "validation" in types_set

                # if has_train_or_val:
                #     training_servers.append(server_info)
                # elif types_set == {"example"}:
                #     example_only_servers.append(server_info)
                is_example_only = server_info.get("example_description", False)
                if is_example_only:
                    example_only_servers.append(server_info)
                else:
                    training_servers.append(server_info)
                # Note: Excludes a server that has neither example nor train/val

    return example_only_servers, training_servers


def generate_example_only_table(servers: list[dict]) -> str:
    """Generate table for example-only resource servers."""
    if not servers:
        return "| Name | Demonstrates | Config | README |\n| ---- | ------------------- | ----------- | ------ |\n"

    col_names = ["Name", "Demonstrates", "Config", "README"]
    rows = []

    for server in servers:
        name = server["display_name"]

        # Optional {example_description} -> Required '{domain} example' -> Fallback: 'Example resource server'
        example_description = (
            server["example_description"] or f"{server.get('domain').title()} example"
            if server.get("domain")
            else "Example resource server"
        )

        config_link = f"<a href='{server['config_path']}'>config</a>"
        readme_link = f"<a href='{server['readme_path']}'>README</a>"

        rows.append([name, example_description, config_link, readme_link])

    rows.sort(
        key=lambda r: (
            normalize_str(r[0]),
            normalize_str(r[1]),
            normalize_str(r[2]),
            normalize_str(r[3]),
        )
    )

    table = [col_names, ["-" for _ in col_names]] + rows
    return format_table(table)


def generate_training_table(servers: list[dict]) -> str:
    """Generate table for training resource servers."""
    if not servers:
        return "| Domain | Resource Server | Train | Validation | Config | License |\n| ------ | --------------- | ----- | ---------- | ------ | ------- |\n"

    col_names = ["Domain", "Resource Server", "Train", "Validation", "Config", "License"]
    rows = []

    for server in servers:
        domain = server["domain"] if server["domain"] else ""
        name = server["display_name"]

        types_set = set(server["types"]) if server["types"] else set()
        train_mark = "✓" if "train" in types_set else "-"
        val_mark = "✓" if "validation" in types_set else "-"

        config_link = f"<a href='{server['config_path']}'>config</a>"

        license_str = server["license"] if server["license"] else "-"

        rows.append([domain, name, train_mark, val_mark, config_link, license_str])

    rows.sort(
        key=lambda r: (
            normalize_str(r[0]),
            normalize_str(r[1]),
            normalize_str(r[2]),
            normalize_str(r[3]),
            normalize_str(r[4]),
            normalize_str(r[5]),
        )
    )

    table = [col_names, ["-" for _ in col_names]] + rows
    return format_table(table)


def normalize_str(s: str) -> str:
    """
    Rows with identical domain values may get reordered differently
    between local and CI runs. We normalize text and
    use all columns as tie-breakers to ensure deterministic sorting.
    """
    if not s:
        return ""
    return unicodedata.normalize("NFKD", s).casefold().strip()


def format_table(table: list[list[str]]) -> str:
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


def main():
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
