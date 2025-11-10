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
    license = None
    types = []

    def visit_domain(data, level=1):
        nonlocal domain
        if level == 4:
            domain = data.get("domain")
            return
        else:
            for k, v in data.items():
                if level == 2 and k != "resources_servers":
                    continue
                visit_domain(v, level + 1)

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

    visit_domain(data)
    visit_license_and_types(data)

    return domain, license, types


def generate_table() -> str:
    """
    Outputs a grid with table data. Raw html <a> tags are used for the links instead of markdown
    to avoid cross-reference warnings in the 'build-docs' CI/CD run (15+ warnings == fail)
    """
    col_names = ["Domain", "Resource Server Name", "Config Path", "License", "Usage"]

    rows = []
    for subdir in TARGET_FOLDER.iterdir():
        if subdir.is_dir():
            path = f"{TARGET_FOLDER.name}/{subdir.name}"
            server_name = subdir.name.replace("_", " ").title()

            configs_folder = subdir / "configs"
            if configs_folder.exists() and configs_folder.is_dir():
                yaml_files = configs_folder.glob("*.yaml")
                if yaml_files:
                    for yaml_file in yaml_files:
                        config_path = path + "/configs/" + yaml_file.name
                        config_path_link = f"<a href='{config_path}'>{config_path}</a>"
                        extraction = extract_config_metadata(yaml_file)
                        if extraction:
                            domain, license, usages = extraction
                            rows.append(
                                [
                                    domain,
                                    server_name,
                                    config_path_link,
                                    license,
                                    ", ".join([u.title() for u in usages]),
                                ]
                            )

    def normalize_str(s: str) -> str:
        """
        Rows with identical domain values may get reordered differently
        between local and CI runs. We normalize text and
        use all columns as tie-breakers to ensure deterministic sorting.
        """
        if not s:
            return ""
        return unicodedata.normalize("NFKD", s).casefold().strip()

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
    pattern = re.compile(
        r"(<!-- START_RESOURCE_TABLE -->)(.*?)(<!-- END_RESOURCE_TABLE -->)",
        flags=re.DOTALL,
    )

    if not pattern.search(text):
        sys.stderr.write(
            "Error: README.md does not contain <!-- START_RESOURCE_TABLE --> and <!-- END_RESOURCE_TABLE --> markers.\n"
        )
        sys.exit(1)

    table_str = generate_table()

    new_text = pattern.sub(lambda m: f"{m.group(1)}\n{table_str}\n{m.group(3)}", text)
    README_PATH.write_text(new_text)


if __name__ == "__main__":
    main()
