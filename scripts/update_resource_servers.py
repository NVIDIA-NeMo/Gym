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
from pathlib import Path

import yaml


README_PATH = Path("README.md")

TARGET_FOLDER = Path("resources_servers")


def extract_domain_and_license(yaml_path: Path) -> list[tuple[str, str]]:
    """
    {name}_resources_server:
        resources_servers:
            {name}:
                entrypoint: app.py
                license: Test License <--
                domain: Test Domain   <--
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    results = []

    def visit(data, level=1):
        if level == 4:  # domain and license
            domain = data.get("domain")
            license = data.get("license")
            if domain is not None or license is not None:
                results.append((domain, license))
        else:
            for k, v in data.items():
                if level == 2 and k != "resources_servers":
                    continue
                visit(v, level + 1)

    visit(data)
    return results


def generate_table() -> str:
    """Outputs a grid with table data"""
    col_names = ["Domain", "License", "Server Type Name", "Path"]

    rows = []
    for subdir in sorted(TARGET_FOLDER.iterdir()):
        path = f"`{TARGET_FOLDER.name}/{subdir.name}`"
        server_name = subdir.name.replace("_", " ").title()

        configs_folder = subdir / "configs"
        if configs_folder.exists() and configs_folder.is_dir():
            yaml_files = sorted(configs_folder.glob("*.yaml"))
            if yaml_files:
                for yaml_file in yaml_files:
                    extraction = extract_domain_and_license(yaml_file)
                    if extraction:
                        for domain, license in extraction:
                            rows.append([domain, license, server_name, path])
                    else:
                        rows.append(["?", "?", server_name, path])
            else:
                rows.append(["?", "?", server_name, path])
        else:
            rows.append(["?", "?", server_name, path])

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

    new_text = pattern.sub(lambda m: f"{m.group(1)}\n{generate_table()}\n{m.group(3)}", text)
    README_PATH.write_text(new_text)


if __name__ == "__main__":
    main()
