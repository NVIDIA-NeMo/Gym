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
"""Write one JSONL row per VAB-CSS task folder, with upstream's filter.

    python resources_servers/vab_css/prepare_data.py --data-dir <css_dataset> --output <file>.jsonl

Upstream (task.py) drops tasks whose corruption distance is below 0.2 and tasks whose instruction says
"This is an invalid example"; the paper's test split has 165 tasks after that filter.
"""

import argparse
import json
from pathlib import Path


INSTRUCTIONS = """
You are a CSS agent. You will be given a target screenshot and an html file. Fix the css rules used in the html file to match the target screenshot.
To facilitate the process, you can use the following tools provided by the system:
1. get_selectors_by_html_elements
Sometimes, the exact selector of the rule you want to edit is not clear. This tool takes the html element specification that could be directly passed to soup.find_all as input and returns the matched selectors. For example, get_selectors_by_html_elements("'a', {'data-custom': 'custom-value'}, string='haha', class_='xxx'"). The argument should be the string representation of valid arguments of the find_all method in BeautifulSoup, which means we can directly do eval(f"soup.find_all({argument})"). Please strictly stick to the usage of BeautifulSoup. Make sure the arguments are valid (e.g., the tag name must be wrapped with quotes, attributes should be a dictionary). You can use this tool to first find the selector of the rule of a specific html element whose style you want to fix.
2. select_rule
This takes the css rule's selectorText as input and returns the rule. You can use this tool to view the properties of a rule, which may help you to decide which rule to edit. Usually, it's recommended to first use this tool to view the rule before deciding which rule to edit.
3. edit_rule
This takes the css rule's selectorText, the property name, and the value of the property as input. You can use this tool to change the value of a property of a rule or insert a new property to the rule, if you believe this change would make the rule closer to the target screenshot. Note that, most of the layout issues are related to the categorical properties, such as border, float, display, overflow, position, etc.
4. revert_last_edit
This tool reverts the last single edit you made. You can use this tool to undo the last edit, if you believe it was a mistake. This action takes no arguments.

Make sure the selectorText is valid based on the html file, i.e., it's from the class or id of the html elements. In addition, please focus on the major layout issue! Ignore the font size, font family, and color of the text, even if you believe they are not perfect.

You can only take ONE action at a time!! For each step, you may first state your thought, then take an action following the format of Thought: ...\n Action: ... (do not add any linebreak after the colon).
For example, you may output\x20
"Thought: I think I should adjust the alignment property of the rule, because the target screenshot shows the text should be centered.\n Action: edit_rule('.templatemo_menu li', 'text-align', 'center')".

After editing a rule or inserting a rule, you will see the updated screenshot of the html file. You should decide your next action (e.g., to revert the last edit or keep adjusting the css) based on the updated screenshot. If you think you have already fixed the css style, please say exactly "I have fixed the css style".

Please strictly follow the format specified above, and please don't repeat the same action in multiple rounds. Also note that, you don't need to worry about how these tools are executed, your job is just to correctly predict the tool invocation.
"""  # noqa: E501 -- upstream's prompt, verbatim (task.py INSTRUCTIONS)

ACKNOWLEDGEMENT = "I've understood your instruction, start please."


def build_rows(data_dir: Path, *, min_distance: float, agent_name: str) -> list[dict]:
    rows = []
    for task_dir in sorted(p for p in data_dir.iterdir() if (p / "index.html").is_file()):
        distance = float((task_dir / "corruption" / "distance.txt").read_text().strip())
        instruction_file = task_dir / "corruption" / "instruction.txt"
        if distance < min_distance:
            continue
        if instruction_file.is_file() and "This is an invalid example" in instruction_file.read_text():
            continue
        rows.append(
            {
                "responses_create_params": {
                    "input": [
                        {"role": "user", "content": INSTRUCTIONS},
                        {"role": "assistant", "content": ACKNOWLEDGEMENT},
                    ]
                },
                "task_id": task_dir.name,
                "initial_distance": distance,
                "agent_ref": {"type": "responses_api_agents", "name": agent_name},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder of task folders (upstream css_dataset).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-distance", type=float, default=0.2, help="Upstream's filter; 0 keeps every task.")
    parser.add_argument("--agent-name", default="vab_css_gymnasium_agent")
    args = parser.parse_args()
    rows = build_rows(args.data_dir, min_distance=args.min_distance, agent_name=args.agent_name)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
