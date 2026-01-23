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
import json

from openai.types.responses import FunctionToolParam


TOOL_DEFINITION_TEMPLATE = """<tool>
Name: {name}
Documentation: {documentation}
Parameters in JSON Schema format: {parameters}
</tool>"""


def format_tool_definitions(tool_definitions: list[FunctionToolParam]) -> str:
    tool_definition_list = []
    for tool_definition in tool_definitions:
        description = tool_definition.get("description")
        if description is None:
            description = "null"

        tool_definition_string = TOOL_DEFINITION_TEMPLATE.format(
            name=tool_definition["name"],
            documentation=description,
            parameters=json.dumps(tool_definition.get("parameters")),
        )
        tool_definition_list.append(tool_definition_string)

    return "\n\n".join(tool_definition_list)
