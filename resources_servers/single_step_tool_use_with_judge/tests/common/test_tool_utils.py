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
from resources_servers.single_step_tool_use_with_judge.common.tool_utils import format_tool_definitions


class TestToolUtils:
    def test_format_tool_definitions(self) -> None:
        assert format_tool_definitions([]) == ""

        first_tool = {
            "type": "function",
            "name": "first_function",
            "parameters": None,
            "strict": False,
        }
        expected_first_tool_definition = """<tool>
Name: first_function
Documentation: null
Parameters in JSON Schema format: null
</tool>"""
        assert format_tool_definitions([first_tool]) == expected_first_tool_definition

        second_tool = {
            "type": "function",
            "name": "second_function",
            "parameters": {},
            "strict": True,
            "description": "This is the second function.",
        }
        expected_second_tool_definition = """<tool>
Name: second_function
Documentation: This is the second function.
Parameters in JSON Schema format: {}
</tool>"""
        assert format_tool_definitions([second_tool]) == expected_second_tool_definition
        assert (
            format_tool_definitions(
                [
                    first_tool,
                    second_tool,
                ]
            )
            == expected_first_tool_definition + "\n\n" + expected_second_tool_definition
        )

        third_tool = {
            "type": "function",
            "name": "third_function",
            "parameters": {"inner3": "inner_value3"},
            "strict": False,
            "description": "The purpose of the third function",
        }
        expected_third_tool_definition = """<tool>
Name: third_function
Documentation: The purpose of the third function
Parameters in JSON Schema format: {"inner3": "inner_value3"}
</tool>"""
        assert format_tool_definitions([third_tool]) == expected_third_tool_definition
        assert (
            format_tool_definitions(
                [
                    second_tool,
                    third_tool,
                    first_tool,
                ]
            )
            == expected_second_tool_definition
            + "\n\n"
            + expected_third_tool_definition
            + "\n\n"
            + expected_first_tool_definition
        )
