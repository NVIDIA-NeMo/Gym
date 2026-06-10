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

"""Tests for CLI help system parameter printing and example formatting."""

import re
from typing import Dict, Optional, TypedDict

import pytest
from pydantic import Field

from nemo_gym.config_types import BaseNeMoGymCLIConfig


class ParsedHelpOutput(TypedDict):
    """Structured representation of parsed CLI help output."""

    title: str
    description: str
    parameters: Dict[str, str]
    examples: str


def parse_help_output(output: str) -> ParsedHelpOutput:
    """
    Parse help output into structured sections.

    Args:
        output: Raw help output string from CLI command

    Returns:
        ParsedHelpOutput with:
            - title: The "Displaying help for X" line
            - description: The description section content
            - examples: The examples section content
            - parameters: Dict mapping parameter names to their full formatted lines
    """
    lines = output.split("\n")
    result: ParsedHelpOutput = {"title": "", "description": "", "examples": "", "parameters": {}}

    # Find the title line
    title_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Displaying help for"):
            result["title"] = line
            title_idx = i
            break

    if title_idx is None:
        return result

    # Find the Parameters section
    params_idx = None
    for i in range(title_idx + 1, len(lines)):
        if lines[i].strip() == "Parameters":
            params_idx = i
            break

    if params_idx is None:
        # No parameters section, everything after title is description/examples
        content_lines = lines[title_idx + 1 :]
        result["description"] = "\n".join(content_lines).strip()
        return result

    # Everything between title and Parameters is description + examples
    content_lines = lines[title_idx + 1 : params_idx]

    # Find where "Examples:" appears in the content
    examples_idx = None
    for i, line in enumerate(content_lines):
        if line.strip().startswith("Examples:"):
            examples_idx = i
            break

    if examples_idx is not None:
        # Split into description and examples
        description_lines = content_lines[:examples_idx]
        examples_lines = content_lines[examples_idx + 1 :]  # Skip the "Examples:" line

        result["description"] = "\n".join(description_lines).strip()
        result["examples"] = "\n".join(examples_lines).strip()
    else:
        # No examples, everything is description
        result["description"] = "\n".join(content_lines).strip()

    # Parse parameters section
    for i in range(params_idx + 1, len(lines)):
        line = lines[i]
        # Skip the "----------" separator line
        if line.strip() == "-" * len(line.strip()) and line.strip():
            continue
        # Parse parameter lines that start with "-"
        if line.strip().startswith("-"):
            match = re.match(r"-\s+(\w+)", line)
            if match:
                param_name = match.group(1)
                result["parameters"][param_name] = line.strip()

    return result


class TestHelpOutput:
    """Test CLI help output formatting and content."""

    def test_help_output_with_examples(self, capsys):
        """Test complete help output with parameters and examples."""

        class TestConfig(BaseNeMoGymCLIConfig):
            """
            Test configuration with all features.

            Examples:

            ```bash
            # Comment line
            ng_cmd \
                +required_param=value \
                +int_param=42
            ```
            """

            required_param: str = Field(description="Required parameter.")
            optional_param: Optional[str] = Field(default=None, description="Optional parameter.")
            bool_param: bool = Field(default=False, description="Boolean parameter.")
            int_param: int = Field(default=42, description="Integer parameter.")
            dict_param: Dict = Field(default_factory=dict, description="Dictionary parameter.")

        with pytest.raises(SystemExit):
            TestConfig.model_validate({"help": True})

        parsed = parse_help_output(capsys.readouterr().out)

        # Validate title
        assert "Displaying help for TestConfig" in parsed["title"]

        # Validate description
        assert "Test configuration with all features." in parsed["description"]

        # Validate parameter suffixes
        assert "required_param" in parsed["parameters"]
        assert "[required]" in parsed["parameters"]["required_param"]

        assert "optional_param" in parsed["parameters"]
        assert "[required]" not in parsed["parameters"]["optional_param"]
        assert "[default:" not in parsed["parameters"]["optional_param"]

        assert "bool_param" in parsed["parameters"]
        assert "[default: False]" in parsed["parameters"]["bool_param"]

        assert "int_param" in parsed["parameters"]
        assert "[default: 42]" in parsed["parameters"]["int_param"]

        assert "dict_param" in parsed["parameters"]
        assert "<factory>]" in parsed["parameters"]["dict_param"]

        # Validate examples formatting
        assert ".. code-block::" not in parsed["examples"]
        assert "# Comment line" in parsed["examples"]
        assert "ng_cmd" in parsed["examples"]
        assert "+required_param=value" in parsed["examples"]
        assert "+int_param=42" in parsed["examples"]

    def test_help_output_without_examples(self, capsys):
        """Test help output when Examples section is not present."""

        class TestConfig(BaseNeMoGymCLIConfig):
            """Configuration without examples."""

            param: str = Field(description="A parameter.")

        with pytest.raises(SystemExit):
            TestConfig.model_validate({"help": True})

        parsed = parse_help_output(capsys.readouterr().out)

        # Validate title
        assert "Displaying help for TestConfig" in parsed["title"]

        # Validate description
        assert "Configuration without examples." in parsed["description"]

        # Validate examples section is empty
        assert parsed["examples"] == ""

        # Validate parameters section
        assert "param" in parsed["parameters"]
        assert "[required]" in parsed["parameters"]["param"]
