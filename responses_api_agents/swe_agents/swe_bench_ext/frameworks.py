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
"""Test framework output configuration mapping."""

from typing import Dict


FRAMEWORK_CONFIGS: Dict[str, Dict] = {
    "pytest": {
        "output_flag": "--junitxml=/workspace/test-results/output.xml",
        "result_file": "/workspace/test-results/output.xml",
    },
    "unittest": {
        "output_flag": "--junitxml=/workspace/test-results/output.xml",
        "result_file": "/workspace/test-results/output.xml",
    },
    "go": {
        "output_flag": "-json",
        "result_file": None,
    },
    "jest": {
        "output_flag": "--json --outputFile=/workspace/test-results/output.json",
        "result_file": "/workspace/test-results/output.json",
    },
    "vitest": {
        "output_flag": "--reporter=json --outputFile=/workspace/test-results/output.json",
        "result_file": "/workspace/test-results/output.json",
    },
    "mocha": {
        "output_flag": "--reporter json --reporter-options output=/workspace/test-results/output.json",
        "result_file": "/workspace/test-results/output.json",
    },
    "bun": {
        "output_flag": None,  # Bun doesn't have structured JSON output flag by default
        "result_file": None,  # Parse from stdout
    },
    "junit": {
        "output_flag": None,
        "result_file": "find:/workspace/repo:*/target/surefire-reports:TEST-*.xml",
    },
    "maven": {
        "output_flag": None,
        "result_file": "find:/workspace/repo:*/target/surefire-reports:TEST-*.xml",
    },
    "gtest": {
        "output_flag": "--gtest_output=json:/workspace/test-results/output.json",
        "result_file": "/workspace/test-results/output.json",
    },
    "cargo-nextest": {
        "output_flag": None,  # Profile is already in test_command
        "result_file": None,  # JUnit XML is output to repo/junit.xml by profile config
    },
    "ctest": {
        "output_flag": "--output-on-failure --output-junit /workspace/test-results/output.xml",
        "result_file": "/workspace/test-results/output.xml",
    },
    "xctest": {
        # For SwiftPM with XCTest framework
        "output_flag": "--parallel --num-workers=1 --xunit-output /workspace/test-results/output.xml",
        "result_file": "/workspace/test-results/output.xml",
    },
    "testing": {
        # For SwiftPM with new Swift Testing framework (Swift 6+)
        "output_flag": "--disable-xctest --parallel --xunit-output /workspace/test-results/output.xml",
        "result_file": "/workspace/test-results/output.xml",
    },
    "cppunit": {
        "output_flag": None,
        "result_file": None,
    },
    # Lua test frameworks - Tier 1 (Standard XML output)
    "busted": {
        "output_flag": "--output=junit",
        "result_file": "/workspace/test-results/output.xml",
    },
    "luaunit": {
        "output_flag": "-o junit -n /workspace/test-results/output.xml",
        "result_file": "/workspace/test-results/output.xml",
    },
    # Lua test frameworks - Tier 2 (Custom parsers)
    "telescope": {
        "output_flag": None,
        "result_file": None,
    },
    "lust": {
        "output_flag": None,
        "result_file": None,
    },
    "minitest": {
        "output_flag": None,
        "result_file": None,
    },
    "bespoke_libgeos": {
        "output_flag": None,
        "result_file": None,
    },
    # TAP (Test Anything Protocol) - used by tape, node-tap
    "tap": {
        "output_flag": None,  # TAP outputs to stdout
        "result_file": None,  # Parse from stdout
    },
    "tape": {
        "output_flag": None,  # tape outputs TAP to stdout
        "result_file": None,  # Parse from stdout
    },
    # Hardhat (Solidity) - uses Mocha under the hood
    "hardhat": {
        "output_flag": None,  # Uses Mocha console reporter by default
        "result_file": None,  # Parse from stdout
    },
}


def get_framework_config(framework: str, test_command: str = "") -> Dict:
    """Get configuration for a test framework.

    Args:
        framework: Test framework name
        test_command: The test command (optional, used to detect Gradle vs Maven)
    """
    config = FRAMEWORK_CONFIGS.get(
        framework,
        {
            "output_flag": None,
            "result_file": None,
        },
    )

    # Special handling for JUnit: detect Gradle vs Maven from command
    if framework == "junit" and test_command:
        if "gradlew" in test_command or "gradle " in test_command:
            # Gradle uses different output location than Maven
            # Use */TEST-*.xml to match both standard Gradle (test/) and Android (testDebugUnitTest/)
            config = {
                "output_flag": None,
                "result_file": "find:/workspace/repo:*/build/test-results*:TEST-*.xml",
            }

    # Special handling for xctest: detect Swift Testing vs XCTest from command
    # When --disable-xctest is used, the task is using Swift Testing, not XCTest
    # Use the 'testing' framework config to avoid adding XCTest-only flags like --num-workers
    if framework == "xctest" and test_command:
        if "--disable-xctest" in test_command:
            config = FRAMEWORK_CONFIGS.get("testing", config)

    return config


def get_test_command_with_output(base_command: str, framework: str) -> str:
    """
    Add structured output flags to test command.

    Returns: command_with_output_flags
    """
    config = get_framework_config(framework, base_command)
    output_flag = config.get("output_flag")

    enhanced = f"{base_command} {output_flag}" if output_flag else base_command

    return enhanced
