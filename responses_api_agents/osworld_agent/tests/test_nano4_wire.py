# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nano4 wire decoding independent of a deployment's producer checkout."""

import json
import unittest
from types import SimpleNamespace

from responses_api_agents.osworld_agent.nano4.coordinate_compat import STEP1500_COORDINATES
from responses_api_agents.osworld_agent.nano4.protocol import parse_action
from responses_api_agents.osworld_agent.nano4.runtime import register_runner
from responses_api_agents.osworld_agent.runner_registry import RUNNER_REGISTRY, RunnerSpec


class Nano4WireTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = SimpleNamespace(coordinates=SimpleNamespace(PYAUTOGUI_PARAM_NAMES={"click": ["x", "y"]}))

    def parse(self, value: object):
        return parse_action(
            value, reference=self.reference, screen_size=(1920, 1080), coordinate_policy=STEP1500_COORDINATES
        )

    def test_structured_tool_arguments_and_reasoning_are_retained(self) -> None:
        source = "import pyautogui\npyautogui.click(608,731)"
        action = self.parse(
            {
                "content": "Click the button.",
                "reasoning": "The dialog is ready.",
                "finish_reason": "tool_calls",
                "tool_calls": [{"function": {"name": "exec_py", "arguments": json.dumps({"code_block": source})}}],
            }
        )
        self.assertEqual(action.model_source, source)
        self.assertEqual(action.reasoning, "The dialog is ready.")
        self.assertEqual(action.summary, "Click the button.")
        self.assertIn("__n4_coordinate_adapter", action.desktop_action)

    def test_xml_tool_and_thinking_are_decoded_without_repair(self) -> None:
        action = self.parse(
            "<think>Inspect the dialog.</think>Wait.\n<tool_call><function=computer.wait></function></tool_call>"
        )
        self.assertEqual(action.reasoning, "Inspect the dialog.")
        self.assertEqual(action.desktop_action, "WAIT")

    def test_terminal_status_is_preserved(self) -> None:
        for status, expected in [("success", "DONE"), ("failure", "FAIL")]:
            action = self.parse(
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "computer.terminate",
                                "arguments": {"status": status, "answer": "observed"},
                            }
                        }
                    ]
                }
            )
            self.assertEqual(action.desktop_action, expected)
            self.assertEqual(action.arguments["answer"], "observed")

    def test_truncated_or_duplicate_argument_output_is_rejected(self) -> None:
        for value in [
            {"content": "partial", "finish_reason": "length"},
            {
                "tool_calls": [
                    {"function": {"name": "exec_py", "arguments": '{"code_block":"print(1)","code_block":"print(2)"}'}}
                ]
            },
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)

    def test_multiple_calls_and_unknown_tools_are_rejected(self) -> None:
        call = {"function": {"name": "computer.wait", "arguments": "{}"}}
        with self.assertRaises(ValueError):
            self.parse({"tool_calls": [call, call]})
        with self.assertRaises(ValueError):
            self.parse({"tool_calls": [{"function": {"name": "other_tool", "arguments": {}}}]})

    def test_registration_preserves_every_existing_runner(self) -> None:
        before = dict(RUNNER_REGISTRY)
        if "model_protocol_id" not in RunnerSpec.__dataclass_fields__:
            with self.assertRaisesRegex(RuntimeError, "PR #5101"):
                register_runner(STEP1500_COORDINATES)
            self.assertEqual(RUNNER_REGISTRY, before)
        else:
            name = register_runner(STEP1500_COORDINATES)
            self.assertEqual(
                RUNNER_REGISTRY[name].agent_class_path, "responses_api_agents.osworld_agent.nano4.agent.Nano4Agent"
            )
        for name, value in before.items():
            self.assertIs(RUNNER_REGISTRY[name], value)
