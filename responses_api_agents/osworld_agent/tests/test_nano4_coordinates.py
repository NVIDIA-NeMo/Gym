# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observed step1500 failures, executed against a deterministic device boundary."""

import contextlib
import copy
import hashlib
import io
import os
import sys
import types
import unittest
from unittest.mock import patch

from responses_api_agents.osworld_agent.nano4.agent import Nano4Agent
from responses_api_agents.osworld_agent.nano4.coordinate_compat import (
    STEP1500_COORDINATES,
    compile_compatible,
    extract_projection_audit,
)
from responses_api_agents.osworld_agent.nano4.coordinate_runtime import resolve_pair
from responses_api_agents.osworld_agent.nano4.environment import CanonicalEnvironment
from responses_api_agents.osworld_agent.nano4.reference import Reference
from responses_api_agents.osworld_agent.nano4.session import CURRENT_SESSION, Session


class Step1500Coordinates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        location = os.environ.get("OSWORLD_V4_PROMPT_FORMAT_REPO")
        cls.reference = (
            Reference(location)
            if location
            else types.SimpleNamespace(
                coordinates=types.SimpleNamespace(
                    PYAUTOGUI_PARAM_NAMES={
                        "click": ["x", "y"],
                        "moveTo": ["x", "y"],
                        "scroll": ["clicks", "x", "y"],
                        "mouseDown": ["x", "y"],
                        "mouseUp": ["x", "y"],
                    }
                )
            )
        )

    def execute(self, code):
        calls = []
        module = types.ModuleType("pyautogui")
        for name in ["click", "scroll", "moveTo", "moveRel", "mouseDown", "mouseUp"]:
            setattr(module, name, lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)))
        module.screenshot = lambda: "image-result"
        source = compile_compatible(code, (1920, 1080), self.reference.coordinates.PYAUTOGUI_PARAM_NAMES)
        stream = io.StringIO()
        namespace = {}
        with patch.dict(sys.modules, {"pyautogui": module}), contextlib.redirect_stdout(stream):
            exec(source, namespace)
        stdout, events = extract_projection_audit(stream.getvalue(), code)
        return calls, stdout, events, namespace

    def test_all_five_observed_chrome_grid_clicks_reach_the_button(self):
        for raw, expected in [
            ((608, 731), (1167, 789)),
            ((621, 733), (1192, 791)),
            ((606, 735), (1163, 793)),
            ((612, 733), (1174, 791)),
            ((612, 732), (1174, 790)),
        ]:
            calls, _, events, _ = self.execute(f"import pyautogui\npyautogui.click{raw!r}")
            self.assertEqual(calls, [("click", expected, {})])
            self.assertEqual(events[0]["source_space"], "grid1000")
            self.assertTrue(events[0]["grid_pixel_ambiguous"])

    def test_unit_coordinates_retain_the_reference_projection(self):
        calls, _, events, _ = self.execute("import pyautogui\npyautogui.click(0.5, 1.0)")
        self.assertEqual(calls[0][1], (960, 1079))
        self.assertEqual(events[0]["source_space"], "unit")

    def test_pixel_override_and_outside_grid_pixel_fallback(self):
        calls, _, events, _ = self.execute("# coordinate_space: pixel\nimport pyautogui\npyautogui.click(608,731)")
        self.assertEqual(calls[0][1], (608, 731))
        self.assertEqual(events[0]["selection_reason"], "explicit_coordinate_space_comment")
        calls, _, events, _ = self.execute("import pyautogui\npyautogui.click(1043,281)")
        self.assertEqual(calls[0][1], (1043, 281))
        self.assertEqual(events[0]["selection_reason"], "outside_grid_inside_screenshot")

    def test_actual_gimp_coordinate_variables_and_arithmetic_execute(self):
        calls, _, _, _ = self.execute(
            "import pyautogui\ngimp_x=15/1280\ngimp_y=474/720\npyautogui.click(gimp_x,gimp_y)"
        )
        self.assertEqual(calls[0][1], (22, 710))

    def test_screenshot_assignment_aliases_and_from_imports_work(self):
        calls, _, _, namespace = self.execute(
            "import pyautogui as p\nname_box=p.screenshot()\nfrom pyautogui import click as c\nc(0.13,0.355)\nq=p\nq.click(0.5,0.5)"
        )
        self.assertEqual(namespace["name_box"], "image-result")
        self.assertEqual(calls, [("click", (249, 383), {}), ("click", (960, 540), {})])

    def test_dynamic_loop_and_function_values_are_evaluated_in_the_vm(self):
        calls, _, _, _ = self.execute(
            "import pyautogui\ndef point(v):\n return v/1000\nfor x in [250,750]:\n pyautogui.click(point(x),0.5)"
        )
        self.assertEqual([call[1] for call in calls], [(480, 540), (1439, 540)])

    def test_scroll_keywords_and_cursor_only_calls_preserve_other_arguments(self):
        calls, _, events, _ = self.execute(
            "import pyautogui\npyautogui.scroll(-3,x=0.5,y=0.5)\npyautogui.click()\npyautogui.moveRel(5,-8)\npyautogui.click((0.5,0.5),button='right')"
        )
        self.assertEqual(calls[0], ("scroll", (-3,), {"x": 960, "y": 540}))
        self.assertEqual(calls[1], ("click", (), {}))
        self.assertEqual(calls[2], ("moveRel", (5, -8), {}))
        self.assertEqual(calls[3], ("click", ((960, 540),), {"button": "right"}))
        self.assertEqual(len(events), 2)

    def test_negative_nonfinite_and_offscreen_values_do_not_silently_clamp(self):
        for values in [(-1, 500), (2001, 500), (500, 1081), (True, 0.5), (float("nan"), 0.5), (float("inf"), 0.5)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                resolve_pair(values, (1920, 1080))
        self.assertEqual(resolve_pair((1920, 1080), (1920, 1080), declared_space="pixel")[0], [1919, 1079])

    def test_future_import_and_model_json_names_do_not_break_the_adapter(self):
        calls, stdout, _, _ = self.execute(
            "from __future__ import annotations\nimport pyautogui\njson=None\npyautogui.click(0.5,0.5)\nprint('kept')"
        )
        self.assertEqual(calls[0][1], (960, 540))
        self.assertEqual(stdout, "kept\n")

    def test_marker_inside_a_string_does_not_override_units(self):
        calls, _, events, _ = self.execute("s='# coordinate_space: pixel'\nimport pyautogui\npyautogui.click(608,731)")
        self.assertEqual(calls[0][1], (1167, 789))
        self.assertEqual(events[0]["source_space"], "grid1000")

    @unittest.skipUnless(os.environ.get("OSWORLD_V4_PROMPT_FORMAT_REPO"), "External producer integration fixture")
    def test_compatible_parse_retry_delivers_the_actual_error(self):
        agent = Nano4Agent("fixture", 100, reference_root=self.reference.root, coordinate_policy=STEP1500_COORDINATES)
        messages = []

        def model(payload, _model):
            messages.append(copy.deepcopy(payload["messages"]))
            if len(messages) == 1:
                return "missing tool"
            return self.reference.converter.tool_call_xml("computer.wait", {})

        agent.messages = lambda *_args: ([{"role": "user", "content": "fixture"}], {})
        agent.call_llm = model
        _, actions, info = agent.predict("fixture", {})
        self.assertEqual(actions, ["WAIT"])
        self.assertEqual(info["parse_attempts"], 2)
        self.assertNotEqual(messages[0], messages[1])
        self.assertIn("Expected the reference", messages[1][-1]["content"])
        self.assertEqual(messages[1][-2]["content"], "missing tool")

    @unittest.skipUnless(os.environ.get("OSWORLD_V4_PROMPT_FORMAT_REPO"), "External producer integration fixture")
    def test_actual_device_audit_survives_feedback_without_changing_command_identity(self):
        captured = []
        module = types.ModuleType("pyautogui")
        module.click = lambda *args, **kwargs: captured.append((args, kwargs))

        class Backend:
            def __init__(self):
                self.controller = types.SimpleNamespace(execute_python_command=self.execute)

            def execute(self, command):
                stream = io.StringIO()
                with patch.dict(sys.modules, {"pyautogui": module}), contextlib.redirect_stdout(stream):
                    exec(command, {})
                return {"returncode": 0, "output": stream.getvalue(), "error": ""}

            def step(self, command, _pause):
                self.controller.execute_python_command(command)
                return {"screenshot": b"fixture"}, 0, False, {}

        session = Session(reference=self.reference)
        token = CURRENT_SESSION.set(session)
        try:
            agent = Nano4Agent(
                "fixture", 100, reference_root=self.reference.root, coordinate_policy=STEP1500_COORDINATES
            )
            code = "import pyautogui\nprint('kept')\npyautogui.click(608,731)"
            agent.messages = lambda *_args: ([{"role": "user", "content": "fixture"}], {})
            agent.call_llm = lambda *_args: self.reference.converter.tool_call_xml("exec_py", {"code_block": code})
            _, actions, _ = agent.predict("fixture", {})
            env = CanonicalEnvironment(Backend(), session)
            observation, _, _, info = env.step(actions[0], 0)
            feedback = info["nemotron_v4_execution_feedback"]["execution_feedback"]
            self.assertEqual(feedback["stdout"], "kept\n")
            self.assertEqual(feedback["model_command_sha256"], hashlib.sha256(code.encode()).hexdigest())
            self.assertEqual(captured, [((1167, 789), {})])
            self.assertEqual(info["nemotron_v4_coordinate_projection"][0]["projected_xy"], [1167, 789])
            self.assertIn("__N4_COORDINATE_AUDIT_V1__", env._responses[0]["output"])
            agent._receive_feedback(observation)
        finally:
            CURRENT_SESSION.reset(token)


if __name__ == "__main__":
    unittest.main()
