# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure helpers of swe_if_agents (plain python, no gym dependency).

Run:  python3 responses_api_agents/swe_if_agents/tests/test_hooks.py
  or: pytest -q responses_api_agents/swe_if_agents/tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from responses_api_agents.swe_if_agents.hooks import (  # noqa: E402
    normalize_tool_name_overrides,
    row_metadata_summary,
    tag_replay_observation_suffix,
    write_row_templates,
)

NOTE = "<user_note>from now on...</user_note>"


class TestToolNameOverrides(unittest.TestCase):
    def test_absent_is_none(self):
        self.assertIsNone(normalize_tool_name_overrides(None))
        self.assertIsNone(normalize_tool_name_overrides(""))

    def test_json_string_and_dict_are_canonicalised(self):
        raw = {"READ_TOOL_NAME": "cat_file", "BASH_TOOL_NAME": "terminal"}
        out = normalize_tool_name_overrides(json.dumps(raw))
        self.assertEqual(json.loads(out), raw)
        self.assertEqual(out, json.dumps(raw, sort_keys=True))
        self.assertEqual(normalize_tool_name_overrides(raw), out)

    def test_bad_identifier_or_name_raises(self):
        with self.assertRaises(ValueError):
            normalize_tool_name_overrides({"bash": "terminal"})  # not an identifier
        with self.assertRaises(ValueError):
            normalize_tool_name_overrides({"BASH_TOOL_NAME": "term inal"})  # not a tool name
        with self.assertRaises(ValueError):
            normalize_tool_name_overrides({"BASH_TOOL_NAME": "x", "READ_TOOL_NAME": "x"})  # duplicate name
        with self.assertRaises(ValueError):
            normalize_tool_name_overrides("[]")  # not an object


class TestReplayObservationSuffix(unittest.TestCase):
    def _msgs(self):
        return [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "out1"},
            {"role": "assistant", "tool_calls": [{"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c2", "content": "out2"},
        ]

    def test_no_spec_is_noop(self):
        msgs = self._msgs()
        self.assertIsNone(tag_replay_observation_suffix(msgs, None))
        self.assertNotIn("observation_suffix", msgs[-1])

    def test_tags_last_tool_message_by_default(self):
        msgs = self._msgs()
        tagged = tag_replay_observation_suffix(msgs, json.dumps({"text": NOTE}))
        self.assertIs(tagged, msgs[-1])
        self.assertEqual(msgs[-1]["observation_suffix"], NOTE)
        self.assertNotIn("observation_suffix", msgs[3])

    def test_tags_named_tool_call(self):
        msgs = self._msgs()
        tag_replay_observation_suffix(msgs, {"text": "note", "tool_call_id": "c1"})
        self.assertEqual(msgs[3]["observation_suffix"], "note")
        self.assertNotIn("observation_suffix", msgs[-1])

    def test_missing_target_or_bad_spec_raises(self):
        with self.assertRaises(ValueError):
            tag_replay_observation_suffix(self._msgs(), {"text": "note", "tool_call_id": "nope"})
        with self.assertRaises(ValueError):
            tag_replay_observation_suffix(self._msgs(), {"text": ""})
        with self.assertRaises(ValueError):
            tag_replay_observation_suffix([{"role": "user", "content": "u"}], {"text": "note"})


class TestRowTemplates(unittest.TestCase):
    def test_writes_only_non_empty_texts(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            sp, up = write_row_templates(Path(d), "SYSTEM {{ x }}", "")
            self.assertTrue(sp and Path(sp).read_text() == "SYSTEM {{ x }}")
            self.assertIsNone(up)
            sp2, up2 = write_row_templates(Path(d), None, "USER {{ instance.problem_statement }}")
            self.assertIsNone(sp2)
            self.assertTrue(up2 and up2.endswith("row_user_prompt.j2"))

    def test_summary_has_no_prompt_text(self):
        s = row_metadata_summary({"system_prompt_template_text": "abc", "sdg_item": "{}", "tool_name_overrides": "{}"})
        self.assertEqual(s["system_prompt_template_text"], 3)
        self.assertTrue(s["sdg_item"])
        self.assertTrue(s["tool_name_overrides"])
        self.assertFalse(s["replay_observation_suffix"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
