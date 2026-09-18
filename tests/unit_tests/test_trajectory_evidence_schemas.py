# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "fern/assets/trajectory-evidence-p0.schema.json"
TOKEN_FIELDS = {
    "TE3.1": ("prompt_tokens", "tokens_in"),
    "TE3.2": ("completion_tokens", "tokens_out"),
    "TE3.3": ("reasoning_tokens", "tokens_reasoning"),
    "TE3.4": ("cached_tokens", "cached_tokens"),
    "TE3.5": ("total_tokens", "tokens_total"),
}


def rollout(source, calls):
    if source == "trajectory":
        return {"ng_trajectory": {"schema_version": "1.0", "model_calls": copy.deepcopy(calls)}}
    return {"ng_model_call_capture": {"calls": copy.deepcopy(calls)}}


def token_call(source, claim, value):
    normal, raw = TOKEN_FIELDS[claim]
    if source == "trajectory":
        return {"model_call_id": "call-1", "token_stats": {normal: value}}
    return {"model_call_id": "call-1", raw: value}


class TestTrajectoryEvidenceSchemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = json.loads(SCHEMA_PATH.read_text())
        cls.schemas = {
            name: Draft202012Validator(
                {"$schema": cls.bundle["$schema"], "$defs": cls.bundle["$defs"], "$ref": f"#/$defs/{name}"}
            )
            for name in cls.bundle["$defs"]
            if name.startswith("TE")
        }

    def accepts(self, claim, row):
        return self.schemas[claim].is_valid(row)

    def test_bundle_requires_explicit_claim_selection(self):
        Draft202012Validator.check_schema(self.bundle)
        self.assertEqual(len(self.schemas), 14)
        self.assertFalse(Draft202012Validator(self.bundle).is_valid({}))

    def test_every_claim_rejects_missing_empty_and_extracted_inputs(self):
        invalid = [
            {},
            None,
            [],
            {"model_call_id": "call-1", "token_stats": {"prompt_tokens": 3}},
            {"ng_trajectory": {}},
            {"ng_trajectory": {"schema_version": "1.0", "model_calls": [], "turns": [], "invocations": []}},
            {"ng_model_call_capture": {"calls": []}},
            {"ng_agent_observations": {"records": []}},
            {"response": {"usage": {"input_tokens": 3}}},
        ]
        for claim in self.schemas:
            for row in invalid:
                with self.subTest(claim=claim, row=row):
                    self.assertFalse(self.accepts(claim, row))

    def test_token_aliases_require_valid_values_on_every_call(self):
        for source in ("trajectory", "capture"):
            for claim in TOKEN_FIELDS:
                for value in (0, 123):
                    with self.subTest(source=source, claim=claim, value=value):
                        self.assertTrue(self.accepts(claim, rollout(source, [token_call(source, claim, value)])))
                for value in (None, -1, 1.5, "123", True):
                    with self.subTest(source=source, claim=claim, value=value):
                        calls = [token_call(source, claim, 3), token_call(source, claim, value)]
                        calls[1]["model_call_id"] = "call-2"
                        self.assertFalse(self.accepts(claim, rollout(source, calls)))
                self.assertFalse(self.accepts(claim, rollout(source, [token_call(source, claim, 3), {}])))
                missing_id = token_call(source, claim, 3)
                del missing_id["model_call_id"]
                self.assertFalse(self.accepts(claim, rollout(source, [missing_id])))

    def test_either_complete_source_can_supply_tokens(self):
        for claim in TOKEN_FIELDS:
            trajectory = rollout("trajectory", [token_call("trajectory", claim, 3)])
            capture = rollout("capture", [token_call("capture", claim, 3)])
            self.assertTrue(self.accepts(claim, trajectory | capture))
            self.assertTrue(self.accepts(claim, trajectory | rollout("capture", [{}])))
            self.assertTrue(self.accepts(claim, rollout("trajectory", [{}]) | capture))
            trajectory["ng_trajectory"]["model_calls"].append({"model_call_id": "call-2"})
            capture["ng_model_call_capture"]["calls"][0] = {"model_call_id": "call-1"}
            extra = token_call("capture", claim, 3)
            extra["model_call_id"] = "call-2"
            capture["ng_model_call_capture"]["calls"].append(extra)
            self.assertFalse(self.accepts(claim, trajectory | capture))

    def test_identity_and_outcome_locations(self):
        outcomes = [
            {"status_code": 200, "finish_reason": "stop"},
            {"response_status": "completed", "finish_reason": None},
            {"status_code": 500, "error_category": "upstream_error"},
            {"status_code": None, "error_category": "cancelled"},
        ]
        for source in ("trajectory", "capture"):
            self.assertTrue(self.accepts("TE1.1", rollout(source, [{"model_call_id": "call-1"}])))
            self.assertFalse(self.accepts("TE1.1", rollout(source, [{"model_call_id": "call-1"}, {}])))
            for outcome in outcomes + [{"status_code": 200}, {"error_category": "unknown"}]:
                call = {"response_metadata": outcome} if source == "trajectory" else outcome
                with self.subTest(source=source, outcome=outcome):
                    self.assertEqual(self.accepts("TE1.2", rollout(source, [call])), outcome in outcomes)

    def test_payload_locations_and_removed_duplicates(self):
        for claim, key in (("TE2.1", "request"), ("TE2.2", "response")):
            for payload in ({"messages": []}, "malformed native payload"):
                for source in ("trajectory", "capture"):
                    self.assertTrue(self.accepts(claim, rollout(source, [{key: payload}])))
                    self.assertFalse(self.accepts(claim, rollout(source, [{key: payload}, {}])))
            self.assertTrue(self.accepts(claim, rollout("capture", [{key: None, key + "_raw": "raw bytes"}])))
            self.assertFalse(self.accepts(claim, rollout("capture", [{key + "_raw": ""}])))
            row = rollout("trajectory", [{key: {"fixture": "retained"}}]) | rollout("capture", [{}])
            self.assertTrue(self.accepts(claim, row))

    def test_reasoning_locations_are_explicit(self):
        responses = [
            {"choices": [{"message": {"reasoning_content": "reason"}}]},
            {"choices": [{"message": {"reasoning": "reason"}}]},
            {"output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "reason"}]}]},
            {"output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "reason"}]}]},
            {"content": [{"type": "thinking", "thinking": "reason"}]},
        ]
        for source in ("trajectory", "capture"):
            for response in responses:
                self.assertTrue(self.accepts("TE2.3", rollout(source, [{"response": response}])))
                self.assertFalse(self.accepts("TE2.3", rollout(source, [{"response": response}, {}])))
            for call in (
                {"response": {"choices": [{"message": {"content": "ordinary answer"}}]}},
                {"response": {"metadata": {"reasoning_content": "wrong location"}}},
                {"response_raw": json.dumps(responses[0])},
                {"token_stats": {"reasoning_tokens": 4}},
            ):
                self.assertFalse(self.accepts("TE2.3", rollout(source, [call])))
        self.assertTrue(self.accepts("TE2.3", rollout("capture", [{"reasoning_content": "retained"}])))
        self.assertFalse(self.accepts("TE2.3", rollout("trajectory", [{"reasoning_content": "wrong location"}])))

    def test_media_locations_and_malformed_second_block(self):
        image = {"type": "input_image", "image_url": "data:image/png;base64,YQ=="}
        calls = [
            {"request": {"input": [{"role": "user", "content": [image]}]}},
            {"request": {"input": [image]}},
            {
                "request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": "https://example.org/image.png"}}],
                        }
                    ]
                }
            },
            {
                "request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": "YQ=="},
                                }
                            ],
                        }
                    ]
                }
            },
            {
                "request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "input_audio", "input_audio": {"data": "YQ==", "format": "wav"}}],
                        }
                    ]
                }
            },
            {"response": {"choices": [{"message": {"audio": {"id": "audio-1", "data": "YQ=="}}}]}},
        ]
        for source in ("trajectory", "capture"):
            for call in calls:
                self.assertTrue(self.accepts("TE2.4", rollout(source, [call])))
                self.assertFalse(self.accepts("TE2.4", rollout(source, [call, {}])))
            for call in (
                {"request": {"metadata": image}},
                {"response": {"id": "not-audio", "data": "YQ=="}},
                {"request": {"input": [image, {"type": "input_image"}]}},
                {"request": {"messages": [{"content": [image, {"type": "input_audio"}]}]}},
            ):
                self.assertFalse(self.accepts("TE2.4", rollout(source, [call])))

    def test_invocations_in_trajectory_or_mixed_observations(self):
        invocation = {"kind": "agent_invocation", "invocation_id": "agent-1", "model_calls": [{"model_call_id": "c"}]}
        no_calls = {"kind": "agent_invocation", "invocation_id": "agent-2", "model_calls": []}
        for records in ([invocation], [invocation, no_calls]):
            self.assertTrue(
                self.accepts("TE6.4", {"ng_trajectory": {"schema_version": "1.0", "invocations": records}})
            )
            self.assertTrue(
                self.accepts("TE6.4", {"ng_agent_observations": {"records": [{"kind": "tool_call"}, *records]}})
            )
        for records in ([no_calls], [{"kind": "tool_call"}], [invocation, {"kind": "agent_invocation"}]):
            self.assertFalse(self.accepts("TE6.4", {"ng_agent_observations": {"records": records}}))
        bad = copy.deepcopy(invocation)
        bad["model_calls"] = [{"response_id": "no-model-ref"}]
        self.assertFalse(self.accepts("TE6.4", {"ng_agent_observations": {"records": [bad]}}))

    def test_turn_selection_is_not_inferred_from_reference_list(self):
        turn = {
            "task_id": "t",
            "rollout_id": "r",
            "invocation_id": "agent-1",
            "turn_no": 1,
            "model_calls": [{"model_call_id": "c"}],
        }
        row = {"ng_trajectory": {"schema_version": "1.0", "turns": [turn]}}
        self.assertTrue(self.accepts("TE4.4", row))
        self.assertFalse(self.accepts("TE4.3", row))
        turn["selected_model_call"] = {"model_call_id": "c"}
        self.assertTrue(self.accepts("TE4.3", row))
        turn["model_calls"] = []
        self.assertFalse(self.accepts("TE4.4", row))

    def test_unknown_trajectory_version_does_not_get_interpreted(self):
        row = rollout("trajectory", [token_call("trajectory", "TE3.1", 3)])
        row["ng_trajectory"]["schema_version"] = "unsupported"
        self.assertFalse(self.accepts("TE3.1", row))
        row.update(rollout("capture", [token_call("capture", "TE3.1", 3)]))
        self.assertTrue(self.accepts("TE3.1", row))
