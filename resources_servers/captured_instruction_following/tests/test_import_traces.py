# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline import contracts, with independently authored fictional traces."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from resources_servers.captured_instruction_following.import_traces import (
    HISTORY_MARKER,
    TOOLS_MARKER,
    canonical_json,
    captured_prefix,
    convert_record,
    file_sha256,
    prepare_dataset,
    value_sha256,
)


def output_message(text):
    return {
        "type": "message",
        "id": "held-message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def source_fixture(identifier="trace-1", native=False):
    tools = [
        {
            "type": "function",
            "name": "lookup_record",
            "strict": False,
            "description": "Read a fictional local registry record.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        }
    ]
    request = {
        "input": [
            {"type": "message", "role": "user", "content": "Return the fictional record's status in one sentence."}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "model": "historical-model",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_output_tokens": 4096,
        "reasoning": {"effort": "high"},
        "metadata": {"captured": True},
    }
    trace = {
        "responses_create_params": request,
        "response": {
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "SECRET HELD OUT REASONING"}]},
                output_message("SECRET FAILED BASELINE"),
            ]
        },
        "reward": 0.0,
        "verifier_metadata": {},
    }
    if native:
        trace["verifier_metadata"] = {"expected_function": "lookup_record", "expected_params": {"id": "fictional-7"}}
    row = {
        "benchmark": "fictional_records",
        "orig_trajectory_id": identifier,
        "task_index": 5,
        "comparison_profile": "case_insensitive_scalars",
    }
    record = {
        "benchmark": row["benchmark"],
        "trajectory_id": identifier,
        "task_index": 5,
        "source_trajectory": trace,
        "effective_verdict": {"primary_failure_mode": "illustrative-obligation"},
    }
    return row, record


class TestCapturedTraceImport(unittest.TestCase):
    def test_native_request_keeps_input_and_tool_definitions_exactly(self):
        row, record = source_fixture(native=True)
        original = copy.deepcopy(record)
        prepared, baseline, receipt = convert_record(row, record)
        params = prepared["responses_create_params"]
        self.assertEqual(params["input"], record["source_trajectory"]["responses_create_params"]["input"])
        self.assertEqual(params["tools"], record["source_trajectory"]["responses_create_params"]["tools"])
        self.assertEqual(params["max_output_tokens"], 32768)
        self.assertEqual(params["temperature"], 0.7)
        self.assertNotIn("model", params)
        self.assertNotIn("metadata", params)
        self.assertNotIn("expected_function", canonical_json(params))
        self.assertNotIn("SECRET", canonical_json(params))
        self.assertEqual(prepared["verifier_metadata"]["expected_params"], {"id": "fictional-7"})
        self.assertEqual(prepared["verifier_metadata"]["evaluation_request_sha256"], value_sha256(params))
        self.assertEqual(baseline["original_reward"], 0.0)
        self.assertEqual(receipt["omitted_request_fields"]["reasoning"], {"effort": "high"})
        self.assertNotIn("reward", prepared)
        self.assertEqual(record, original)

    def test_text_wrapper_omits_entire_held_out_initial_turn(self):
        row, record = source_fixture()
        prepared, baseline, _ = convert_record(row, record)
        request = prepared["responses_create_params"]
        content = request["input"][0]["content"]
        tool_text, history_text = content.split(TOOLS_MARKER)[1].split(HISTORY_MARKER)
        self.assertEqual(json.loads(tool_text), record["source_trajectory"]["responses_create_params"]["tools"])
        self.assertEqual(json.loads(history_text), record["source_trajectory"]["responses_create_params"]["input"])
        self.assertEqual(request["tools"], [])
        self.assertNotIn("SECRET", content)
        self.assertIn("SECRET FAILED BASELINE", canonical_json(baseline))
        self.assertEqual(prepared["verifier_metadata"]["criteria_status"], "pending_review")

    def test_prefix_after_tool_results_keeps_ids_without_final_reasoning(self):
        row, record = source_fixture()
        previous = [
            {
                "type": "reasoning",
                "id": "past-reasoning",
                "summary": [{"type": "summary_text", "text": "Earlier tool decision"}],
            },
            {
                "type": "function_call",
                "id": "raw-function-id",
                "call_id": "original-call-id",
                "name": "lookup_record",
                "arguments": '{"id":"fictional-7"}',
            },
            {"type": "function_call_output", "call_id": "original-call-id", "output": '{"status":"available"}'},
        ]
        record["source_trajectory"]["response"]["output"] = (
            previous + record["source_trajectory"]["response"]["output"]
        )
        prepared, baseline, receipt = convert_record(row, record)
        content = prepared["responses_create_params"]["input"][0]["content"]
        history = json.loads(content.split(HISTORY_MARKER)[1])
        self.assertEqual(history[1:], previous)
        self.assertNotIn("SECRET", content)
        self.assertEqual(receipt["continuation_boundary"]["held_out_begin_index"], 3)
        self.assertEqual(baseline["continuation_boundary"]["held_out_types"], ["reasoning", "message"])
        self.assertFalse(receipt["continuation_boundary"]["earlier_action_correctness_retested"])

    def test_request_instructions_and_string_input_have_explicit_history_receipts(self):
        row, record = source_fixture()
        record["source_trajectory"]["responses_create_params"].update(
            input="A plain fictional request.", instructions="Keep the response short."
        )
        prepared, _, receipt = convert_record(row, record)
        content = prepared["responses_create_params"]["input"][0]["content"]
        history = json.loads(content.split(HISTORY_MARKER)[1])
        self.assertEqual(
            [(m["role"], m["content"]) for m in history],
            [("system", "Keep the response short."), ("user", "A plain fictional request.")],
        )
        self.assertEqual(len(receipt["continuation_boundary"]["history_insertions"]), 2)
        self.assertEqual(receipt["continuation_boundary"]["history_insertions"][0]["history_index"], 1)

    def test_unmaterialized_remote_requests_and_unknown_output_types_fail(self):
        for change in ("remote", "output"):
            row, record = source_fixture()
            if change == "remote":
                record["source_trajectory"]["responses_create_params"]["previous_response_id"] = "external-state"
            else:
                record["source_trajectory"]["response"]["output"] = [{"type": "unknown_tool_action"}]
            with self.assertRaises(ValueError):
                convert_record(row, record)

    def test_missing_continuation_after_tool_result_fails(self):
        with self.assertRaisesRegex(ValueError, "No assistant continuation"):
            captured_prefix({"input": []}, [{"type": "function_call_output", "call_id": "c", "output": "ok"}])

    def test_reference_map_must_match_original_gold_and_explicit_profile(self):
        row, record = source_fixture(native=True)
        reference = {
            "expected_call": {"name": "another_function", "arguments": {}},
            "native_reference_profile": "normalized_scalars",
        }
        with self.assertRaisesRegex(ValueError, "differs"):
            convert_record(row, record, reference=reference)
        row.pop("comparison_profile")
        with self.assertRaisesRegex(ValueError, "comparison_profile"):
            convert_record(row, record)
        reference["expected_call"] = {"name": "lookup_record", "arguments": {"id": "fictional-7"}}
        prepared, _, _ = convert_record(row, record, reference=reference)
        self.assertEqual(prepared["verifier_metadata"]["comparison_profile"], "normalized_scalars")

    def make_inventory(self, root, native=True):
        sources = root / "source_records"
        sources.mkdir()
        inventory = []
        for index in range(2):
            row, record = source_fixture("attempt-" + str(index), native=native and index == 0)
            path = sources / (str(index) + ".json")
            path.write_text(canonical_json(record))
            row.update(source_record_path=str(path.relative_to(root)), source_record_sha256=file_sha256(path))
            inventory.append(row)
        manifest = root / "selected.jsonl"
        manifest.write_text("".join(canonical_json(row) + "\n" for row in inventory))
        return manifest, inventory

    def test_prepare_relative_paths_hashes_baseline_refs_and_repeat_groups(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, _ = self.make_inventory(root)
            out = root / "private"
            summary = prepare_dataset(manifest, out, expected_count=2)
            rows = [json.loads(line) for line in (out / "prepared.jsonl").read_text().splitlines()]
            baseline = [json.loads(line) for line in (out / "baseline_outputs.jsonl").read_text().splitlines()]
            self.assertEqual(summary["evaluation_modes"], {"native_next_turn": 1, "single_turn_text_replay": 1})
            self.assertEqual(summary["num_task_groups"], 1)
            self.assertNotEqual(rows[0]["task_id"], rows[1]["task_id"])
            self.assertEqual(rows[0]["task_group_id"], rows[1]["task_group_id"])
            for row, historical in zip(rows, baseline, strict=True):
                self.assertEqual(row["verifier_metadata"]["private_baseline_ref"]["sha256"], value_sha256(historical))
                self.assertEqual(
                    row["verifier_metadata"]["source_identity"]["orig_trajectory_id"], row["orig_trajectory_id"]
                )
                self.assertNotIn("SECRET", canonical_json(row["responses_create_params"]))
            for path in out.iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)

    def test_tampered_source_duplicate_identity_and_count_mismatch_fail_without_publish(self):
        for defect in ("tamper", "duplicate", "count"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                manifest, inventory = self.make_inventory(root)
                if defect == "tamper":
                    (root / inventory[0]["source_record_path"]).write_text("{}")
                if defect == "duplicate":
                    manifest.write_text(canonical_json(inventory[0]) + "\n" + canonical_json(inventory[0]) + "\n")
                with self.assertRaises(ValueError):
                    prepare_dataset(manifest, root / "private", expected_count=3 if defect == "count" else 2)
                self.assertFalse((root / "private").exists())

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, _ = self.make_inventory(root)
            out = root / "private"
            out.mkdir()
            marker = out / "keep"
            marker.write_text("user data")
            with self.assertRaisesRegex(ValueError, "exists"):
                prepare_dataset(manifest, out)
            self.assertEqual(marker.read_text(), "user data")

    def test_private_dataset_cannot_be_created_inside_git_checkout(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, _ = self.make_inventory(root)
            checkout = root / "checkout"
            checkout.mkdir()
            (checkout / ".git").write_text("gitdir: private-worktree-metadata\n")
            with self.assertRaisesRegex(ValueError, "outside Git"):
                prepare_dataset(manifest, checkout / "private")
            self.assertFalse((checkout / "private").exists())


if __name__ == "__main__":
    unittest.main()
