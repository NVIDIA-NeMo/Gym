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
import pytest

from resources_servers.enterpriseops_gym.export_eval_telemetry import (
    align_tool_latencies,
    export_rows,
    render_question,
)


def make_turn(turn, input_tokens, output_tokens, cached=0, tools=(), text="", start=0, end=0):
    return {
        "turn": turn,
        "timestamp": f"2026-07-07T00:00:0{turn}+00:00",
        "duration_ms": 100.0,
        "model": "test-model",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached,
        "reasoning_tokens": 0,
        "tool_call_names": list(tools),
        "num_tool_calls": len(tools),
        "assistant_text": text,
        "output_start_index": start,
        "output_end_index": end,
    }


def make_row(task_index, rollout_index, turns, latencies, reward=1.0, output=None):
    return {
        "_ng_task_index": task_index,
        "_ng_rollout_index": rollout_index,
        "reward": reward,
        "verifier_metadata": {"task_id": f"task_{task_index}", "domain": "csm", "mode": "oracle"},
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Policy."},
                {"role": "user", "content": "Do the thing."},
            ]
        },
        "response": {"output": output or []},
        "turns": turns,
        "num_turns": len(turns),
        "tool_latencies_ms": latencies,
    }


class TestExport:
    def test_full_schema_two_passes(self) -> None:
        # Pass 1: two turns (tool call then final answer); pass 2: single-turn answer.
        pass1 = make_row(
            0,
            0,
            turns=[
                make_turn(0, 1000, 20, cached=0, tools=("update_entitlement",), start=0, end=2),
                make_turn(1, 1500, 30, cached=990, text="Done.", start=2, end=3),
            ],
            latencies=[{"tool": "update_entitlement", "gym": "g", "latency_ms": 42.0}],
        )
        pass2 = make_row(0, 1, turns=[make_turn(0, 900, 15, text="Nope.")], latencies=[], reward=0.0)

        rows = export_rows([pass1, pass2])
        assert len(rows) == 3

        first, second, third = rows
        assert first["task_id"] == "task_0"
        assert first["trial_name"] == "task_0.1-of-2.default"
        assert third["trial_name"] == "task_0.2-of-2.default"

        # Turn indexing and totals
        assert (first["turn"], first["num_turns"]) == (0, 2)
        assert (second["turn"], second["num_turns"]) == (1, 2)

        # Token accounting: stacked == input (no compaction), new = input - cached
        assert first["input_length"] == 1000 and first["stacked_input"] == 1000
        assert second["cached input length"] == 990
        assert second["new input length"] == 510

        # Tool fields
        assert first["tool_call names"] == "update_entitlement"
        assert first["num_steps"] == 1
        assert first["per-tool latency"] == 42
        assert second["per-tool latency"] is None

        # Completion / resolution semantics
        assert first["task_complete"] is False
        assert second["task_complete"] is True
        assert first["is_resolved"] is True and third["is_resolved"] is False

        # Compaction fields are explicitly null (no compaction in this harness)
        for field in ("tokensBefore", "tokensAfter", "content before compaction", "content after compaction"):
            assert first[field] is None

        # Prompt rendering
        assert "[system]\nPolicy." in first["question"]
        assert "[user]\nDo the thing." in first["question"]
        assert second["answer"] == "Done."
        assert second["timestamp"] == "2026-07-07T00:00:01+00:00"

    def test_answer_includes_reasoning_summaries(self) -> None:
        output = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I should update it."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done."}]},
        ]
        row = make_row(0, 0, turns=[make_turn(0, 10, 5, text="Done.", start=0, end=2)], latencies=[], output=output)
        exported = export_rows([row])[0]
        assert exported["answer"] == "I should update it.\n\nDone."

    def test_malformed_tool_calls_align_to_null_without_shifting(self) -> None:
        # Turn 0 issued two calls but the second was malformed (never reached the proxy);
        # turn 1 issued one good call. Latencies: [good_call_A, good_call_C].
        turns = [
            make_turn(0, 10, 5, tools=("tool_a", "tool_b")),
            make_turn(1, 10, 5, tools=("tool_c",)),
        ]
        latencies = [
            {"tool": "tool_a", "gym": "g", "latency_ms": 10.0},
            {"tool": "tool_c", "gym": "g", "latency_ms": 30.0},
        ]
        aligned = align_tool_latencies(turns, latencies)
        assert aligned[0][0]["tool"] == "tool_a"
        assert aligned[0][1] is None  # malformed: no latency, no shift
        assert aligned[1][0]["tool"] == "tool_c"

    def test_rows_without_turns_raise_helpful_error(self) -> None:
        row = make_row(0, 0, turns=[], latencies=[])
        del row["turns"]
        with pytest.raises(ValueError, match="turn_logging_agent"):
            export_rows([row])

    def test_render_question_handles_structured_content(self) -> None:
        text = render_question({"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]})
        assert text.startswith("[user]")
