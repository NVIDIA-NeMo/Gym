"""Self-check suite: synthetic rollouts exercising the ported semantics.

Run: ``python3 -m verify_spike.tests``
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from verify_spike.checks import verify
from verify_spike.gym_records import TOKEN_SEMANTICS, map_rollout
from verify_spike.models import Metrics, Step, ToolCall, Trajectory, WireCall, WireRecord
from verify_spike.wire import find_model_calls_dir, load_wire_record


def _row(
    *,
    output: "list[dict[str, Any]]",
    calls: "list[dict[str, Any]] | None" = None,
    task: int = 0,
    trial: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "_ng_task_index": task,
        "_ng_rollout_index": trial,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "policy system prompt"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "Hello"},
            ]
        },
        "response": {"output": output},
    }
    if calls is not None:
        row["ng_model_call_capture"] = {"rollout_id": f"{task}-{trial}", "calls": calls}
    return row


def _call(
    index: int,
    *,
    name: str = "policy_model",
    tokens_in: int = 10,
    tokens_out: int = 5,
    tool_call_ids: "tuple[str, ...]" = (),
) -> dict[str, Any]:
    return {
        "model_call_id": f"call-{index}",
        "call_index": index,
        "model_ref": {"name": name},
        "status_code": 200,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tool_calls": [{"call_id": call_id} for call_id in tool_call_ids],
    }


def _agent_step(
    step_id: int,
    *,
    metrics: "Metrics | None",
    llm_call_count: "int | None" = 1,
    message: str = "text",
    tool_calls: "tuple[ToolCall, ...]" = (),
) -> Step:
    return Step(
        step_id=step_id,
        source="agent",
        message=message,
        metrics=metrics,
        llm_call_count=llm_call_count,
        tool_calls=tool_calls,
    )


class TranscriptProjectionTest(unittest.TestCase):
    def test_assistant_run_folds_into_one_agent_step(self) -> None:
        # reasoning + message + function_call are one model call; only
        # the function_call_output closes the run.
        output = [
            {"type": "reasoning", "summary": [{"text": "think"}]},
            {"type": "message", "role": "assistant", "content": "doing it"},
            {
                "type": "function_call",
                "call_id": "fc1",
                "name": "lookup",
                "arguments": "{\"q\": 1}",
            },
            {"type": "function_call_output", "call_id": "fc1", "output": "ok"},
            {"type": "message", "role": "user", "content": "thanks"},
            {"type": "message", "role": "assistant", "content": "done"},
        ]
        trajectory = map_rollout(_row(output=output), dataset="test")
        agent = [s for s in trajectory.steps if s.source == "agent"]
        # seed greeting + folded step + final message
        self.assertEqual(len(agent), 3)
        self.assertEqual(agent[0].llm_call_count, 0)  # scripted greeting
        folded = agent[1]
        self.assertEqual(folded.reasoning_content, "think")
        self.assertEqual(folded.message, "doing it")
        self.assertEqual(
            [t.tool_call_id for t in folded.tool_calls], ["fc1"]
        )
        self.assertIsNotNone(folded.observation)
        self.assertEqual(folded.observation.results[0].content, "ok")
        self.assertEqual(trajectory.session_id, "0-0")

    def test_metrics_bind_positionally_and_by_threading(self) -> None:
        output = [
            {
                "type": "function_call",
                "call_id": "fc1",
                "name": "lookup",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "fc1", "output": "ok"},
            {"type": "message", "role": "assistant", "content": "done"},
        ]
        calls = [
            _call(0, tokens_in=100, tokens_out=10, tool_call_ids=("fc1",)),
            _call(1, tokens_in=200, tokens_out=20),
            # user-sim group must be excluded from positional binding
            _call(2, name="user_sim", tokens_in=999, tokens_out=99),
        ]
        trajectory = map_rollout(_row(output=output, calls=calls), dataset="t")
        body = [
            s
            for s in trajectory.steps
            if s.source == "agent" and not (s.extra or {}).get("from_seed")
        ]
        self.assertEqual(body[0].metrics.prompt_tokens, 100)
        self.assertEqual(body[1].metrics.prompt_tokens, 200)
        self.assertTrue(all(s.llm_call_count == 1 for s in body))

    def test_undecidable_policy_group_leaves_steps_unbound(self) -> None:
        # No tool calls anywhere: with two model groups the policy side
        # is not decidable in-row, so steps stay unbound and the verdict
        # flags missed metrics (never a zero-token step).
        output = [{"type": "message", "role": "assistant", "content": "hi"}]
        calls = [_call(0), _call(1, name="user_sim")]
        trajectory = map_rollout(_row(output=output, calls=calls), dataset="t")
        health = verify(trajectory, token_semantics=TOKEN_SEMANTICS)
        self.assertEqual(health.missed_metrics_count, 1)
        self.assertEqual(health.zero_token_turn_count, 0)
        self.assertFalse(health.is_clean)


class ScanTest(unittest.TestCase):
    def test_zero_token_classification(self) -> None:
        steps = (
            _agent_step(0, metrics=None, llm_call_count=0),  # deterministic
            _agent_step(1, metrics=None, llm_call_count=None),  # missed
            _agent_step(2, metrics=Metrics(0, 0), llm_call_count=1),  # anomalous
            _agent_step(3, metrics=Metrics(0, 0), llm_call_count=None, message=""),  # unknown+hollow
            _agent_step(4, metrics=Metrics(10, 5), llm_call_count=1),  # fine
        )
        health = verify(
            Trajectory(session_id="s", steps=steps),
            token_semantics="per_call",
        )
        self.assertEqual(health.deterministic_step_count, 1)
        self.assertEqual(health.missed_metrics_count, 1)
        self.assertEqual(health.anomalous_zero_token_count, 1)
        self.assertEqual(health.unknown_zero_token_count, 1)
        self.assertEqual(health.hollow_step_count, 1)
        self.assertEqual(health.zero_token_turn_count, 2)
        self.assertFalse(health.is_fully_zero)
        self.assertFalse(health.is_clean)

    def test_only_deterministic_steps_is_no_agent_work(self) -> None:
        steps = (_agent_step(0, metrics=None, llm_call_count=0),)
        health = verify(
            Trajectory(session_id="s", steps=steps),
            token_semantics="per_call",
        )
        self.assertTrue(health.has_no_agent_steps)
        self.assertFalse(health.is_clean)


class ReconcileTest(unittest.TestCase):
    def test_success_filter_runs_before_dedup(self) -> None:
        # A failed call and its byte-identical successful retry must NOT
        # collapse onto the failed row: 1 unique successful call, 1
        # failed call, no duplicates among successful calls.
        steps = (_agent_step(0, metrics=Metrics(10, 5), llm_call_count=1),)
        capture = WireRecord(
            found=True,
            calls=(
                WireCall("a", status_code=500, request_hash="h1"),
                WireCall(
                    "b",
                    status_code=200,
                    request_hash="h1",
                    prompt_tokens=10,
                    completion_tokens=5,
                ),
            ),
        )
        health = verify(
            Trajectory(session_id="s", steps=steps),
            token_semantics="per_call",
            capture=capture,
        )
        self.assertEqual(health.unique_capture_calls, 1)
        self.assertEqual(health.failed_captured_calls, 1)
        self.assertEqual(health.duplicate_captured_call_count, 0)
        self.assertEqual(health.duplicate_captured_call_all_count, 1)
        self.assertEqual(health.step_capture_delta, 0)
        self.assertTrue(health.token_sums_match)
        self.assertFalse(health.binding_disagreement)
        self.assertFalse(health.last_captured_call_non_200)
        self.assertTrue(health.is_clean)

    def test_binding_disagreement_on_token_pair_shift(self) -> None:
        steps = (
            _agent_step(0, metrics=Metrics(10, 5), llm_call_count=1),
            _agent_step(1, metrics=Metrics(20, 6), llm_call_count=1),
        )
        capture = WireRecord(
            found=True,
            calls=(
                WireCall("a", 200, "h1", prompt_tokens=20, completion_tokens=6),
                WireCall("b", 200, "h2", prompt_tokens=10, completion_tokens=5),
            ),
        )
        health = verify(
            Trajectory(session_id="s", steps=steps),
            token_semantics="per_call",
            capture=capture,
        )
        self.assertTrue(health.binding_disagreement)
        # Flags never gate: the rollout stays clean.
        self.assertTrue(health.is_clean)

    def test_deterministic_steps_are_not_countable(self) -> None:
        steps = (
            _agent_step(0, metrics=None, llm_call_count=0),
            _agent_step(1, metrics=Metrics(10, 5), llm_call_count=1),
        )
        capture = WireRecord(
            found=True,
            calls=(
                WireCall("a", 200, "h1", prompt_tokens=10, completion_tokens=5),
            ),
        )
        health = verify(
            Trajectory(session_id="s", steps=steps),
            token_semantics="per_call",
            capture=capture,
        )
        self.assertEqual(health.step_capture_delta, 0)


class SidecarTest(unittest.TestCase):
    def _write_sidecar(
        self, capture_dir: Path, name: str, records: "list[dict[str, Any]]"
    ) -> None:
        path = capture_dir / name
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def _record(
        self,
        call_id: str,
        *,
        name: str,
        first_message: str,
        tool_ids: "tuple[str, ...]" = (),
    ) -> dict[str, Any]:
        return {
            "model_call_id": call_id,
            "model_ref": {"name": name},
            "status_code": 200,
            "request": {"messages": [{"role": "system", "content": first_message}]},
            "response": {
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"id": tid} for tid in tool_ids]
                        }
                    }
                ],
            },
        }

    def test_policy_discrimination_by_seed_system_prompt(self) -> None:
        output = [{"type": "message", "role": "assistant", "content": "hi"}]
        trajectory = map_rollout(_row(output=output), dataset="t")
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "model_calls"
            capture_dir.mkdir()
            self._write_sidecar(
                capture_dir,
                "0-0.capture.jsonl",
                [
                    self._record("p1", name="policy", first_message="policy system prompt"),
                    self._record("u1", name="usersim", first_message="user sim prompt"),
                    self._record("p2", name="policy", first_message="policy system prompt"),
                ],
            )
            record = load_wire_record(capture_dir, trajectory)
        self.assertTrue(record.found)
        self.assertEqual(len(record.calls), 2)
        self.assertEqual(record.non_policy_calls, 1)
        self.assertFalse(record.discriminator_disagreement)

    def test_attempt_selection_keeps_last_and_counts_dropped(self) -> None:
        output = [{"type": "message", "role": "assistant", "content": "hi"}]
        trajectory = map_rollout(_row(output=output), dataset="t")
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "model_calls"
            capture_dir.mkdir()
            self._write_sidecar(
                capture_dir,
                "0-0.capture.jsonl",
                [
                    self._record("a", name="m", first_message="x"),
                    self._record("b", name="m", first_message="x"),
                ],
            )
            self._write_sidecar(
                capture_dir,
                "0-0-a1.capture.jsonl",
                [self._record("c", name="m", first_message="x")],
            )
            # A different rollout's file must not match rollout 0-0.
            self._write_sidecar(
                capture_dir,
                "0-01.capture.jsonl",
                [self._record("z", name="m", first_message="x")],
            )
            record = load_wire_record(capture_dir, trajectory)
        self.assertEqual(len(record.calls), 1)
        self.assertEqual(record.calls[0].call_id, "c")
        self.assertEqual(record.retry_sessions, 1)
        self.assertEqual(record.dropped_calls, 2)

    def test_missing_capture_dir_is_fail_soft(self) -> None:
        output = [{"type": "message", "role": "assistant", "content": "hi"}]
        trajectory = map_rollout(_row(output=output), dataset="t")
        record = load_wire_record(None, trajectory)
        self.assertFalse(record.found)
        health = verify(
            trajectory, token_semantics=TOKEN_SEMANTICS, capture=record
        )
        self.assertFalse(health.capture_found)
        self.assertIsNone(health.unique_capture_calls)

    def test_find_model_calls_dir_checks_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model_calls").mkdir()
            results = root / "results"
            results.mkdir()
            self.assertEqual(
                find_model_calls_dir(results), root / "model_calls"
            )
            self.assertEqual(
                find_model_calls_dir(root), root / "model_calls"
            )


if __name__ == "__main__":
    unittest.main()
