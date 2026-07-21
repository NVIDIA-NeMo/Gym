# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from responses_api_agents.swe_agents.app import SWEBenchMetrics
from responses_api_agents.swe_agents.refine_app import (
    SWEBenchRefineConfig,
    _append_seed_to_problem_metadata,
    _build_chain_metrics,
    _build_group_hash,
    _build_padding_transport,
    _build_refine_v1_seed,
    _build_refine_v3_seed,
    _extract_failure_snippet,
    _split_key_and_raw_verify_context,
    _truncate_middle,
)


def test_group_hash_uses_swe_metadata_when_input_is_empty() -> None:
    first = SimpleNamespace(
        input=[],
        metadata={
            "dataset_name": "swe-dataset",
            "instance_id": "repo__issue-1",
            "problem_statement": "A",
        },
    )
    same_reordered = SimpleNamespace(
        input=[],
        metadata={
            "problem_statement": "changed transport copy",
            "instance_id": "repo__issue-1",
            "dataset_name": "swe-dataset",
        },
    )
    second = SimpleNamespace(
        input=[],
        metadata={
            "dataset_name": "swe-dataset",
            "instance_id": "repo__issue-2",
            "problem_statement": "B",
        },
    )

    assert _build_group_hash(first) == _build_group_hash(same_reordered)
    assert _build_group_hash(first) != _build_group_hash(second)


def _refine_config(**overrides) -> SWEBenchRefineConfig:
    values = {
        "host": "localhost",
        "port": 9003,
        "name": "test_refine_agent",
        "entrypoint": "responses_api_agents/swe_agents/refine_app.py",
        "model_server": ModelServerRef(type="responses_api_models", name="test"),
    }
    values.update(overrides)
    return SWEBenchRefineConfig(**values)


def test_refine_round_config_accepts_canonical_and_legacy_names() -> None:
    assert _refine_config(max_refine_rounds=3).max_refine_rounds == 3
    assert _refine_config(max_attempts=4).max_refine_rounds == 4
    assert _refine_config(refine_strategy="compact_raw").refine_strategy == "compact_raw"
    assert _refine_config().refine_failure_snippet_chars == 3000
    assert _refine_config(skip_reset_after_first=True).skip_reset_after_initial_round is True


def test_truncate_middle_respects_approximate_token_budget() -> None:
    text = "abcdefgh12345678"

    truncated = _truncate_middle(text, max_tokens=2)

    assert truncated.startswith("abcd")
    assert truncated.endswith("5678")
    assert "diff truncated to fit carry-over budget" in truncated


def test_truncate_middle_zero_budget_keeps_full_patch() -> None:
    text = "full patch"

    assert _truncate_middle(text, max_tokens=0) == text


def test_build_refine_v1_seed_matches_eval_handoff() -> None:
    seed = _build_refine_v1_seed(
        patch="diff --git a/a.py b/a.py\n+fix",
        verify_feedback="FAILED tests/test_a.py::test_bug",
        max_patch_tokens=40000,
    )

    assert "previous automated refinement round did NOT resolve" in seed
    assert "```diff\ndiff --git a/a.py b/a.py\n+fix\n```" in seed
    assert "FAILED tests/test_a.py::test_bug" in seed
    assert "produce a correct, complete patch" in seed


def test_extract_failure_snippet_prefers_latest_traceback() -> None:
    feedback = (
        "Traceback (most recent call last):\nold failure\n"
        "noise\n"
        "Traceback (most recent call last):\nnew failure\nAssertionError: expected 2"
    )

    snippet = _extract_failure_snippet(feedback, max_chars=3000)

    assert "old failure" not in snippet
    assert snippet.startswith("Traceback (most recent call last):\nnew failure")
    assert "AssertionError: expected 2" in snippet


def test_split_key_and_raw_verify_context_removes_overlap() -> None:
    key = "FAILED tests/test_a.py::test_bug\nAssertionError: expected 2"
    raw = f"setup output\n{key}\nshort test summary"

    extracted_key, additional = _split_key_and_raw_verify_context(key, raw)

    assert extracted_key == key
    assert key not in additional
    assert "key verifier output shown above" in additional
    assert "short test summary" in additional


def test_build_refine_v3_seed_frontloads_compact_raw_evidence() -> None:
    failure = (
        "Traceback (most recent call last):\n"
        '  File "tests/test_a.py", line 10, in test_bug\n'
        "AssertionError: expected 2"
    )
    seed = _build_refine_v3_seed(
        patch="diff --git a/a.py b/a.py\n+fix",
        verify_feedback=f"setup output\n{failure}",
        max_patch_tokens=30000,
        max_failure_snippet_chars=3000,
    )

    assert "previous automated refine round did NOT resolve" in seed
    assert seed.index("Key verifier output:") < seed.index("Previous patch:")
    assert seed.count("AssertionError: expected 2") == 1
    assert "Additional verifier context:" in seed
    assert "You may keep, revise, or discard it" in seed
    assert "complete minimal patch from the clean repository" in seed


def test_append_seed_updates_openhands_instance_problem() -> None:
    metadata = {
        "problem_statement": "Fix the bug.",
        "instance_dict": json.dumps(
            {
                "instance_id": "repo__issue-1",
                "problem_statement": "Fix the bug.",
            }
        ),
    }

    refined = _append_seed_to_problem_metadata(metadata, "\n\n---\nRefine this patch.")
    instance_dict = json.loads(refined["instance_dict"])

    assert refined["problem_statement"].endswith("---\nRefine this patch.")
    assert instance_dict["problem_statement"] == refined["problem_statement"]
    assert json.loads(metadata["instance_dict"])["problem_statement"] == "Fix the bug."


def test_build_chain_metrics_reports_refine_rescue() -> None:
    refine_rounds = [
        {"metrics": SWEBenchMetrics(resolved=False)},
        {"metrics": SWEBenchMetrics(resolved=True)},
    ]

    metrics = _build_chain_metrics(
        refine_rounds,
        max_refine_rounds=2,
        refine_strategy="compact_raw",
    )

    assert metrics == {
        "refine_strategy": "compact_raw",
        "num_refine_rounds": 2,
        "max_refine_rounds": 2,
        "chain_resolved": True,
        "resolved_at_refine_round": 1,
        "initial_round_resolved": False,
        "refine_continued": True,
        "refine_rescued": True,
    }


def test_build_chain_metrics_reports_early_success_without_refine() -> None:
    refine_rounds = [{"metrics": SWEBenchMetrics(resolved=True)}]

    metrics = _build_chain_metrics(refine_rounds, max_refine_rounds=2)

    assert metrics["resolved_at_refine_round"] == 0
    assert metrics["initial_round_resolved"] is True
    assert metrics["refine_continued"] is False
    assert metrics["refine_rescued"] is False


def test_build_padding_transport_drops_token_heavy_fields_without_mutating_source() -> None:
    source_response = NeMoGymResponse.model_construct(
        id="response-1",
        created_at=123,
        model="test-model",
        object="response",
        output=[{"generation_token_ids": list(range(1000))}],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[{"type": "function", "name": "large_tool_schema"}],
        metadata={"large": "metadata" * 100},
    )

    padding_params, padding_response = _build_padding_transport(source_response)

    assert padding_params == {"input": ""}
    assert padding_response.output == []
    assert padding_response.tools == []
    assert padding_response.metadata is None
    assert source_response.output
    assert source_response.tools
    assert source_response.metadata
