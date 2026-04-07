# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for two levels of quality filtering of generated multi-turn datasets."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _parse_scores(scores: Any) -> dict[str, Any]:
    """Normalize judge outputs to dictionaries."""
    if isinstance(scores, str):
        return json.loads(scores)
    return scores or {}


def filter_high_quality(
    df: pd.DataFrame,
    # Stage 1: Ambiguous query quality
    min_correct_ambiguity_type: int = 3,
    min_naturalness: int = 3,
    min_clarification_usefulness: int = 3,
    min_schema_consistency: int = 4,
    # Stage 2: Conversation quality
    min_clarification_completeness: int = 3,
    min_clarification_structure: int = 3,
    min_user_behavior: int = 3,
    min_tool_validity: int = 4,
    min_conversation_coherence: int = 3,
    min_task_completion: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter generated data with two levels of quality control.

    Stage 1 checks ambiguous query quality.
    Stage 2 checks conversation quality.
    Records must pass both stages.
    """
    out = df.copy()
    out["_query_scores"] = out["ambiguous_query_judge"].apply(_parse_scores)
    out["_conv_scores"] = out["conversation_judge"].apply(_parse_scores)

    # Stage 1: ambiguous query quality
    query_is_valid = out["_query_scores"].apply(lambda x: x.get("is_valid", False)) == True  # noqa: E712
    query_ambiguity_ok = (
        out["_query_scores"].apply(lambda x: x.get("correct_ambiguity_type", 0)) >= min_correct_ambiguity_type
    )
    query_natural_ok = out["_query_scores"].apply(lambda x: x.get("naturalness", 0)) >= min_naturalness
    query_useful_ok = (
        out["_query_scores"].apply(lambda x: x.get("clarification_usefulness", 0)) >= min_clarification_usefulness
    )
    query_consistent_ok = (
        out["_query_scores"].apply(lambda x: x.get("schema_consistency", 0)) >= min_schema_consistency
    )
    query_passed = query_is_valid & query_ambiguity_ok & query_natural_ok & query_useful_ok & query_consistent_ok

    # Stage 2: conversation quality
    conv_is_valid = out["_conv_scores"].apply(lambda x: x.get("is_valid", False)) == True  # noqa: E712
    conv_clarification_ok = (
        out["_conv_scores"].apply(lambda x: x.get("clarification_completeness", 0)) >= min_clarification_completeness
    )
    conv_structure_ok = (
        out["_conv_scores"].apply(lambda x: x.get("clarification_structure", 0)) >= min_clarification_structure
    )
    conv_user_ok = out["_conv_scores"].apply(lambda x: x.get("user_behavior", 0)) >= min_user_behavior
    conv_tool_ok = out["_conv_scores"].apply(lambda x: x.get("tool_validity", 0)) >= min_tool_validity
    conv_coherence_ok = (
        out["_conv_scores"].apply(lambda x: x.get("conversation_coherence", 0)) >= min_conversation_coherence
    )
    conv_completion_ok = out["_conv_scores"].apply(lambda x: x.get("task_completion", 0)) >= min_task_completion
    conv_passed = (
        conv_is_valid
        & conv_clarification_ok
        & conv_structure_ok
        & conv_user_ok
        & conv_tool_ok
        & conv_coherence_ok
        & conv_completion_ok
    )

    final_passed = query_passed & conv_passed

    if verbose:
        n = len(out)
        print("\n=== Quality Filtering Results ===")
        print(f"Total records: {n}")
        print(f"\nStage 1 (Ambiguous Query): {query_passed.sum()}/{n} passed ({query_passed.mean() * 100:.0f}%)")
        print(
            f"  is_valid: {query_is_valid.sum()} "
            f"| ambiguity_type>={min_correct_ambiguity_type}: {query_ambiguity_ok.sum()} "
            f"| naturalness>={min_naturalness}: {query_natural_ok.sum()} "
            f"| usefulness>={min_clarification_usefulness}: {query_useful_ok.sum()} "
            f"| consistency>={min_schema_consistency}: {query_consistent_ok.sum()}"
        )
        print(f"\nStage 2 (Conversation): {conv_passed.sum()}/{n} passed ({conv_passed.mean() * 100:.0f}%)")
        print(
            f"  is_valid: {conv_is_valid.sum()} "
            f"| clarification>={min_clarification_completeness}: {conv_clarification_ok.sum()} "
            f"| structure>={min_clarification_structure}: {conv_structure_ok.sum()} "
            f"| user_behavior>={min_user_behavior}: {conv_user_ok.sum()} "
            f"| tool_validity>={min_tool_validity}: {conv_tool_ok.sum()} "
            f"| coherence>={min_conversation_coherence}: {conv_coherence_ok.sum()} "
            f"| completion>={min_task_completion}: {conv_completion_ok.sum()}"
        )
        print(f"\nFinal: {final_passed.sum()}/{n} passed ({final_passed.mean() * 100:.0f}%)")

    return out[final_passed].drop(columns=["_query_scores", "_conv_scores"]).reset_index(drop=True)


def show_rejection_reasons(df: pd.DataFrame, num_examples: int = 5) -> None:
    """Print example rejection reasons from both judges."""
    query_scores = df["ambiguous_query_judge"].apply(_parse_scores)
    conv_scores = df["conversation_judge"].apply(_parse_scores)

    for label, scores, query_col in [
        ("Ambiguous Query", query_scores, "ambiguity_query"),
        ("Conversation", conv_scores, "conversation"),
    ]:
        rejected = scores[scores.apply(lambda x: not x.get("is_valid", True))]
        print(f"\n=== {label} Issues ({len(rejected)}/{len(df)} rejected) ===")
        if len(rejected) == 0:
            print("  No issues found.")
            continue
        for i, (idx, s) in enumerate(rejected.head(num_examples).items()):
            try:
                row_preview = str(df.loc[idx, query_col])[:100]
            except (KeyError, IndexError):
                row_preview = "(preview unavailable)"
            print(f"  [{i + 1}] {row_preview}...")
            print(f"      Issues: {s.get('issues', 'N/A')}")
