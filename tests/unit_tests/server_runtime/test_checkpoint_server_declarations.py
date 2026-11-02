# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources servers that declare how they take part in partial-rollout checkpoints.

A server that declares nothing is restart-only, so a checkpoint retires every rollout that uses it.
These declarations let checkpoints continue rollouts on the servers most used for training.
The declarations are read from source because each server's dependencies live in its own environment.
"""

import ast
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]

DECLARED = {
    "code_gen": "CompCodingResourcesServer",
    "competitive_coding_challenges": "CompetitiveCodingChallengesResourcesServer",
    "equivalence_llm_judge": "LLMJudgeResourcesServer",
    "math_with_judge": "LibraryJudgeMathResourcesServer",
    "abstention": "AbstentionServer",
    "calendar": "CalendarResourcesServer",
    "equivalence_rule": "EquivalenceRuleResourcesServer",
    "ether0": "Ether0ResourcesServer",
    "format_verification": "FormatVerificationResourcesServer",
    "instruction_following": "InstructionFollowingResourcesServer",
    "inverse_if": "InverseIFServer",
    "jailbreak_detection": "JailbreakDetectionResourcesServer",
    "mcqa": "MCQAResourcesServer",
    "multichallenge": "MultiChallengeServer",
    "nvarc": "NVARCResourcesServer",
    "reasoning_gym": "ReasoningGymResourcesServer",
    "structured_outputs": "StructuredOutputsResourcesServer",
    "terminus_judge": "TerminusJudgeResourcesServer",
    "string_match": "StringMatchResourcesServer",
    "partner_if": "PartnerIFResourcesServer",
}


def class_attributes(server: str, class_name: str) -> dict[str, object]:
    tree = ast.parse((REPO / "resources_servers" / server / "app.py").read_text())
    [cls] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    return {
        target.id: ast.literal_eval(node.value)
        for node in cls.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.startswith("checkpoint_")
    }


@pytest.mark.parametrize("server", sorted(DECLARED))
def test_server_is_stateless_with_a_replayable_verification(server: str) -> None:
    assert class_attributes(server, DECLARED[server]) == {
        "checkpoint_mode": "stateless",
        "checkpoint_verify": "replay",
    }


def test_genrm_exports_completed_cohorts_and_replays_verification() -> None:
    """GenRM's cohorts span sessions, so it exports each session's share of a completed cohort."""
    assert class_attributes("genrm_compare", "GenRMCompareResourcesServer") == {
        "checkpoint_mode": "exported",
        "checkpoint_verify": "replay",
    }
