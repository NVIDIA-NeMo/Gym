# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every outcome identifies the code and literal grading contract that produced it."""

import pytest

from resources_servers.agentic_if.if_grading.configuration import config_hash, load_config
from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.engine import grade
from resources_servers.agentic_if.if_grading.judge import Judge


def test_every_outcome_retains_scoring_and_check_identity():
    # Identity is attached after either grading branch, so even a check that was never graded carries it.
    check = ResolvedCheck(
        id="test",
        kind="exact",
        parameters={"value": "DONE"},
        scope={"places": ["final"]},
        provenance={"catalog_revision": "catalog-abc", "rule_hash": "rule-123"},
    )
    evidence = SessionEvidence(
        status="incomplete",
        steps=[{"id": "a", "session_id": "s", "role": "assistant", "text": "other"}],
        issues=["missing call"],
    )
    config = load_config()
    judge = Judge(lambda body: pytest.fail("text check must not call model"), config=config)
    result = grade([check], evidence, judge=judge)
    assert result.checks[0].status == "error"
    assert result.checks[0].details["provenance"] == check.provenance
    assert len(result.checks[0].details["check_sha256"]) == 64
    assert result.metadata["scorer"] == {
        "module": "resources_servers.agentic_if.if_grading",
        "source_sha256": result.metadata["scorer"]["source_sha256"],
    }
    assert len(result.metadata["scorer"]["source_sha256"]) == 64
    assert result.metadata["judge"] == {"version": config["version"], "config_hash": config_hash(config)}
