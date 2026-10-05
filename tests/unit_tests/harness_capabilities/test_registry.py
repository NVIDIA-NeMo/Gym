# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry wiring must retain diagnostic meaning, ordering and short-circuiting."""

import json

import pytest

from nemo_gym.harness_capabilities import checker
from nemo_gym.harness_capabilities.contracts import PATH_MODELS, model_errors
from nemo_gym.harness_capabilities.reader import hydrate_record
from nemo_gym.harness_capabilities.registry import (
    ARTIFACT_CHECK_GROUPS,
    BEHAVIORAL_CHECKS,
    CHECKS,
    NATIVE_CHECKS,
    REPORT_CHECKS,
    behavioral_issue,
    check_catalog,
)
from tests.unit_tests.harness_capabilities.synthetic import evidence_record


def test_catalog_is_complete_and_resolves_implementations():
    from scripts.harness_conformance import runner

    definitions = [
        *NATIVE_CHECKS,
        *(check for group in ARTIFACT_CHECK_GROUPS for check in group.checks),
        *REPORT_CHECKS,
        *BEHAVIORAL_CHECKS,
    ]
    assert len(CHECKS) == len(definitions)  # Do not silently overwrite a duplicated ID.
    assert set(PATH_MODELS) == {check.locations[0].removeprefix("$.") for check in NATIVE_CHECKS}
    owners = {"checker": checker._RecordInspector, "contracts": {"model_errors": model_errors}, "runner": runner}
    for check in definitions:
        assert check.locations and check.requirement
        assert check.kind in {"schema", "semantic", "behavioral"}
        module, name = check.implementation.split(".")
        owner = owners[module]
        implementation = owner[name] if isinstance(owner, dict) else getattr(owner, name)
        assert callable(implementation), check.id
    assert json.loads(json.dumps(check_catalog()))[0]["id"] == NATIVE_CHECKS[0].id


def test_existing_failure_ids_resolve_to_requirements():
    record = evidence_record()
    record.update(mask_sample=True, failure_kind="", failure_reason="")
    result = checker.inspect_record(hydrate_record(record))
    failures = [f for f in result["findings"] if f["evidence"] == "TE-6"]
    assert [f["location"] for f in failures] == ["record/failure_kind", "record/failure_reason"]
    assert {f["assertion"] for f in failures} == {"verifier.error"}
    for finding in failures:
        check = CHECKS[f"{finding['evidence']}.{finding['assertion']}"]
        assert check.kind == "schema"
        assert "$" + finding["location"].removeprefix("record").replace("/", ".") in check.locations


def test_dispatch_uses_registered_groups(monkeypatch):
    verifier = next(group for group in ARTIFACT_CHECK_GROUPS if group.implementation == "check_verifier")
    monkeypatch.setattr(checker, "ARTIFACT_CHECK_GROUPS", (verifier,))
    record = evidence_record()
    record.pop("_ng_task_index")
    record["ng_trajectory"].pop("task_id")
    record["reward"] = None
    result = checker.inspect_record(hydrate_record(record))
    # Only the registered verifier ran; identity checking would also reject this input.
    assert [(f["evidence"], f["assertion"]) for f in result["findings"]] == [("TE-6", "reward.required")]


def test_schema_failure_short_circuits_artifact_dispatch(monkeypatch):
    def forbidden(_self):
        pytest.fail("semantic dispatch ran on an invalid native object")

    monkeypatch.setattr(checker._RecordInspector, "check_identity", forbidden)
    record = evidence_record()
    record["ng_trajectory"]["turns"][0]["turn_no"] = "one"
    result = checker.inspect_record(record)
    assert result["verdict"] == "not_fulfilled"
    assert any(f["assertion"] == "model.int_type" for f in result["findings"])
    assert all(row["verdict"] == "not_fulfilled" for row in result["evidence"].values())


def test_unknown_assertions_cannot_silently_enter_reports():
    inspector = checker._RecordInspector({}, "record", checker.EvidenceScope())
    with pytest.raises(KeyError, match="TE-6.unregistered"):
        inspector._fail("TE-6", "unregistered", "/reward", "new assertion")


def test_behavioral_ids_keep_existing_messages():
    assert behavioral_issue("probe.verifier.reward") == "rollout reward differs from the verifier witness"
    with pytest.raises(ValueError, match="no behavioral failure message"):
        behavioral_issue("TE-6.reward.required")
    with pytest.raises(ValueError, match="no behavioral failure message"):
        behavioral_issue("probe.model.protocol")
