# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Task intents: https://github.com/jayl940712/webarena_benchmarks/blob/6a2977939b157b0ab9de7799bb089c721f1ac115/visualwebarena.jsonl
# Revision: 6a2977939b157b0ab9de7799bb089c721f1ac115; license: Apache-2.0.

import copy
import fcntl
import json

import pytest

from nemo_gym.web import EvaluatorInfrastructureError, WebTask
from resources_servers.webarena_browser import shared_state as state


INTENTS = {
    "92": "Add something like the blue one to my shopping cart.",
    "335": "Buy me a chocolate ice cream to the address same as the label shown in the image. My phone number is also (650) 555-0111 in case it is needed.",
    "336": 'Buy me the cheapest color photo printer (from the "Printers & Accessories") category to Emily\'s place as shown in the image. My phone number is also (650) 555-0111 in case it is needed..',
    "340": "Change my default shipping address to the place shown in the image.",
    "345": "Prepend the country code of South Korea to the phone number of my account profile.",
    "346": "Prepend the country code of the country shown in the image to the phone number of my account profile.",
    "678": "Navigate to my listing of the white car and change the price to $25000. Update the price in the description as well.",
    "751": "Navigate to my listing of the all-black bike and change the price (including in the description) to one dollar more than the cheapest bike on the site (ignoring bikes that cost $1).",
}
RUN_ID = "e" * 32


def _protected(task_id, *, epoch=1, run_id=RUN_ID):
    return WebTask(
        benchmark="visualwebarena",
        task_id=task_id,
        intent=INTENTS[task_id],
        sites=["classifieds" if task_id in {"678", "751"} else "shopping"],
        task_kwargs={"shared_state": {"run_id": run_id, "epoch": epoch}},
    )


@pytest.fixture
def active_site(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    (root / RUN_ID).mkdir()
    with (root / "site.lease").open("w+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)

        def epoch(number):
            lease.seek(0)
            lease.truncate()
            json.dump({"run_id": RUN_ID, "epoch": number, "phase": "running", "reset_verified": True}, lease)
            lease.flush()

        epoch(1)
        yield root, epoch


@pytest.mark.parametrize("task_id", list(INTENTS))
def test_audited_public_task_identities_are_retained(task_id):
    state.validate_state_identity(_protected(task_id))


def test_state_verification_compares_across_reset_epochs_and_rejects_contamination(monkeypatch, active_site):
    root, epoch = active_site
    observed = {"cart": ["snapshot item"], "address": "snapshot address"}
    monkeypatch.setattr(state, "_observe_account_state", lambda *a: ("shopping", copy.deepcopy(observed)))
    first = state.verify_initial_state(_protected("346"), None, {}, state_root=str(root), required=True)
    epoch(2)
    second = state.verify_initial_state(_protected("345", epoch=2), None, {}, state_root=str(root), required=True)
    assert first["state_sha256"] == second["state_sha256"]
    assert json.loads((root / RUN_ID / "baseline-shopping.json").read_text()) == observed
    epoch(3)
    observed["cart"].append("mutation from another task")
    with pytest.raises(EvaluatorInfrastructureError, match="did not restore"):
        state.verify_initial_state(_protected("336", epoch=3), None, {}, state_root=str(root), required=True)
    assert not (root / RUN_ID / "epoch-3-task-336.json").exists()


@pytest.mark.parametrize("failure", ["wrong-epoch", "wrong-intent", "escaped-run", "unlocked", "missing-context"])
def test_state_verification_rejects_invalid_ownership_before_browser_reads(monkeypatch, active_site, failure):
    root, epoch = active_site
    task = _protected("346")
    if failure == "wrong-epoch":
        epoch(2)
    elif failure == "wrong-intent":
        task.intent += " changed"
    elif failure == "escaped-run":
        task.task_kwargs["shared_state"]["run_id"] = "../other"
    elif failure == "unlocked":
        root = root.parent / "unlocked"
        (root / RUN_ID).mkdir(parents=True)
        (root / "site.lease").write_text("{}")
    else:
        task.task_kwargs.clear()
    monkeypatch.setattr(
        state, "_observe_account_state", lambda *a: pytest.fail("invalid ownership must precede reads")
    )
    with pytest.raises(EvaluatorInfrastructureError):
        state.verify_initial_state(task, None, {}, state_root=str(root), required=True)


def test_unconfigured_generic_evaluator_retains_legacy_behavior():
    task = _protected("345")
    task.task_kwargs.clear()
    assert state.verify_initial_state(task, None, {}, state_root=None, required=False) is None
    with pytest.raises(EvaluatorInfrastructureError, match="requires a clean-state coordinator"):
        state.verify_initial_state(task, None, {}, state_root=None, required=True)


def test_shared_normal_lease_cannot_impersonate_a_protected_epoch(monkeypatch, tmp_path):
    (tmp_path / RUN_ID).mkdir()
    with (tmp_path / "site.lease").open("w+") as lease:
        json.dump({"run_id": RUN_ID, "epoch": 1, "phase": "running", "reset_verified": True}, lease)
        lease.flush()
        fcntl.flock(lease, fcntl.LOCK_SH)
        monkeypatch.setattr(state, "_observe_account_state", lambda *a: pytest.fail("shared lease is not exclusive"))
        with pytest.raises(EvaluatorInfrastructureError, match="exclusive site lease"):
            state.verify_initial_state(_protected("346"), None, {}, state_root=str(tmp_path), required=True)


def test_visualwebarena_policy_does_not_change_webarena_task_kwargs():
    task = WebTask(benchmark="webarena", task_id="345", task_kwargs={"shared_state": {"unrelated": True}})
    assert state.verify_initial_state(task, None, {}, state_root=None, required=True) is None
