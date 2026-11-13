# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier control metadata stays out of the pinned actor conversation."""

import copy

import pytest
from fastapi import HTTPException

from benchmarks.partner_if.tests.test_replay import PARAMS
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.ngc_recorded_context.native_agent import ReplayActivation, replay_activation
from responses_api_agents.ngc_recorded_context.saved_prefix import verification_digest


def test_first_activation_accepts_the_unchanged_registry_input():
    body = ReplayActivation.model_validate(PARAMS)
    assert body.parent_turn_limit == 1
    expected = NeMoGymResponseCreateParamsNonStreaming.model_validate(PARAMS)
    assert body.model_dump(exclude_none=True, exclude_unset=True) == expected.model_dump(
        exclude_none=True, exclude_unset=True
    )


def test_continuation_controls_do_not_change_the_actor_input(tmp_path):
    params = copy.deepcopy(PARAMS)
    params["metadata"] = {"public_fixture": "synthetic"}
    body = ReplayActivation.model_validate(
        params
        | {
            "metadata": params["metadata"]
            | {
                "ngc_replay_parent_turn_limit": "2",
                "ngc_replay_saved_prefix": str(tmp_path),
                "ngc_replay_saved_prefix_sha256": "a" * 64,
            }
        }
    )
    decoded = replay_activation(body)
    assert decoded.parent_turn_limit == 2
    assert decoded.saved_prefix == tmp_path
    assert decoded.saved_prefix_sha256 == "a" * 64
    assert decoded.model_dump(
        exclude_none=True, exclude_unset=True, exclude={"parent_turn_limit", "saved_prefix", "saved_prefix_sha256"}
    ) == NeMoGymResponseCreateParamsNonStreaming.model_validate(params).model_dump(
        exclude_none=True, exclude_unset=True
    )


@pytest.mark.parametrize("limit", ["0", "33", "not-a-number"])
def test_invalid_continuation_limits_are_rejected(limit):
    with pytest.raises(HTTPException, match="Invalid replay continuation metadata"):
        replay_activation(
            ReplayActivation.model_validate(PARAMS | {"metadata": {"ngc_replay_parent_turn_limit": limit}})
        )


def test_ambiguous_direct_and_metadata_controls_are_rejected():
    with pytest.raises(HTTPException, match="cannot mix"):
        replay_activation(
            ReplayActivation.model_validate(
                PARAMS | {"parent_turn_limit": 1, "metadata": {"ngc_replay_parent_turn_limit": "2"}}
            )
        )


def test_verification_digest_covers_receipts_and_rejects_escaping_symlinks(tmp_path):
    names = (
        "identity.json",
        "events.json",
        "drafts.json",
        "terminal_reply.json",
        "result.json",
        "visible-output.json",
        "ordering-verification-input.json",
        "native-step.json",
    )
    for name in names:
        (tmp_path / name).write_text("{}")
    original = verification_digest(tmp_path)
    (tmp_path / "events.json").write_text("[]")
    assert verification_digest(tmp_path) != original
    (tmp_path / "events.json").unlink()
    (tmp_path / "events.json").symlink_to(tmp_path.parent / "outside.json")
    with pytest.raises(ValueError, match="escapes capture"):
        verification_digest(tmp_path)
