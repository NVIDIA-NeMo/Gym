# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The real saved-row payload builder shared by core and component contract tests."""

from nemo_gym.eval.rollout_reverification import InputRolloutPair, _build_verify_payload


def build_reverification_payload(*, input_row: dict[str, object], saved_row: dict[str, object]) -> dict[str, object]:
    """Build a verifier request without exposing evaluation internals as a runtime API."""
    return _build_verify_payload(InputRolloutPair(input=input_row, rollout=saved_row))
