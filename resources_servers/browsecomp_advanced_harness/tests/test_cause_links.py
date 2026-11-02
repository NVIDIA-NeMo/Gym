# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.judge_adapter_cause_links import (
    assert_browsecomp_received_unparseable_verdict_keeps_zero_and_transport_observations,
    assert_real_verifier_reply_preserves_the_original_judge_model_cause,
)


async def test_real_verifier_reply_preserves_the_original_judge_model_cause(tmp_path, monkeypatch):
    await assert_real_verifier_reply_preserves_the_original_judge_model_cause(
        tmp_path, monkeypatch, "browsecomp_advanced_harness"
    )


@pytest.mark.parametrize("error_first", [False, True])
async def test_received_unparseable_verdict_keeps_zero_and_transport_observations(monkeypatch, error_first):
    await assert_browsecomp_received_unparseable_verdict_keeps_zero_and_transport_observations(
        monkeypatch, error_first
    )
