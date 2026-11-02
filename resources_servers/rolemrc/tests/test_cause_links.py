# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.judge_adapter_cause_links import (
    assert_real_verifier_reply_preserves_the_original_judge_model_cause,
)


@pytest.mark.parametrize("adapter", ["rolemrc", "rolemrc_multi"])
async def test_real_verifier_reply_preserves_the_original_judge_model_cause(tmp_path, monkeypatch, adapter):
    await assert_real_verifier_reply_preserves_the_original_judge_model_cause(tmp_path, monkeypatch, adapter)
