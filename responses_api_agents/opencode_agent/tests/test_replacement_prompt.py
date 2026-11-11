# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from responses_api_agents.opencode_agent.app import _serialize_replacement_config


@pytest.mark.parametrize("config", [{}, {"agent": {}}, {"agent": {"build": {"steps": 8}}}])
def test_absent_replacement_prompt_preserves_configuration(config):
    assert json.loads(_serialize_replacement_config(config)) == config


def test_replacement_prompt_preserves_literal_braces():
    config = {"agent": {"build": {"prompt": "Use {env:DO_NOT_SUBSTITUTE} literally."}}}
    encoded = _serialize_replacement_config(config)
    assert json.loads(encoded) == config
    assert "{env:DO_NOT_SUBSTITUTE}" not in encoded
