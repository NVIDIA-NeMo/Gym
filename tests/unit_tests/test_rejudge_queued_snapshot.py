# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollout_recovery import run_lock
from nemo_gym.rollout_reverification import validate_settle_source_snapshot


def test_queued_snapshot_rechecked_under_original_lock(tmp_path):
    names = [
        "rollouts.jsonl",
        "rollouts_failures.jsonl",
        "rollouts_manifest.json",
        "rollouts_materialized_inputs.jsonl",
        "rollouts_resolved_config.yaml",
        "rollouts_attempts.jsonl",
    ]
    expected = {}
    for name in names:
        (tmp_path / name).write_bytes(b"original")
        expected[name] = hashlib.sha256(b"original").hexdigest()
    source = tmp_path / "rollouts.jsonl"
    with run_lock(source):
        validate_settle_source_snapshot(source, json.dumps(expected))
    for name in names:
        (tmp_path / name).write_bytes(b"changed while queued")
        with run_lock(source), pytest.raises(ConfigError, match="changed while queued"):
            validate_settle_source_snapshot(source, json.dumps(expected))
        (tmp_path / name).write_bytes(b"original")
    for invalid in (
        {},
        {"../elsewhere": "a" * 64},
        expected | {"rollouts.jsonl": None},
        expected | {"rollouts.jsonl": "bad"},
    ):
        with run_lock(source), pytest.raises(ConfigError):
            validate_settle_source_snapshot(source, json.dumps(invalid))
