# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


@pytest.mark.parametrize("metadata_source", ["row", "run"])
def test_repeat_seeds_do_not_mutate_shared_metadata(tmp_path: Path, metadata_source: str) -> None:
    metadata = {"extra_body": '{"top_k": 64}', "chat_template_kwargs": '{"enable_thinking": true}'}
    request = {"input": []}
    overrides = {}
    if metadata_source == "row":
        request["metadata"] = metadata
    else:
        overrides["metadata"] = metadata
    examples = [{"responses_create_params": request, "agent_ref": {"name": "my_agent"}}]
    config = RolloutCollectionConfig(
        input_jsonl_fpath="unused",
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        num_repeats=4,
        num_repeats_add_seed=True,
        responses_create_params=overrides,
    )
    rows = RolloutCollectionHelper._preprocess_raw_rows([(0, json.dumps(examples[0]), examples[0].copy())], config)
    assert [json.loads(row["responses_create_params"]["metadata"]["extra_body"]) for row in rows] == [
        {"top_k": 64, "seed": seed} for seed in range(4)
    ]
    assert all(
        row["responses_create_params"]["metadata"]["chat_template_kwargs"] == metadata["chat_template_kwargs"]
        for row in rows
    )
    assert metadata["extra_body"] == '{"top_k": 64}'
    if metadata_source == "run":
        assert config.responses_create_params["metadata"]["extra_body"] == '{"top_k": 64}'
