# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporarily convert ``environment.yaml`` into JSONL for the existing rollout pipeline.

The planned ``gym eval run --environment`` path will perform this conversion
internally, so environment authors will not need a preparation script.
"""

from pathlib import Path

from nemo_gym.environment.authoring import load_environment, materialize_tasks_jsonl


ENVIRONMENT_ROOT = Path(__file__).parent
OUTPUT_PATH = ENVIRONMENT_ROOT / "data" / "materialized.jsonl"


def prepare() -> Path:
    environment = load_environment(ENVIRONMENT_ROOT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(materialize_tasks_jsonl(environment), encoding="utf-8")
    return OUTPUT_PATH


if __name__ == "__main__":
    print(prepare())
