# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned DeepSWE v1.1 benchmark dataset."""

from pathlib import Path

from resources_servers.deepswe.prepare import prepare as prepare_deepswe


def prepare() -> Path:
    """Materialize private verifier assets and return the model-visible JSONL."""

    _, jsonl_path = prepare_deepswe()
    return jsonl_path


if __name__ == "__main__":
    print(prepare())
