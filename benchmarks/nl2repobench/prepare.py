# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned NL2RepoBench benchmark dataset."""

from pathlib import Path


def prepare() -> Path:
    """Materialize private verifier assets and return the model-visible JSONL."""

    # Import lazily so cached evals stay runnable from Gym's base launcher environment without
    # pulling nl2repobench's preparation-only dependencies.
    from resources_servers.nl2repobench.prepare import prepare as prepare_nl2repobench

    _, jsonl_path = prepare_nl2repobench()
    return jsonl_path


if __name__ == "__main__":
    print(prepare())
