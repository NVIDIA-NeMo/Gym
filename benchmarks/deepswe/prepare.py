# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned DeepSWE v1.1 benchmark dataset."""

from pathlib import Path


def prepare() -> Path:
    """Materialize private verifier assets and return the model-visible JSONL."""

    # Pier is isolated to the DeepSWE resources-server environment. Import it
    # only when preparation is actually needed so cached evals stay runnable
    # from Gym's base launcher environment.
    from resources_servers.deepswe.prepare import prepare as prepare_deepswe

    _, jsonl_path = prepare_deepswe()
    return jsonl_path


def is_prepared_data_current(fpath: str | Path) -> bool:
    """Whether a cached JSONL can be reused: it exists and the pinned ripgrep the agent needs is in place."""

    from resources_servers.deepswe.prepare import ensure_ripgrep

    if not Path(fpath).is_file():
        return False
    try:
        ensure_ripgrep(allow_download=False)
    except FileNotFoundError:
        return False
    return True


if __name__ == "__main__":
    print(prepare())
