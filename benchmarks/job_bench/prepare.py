# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the JobBench main-split benchmark dataset."""

from pathlib import Path


def prepare() -> Path:
    """Materialize the private rubric cache and return the model-visible JSONL."""

    # Imported lazily so the Hugging Face dependency is only needed when
    # preparation actually runs, keeping cached evals runnable from Gym's
    # base launcher environment.
    from resources_servers.job_bench.prepare import prepare as prepare_job_bench

    _, jsonl_path = prepare_job_bench(split="main")
    return jsonl_path


if __name__ == "__main__":
    print(prepare())
