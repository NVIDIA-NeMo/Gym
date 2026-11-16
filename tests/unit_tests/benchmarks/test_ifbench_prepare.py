# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path

from benchmarks.ifbench import prepare


METRICS_FPATH = Path(prepare.__file__).parent / "data" / "ifbench_benchmark_metrics.json"
PROMPTS = 300  # rows of IFBench_test.jsonl at the pinned commit


def test_the_download_is_pinned_to_a_commit() -> None:
    assert re.search(r"/allenai/IFBench/[0-9a-f]{40}/data/IFBench_test\.jsonl$", prepare.URL)


def test_the_committed_metrics_describe_the_pinned_prompts() -> None:
    metrics = json.loads(METRICS_FPATH.read_text())
    assert metrics["Number of examples"] == PROMPTS * metrics["num_repeats"]
    assert metrics["prompt"]["unique_count"] == PROMPTS
    assert metrics["id"]["Max"] == PROMPTS - 1
