# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Gym's native collector with a privately loaded Inference Hub key."""

import argparse
import os
import sys
from pathlib import Path

from nemo_gym.cli.env import RunHelper
from nemo_gym.cli.eval import collect_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("benchmarks/indic_alpaca_eval/gemma.yaml"))
    parser.add_argument(
        "--judge-key-file", type=Path, help="Read the judge API key privately; never put it in CLI overrides."
    )
    args, overrides = parser.parse_known_args()
    if args.judge_key_file:
        key = args.judge_key_file.read_text().strip()
        if not key:
            parser.error("Judge key file is empty")
        os.environ["INDIC_ALPACA_JUDGE_API_KEY"] = key
    if not os.environ.get("INDIC_ALPACA_JUDGE_API_KEY"):
        parser.error("Set INDIC_ALPACA_JUDGE_API_KEY or supply --judge-key-file")
    sys.argv = [sys.argv[0], f"+config_paths=[{args.config.resolve()}]", *overrides]
    # RunHelper owns server and optional Ray startup on current Gym.
    runner = RunHelper()
    try:
        runner.start(None)
        collect_rollouts()
    finally:
        if hasattr(runner, "_processes"):
            runner.shutdown()


if __name__ == "__main__":
    main()
