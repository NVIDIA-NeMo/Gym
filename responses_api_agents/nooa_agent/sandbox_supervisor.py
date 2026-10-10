# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the NOOA worker under Gym's unchanged Linux descendant supervisor."""

import json
import math
import os
import sys
from pathlib import Path

from nemo_gym.agent_utils import process_supervisor


def supervise(directory: Path) -> int:
    """Keep one episode clock and publish the shared supervisor's cleanup receipt.

    This is the sole dependency on Gym's private ``_supervise`` helper. Its CLI
    requires a finite deadline; infinity here leaves cancellation to the NOOA
    environment without changing the shared supervisor or copying its reaper.
    The worker remains alive while successful task services await verification.
    """
    (directory / "runner.pid").write_text(str(os.getpid()))
    receipt = process_supervisor._supervise(
        [
            sys.executable,
            "-I",
            "-m",
            "responses_api_agents.nooa_agent.sandbox_entrypoint",
            str(directory / "input.json"),
            str(directory / "result.json"),
            str(directory / "runner.stop"),
            str(directory / "completion.json"),
        ],
        timeout=math.inf,
        cleanup_timeout=5,
        stop_path=directory / "runner.stop",
    )
    temporary = directory / "cleanup.tmp"
    temporary.write_text(json.dumps(receipt))
    temporary.replace(directory / "cleanup.json")
    return 0 if receipt["cleanup_confirmed"] and receipt["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(supervise(Path(sys.argv[1])))
