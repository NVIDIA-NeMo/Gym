# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run scripted requests through real Gym harnesses and inspect the resulting rollouts."""

import sys
from pathlib import Path


# Run as a script, only scripts/ is on sys.path; `scripts.check_suites` needs the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_conformance.runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
