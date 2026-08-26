#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the repo's required test coverage threshold (whole percent).

pyproject.toml's [tool.coverage.report] fail_under is the single source of truth for this
number. CI workflows call this instead of hardcoding the threshold, so there is exactly one
place to change it. See unit-tests.yml and full-test-suite.yml.
"""

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    fail_under = pyproject["tool"]["coverage"]["report"]["fail_under"]
    print(int(fail_under))


if __name__ == "__main__":
    main()
