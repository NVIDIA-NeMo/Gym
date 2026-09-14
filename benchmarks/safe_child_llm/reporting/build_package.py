# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a validated Safe-Child-LLM run package: ``python -m benchmarks.safe_child_llm.reporting.build_package --help``."""

from .cli import build_main
from .normalize import normalize


def main() -> None:
    build_main(normalize, benchmark_id="safe_child_llm", description=__doc__)


if __name__ == "__main__":
    main()
