# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry point: ``python -m benchmarks.<benchmark>.reporting.generate_model_card_report --package <dir> ...``."""

from .model_card import main


if __name__ == "__main__":  # pragma: no cover
    main()
