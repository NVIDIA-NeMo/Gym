# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the prose model-card report for a run package with any OpenAI-compatible endpoint.

Example::

    python -m benchmarks.<benchmark>.reporting.generate_model_card_report \\
      --package results/<run-package> --base-url "$REPORT_LLM_BASE_URL" \\
      --model "$REPORT_LLM_MODEL" --api-key-env REPORT_LLM_API_KEY
"""

from .model_card import main


if __name__ == "__main__":
    main()
