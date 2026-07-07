# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enable ``python -m tool_sandbox.cli``."""
from __future__ import annotations

import sys

from tool_sandbox.cli import main

if __name__ == "__main__":
    sys.exit(main())
