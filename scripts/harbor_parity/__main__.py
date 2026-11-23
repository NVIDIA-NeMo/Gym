# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`python -m scripts.harbor_parity`: compare a native-path run with a Harbor-path run."""

import sys

from scripts.harbor_parity.parity import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
