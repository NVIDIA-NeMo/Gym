# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect retained harness capability evidence in rollout artifacts."""

import sys

from nemo_gym.harness_capabilities.cli import main


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args if args and args[0] == "matrix" else ["inspect", *args]))
