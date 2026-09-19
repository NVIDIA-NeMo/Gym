#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# An optional output path also permits an offline oracle check without Docker.
answer_file="${1:-/app/answer.txt}"
printf '42\n' > "$answer_file"
