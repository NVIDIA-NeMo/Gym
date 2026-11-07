#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -o pipefail
mkdir -p /logs/verifier
python /tests/verify.py > /logs/verifier/test-stdout.txt 2>&1
