#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The reference solution: `gym dataset validate` plays it with the oracle and expects reward 1.
set -eu
# The tool's answer for the name "NeMo Gym" (environment/tools/server.py), written where the verifier reads it.
printf 'Hello, NeMo Gym!\n' > /workspace/tool-greeting.txt
