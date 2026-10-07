#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Match SGLang 0.5.20's optional fastokens dependency.
python3 -m pip install 'fastokens>=0.1.1,<0.2.0'
python3 -c 'import fastokens'
