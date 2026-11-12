#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../.."
# uv.lock remains authoritative. Refuse stale project metadata instead of changing
# the baseline's dependency versions as a side effect of this experiment.
uv export --locked --extra dev --no-emit-project --no-hashes --no-annotate --prune mypy --prune pre-commit --prune ruff \
  --output-file scripts/pants/pilot.requirements.txt >/dev/null
"${PANTS_BIN:-pants}" generate-lockfiles --resolve=pilot
