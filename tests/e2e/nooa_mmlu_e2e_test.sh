#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NOOA_E2E_CONFIG="$ROOT_DIR/benchmarks/mmlu/config_nooa.yaml"
export NOOA_E2E_AGENT=mmlu_nooa_agent
export NOOA_E2E_INPUT="$ROOT_DIR/tests/e2e/nooa_mmlu_smoke.jsonl"
export NOOA_E2E_LIMIT=1
export NOOA_E2E_VERIFY="$ROOT_DIR/tests/e2e/verify_nooa_mmlu_rollout.py"
export NOOA_E2E_MODEL_TYPE=openai_model
export NOOA_E2E_PROMPT_CONFIG="$ROOT_DIR/benchmarks/prompts/eval/aai/mcq-4choices-boxed.yaml"

bash "$ROOT_DIR/tests/e2e/nooa_calculate_e2e_test.sh"
