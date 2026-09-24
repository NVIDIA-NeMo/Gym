#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared settings for the fast profiles: capped generation on the fast vLLM profile.
VLLM_CONFIG="$(dirname -- "${BASH_SOURCE[0]}")/vllm_profile_fast.sh"
BENCHMARK_EXTRA_ARGS=(
  "++policy_model.responses_api_models.vllm_model.sampling_overrides.max_tokens=49152"
  "++policy_model.responses_api_models.vllm_model.sampling_overrides.thinking_token_budget=32768"
  # a failed agent /run (sandbox infra error -> HTTP 500) becomes a sidecar failure row instead of aborting the run
  "++route_failures_to_sidecar=true"
)
