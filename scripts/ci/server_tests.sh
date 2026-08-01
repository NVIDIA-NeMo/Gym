#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 SHARD_INDEX NUM_SHARDS" >&2
    exit 2
fi

readonly shard_index="$1"
readonly num_shards="$2"
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root

if [[ ! "${shard_index}" =~ ^[0-9]+$ ]]; then
    echo "SHARD_INDEX must be a non-negative integer: ${shard_index}" >&2
    exit 2
fi
if [[ ! "${num_shards}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SHARDS must be a positive integer: ${num_shards}" >&2
    exit 2
fi
if ((shard_index >= num_shards)); then
    echo "SHARD_INDEX (${shard_index}) must be less than NUM_SHARDS (${num_shards})" >&2
    exit 2
fi

cd "${repo_root}"
# shellcheck source=scripts/ci/sanitize_env.sh
source "${ci_dir}/sanitize_env.sh"
gym_ci_sanitize_environment server
unset -f gym_ci_sanitize_environment
# shellcheck source=scripts/ci/setup_dev.sh
source "${ci_dir}/setup_dev.sh"
# Nested pytest processes inherit this through ng_test_all even when Slurm provides no TTY.
export PY_COLORS=1
exec ng_test_all \
    +fail_on_total_and_test_mismatch=true \
    +delete_venvs_after_each_test=true \
    +num_shards="${num_shards}" \
    +shard_index="${shard_index}"
