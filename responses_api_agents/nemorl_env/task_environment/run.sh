#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
stage=${1:?Expected train or eval}
mode=${2:-run}
case "$stage" in
  train) runner=launch_inner.py ;;
  eval) runner=evaluate.py ;;
  *) exit 2 ;;
esac
if [[ "$mode" == execute ]]; then
  export INNER_TRAIN_STARTED_AT=$(date +%s)
fi
export NEMORL_ROOT=/testbed/NeMo-RL
export PATH="/opt/conda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

show_failure_logs() {
  test ! -f /testbed/train.log || tail -n 160 /testbed/train.log
  test ! -f /testbed/aime25.log || tail -n 160 /testbed/aime25.log
}
trap show_failure_logs ERR

if [[ "$mode" != execute ]]; then
  if [[ ! -d "$NEMORL_ROOT/.git" ]]; then
    for attempt in {1..5}; do
      git clone https://github.com/NVIDIA-NeMo/RL.git "$NEMORL_ROOT" && break
      [[ "$attempt" -lt 5 ]] || exit 1
      sleep 5
    done
  fi
  git -C "$NEMORL_ROOT" checkout 1cee83587d0f0d2ba82e7cdeced9772641fddbe3
  for attempt in {1..5}; do
    git -C "$NEMORL_ROOT" submodule update --init --recursive --depth 1 && break
    [[ "$attempt" -lt 5 ]] || exit 1
    sleep 5
  done
fi

if [[ "$stage" == train && "$mode" != setup ]]; then
  cd /testbed
  git apply --check /root/change.diff
  git apply /root/change.diff
fi

if [[ "$mode" != setup ]]; then
  sed -i 's/exclude-dependencies = \["nvidia-cutlass-dsl-libs-base"\]/exclude-dependencies = ["nvidia-cutlass-dsl-libs-base", "deep-ep", "deep-gemm", "mamba-ssm", "causal-conv1d"]/' "$NEMORL_ROOT/pyproject.toml"
fi

if [[ "$mode" != execute ]]; then
  apt-get update -qq
  apt-get install -y -qq build-essential
  /opt/conda/bin/python -m pip install -q --upgrade uv==0.12.9
fi
cd "$NEMORL_ROOT"
sync_args=(--frozen)
if [[ "$stage" == eval ]]; then
  sync_args+=(--extra vllm --extra nemo_gym)
fi
for attempt in {1..5}; do
  /opt/conda/bin/uv sync "${sync_args[@]}" \
    --no-install-package nvidia-cutlass-dsl-libs-base \
    --no-install-package deep-ep \
    --no-install-package deep-gemm \
    --no-install-package mamba-ssm \
    --no-install-package causal-conv1d && break
  [[ "$attempt" -lt 5 ]] || exit 1
  sleep 5
done
if [[ "$mode" != setup ]]; then
  cd /testbed
  "$NEMORL_ROOT/.venv/bin/python" "$runner"
  show_failure_logs
fi
