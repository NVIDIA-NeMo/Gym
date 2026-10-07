# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Sourced by run.sh before `gym eval prepare`. Gym runs the xCOMET-XXL scorer as Ray workers
# and expects a separate scoring machine; these lines let it score on this machine's GPUs.

# GPUs for xCOMET-XXL: those in CUDA_VISIBLE_DEVICES if set, else every GPU with 60 GB free.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits \
    | awk -F', ' '$2 >= 60000 {print $1}' | paste -sd, -)
  [ -n "$CUDA_VISIBLE_DEVICES" ] || {
    echo "wmt24pp: no GPU with 60 GB free for xCOMET-XXL; set CUDA_VISIBLE_DEVICES" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES
fi
echo "wmt24pp: xCOMET-XXL scores on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
N=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)

# Gym's scorers only start on GPUs Ray offers as `extra_gpu`: offer one per chosen GPU.
export RAY_OVERRIDE_RESOURCES="{\"extra_gpu\": $N}"
EXTRA_ARGS+=(++wmt24pp_wmt_translation_resources_server.resources_servers.wmt_translation.comet_num_shards=$N)

# Gym copies Python for the scorers into /opt/Gym by default; use a writable folder instead.
export WMT_TRANSLATION_COMET_PY_CACHE="$PWD/cache/comet-python"

# Gym looks for the scorer's packages under python3.12, but its venvs are Python 3.13 at this
# commit: point the old folder name at the real one.
SCORER_VENV_LIB=resources_servers/wmt_translation/.venv/lib
mkdir -p "$SCORER_VENV_LIB" && ln -sfn python3.13 "$SCORER_VENV_LIB/python3.12"

# Download the scorer model once (43 GB) into the cache Gym uses; each scorer gets only
# 5 minutes to start, too short to download it then.
HF_HOME="${HF_HOME:-$PWD/cache/huggingface}" hf download Unbabel/XCOMET-XXL > /dev/null
