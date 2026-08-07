#!/usr/bin/env bash
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
# Builds the Apptainer sandbox GDPval runs deliverables in — once, before
# gdpval.sh. Needs apptainer on the host; the definition ships with Gym, so run
# this from the Gym repo root (or set GYM_ROOT).
#
#   ./build-gdpval-sif.sh                 # writes ./gdpval.sif
#   ./build-gdpval-sif.sh /abs/out.sif    # writes somewhere else
#
# Then export the path gdpval.sh reads:
#   export GDPVAL_CONTAINER_PATH=/abs/out.sif
#
# Note: building inside Docker needs --privileged --device /dev/fuse, a subuid
# mapping for the build user (echo "root:100000:65536" >> /etc/subuid, same for
# /etc/subgid), and an output path on a container-local dir rather than a host
# bind-mount. On a Linux host, VM or cluster node it runs natively.
set -euo pipefail

GYM_ROOT="${GYM_ROOT:-$PWD}"
OUT="${1:-$PWD/gdpval.sif}"
DEF="${GDPVAL_DEF:-$GYM_ROOT/responses_api_agents/stirrup_agent/containers/gdpval.def}"

command -v apptainer >/dev/null || { echo "apptainer not installed (https://apptainer.org/docs/admin/main/installation.html)" >&2; exit 1; }
[ -r "$DEF" ] || { echo "gdpval.def not readable at $DEF — set GYM_ROOT to your Gym checkout (or GDPVAL_DEF to the .def)" >&2; exit 1; }

echo "Building $OUT from $DEF  (multi-GB; several minutes)…"
apptainer build --fakeroot "$OUT" "$DEF"
echo
echo "Built: $OUT"
echo "Next:  export GDPVAL_CONTAINER_PATH=$OUT"
