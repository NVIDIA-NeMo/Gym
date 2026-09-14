#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
uv run --with grpcio-tools==1.75.1 python -m grpc_tools.protoc \
  --proto_path="$root/nemo_gym/sandbox/providers/mfn/protos" \
  --python_out="$root/nemo_gym/sandbox/providers/mfn/protos" \
  --pyi_out="$root/nemo_gym/sandbox/providers/mfn/protos" \
  "$root/nemo_gym/sandbox/providers/mfn/protos/mfn_sandbox.proto"

# protoc output carries no license header; re-add this script's header (lines 2-15) so regenerating keeps
# the files compliant.
header="$(sed -n '2,15p' "$0")"
for generated in "$root"/nemo_gym/sandbox/providers/mfn/protos/mfn_sandbox_pb2.py{,i}; do
  { printf '%s\n\n' "$header"; cat "$generated"; } >"$generated.tmp"
  mv "$generated.tmp" "$generated"
done
