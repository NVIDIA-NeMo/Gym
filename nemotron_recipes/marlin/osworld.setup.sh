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
# Sourced by run.sh before `gym eval prepare`. Downloads the OSWorld VM disk once with Gym's own
# script (later runs only re-check its checksum) and points the agent at it.
# OSWORLD_VM_DIR: where to keep it (default: docker_vm_data in the Gym repo root).
OSWORLD_VM_DIR="${OSWORLD_VM_DIR:-$PWD/docker_vm_data}"
VM_DIR="$OSWORLD_VM_DIR" bash benchmarks/osworld/tools/prepare_osworld_vm.sh
export OSWORLD_VM_PATH="$OSWORLD_VM_DIR/Ubuntu.qcow2"
# Absolute paths for osworld.yaml: the agent server runs in its own directory.
export OSWORLD_TASKS="$PWD/benchmarks/osworld/data/test_nogdrive.jsonl"
export OSWORLD_SETUP_CACHE="$PWD/benchmarks/osworld/.cache/setup"
