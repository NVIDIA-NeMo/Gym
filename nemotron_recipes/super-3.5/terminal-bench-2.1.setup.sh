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
# Sourced by run.sh before `gym eval prepare`. Pins the tasks to the commit of the published
# run; Gym otherwise clones the latest terminal-bench-2-1.

TB21_TASKS_REF=36d417f56c293b8271b306a0e4c566f58e98c153
TB21_DIR=benchmarks/terminal_bench_2_1/terminal-bench-2-1

if [ ! -d "$TB21_DIR" ]; then
  git clone -q https://github.com/harbor-framework/terminal-bench-2-1 "$TB21_DIR"
fi
git -C "$TB21_DIR" checkout -q "$TB21_TASKS_REF" || {
  echo "terminal-bench-2.1: could not check out $TB21_TASKS_REF in $TB21_DIR" >&2; exit 1; }
echo "terminal-bench-2.1: pinned tasks to $TB21_TASKS_REF"
