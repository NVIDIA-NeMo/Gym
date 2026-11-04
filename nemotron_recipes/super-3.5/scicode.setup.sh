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
# Sourced by run.sh before `gym eval prepare`. SciCode's test_data.h5 must be readable by the
# user running the recipe; otherwise every test case fails and the run scores 0 with no error.
TEST_DATA="${TEST_DATA:?export TEST_DATA (path to SciCode test_data.h5)}"
[ -r "$TEST_DATA" ] || { echo "test_data.h5 not readable at $TEST_DATA" >&2; exit 1; }
export TEST_DATA="$(cd "$(dirname "$TEST_DATA")" && pwd)/$(basename "$TEST_DATA")"
