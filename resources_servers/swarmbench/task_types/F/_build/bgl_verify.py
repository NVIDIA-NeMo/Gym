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
import collections, sys
sys.path.insert(0, sys.argv[2])
from bgl_rules import categorize
bad = collections.Counter(); n = 0; ex = {}
for line in open(sys.argv[1], errors="replace"):
    line = line.rstrip("\n")
    lab, rest = line.split(" ", 1)
    got = categorize(rest) or "-"
    n += 1
    if got != lab:
        bad[(lab, got)] += 1; ex.setdefault((lab, got), rest[:200])
print("lines", n, "mismatches", sum(bad.values()))
for k, v in bad.most_common(30): print(v, k, ex[k])
