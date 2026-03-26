# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Usage:
```bash
python resources_servers/tavily_search/plot_session_id_metrics.py \
    --fpath resources_servers/tavily_search/session_id_metrics.json
```
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from pandas import DataFrame


parser = ArgumentParser()
parser.add_argument("--fpath", type=str, required=True)
args = parser.parse_args()

with open(args.fpath) as f:
    data = json.load(f)

rows = []
for session_id_metrics in data.values():
    rows.extend(session_id_metrics["async_tavily_calls"])

df = DataFrame.from_records(rows)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.hist(df["time_taken"])
ax.set_xlabel("Time taken per call (s)")
ax.set_ylabel("Count")
ax.set_title(f"Call time distribution (total {len(df)} Tavily API calls)")

fig.tight_layout()
out_path = Path(__file__).parent / "metrics.png"
fig.savefig(out_path)
