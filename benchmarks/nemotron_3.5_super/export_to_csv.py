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
import json
from argparse import ArgumentParser
from csv import DictWriter
from datetime import datetime
from subprocess import check_output
from typing import Union
from zoneinfo import ZoneInfo


parser = ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--jsonl-fpath-base", type=str, required=True)
args = parser.parse_args()

aggregate_metrics_fpath = f"{args.jsonl_fpath_base}_aggregate_metrics.json"
csv_fpath = f"{args.jsonl_fpath_base}_export.csv"

with open(aggregate_metrics_fpath) as f:
    aggregate_metrics = json.load(f)

agent_to_metrics = {d["agent_ref"]["name"]: d for d in aggregate_metrics}


def to_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def v(agent: str, value: str, is_pct: bool = True) -> Union[float, str]:
    value = agent_to_metrics[agent]["agent_metrics"][value]
    return to_pct(value) if is_pct else value


row = {
    "Model nickname": "",
    "Model path": args.model_path,
    "Note": "",
    "Date run": datetime.now(ZoneInfo("America/Los_Angeles")),
    "Gym commit": check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "Results dir": args.jsonl_fpath_base,
    "Tau3-Banking": v("tau2_banking_knowledge_bm25_grep_artificial_analysis_agent", "mean/reward"),
    "Tau3-Average": to_pct(
        sum(
            (
                v("tau2_benchmark_agent", "airline/reward", is_pct=False),
                v("tau2_benchmark_agent", "retail/reward", is_pct=False),
                v("tau2_benchmark_agent", "telecom/reward", is_pct=False),
                v("tau2_banking_knowledge_bm25_grep_artificial_analysis_agent", "mean/reward", is_pct=False),
            )
        )
        / 4
    ),
    "SciCode": v("scicode_benchmark_agent", "mean/reward"),
    "AA-LCR": v("aalcr_benchmark_simple_agent", "mean/reward"),
    "AA-Omniscience (OmniIndex)": v("omniscience_omniscience_simple_agent", "mean/reward"),
    "GPQA Diamond": v("gpqa_mcqa_simple_agent", "mean/reward"),
}

with open(csv_fpath, "w") as f:
    writer = DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)

with open(csv_fpath) as f:
    content = f.read()
print(f"Wrote out csv to {csv_fpath}:\n{content}")
