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
from csv import DictWriter
from glob import glob
from pathlib import Path
from shutil import copytree

from swebench.harness.constants import MAP_REPO_TO_EXT


benchmark_fpath = Path("benchmarks/swebench/data/swebench_multilingual_benchmark.jsonl")
failed_sample_dirpath = Path("results/failed_swe_multilingual_golden_patch")
failed_sample_dirpath.mkdir(parents=True, exist_ok=True)

instance_id_to_row = dict()
with benchmark_fpath.open() as f:
    for line in f:
        row = json.loads(line)
        instance_id_to_row[row["instance_id"]] = row

writer = DictWriter(open("temp.csv", "w"), fieldnames=["Instance ID", "Language", "Resolved"])
writer.writeheader()


def copy_sample(path: Path, instance_id: str):
    relative_report_path = path.relative_to("resources_servers/swebench/logs/run_evaluation")
    session_id = relative_report_path.parts[0]
    copytree(
        src=Path("resources_servers/swebench/logs/run_evaluation") / session_id,
        dst=failed_sample_dirpath / session_id,
        dirs_exist_ok=True,
    )
    row = instance_id_to_row[instance_id]
    sample_path = (failed_sample_dirpath / relative_report_path).parent / "sample.json"
    sample_path.write_text(json.dumps(row, indent=4))


seen_instance_ids = []
failed_instance_ids = []
for path in glob("resources_servers/swebench/logs/run_evaluation/**/report.json", recursive=True):
    path = Path(path)
    report = json.loads(path.read_text())
    instance_id: str = list(report.keys())[0]
    repo_name = instance_id.rsplit("-", maxsplit=1)[0].replace("__", "/")
    repo_ext = MAP_REPO_TO_EXT[repo_name]

    writer.writerow(
        {
            "Instance ID": instance_id,
            "Language": repo_ext,
            "Resolved": report[instance_id]["resolved"],
        }
    )

    seen_instance_ids.append(instance_id)
    if not report[instance_id]["resolved"]:
        failed_instance_ids.append(instance_id)
        copy_sample(path, instance_id)

seen_instance_ids = set(seen_instance_ids)

for instance_id in instance_id_to_row:
    if instance_id in seen_instance_ids:
        continue

    dirpaths = glob(f"resources_servers/swebench/logs/run_evaluation/**/{instance_id}", recursive=True)
    assert len(dirpaths) == 1
    report_path = Path(dirpaths[0]) / "report.json"
    copy_sample(report_path, instance_id)

    failed_instance_ids.append(instance_id)

failed_instance_ids = set(failed_instance_ids)

temp_jsonl_fpath = Path("temp.jsonl")
with benchmark_fpath.open() as f, temp_jsonl_fpath.open("w") as f_out:
    for line in f:
        row = json.loads(line)
        if row["instance_id"] in failed_instance_ids:
            f_out.write(line)
