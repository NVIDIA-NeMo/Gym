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
import importlib
import json
import os

from datasets import concatenate_datasets, load_dataset

FORMATS = ["json", "yaml", "xml"]
FILE_PREFIX = "260309_nano_v3_sdg_structured_outputs"


def main():
    local_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(local_dir, "data")

    for fmt in FORMATS:
        module = importlib.import_module(f"260309_nano_v3_sdg_{fmt}")
        print(f"\n{'='*60}\nRunning {fmt} SDG...\n{'='*60}")
        module.main()

    metrics = {}

    for split in ["train", "val"]:
        split_datasets = {}
        for fmt in FORMATS:
            fpath = os.path.join(data_dir, f"{FILE_PREFIX}_{fmt}_{split}.jsonl")
            ds = load_dataset("json", data_files=fpath, split="train")
            split_datasets[fmt] = ds

        combined = concatenate_datasets(list(split_datasets.values()))
        combined = combined.shuffle(seed=42)
        out_path = os.path.join(data_dir, f"{FILE_PREFIX}_all_{split}.jsonl")
        combined.to_json(out_path)

        total = len(combined)
        metrics[split] = {
            "total": total,
            "per_format": {
                fmt: {
                    "count": len(ds),
                    "proportion": round(len(ds) / total, 4) if total else 0,
                }
                for fmt, ds in split_datasets.items()
            },
            "output_path": out_path,
        }

    print(f"\n{'='*60}\nDataset Metrics\n{'='*60}")
    for split, info in metrics.items():
        print(f"\n  {split}:")
        print(f"    total: {info['total']}")
        for fmt, fmt_info in info["per_format"].items():
            print(f"    {fmt}: {fmt_info['count']} ({fmt_info['proportion']:.1%})")
        print(f"    -> {info['output_path']}")

    metrics_path = os.path.join(data_dir, f"{FILE_PREFIX}_all_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
