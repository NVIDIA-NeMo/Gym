# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from resources_servers.mimo_rl_oss.webdev.environment import DELIVERY_INSTRUCTIONS


REPO = "XiaomiMiMo/MiMo-V2.6-RL-oss"
FILES = {
    "code": "code.parquet",
    "cyber": "cyber.parquet",
    "music": "music.parquet",
    "terminal_bench": "general/train.parquet",
    "general_agent": "general/train.parquet",
    "webdev": "webdev.parquet",
}
OUT_DIR = Path(__file__).parent / "data"
MUSIC_DIR = Path(__file__).parent.parent / "mimo_music" / "data"


def _download(name: str) -> str:
    return hf_hub_download(REPO, name, repo_type="dataset")


def convert_music() -> list[dict]:
    rows = []
    for _, row in pd.read_parquet(_download(FILES["music"])).iterrows():
        messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
        meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(row["extra_info"]).items()}
        rows.append({"responses_create_params": {"input": messages}, "verifier_metadata": meta})
    return rows


def convert(subset: str, image_map: dict[str, str]) -> list[dict]:
    if subset == "music":
        return convert_music()
    rows = []
    for _, row in pd.read_parquet(_download(FILES[subset])).iterrows():
        extra = dict(row["extra_info"])
        instance = json.loads(extra["instance_json"])
        if subset in ("terminal_bench", "general_agent") and instance["dataset_type"] != subset:
            continue
        instance["docker_image"] = image_map[instance["docker_image"]]
        task = instance["problem_statement"]
        if subset == "webdev":
            task = DELIVERY_INSTRUCTIONS.format(cwd=instance["cwd"].rstrip("/"), task=task)
        messages = [{"role": "user", "content": task}]
        rows.append(
            {
                "responses_create_params": {"input": messages, "metadata": {"workdir": instance["cwd"]}},
                "instance": instance,
                "subset": subset,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=list(FILES))
    args = parser.parse_args()
    image_map = {}
    for line in Path(_download("image-mapping.jsonl")).read_text().splitlines():
        entry = json.loads(line)
        image_map[entry["dataset_image"]] = entry["dockerhub_image"]
    for subset in args.subsets:
        rows = convert(subset, image_map)
        out = MUSIC_DIR / "train.jsonl" if subset == "music" else OUT_DIR / f"{subset}.jsonl"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"{subset}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
