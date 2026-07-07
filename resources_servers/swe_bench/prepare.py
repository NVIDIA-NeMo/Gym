# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
    python prepare.py                          # full SWE-bench Verified + all SIFs
    python prepare.py --limit 5                # 5 instances + their 5 SIFs (smoke test)
    python prepare.py --instance-id django__django-13741
    python prepare.py --no-images              # dataset only, skip image builds
    python prepare.py --no-dataset --sif-dir PATH # build images only

schema anyswe_agent expects: each line has
`responses_create_params.metadata` with `instance_id`, `dataset_name`, `split`,
`problem_statement`, and `instance_dict` (the full SWE-bench instance the eval
harness needs). Images are Apptainer SIFs named `{instance_id}.sif` so the
agent's container_formatter is simply `<sif-dir>/{instance_id}.sif`.

Prerequisites for image builds: `apptainer` on PATH and network access to the
SWE-bench image registry. Each SIF is multiple GB, building all of SWE-bench
Verified (500 tasks) needs hundreds of GB of disk. Can use --limit and iterate.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HF_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
# SWE-bench publishes eval images with `__` -> `_1776_` and lowercased.
DOCKER_IMAGE_TMPL = "docker://swebench/sweb.eval.x86_64.{tag}:latest"
DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

_THIS_DIR = Path(__file__).parent


def _docker_tag(instance_id: str) -> str:
    return instance_id.replace("__", "_1776_").lower()


def _to_gym_row(inst: dict, split: str, sampling: dict) -> dict:
    swe_meta = {
        "instance_id": inst["instance_id"],
        "dataset_name": HF_DATASET,
        "split": split,
        "problem_statement": inst["problem_statement"],
        "instance_dict": json.dumps(inst),
    }
    user_text = inst["problem_statement"]
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": user_text}],
            **sampling,
            "metadata": swe_meta,
        },
        "verifier_metadata": swe_meta,
    }


def build_dataset(output: Path, split: str, limit: int | None, instance_id: str | None, sampling: dict) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` is required for dataset prep: pip install datasets")

    print(f"Loading {HF_DATASET} [{split}]...", flush=True)
    rows = load_dataset(HF_DATASET, split=split)

    if instance_id:
        rows = [r for r in rows if r["instance_id"] == instance_id]
        if not rows:
            sys.exit(f"instance_id {instance_id!r} not found in {HF_DATASET}")
    elif limit:
        rows = rows.select(range(min(limit, len(rows))))

    output.parent.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    with output.open("w") as f:
        for inst in rows:
            inst = dict(inst)
            f.write(json.dumps(_to_gym_row(inst, split, sampling)) + "\n")
            ids.append(inst["instance_id"])
    print(f"Wrote {len(ids)} rows -> {output}", flush=True)
    return ids


def _build_one_sif(instance_id: str, sif_dir: Path, force: bool) -> tuple[str, bool, str]:
    sif_path = sif_dir / f"{instance_id}.sif"
    if sif_path.exists() and not force:
        return instance_id, True, "exists"
    image = DOCKER_IMAGE_TMPL.format(tag=_docker_tag(instance_id))
    proc = subprocess.run(
        ["apptainer", "build", "--force", str(sif_path), image],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0:
        return instance_id, False, proc.stderr.strip()[-500:]
    return instance_id, True, "built"


def build_images(instance_ids: list[str], sif_dir: Path, jobs: int, force: bool) -> None:
    if not _which("apptainer"):
        sys.exit("`apptainer` not found on PATH. Install it or pass --no-images")
    sif_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building {len(instance_ids)} SIF(s) into {sif_dir} with {jobs} worker(s)...", flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_build_one_sif, iid, sif_dir, force): iid for iid in instance_ids}
        for done in as_completed(futures):
            iid, ok, detail = done.result()
            print(f"  [{'ok' if ok else 'FAIL'}] {iid}: {detail}", flush=True)
            if not ok:
                failures.append(iid)
    if failures:
        print(f"\n{len(failures)} image build(s) failed:", flush=True)
        for iid in failures:
            print(f"  - {iid}", flush=True)
        sys.exit(1)
    print(f"All images ready. Use: container_formatter='{sif_dir}/{{instance_id}}.sif'", flush=True)


def _which(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=_THIS_DIR / "data" / "swebench_verified.jsonl")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=None, help="Only the first N instances (default: all)")
    p.add_argument("--instance-id", default=None, help="Only this instance")
    p.add_argument("--sif-dir", type=Path, default=_THIS_DIR / "data" / "sifs")
    p.add_argument("--no-dataset", action="store_true", help="Skip dataset build")
    p.add_argument("--no-images", action="store_true", help="Skip image build")
    p.add_argument("--jobs", type=int, default=4, help="Parallel image builds")
    p.add_argument("--force", action="store_true", help="Rebuild SIFs that already exist")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Default model baked into each row")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--max-output-tokens", type=int, default=12288)
    args = p.parse_args()

    sampling = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
    }

    instance_ids: list[str]
    if args.no_dataset:
        if not args.output.exists():
            sys.exit(f"--no-dataset but {args.output} does not exist")
        instance_ids = [
            json.loads(line)["responses_create_params"]["metadata"]["instance_id"]
            for line in args.output.read_text().splitlines()
            if line.strip()
        ]
    else:
        instance_ids = build_dataset(args.output, args.split, args.limit, args.instance_id, sampling)

    if not args.no_images:
        build_images(instance_ids, args.sif_dir, args.jobs, args.force)


if __name__ == "__main__":
    main()
