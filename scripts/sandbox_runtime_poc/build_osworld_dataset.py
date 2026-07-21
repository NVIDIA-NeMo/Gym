#!/usr/bin/env -S uv run --no-config --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Build the full OSWorld nemo_gym dataset jsonl from the upstream task set.

For each task in ``evaluation_examples/test_all.json`` (369 tasks / 10 domains), read the
per-task config ``evaluation_examples/examples/<domain>/<id>.json`` and emit one nemo_gym
row: ``responses_create_params.input`` = the instruction (user message) and
``verifier_metadata`` = the *entire* upstream task config (id/instruction/config/evaluator/
related_apps/…). The resources server uses ``verifier_metadata["config"]`` for guest setup
and passes the whole config to the complete upstream OSWorld evaluator at verify time.

Usage:
  ./build_osworld_dataset.py <osworld_repo_dir> <out.jsonl> [--meta test_all.json]
"""
import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("osworld_dir")
    ap.add_argument("out")
    ap.add_argument("--meta", default="evaluation_examples/test_all.json")
    args = ap.parse_args()

    meta_path = os.path.join(args.osworld_dir, args.meta)
    with open(meta_path) as f:
        meta = json.load(f)

    rows = 0
    missing = 0
    with open(args.out, "w") as out:
        for domain, ids in meta.items():
            for tid in ids:
                cfg_path = os.path.join(args.osworld_dir, "evaluation_examples", "examples", domain, f"{tid}.json")
                if not os.path.exists(cfg_path):
                    missing += 1
                    print(f"WARN missing config: {cfg_path}", file=sys.stderr)
                    continue
                with open(cfg_path) as f:
                    cfg = json.load(f)
                instruction = cfg.get("instruction", "")
                row = {
                    "responses_create_params": {"input": [{"role": "user", "content": instruction}]},
                    "verifier_metadata": cfg,
                }
                out.write(json.dumps(row) + "\n")
                rows += 1
    print(f"wrote {rows} rows to {args.out} (missing {missing})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
