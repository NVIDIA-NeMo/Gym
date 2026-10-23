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
"""Prepare data and the gRNAde environment.

python prepare.py rnainvbench_eterna100_v1 --output data/eterna100_v1.jsonl --limit 20   # builds suites/ on first use
python prepare.py --setup ~/rna_env      # gRNAde checkout, model weights, and a Python env for it
"""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml


SUITES_DIR = Path(__file__).resolve().parent / "suites"

# RNAInvBench (MIT) at a pinned commit. Its benchmark sets are pickled DataFrames, so each file is checked against
# its hash before it is unpickled.
RNAINVBENCH_COMMIT = "4aa5115bac70a086c76e646f2d758ecf0724536c"  # pragma: allowlist secret
RNAINVBENCH_URL = f"https://raw.githubusercontent.com/COLA-Laboratory/RNAInvBench/{RNAINVBENCH_COMMIT}/data/"
RNAINVBENCH_FILES = {
    "inverse_rna_folding_benchmark_dotbracket.pkl.gz": "f599461344b115753fd2683132fb3671c90fe3b456887702c38065521bde5685",  # pragma: allowlist secret
    "constrained_design_benchmark_dotbracket.pkl.gz": "c563a9af081308f40c152f49d7ff4762a9027997d1db007cd01cadcc6cdd7272",  # pragma: allowlist secret
}


def ensure_suites() -> None:
    """Write the eight RNAInvBench suites to suites/, one per origin, unless they are there. Idempotent."""
    if len(list(SUITES_DIR.glob("rnainvbench_*.yaml"))) == 8:
        return
    from rds.structure import to_dot_bracket

    SUITES_DIR.mkdir(exist_ok=True)
    for name, digest in RNAINVBENCH_FILES.items():
        print(f"Building suites from RNAInvBench {name}", file=sys.stderr)
        raw = urllib.request.urlopen(RNAINVBENCH_URL + name, timeout=120).read()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError(f"{name} does not match its pinned SHA-256, so it is not unpickled")
        constrained = name.startswith("constrained")
        for origin, group in pickle.loads(gzip.decompress(raw)).groupby("origin"):
            targets = []
            for _, row in group.iterrows():
                sequence = "".join(row["sequence"])
                # From the pair list, not the shipped dot-bracket column, which drops every crossing pair.
                structure = to_dot_bracket([(int(a), int(b)) for a, b, *_ in row["pairs"]], len(sequence))
                target = {
                    "name": f"{origin}_{row['Id']}",
                    "length": len(structure),
                    "target_structure": structure,
                    "origin": str(origin),
                }
                # In the constrained sets an IUPAC letter other than N pins that position.
                pinned = {i + 1: base for i, base in enumerate(sequence) if base.upper() in "ACGU"}
                if constrained and pinned:
                    target["conserved"] = pinned
                targets.append(target)
            knotted = sum("[" in t["target_structure"] for t in targets) > len(targets) // 2
            category = "constrained_design" if constrained else "pseudoknot_design" if knotted else "inverse_folding"
            provides = ["secondary_structure", "sequence_constraints"] if constrained else ["secondary_structure"]
            suite = {
                "name": f"rnainvbench_{origin.replace('-', '_')}",
                "category": category,
                "source": f"RNAInvBench / {origin}",
                "provides": provides,
                "scoring": {
                    # Solved means the exact fold, as these sets are published. A pseudoknot set ranks on its
                    # crossing pairs instead, and 0.8 is this benchmark's cutoff since none is published.
                    "primary": "struct.rnet_ss.match.crossed_pair_f1" if knotted else "score.fold_match.match.f1",
                    "direction": "max",
                    "gates": {"score.constraints.passed": True},
                    "top_k": 20,
                    "pass_threshold": 0.8 if knotted else 1.0,
                },
                "budget": {"candidates": 600, "rounds": 12, "tool_calls": 40, "gpu_seconds": 3600},
                "generator": "generate.grnade",
                "exclude": ["generate.inverse_fold", "generate.mutate", "fold.vienna"],
                "targets": targets,
            }
            path = SUITES_DIR / f"{suite['name']}.yaml"
            # Both servers may build at once, so each file appears whole or not at all.
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(yaml.safe_dump(suite, sort_keys=False, width=100))
            os.replace(tmp, path)
            print(f"  {path.name}: {len(targets)} targets", file=sys.stderr)


def task_description(target: dict) -> str:
    """Placeholder input. The campaign builds its own prompt, so the model does not read this."""
    lines = [f"Design an RNA sequence of length {target['length']} for this target structure:"]
    lines.append(target["target_structure"])
    if target.get("fixed_prefix"):
        lines.append(f"Fixed prefix: {target['fixed_prefix']}")
    return "\n".join(lines)


def build_rows(suite_name: str, limit: int = 0, seed: int = 0) -> list[dict]:
    ensure_suites()
    path = SUITES_DIR / f"{suite_name}.yaml"
    if not path.is_file():
        known = sorted(p.stem for p in SUITES_DIR.glob("*.yaml"))
        raise SystemExit(f"unknown suite {suite_name!r}; choose from {known}")
    suite = yaml.safe_load(path.read_text())
    targets = list(suite["targets"])
    if limit:
        random.Random(seed).shuffle(targets)
        targets = targets[:limit]
    return [
        {
            "responses_create_params": {"input": [{"role": "user", "content": task_description(t)}]},
            "suite": suite["name"],
            "target": t,
        }
        for t in targets
    ]


# What gRNAde imports. torch-scatter and torch-cluster are left out: the agent ships pure PyTorch stand-ins.
GRNADE_PACKAGES = (
    "numpy scipy pandas pyyaml tqdm einops ml_collections python-dotenv torchmetrics biopython biotite mdanalysis "
    "torch-geometric arnie wandb gdown draw_rna cpdb-protein huggingface_hub"
).split()


def setup(home: Path) -> None:
    """Check out gRNAde, download the checkpoints it loads (public, no login) and build a Python env for it.

    Afterwards set `GRNADE_REPO=<home>/geometric-rna-design` and `RNA_WORKER_PYTHON=<home>/venv/bin/python`.
    """
    from huggingface_hub import snapshot_download

    home.mkdir(parents=True, exist_ok=True)
    repo = home / "geometric-rna-design"
    if not repo.exists():
        url = "https://github.com/chaitjo/geometric-rna-design.git"
        subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
    # design.py loads all three models from checkpoints/ inside the checkout.
    patterns = ["gRNAde_drop3d@0.75_maxlen@500.h5", "ribonanzanet/*", "ribonanzanet_sec_struct/*", "config.yaml"]
    snapshot_download("chaitjo/gRNAde", local_dir=repo / "checkpoints", allow_patterns=patterns)
    shutil.rmtree(repo / "checkpoints" / ".cache", ignore_errors=True)

    venv = home / "venv"
    if not venv.exists():
        # The venv is built from the interpreter running this script.
        if sys.version_info < (3, 10):
            raise SystemExit(f"run prepare.py with Python 3.10 or newer, not {sys.version.split()[0]}")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = [str(venv / "bin" / "python"), "-m", "pip", "install", "--quiet"]
    # The CUDA 12.8 wheel on Linux, the default wheel elsewhere.
    index = ["--index-url", "https://download.pytorch.org/whl/cu128"] if sys.platform == "linux" else []
    subprocess.run([*pip, "torch", *index], check=True)
    subprocess.run([*pip, *GRNADE_PACKAGES], check=True)
    print(f"ready: GRNADE_REPO={repo}  python={venv / 'bin' / 'python'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", nargs="?", help="a file stem under suites/")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--setup", type=Path, metavar="DIR", help="set up gRNAde and its weights in DIR")
    parser.add_argument("--limit", type=int, default=0, help="a seeded random subset of this many targets")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.setup:
        return setup(args.setup)
    if not (args.suite and args.output):
        parser.error("give a suite and --output, or --setup DIR")

    rows = build_rows(args.suite, args.limit, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
