# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare MolDeTox for NeMo Gym.

The `question` field is used verbatim as the user message, and the system prompt
for the config's task family is prepended from `prompts.py`. Both come from
upstream: the dataset ships the user message, and the paper publishes the system
prompts as Tables H, I, J and K. This matches the paper's own baseline rows,
which are zero-shot; the separately labelled few-shot and chain-of-thought rows
are inference-strategy ablations, not the baseline.

`--no-system-prompt` reproduces the earlier omit-the-system-prompt behaviour. It
exists so the effect stays measurable, not because omitting it is supported.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts import SYSTEM_PROMPTS  # noqa: E402


DATASET = "MolDeTox/MolDeTox"

# Files were revised 2026-08-27 with split sizes changed, so the revision is
# pinned rather than tracking `main`. The pragma is for detect-secrets, which
# reads any 40-character hex string as a high-entropy secret; this one is a
# public Hugging Face commit SHA and is meant to be read.
REVISION = "059e1926fa62a07c3f01afadadf72cce154cfb7f"  # pragma: allowlist secret

# All eight QA configs. The ninth config, `toxicitycliff`, is the source pair
# table rather than a task and is loaded separately for the toxic-molecule join.
#
# `scoring_mode` tells the resources server how to compare the answer: Task 3
# answers are whole molecules written as SMILES or SAFE, while Task 1 and 2
# answers are SAFE fragment sets, which are not parseable as molecules.
SCORING_MODES = {
    "task1_single": "fragments_task1",
    "task1_multi": "fragments_task1",
    "task2_single": "fragments_task2",
    "task2_multi": "fragments_task2",
    "task3_smiles_gen_single": "smiles",
    "task3_smiles_gen_multi": "smiles",
    "task3_safe_gen_single": "safe",
    "task3_safe_gen_multi": "safe",
}
SUPPORTED_CONFIGS = tuple(SCORING_MODES)


def _download(filename: str) -> str:
    """Fetch one file from the pinned dataset revision.

    Every file is fetched directly rather than through `datasets.load_dataset`.
    The repository is eight plain `.jsonl` files under a fixed path plus two
    CSVs, so a builder buys nothing, and this repository's dataset card is not
    honoured anyway: `load_dataset` reports `Some datasets params were ignored:
    ['type']` and selects the JSON builder for the CSV configs, which then fails.

    Direct download was checked against `load_dataset` on all eight QA configs of
    the pinned revision and produced identical rows, so dropping the builder does
    not change the prepared data.

    This does not shrink the install. `nemo-gym` declares `datasets` in its own
    base dependencies, so it, PyArrow and Pandas arrive regardless; what goes
    away is a duplicate direct declaration and a builder that mis-selects on this
    repository.
    """
    return hf_hub_download(repo_id=DATASET, filename=filename, repo_type="dataset", revision=REVISION)


def _load_qa_rows(config: str, split: str) -> list[dict]:
    """Read one QA config as a list of rows."""
    with open(_download(f"MolDeTox_QA/{split}/{split}_{config}.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _toxic_smiles_by_index(split: str) -> dict[int, str]:
    """Map `source_index` to the toxic input molecule.

    The QA rows embed the toxic molecule in their prompt text but do not carry it
    as a field. `source_index` joins to `orig_test_index` in the toxicitycliff
    config, which is exact over the test split, so the molecule is looked up
    rather than parsed back out of the prompt.
    """
    with open(_download(f"toxicitycliff_{split}.csv"), newline="") as f:
        return {int(row["orig_test_index"]): row["toxic_smiles"] for row in csv.DictReader(f)}


def format_row(row: dict, config: str, toxic_smiles: dict[int, str], system_prompt: bool = True) -> dict:
    metadata = {
        "answer": row["answer"]["answer"],
        "scoring_mode": SCORING_MODES[config],
        "config": config,
        "task": config.partition("_")[0],
        "variant": config.rsplit("_", 1)[-1],
        "endpoint": row["endpoint"],
        "dataset_name": row["dataset_name"],
        "toxic_smiles": toxic_smiles.get(int(row["source_index"])),
        "id": row["id"],
    }
    # Only the Task 2 configs carry the fragments shared by both members of the pair.
    if "common_safe_fragments" in row:
        metadata["common_safe_fragments"] = row["common_safe_fragments"]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPTS[config]})
    messages.append({"role": "user", "content": row["question"]})

    return {
        "responses_create_params": {"input": messages},
        "verifier_metadata": metadata,
        "agent_ref": {"type": "responses_api_agents", "name": "moldetox_simple_agent"},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare MolDeTox for NeMo Gym")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--config", default="task3_smiles_gen_single", choices=SUPPORTED_CONFIGS)
    # Constrained rather than documented: the server README states that nothing
    # here reads the ~202,000-row train split, and a free-form string silently
    # made that false. `train` is a plausible fine-tuning source and is tracked
    # separately; it is not an evaluation input, so it is not reachable here.
    parser.add_argument(
        "--split",
        default="test",
        choices=("test",),
        help="Only 'test' is in scope for evaluation",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to output")
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Omit upstream's system prompt; for measuring its effect, not for reporting",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    toxic_smiles = _toxic_smiles_by_index(args.split)
    ds = _load_qa_rows(args.config, args.split)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = 0
    with open(output_path, "w") as fout:
        for count, row in enumerate(ds, start=1):
            formatted = format_row(row, args.config, toxic_smiles, system_prompt=not args.no_system_prompt)
            missing += formatted["verifier_metadata"]["toxic_smiles"] is None
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {min(count, args.limit or count)} rows to {output_path}", file=sys.stderr)
    if missing:
        print(f"WARNING: {missing} rows had no toxicitycliff join for source_index", file=sys.stderr)


# python scripts/prepare_moldetox.py --output data/val.jsonl
# python scripts/prepare_moldetox.py --output data/example.jsonl --limit 5
if __name__ == "__main__":
    main()
