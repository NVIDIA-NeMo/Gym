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

"""Prepare FukuyamaBench for NeMo Gym.

FukuyamaBench publishes no Hugging Face dataset: the 319 cases exist only as a
directory tree in the upstream GitHub repository, so this fetches the source
tarball at a pinned commit rather than calling `load_dataset`.

Prompts are reproduced verbatim from upstream `eval/prompts/infer_pathway_prompts.yaml`
at the same revision, so a run of this data exercises the shipped prompt. Note
that the paper's Appendix C prints a longer prompt than the repository ships
(it adds an output-guidelines block and a worked example), so neither this nor
upstream's own harness is byte-identical to what produced the published tables.
"""

import argparse
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


REPO = "HaCTang/ReactionMechanismReasoning"

# The paper cites commit 7c1cb69 for its reported results, but that SHA does not
# exist in the public repository. This pins the current main HEAD instead, which
# is an unverified proxy for the evaluated state.
REVISION = "63bb79f912f0b2de593996b80ffeea894f6f1a59"
TARBALL_URL = f"https://codeload.github.com/{REPO}/tar.gz/{REVISION}"

# Verbatim from upstream eval/prompts/infer_pathway_prompts.yaml, key
# `pathway_student_system_prompt`.
SYSTEM_PROMPT = """Role: You are an expert Computational Chemist and Mechanism Designer. You prioritize chemical validity and strict SMILES syntax.

Task: Given the starting reactants (SMILES with atom mapping), predict the complete multi-step reaction mechanism.

## Instructions

1. **Reasoning Section (Mandatory)**
   Analyze the mechanism step-by-step. For each step, you MUST follow this structured process:

   A. **Reaction Group Property Analysis**:
      - Current reactive species and their electronic properties.
      - Which interaction will occur and why.
      - *Example*: [Li+][C-]CCC has strong alkalinity, weak nucleophilicity, prone to acid-base reaction rather than SN reaction

   B. **Mechanism Analysis**:
      - Analyze electron flow (Source -> Sink).
      - Identify bond formation/breaking using Atom Map IDs.

   C. **Atom State Definition (CRITICAL)**:
      - Before generating SMILES, explicitly define the state of the reaction centers.
      - Format: "Atom [ID]: [Element], [Valence/Bonds], [Charge]"
      - *Example*: "Atom :5 (Oxygen): 1 bond, 3 lone pairs -> Charge -1."
      - *Example*: "Atom :3 (Carbon): 4 single bonds -> Neutral."

   D. **SMILES Syntax Translation**:
      - **Rule Check**: If Atom X is defined as Negative, the SMILES segment MUST be `[Element-:ID]`.
      - **Sanity Check**: Does the draft SMILES imply a pentavalent carbon (5 bonds) or a negative carbon with 4 bonds? If yes, CORRECT IT immediately by moving the charge to the heteroatom.

   E. **Predicted Intermediate**:
      - The final SMILES string derived from the logic above.

2. **Result Section**
   Provide a JSON array with each elementary step. Continue until you reach the terminal
   stable product.

## Output Format

## Reasoning

**Step 1:**
[Your analysis of the first elementary step...]
Predicted intermediate: [describe or show SMILES]

**Step 2:**
[Your analysis...]
...

**Step N (Terminal):**
[Analysis of the final step leading to stable product...]

## Result

```json
[
  {
    "step_id": 1,
    "reaction_type": "e.g., Nucleophilic Addition",
    "description": "Brief description of what happens",
    "reactive_sites": {"nucleophile": "[O:5]", "electrophile": "[C:3]"},
    "product_smiles": "SMILES of the intermediate/product",
    "is_terminal": false
  },
  ...
  {
    "step_id": N,
    "reaction_type": "...",
    "description": "...",
    "reactive_sites": {...},
    "product_smiles": "SMILES of final product",
    "is_terminal": true
  }
]
```
"""

# Verbatim from the same file, key `pathway_student_template_prompt`.
USER_TEMPLATE = """Starting Reactants:
{starting_reactants}

Reaction Conditions:
{conditions}

Predict the complete reaction mechanism from these starting materials to the final product.
"""


def fetch_mechanisms(dest: Path) -> Path:
    """Download and unpack the upstream tarball, returning the mechanisms directory."""
    dest.mkdir(parents=True, exist_ok=True)
    mechanisms = dest / f"ReactionMechanismReasoning-{REVISION}" / "fukuyama_bench" / "mechanisms"
    if mechanisms.is_dir():
        return mechanisms

    print(f"Downloading {TARBALL_URL}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        with urllib.request.urlopen(TARBALL_URL) as response:
            tmp.write(response.read())
        tmp.flush()
        with tarfile.open(tmp.name) as tar:
            tar.extractall(dest, filter="data")

    if not mechanisms.is_dir():
        raise SystemExit(f"Expected {mechanisms} in the downloaded archive")
    return mechanisms


def render_conditions(reactions: list[dict]) -> str:
    """Flatten every reaction's conditions the way upstream's prompt formatter does."""
    parts = []
    for rxn in reactions:
        for cond in rxn.get("conditions", []):
            text = cond.get("text") or cond.get("smiles") or ""
            if not text:
                continue
            role = cond.get("role", "")
            step = cond.get("reaction_step", "")
            parts.append(f"{role}: {text}" + (f" (step {step})" if step else ""))
    return "\n".join(parts)


def load_case(case_dir: Path) -> Optional[dict]:
    """Read one case directory into the fields the prompt and scorer need."""
    mechanism_path = case_dir / "mechanism.json"
    ckpt_path = case_dir / "ckpt.txt"
    if not mechanism_path.is_file() or not ckpt_path.is_file():
        return None

    mechanism_data = json.loads(mechanism_path.read_text(encoding="utf-8"))
    reactions = mechanism_data.get("reactions", [])
    steps = mechanism_data.get("mechanism", [])
    if not reactions or not steps:
        return None

    checkpoints = []
    for line in ckpt_path.read_text(encoding="utf-8").splitlines():
        ids = [int(x) for x in line.strip().split(",") if x.strip()]
        if ids:
            checkpoints.append(ids)
    if not checkpoints:
        return None

    gt_pathway = [
        {
            "step_id": step.get("mechanism_id"),
            "products": [p.get("smiles") for p in step.get("products", []) if p.get("smiles")],
        }
        for step in steps
    ]

    # The model is prompted with atom-mapped reactants, matching upstream's
    # SFT/RL input format; plain reaction SMILES is the fallback.
    starting_reactants = ""
    mapped_path = case_dir / "mapped_rxn.json"
    if mapped_path.is_file():
        mapped = json.loads(mapped_path.read_text(encoding="utf-8")).get("mechanism_rxn", [])
        if mapped:
            starting_reactants = mapped[0].get("mapped_rxn", "").split(">>")[0]

    if not starting_reactants:
        rxn_path = case_dir / "rxn.json"
        if rxn_path.is_file():
            rxn = json.loads(rxn_path.read_text(encoding="utf-8")).get("mechanism_rxn", [])
            if rxn:
                starting_reactants = rxn[0].get("rxn_smiles", "").split(">>")[0]

    if not starting_reactants:
        starting_reactants = ".".join(p.get("smiles", "") for p in steps[0].get("reactants", []) if p.get("smiles"))
    if not starting_reactants:
        return None

    return {
        "case_id": case_dir.name,
        "case_set": case_dir.name[0].upper(),
        "starting_reactants": starting_reactants,
        "conditions": render_conditions(reactions),
        "gt_pathway": gt_pathway,
        "checkpoints": checkpoints,
        "n_gt_steps": len(steps),
    }


def format_row(case: dict, lenient: bool) -> dict:
    user_prompt = USER_TEMPLATE.replace("{starting_reactants}", case["starting_reactants"]).replace(
        "{conditions}", case["conditions"]
    )
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        },
        "verifier_metadata": {
            "case_id": case["case_id"],
            "case_set": case["case_set"],
            "gt_pathway": case["gt_pathway"],
            "checkpoints": case["checkpoints"],
            "lenient": lenient,
            "n_gt_steps": case["n_gt_steps"],
            "conditions": case["conditions"],
            "starting_reactants": case["starting_reactants"],
        },
        "agent_ref": {"type": "responses_api_agents", "name": "fukuyamabench_simple_agent"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare FukuyamaBench for NeMo Gym")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["B", "C"],
        help=(
            "Case-ID leading characters to include, matching upstream's prefix filter. "
            "Upstream tiers are A, B and C; B and C carry the sub-20%% published figures"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require exact product-set equality instead of upstream's default subset match",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Existing mechanisms directory; skips the download when set",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "fukuyamabench",
        help="Where the upstream tarball is unpacked",
    )
    args = parser.parse_args()

    mechanisms = args.source_dir or fetch_mechanisms(args.cache_dir)
    wanted = {s.upper() for s in args.sets}
    case_dirs = sorted(d for d in mechanisms.iterdir() if d.is_dir() and d.name[0].upper() in wanted)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(output_path, "w") as fout:
        for case_dir in case_dirs:
            case = load_case(case_dir)
            if case is None:
                skipped += 1
                continue
            fout.write(json.dumps(format_row(case, not args.strict), ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"Wrote {written} rows to {output_path}", file=sys.stderr)
    if skipped:
        print(f"WARNING: skipped {skipped} cases with missing or empty mechanism/checkpoint files", file=sys.stderr)


# python scripts/prepare_fukuyamabench.py --output data/val.jsonl
# python scripts/prepare_fukuyamabench.py --output data/example.jsonl --limit 5
if __name__ == "__main__":
    main()
