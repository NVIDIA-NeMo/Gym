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
#
# `oracle_solubility_eval` below is adapted from Future-House/ether0, file `src/ether0/rewards.py`
# at commit b77e0d8eb06f62c8528da72bff22d4c7fee11371
# (https://github.com/Future-House/ether0/blob/b77e0d8eb06f62c8528da72bff22d4c7fee11371/src/ether0/rewards.py),
# licensed under the Apache License 2.0 (upstream `LICENSE`; the upstream file carries no per-file
# copyright notice). The adaptation requires every listed functional group instead of any one,
# looks helpers up on `ether0.rewards` at call time, and reformats comments; the scaffold,
# tanimoto, admissibility and numerical-tolerance branches are unchanged. `str_eval` and
# `get_largest_mol` replace the upstream functions of the same name and signature with independent
# implementations. Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, licensed
# under the Apache License 2.0 (header above).

"""The reward functions the ether0 verifier grades with.

``ether0.rewards`` (Future-House/ether0, cloned by ``setup_ether0``) provides the graders; this
module fixes three of them and exposes the table ``app.py`` dispatches on:

1. ``product_eval`` / ``caption_eval``: every ``.``-separated fragment of the answer must be a
   valid, non-empty SMILES before the principal fragment is compared. Upstream silently dropped
   short or unparsable fragments, so hedged or garbage fragments slipped through.
2. ``sol_eval``: a functional-group constraint requires all listed groups, not any one of them.
3. ``str_eval``: a molecule reference is compared as a molecule (any SMILES of the same molecule
   matches, text does not); text references keep the case-insensitive comparison.

Imports happen after ``setup_ether0.ensure_ether0()`` has run.
"""

import ast
from typing import cast

import exmol
from ether0 import rewards
from ether0.data import mol_from_smiles
from ether0.models import RewardReason
from pydantic import JsonValue
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolToSmiles  # pylint: disable=no-name-in-module


def get_largest_mol(smiles: str) -> Chem.Mol | None:
    """Reject empty or invalid fragments, then select a deterministic principal fragment."""
    parts = smiles.split(".")
    if any(not part.strip() for part in parts):
        return None
    mols = [mol_from_smiles(part) for part in parts]
    if any(mol is None for mol in mols):
        return None
    return max(mols, key=lambda mol: (mol.GetNumHeavyAtoms(), MolToSmiles(mol)))


# product_eval and caption_eval look this helper up in their module at call time, so rebinding it
# is the whole of fix 1; the two graders themselves are unchanged.
rewards.get_largest_mol = get_largest_mol


def str_eval(
    yhat: str,
    y: str,
    soft: bool = False,  # noqa: ARG001
    test: bool = False,  # noqa: ARG001
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Choose molecular or text comparison from the reference answer."""
    rewards.set_full_smiles(yhat, metadata)
    m = mol_from_smiles(y)
    if m is None:
        return 1.0 if rewards.normalize_iupac(yhat) == rewards.normalize_iupac(y) else 0.0
    mhat = mol_from_smiles(yhat)
    if mhat is None:
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0
    return rewards.exact_mol_match(mhat, m)


def oracle_solubility_eval(
    yhat: str,
    y: str,
    soft: bool = False,  # noqa: ARG001
    test: bool = False,
    metadata: dict[str, JsonValue] | None = None,
) -> float:
    """Upstream ``oracle_solubility_eval`` with ``all`` for group constraints.

    The oracle and admissibility helpers are looked up on ``ether0.rewards`` at call time so that
    deployments which rebind them (local solubility models, tests) keep working.
    """
    rewards.set_full_smiles(yhat, metadata)
    if "." in yhat:  # single molecules only
        return 0.0
    mhat = mol_from_smiles(yhat)
    if mhat is None:
        RewardReason.INVALID_MOL.set_reason(metadata)
        return 0.0

    y_args: tuple[str, str | list[str], float | str, str] = ast.literal_eval(y)
    constraint_type, constraint_data = y_args[:2]
    target = float(y_args[2])

    ref_mol: Chem.Mol | None = None
    if constraint_type == "scaffold":
        ref_mol = mol_from_smiles(cast(str, constraint_data))
        if ref_mol is None:
            raise NotImplementedError(f"Didn't handle when {constraint_data=} is invalid.")
        if not mhat.HasSubstructMatch(ref_mol):
            RewardReason.FAILED_CONSTRAINT.set_reason(metadata)
            return 0.0
    elif constraint_type == "groups":
        groups = [g.lower() for g in exmol.get_functional_groups(mhat, return_all=True)]
        if not all(group.lower() in groups for group in constraint_data):
            RewardReason.FAILED_CONSTRAINT.set_reason(metadata)
            return 0.0
    elif constraint_type == "tanimoto":
        ref_mol = mol_from_smiles(cast(str, constraint_data))
        if (
            rewards.tanimoto_similarity(mhat, ref_mol, atom_threshold=0.2)
            < rewards.SMOOTH_THRESHOLD_TANIMOTO_SIMILARITY
        ):
            RewardReason.FAILED_CONSTRAINT.set_reason(metadata)
            return 0.0
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    if not rewards.is_reasonable_molecule(mhat, metadata, test, ref_mol=ref_mol):
        return 0.0

    result = rewards.fetch_solubility(yhat)
    if "solubility" in result:
        # Typical solubility accuracy; 0.01 below 0.5 so a restatement of a "change by 0.5"
        # question does not count as a match.
        if abs(result["solubility"] - target) > (0.5 - 0.01):
            RewardReason.WRONG_NUMERICAL_ANSWER.set_reason(metadata)
            return 0.0
        return 1.0
    RewardReason.INVALID_MOL.set_reason(metadata)
    return 0.0


EVAL_FUNCTIONS: dict[str, rewards.RewardEvalFn] = {
    **rewards.EVAL_FUNCTIONS,
    "str_eval": str_eval,
    "sol_eval": oracle_solubility_eval,
}
