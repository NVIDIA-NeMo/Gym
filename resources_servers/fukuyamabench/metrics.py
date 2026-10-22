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

"""Deterministic mechanism-pathway scoring for FukuyamaBench.

Ported from the upstream scorer `eval/eval_infer_pathway.py` at revision
63bb79f912f0b2de593996b80ffeea894f6f1a59 of
https://github.com/HaCTang/ReactionMechanismReasoning (Apache-2.0).

No judge model is involved. A predicted pathway is compared to ground truth by
canonical-SMILES set equality at each checkpoint, with checkpoints matched in
order against the predicted steps.
"""

import re
from typing import Optional

from rdkit import Chem, RDLogger


# A model emitting an unparseable SMILES is an ordinary scoring outcome, not an
# error worth a stderr line per occurrence.
RDLogger.DisableLog("rdApp.*")

_ATOM_MAP_RE = re.compile(r":\d+")


def strip_atom_mapping(smiles: str) -> str:
    """Drop atom-map indices, e.g. ``[CH3:1][O:2]`` -> ``[CH3][O]``.

    Prompts supply atom-mapped reactants and models echo the mapping, while gold
    products are unmapped, so mapping is removed before any comparison.
    """
    if not smiles:
        return smiles
    return _ATOM_MAP_RE.sub("", smiles)


def _organometallic_mol(smiles: str):
    """Reparse without valence checking, for transition-metal complexes.

    RDKit's valence model rejects ligand counts that are ordinary for Rh, Cr, Ti
    and Mo centres, so those species fail the default parse. Sanitising with
    everything except ``SANITIZE_PROPERTIES`` keeps ring and aromaticity
    perception — and therefore canonical output — while dropping only the
    valence assertion the metals violate.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        return mol
    except Exception:
        return None


def canonical_smiles(smiles: str) -> Optional[str]:
    """Canonical SMILES, or None if RDKit cannot parse it even permissively.

    The permissive path is a fallback rather than the default, so the 1,716
    species that parse strictly are canonicalised exactly as before. It recovers
    17 organometallic species across sets B and C, each of which canonicalises to
    its own distinct form — no previously-separate species are merged.
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        mol = _organometallic_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonical_species(smiles_list: list[str]) -> tuple[set[str], bool]:
    """Canonicalise into a set of species, reporting whether everything parsed.

    Entries are split on ``.`` so that one dotted string and several separate
    entries describing the same species compare equal. Upstream splits only the
    prediction side, which leaves any checkpoint whose gold products contain a
    dotted entry unmatchable by construction: feeding gold back in as the
    prediction fails 11 of the 241 set-B/C cases under upstream's rule.
    Splitting both sides is a deliberate divergence that makes them scoreable.

    The second return value matters because dropping unparseable components
    silently is what lets a prediction earn credit for output it never wrote
    validly. Callers decide whether a parse failure is disqualifying; it always
    is on the prediction side.
    """
    species: set[str] = set()
    all_parsed = True
    for smi in smiles_list:
        if not smi:
            continue
        for part in str(smi).split("."):
            part = part.strip()
            if not part:
                continue
            canon = canonical_smiles(part)
            if canon is None:
                all_parsed = False
            else:
                species.add(canon)
    return species, all_parsed


def _split(smiles: str) -> list[str]:
    return [part.strip() for part in strip_atom_mapping(smiles).split(".") if part.strip()]


def product_smiles_from_step(step: dict) -> tuple[list[str], bool]:
    """Pull product SMILES out of one predicted step, reporting well-formedness.

    Accepts ``product_smiles`` (a dot-joined string or a list) or ``products``
    (a list of strings or of ``{"smiles": ...}`` dicts), matching the shapes the
    upstream prompt and dataset both produce.

    The second return value is false when the step, or any member of a list it
    holds, is not one of those shapes. Dropping unrecognised members silently is
    what let ``["CCO", null]`` score an exact match against gold ``CCO`` — the
    same invalid-output credit the parse-validity rule exists to prevent — so
    callers must treat a malformed member as disqualifying rather than absent.
    """
    if not isinstance(step, dict):
        return [], False

    if "product_smiles" in step:
        smiles = step["product_smiles"]
        if isinstance(smiles, str):
            return _split(smiles), True
        if isinstance(smiles, list):
            out: list[str] = []
            well_formed = True
            for member in smiles:
                if isinstance(member, str):
                    out.extend(_split(member))
                else:
                    well_formed = False
            return out, well_formed
        return [], False

    if "products" in step:
        products = step["products"]
        if isinstance(products, list):
            out = []
            well_formed = True
            for member in products:
                if isinstance(member, str):
                    out.extend(_split(member))
                elif isinstance(member, dict) and isinstance(member.get("smiles"), str):
                    out.extend(_split(member["smiles"]))
                else:
                    well_formed = False
            return out, well_formed
        return [], False

    return [], False


def compare_step_products(
    pred_products: list[str],
    gt_products: list[str],
    lenient: bool = True,
    pred_well_formed: bool = True,
) -> tuple[bool, Optional[str]]:
    """Compare one predicted step's products against one gold step's products.

    Returns ``(is_match, error_type)``. Under ``lenient`` (the upstream default)
    a non-empty proper subset of the gold products counts as a match, which is
    how a model that names the main product but omits a leaving group is
    credited.

    A prediction containing any unparseable component fails outright. Upstream
    drops such components before comparing, so a prediction of the right product
    plus one malformed fragment scores an exact match even under ``strict`` —
    crediting output the model never wrote validly, and making strict mode mean
    something other than product-set equality.
    """
    pred_canon, pred_all_parsed = canonical_species(pred_products)
    gt_canon, _ = canonical_species(gt_products)

    if not pred_canon and not gt_canon:
        return False, "both_invalid"
    if not pred_canon:
        return False, "invalid_pred"
    if not gt_canon:
        return False, "invalid_gt"
    if not pred_all_parsed or not pred_well_formed:
        return False, "invalid_pred_component"

    if pred_canon == gt_canon:
        return True, None
    if lenient and pred_canon.issubset(gt_canon):
        return True, "subset_match"
    if gt_canon.issubset(pred_canon):
        return False, "superset_match"
    if pred_canon & gt_canon:
        return False, "partial_match"
    return False, "mismatch"


def score_pathway(
    pred_pathway: list[dict],
    gt_pathway: list[dict],
    checkpoints: list[list[int]],
    lenient: bool = True,
) -> dict:
    """Score a predicted pathway against gold using ordered checkpoint matching.

    Each checkpoint is a group of gold step IDs that are mutually acceptable at
    that point. Checkpoints are consumed in order and the scan over predicted
    steps only ever moves forward, so a predicted step that satisfies checkpoint
    *i* cannot also satisfy checkpoint *i+1*. A checkpoint that exhausts the
    remaining predicted steps fails, and — because the cursor is then spent —
    every later checkpoint fails with it.

    This tolerates a model predicting finer-grained steps than the gold
    mechanism, which is the reason checkpoints exist rather than a positional
    step-for-step comparison.
    """
    pred_length = len(pred_pathway)
    matches: list[dict] = []
    correct = 0
    pred_idx = 0

    for ckpt_idx, step_ids in enumerate(checkpoints):
        # Out-of-range IDs would come from a malformed ckpt.txt; drop them rather
        # than letting them shift the pairing of IDs to products.
        candidates = [
            (step_id, gt_pathway[step_id - 1].get("products", []))
            for step_id in step_ids
            if 1 <= step_id <= len(gt_pathway)
        ]

        match = {
            "checkpoint": ckpt_idx + 1,
            "equivalent_steps": step_ids,
            "match": False,
            "matched_pred_step": None,
            "matched_gt_step": None,
            "error": None,
        }

        found = False
        while pred_idx < pred_length and not found:
            pred_products, pred_well_formed = product_smiles_from_step(pred_pathway[pred_idx])
            for gt_step_id, gt_products in candidates:
                is_match, _ = compare_step_products(
                    pred_products, gt_products, lenient, pred_well_formed=pred_well_formed
                )
                if is_match:
                    match.update(
                        match=True,
                        matched_pred_step=pred_idx + 1,
                        matched_gt_step=gt_step_id,
                    )
                    correct += 1
                    found = True
                    break
            pred_idx += 1

        if not found:
            match["error"] = "checkpoint_not_found"
        matches.append(match)

    total = len(checkpoints)
    return {
        "exact_match": total > 0 and correct == total,
        "checkpoints_correct": correct,
        "checkpoints_total": total,
        "checkpoint_accuracy": correct / total if total else 0.0,
        "any_checkpoint_correct": correct > 0,
        "pred_length": pred_length,
        "gt_steps": len(gt_pathway),
        "checkpoint_matches": matches,
    }
