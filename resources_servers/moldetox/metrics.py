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

"""Deterministic scoring primitives for MolDeTox.

No model is involved in any function here. PRS (Property Retention Score) is
deliberately absent. Appendix D.3 is unambiguous about its shape and reference
point: `PRS = exp(-|S(Mt) - S(M_hat)|)`, where `Mt` is the toxic input molecule
and `S` aggregates six Lipinski/Veber properties. What it never states are the
per-property desirability curves `d_i` and their weights `w_i`, and there is no
upstream reference implementation to check against. Those curves determine the
value entirely, so implementing a guess would put an unfaithful number inside
the scorer under a name readers would compare to the paper.
"""

from collections import Counter
from typing import Optional

import safe
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity


# RDKit logs every unparseable SMILES to stderr; model output produces these routinely.
RDLogger.DisableLog("rdApp.*")

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_RDKIT_GEN = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)

FINGERPRINTS = ("rdk", "maccs", "morgan")


def to_mol(smiles: Optional[str]) -> Optional[Chem.Mol]:
    if not smiles or not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles.strip())


def canonical_smiles(smiles: Optional[str], keep_stereo: bool = True) -> Optional[str]:
    """Canonical SMILES, or None if unparseable.

    Handles multi-component species (the dot-separated salts in the gold data)
    natively. `keep_stereo=False` erases stereochemistry before canonicalising,
    which is the lenient comparison mode.
    """
    mol = to_mol(smiles)
    if mol is None:
        return None
    if not keep_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def is_valid(smiles: Optional[str]) -> bool:
    return to_mol(smiles) is not None


def exact_match(pred: Optional[str], gold: str, keep_stereo: bool = True) -> bool:
    """Structural equality via canonical SMILES. An unparseable prediction is a miss."""
    pred_canon = canonical_smiles(pred, keep_stereo=keep_stereo)
    if pred_canon is None:
        return False
    gold_canon = canonical_smiles(gold, keep_stereo=keep_stereo)
    if gold_canon is None:
        raise ValueError(f"gold answer is not valid SMILES: {gold!r}")
    return pred_canon == gold_canon


def _fingerprint(mol: Chem.Mol, kind: str):
    if kind == "morgan":
        return _MORGAN_GEN.GetFingerprint(mol)
    if kind == "rdk":
        return _RDKIT_GEN.GetFingerprint(mol)
    if kind == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    raise ValueError(f"unknown fingerprint {kind!r}; expected one of {FINGERPRINTS}")


def tanimoto(pred: Optional[str], gold: str, kind: str) -> float:
    """Tanimoto similarity over `kind` fingerprints. Unparseable prediction scores 0.0."""
    pred_mol, gold_mol = to_mol(pred), to_mol(gold)
    if pred_mol is None or gold_mol is None:
        return 0.0
    return TanimotoSimilarity(_fingerprint(pred_mol, kind), _fingerprint(gold_mol, kind))


def levenshtein(a: Optional[str], b: str) -> int:
    """Character-level edit distance. A missing prediction costs the full length of `b`."""
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def score_smiles(pred: Optional[str], gold: str, keep_stereo: bool = True) -> dict:
    """Every SMILES-level metric for one prediction.

    `levenshtein` is None when the prediction is not a usable molecule. Charging
    the full gold length instead inflates the mean badly — measured at 10.07
    against upstream's 6.70 on `task3_safe_gen_single`, where roughly 7% of
    predictions do not decode and so have no string to compare at all.
    """
    scores = {
        "exact_match": float(exact_match(pred, gold, keep_stereo=keep_stereo)),
        "validity": float(is_valid(pred)),
        "levenshtein": float(levenshtein(pred, gold)) if is_valid(pred) else None,
    }
    for kind in FINGERPRINTS:
        scores[f"tanimoto_{kind}"] = tanimoto(pred, gold, kind)
    return scores


def decode_safe(safe_string: Optional[str]) -> Optional[str]:
    """SAFE string to SMILES via the datamol decoder, or None if reconstruction fails.

    A whole SAFE string also parses as SMILES, but as disconnected fragments
    rather than the assembled molecule, so `Chem.MolFromSmiles` is not a
    substitute for decoding here.
    """
    if not safe_string or not isinstance(safe_string, str):
        return None
    try:
        return safe.decode(safe_string.strip())
    except Exception:
        return None


def score_safe(pred: Optional[str], gold: str, keep_stereo: bool = True) -> dict:
    """Task 3 SAFE generation: decode, then score as SMILES.

    Upstream counts a failed SAFE-to-SMILES reconstruction as invalid rather than
    masking it, so a prediction that will not decode scores zero everywhere. The
    gold is decoded too, since the stored answer is itself a SAFE string.
    """
    gold_smiles = decode_safe(gold)
    if gold_smiles is None:
        raise ValueError(f"gold SAFE string does not decode: {gold!r}")
    return score_smiles(decode_safe(pred), gold_smiles, keep_stereo=keep_stereo)


def parse_fragments(fragment_string: Optional[str]) -> list[str]:
    """Split a dot-separated SAFE fragment set into its parts."""
    if not fragment_string or not isinstance(fragment_string, str):
        return []
    return [f for f in (part.strip() for part in fragment_string.strip().split(".")) if f]


def _fragment_f1(pred_frags: list[str], gold_frags: list[str]) -> float:
    """Fragment-overlap F1 over **sets**, per Appendix D.1.

    The paper writes `G = set(T(s))`, `P = set(T(ŝ))`, so a fragment repeated in
    the prediction neither helps nor hurts. Counting multiplicities instead
    understates F1 whenever a model repeats a fragment: `A.A` against `A.B`
    scores 0.50 as a multiset and 0.67 as the paper defines it.

    Exact-match accuracy is unaffected; that one is defined over the multiset.
    """
    pred_set, gold_set = set(pred_frags), set(gold_frags)
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def fragment_levenshtein(pred_frags: list[str], gold_frags: list[str]) -> Optional[float]:
    """Task 2's fragment-level Levenshtein distance, per Appendix D.2.

    The paper is explicit that this "is not computed as a single edit distance
    over the whole SAFE string". For each predicted fragment it takes the minimum
    distance to any gold fragment, then averages those minima over the predicted
    fragments. A whole-string distance is a different and much larger quantity:
    `A.A` against `A.B` is 1 character but 0.0 by this definition.

    Returns None when there is nothing to average over, so an absent value is
    never read as a perfect zero.
    """
    if not pred_frags or not gold_frags:
        return None
    return sum(min(levenshtein(p, g) for g in gold_frags) for p in pred_frags) / len(pred_frags)


def score_fragments(pred: Optional[str], gold: str, task: str = "task2") -> dict:
    """Tasks 1 and 2: multiset exact match over SAFE fragments.

    Individual SAFE fragments carry unmatched attachment digits (`Cc1cccc6c1`,
    `C46`) and are not parseable as molecules, so canonicalisation is impossible
    and fragments are compared as strings. Order does not matter, multiplicity
    does. Task 1 fragments are selected from the decomposition given in the
    prompt, so verbatim comparison is what upstream intends.

    The auxiliary metrics differ by task because the paper defines them that way.
    Appendix D.1 gives Task 1 exact match and F1 only, with no edit distance of
    any kind; Appendix D.2 adds the fragment-level distance above for Task 2.
    Emitting a Task 1 distance would invent a number upstream never reports.
    """
    pred_frags, gold_frags = parse_fragments(pred), parse_fragments(gold)
    scores = {
        "exact_match": float(Counter(pred_frags) == Counter(gold_frags) and bool(gold_frags)),
        "fragment_f1": _fragment_f1(pred_frags, gold_frags),
        "n_fragments_predicted": float(len(pred_frags)),
    }
    if task == "task2":
        scores["levenshtein"] = fragment_levenshtein(pred_frags, gold_frags)
    return scores
