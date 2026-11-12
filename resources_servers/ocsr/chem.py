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
"""SMILES extraction and OCSR scoring.

Answer extraction: the requested {"smiles": ...} JSON object first, else the last RDKit-valid SMILES
in a backtick span. Scores: exact (canonical isomeric SMILES), graph (no stereo), InChIKey, Tanimoto.
"""

import json
import re
from typing import Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


RDLogger.DisableLog("rdApp.*")

MAX_SMILES_CHARS = 2000
_THINK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|$)", re.I | re.S)
_JSON_OBJ = re.compile(r'\{[^{}]*"smiles"[^{}]*\}', re.S)
_SMILES_FIELD = re.compile(r'"smiles"\s*:\s*"([^"]+)"')
_BACKTICK = re.compile(r"`([^`\n]{2,150})`")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _mol(smi: Optional[str]):
    if not smi or len(smi) > MAX_SMILES_CHARS:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def canon(smi: Optional[str], iso: bool = True) -> Optional[str]:
    """RDKit canonical SMILES, or None. ``iso=False`` drops stereo."""
    m = _mol(smi)
    return Chem.MolToSmiles(m, isomericSmiles=iso) if m is not None else None


def inchikey(smi: Optional[str]) -> Optional[str]:
    m = _mol(smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m) or None
    except Exception:
        return None


def tanimoto(a: Optional[str], b: Optional[str]) -> float:
    ma, mb = _mol(a), _mol(b)
    if ma is None or mb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(_MORGAN.GetFingerprint(ma), _MORGAN.GetFingerprint(mb))


def gold_is_usable(gold: str) -> bool:
    return canon(gold) is not None


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> blocks; if only a closing tag is present, keep what follows it."""
    text = _THINK.sub("", text or "")
    if re.search(r"</(?:think|thinking)\s*>", text, re.I):
        text = re.split(r"</(?:think|thinking)\s*>", text, flags=re.I)[-1]
    return text


def extract_answer(text: str) -> tuple[Optional[str], Optional[str]]:
    """(smiles, how): how is "json" for the requested JSON object, "salvage" otherwise, None if nothing."""
    text = strip_reasoning(text)
    for cand in reversed(_JSON_OBJ.findall(text)):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("smiles"), str) and obj["smiles"].strip():
            return obj["smiles"].strip(), "json"
    fields = _SMILES_FIELD.findall(text)
    if fields:
        return fields[-1].strip(), "json"
    cands = _BACKTICK.findall(text) + fields
    valid = [c.strip() for c in cands if _mol(c.strip()) is not None]
    return (valid[-1], "salvage") if valid else (None, None)


def score(pred: Optional[str], gold: str) -> dict[str, float]:
    pred_c = canon(pred, True)
    valid = pred_c is not None
    pk, gk = inchikey(pred), inchikey(gold)
    return {
        "valid": float(valid),
        "exact": float(valid and pred_c == canon(gold, True)),
        "graph": float(valid and canon(pred, False) == canon(gold, False)),
        "inchikey": float(pk is not None and pk == gk),
        "tanimoto": round(tanimoto(pred, gold), 3) if valid else 0.0,
    }
