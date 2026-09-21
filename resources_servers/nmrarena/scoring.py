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

"""Deterministic NMRArena scoring: candidate extraction, RDKit canonicalisation, Top-k and Tanimoto.

Everything here follows upstream's two notebooks, as *run* rather than as described.

Extraction reproduces ``_extract_json_candidates`` / ``_coerce_candidate_list`` in
``dataset/llm_track.ipynb``: the last ``"candidates"`` key in the text is located,
the enclosing JSON object is brace-matched and loaded (comments and trailing commas
stripped on a second attempt), and the list is read in ``rank`` order. Upstream's
published per-item lists (``results/combined_predictions_105_final.json``) also
contain, for outputs cut off at the token budget, the complete ``{"rank", "smiles"}``
objects that precede the cut; the notebook's parser returns ``format_fail`` there.
``salvage`` reproduces that recovery and is on by default.

Scoring reproduces ``analysis/data_analysis.ipynb``, which takes the published lists
as they are: the candidate at position *i* has rank *i*; an entry that RDKit cannot
parse is a miss at its position; duplicates occupy positions. A molecule is solved
when a candidate equals the truth after ``Chem.RemoveStereochemistry`` and
canonicalisation. Tanimoto (Morgan radius 2, 2048 bits) is between the position-1
candidate and the truth, and is undefined when position 1 does not parse. Note that
this differs from the notebook's ``parse_candidates``, which drops invalid and
duplicate entries before ranking: on every list upstream published the two agree on
Top-1 and Top-10 and differ on Tanimoto in the third decimal; the published table is
computed the way it is done here.

Two things are added. Every model-controlled string is length-capped *before* RDKit
sees it, because ``MolToSmiles`` recurses per atom and a long enough chain overflows
the C stack with no Python exception (measured at ~15,000 characters; the longest
gold SMILES is 92). And a ``strict`` mode disqualifies a prediction that contains any
invalid, oversize or duplicate entry, so the cost of upstream's positional tolerance
can be measured rather than assumed.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


RDLogger.DisableLog("rdApp.*")

# Longest gold SMILES in ``dataset_selected_clean_105.json`` is 92 characters; the cap
# leaves a wide margin for verbose but legitimate spellings while staying well below
# where RDKit's canonical ranking overflows the stack.
DEFAULT_MAX_SMILES_CHARS = 500

# Upstream ``FP_RADIUS = 2`` (ECFP4-equivalent) and ``FP_SIZE = 2048``.
FP_RADIUS = 2
FP_SIZE = 2048
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)


def sanitize_strings(value: Any) -> Any:
    """Replace lone surrogates in every string of a parsed JSON value so it can be re-encoded."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, dict):
        return {sanitize_strings(k): sanitize_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_strings(v) for v in value]
    return value


# --- canonicalisation -----------------------------------------------------------------


def _clean_smiles(smiles: Any) -> Optional[str]:
    """The parsing notebook's ``strip().strip("`").strip()``; ``None`` for non-strings and empties."""
    if not isinstance(smiles, str):
        return None
    s = smiles.strip().strip("`").strip()
    return s or None


def canonical(smiles: Any, max_chars: int = DEFAULT_MAX_SMILES_CHARS) -> Optional[str]:
    """Canonical SMILES with stereochemistry stripped; ``None`` if not a parseable string within the cap.

    The scoring notebook's form (``RemoveStereochemistry`` then ``MolToSmiles``). The
    parsing notebook uses ``MolToSmiles(m, isomericSmiles=False)`` instead; over every
    candidate string upstream published the two agree.
    """
    s = _clean_smiles(smiles)
    if s is None or len(s) > max_chars:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        # Includes a lone surrogate that the RDKit binding cannot encode.
        return None


def tanimoto(smiles_a: Any, smiles_b: Any, max_chars: int = DEFAULT_MAX_SMILES_CHARS) -> Optional[float]:
    """Upstream ``tanimoto``: Morgan r=2 / 2048-bit Tanimoto; ``None`` if either side is unparseable."""
    a, b = _clean_smiles(smiles_a), _clean_smiles(smiles_b)
    if a is None or b is None or len(a) > max_chars or len(b) > max_chars:
        return None
    try:
        ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    except Exception:
        return None
    if ma is None or mb is None:
        return None
    return DataStructs.TanimotoSimilarity(_MORGAN.GetFingerprint(ma), _MORGAN.GetFingerprint(mb))


# --- candidate extraction (upstream ``_extract_json_candidates``) ---------------------------


def _clean_json(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)  # /* ... */ comments
    s = re.sub(r"//[^\n]*", "", s)  # // line comments
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas before } or ]
    return s


def _try_load(s: str) -> Any:
    for cand in (s, _clean_json(s)):
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None


def _rank_key(pair):
    i, it = pair
    rank = it.get("rank") if isinstance(it, dict) else None
    if isinstance(rank, (int, float)) and not isinstance(rank, bool):
        return (0, rank, i)
    return (1, 0, i)


def _coerce_candidate_list(obj: Any) -> Optional[list[Any]]:
    """Parsed JSON -> rank-ordered list of raw ``smiles`` values, or ``None``.

    Upstream keeps ``str`` items and the ``smiles``/``SMILES`` value of ``dict`` items
    when that value is a string, and silently drops everything else. Dropped members
    are kept here as ``None`` so that they occupy their position and strict mode can
    count them; they parse as invalid either way.
    """
    items = None
    if isinstance(obj, dict):
        if isinstance(obj.get("candidates"), list):
            items = obj["candidates"]
        elif "smiles" in obj or "SMILES" in obj:
            items = [obj]
    elif isinstance(obj, list):
        items = obj
    if not items:
        return None
    out: list[Any] = []
    for _, it in sorted(enumerate(items), key=_rank_key):
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            sm = it.get("smiles") or it.get("SMILES")
            out.append(sm if isinstance(sm, str) else None)
        else:
            out.append(None)
    return out


def _match_braces(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_raw_candidates(text: str) -> Optional[list[Any]]:
    """Upstream ``_extract_json_candidates``: the object around the *last* ``"candidates"`` key."""
    idx = text.rfind('"candidates"')
    if idx == -1:
        return None
    ob = text.rfind("{", 0, idx)
    if ob == -1:
        return None
    cb = _match_braces(text, ob)
    if cb == -1:
        return None
    obj = _try_load(text[ob : cb + 1])
    if obj is None:
        return None
    return _coerce_candidate_list(obj)


# One complete candidate object. Only the shape the prompt asks for is recovered.
_CANDIDATE_OBJECT_RE = re.compile(r'\{\s*"rank"\s*:\s*(\d+)\s*,\s*"smiles"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')


def salvage_truncated_candidates(text: str) -> Optional[list[Any]]:
    """Complete ``{"rank": n, "smiles": "..."}`` objects after the last ``"candidates"`` key.

    Reproduces what upstream's published lists contain for outputs cut off at the
    token budget: the notebook's brace matcher fails on the unterminated object, but
    the per-item results carry the candidates that were completed before the cut.
    Returns ``None`` when there is no ``"candidates"`` key or no complete object after it.
    """
    idx = text.rfind('"candidates"')
    if idx == -1:
        return None
    matches = _CANDIDATE_OBJECT_RE.findall(text, idx)
    if not matches:
        return None
    objects = []
    for rank, smiles in matches:
        try:
            decoded = json.loads(f'"{smiles}"')
        except ValueError:
            # A SMILES with a stereo bond (``C/C=C\\C``) is an invalid JSON escape; keep the
            # text as written rather than lose the candidate or raise.
            decoded = smiles
        objects.append({"rank": int(rank), "smiles": decoded})
    return _coerce_candidate_list(objects)


@dataclass
class ParsedCandidates:
    """What came out of the model's text, before comparison with the truth."""

    # Canonical SMILES (or ``None`` for an entry that does not parse), one per position,
    # at most ``n``. Empty when no JSON was found or the prediction was disqualified.
    candidates: list[Optional[str]] = field(default_factory=list)
    # ``None`` when no ``"candidates"`` JSON could be located (upstream ``format_fail``).
    n_raw: Optional[int] = None
    n_invalid: int = 0
    n_oversize: int = 0
    n_duplicate: int = 0
    # True when the list came from complete objects in an unterminated JSON block.
    salvaged: bool = False
    # Strict mode only: the reason the whole prediction was disqualified.
    disqualified: Optional[str] = None

    @property
    def found_json(self) -> bool:
        return self.n_raw is not None

    @property
    def n_valid(self) -> int:
        return sum(c is not None for c in self.candidates)


def parse_candidates(
    text: str,
    n: int = 10,
    max_smiles_chars: int = DEFAULT_MAX_SMILES_CHARS,
    strict: bool = False,
    salvage: bool = True,
) -> ParsedCandidates:
    """Extract the ranked list and canonicalise it position by position.

    The first ``n`` entries are kept (upstream's lists never exceed ten, so the cap is
    not exercised by upstream data). Lenient mode (upstream): an invalid, oversize or
    duplicate entry is a miss at its position. Strict mode: any such entry
    disqualifies the prediction and no candidates are returned.
    """
    text = text or ""
    raw = extract_raw_candidates(text)
    salvaged = False
    if not raw and salvage:
        raw = salvage_truncated_candidates(text)
        salvaged = raw is not None
    if not raw:
        return ParsedCandidates()
    # ``json.loads`` turns the wire-safe escape ``\udcff`` into a lone surrogate that
    # neither RDKit nor the response encoder accepts; normalise the parsed values.
    raw = sanitize_strings(raw)
    parsed = ParsedCandidates(n_raw=len(raw), salvaged=salvaged)
    seen: set[str] = set()
    for item in raw[:n]:
        cleaned = _clean_smiles(item)
        if cleaned is not None and len(cleaned) > max_smiles_chars:
            parsed.n_oversize += 1
            parsed.candidates.append(None)
            if strict:
                parsed.disqualified = "oversize"
                break
            continue
        c = canonical(item, max_chars=max_smiles_chars)
        if c is None:
            parsed.n_invalid += 1
            parsed.candidates.append(None)
            if strict:
                parsed.disqualified = "invalid"
                break
            continue
        if c in seen:
            parsed.n_duplicate += 1
            if strict:
                parsed.disqualified = "duplicate"
                break
        seen.add(c)
        parsed.candidates.append(c)
    if strict and parsed.disqualified is None and len(raw) > n:
        parsed.disqualified = "too_many"
    if parsed.disqualified is not None:
        parsed.candidates = []
    return parsed


# --- metrics (upstream ``analysis/data_analysis.ipynb``) -------------------------------------


def hit_rank(truth_canonical: Optional[str], candidates: list[Optional[str]]) -> Optional[int]:
    """1-based position of the first candidate equal to the truth; ``None`` on a miss.

    ``candidates`` are already canonical (or ``None``), so equality is the structural
    match upstream computes by canonicalising both sides.
    """
    if truth_canonical is None:
        return None
    for i, c in enumerate(candidates):
        if c is not None and c == truth_canonical:
            return i + 1
    return None


@dataclass
class Scores:
    hit_rank: Optional[int]
    top1: float
    top10: float
    # Upstream's Tanimoto is defined only when the position-1 candidate parses; ``None`` otherwise.
    tanimoto_top1: Optional[float]
    # 1.0 when at least one candidate parses.
    answered: float


def score_candidates(truth_canonical: str, candidates: list[Optional[str]], k: int = 10) -> Scores:
    rank = hit_rank(truth_canonical, candidates)
    first = candidates[0] if candidates else None
    return Scores(
        hit_rank=rank,
        top1=1.0 if rank == 1 else 0.0,
        top10=1.0 if rank is not None and rank <= k else 0.0,
        tanimoto_top1=tanimoto(truth_canonical, first) if first is not None else None,
        answered=1.0 if any(c is not None for c in candidates) else 0.0,
    )
