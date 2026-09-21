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

"""Prepare the NMRArena 105-molecule benchmark for NeMo Gym.

The dataset is one JSON file in the upstream GitHub repository
(odanchem/NMRArena, default branch ``release``; ``main`` does not exist). Upstream
has no tags, releases or DOI, and both the results table and the spectra strings
have been edited in place since the repository appeared, so the file is fetched at a
pinned commit and its SHA-256 is checked before anything is written. The benchmark
data is MIT-licensed but is not redistributed here; it is downloaded at run time.

Each output row is upstream's request: the vendored system prompt, the notebook's
user prompt built from the normalised 1H and 13C peak strings, and the notebook's
decoding settings (``temperature`` 1.0, ``max_output_tokens`` 24576). Preparation
fails closed: the digest, the row count, the class structure, the uniqueness of the
compound ids and the parseability of every gold SMILES are all checked, and the
output is written only after every row has been built. ``--limit`` is the one
supported way to prepare a deliberate subset.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting import MAX_OUTPUT_TOKENS, NUM_CANDIDATES, TEMPERATURE, build_messages  # noqa: E402
from scoring import canonical  # noqa: E402


GITHUB_REPO = "odanchem/NMRArena"
# Head of ``release`` on 2026-09-22 ("Updated statistics in the README", 2026-09-17). The
# dataset file is byte-identical to the one at cdb6883dd70633fb548a1830b2591fae6721250b,
# the pin recorded in the Surveyor availability audit of 2026-09-16.
GITHUB_COMMIT = "8b4ca8a8953185c00f0c4d7fa3c16c23aa616326"  # pragma: allowlist secret
DATASET_PATH = "dataset/dataset_selected_clean_105.json"
DATASET_SHA256 = "6c8aed1adeecfb80278d47b4b524de54246b2739f9c568e3f2f2c7d7f7259108"  # pragma: allowlist secret
DATASET_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/{DATASET_PATH}"

EXPECTED_ROWS = 105
EXPECTED_CLASSES = 21
EXPECTED_PER_CLASS = 5

AGENT_REF = {"type": "responses_api_agents", "name": "nmrarena_simple_agent"}


class CorpusError(RuntimeError):
    """The fetched corpus is not what the pinned revision is known to contain."""


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def load_dataset(http_get: Callable[[str], bytes]) -> dict:
    """Fetch the dataset at the pin and verify its digest before parsing it."""
    raw = http_get(DATASET_URL)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATASET_SHA256:
        raise CorpusError(f"{DATASET_PATH} at {GITHUB_COMMIT[:12]} has sha256 {digest}; expected {DATASET_SHA256}")
    return json.loads(raw.decode("utf-8"))


def flatten(dataset: dict) -> list[tuple[str, dict]]:
    """Upstream ``load_records``: ``{class: {key: record}}`` -> ``[(class, record)]`` in file order."""
    out = []
    for cls, group in dataset.items():
        if isinstance(group, dict):
            for rec in group.values():
                if isinstance(rec, dict) and "smiles" in rec:
                    out.append((cls, rec))
    return out


def validate_records(dataset: dict, records: list[tuple[str, dict]]) -> None:
    """Fail closed on anything that would silently change a denominator or a truth."""
    if len(records) != EXPECTED_ROWS:
        raise CorpusError(f"loaded {len(records)} records; expected {EXPECTED_ROWS}")
    classes = {cls for cls, _ in records}
    if len(classes) != EXPECTED_CLASSES:
        raise CorpusError(f"found {len(classes)} classes; expected {EXPECTED_CLASSES}")
    for cls in sorted(classes):
        n = sum(1 for c, _ in records if c == cls)
        if n != EXPECTED_PER_CLASS:
            raise CorpusError(f"class {cls} has {n} records; expected {EXPECTED_PER_CLASS}")
    ids = [rec.get("compound_id") for _, rec in records]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise CorpusError("compound_id values are not unique non-empty strings")
    for cls, rec in records:
        for key in ("h_nmr", "c_nmr"):
            if not isinstance(rec.get(key), str) or not rec[key].strip():
                raise CorpusError(f"{rec.get('compound_id')} ({cls}) has no {key}")
        if canonical(rec.get("smiles")) is None:
            raise CorpusError(f"{rec.get('compound_id')} ({cls}) gold smiles does not parse: {rec.get('smiles')!r}")


def format_row(cls: str, rec: dict, num_candidates: int) -> dict:
    """One Gym row: upstream's two-turn request with its decoding, and what ``verify`` needs."""
    return {
        "responses_create_params": {
            "input": build_messages(rec["h_nmr"], rec["c_nmr"], num_candidates),
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "verifier_metadata": {
            "compound_id": rec["compound_id"],
            "smiles": rec["smiles"],
            "primary_class": cls,
            "n_complex": rec.get("n_complex"),
            # Provenance, never read by ``verify``.
            "publication_id": rec.get("publication_id"),
            "doi": rec.get("doi"),
            "dataset_commit": GITHUB_COMMIT,
            "dataset_sha256": DATASET_SHA256,
        },
        "agent_ref": AGENT_REF,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare NMRArena (105 molecules) for NeMo Gym")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--num-candidates",
        type=_positive_int,
        default=NUM_CANDIDATES,
        help="n in the prompt; upstream asks for 10 and scores Top-10",
    )
    parser.add_argument("--limit", type=_positive_int, default=None, help="Prepare only the first N rows")
    return parser


def main(argv: Optional[list[str]] = None, http_get: Optional[Callable[[str], bytes]] = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = load_dataset(http_get or _http_get)
    records = flatten(dataset)
    validate_records(dataset, records)
    if args.limit is not None:
        records = records[: args.limit]
    rows = [format_row(cls, rec, args.num_candidates) for cls, rec in records]
    if len(rows) != len(records):
        raise CorpusError(f"built {len(rows)} of {len(records)} rows")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output_path} (dataset {GITHUB_COMMIT[:12]})", file=sys.stderr)


# python resources_servers/nmrarena/scripts/prepare_nmrarena.py --output resources_servers/nmrarena/data/nmrarena_105.jsonl
if __name__ == "__main__":
    main()
