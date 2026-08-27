# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the pinned PrimeVul paired test split and shape it into Gym rows."""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Optional


# PrimeVul's upstream release is not hosted on the Hub. This third-party mirror reproduces its
# paired configuration; pinning the commit keeps row-index-derived pair IDs stable.
PRIMEVUL_HF_DATASET = "colin/PrimeVul"
PRIMEVUL_HF_CONFIG = "paired"
PRIMEVUL_HF_REVISION = "4fd7158322872d711e90f091dbd8673ef32cb1be"

HF_SPLITS = {"benchmark": "test"}

# PrimeVul is a C/C++ dataset and upstream does not distinguish the two, so every row is fenced as
# `c`. The fence language is a hint to the model, not a claim about the dialect.
LANG = "c"


def load_pairs(
    split: str,
    *,
    max_pairs: Optional[int] = None,
    seed: int = 0,
    hf_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return the records for `split` (one of `HF_SPLITS`), as whole pairs.

    With `max_pairs` set, pairs are sampled reproducibly from `seed` and the upstream ordering
    is preserved. Without it, the full split is returned.
    """
    if split not in HF_SPLITS:
        raise ValueError(f"split must be one of {sorted(HF_SPLITS)}, got {split!r}")

    from datasets import load_dataset  # heavy import; only needed at preparation time

    dataset = load_dataset(
        PRIMEVUL_HF_DATASET,
        PRIMEVUL_HF_CONFIG,
        split=HF_SPLITS[split],
        revision=PRIMEVUL_HF_REVISION,
        token=hf_token,
    )
    records = [_record(row, split, index) for index, row in enumerate(dataset)]
    _assert_well_paired(records, split)
    return _sample_pairs(records, max_pairs, seed)


def raw_row(record: dict[str, Any]) -> dict[str, Any]:
    """A row with no `responses_create_params`, for use with a `prompt_config`.

    `code` and `lang` are the top-level fields the prompt template fills from.
    """
    return {
        "code": record["code"],
        "lang": record["lang"],
        "verifier_metadata": _verifier_metadata(record),
    }


def _verifier_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "pair_id": record["pair_id"],
        "gold_is_vulnerable": record["gold_is_vulnerable"],
    }


def _record(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    """One upstream row, normalized. `index // 2` groups a pair under one id."""
    return {
        "id": str(row.get("idx", index)),
        "pair_id": f"primevul-{split}-{index // 2}",
        "lang": LANG,
        "code": row.get("func", ""),
        "gold_is_vulnerable": bool(row.get("target")),
    }


def _assert_well_paired(records: list[dict[str, Any]], split: str) -> None:
    """Fail loudly if the consecutive-row pairing does not hold.

    Paired accuracy is the reported metric and it is computed from `pair_id`, so a change in
    upstream ordering has to stop preparation rather than quietly produce a set of pairs that all
    carry the same label.
    """
    for pair_id, members in _by_pair(records).items():
        labels = sorted(member["gold_is_vulnerable"] for member in members)
        if labels != [False, True]:
            raise ValueError(
                f"{PRIMEVUL_HF_DATASET}:{PRIMEVUL_HF_CONFIG} split {split!r}: {pair_id} has labels "
                f"{labels}, expected exactly one vulnerable and one fixed function. The upstream "
                "row ordering the pairing relies on may have changed."
            )


def _by_pair(records: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        grouped.setdefault(record["pair_id"], []).append(record)
    return grouped


def _sample_pairs(records: list[dict[str, Any]], max_pairs: Optional[int], seed: int) -> list[dict[str, Any]]:
    if max_pairs is None:
        return records
    if max_pairs <= 0:
        raise ValueError(f"max_pairs must be positive, got {max_pairs}")
    grouped = _by_pair(records)
    pair_ids = list(grouped)
    if max_pairs >= len(pair_ids):
        return records
    chosen = set(random.Random(seed).sample(pair_ids, max_pairs))
    return [record for pair_id in pair_ids if pair_id in chosen for record in grouped[pair_id]]
