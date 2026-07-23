# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework-free helpers for the SGLang model server."""

from typing import Any, Dict, List, Tuple


def extract_generated_tokens_and_logprobs(
    result: Dict[str, Any],
) -> Tuple[List[int], List[float]]:
    """Return aligned generated token IDs and logprobs from ``/generate``.

    SGLang releases have emitted the selected-token data as dictionaries,
    tuples, or parallel value/index arrays. Missing or malformed data is a
    hard error because silently returning empty arrays would invalidate the
    training loss mask.
    """
    meta = result.get("meta_info") or {}
    if "output_token_logprobs" in meta:
        entries = meta["output_token_logprobs"]
        if not isinstance(entries, (list, tuple)):
            raise RuntimeError(
                f"Malformed SGLang output_token_logprobs field: expected an array, got {type(entries).__name__}"
            )
        if not entries:
            return [], []
        token_ids: List[int] = []
        logprobs: List[float] = []
        for entry in entries:
            if isinstance(entry, dict):
                token_id = entry.get("token_id", entry.get("id"))
                logprob = entry.get("logprob")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                logprob, token_id = entry[0], entry[1]
            else:
                raise RuntimeError(f"Malformed SGLang output_token_logprobs entry: {entry!r}")
            if token_id is None or logprob is None:
                raise RuntimeError(f"Malformed SGLang output_token_logprobs entry: {entry!r}")
            token_ids.append(int(token_id))
            logprobs.append(float(logprob))
        return token_ids, logprobs

    values_present = "output_token_logprobs_val" in meta or "output_token_logprobs_val" in result
    values = meta.get(
        "output_token_logprobs_val",
        result.get("output_token_logprobs_val"),
    )
    indexes = meta.get("output_token_logprobs_idx")
    if indexes is None:
        indexes = result.get("output_token_logprobs_idx")
    output_ids = meta.get("output_ids")
    if output_ids is None:
        output_ids = result.get("output_ids")

    if values_present:
        if not isinstance(values, (list, tuple)):
            raise RuntimeError(
                f"Malformed SGLang output_token_logprobs_val field: expected an array, got {type(values).__name__}"
            )
        selected_ids = indexes if indexes else output_ids
        if not values:
            if selected_ids:
                raise RuntimeError(
                    f"SGLang returned mismatched generation fields: {len(selected_ids)} ids for 0 logprobs"
                )
            return [], []
        if not selected_ids or len(selected_ids) != len(values):
            id_count = len(selected_ids) if selected_ids is not None else 0
            raise RuntimeError(
                f"SGLang returned mismatched generation fields: {id_count} ids for {len(values)} logprobs"
            )
        return [int(token_id) for token_id in selected_ids], [float(logprob) for logprob in values]

    result_keys = sorted(result)
    meta_keys = sorted(meta)
    raise RuntimeError(
        "SGLang /generate returned no generated-token logprobs "
        f"(result keys={result_keys}, meta_info keys={meta_keys}). "
        "Ensure return_logprob=true is supported and honored."
    )
