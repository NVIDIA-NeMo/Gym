# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan and verify one forced pytest batch using the installed Pants partitioner."""

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path


PANTS_VERSION = "2.30.2"


def single_batch_plan(
    partitions: Sequence[Sequence[str]],
    *,
    pants_version: str,
    hash_prefix: Callable[[str], int],
    partition: Callable[..., Iterable[list[str]]],
) -> dict[str, object]:
    """Choose a supported size; the supplied functions belong to installed Pants.

    We do not copy Pants' hashing or splitting algorithms. Its actual partitioner
    must confirm that the selected size preserves every resolved address once.
    """
    if pants_version != PANTS_VERSION:
        raise ValueError(f"Batching requires verified Pants {PANTS_VERSION}, got {pants_version}")
    if len(partitions) != 1:
        raise ValueError("Forced core tests do not form one compatible Pants partition")
    addresses = sorted(partitions[0])
    if not addresses or len(addresses) != len(set(addresses)):
        raise ValueError("Batch plan requires nonempty, distinct resolved addresses")
    maximum_prefix = max(hash_prefix(address) for address in addresses)
    size = 1
    while size < len(addresses) or size.bit_length() - 1 <= maximum_prefix:
        size *= 2
    # The CLI's native integer parser is bounded; do not approximate overflow.
    if size > (1 << 63) - 1:
        raise ValueError("No representable batch-size bound; inspect this target set")
    batches = list(partition(addresses, key=str, size_target=size, size_max=2 * size))
    if batches != [addresses]:
        raise ValueError("Installed Pants partitioner did not confirm one complete batch")
    return {
        "pants_version": pants_version,
        "batch_size": size,
        "expected_batches": 1,
        "addresses": addresses,
        "maximum_hash_prefix_zeros": maximum_prefix,
        "verification": "installed Pants compatibility and stable partitioners",
    }


def validate_plan(plan: dict[str, object], selected: list[str]) -> int:
    if plan.get("pants_version") != PANTS_VERSION or plan.get("expected_batches") != 1:
        raise ValueError("Unsupported forced-batch plan")
    if plan.get("addresses") != sorted(selected) or len(selected) != len(set(selected)):
        raise ValueError("Batch planner changed the selected native target set")
    size = plan.get("batch_size")
    if type(size) is not int or not 1 <= size <= (1 << 63) - 1:
        raise ValueError("Invalid verified batch size")
    return size


def verify_batch_result(
    checkout: Path, reports: Path, output: Path, expected_addresses: list[str]
) -> dict[str, object]:
    """Retain Pants' actual result provenance, rather than infer batching from time."""
    result_files = list(checkout.glob("test_result_info_report_runid*.json"))
    if len(result_files) != 1:
        raise ValueError("Expected one Pants test result provenance report")
    result = json.loads(result_files[0].read_text())
    info = result.get("info")
    if not isinstance(info, dict) or len(info) != 1 or len(list(reports.rglob("*.xml"))) != 1:
        raise ValueError("Forced core execution did not produce exactly one pytest batch")
    if not expected_addresses or list(info) != [sorted(expected_addresses)[0]]:
        raise ValueError("Actual batch representative differs from the verified selection")
    if info[sorted(expected_addresses)[0]].get("source") != "ran":
        raise ValueError("Forced core batch did not execute; cached provenance is invalid")
    (output / "batch-result-info.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"actual_pytest_batches": 1, "result_info": info}
