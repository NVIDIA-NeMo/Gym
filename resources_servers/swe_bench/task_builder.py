# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build ``SweTask`` instances from Gym dataset / verifier metadata."""

from __future__ import annotations

import json
from typing import Any

from resources_servers.swe_bench.harness import SweTask


_BENCHMARK_KEYS: list[tuple[str, str]] = [
    ("R2E-Gym", "r2e-gym"),
    ("SWE-bench_Multilingual", "swe-bench-multilingual"),
    ("SWE-bench", "swe-bench"),
]


def benchmark_key(dataset_name: str) -> str:
    for needle, key in _BENCHMARK_KEYS:
        if needle in dataset_name:
            return key
    return "swe-bench"


def instance_image(container_formatter: Any, instance_id: str) -> str:
    fmt = container_formatter[0] if isinstance(container_formatter, list) else container_formatter
    fmt = fmt or "swebench/sweb.eval.x86_64.{instance_id}"
    if fmt.endswith(".sif") or fmt.startswith(("/", ".")):
        return fmt.format(instance_id=instance_id)
    if fmt.startswith("docker://"):
        fmt = fmt[len("docker://") :]
    tag = instance_id.replace("__", "_1776_").lower()
    image = fmt.format(instance_id=tag)
    if ":" not in image.rsplit("/", 1)[-1]:
        image += ":latest"
    return image


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            return list(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return [value] if value else []
    return list(value or [])


def problem_info_from_row(
    verifier_metadata: dict[str, Any] | None, responses_metadata: dict[str, Any] | None
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if responses_metadata:
        info.update(responses_metadata)
    if verifier_metadata:
        info.update(verifier_metadata)
    return info


def build_swetask(problem_info: dict[str, Any], *, container_formatter: str, flat_eval: bool = True) -> SweTask:
    inst = (
        json.loads(problem_info["instance_dict"])
        if isinstance(problem_info.get("instance_dict"), str)
        else dict(problem_info.get("instance_dict", {}))
    )
    benchmark = benchmark_key(problem_info.get("dataset_name", ""))
    instance_id = problem_info["instance_id"]
    image = instance_image(problem_info.get("container_formatter") or container_formatter, instance_id)

    return SweTask(
        instance_id=instance_id,
        image=image,
        base_commit=inst.get("base_commit"),
        repo_workdir="/testbed",
        test_patch=inst.get("test_patch", ""),
        fail_to_pass=_as_list(inst.get("FAIL_TO_PASS") or inst.get("fail_to_pass")),
        pass_to_pass=_as_list(inst.get("PASS_TO_PASS") or inst.get("pass_to_pass")),
        benchmark=benchmark,
        split=problem_info.get("split", "test"),
        metadata={"instance_dict": inst, "flat_eval": flat_eval, "dataset_name": problem_info.get("dataset_name", "")},
    )
