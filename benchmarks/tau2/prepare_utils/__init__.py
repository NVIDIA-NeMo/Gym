# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare helpers for Tau2/Tau3 Gym benchmark rows."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BENCHMARK_DIR / "data"
NEMO_GYM_DATA_DIR = BENCHMARK_DIR / "nemo_gym_data"

TAU2_DOMAINS = ("airline", "retail", "telecom")
TAU2_OUTPUT_FPATH = DATA_DIR / "tau2_benchmark.jsonl"

BANKING_RETRIEVAL_CONFIGS = (
    "no_knowledge",
    "full_kb",
    "golden_retrieval",
    "qwen_embeddings_grep",
    "openai_embeddings_grep",
    "qwen_embeddings_reranker_grep",
    "openai_embeddings_reranker_grep",
    "bm25_grep",
    "bm25_reranker_grep",
    "qwen_embeddings",
    "openai_embeddings",
    "qwen_embeddings_reranker",
    "openai_embeddings_reranker",
    "bm25",
    "bm25_reranker",
    "grep_only",
    "terminal_use",
    "terminal_use_write",
    "alltools",
    "alltools-qwen",
)

TAU2_BENCH_DATA_REPO_URL_ENV = "NEMO_GYM_TAU2_BENCH_DATA_REPO_URL"
TAU2_BENCH_DATA_REF_ENV = "NEMO_GYM_TAU2_BENCH_DATA_REF"

DEFAULT_TAU2_BENCH_DATA_REPO_URL = "https://github.com/bxyu-nvidia/tau2-bench"
DEFAULT_TAU2_BENCH_DATA_REF = "jk/bxyu-nemo-gym-data-upstream-main-tau3"


def get_tau2_bench_data_repo_url() -> str:
    return os.environ.get(
        TAU2_BENCH_DATA_REPO_URL_ENV,
        DEFAULT_TAU2_BENCH_DATA_REPO_URL,
    )


def get_tau2_bench_data_ref() -> str:
    return os.environ.get(TAU2_BENCH_DATA_REF_ENV, DEFAULT_TAU2_BENCH_DATA_REF)


def banking_dump_dir(retrieval_config: str) -> str:
    return f"banking_knowledge_{retrieval_config}"


def banking_dataset_name(retrieval_config: str) -> str:
    return f"banking_{retrieval_config}"


def banking_output_path(retrieval_config: str) -> Path:
    return DATA_DIR / f"tau2_banking_knowledge_{retrieval_config}.jsonl"


def ensure_nemo_gym_data_dir(
    required_dirs: Sequence[str],
    datasets: Sequence[str],
) -> Path:
    """Ensure local ``nemo_gym_data`` contains the requested dumped datasets."""

    missing_dirs = [dirname for dirname in required_dirs if not (NEMO_GYM_DATA_DIR / dirname).exists()]
    if not missing_dirs:
        return NEMO_GYM_DATA_DIR

    repo_url = get_tau2_bench_data_repo_url()
    ref = get_tau2_bench_data_ref()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="tau2-bench-data-", dir=BENCHMARK_DIR) as tmp_dir:
        checkout_dir = Path(tmp_dir) / "tau2-bench"
        run(["git", "clone", repo_url, str(checkout_dir)], check=True)
        run(["git", "-C", str(checkout_dir), "checkout", ref], check=True)

        dump_script = checkout_dir / "dump_nemo_gym_data.sh"
        if not dump_script.exists():
            raise RuntimeError(f"Tau2 data dump script not found at {dump_script}")

        command = ["bash", str(dump_script)]
        for dataset in datasets:
            command.extend(["--dataset", dataset])
        run(command, cwd=checkout_dir, check=True)

        source_data_dir = checkout_dir / "nemo_gym_data"
        if not source_data_dir.exists():
            raise RuntimeError(f"nemo_gym_data not generated at {source_data_dir}")

        NEMO_GYM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for dirname in required_dirs:
            source_dir = source_data_dir / dirname
            if not source_dir.exists():
                raise RuntimeError(f"Requested Tau2 data dump {dirname!r} missing from {repo_url}@{ref}")
            target_dir = NEMO_GYM_DATA_DIR / dirname
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)

    return NEMO_GYM_DATA_DIR


def strip_nl_assertions(task_dict: dict) -> None:
    reward_basis = task_dict.get("evaluation_criteria", {}).get("reward_basis")
    if isinstance(reward_basis, list):
        task_dict["evaluation_criteria"]["reward_basis"] = [item for item in reward_basis if item != "NL_ASSERTION"]


def normalize_row(row: dict) -> dict:
    row["config"]["save_to"] = ""
    row["evaluation_type"] = "all"
    row["config"].get("llm_args_user", {}).pop("temperature", None)
    strip_nl_assertions(row["task"])

    responses_create_params = row.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        raise ValueError("Tau2 row is missing responses_create_params")
    if "input" not in responses_create_params or "tools" not in responses_create_params:
        raise ValueError("Tau2 row responses_create_params must include input and tools")

    return row


def load_dump_rows(
    dump_dirs: Sequence[str],
    datasets: Sequence[str],
) -> list[dict]:
    data_dir = ensure_nemo_gym_data_dir(dump_dirs, datasets)
    rows = []
    for dump_dir in dump_dirs:
        paths = sorted((data_dir / dump_dir).glob("*.json"))
        if not paths:
            raise RuntimeError(f"No Tau2 dump rows found in {data_dir / dump_dir}")
        for path in paths:
            rows.append(normalize_row(json.loads(path.read_text())))
    return rows


def write_rows(output_fpath: Path, rows: Iterable[dict], label: str) -> Path:
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_fpath.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} {label} problems to {output_fpath}")
    return output_fpath


def prepare_tau2() -> Path:
    rows = load_dump_rows(TAU2_DOMAINS, TAU2_DOMAINS)
    return write_rows(TAU2_OUTPUT_FPATH, rows, "Tau2 benchmark")


def prepare_banking_knowledge(retrieval_config: str) -> Path:
    rows = load_dump_rows(
        [banking_dump_dir(retrieval_config)],
        [banking_dataset_name(retrieval_config)],
    )
    return write_rows(
        banking_output_path(retrieval_config),
        rows,
        f"banking_knowledge/{retrieval_config}",
    )


def prepare_all_banking_knowledge() -> dict[str, Path]:
    dump_dirs = [banking_dump_dir(config) for config in BANKING_RETRIEVAL_CONFIGS]
    datasets = [banking_dataset_name(config) for config in BANKING_RETRIEVAL_CONFIGS]
    load_dump_rows(dump_dirs, datasets)
    return {
        retrieval_config: prepare_banking_knowledge(retrieval_config) for retrieval_config in BANKING_RETRIEVAL_CONFIGS
    }
