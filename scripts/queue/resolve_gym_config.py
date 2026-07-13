#!/usr/bin/env python3
"""Resolve queue model/benchmark YAML configs for shell runners."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merged(cfg: dict[str, Any], section: str, slug: str) -> dict[str, Any]:
    defaults = dict(cfg.get("defaults") or {})
    values = dict((cfg.get(section) or {}).get(slug) or {})
    if not values:
        raise SystemExit(f"unknown {section[:-1]} slug: {slug}")
    defaults.update(values)
    return defaults


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def emit_value(name: str, value: Any) -> None:
    if value is None:
        value = ""
    if isinstance(value, bool):
        value = "1" if value else "0"
    print(f"{name}={shlex.quote(str(value))}")


def emit_array(name: str, values: Any) -> None:
    values = values or []
    print(f"{name}=({' '.join(shlex.quote(str(v)) for v in values)})")


def emit_env(name: str, env: Any) -> None:
    env = env or {}
    emit_array(name, [f"{k}={v}" for k, v in env.items()])


def get_path(root: dict[str, Any], dotted: str) -> Any:
    cur: Any = root
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def emit_pair(args: argparse.Namespace) -> None:
    models_cfg = load_yaml(args.models_config)
    benchmarks_cfg = load_yaml(args.benchmarks_config)
    model = merged(models_cfg, "models", args.model)
    bench = merged(benchmarks_cfg, "benchmarks", args.benchmark)

    emit_value("MODEL_ID", model.get("model"))
    emit_value("MODEL_TEMPERATURE", model.get("temperature", 0))
    emit_value("MODEL_TOP_P", model.get("top_p", ""))
    emit_value("MODEL_MAX_MODEL_LEN", model.get("max_model_len", ""))
    emit_value("MODEL_MAX_OUTPUT_TOKENS", model.get("max_output_tokens", ""))
    emit_value("MODEL_GPU_MEMORY_UTILIZATION", model.get("gpu_memory_utilization", ""))
    emit_value("MODEL_CONCURRENCY", model.get("concurrency", ""))
    emit_value("MODEL_CUDA_VISIBLE_DEVICES", model.get("cuda_visible_devices", ""))
    emit_value("MODEL_TENSOR_PARALLEL_SIZE", model.get("tensor_parallel_size", ""))
    emit_value("MODEL_PIPELINE_PARALLEL_SIZE", model.get("pipeline_parallel_size", ""))
    emit_value("MODEL_VLLM_WAIT_TIMEOUT", model.get("vllm_wait_timeout", ""))
    emit_value("MODEL_GYM_WAIT_TIMEOUT", model.get("gym_wait_timeout", ""))
    emit_array("MODEL_VLLM_EXTRA_ARGS", model.get("vllm_extra_args"))
    emit_array("MODEL_TOOL_VLLM_EXTRA_ARGS", model.get("tool_vllm_extra_args"))
    emit_env("MODEL_ENV_ASSIGNMENTS", model.get("env"))

    emit_value("BENCH_KIND", bench.get("kind", "benchmark"))
    emit_value("BENCH_TARGET", bench.get("target", ""))
    emit_value("BENCH_CONFIG", bench.get("config", ""))
    emit_value("BENCH_INPUT", bench.get("input", ""))
    emit_value("BENCH_AGENT", bench.get("agent", ""))
    emit_value("BENCH_SPLIT", bench.get("split", ""))
    emit_value("BENCH_PREPARE", bench.get("prepare", False))
    emit_value("BENCH_TOOL_CALLS", bench.get("tool_calls", False))
    emit_value("BENCH_LIMIT", bench.get("limit", ""))
    emit_value("BENCH_NUM_REPEATS", bench.get("num_repeats", ""))
    emit_value("BENCH_MAX_MODEL_LEN", bench.get("max_model_len", ""))
    emit_value("BENCH_MAX_OUTPUT_TOKENS", bench.get("max_output_tokens", ""))
    emit_value("BENCH_VLLM_WAIT_TIMEOUT", bench.get("vllm_wait_timeout", ""))
    emit_value("BENCH_GYM_WAIT_TIMEOUT", bench.get("gym_wait_timeout", ""))
    emit_array("BENCH_GYM_EXTRA_ARGS", bench.get("gym_extra_args"))
    emit_array("BENCH_VLLM_EXTRA_ARGS", bench.get("vllm_extra_args"))
    emit_array("BENCH_REQUIRED_FILES", bench.get("required_files"))


def validate_selection(args: argparse.Namespace) -> None:
    models_cfg = load_yaml(args.models_config)
    benchmarks_cfg = load_yaml(args.benchmarks_config)
    env_cfg: dict[str, Any] = {}
    env_path = Path(args.env_yaml)
    if env_path.exists():
        env_cfg = load_yaml(env_path)

    errors: list[str] = []
    repo_root = Path(args.repo_root).resolve()

    for model_slug in csv_values(args.models):
        try:
            merged(models_cfg, "models", model_slug)
        except SystemExit:
            errors.append(f"unknown model slug: {model_slug}")

    for bench_slug in csv_values(args.benchmarks):
        try:
            bench = merged(benchmarks_cfg, "benchmarks", bench_slug)
        except SystemExit:
            errors.append(f"unknown benchmark slug: {bench_slug}")
            continue

        if not args.skip_credential_check:
            for dotted in bench.get("required_config_paths") or []:
                value = get_path(env_cfg, dotted)
                if value in (None, "", "??", "???"):
                    errors.append(f"{bench_slug}: missing env.yaml value {dotted}")

        for fpath in bench.get("required_files") or []:
            path = Path(fpath)
            if not path.is_absolute():
                path = repo_root / path
            if not path.exists():
                errors.append(f"{bench_slug}: missing required file {fpath}")
            elif path.stat().st_size == 0:
                errors.append(f"{bench_slug}: required file is empty {fpath}")

    if errors:
        print("Config validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit-pair")
    emit.add_argument("--models-config", required=True)
    emit.add_argument("--benchmarks-config", required=True)
    emit.add_argument("--model", required=True)
    emit.add_argument("--benchmark", required=True)
    emit.set_defaults(func=emit_pair)

    validate = subparsers.add_parser("validate-selection")
    validate.add_argument("--models-config", required=True)
    validate.add_argument("--benchmarks-config", required=True)
    validate.add_argument("--models", required=True)
    validate.add_argument("--benchmarks", required=True)
    validate.add_argument("--env-yaml", default="env.yaml")
    validate.add_argument("--repo-root", default=".")
    validate.add_argument("--skip-credential-check", action="store_true")
    validate.set_defaults(func=validate_selection)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
