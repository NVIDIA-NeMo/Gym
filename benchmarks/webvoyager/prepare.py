# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the maintained 552-task WebVoyager population for visual browsers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
import urllib.request
from pathlib import Path

from nemo_gym.web.datasets import adapt_webvoyager_record, load_json_records, write_jsonl


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "webvoyager.jsonl"
DEFAULT_ENV_FPATH = BENCHMARK_DIR / "env.yaml"
DEFAULT_ROLLOUT_FPATH = REPO_ROOT / "results" / "webvoyager" / "rollouts.jsonl"
SOURCE_COMMIT = "6a2977939b157b0ab9de7799bb089c721f1ac115"  # pragma: allowlist secret
SOURCE_URL = f"https://raw.githubusercontent.com/jayl940712/webarena_benchmarks/{SOURCE_COMMIT}/webvoyager.jsonl"
SOURCE_SHA256 = "f635a9b27fa1980a63b39bbf64ae8e9e766159cb70fa765451d3d3c0b948ff98"  # pragma: allowlist secret
SOURCE_FPATH = BENCHMARK_DIR / "data" / "webvoyager_source.jsonl"

PROFILE_CONFIGS = {
    "nano_omni": (
        BENCHMARK_DIR / "configs" / "nano_omni.yaml",
        REPO_ROOT / "responses_api_models" / "vllm_model" / "configs" / "vllm_model.yaml",
        BENCHMARK_DIR / "configs" / "nano_omni_policy.yaml",
    ),
    "qwen35_122b_a10b": (BENCHMARK_DIR / "configs" / "qwen35_122b_a10b.yaml",),
}
PROFILE_AGENTS = {
    "nano_omni": "nano_omni_webvoyager_agent",
    "qwen35_122b_a10b": "qwen35_webvoyager_agent",
}
PROFILE_SAMPLING = {
    "nano_omni": {"max_output_tokens": 16384, "temperature": 0.1, "top_p": 0.95},
    "qwen35_122b_a10b": {"max_output_tokens": 32768, "temperature": 0.1, "top_p": 0.9},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download_source(destination: Path = SOURCE_FPATH) -> Path:
    """Materialize the immutable maintained source in Gym's ignored cache."""

    if destination.is_file() and _sha256(destination) == SOURCE_SHA256:
        print(f"Using cached WebVoyager source: {destination}", flush=True)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading WebVoyager source from {SOURCE_URL}", flush=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"WebVoyager source hash mismatch: expected {SOURCE_SHA256}, got {digest}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary_path = Path(handle.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def prepare(source: str | Path | None = None, output: str | Path = OUTPUT_FPATH) -> Path:
    """Prepare one model-independent, hash-pinned 552-task dataset.

    ``gym eval prepare --benchmark webvoyager`` calls this function without
    arguments. Nano Omni and Qwen select different policy adapters and serving
    profiles at runtime but consume these exact same task rows.
    """

    configured_source = source or os.environ.get("WEBVOYAGER_SOURCE_JSONL")
    source_path = Path(configured_source).expanduser() if configured_source else _download_source()
    digest = _sha256(source_path)
    if digest != SOURCE_SHA256:
        raise ValueError(f"WebVoyager source hash mismatch: expected {SOURCE_SHA256}, got {digest}")
    rows = [adapt_webvoyager_record(record) for record in load_json_records(source_path)]
    if len(rows) != 552:
        raise ValueError(f"maintained WebVoyager requires exactly 552 tasks, got {len(rows)}")
    count = write_jsonl(rows, output)
    print(f"Wrote {count} WebVoyager tasks to {output}", flush=True)
    return Path(output)


def _yaml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def write_env(
    env_path: str | Path,
    *,
    profile: str,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    concurrency: int = 1,
    force: bool = False,
) -> bool:
    """Write a private, gitignored Gym composition for one policy profile."""

    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"unsupported WebVoyager profile: {profile!r}")
    if concurrency != 1:
        raise ValueError(
            "one headed visual-browser process owns one DISPLAY; shard across isolated Gym processes instead"
        )
    config_paths = tuple(path.resolve() for path in PROFILE_CONFIGS[profile])
    missing = [path for path in config_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"WebVoyager Gym config does not exist: {missing[0]}")

    env_path = Path(env_path).expanduser().resolve()
    if env_path.exists() and not force:
        print(f"Keeping existing configuration: {env_path}")
        return False
    env_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampling = PROFILE_SAMPLING[profile]
    lines = [
        "# Generated by benchmarks/webvoyager/prepare.py. This file is gitignored.",
        "config_paths:",
        *(f"  - {_yaml_string(path)}" for path in config_paths),
        f"agent_name: {PROFILE_AGENTS[profile]}",
        f"input_jsonl_fpath: {_yaml_string(Path(input_jsonl).expanduser().resolve())}",
        f"output_jsonl_fpath: {_yaml_string(output_path)}",
        "num_repeats: 1",
        "num_samples_in_parallel: 1",
        "upload_rollouts: false",
        "responses_create_params:",
        f"  max_output_tokens: {sampling['max_output_tokens']}",
        f"  temperature: {sampling['temperature']}",
        f"  top_p: {sampling['top_p']}",
        "policy_base_url: ${oc.env:POLICY_BASE_URL,http://127.0.0.1:8000/v1}",
        "policy_api_key: ${oc.env:POLICY_API_KEY,local-vllm}",
        "policy_model_name: ${oc.env:POLICY_MODEL_NAME,webvoyager-policy}",
        "webvoyager_judge_base_url: ${oc.env:WEBARENA_JUDGE_BASE_URL,https://inference-api.nvidia.com/v1}",
        "webvoyager_judge_api_key: ${oc.env:WEBARENA_JUDGE_API_KEY,unset}",
        "webvoyager_judge_model_name: ${oc.env:WEBARENA_JUDGE_MODEL,gcp/google/gemini-3-flash-preview}",
        "",
    ]
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if force else os.O_EXCL)
    descriptor = os.open(env_path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    os.chmod(env_path, 0o600)
    print(f"Wrote private configuration: {env_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="nano_omni")
    parser.add_argument("--source", type=Path, default=None, help="Source WebVoyager JSONL")
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH, help="Prepared Gym JSONL")
    parser.add_argument("--rollout-output", type=Path, default=DEFAULT_ROLLOUT_FPATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FPATH)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-env", action="store_true", help="Prepare data without writing env.yaml")
    parser.add_argument("--force-env", action="store_true", help="Replace an existing generated env.yaml")
    args = parser.parse_args()

    prepared = prepare(args.source, args.output)
    if not args.no_env:
        write_env(
            args.env_file,
            profile=args.profile,
            input_jsonl=prepared,
            output_jsonl=args.rollout_output,
            concurrency=args.concurrency,
            force=args.force_env,
        )

    print("\nNext steps:")
    print(f"  cd {_yaml_string(args.env_file.expanduser().resolve().parent)}")
    gym_cli = shlex.quote(str(REPO_ROOT / ".venv" / "bin" / "gym"))
    print(f"  {gym_cli} env prefetch")
    print(f"  {gym_cli} env start")
    print(f"  {gym_cli} eval run --no-serve")


if __name__ == "__main__":
    main()
