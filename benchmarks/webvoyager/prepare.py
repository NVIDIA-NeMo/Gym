# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the official WebVoyager JSONL into normalized Gym rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
import urllib.request
from pathlib import Path

from nemo_gym.web.datasets import (
    adapt_native_webvoyager_record,
    adapt_webvoyager_record,
    load_json_records,
    write_jsonl,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "webvoyager_benchmark.jsonl"
NATIVE_OUTPUT_FPATH = BENCHMARK_DIR / "data" / "webvoyager_native_v3.jsonl"
DEFAULT_ENV_FPATH = BENCHMARK_DIR / "env.yaml"
DEFAULT_ROLLOUT_FPATH = REPO_ROOT / "results" / "webvoyager" / "rollouts.jsonl"
LEGACY_SOURCE_REVISION = "5a7896738c10bfb8b9edccce6bb0e0411f8ae569"  # pragma: allowlist secret
LEGACY_SOURCE_URL = (
    f"https://raw.githubusercontent.com/MinorJerry/WebVoyager/{LEGACY_SOURCE_REVISION}/data/WebVoyager_data.jsonl"
)
LEGACY_SOURCE_SHA256 = "69b19fd86c23f1a500244a3724e039aa7ca6a1223d03e11eb10e308d4f11c488"  # pragma: allowlist secret
LEGACY_SOURCE_FPATH = BENCHMARK_DIR / "data" / "WebVoyager_data.jsonl"
NATIVE_V3_SOURCE_COMMIT = "6a2977939b157b0ab9de7799bb089c721f1ac115"  # pragma: allowlist secret
NATIVE_V3_SOURCE_URL = (
    f"https://raw.githubusercontent.com/jayl940712/webarena_benchmarks/{NATIVE_V3_SOURCE_COMMIT}/webvoyager.jsonl"
)
NATIVE_V3_SOURCE_SHA256 = (
    "f635a9b27fa1980a63b39bbf64ae8e9e766159cb70fa765451d3d3c0b948ff98"  # pragma: allowlist secret
)
NATIVE_V3_SOURCE_FPATH = BENCHMARK_DIR / "data" / "webvoyager_native_v3_source.jsonl"

PROFILE_CONFIGS = {
    "legacy": (
        BENCHMARK_DIR / "config.yaml",
        REPO_ROOT / "responses_api_models" / "openai_model" / "configs" / "openai_model.yaml",
    ),
    "native_v3": (
        BENCHMARK_DIR / "configs" / "native_v3.yaml",
        REPO_ROOT / "responses_api_models" / "vllm_model" / "configs" / "vllm_model.yaml",
        BENCHMARK_DIR / "configs" / "native_v3_policy.yaml",
    ),
}
PROFILE_AGENTS = {
    "legacy": "webvoyager_benchmark_agent",
    "native_v3": "native_webvoyager_agent",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download_source(*, url: str, sha256: str, destination: Path, label: str) -> Path:
    """Materialize one hash-pinned source inside Gym's ignored data cache."""

    if destination.is_file() and _sha256(destination) == sha256:
        print(f"Using cached {label}: {destination}", flush=True)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} from {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise ValueError(f"{label} hash mismatch: expected {sha256}, got {digest}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            handle.write(payload)
            temporary_path = Path(handle.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _download_legacy_source(destination: Path = LEGACY_SOURCE_FPATH) -> Path:
    """Materialize the pinned official 643-task source."""

    return _download_source(
        url=LEGACY_SOURCE_URL,
        sha256=LEGACY_SOURCE_SHA256,
        destination=destination,
        label="official WebVoyager source",
    )


def _download_native_v3_source(destination: Path = NATIVE_V3_SOURCE_FPATH) -> Path:
    """Materialize the pinned maintained 552-task source."""

    return _download_source(
        url=NATIVE_V3_SOURCE_URL,
        sha256=NATIVE_V3_SOURCE_SHA256,
        destination=destination,
        label="maintained WebVoyager source",
    )


def prepare(source: str | Path | None = None, output: str | Path = OUTPUT_FPATH) -> Path:
    """Prepare the official 643-task BrowserGym-compatible profile.

    ``gym eval prepare --benchmark webvoyager`` calls this function without
    arguments. In that case it downloads an immutable upstream source into the
    benchmark's gitignored data directory. Operators can still provide an
    explicit source path for an already-populated cache.
    """

    configured_source = source or os.environ.get("WEBVOYAGER_SOURCE_JSONL")
    source_path = Path(configured_source).expanduser() if configured_source else _download_legacy_source()
    records = load_json_records(source_path)
    if len(records) != 643:
        raise ValueError(f"BrowserGym-compatible WebVoyager requires exactly 643 tasks, got {len(records)}")
    rows = [adapt_webvoyager_record(record) for record in records]
    count = write_jsonl(rows, output)
    print(f"Wrote {count} WebVoyager tasks to {output}", flush=True)
    return Path(output)


def prepare_native(
    source: str | Path | None = None,
    output: str | Path = NATIVE_OUTPUT_FPATH,
) -> Path:
    """Prepare the pinned maintained population for native runs."""

    configured_source = source or os.environ.get("WEBVOYAGER_SOURCE_JSONL")
    source_path = Path(configured_source).expanduser() if configured_source else _download_native_v3_source()
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != NATIVE_V3_SOURCE_SHA256:
        raise ValueError(f"native WebVoyager source hash mismatch: expected {NATIVE_V3_SOURCE_SHA256}, got {digest}")
    records = load_json_records(source_path)
    rows = [adapt_native_webvoyager_record(record) for record in records]
    if len(rows) != 552:
        raise ValueError(f"native WebVoyager v3 requires exactly 552 tasks, got {len(rows)}")
    count = write_jsonl(rows, output)
    print(f"Wrote {count} native WebVoyager tasks to {output}", flush=True)
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
    """Write a private, gitignored Gym composition for a prepared profile."""

    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"unsupported WebVoyager profile: {profile!r}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if profile == "native_v3" and concurrency != 1:
        raise ValueError(
            "one native WebVoyager resource server owns one X display; use isolated Gym processes for parallelism"
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
    native = profile == "native_v3"
    lines = [
        "# Generated by benchmarks/webvoyager/prepare.py. This file is gitignored.",
        "config_paths:",
        *(f"  - {_yaml_string(path)}" for path in config_paths),
        f"agent_name: {PROFILE_AGENTS[profile]}",
        f"input_jsonl_fpath: {_yaml_string(Path(input_jsonl).expanduser().resolve())}",
        f"output_jsonl_fpath: {_yaml_string(output_path)}",
        "num_repeats: 1",
        f"num_samples_in_parallel: {concurrency}",
        "upload_rollouts: false",
        "responses_create_params:",
        f"  max_output_tokens: {16384 if native else 1000}",
        f"  temperature: {0.1 if native else 1.0}",
        *(["  top_p: 0.95"] if native else []),
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
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="legacy")
    parser.add_argument("--source", type=Path, default=None, help="Source WebVoyager JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Prepared Gym JSONL")
    parser.add_argument("--rollout-output", type=Path, default=DEFAULT_ROLLOUT_FPATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FPATH)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-env", action="store_true", help="Prepare data without writing env.yaml")
    parser.add_argument("--force-env", action="store_true", help="Replace an existing generated env.yaml")
    args = parser.parse_args()

    if args.profile == "native_v3":
        prepared = prepare_native(args.source, args.output or NATIVE_OUTPUT_FPATH)
    else:
        prepared = prepare(args.source, args.output or OUTPUT_FPATH)
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
