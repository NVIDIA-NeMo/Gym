# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the maintained 908-task VisualWebArena population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from nemo_gym import CACHE_DIR
from nemo_gym.web.datasets import adapt_visualwebarena_records, load_json_records, write_jsonl


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "visualwebarena.jsonl"
DEFAULT_ENV_FPATH = BENCHMARK_DIR / "env.yaml"
DEFAULT_ROLLOUT_FPATH = REPO_ROOT / "results" / "visualwebarena" / "rollouts.jsonl"
EXPECTED_TASKS = 908
SOURCE_COMMIT = "6a2977939b157b0ab9de7799bb089c721f1ac115"  # pragma: allowlist secret
SOURCE_SHA256 = "923a4ec5a2a306d497a0a2f0d267db2c47b40b57c6be1965de0b19dd5041e04a"  # pragma: allowlist secret
SOURCE_REPOSITORY = "jayl940712/webarena_benchmarks"
SOURCE_ARCHIVE_URL = f"https://github.com/{SOURCE_REPOSITORY}/archive/{SOURCE_COMMIT}.tar.gz"
DEFAULT_SOURCE_ROOT = CACHE_DIR / "webarena_benchmarks" / SOURCE_COMMIT
SOURCE_JSONL_NAME = "visualwebarena.jsonl"


def _source_records(source_path: Path, image_root: Path) -> list[dict]:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"VisualWebArena source hash mismatch: expected {SOURCE_SHA256}, got {digest}")
    records = load_json_records(source_path)
    if len(records) != EXPECTED_TASKS:
        raise ValueError(f"maintained VisualWebArena requires exactly {EXPECTED_TASKS} tasks, got {len(records)}")

    root = image_root.expanduser().resolve()
    missing: list[str] = []
    for record in records:
        image_value = record.get("image") or record.get("images") or []
        references = [image_value] if isinstance(image_value, str) else image_value
        for reference_value in references:
            reference = str(reference_value).strip()
            if not reference or urlparse(reference).scheme:
                continue
            candidate = (root / reference).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"VisualWebArena image path escapes source root: {reference!r}") from exc
            if not candidate.is_file():
                missing.append(reference)
    if missing:
        examples = ", ".join(repr(path) for path in missing[:3])
        raise FileNotFoundError(
            f"VisualWebArena source is missing {len(missing)} referenced image(s) below {root}; examples: {examples}"
        )
    return records


def _download_pinned_source(source_root: Path) -> Path:
    """Download and atomically cache the public JSONL plus reference images."""

    source_root = source_root.expanduser().resolve()
    source_path = source_root / SOURCE_JSONL_NAME
    if source_root.exists():
        if not source_root.is_dir():
            raise RuntimeError(f"VisualWebArena source root is not a directory: {source_root}")
        try:
            _source_records(source_path, source_root)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"VisualWebArena cache is incomplete or invalid: {source_root}. "
                "Remove that benchmark cache directory or select a fresh VISUALWEBARENA_SOURCE_ROOT."
            ) from exc
        print(f"Using cached VisualWebArena source: {source_root}", flush=True)
        return source_path

    source_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="visualwebarena-source-", dir=source_root.parent) as temp_value:
            temp_dir = Path(temp_value)
            archive_path = temp_dir / "source.tar.gz"
            extracted_dir = temp_dir / "extracted"
            extracted_dir.mkdir()
            print(f"Downloading pinned VisualWebArena source from {SOURCE_ARCHIVE_URL}", flush=True)
            with urllib.request.urlopen(SOURCE_ARCHIVE_URL, timeout=120) as response:  # noqa: S310
                with archive_path.open("wb") as archive_handle:
                    shutil.copyfileobj(response, archive_handle, length=1024 * 1024)
            with tarfile.open(archive_path, mode="r:gz") as archive:
                archive.extractall(extracted_dir, filter="data")

            candidates = [path for path in extracted_dir.iterdir() if path.is_dir()]
            if len(candidates) != 1:
                raise RuntimeError(f"expected one source directory in the pinned archive, found {len(candidates)}")
            extracted_root = candidates[0]
            _source_records(extracted_root / SOURCE_JSONL_NAME, extracted_root)
            try:
                extracted_root.replace(source_root)
            except OSError:
                if not source_root.is_dir():
                    raise
                _source_records(source_path, source_root)
    except (OSError, tarfile.TarError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"could not download pinned VisualWebArena data from {SOURCE_ARCHIVE_URL}; "
            f"set VISUALWEBARENA_SOURCE_ROOT to a local checkout of {SOURCE_REPOSITORY}@{SOURCE_COMMIT}"
        ) from exc

    print(f"Cached VisualWebArena source: {source_root}", flush=True)
    return source_path


def prepare(
    source: str | Path | None = None,
    output: str | Path = OUTPUT_FPATH,
    source_root: str | Path | None = None,
) -> Path:
    """Validate and adapt one hash-pinned, model-independent task population."""

    source_value = str(source or os.environ.get("VISUALWEBARENA_SOURCE_JSONL", "")).strip()
    configured_root = str(os.environ.get("VISUALWEBARENA_SOURCE_ROOT", "")).strip()
    if source_value:
        source_path = Path(source_value).expanduser().resolve()
        image_root = Path(configured_root).expanduser().resolve() if configured_root else source_path.parent
    else:
        image_root = Path(configured_root or source_root or DEFAULT_SOURCE_ROOT).expanduser().resolve()
        source_path = _download_pinned_source(image_root)

    records = _source_records(source_path, image_root)
    count = write_jsonl(adapt_visualwebarena_records(records), output)
    print(f"Wrote {count} VisualWebArena tasks to {output}\nTask image root: {image_root}", flush=True)
    return Path(output)


def _yaml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def write_env(
    env_path: str | Path,
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    source_root: str | Path,
    concurrency: int = 1,
    force: bool = False,
) -> bool:
    """Write a private, gitignored Nano Omni composition for VisualWebArena."""

    if concurrency != 1:
        raise ValueError(
            "one headed visual-browser process owns one DISPLAY; shard across isolated Gym processes instead"
        )
    config_path = (BENCHMARK_DIR / "configs" / "nano_omni.yaml").resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"VisualWebArena Gym config does not exist: {config_path}")

    env_path = Path(env_path).expanduser().resolve()
    if env_path.exists() and not force:
        print(f"Keeping existing configuration: {env_path}")
        return False
    env_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by benchmarks/visualwebarena/prepare.py. This file is gitignored.",
        "config_paths:",
        f"  - {_yaml_string(config_path)}",
        "agent_name: visualwebarena_benchmark_agent",
        f"input_jsonl_fpath: {_yaml_string(Path(input_jsonl).expanduser().resolve())}",
        f"output_jsonl_fpath: {_yaml_string(output_path)}",
        "num_repeats: 1",
        "num_samples_in_parallel: 1",
        "upload_rollouts: false",
        f"visualwebarena_source_root: {_yaml_string(Path(source_root).expanduser().resolve())}",
        "responses_create_params:",
        "  max_output_tokens: 16384",
        "  temperature: 0.1",
        "  top_p: 0.95",
        "policy_base_url: ${oc.env:POLICY_BASE_URL,http://127.0.0.1:8000/v1}",
        "policy_api_key: ${oc.env:POLICY_API_KEY,local-vllm}",
        "policy_model_name: ${oc.env:POLICY_MODEL_NAME,visualwebarena-policy}",
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
    parser.add_argument("--source", type=Path, default=None, help="Source VisualWebArena JSONL")
    parser.add_argument("--source-root", type=Path, default=None, help="Source checkout and task-image root")
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH, help="Prepared Gym JSONL")
    parser.add_argument("--rollout-output", type=Path, default=DEFAULT_ROLLOUT_FPATH)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FPATH)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-env", action="store_true", help="Prepare data without writing env.yaml")
    parser.add_argument("--force-env", action="store_true", help="Replace an existing generated env.yaml")
    args = parser.parse_args()

    configured_root = str(os.environ.get("VISUALWEBARENA_SOURCE_ROOT", "")).strip()
    source_root = Path(configured_root or args.source_root or DEFAULT_SOURCE_ROOT).expanduser().resolve()
    prepared = prepare(args.source, args.output, source_root)
    if not args.no_env:
        write_env(
            args.env_file,
            input_jsonl=prepared,
            output_jsonl=args.rollout_output,
            source_root=source_root,
            concurrency=args.concurrency,
            force=args.force_env,
        )


if __name__ == "__main__":
    main()
