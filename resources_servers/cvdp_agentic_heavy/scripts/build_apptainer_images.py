#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prebuild Apptainer SIFs for CVDP tool and verifier containers.

The CVDP resource server can lazily pull/build SIFs at runtime, but production
rollout runs are easier to operate when images are prepared up front. This
script mirrors the resource server's image resolution rules:

- Tool container: the OSS simulator image used for visible iverilog/vvp calls.
- Verifier containers: every image referenced by converted CVDP harness compose
  services, including simple Docker images and Dockerfile-derived images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources_servers.cvdp_agentic_heavy.app import (  # noqa: E402
    CVDPAgenticHeavyConfig,
    _apply_substitutions,
    _filter_code_volumes,
    _resolve_image_for_service,
)


COMMERCIAL_EDA_MARKERS = ("__VERIF_EDA_IMAGE__", "xrun", "xcelium", "licnetwork")


def _make_config(args: argparse.Namespace) -> CVDPAgenticHeavyConfig:
    return CVDPAgenticHeavyConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="cvdp_agentic_heavy",
        oss_sim_image=args.oss_sim_image,
        oss_pnr_image=args.oss_pnr_image,
        sif_cache_dir=args.sif_cache_dir,
    )


def _safe_sif_name(image: str) -> str:
    return image.replace("/", "_").replace(":", "_") + ".sif"


def _built_sif_name(base_image: str, post_commands: tuple[str, ...]) -> str:
    cmd_hash = hashlib.md5("\n".join(post_commands).encode()).hexdigest()[:12]
    return base_image.replace("/", "_").replace(":", "_") + f"__built_{cmd_hash}.sif"


def _sif_path(cache_dir: Path, image: str, post_commands: tuple[str, ...] = ()) -> Path:
    name = _built_sif_name(image, post_commands) if post_commands else _safe_sif_name(image)
    return cache_dir / name


def _add_target(
    targets: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
    image: str,
    *,
    kind: str,
    source: str,
    post_commands: tuple[str, ...] = (),
) -> None:
    if not image:
        return
    key = (image, post_commands)
    target = targets.setdefault(key, {"kinds": set(), "sources": []})
    target["kinds"].add(kind)
    target["sources"].append(source)


def _compose_contains_commercial_eda(compose_content: str) -> bool:
    text = compose_content.lower()
    return any(marker.lower() in text for marker in COMMERCIAL_EDA_MARKERS)


def _iter_rows(path: Path):
    with path.open(encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue
            yield line_num, json.loads(line)


def discover_verifier_targets(
    jsonl_paths: list[Path],
    config: CVDPAgenticHeavyConfig,
) -> tuple[dict[tuple[str, tuple[str, ...]], dict[str, Any]], list[str]]:
    targets: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    warnings: list[str] = []

    for jsonl_path in jsonl_paths:
        for line_num, row in _iter_rows(jsonl_path):
            meta = row.get("verifier_metadata") or {}
            task_id = meta.get("task_id") or f"{jsonl_path}:{line_num}"
            harness_files = dict(meta.get("harness_files") or {})
            compose_key = next((key for key in harness_files if key.endswith("docker-compose.yml")), "")
            if not compose_key:
                warnings.append(f"{task_id}: no docker-compose.yml in harness_files")
                continue

            compose_content = harness_files.get(compose_key) or ""
            compose_content = _filter_code_volumes(_apply_substitutions(compose_content, config))
            if _compose_contains_commercial_eda(compose_content):
                warnings.append(f"{task_id}: commercial EDA/Xcelium compose service remains; skipping image discovery")
                continue

            try:
                compose_data = yaml.safe_load(compose_content) or {}
            except Exception as exc:
                warnings.append(f"{task_id}: could not parse docker-compose.yml: {exc}")
                continue

            services = compose_data.get("services") or {}
            if not isinstance(services, dict):
                warnings.append(f"{task_id}: docker-compose.yml has no service map")
                continue

            for service_name in services:
                image, post_commands = _resolve_image_for_service(compose_data, service_name, harness_files, config)
                if not image:
                    warnings.append(f"{task_id}:{service_name}: no image or supported Dockerfile base image found")
                    continue
                _add_target(
                    targets,
                    image,
                    kind="verifier",
                    source=f"{task_id}:{service_name}",
                    post_commands=tuple(post_commands),
                )

    return targets, warnings


def _print_plan(targets: dict[tuple[str, tuple[str, ...]], dict[str, Any]], cache_dir: Path) -> None:
    print(f"sif_cache_dir={cache_dir}")
    print(f"image_targets={len(targets)}")
    for index, ((image, post_commands), data) in enumerate(sorted(targets.items()), 1):
        kinds = ",".join(sorted(data["kinds"]))
        sif = _sif_path(cache_dir, image, post_commands)
        sources = data["sources"][:5]
        more = f" (+{len(data['sources']) - len(sources)} more)" if len(data["sources"]) > len(sources) else ""
        derived = "yes" if post_commands else "no"
        print(f"[{index}] kinds={kinds} image={image} derived={derived} sif={sif}")
        print(f"    sources={'; '.join(sources)}{more}")
        if post_commands:
            print("    post_commands:")
            for cmd in post_commands:
                print(f"      - {cmd}")


def _run(cmd: list[str], *, timeout: int = 0) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout or None)


def ensure_sif(image: str, cache_dir: Path, *, force: bool, dry_run: bool, timeout: int) -> Path:
    sif_path = _sif_path(cache_dir, image)
    if sif_path.exists() and not force:
        print(f"exists {sif_path}")
        return sif_path
    tmp_path = sif_path.with_suffix(sif_path.suffix + ".pulling")
    cmd = ["apptainer", "pull", "--force", str(tmp_path), f"docker://{image}"]
    if dry_run:
        print("dry-run + " + " ".join(cmd))
        return sif_path
    _run(cmd, timeout=timeout)
    tmp_path.replace(sif_path)
    return sif_path


def ensure_built_sif(
    base_image: str,
    post_commands: tuple[str, ...],
    cache_dir: Path,
    *,
    force: bool,
    dry_run: bool,
    timeout: int,
) -> Path:
    if not post_commands:
        return ensure_sif(base_image, cache_dir, force=force, dry_run=dry_run, timeout=timeout)

    sif_path = _sif_path(cache_dir, base_image, post_commands)
    if sif_path.exists() and not force:
        print(f"exists {sif_path}")
        return sif_path

    base_sif = ensure_sif(base_image, cache_dir, force=force, dry_run=dry_run, timeout=timeout)
    post_section = "\n    ".join(post_commands)
    def_content = f"Bootstrap: localimage\nFrom: {base_sif}\n\n%post\n    {post_section}\n"
    tmp_def = sif_path.with_suffix(sif_path.suffix + ".def")
    tmp_sif = sif_path.with_suffix(sif_path.suffix + ".building")
    cmd = ["apptainer", "build", "--force", str(tmp_sif), str(tmp_def)]

    if dry_run:
        print(f"dry-run write {tmp_def}")
        print(def_content.rstrip())
        print("dry-run + " + " ".join(cmd))
        return sif_path

    tmp_def.write_text(def_content, encoding="utf-8")
    try:
        _run(cmd, timeout=timeout)
        tmp_sif.replace(sif_path)
    finally:
        tmp_def.unlink(missing_ok=True)
        if tmp_sif.exists() and not sif_path.exists():
            tmp_sif.unlink()
    return sif_path


def check_targets(targets: dict[tuple[str, tuple[str, ...]], dict[str, Any]], cache_dir: Path) -> int:
    missing = []
    for image, post_commands in targets:
        path = _sif_path(cache_dir, image, post_commands)
        if path.exists():
            print(f"present {path}")
        else:
            print(f"missing {path}")
            missing.append(path)
    if missing:
        print(f"missing_count={len(missing)}")
        return 1
    print("all_sifs_present=yes")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--converted-jsonl",
        action="append",
        default=[],
        help="Converted CVDP Gym JSONL. Pass multiple times to discover verifier images across files.",
    )
    parser.add_argument(
        "--sif-cache-dir",
        default=os.environ.get("CVDP_SIF_CACHE_DIR", str(Path.home() / ".cache" / "nemo-gym" / "sif")),
        help="Directory where SIFs are written. Defaults to CVDP_SIF_CACHE_DIR or ~/.cache/nemo-gym/sif.",
    )
    parser.add_argument("--oss-sim-image", default="ghcr.io/hdl/sim/osvb", help="Tool container image for iverilog/vvp.")
    parser.add_argument("--oss-pnr-image", default="ghcr.io/hdl/impl/pnr", help="Default PNR image placeholder.")
    parser.add_argument("--skip-tool-container", action="store_true", help="Do not prebuild the visible tool container SIF.")
    parser.add_argument(
        "--include-default-pnr-image",
        action="store_true",
        help="Also prebuild the default OSS PNR image even if no converted row references it.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the image plan and commands without running Apptainer.")
    parser.add_argument("--check", action="store_true", help="Only check whether planned SIF files already exist.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing SIF files.")
    parser.add_argument("--timeout", type=int, default=0, help="Optional timeout in seconds for each Apptainer command.")
    args = parser.parse_args()

    cache_dir = Path(args.sif_cache_dir).expanduser().resolve()
    config = _make_config(args)
    targets: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    if not args.skip_tool_container:
        _add_target(targets, args.oss_sim_image, kind="tool", source="visible iverilog/vvp tools")
    if args.include_default_pnr_image:
        _add_target(targets, args.oss_pnr_image, kind="verifier", source="default OSS PNR image")

    verifier_targets, warnings = discover_verifier_targets([Path(p) for p in args.converted_jsonl], config)
    for key, data in verifier_targets.items():
        image, post_commands = key
        for source in data["sources"]:
            _add_target(targets, image, kind="verifier", source=source, post_commands=post_commands)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not targets:
        print("No image targets found. Provide --converted-jsonl or leave tool prebuild enabled.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    _print_plan(targets, cache_dir)

    if args.check:
        raise SystemExit(check_targets(targets, cache_dir))
    if args.dry_run:
        for image, post_commands in sorted(targets):
            ensure_built_sif(image, post_commands, cache_dir, force=args.force, dry_run=True, timeout=args.timeout)
        print("dry_run=complete")
        return
    if shutil.which("apptainer") is None:
        print("ERROR: apptainer was not found on PATH. Install Apptainer or rerun with --dry-run/--check.", file=sys.stderr)
        raise SystemExit(2)

    for image, post_commands in sorted(targets):
        ensure_built_sif(image, post_commands, cache_dir, force=args.force, dry_run=False, timeout=args.timeout)

    print("apptainer_prebuild=complete")


if __name__ == "__main__":
    main()
