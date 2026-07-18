# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prebuild OCI images for CVDP Compose ``build:`` services."""

from __future__ import annotations

import argparse
import json
import posixpath
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from resources_servers.cvdp.app import CVDPResourcesServerConfig
from resources_servers.cvdp.testbench_runner import (
    _apply_substitutions,
    _safe_workspace_path,
    _service_build_key,
)


def _rows(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def _build_paths(build: Any) -> tuple[str, str]:
    if isinstance(build, str):
        return build, "Dockerfile"
    build = build or {}
    return str(build.get("context") or "."), str(build.get("dockerfile") or "Dockerfile")


def _write_context(root: Path, harness_files: Dict[str, Optional[str]], config: CVDPResourcesServerConfig) -> None:
    for relative, content in harness_files.items():
        if content is None:
            continue
        destination = _safe_workspace_path(root, relative)
        if destination is None:
            raise ValueError(f"unsafe CVDP build-context path: {relative!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_apply_substitutions(content, config), encoding="utf-8")


def prepare_images(
    inputs: list[Path],
    image_prefix: str,
    manifest_path: Path,
    config: CVDPResourcesServerConfig,
    *,
    push: bool,
    force: bool,
) -> Dict[str, str]:
    existing: Dict[str, str] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = {str(key): str(value) for key, value in payload.get("images", payload).items()}

    prepared = dict(existing)
    processed: set[str] = set()
    for row in _rows(inputs):
        harness_files = (row.get("verifier_metadata") or {}).get("harness_files") or {}
        compose_text = harness_files.get("docker-compose.yml")
        if not compose_text:
            continue
        compose_text = _apply_substitutions(compose_text, config)
        compose = yaml.safe_load(compose_text) or {}
        for service_name, service in (compose.get("services") or {}).items():
            build = (service or {}).get("build")
            if not build:
                continue
            key = _service_build_key(compose, service_name, harness_files, config)
            image = f"{image_prefix.rstrip(':')}:{key}"
            if key in processed:
                continue
            if key in prepared and not force:
                continue

            context_relative, dockerfile_relative = _build_paths(build)
            with tempfile.TemporaryDirectory(prefix=f"cvdp-image-{key}-") as temporary:
                root = Path(temporary)
                _write_context(root, harness_files, config)
                normalized_context = posixpath.normpath(context_relative)
                context = root if normalized_context == "." else _safe_workspace_path(root, normalized_context)
                dockerfile_path = posixpath.join(normalized_context, dockerfile_relative)
                dockerfile = _safe_workspace_path(root, dockerfile_path)
                if context is None or dockerfile is None or not dockerfile.is_file():
                    raise ValueError(
                        f"service {service_name!r} has an unavailable build context or Dockerfile: "
                        f"context={context_relative!r}, dockerfile={dockerfile_relative!r}"
                    )
                subprocess.run(
                    ["docker", "build", "-f", str(dockerfile), "-t", image, str(context)],
                    check=True,
                )
                if push:
                    subprocess.run(["docker", "push", image], check=True)
            prepared[key] = image
            processed.add(key)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"images": dict(sorted(prepared.items()))}, indent=2) + "\n", encoding="utf-8")
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Converted CVDP JSONL dataset(s)")
    parser.add_argument("--image-prefix", required=True, help="Registry/repository for prepared verifier images")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("resources_servers/cvdp/data/prepared_images.json"),
    )
    parser.add_argument("--oss-sim-image", default="ghcr.io/hdl/sim/osvb")
    parser.add_argument("--oss-pnr-image", default="ghcr.io/hdl/impl/pnr")
    parser.add_argument("--eda-sim-image", default="")
    parser.add_argument("--push", action="store_true", help="Push built images to the configured repository")
    parser.add_argument("--force", action="store_true", help="Rebuild keys already present in the manifest")
    args = parser.parse_args()

    config = CVDPResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        name="cvdp",
        entrypoint="app.py",
        oss_sim_image=args.oss_sim_image,
        oss_pnr_image=args.oss_pnr_image,
        eda_sim_image=args.eda_sim_image,
    )
    images = prepare_images(
        args.inputs,
        args.image_prefix,
        args.manifest,
        config,
        push=args.push,
        force=args.force,
    )
    print(f"Prepared {len(images)} image recipe(s). Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
