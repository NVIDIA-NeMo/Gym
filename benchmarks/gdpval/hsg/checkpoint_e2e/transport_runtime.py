#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the pinned PR #2588 GDPVal transport runtime per campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REVISION = "d3f146d386c7dfe07d4fabce32c4c8b14c7917d2"
SCHEMA = "gdpval.transport-runtime.v3"
BASE_HASHES = {
    "pyproject.toml": "3897f79b5d66e69fb119570cb4c268c3b85c38510e2e5caba0d8309fd952d25a",
    "resources_servers/gdpval/requirements.txt": "eed0001155f4d85df6501d42852aa85c5b9f2549fb5cd38effa6267a0b1d6506",
    "resources_servers/gdpval/__init__.py": "c4dc0cf54a15db963e684ed996a7ceb38a3737390327929f05930ac903f33514",
    "resources_servers/gdpval/app.py": "2eb146d0cdab8e03a6e5b2b42d04d63c47419e83a462a638aa142ac2ffe6d909",
    "resources_servers/gdpval/comparison.py": "f6bc837ec0d3ae82bde3f1b8a10f34f0edd58078976d198b0d619718581b750a",
    "resources_servers/gdpval/judge_panel.py": "2622ff6c655775900271a338c7430923ca2660b5fcbc85c216c107717e600225",
    "resources_servers/gdpval/media_conversion.py": "282c10b5d7c858ff798ee3aae9b6e0bdd87be277b412006e13a9bd0c37fd1b64",
    "resources_servers/gdpval/multistage_elo.py": "79b5290c2a9c2e767e629df88b106c8a80853bedd164a23223d15011fe2eecc1",
    "resources_servers/gdpval/multistage_orchestrator.py": "c5862484e7a371bece91aa8871d6f816fa65f1a934b97b8d834fb104579afd80",
    "responses_api_models/openai_model/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "responses_api_models/openai_model/app.py": "f11c08b3bc1b52b93b8a3d1fa7d36e59eae2e7cc2e252f086e1f1487d19c078c",
    "responses_api_models/openai_model/requirements.txt": "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d",
}

COMPONENT_REQUIREMENTS = (
    "resources_servers/gdpval/requirements.txt",
    "responses_api_models/openai_model/requirements.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(gym_root: Path) -> str:
    marker = gym_root / ".checkpoint_e2e_revision"
    if marker.is_file() and not marker.is_symlink():
        revision = marker.read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"invalid staged Gym revision marker: {marker}")
        return revision
    return subprocess.check_output(["git", "-C", str(gym_root), "rev-parse", "HEAD"], text=True).strip()


def _validate_base(gym_root: Path) -> None:
    if _revision(gym_root) != REVISION:
        raise ValueError(f"Gym revision must be exact {REVISION}")
    for relative, expected in BASE_HASHES.items():
        path = gym_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"pinned Gym source drift: {path}")


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o400)
    os.replace(temporary, path)


def _output_hashes(runtime_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(runtime_root.rglob("*")):
        if path.is_file() and path.name != "runtime_manifest.json":
            result[path.relative_to(runtime_root).as_posix()] = _sha256(path)
    return result


def _validate_component_resolution(gym_root: Path, runtime_root: Path) -> None:
    """Prove Gym will launch both patched components from the runtime overlay."""

    python = gym_root / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)
    script = """
import sys
from pathlib import Path

from nemo_gym.cli.env import _resolve_server_dir

runtime_root = Path(sys.argv[1]).resolve()
for relative in (
    "resources_servers/gdpval",
    "responses_api_models/openai_model",
):
    expected = (runtime_root / relative).resolve()
    resolved = _resolve_server_dir(Path(relative)).resolve()
    if resolved != expected:
        raise SystemExit(f"component resolver selected {resolved}, expected {expected}")
"""
    environment = {
        **os.environ,
        "NEMO_GYM_EXTRA_ROOTS": str(runtime_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(runtime_root), str(gym_root))),
    }
    completed = subprocess.run(
        [str(python), "-c", script, str(runtime_root)],
        cwd=gym_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ValueError(f"runtime component resolution failed: {detail}")


def validate(gym_root: Path, runtime_root: Path) -> dict[str, Any]:
    gym_root = gym_root.resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=True)
    _validate_base(gym_root)
    manifest_path = runtime_root / "runtime_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"runtime manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("revision") != REVISION:
        raise ValueError("runtime manifest contract mismatch")
    if Path(manifest["gym_root"]) != gym_root:
        raise ValueError("runtime manifest points at a different Gym root")
    actual = _output_hashes(runtime_root)
    if actual != manifest.get("output_sha256"):
        raise ValueError("materialized GDPVal runtime drift")
    expected = f"-e nemo-gym[dev] @ {gym_root.as_uri()}"
    for relative in COMPONENT_REQUIREMENTS:
        requirements = runtime_root / relative
        lines = requirements.read_text().splitlines()
        if not lines or lines[0] != expected:
            raise ValueError(f"component requirements do not pin the exact Gym root: {relative}")
    _validate_component_resolution(gym_root, runtime_root)
    return manifest


def materialize(gym_root: Path, runtime_root: Path, package_root: Path) -> dict[str, Any]:
    gym_root = gym_root.resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=False)
    package_root = package_root.resolve(strict=True)
    _validate_base(gym_root)
    if (runtime_root / "runtime_manifest.json").is_file():
        return validate(gym_root, runtime_root)
    if runtime_root.exists():
        raise ValueError(f"incomplete runtime root already exists: {runtime_root}")
    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{runtime_root.name}.", dir=runtime_root.parent))
    try:
        for relative in BASE_HASHES:
            source = gym_root / relative
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        for relative in COMPONENT_REQUIREMENTS:
            requirements = temporary / relative
            lines = requirements.read_text().splitlines()
            if not lines or lines[0] != "-e nemo-gym[dev] @ ../../":
                raise ValueError(f"unexpected component requirements header: {relative}")
            lines[0] = f"-e nemo-gym[dev] @ {gym_root.as_uri()}"
            requirements.write_text("\n".join(lines) + "\n")

        runtime_sources = package_root / "runtime_sources"
        patch_path = runtime_sources / "pr2588_true3_transport.patch"
        image_cap_patch = runtime_sources / "provider_image_caps.patch"
        aggregate_media_cap_patch = runtime_sources / "provider_aggregate_media_caps.patch"
        recursive_reference_patch = runtime_sources / "recursive_reference_assets.patch"
        strict_comparison_trials_patch = runtime_sources / "strict_comparison_trials.patch"
        provider_context_fallback_patch = runtime_sources / "provider_context_fallback.patch"
        provider_rate_limit_backoff_patch = runtime_sources / "provider_rate_limit_backoff.patch"
        partial_pdf_overflow_patch = runtime_sources / "partial_pdf_overflow.patch"
        gemini_pdf_part_cap_patch = runtime_sources / "gemini_pdf_part_cap.patch"
        assignment_source = runtime_sources / "transport_assignment.py"
        # ``git apply`` has deterministic unified-diff matching across the HSG
        # hosts, while GNU patch can reject an otherwise exact hunk after prior
        # hunks shift the same file. It also works in this non-repository
        # materialization directory.
        subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=temporary, check=True)
        subprocess.run(["git", "apply", str(patch_path)], cwd=temporary, check=True)
        subprocess.run(["git", "apply", "--check", str(image_cap_patch)], cwd=temporary, check=True)
        subprocess.run(["git", "apply", str(image_cap_patch)], cwd=temporary, check=True)
        subprocess.run(
            ["git", "apply", "--check", str(aggregate_media_cap_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(aggregate_media_cap_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--check", str(recursive_reference_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(recursive_reference_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--check", str(strict_comparison_trials_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(strict_comparison_trials_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--check", str(provider_context_fallback_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(provider_context_fallback_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--check", str(provider_rate_limit_backoff_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(provider_rate_limit_backoff_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--check", str(partial_pdf_overflow_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(partial_pdf_overflow_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--check", str(gemini_pdf_part_cap_patch)],
            cwd=temporary,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(gemini_pdf_part_cap_patch)],
            cwd=temporary,
            check=True,
        )
        shutil.copyfile(
            assignment_source,
            temporary / "resources_servers/gdpval/transport_assignment.py",
        )
        for path in temporary.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o400)
        manifest = {
            "schema": SCHEMA,
            "revision": REVISION,
            "gym_root": str(gym_root),
            "base_sha256": BASE_HASHES,
            "patch_sha256": _sha256(patch_path),
            "provider_image_caps_patch_sha256": _sha256(image_cap_patch),
            "provider_aggregate_media_caps_patch_sha256": _sha256(aggregate_media_cap_patch),
            "recursive_reference_assets_patch_sha256": _sha256(recursive_reference_patch),
            "strict_comparison_trials_patch_sha256": _sha256(strict_comparison_trials_patch),
            "provider_context_fallback_patch_sha256": _sha256(provider_context_fallback_patch),
            "provider_rate_limit_backoff_patch_sha256": _sha256(provider_rate_limit_backoff_patch),
            "partial_pdf_overflow_patch_sha256": _sha256(partial_pdf_overflow_patch),
            "gemini_pdf_part_cap_patch_sha256": _sha256(gemini_pdf_part_cap_patch),
            "assignment_source_sha256": _sha256(assignment_source),
            "output_sha256": _output_hashes(temporary),
        }
        _atomic_json(temporary / "runtime_manifest.json", manifest)
        os.replace(temporary, runtime_root)
        temporary = None
        return validate(gym_root, runtime_root)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("materialize", "validate"))
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.action == "materialize":
        manifest = materialize(args.gym_root, args.runtime_root, args.package_root)
    else:
        manifest = validate(args.gym_root, args.runtime_root)
    print(f"TRANSPORT_RUNTIME_PASS revision={manifest['revision']} root={args.runtime_root}")


if __name__ == "__main__":
    main()
