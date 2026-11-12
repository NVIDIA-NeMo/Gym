# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a Codex catalog from pinned source without changing the checkout or CLI runtime."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path


SOURCE_REVISION = "8c68d4c87dc54d38861f5114e920c3de2efa5876"
SOURCE_URL = "https://github.com/openai/codex"
CODEX_VERSION = "0.144.4"


def sha256(data: bytes) -> str:
    """Hash preparation inputs and outputs for the provenance record."""
    return hashlib.sha256(data).hexdigest()


def check_lock_change(before: bytes, after: bytes, *, local_packages: set[str]) -> None:
    """Allow only the release's workspace-package version reconciliation."""
    expected = tomllib.loads(before.decode())
    for package in expected["package"]:
        if "source" not in package and package["name"] in local_packages and package["version"] == "0.0.0":
            package["version"] = CODEX_VERSION
    if tomllib.loads(after.decode()) != expected:
        raise ValueError("Cargo.lock changed beyond source-local workspace versions; dependency changes are forbidden")


def reconcile_lock(workspace: Path) -> tuple[str, str]:
    """Reconcile the release manifest version while retaining all dependency resolution."""
    manifest = tomllib.loads((workspace / "Cargo.toml").read_text())
    if manifest["workspace"]["package"]["version"] != CODEX_VERSION:
        raise ValueError("Unexpected Codex workspace version")
    local_packages = set()
    for path in workspace.rglob("Cargo.toml"):
        package = tomllib.loads(path.read_text()).get("package", {})
        if package.get("version") == {"workspace": True}:
            local_packages.add(package["name"])
    lock = workspace / "Cargo.lock"
    original = lock.read_bytes()
    blocks = original.decode().split("[[package]]")
    for index in range(1, len(blocks)):
        package = tomllib.loads("[[package]]" + blocks[index])["package"][0]
        if "source" not in package and package["name"] in local_packages and package["version"] == "0.0.0":
            blocks[index] = re.sub(
                r'^version = "0\.0\.0"$', f'version = "{CODEX_VERSION}"', blocks[index], count=1, flags=re.M
            )
    reconciled = "[[package]]".join(blocks).encode()
    check_lock_change(original, reconciled, local_packages=local_packages)
    lock.write_bytes(reconciled)
    return sha256(original), sha256(reconciled)


def check_catalog_export(fallback: bytes, catalog: bytes, *, model: str) -> None:
    """Independently check the export's serialized descriptor against its resolved baseline."""
    baseline = json.loads(fallback)
    matches = [entry for entry in json.loads(catalog)["models"] if entry["slug"] == model]
    if baseline["slug"] != model or len(matches) != 1:
        raise ValueError("Export must contain exactly one descriptor for the resolved fallback alias")
    modified = matches[0]
    for field in ("context_window", "max_context_window", "auto_compact_token_limit"):
        if modified.pop(field, None) is not None:
            raise ValueError(f"Export retained finite {field}")
        baseline.pop(field, None)
    if baseline != modified:
        raise ValueError("Export changed fields other than the three context fields")


def prepare_catalog(
    *, source_checkout: Path, output: Path, model: str, cargo_cache: Path | None = None, allow_downloads: bool = False
) -> Path:
    """Archive pinned source and build the exporter offline with isolated build/cache directories."""
    source_checkout, output = source_checkout.resolve(), output.resolve()
    if not model or output.is_relative_to(source_checkout):
        raise ValueError("Use a nonempty exact model alias and a new output directory outside the source checkout")
    if cargo_cache is not None and (not cargo_cache.is_dir() or output.is_relative_to(cargo_cache.resolve())):
        raise ValueError("Use an existing Cargo cache and keep the output directory outside it")
    output.mkdir(parents=True, exist_ok=False)
    provenance: dict[str, object] = {
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "codex_version": CODEX_VERSION,
        "rust_toolchain": "1.95.0",
        "model": model,
        "status": "preparing",
        "qualification": "Exporter checks only; native CLI behavior and engine context enforcement are unqualified",
        "warning_marker": "Catalog resolution clears internal used_fallback_model_metadata",
    }
    try:
        archive = output / "source.tar"
        with archive.open("wb") as destination:
            subprocess.run(
                ["git", "-C", str(source_checkout), "archive", SOURCE_REVISION], stdout=destination, check=True
            )
        provenance["source_archive_sha256"] = sha256(archive.read_bytes())
        extracted = output / "source"
        extracted.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(extracted, filter="data")
        workspace = extracted / "codex-rs"
        exporter = Path(__file__).with_name("export_model_catalog.rs").read_bytes()
        example = workspace / "models-manager/examples/gym_export_catalog.rs"
        example.parent.mkdir(exist_ok=True)
        example.write_bytes(exporter)
        provenance["exporter_sha256"] = sha256(exporter)
        original_hash, reconciled_hash = reconcile_lock(workspace)
        provenance.update(original_lock_sha256=original_hash, reconciled_lock_sha256=reconciled_hash)
        for name in ("prompt.md", "models.json"):
            provenance[f"bundled_{name}_sha256"] = sha256((workspace / "models-manager" / name).read_bytes())
        cargo_home = output / "cargo-home"
        cargo_home.mkdir()
        if cargo_cache is not None:
            for name in ("registry", "git"):
                if (cargo_cache / name).is_dir():
                    shutil.copytree(cargo_cache / name, cargo_home / name)
        cargo = shutil.which("cargo")
        if cargo is None:
            raise RuntimeError(
                "Cargo is unavailable; source and provenance are prepared, but no catalog was generated"
            )
        env = {
            **os.environ,
            "CARGO_HOME": str(cargo_home),
            "CARGO_TARGET_DIR": str(output / "target"),
            "CARGO_NET_OFFLINE": "true",
            "RUSTUP_AUTO_INSTALL": "0",
            "RUSTUP_TOOLCHAIN": "1.95.0",
        }
        provenance["cargo_version"] = subprocess.check_output(
            [cargo, "--version"], cwd=workspace, env=env, text=True
        ).strip()
        provenance["rustc_version"] = subprocess.check_output(
            ["rustc", "--version", "--verbose"], cwd=workspace, env=env, text=True
        ).strip()
        command = [
            cargo,
            "run",
            "--offline",
            "--locked",
            "--package",
            "codex-models-manager",
            "--example",
            "gym_export_catalog",
            "--",
            model,
            str(output),
        ]
        provenance["command"] = command
        with (output / "build.log").open("wb") as log:
            if allow_downloads:
                fetch = [cargo, "fetch", "--locked"]
                provenance["fetch_command"] = fetch
                subprocess.run(
                    fetch,
                    cwd=workspace,
                    env={**env, "CARGO_NET_OFFLINE": "false"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=900,
                )
            subprocess.run(
                command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900
            )
        if sha256((workspace / "Cargo.lock").read_bytes()) != reconciled_hash:
            raise ValueError("Cargo modified the reconciled lockfile")
        check_catalog_export(
            (output / "fallback.json").read_bytes(), (output / "models.json").read_bytes(), model=model
        )
        for name in ("fallback.json", "models.json"):
            provenance[f"{name}_sha256"] = sha256((output / name).read_bytes())
        provenance["status"] = "exporter_checks_passed"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as error:
        provenance.update(status="failed", error=str(error))
        raise
    finally:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output / "models.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Exact CLI alias, commonly gym-policy-model")
    parser.add_argument("--cargo-cache", type=Path, help="Existing Cargo cache to copy; never modified in place")
    parser.add_argument(
        "--allow-downloads", action="store_true", help="Fetch locked dependencies into the isolated cache"
    )
    print(prepare_catalog(**vars(parser.parse_args())))
