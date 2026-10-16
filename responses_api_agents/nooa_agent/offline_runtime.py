# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a source-bound wheelhouse outside a network-restricted task sandbox."""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from responses_api_agents.nooa_agent.sandbox_runtime import (
    _PYTHON_SHA256,
    _PYTHON_VERSION,
    _python_download_url,
    _runtime_archive,
)


_TARGETS = {"x86_64/gnu": "amd64", "aarch64/gnu": "arm64"}


def file_sha256(path: Path) -> str:
    """Hash an artifact without loading large wheel archives into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_runtime_wheelhouse(output: Path, *, target: str, source_fingerprint: str) -> dict[str, object]:
    """Validate the exact source, platform and artifact bytes before staging."""
    if target not in _TARGETS:
        raise ValueError(f"Offline NOOA runtime supports only {', '.join(_TARGETS)}; got {target!r}")
    try:
        manifest = json.loads((output / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Prepare the NOOA runtime wheelhouse first: {output}") from error
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != expected
        for key, expected in {
            "version": 1,
            "target": target,
            "source_fingerprint": source_fingerprint,
            "python_version": _PYTHON_VERSION,
            "python_sha256": _PYTHON_SHA256[target],
        }.items()
    ):
        raise ValueError(
            f"NOOA wheelhouse is incompatible with this source/platform; prepare a fresh directory: {output}"
        )
    glibc = manifest.get("glibc")
    if not isinstance(glibc, str) or len(glibc.split(".")) != 2 or not all(p.isdigit() for p in glibc.split(".")):
        raise ValueError("NOOA wheelhouse manifest requires its builder glibc version")
    for name, field in (("python.tar.gz", "python_sha256"), ("wheels.tar.gz", "wheels_sha256")):
        try:
            actual = file_sha256(output / name)
        except OSError as error:
            raise ValueError(f"NOOA wheelhouse is missing {name}: {output}") from error
        if actual != manifest.get(field):
            raise ValueError(f"NOOA wheelhouse checksum mismatch: {output / name}")
    with tarfile.open(output / "wheels.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        if not members or any(
            not m.isfile() or Path(m.name).name != m.name or not m.name.endswith(".whl") for m in members
        ):
            raise ValueError("NOOA wheelhouse must contain only top-level wheel files")
    return manifest


def validate_runtime_wheelhouse(
    output: Path, *, target: str, requirements_path: Path | None = None
) -> dict[str, object]:
    """Check an existing prepared bundle against the current checkout."""
    _, fingerprint = _runtime_archive(Path(__file__).resolve().parents[2], requirements_path=requirements_path)
    return load_runtime_wheelhouse(output, target=target, source_fingerprint=fingerprint)


async def _command(*arguments: str, timeout: int = 1800) -> str:
    process = await asyncio.create_subprocess_exec(
        *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    output = stdout.decode(errors="replace")
    if process.returncode:
        raise RuntimeError(
            f"Offline NOOA preparation failed ({arguments[0]}, exit {process.returncode}):\n{output[-12000:]}"
        )
    return output


async def prepare_runtime_wheelhouse(
    output: Path,
    *,
    target: str,
    requirements_path: Path | None = None,
    builder_image: str | None = None,
) -> Path:
    """Build every runtime wheel on compatible Linux, using Docker outside the task.

    Requires controller package/GitHub access and Docker. Existing bundles are
    reused only when source, platform and checksums match. A new checkout needs
    a fresh output directory. No benchmark image or task Python is modified.
    """
    if target not in _TARGETS:
        raise ValueError(f"Unsupported offline NOOA target {target!r}; expected one of {tuple(_TARGETS)}")
    root = Path(__file__).resolve().parents[2]
    source, fingerprint = _runtime_archive(root, requirements_path=requirements_path)
    output = output.expanduser().resolve()
    if output.exists():
        load_runtime_wheelhouse(output, target=target, source_fingerprint=fingerprint)
        return output
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Offline NOOA preparation requires Docker on the controller, or a compatible prepared bundle"
        )
    from nemo_gym.server_utils import (
        GlobalAIOHTTPAsyncClientConfig,
        is_global_aiohttp_client_setup,
        set_global_aiohttp_client,
    )
    from responses_api_agents.miniswe_sandboxed_agent.bootstrap import cached_download

    # The standalone preparation CLI has argparse flags, not Gym/Hydra overrides.
    # Reuse an existing Gym transport, or initialize just its HTTP defaults.
    if not is_global_aiohttp_client_setup():
        set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    python_archive = await cached_download(_python_download_url(target), sha256=_PYTHON_SHA256[target])
    output.parent.mkdir(parents=True, exist_ok=True)
    arch = target.split("/")[0]
    builder_image = builder_image or f"quay.io/pypa/manylinux_2_28_{arch}:latest"
    await _command("docker", "pull", "--platform", f"linux/{_TARGETS[target]}", builder_image)
    image_id = (await _command("docker", "image", "inspect", "--format", "{{.Id}}", builder_image)).strip()
    with tempfile.TemporaryDirectory(prefix=".nooa-wheelhouse-", dir=output.parent) as temporary:
        work = Path(temporary)
        bundle = work / "bundle"
        bundle.mkdir()
        shutil.copyfile(python_archive, bundle / "python.tar.gz")
        if file_sha256(bundle / "python.tar.gz") != _PYTHON_SHA256[target]:
            raise RuntimeError("NOOA Python artifact changed while preparing the offline bundle")
        (work / "source.tar.gz").write_bytes(source)
        container = f"nooa-wheelhouse-{uuid4().hex}"
        # Build scripts use only paths within the fresh bind mount, never the task image.
        script = """set -Eeuo pipefail
umask 022
mkdir -p /build/python /build/source/cache /build/wheels
uname -m > /build/architecture.txt
getconf GNU_LIBC_VERSION > /build/libc.txt
tar -xzf /build/bundle/python.tar.gz -C /build/python --strip-components=1
tar -xzf /build/source.tar.gz -C /build/source
/build/python/bin/python3 -I -B -m pip --isolated wheel --no-cache-dir --wheel-dir /build/wheels /build/source -r /build/source/runtime-requirements.txt
"""
        try:
            log = await _command(
                "docker",
                "run",
                "--rm",
                "--name",
                container,
                "--platform",
                f"linux/{_TARGETS[target]}",
                "--volume",
                f"{work}:/build",
                image_id,
                "bash",
                "-c",
                script,
            )
            (bundle / "build.log").write_text(log)
        finally:
            # Also remove a live builder after cancellation/timeout of the Docker client.
            try:
                await _command("docker", "rm", "--force", container, timeout=30)
            except RuntimeError:
                pass
        built_arch = (work / "architecture.txt").read_text().strip()
        libc = (work / "libc.txt").read_text().strip().split()
        if built_arch != arch or len(libc) != 2 or libc[0] != "glibc":
            raise RuntimeError(f"Builder does not match {target}: architecture={built_arch}, libc={libc}")
        wheels = sorted((work / "wheels").glob("*.whl"))
        if not wheels:
            raise RuntimeError("NOOA preparation produced no wheels")
        with tarfile.open(bundle / "wheels.tar.gz", "w:gz") as archive:
            for wheel in wheels:
                archive.add(wheel, arcname=wheel.name)
        manifest = {
            "version": 1,
            "target": target,
            "glibc": libc[1],
            "builder_image": image_id,
            "source_fingerprint": fingerprint,
            "python_version": _PYTHON_VERSION,
            "python_sha256": _PYTHON_SHA256[target],
            "wheels_sha256": file_sha256(bundle / "wheels.tar.gz"),
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        load_runtime_wheelhouse(bundle, target=target, source_fingerprint=fingerprint)
        try:
            os.rename(bundle, output)
        except OSError:
            # Concurrent builders may finish first; reuse only a verified compatible result.
            if not output.exists():
                raise
            load_runtime_wheelhouse(output, target=target, source_fingerprint=fingerprint)
    return output


def main() -> None:
    """Prepare or validate an offline bundle for the selected controller source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(_TARGETS), default="x86_64/gnu")
    parser.add_argument("--requirements-file", type=Path)
    parser.add_argument("--builder-image")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_runtime_wheelhouse(args.output, target=args.target, requirements_path=args.requirements_file)
    else:
        asyncio.run(
            prepare_runtime_wheelhouse(
                args.output,
                target=args.target,
                requirements_path=args.requirements_file,
                builder_image=args.builder_image,
            )
        )
    print(f"NOOA runtime wheelhouse ready: {args.output.resolve()}")


if __name__ == "__main__":
    main()
