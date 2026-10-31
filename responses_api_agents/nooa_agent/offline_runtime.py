# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a source-bound wheelhouse outside a network-restricted task sandbox."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from responses_api_agents.nooa_agent.sandbox_runtime import (
    _PYTHON_SHA256,
    _PYTHON_VERSION,
    _python_download_url,
    _runtime_archive,
    _runtime_io,
)


# Target -> Docker platform. Wheels are collected on the PyPA image for the target's C library.
_TARGETS = {"x86_64/gnu": "amd64", "aarch64/gnu": "arm64", "x86_64/musl": "amd64", "aarch64/musl": "arm64"}
_BUILDER_IMAGES = {
    "x86_64/gnu": "quay.io/pypa/manylinux2014_x86_64:latest",
    "aarch64/gnu": "quay.io/pypa/manylinux2014_aarch64:latest",
    "x86_64/musl": "quay.io/pypa/musllinux_1_2_x86_64:latest",
    "aarch64/musl": "quay.io/pypa/musllinux_1_2_aarch64:latest",
}
_LIBC_KIND = {"gnu": "glibc", "musl": "musl"}
# The oldest C library the portable interpreter itself runs on: Astral's builds link glibc 2.17 (manylinux2014) and
# musl 1.2.3 (verified on Alpine 3.17 task images; the x86_64 musl build is dynamically linked against libc.so).
_INTERPRETER_LIBC_FLOOR = {"gnu": "2.17", "musl": "1.2.3"}
_LEGACY_MANYLINUX = {"manylinux1": "2.5", "manylinux2010": "2.12", "manylinux2014": "2.17"}
_PLATFORM_TAG = re.compile(r"^(manylinux|musllinux)_(\d+)_(\d+)_")


def libc_floor(wheels: Iterable[str], *, libc: str, builder_version: str) -> str:
    """The oldest C library version a bundle's wheels and interpreter run on.

    PyPI wheels declare it in their platform tag (``musllinux_1_2``, ``manylinux_2_28``); a wheel compiled in the
    builder (a plain ``linux_<arch>`` tag) links the builder's own C library. The builder's version alone would
    exclude every task image older than the builder, which for musl is most Alpine releases.
    """
    kind = "manylinux" if libc == "gnu" else "musllinux"
    floors = [_INTERPRETER_LIBC_FLOOR[libc]]
    for wheel in wheels:
        for platform in Path(wheel).name.removesuffix(".whl").split("-")[-1].split("."):
            if platform == "any":
                continue
            tag = _PLATFORM_TAG.match(platform)
            legacy = next((v for k, v in _LEGACY_MANYLINUX.items() if platform.startswith(f"{k}_")), None)
            if tag is not None and tag.group(1) == kind:
                floors.append(f"{tag.group(2)}.{tag.group(3)}")
            elif legacy is not None and libc == "gnu":
                floors.append(legacy)
            elif tag is None and legacy is None:
                floors.append(builder_version)
    return max(floors, key=lambda version: tuple(int(part) for part in version.split(".")))


def file_sha256(path: Path) -> str:
    """Hash an artifact without loading large wheel archives into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_runtime_wheelhouse(output: Path, *, target: str, source_fingerprint: str) -> dict[str, object]:
    """Validate the exact source, platform and artifact bytes before staging."""
    output = output.expanduser().resolve()
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
    kind = _LIBC_KIND[target.split("/")[1]]
    manifest_kind = manifest.get("libc") or ("glibc" if "glibc" in manifest else None)
    version = manifest.get("libc_version") or manifest.get("glibc")
    if manifest_kind != kind:
        raise ValueError(
            f"NOOA wheelhouse was built for {manifest_kind or 'an unknown C library'}, target {target} needs {kind}"
        )
    if (
        not isinstance(version, str)
        or not all(p.isdigit() for p in version.split("."))
        or len(version.split(".")) not in (2, 3)
    ):
        raise ValueError(f"NOOA wheelhouse manifest requires its builder {kind} version")
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


def validate_runtime_wheelhouse(output: Path, *, target: str) -> dict[str, object]:
    """Check an existing prepared bundle against the current checkout."""
    _, fingerprint = _runtime_archive(Path(__file__).resolve().parents[2])
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
    source, fingerprint = await _runtime_io(_runtime_archive, root)
    output = output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        await _runtime_io(load_runtime_wheelhouse, output, target=target, source_fingerprint=fingerprint)
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
    arch, libc = target.split("/")
    builder_image = builder_image or _BUILDER_IMAGES[target]
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
if getconf GNU_LIBC_VERSION > /build/libc.txt 2>/dev/null; then :; else
  # musl reports its version on stderr: "musl libc (x86_64)" then "Version 1.2.5".
  # ldd exits non-zero on musl; the pipeline must not abort under pipefail before awk prints.
  (ldd --version 2>&1 || true) | awk '/^musl libc/ { kind = "musl" } /^Version / && kind { print kind, $2; exit }' > /build/libc.txt
fi
tar -xzf /build/bundle/python.tar.gz -C /build/python --strip-components=1
tar -xzf /build/source.tar.gz -C /build/source
/build/python/bin/python3 -I -B -m pip --isolated wheel --no-cache-dir --wheel-dir /build/wheels /build/source -r /build/source/runtime-requirements.txt
"""
        docker_run = (
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--platform",
            f"linux/{_TARGETS[target]}",
            "--volume",
            f"{work}:/build",
        )
        try:
            # Use the mount's container-visible owner: rootless Docker remaps host IDs.
            # Creating artifacts as this owner also allows cleanup after a killed builder.
            owner = (await _command(*docker_run, image_id, "stat", "-c", "%u:%g", "/build")).strip()
            log = await _command(
                *docker_run,
                "--user",
                owner,
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
            except RuntimeError as error:
                if "No such container" not in str(error):
                    raise
        built_arch = (work / "architecture.txt").read_text().strip()
        built_libc = (work / "libc.txt").read_text().strip().split()
        if built_arch != arch or len(built_libc) != 2 or built_libc[0] != _LIBC_KIND[libc]:
            raise RuntimeError(f"Builder does not match {target}: architecture={built_arch}, libc={built_libc}")
        wheels = sorted((work / "wheels").glob("*.whl"))
        if not wheels:
            raise RuntimeError("NOOA preparation produced no wheels")
        with tarfile.open(bundle / "wheels.tar.gz", "w:gz") as archive:
            for wheel in wheels:
                archive.add(wheel, arcname=wheel.name)
        floor = libc_floor((wheel.name for wheel in wheels), libc=libc, builder_version=built_libc[1])
        manifest = {
            "version": 1,
            "target": target,
            "libc": built_libc[0],
            "libc_version": floor,
            # Older loaders read the glibc floor from this key; musl bundles never had one.
            **({"glibc": floor} if libc == "gnu" else {}),
            "builder_libc_version": built_libc[1],
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
    parser.add_argument("--builder-image")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_runtime_wheelhouse(args.output, target=args.target)
    else:
        asyncio.run(
            prepare_runtime_wheelhouse(
                args.output,
                target=args.target,
                builder_image=args.builder_image,
            )
        )
    print(f"NOOA runtime wheelhouse ready: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
