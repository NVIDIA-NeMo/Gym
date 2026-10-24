# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned SWE-Together runtimes on the trusted environment host.

Run ``python -m benchmarks.swe_together.prepare_runtime --output /path/to/cache``.
The resulting manifest is consumed by the environment before any task sandbox starts.
These artifacts support Linux x86_64 task images with glibc. Preparation may run
on any trusted host. Python is the complete install-only archive, including its
standard library and shared libraries; the environment unpacks it before task setup.
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


RUNTIMES: dict[str, dict[str, str]] = {
    "python": {
        "version": "3.12.8+20250115",
        "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20250115/cpython-3.12.8%2B20250115-x86_64-unknown-linux-gnu-install_only.tar.gz",
        "sha256": "e5435e717c934ed30d4066f64e858497c27f37c1ba547f403b050d9221e50ea4",  # pragma: allowlist secret (public runtime digest)
        "format": "tar.gz",
    },
    "opencode": {
        "version": "1.15.13",
        "url": "https://github.com/anomalyco/opencode/releases/download/v1.15.13/opencode-linux-x64.tar.gz",
        "sha256": "8e97f83175c84daeb51e96b34baf9ee9839ae54e5e848a3cce1e0e32dcf41a6c",  # pragma: allowlist secret (public runtime digest)
        "member": "opencode",
    },
    "rg": {
        "version": "15.1.0",
        "url": "https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz",
        "sha256": "ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2",  # pragma: allowlist secret (public runtime digest)
        "archive_sha256": "1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599",  # pragma: allowlist secret (public runtime digest)
        "member": "rg",
    },
    "claude": {
        "version": "2.1.108",
        "url": "https://downloads.claude.ai/claude-code-releases/2.1.108/linux-x64/claude",
        "sha256": "1f641676eb086ceafd76a5f51ffb260b948fa1f6d77991d8cd6e480267dfff37",  # pragma: allowlist secret (public runtime digest)
    },
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink() or digest(path) != expected:
        raise ValueError(f"Pinned runtime digest mismatch: {path}")


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "NeMo-Gym-SWE-Together"})
    with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def prepare(output: Path, seed_manifest: Path | None = None) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = {}
    if seed_manifest is not None:
        seed = json.loads(seed_manifest.read_text())
        if seed.get("version") != 1:
            raise ValueError("Unsupported seed manifest version")
        seeds = seed["artifacts"]
    artifacts = {}
    for name, runtime in RUNTIMES.items():
        suffix = ".tar.gz" if runtime.get("format") else ""
        target = output / "sha256" / runtime["sha256"] / (name + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            check(target, runtime["sha256"])
        else:
            with tempfile.TemporaryDirectory(dir=target.parent, prefix=".prepare-") as temp:
                staged = Path(temp) / "runtime"
                if name in seeds:
                    source = Path(seeds[name]["path"]).resolve()
                    check(source, runtime["sha256"])
                    shutil.copyfile(source, staged)
                elif runtime.get("member"):
                    archive = Path(temp) / "download.tar.gz"
                    download(runtime["url"], archive)
                    if runtime.get("archive_sha256"):
                        check(archive, runtime["archive_sha256"])
                    with tarfile.open(archive, "r:gz") as tar:
                        members = [
                            member for member in tar if member.isfile() and Path(member.name).name == runtime["member"]
                        ]
                        if len(members) != 1:
                            raise ValueError(f"Expected one regular {name} binary in {runtime['url']}")
                        with tar.extractfile(members[0]) as source, staged.open("wb") as dest:
                            shutil.copyfileobj(source, dest, length=1024 * 1024)
                else:
                    download(runtime["url"], staged)
                check(staged, runtime["sha256"])
                staged.chmod(0o644 if runtime.get("format") else 0o755)
                os.replace(staged, target)
        artifacts[name] = {"path": str(target), "sha256": runtime["sha256"]}
        if runtime.get("format"):
            artifacts[name]["format"] = runtime["format"]
    binaries = output / "bin"
    binaries.mkdir(exist_ok=True)
    for name, artifact in artifacts.items():
        if name != "python":
            alias = binaries / name
            alias.unlink(missing_ok=True)
            alias.symlink_to(artifact["path"])
    manifest = output / "manifest.json"
    payload = {"version": 1, "artifacts": artifacts}
    with tempfile.NamedTemporaryFile(mode="w", dir=output, prefix=".manifest-", delete=False) as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Trusted runtime cache directory")
    parser.add_argument("--seed-manifest", type=Path, help="Reuse an existing verified runtime manifest")
    args = parser.parse_args()
    print(prepare(args.output, args.seed_manifest))


if __name__ == "__main__":
    main()
