# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the external runtime under the normal trusted Pants-cache assumption."""

import argparse
import hashlib
import json
import os
import platform
import socket
import stat
import subprocess
import sys
from pathlib import Path


CONFIGURATION = (
    "/etc/ld.so.cache",
    "/etc/ld.so.preload",
    "/etc/nsswitch.conf",
    "/etc/os-release",
    "/etc/passwd",
    "/etc/ssl/openssl.cnf",
    "/etc/localtime",
    "/usr/share/zoneinfo/UTC",
    "/usr/share/locale/locale.alias",
    "/usr/bin/uname",
    "/bin/sh",
    "/bin/bash",
)


def fingerprint(roots: list[str], paths: list[str]) -> dict[str, object]:
    """Hash complete interpreter/system-library content; preserve negative inputs.

    The test uses a sandbox-owned PYTHONPYCACHEPREFIX. Ambient bytecode is
    neither read nor written, so it is excluded from the interpreter inventory.
    Dependency PEX bytes and validated resolve artifacts are already immutable
    Pants process inputs; their extracted named caches are trusted as Pants does.
    """
    entries = {}
    count = 0
    pending = [Path(name) for name in roots]
    visited = set()
    while pending:
        base = pending.pop().resolve()
        if base in visited:
            continue
        visited.add(base)
        if not base.is_dir():
            raise ValueError(f"Missing runtime root: {base}")
        for directory, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name != "__pycache__")
            for name in [*dirs, *sorted(files)]:
                path = Path(directory) / name
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    entries[str(path)] = {"link": os.readlink(path)}
                    resolved = path.resolve()
                    if resolved.is_dir() and not any(resolved.is_relative_to(Path(root)) for root in roots):
                        pending.append(resolved)
                    elif resolved.is_file():
                        data = resolved.read_bytes()
                        count += len(data)
                        entries[str(path)]["target_sha256"] = hashlib.sha256(data).hexdigest()
                elif stat.S_ISREG(info.st_mode):
                    data = path.read_bytes()
                    count += len(data)
                    entries[str(path)] = {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "mode": stat.S_IMODE(info.st_mode),
                    }
                elif not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f"Unexpected runtime special file: {path}")
    for name in paths:
        path = Path(name)
        if path.is_file():
            data = path.read_bytes()
            count += len(data)
            entries[name] = {"sha256": hashlib.sha256(data).hexdigest(), "resolved": str(path.resolve())}
        elif not path.exists():
            entries[name] = {"missing": True}
        else:
            raise ValueError(f"Unsupported configuration input: {path}")
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
        ipv6 = True
    except OSError:
        ipv6 = False
    body = {
        "entries": entries,
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "ipv6_bind_available": ipv6,
    }
    return {
        "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "body": body,
        "bytes_hashed": count,
    }


def capture(output: Path) -> None:
    """Capture unconditionally before Pants can perform a successful-result lookup."""
    if sys.platform != "linux":
        raise ValueError("Incremental runtime verification currently supports Linux only")
    roots = sorted({str(Path(sys.base_prefix).resolve()), str(Path("/usr/lib").resolve())})
    paths = sorted(
        set(CONFIGURATION)
        | external_configuration(
            output.resolve().parents[3], inventory_output=output.with_name("configuration-inputs.json")
        )
    )
    result = fingerprint(roots, paths)
    output.write_text(
        json.dumps({"schema_version": 1, "roots": roots, "paths": paths, "fingerprint": result}, sort_keys=True) + "\n"
    )


def external_configuration(source: Path, *, inventory_output: Path | None = None) -> set[str]:
    """Fingerprint absolute YAML inheritance, including absent referenced files.

    Paths come from the sampled public configuration, never a controller-specific
    hardcoded location. Existing external YAML is traversed recursively so its
    own inheritance cannot bypass the content fingerprint.
    """
    import yaml
    from incremental import SCANNED_ROOTS, declared_files

    source = source.resolve()
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0"))
    pending = [
        source / name
        for name in declared_files(source)
        if name.startswith(SCANNED_ROOTS) and name.endswith((".yaml", ".yml"))
    ]
    visited = set()
    external = set()
    included = set()
    missing = set()

    def include(value: str) -> None:
        candidate = Path(value)
        if candidate.is_absolute():
            external.add(value)
            pending.append(candidate)
            return
        candidate = source / os.path.normpath(value)
        if not candidate.resolve().is_relative_to(source):
            raise ValueError("Relative configuration inheritance escapes the sampled checkout")
        name = candidate.relative_to(source).as_posix()
        if candidate.is_file():
            if name not in tracked:
                raise ValueError(f"Configuration inheritance reads an untracked file: {name}")
            included.add(name)
            pending.append(candidate)
        elif not candidate.exists():
            missing.add(name)
        else:
            raise ValueError(f"Configuration inheritance is not a file: {name}")

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from strings(key)
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    def check_resource_entrypoints(value):
        if isinstance(value, dict):
            resources = value.get("resources_servers")
            if isinstance(resources, dict):
                for implementation, config in resources.items():
                    if not isinstance(config, dict):
                        continue
                    entrypoint = config.get("entrypoint") or "app.py"
                    if not isinstance(implementation, str) or not isinstance(entrypoint, str):
                        raise ValueError("Unsupported resource entrypoint syntax")
                    root = source / "resources_servers"
                    candidate = root / implementation / entrypoint
                    if (
                        Path(implementation).is_absolute()
                        or Path(entrypoint).is_absolute()
                        or not candidate.resolve().is_relative_to(root.resolve())
                    ):
                        raise ValueError("Resource AST entrypoint escapes declared resources_servers inputs")
                    if candidate.is_file() and candidate.relative_to(source).as_posix() not in tracked:
                        raise ValueError("Resource AST entrypoint is not a tracked source input")
            for item in value.values():
                check_resource_entrypoints(item)
        elif isinstance(value, list):
            for item in value:
                check_resource_entrypoints(item)

    while pending:
        path = pending.pop()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        for document in yaml.safe_load_all(path.read_text()):
            check_resource_entrypoints(document)
            if isinstance(document, dict):
                references = document.get("config_paths") or []
                if not isinstance(references, list) or not all(isinstance(value, str) for value in references):
                    raise ValueError("Unsupported configuration inheritance syntax")
                for reference in references:
                    include(reference)
            for value in strings(document):
                candidate = Path(value)
                if (
                    candidate.is_absolute()
                    and candidate.suffix in {".yaml", ".yml"}
                    and not candidate.is_relative_to(source)
                ):
                    external.add(value)
                    pending.append(candidate)
    if inventory_output is not None:
        inventory_output.write_text(json.dumps({"files": sorted(included), "missing": sorted(missing)}) + "\n")
    return external


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    capture(parser.parse_args().output)
