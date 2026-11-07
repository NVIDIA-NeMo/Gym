# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonicalize sampled editable installation paths without changing package metadata."""

import ast
import base64
import csv
import hashlib
import io
import json
import os
import re
from importlib.metadata import Distribution
from pathlib import Path


SOURCE_ROOT = "/__gym_sampled_source__"


def relocate_finder(contents: str, *, original: str, replacement: str) -> str:
    """Relocate only complete source-prefix string constants in the actual finder."""
    tree = ast.parse(contents)

    class Relocate(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and (node.value == original or node.value.startswith(original + "/")):
                node.value = replacement + node.value[len(original) :]
            return node

    result = ast.unparse(Relocate().visit(tree)) + "\n"
    if original != replacement and original in result:
        raise ValueError("Editable finder retained an unsupported source path")
    return result


def record_bytes(rows: list[tuple[str, bytes | None]]) -> bytes:
    """Write a complete RECORD with hashes for the represented file contents."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, contents in sorted(rows):
        if contents is None:
            writer.writerow((name, "", ""))
        else:
            digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).decode().rstrip("=")
            writer.writerow((name, "sha256=" + digest, str(len(contents))))
    return stream.getvalue().encode()


def cache_metadata(contents: bytes, *, mtime_ns: int) -> bytes:
    """Relocate uv's installer cache timestamp, preserving all other cache fields."""
    cache = json.loads(contents)
    timestamp = cache.get("timestamp")
    if not isinstance(timestamp, dict) or set(timestamp) != {"secs_since_epoch", "nanos_since_epoch"}:
        raise ValueError("Unsupported uv editable cache timestamp")
    cache["timestamp"] = {"secs_since_epoch": mtime_ns // 10**9, "nanos_since_epoch": mtime_ns % 10**9}
    return (json.dumps(cache, sort_keys=True, separators=(",", ":")) + "\n").encode()


def snapshot_editable(metadata: Path, *, source: Path, destination: Path) -> None:
    """Snapshot actual editable artifacts, with canonical paths and valid RECORD hashes."""
    source = source.resolve()
    site = metadata.parent
    direct = json.loads((metadata / "direct_url.json").read_text())
    if direct != {"dir_info": {"editable": True}, "url": source.as_uri()}:
        raise ValueError("Collected Gym distribution is not the sampled editable checkout")
    hooks = list(site.glob("__editable__*nemo_gym*.pth"))
    if len(hooks) != 1:
        raise ValueError("Expected the sampled Gym setuptools editable finder hook")
    hook = hooks[0]
    match = re.fullmatch(r"import ([A-Za-z0-9_]+); \1\.install\(\)\s*", hook.read_text())
    if not match:
        raise ValueError("Unsupported sampled Gym editable installation hook")
    finder = site / (match[1] + ".py")
    entries = Distribution.at(metadata).entry_points
    scripts = {entry.name for entry in entries if entry.group == "console_scripts"}
    files: dict[str, bytes] = {}
    paths: dict[str, str] = {}
    rows: list[tuple[str, bytes | None]] = []
    record_name = metadata.name + "/RECORD"
    for row in csv.reader(io.StringIO((metadata / "RECORD").read_text())):
        if len(row) != 3 or row[0] in paths or Path(row[0]).is_absolute():
            raise ValueError("Unsupported or duplicate editable RECORD entry")
        name, digest, size = row
        if name == record_name:
            if digest or size:
                raise ValueError("Editable RECORD must not hash itself")
            paths[name] = name
            rows.append((name, None))
            continue
        recorded = site / name
        contents = recorded.read_bytes()
        if digest or size:
            actual = next(csv.reader(io.StringIO(record_bytes([(name, contents)]).decode())))
            if [name, digest, size] != actual:
                raise ValueError(f"Sampled editable RECORD does not match {name}")
        stored = name
        if name.startswith("../../../bin/") and Path(name).name in scripts:
            if name != "../../../bin/" + Path(name).name:
                raise ValueError("Unsupported editable script path")
            interpreter = str(source / ".venv/bin/python")
            lines = contents.decode().splitlines(keepends=True)
            if not lines or lines[0] != f"#!{interpreter}\n":
                raise ValueError("Unsupported sampled console-script launcher")
            contents = (f"#!{SOURCE_ROOT}/.venv/bin/python\n" + "".join(lines[1:])).encode()
            stored = "console-scripts/" + Path(name).name
        elif name == finder.name:
            contents = relocate_finder(contents.decode(), original=str(source), replacement=SOURCE_ROOT).encode()
        elif name == hook.name:
            pass
        elif name.startswith(metadata.name + "/") and ".." not in Path(name).parts:
            if name == metadata.name + "/direct_url.json":
                contents = (
                    json.dumps({"dir_info": {"editable": True}, "url": Path(SOURCE_ROOT).as_uri()}) + "\n"
                ).encode()
            elif name == metadata.name + "/uv_cache.json":
                contents = cache_metadata(contents, mtime_ns=0)
        else:
            raise ValueError(f"Unsupported editable RECORD file: {name}")
        if str(source).encode() in contents or source.as_uri().encode() in contents:
            raise ValueError(f"Editable artifact retained an unsupported source path: {name}")
        paths[name] = stored
        files[stored] = contents
        rows.append((stored, contents))
    required = {record_name, finder.name, hook.name, metadata.name + "/direct_url.json"}
    if not required <= paths.keys() or scripts != {
        Path(name).name for name in paths if name.startswith("../../../bin/")
    }:
        raise ValueError("Sampled editable RECORD omits required installation artifacts")
    files[record_name] = record_bytes(rows)
    files["editable.json"] = (
        json.dumps(
            {
                "schema_version": 1,
                "source_root": SOURCE_ROOT,
                "hook": hook.name,
                "finder": finder.name,
                "record_files": paths,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    destination.mkdir(parents=True, exist_ok=False)
    for name, contents in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)


def restore_records(snapshot: Path, *, root: Path, installed: Path, child_site: Path, wheel_site: Path) -> None:
    """Describe the real relocated files and genuine wheel scripts in both metadata views."""
    manifest = json.loads((snapshot / "editable.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("source_root") != SOURCE_ROOT:
        raise ValueError("Unsupported canonical editable snapshot")
    metadata_name = installed.name
    for metadata in (installed, child_site / metadata_name):
        (metadata / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}, "url": root.as_uri()}))
        cache = metadata / "uv_cache.json"
        if cache.exists():
            cache.write_bytes(
                cache_metadata(cache.read_bytes(), mtime_ns=(root / "pyproject.toml").stat().st_mtime_ns)
            )
        rows = []
        for name in manifest["record_files"]:
            if name.startswith("../../../bin/"):
                target = (wheel_site / name).resolve()
            elif name.startswith(metadata_name + "/"):
                target = metadata / name.removeprefix(metadata_name + "/")
            elif name in (manifest["finder"], manifest["hook"]):
                target = child_site / name
            else:
                raise ValueError(f"Unsupported canonical RECORD path: {name}")
            relative = os.path.relpath(target, metadata.parent)
            rows.append((relative, None if target == metadata / "RECORD" else target.read_bytes()))
        (metadata / "RECORD").write_bytes(record_bytes(rows))
