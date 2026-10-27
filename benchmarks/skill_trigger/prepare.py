# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare immutable skill catalogs and a host-only manifest from pinned public sources."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, ConfigDict, Field


BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parents[1]
EXCLUDED = {"evals", ".git", "__pycache__"}


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Source(Record):
    url: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    notices: list[str]


class Catalog(Record):
    source: str
    directories: list[str]
    mounted_sha256: dict[str, str]


class Provenance(Record):
    kind: str
    path: str
    pointer: str | None = None
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str


class Case(Record):
    id: str
    suite: str
    skill: str
    target: str
    target_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog: str
    prompt: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: Provenance


class ReleaseBlocker(Record):
    source: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str


class Selection(Record):
    version: int
    expected_counts: dict[str, int]
    sources: dict[str, Source]
    catalogs: dict[str, Catalog]
    cases: list[Case]
    release_blockers: list[ReleaseBlocker] = Field(default_factory=list)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: object) -> bytes:
    """Stable UTF-8 serialization for published identities and artifacts."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def source_path(root: Path, relative: str) -> Path:
    """Reject traversal and symlinks, including links that stay inside a checkout."""
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or any(p in {"..", "."} for p in parts):
        raise ValueError(f"Unsafe source path: {relative}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlinks are not permitted in a frozen catalog: {relative}")
    if not path.exists() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing or escaping source path: {relative}")
    return path


def checkout(root: Path, source: Source, *, fetch: bool) -> list[str]:
    """Fetch only the pinned revision into a new cache; never reset an existing checkout."""
    if not root.exists():
        if not fetch:
            raise FileNotFoundError(f"Missing checkout {root}; run preparation with --fetch")
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", source.url, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", source.commit], check=True)
    if git(root, "rev-parse", "HEAD") != source.commit:
        raise ValueError(f"Wrong revision in {root}; use a separate clean cache")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Modified or untracked source files in {root}")
    return git(root, "ls-files", "-z").split("\0")


def skill_name(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Missing skill front matter: {path}")
    fields = yaml.safe_load(text.split("---", 2)[1])
    if not isinstance(fields, dict) or not isinstance(fields.get("name"), str) or not fields.get("description"):
        raise ValueError(f"Missing skill name/description: {path}")
    name = fields["name"]
    if name in {".", ".."} or any(char in name for char in "/\\\0") or any(ord(char) < 32 for char in name):
        raise ValueError(f"Unsafe skill name: {name}")
    return name


def tree_entries(repo: Path, directory: str, tracked: list[str], *, exclude_labels: bool) -> list[tuple[str, str]]:
    """Hash the same tracked file bytes as the historical skill-tree identity."""
    prefix = directory + "/"
    entries = []
    for relative in sorted(p for p in tracked if p.startswith(prefix)):
        local = relative[len(prefix) :]
        if exclude_labels and EXCLUDED.intersection(PurePosixPath(local).parts):
            continue
        path = source_path(repo, relative)
        if not path.is_file():
            raise ValueError(f"Non-file catalog entry: {relative}")
        entries.append((local, digest(path.read_bytes())))
    if not entries:
        raise ValueError(f"Empty skill directory: {directory}")
    if exclude_labels:
        # The historical mounted-tree hash sorts Path components, whereas its
        # complete tracked-tree hash sorts string paths. Preserve both formats.
        entries.sort(key=lambda entry: PurePosixPath(entry[0]))
    return entries


def tree_digest(entries: list[tuple[str, str]]) -> str:
    # Deliberately preserve the historical JSON tuple serialization for comparison.
    return digest(json.dumps(entries, separators=(",", ":")).encode())


def upstream_prompt(path: Path, pointer: str | None) -> str | None:
    """Read a published prompt by JSON pointer; authored cases have no pointer."""
    if pointer is None:
        return None
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {pointer}")
    value = json.loads(path.read_text(encoding="utf-8"))
    for token in pointer[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pointer does not identify a prompt: {path}#{pointer}")
    return value


def prepare_release(
    selection: Selection, *, sources_root: Path, output: Path, fetch: bool = False, audit_only: bool = False
) -> Path:
    """Validate all inputs before atomically publishing an immutable prepared release.

    Only catalogs/ is model-visible. The manifest and upstream prompt provenance
    contain labels and must remain on the trusted host when tasks are packaged.
    Audit mode emits metadata only; it never copies catalogs or licenses and is
    not a runnable release. A release blocker must be resolved explicitly.
    """
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared release: {output}")
    if selection.release_blockers and not audit_only:
        raise ValueError("Unresolved release blockers; use --audit-only to inspect metadata without packaging skills")
    if dict(Counter(case.suite for case in selection.cases)) != selection.expected_counts:
        raise ValueError("Case counts differ from the declared release")
    if len({c.id for c in selection.cases}) != len(selection.cases):
        raise ValueError("Duplicate case ID")
    if len({(c.suite, c.skill) for c in selection.cases}) != len(selection.cases):
        raise ValueError("Duplicate selected skill")
    tracked = {}
    for name, source in selection.sources.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError(f"Unsafe source name: {name}")
        tracked[name] = checkout(sources_root / name, source, fetch=fetch)
    for blocker in selection.release_blockers:
        path = source_path(sources_root / blocker.source, blocker.path)
        if digest(path.read_bytes()) != blocker.sha256:
            raise ValueError(f"Release-blocker source changed: {blocker.path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skill-trigger-", dir=output.parent) as temporary:
        stage = Path(temporary) / "release"
        stage.mkdir()
        catalogs = {}
        for catalog_id, catalog in selection.catalogs.items():
            if not re.fullmatch(r"[a-z0-9-]+", catalog_id):
                raise ValueError(f"Unsafe catalog ID: {catalog_id}")
            repo = sources_root / catalog.source
            if set(catalog.mounted_sha256) != set(catalog.directories):
                raise ValueError(f"Incomplete catalog hash lock: {catalog_id}")
            members = []
            names = set()
            for directory in catalog.directories:
                folder = source_path(repo, directory)
                name = skill_name(source_path(folder, "SKILL.md"))
                if name in names:
                    raise ValueError(f"Duplicate skill name in {catalog_id}: {name}")
                names.add(name)
                files = tree_entries(repo, directory, tracked[catalog.source], exclude_labels=True)
                all_files = tree_entries(repo, directory, tracked[catalog.source], exclude_labels=False)
                if tree_digest(files) != catalog.mounted_sha256[directory]:
                    raise ValueError(f"Mounted skill tree changed: {directory}")
                if not audit_only:
                    for local, expected_hash in files:
                        src = source_path(folder, local)
                        dest = stage / "catalogs" / catalog_id / name / local
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dest)
                        if digest(dest.read_bytes()) != expected_hash:
                            raise ValueError(f"Source changed during preparation: {directory}/{local}")
                        dest.chmod(0o755 if src.stat().st_mode & 0o111 else 0o644)
                members.append(
                    {
                        "name": name,
                        "source_directory": directory,
                        "tree_sha256": tree_digest(all_files),
                        "mounted_sha256": tree_digest(files),
                        "files": files,
                        "excluded_files": sorted(set(p for p, _ in all_files) - set(p for p, _ in files)),
                    }
                )
            catalogs[catalog_id] = {"source": catalog.source, "members": members, "sha256": digest(encode(members))}
        cases = []
        for case in selection.cases:
            if not case.prompt.strip() or digest(case.prompt.encode()) != case.prompt_sha256:
                raise ValueError(f"Prompt changed: {case.id}")
            catalog = catalogs[case.catalog]
            if catalog["source"] != case.suite:
                raise ValueError(f"Catalog belongs to another source: {case.id}")
            matches = [
                m for m in catalog["members"] if m["name"] == case.skill and m["source_directory"] == case.target
            ]
            if len(matches) != 1 or matches[0]["tree_sha256"] != case.target_tree_sha256:
                raise ValueError(f"Selected skill version is missing or changed: {case.id}")
            provenance_path = source_path(sources_root / case.suite, case.provenance.path)
            if digest(provenance_path.read_bytes()) != case.provenance.source_sha256:
                raise ValueError(f"Provenance file changed: {case.id}")
            original = upstream_prompt(provenance_path, case.provenance.pointer)
            record = case.model_dump()
            record.update({"task_name": "case-" + digest(case.id.encode())[:12], "catalog_sha256": catalog["sha256"]})
            record["provenance"].update(
                {
                    "upstream_prompt": original,
                    "wording_changed": original != case.prompt if original is not None else None,
                }
            )
            cases.append(record)
        if len({c["task_name"] for c in cases}) != len(cases):
            raise ValueError("Task-name hash collision")
        for name, source in selection.sources.items():
            for relative in source.notices:
                src = source_path(sources_root / name, relative)
                if not audit_only:
                    dest = stage / "notices" / name / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
        manifest = {
            "artifact_kind": "source-audit" if audit_only else "prepared-release",
            "release_blockers": [b.model_dump() for b in selection.release_blockers],
            "version": selection.version,
            "selection_sha256": digest(encode(selection.model_dump())),
            "sources": {k: v.model_dump() for k, v in selection.sources.items()},
            "catalogs": catalogs,
            "cases": cases,
        }
        (stage / "manifest.json").write_bytes(encode(manifest))
        stage.rename(output)
    return output / "manifest.json"


def prepare(
    *,
    sources_root: Path = ROOT / ".cache/skill-trigger-sources",
    output: Path = BENCHMARK_DIR / "data/release",
    fetch: bool = False,
    audit_only: bool = False,
) -> Path:
    selection = Selection.model_validate_json((BENCHMARK_DIR / "selection.json").read_text())
    return prepare_release(selection, sources_root=sources_root, output=output, fetch=fetch, audit_only=audit_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-root", type=Path, default=ROOT / ".cache/skill-trigger-sources")
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "data/release")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--audit-only", action="store_true", help="Validate inputs and write metadata without packaging"
    )
    args = parser.parse_args()
    print(prepare(sources_root=args.sources_root, output=args.output, fetch=args.fetch, audit_only=args.audit_only))
