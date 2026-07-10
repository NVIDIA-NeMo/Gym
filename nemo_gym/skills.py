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
"""Agent skills: a directory of skills made available to an agent at rollout time.

Skills follow the open `Agent Skills standard <https://agentskills.io/specification>`_
used by Codex CLI and Claude Code. A skill is a *directory* containing a ``SKILL.md``
file (YAML frontmatter + markdown body) plus optional supporting files. The ``skills.path``
config points at a directory of such skill directories.

Skills are a run-level knob (specified on ``gym eval run``), applied to a fixed,
skill-agnostic dataset -- mirroring how ``prompt.py`` applies a prompt template. They are
*not* a dataset-row field, so the same dataset is reusable across skill variants. Each
rollout result is stamped with a ``skills_ref`` for provenance/grouping in reward profiling.

The ``skills_ref`` carries a content ``hash`` (a short sha256 over the skill directory's
sorted relative paths + file bytes) so that variants that mutate a skill *in place* at the
same path -- as optimizer loops like ACE, GEPA, and EvoSkill commonly do -- remain
distinguishable when comparing rollouts. Identity is derived from bytes on disk, so it
requires no cooperation from the optimizer.
"""

import hashlib
import shutil
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

from nemo_gym import PARENT_DIR


SKILL_MD_FILENAME = "SKILL.md"
# 12 hex chars (48 bits) is plenty to separate the handful of variants in one experiment
# while staying readable in tables / W&B.
_HASH_PREFIX_LEN = 12


class SkillMetadata(BaseModel):
    """Metadata parsed from a single skill's ``SKILL.md`` YAML frontmatter."""

    name: str
    description: Optional[str] = None
    version: Optional[str] = None


class SkillsRef(BaseModel):
    """Provenance stamp describing the skills made available for a run.

    Stamped onto rollout result rows (not source datasets). ``hash`` is a content
    digest so two skill *versions* at the same ``path`` do not collide in profiling.
    """

    path: str
    hash: str
    skills: List[SkillMetadata]


class SkillsConfig(BaseModel):
    """Run-level skills config: ``skills.path`` points at a directory of skill directories."""

    path: str = Field(description="Directory of Agent Skills standard skill directories to make available.")


def _resolve_skills_path(path: str) -> Path:
    """Resolve a skills path. Relative paths check cwd first, then the Gym root (PARENT_DIR).

    This matches how ``input_jsonl_fpath`` and ``config_paths`` are resolved.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    cwd_path = Path.cwd() / p
    return cwd_path if cwd_path.exists() else PARENT_DIR / p


def hash_skill_dir(root: Path) -> str:
    """Compute a stable short sha256 over a skill directory's contents.

    Walks files in sorted relative-path order, folding each file's relative path and bytes
    into the digest. Including the relative path means renaming or adding/removing files also
    changes the hash -- a skill *is* its file layout, not just ``SKILL.md``'s bytes.
    """
    h = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(file_path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(file_path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:_HASH_PREFIX_LEN]


def parse_skill_md(skill_md_path: Path) -> SkillMetadata:
    """Parse a ``SKILL.md`` file's YAML frontmatter into ``SkillMetadata``.

    Frontmatter is delimited by lines containing only ``---``. Raises ``ValueError`` with a
    clear message if the frontmatter is missing, malformed, or lacks a ``name``.
    """
    # utf-8-sig transparently strips a leading BOM if present; Claude Code's own loader tolerates
    # one, so a BOM should not make Gym reject a skill that would otherwise run.
    text = skill_md_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"Skill file {skill_md_path} is missing YAML frontmatter. "
            f"It must start with a '---' line followed by 'name:' and 'description:' fields."
        )

    closing_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_idx = idx
            break
    if closing_idx is None:
        raise ValueError(
            f"Skill file {skill_md_path} has an unterminated YAML frontmatter block (no closing '---' line)."
        )

    frontmatter_text = "\n".join(lines[1:closing_idx])
    try:
        data = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Skill file {skill_md_path} has malformed YAML frontmatter: {e}") from None
    if not isinstance(data, dict):
        raise ValueError(f"Skill file {skill_md_path} frontmatter must be a YAML mapping, got {type(data).__name__}.")
    if not data.get("name"):
        raise ValueError(f"Skill file {skill_md_path} frontmatter is missing a required 'name' field.")

    # The Agent Skills spec nests version under an optional `metadata:` map
    # (e.g. `metadata: {version: "1.0"}`), not as a top-level key.
    metadata = data.get("metadata")
    version = str(metadata["version"]) if isinstance(metadata, dict) and metadata.get("version") is not None else None
    return SkillMetadata(
        name=str(data["name"]),
        description=data.get("description"),
        version=version,
    )


def load_skill_directory(path: str) -> SkillsRef:
    """Load a directory of skills, returning a ``SkillsRef`` (path, content hash, metadata).

    The directory at ``path`` contains one subdirectory per skill, each with a ``SKILL.md``.
    Raises ``ValueError`` with an actionable message if the path is missing, is not a
    directory, contains no skills, or contains a malformed skill.
    """
    resolved = _resolve_skills_path(path)
    if not resolved.exists():
        raise ValueError(f"Skills path does not exist: {resolved} (from skills.path={path!r}).")
    if not resolved.is_dir():
        raise ValueError(
            f"Skills path must be a directory of skill directories, but {resolved} is a file "
            f"(from skills.path={path!r})."
        )

    skill_dirs = sorted(d for d in resolved.iterdir() if d.is_dir())
    skills: List[SkillMetadata] = []
    for skill_dir in skill_dirs:
        skill_md = skill_dir / SKILL_MD_FILENAME
        if not skill_md.is_file():
            raise ValueError(
                f"Skill directory {skill_dir} is missing a {SKILL_MD_FILENAME} file. "
                f"Each skill must be a directory containing a {SKILL_MD_FILENAME}."
            )
        skills.append(parse_skill_md(skill_md))

    if not skills:
        raise ValueError(
            f"Skills path {resolved} contains no skills. Expected one or more subdirectories, "
            f"each containing a {SKILL_MD_FILENAME} file (from skills.path={path!r})."
        )

    return SkillsRef(path=path, hash=hash_skill_dir(resolved), skills=skills)


def stage_skills(path: str, dest_skills_dir: Path) -> None:
    """Copy the directory of skills at ``path`` into ``dest_skills_dir``.

    Used by agent runtimes to materialize skills into a location their native discovery
    mechanism scans (e.g. ``<CLAUDE_CONFIG_DIR>/skills/`` for Claude Code). ``dest_skills_dir``
    must not already exist. Raises ``ValueError`` if the source path is missing or not a directory.
    """
    resolved = _resolve_skills_path(path)
    if not resolved.is_dir():
        raise ValueError(
            f"Cannot stage skills: {resolved} is not a directory (from skills path {path!r}). "
            f"For distributed runs the skills directory must be on storage accessible to the agent."
        )
    shutil.copytree(resolved, dest_skills_dir)
