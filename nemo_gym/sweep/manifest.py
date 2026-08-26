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
"""Schema, loading, and validation for sweep manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


AGENT_REF_KEY = "agent_ref"

# Reading every row of a multi-GB training file just to learn its agent_ref is wasteful, so
# validation samples the head by default. Raise it (or pass 0 for a full scan) when a file is
# suspected to mix agents partway through.
DEFAULT_SAMPLE_ROWS = 1000


class SweepValidationError(Exception):
    """Raised when a manifest is internally inconsistent or disagrees with its data."""


class SweepDefaults(BaseModel):
    """Sweep-wide settings that individual entries may override."""

    model_config = ConfigDict(extra="forbid")

    num_repeats: int = Field(default=1, ge=1)


class SweepEntry(BaseModel):
    """One (dataset, config) pair.

    ``agent`` is the ``agent_ref.name`` every row of ``data`` is expected to carry. It is
    declared rather than inferred so that a mispaired dataset and config fails loudly instead
    of silently scoring rollouts with the wrong verifier.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    data: str
    configs: List[str] = Field(min_length=1)
    agent: str
    # Who to ask about this dataset. Datasets outlive the person who built them, and the /lustre
    # path is not a reliable answer once data is copied or a blend is re-cut.
    owner: Optional[str] = None
    # Set only to deliberately run a dataset through a different agent than its rows name.
    # The rewrite is applied while concatenating and recorded in the build report.
    agent_ref_override: Optional[str] = None
    num_repeats: Optional[int] = Field(default=None, ge=1)
    # Free-form provenance (row counts, source paths, measured throughput). Never interpreted.
    notes: Dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_agent(self) -> str:
        """The agent rows will carry after any override is applied."""
        return self.agent_ref_override or self.agent


class SweepManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identifies this run and scopes every artifact it writes, so profiling the same blend
    # against a different checkpoint never collides. Mirrors NICKNAME in the pre-sweep scripts,
    # which scoped `rollouts/$NICKNAME/<env>/`.
    nickname: str
    # Split the concatenated input into N files. Leave unset (the default) to emit one file and
    # run the sweep as a single invocation; set it only when the driver process cannot hold every
    # materialized row at once, since rollout collection keeps them all resident.
    num_shards: Optional[int] = Field(default=None, ge=1)
    defaults: SweepDefaults = Field(default_factory=SweepDefaults)
    # Sweep-wide configs merged ahead of every entry's: the model server the agents reference,
    # plus optional judge and sandbox bindings. Entry configs declare agents and verifiers; they
    # do not declare the policy, so without at least a model-server config here the composed
    # config fails with "references responses_api_models/'policy_model', which is not defined".
    extra_configs: List[str] = Field(default_factory=list)
    entries: List[SweepEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_labels(self) -> "SweepManifest":
        dupes = [label for label, count in Counter(e.label for e in self.entries).items() if count > 1]
        if dupes:
            raise ValueError(f"Duplicate entry labels: {sorted(dupes)}")
        return self

    def config_paths(self) -> List[str]:
        """Union of every entry's configs, de-duplicated, first-seen order preserved."""
        seen: Dict[str, None] = {config: None for config in self.extra_configs}
        for entry in self.entries:
            for config in entry.configs:
                seen.setdefault(config, None)
        return list(seen)

    def num_repeats(self) -> Dict[str, int]:
        """A ``num_repeats`` mapping for ``gym eval run``.

        Keyed by agent because that is what rollout collection resolves against, so entries
        sharing an agent necessarily share a repeat count. ``resolve_conflicts`` reports those
        collisions rather than silently letting the last entry win.
        """
        repeats: Dict[str, int] = {"_default": self.defaults.num_repeats}
        for entry in self.entries:
            if entry.num_repeats is not None:
                repeats[entry.effective_agent] = entry.num_repeats
        return repeats

    def entries_by_agent(self) -> Dict[str, List[SweepEntry]]:
        grouped: Dict[str, List[SweepEntry]] = defaultdict(list)
        for entry in self.entries:
            grouped[entry.effective_agent].append(entry)
        return dict(grouped)


def load_manifest(path: str | Path) -> SweepManifest:
    path = Path(path)
    if not path.is_file():
        raise SweepValidationError(f"Manifest not found: {path}")
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise SweepValidationError(f"Manifest must be a mapping, got {type(raw).__name__}: {path}")
    return SweepManifest.model_validate(raw)


def observed_agents(data_path: Path, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> Counter:
    """Count the ``agent_ref.name`` values appearing in a JSONL file.

    Rows missing ``agent_ref`` are counted under ``None`` so the caller can report them; they
    are a real failure mode because ``--no-serve`` collection has no collation step to stamp one.
    """
    counts: Counter = Counter()
    with open(data_path) as handle:
        for line_number, line in enumerate(handle):
            if sample_rows and line_number >= sample_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SweepValidationError(f"{data_path}:{line_number + 1} is not valid JSON: {exc}") from exc
            ref = row.get(AGENT_REF_KEY)
            counts[ref.get("name") if isinstance(ref, dict) else None] += 1
    return counts


def _declared_top_level_keys(config_path: Path) -> set[str]:
    """Top-level server-instance keys a config file declares.

    Parsed directly rather than through Gym's config resolver so that validation stays offline
    and fast; nested ``config_paths`` are deliberately not followed, since an entry should name
    the config that actually declares its agent.
    """
    with open(config_path) as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        return set()
    return {key for key in raw if key != "config_paths"}


def validate_manifest(
    manifest: SweepManifest,
    *,
    repo_root: str | Path = ".",
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    check_data: bool = True,
) -> List[str]:
    """Check a manifest against the filesystem. Returns warnings; raises on errors."""
    repo_root = Path(repo_root)
    errors: List[str] = []
    warnings: List[str] = []

    for entry in manifest.entries:
        for config in entry.configs:
            config_path = config if Path(config).is_absolute() else repo_root / config
            if not Path(config_path).is_file():
                errors.append(f"[{entry.label}] config not found: {config}")
                continue
            if entry.agent not in _declared_top_level_keys(Path(config_path)):
                errors.append(
                    f"[{entry.label}] agent '{entry.agent}' is not declared by {config}. "
                    f"Point the entry at the config that defines it, or fix the agent name."
                )

        if not check_data:
            continue

        data_path = Path(entry.data)
        if not data_path.is_file():
            errors.append(f"[{entry.label}] data not found: {entry.data}")
            continue

        counts = observed_agents(data_path, sample_rows=sample_rows)
        if None in counts:
            errors.append(
                f"[{entry.label}] {counts[None]} sampled row(s) have no agent_ref. "
                f"--no-serve collection cannot route them."
            )
        unexpected = {name for name in counts if name is not None and name != entry.agent}
        if unexpected:
            errors.append(
                f"[{entry.label}] data declares agent_ref {sorted(unexpected)} but the manifest "
                f"says '{entry.agent}'. Fix the pairing, or set agent_ref_override to rewrite it."
            )

    for agent, entries in manifest.entries_by_agent().items():
        if len(entries) > 1:
            labels = sorted(e.label for e in entries)
            distinct = {e.num_repeats for e in entries}
            if len(distinct) > 1:
                errors.append(
                    f"Entries {labels} share agent '{agent}' but request different num_repeats "
                    f"{sorted(r for r in distinct if r is not None)}. num_repeats resolves per agent, "
                    f"so give them distinct agents or one repeat count."
                )
            else:
                warnings.append(f"Entries {labels} share agent '{agent}' and therefore share num_repeats.")

    if errors:
        raise SweepValidationError("Manifest validation failed:\n  " + "\n  ".join(errors))
    return warnings
