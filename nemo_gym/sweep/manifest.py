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
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


AGENT_REF_KEY = "agent_ref"

# Reading every row of a multi-GB training file just to learn its agent_ref is wasteful, so
# validation samples the head by default. Raise it (or pass 0 for a full scan) when a file is
# suspected to mix agents partway through.
DEFAULT_SAMPLE_ROWS = 1000


class SweepValidationError(ValueError):
    """Raised when a manifest is internally inconsistent or disagrees with its data."""


def _within_one_edit(a: str, b: str) -> bool:
    """True when a and b differ by one insertion, deletion or substitution, and are not equal."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):
        if short == long[:i] + long[i + 1 :]:
            return True
    return False


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
    # How much of this dataset is in scope, overriding materialize.limit_per_entry. Part of what
    # the sweep *is*, so it is committed -- unlike a smoke run's LIMIT_PER_ENTRY, which is a
    # property of one invocation and belongs on the command line.
    limit: Optional[int] = Field(default=None, ge=1)
    # Free-form provenance (row counts, source paths, measured throughput). Never interpreted.
    notes: Dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_agent(self) -> str:
        """The agent rows will carry after any override is applied."""
        return self.agent_ref_override or self.agent


class GymEnvStart(BaseModel):
    """What becomes ``sweep_config.yaml``, which is passed to ``gym env start --config``.

    ``config_paths`` are merged ahead of every entry's own configs: the model server the agents
    reference, plus anything shared. Every other key is ordinary Gym config, spliced in verbatim.

    Those extra keys act as an overlay. Per ``global_config.load_extra_config_paths``: "A config
    named in another config's config_paths is *inner*. The config that pulled it in overrides it."
    ``sweep_config.yaml`` is the outer config, so its own top-level keys beat every file it lists.
    Use that for bindings this sweep owns -- which judge a resources server talks to, say -- rather
    than editing an upstream config. The container is built from a Gym ref and does not contain
    this repo, so a repo-relative config path will not resolve inside it, whereas
    ``sweep_config.yaml`` lives in SWEEP_DIR and is always mounted.
    """

    model_config = ConfigDict(extra="allow")

    config_paths: List[str] = Field(default_factory=list)

    def overlay(self) -> Dict[str, Any]:
        """Every key except config_paths: the Gym config spliced in alongside it."""
        return dict(self.model_extra or {})


class GymEvalRun(BaseModel):
    """Runtime settings for the collection command, emitted as ``++key=value``.

    Committed with the manifest so a sweep's settings travel with it. A launcher passes an override
    only when its own env var is set, so these are defaults rather than something clobbered:
    manifest -> script env var -> command line, lowest to highest.
    """

    model_config = ConfigDict(extra="allow")

    # Rollouts per task; the spread across them is the profile. Also read by materialize, which
    # writes this many copies of each row -- it is a gym eval run argument that the expander
    # happens to need too.
    num_repeats: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _reject_near_miss_keys(self) -> "GymEvalRun":
        """Catch `num_repeat` for `num_repeats`, which otherwise costs a whole run.

        Extra keys are forwarded verbatim as ``++key=value`` and Hydra force-adds unknown ones
        without complaint, so a one-character typo leaves the real field at its default and the
        override lands somewhere nothing reads. For num_repeats that means a reward profile with
        one rollout per task -- no spread, which is the entire point of the run.
        """
        known = set(type(self).model_fields)
        for key in self.model_extra or {}:
            near = [k for k in known if _within_one_edit(key, k)]
            if near:
                raise ValueError(f"gym_eval_run has '{key}'; did you mean '{near[0]}'?")
        return self

    def overrides(self) -> Dict[str, Any]:
        """Everything except num_repeats, which the launcher derives from the sweep report."""
        return dict(self.model_extra or {})


class Materialize(BaseModel):
    """What ``nemo_gym.sweep materialize`` reads, i.e. how the input file gets built.

    Separate from ``gym_eval_run`` because these are decided before collection exists: they
    determine which rows are in the sweep at all, not how the sweep is run.
    """

    model_config = ConfigDict(extra="forbid")

    # Rows to take from every entry, unless the entry sets its own ``limit``. Unset means all.
    limit_per_entry: Optional[int] = Field(default=None, ge=1)
    # ``head`` takes the first N and can stop reading there, which is why it is the default and
    # what a smoke run wants. ``random`` must read the whole file to sample it -- 36.9 GB here --
    # but the first N rows of a sorted dataset are a biased subset, so prefer it when the limit is
    # a deliberate subset rather than a smoke test.
    sample: Literal["head", "random"] = "head"
    # Seeds ``random`` per entry label, so selection is reproducible and does not shift when
    # another entry is added.
    seed: int = 0

    def limit_for(self, entry: "SweepEntry") -> Optional[int]:
        return entry.limit if entry.limit is not None else self.limit_per_entry


class Sbatch(BaseModel):
    """Slurm settings for the launchers, which shell out to ``sbatch``.

    Keys are free-form and become ``SBATCH_<KEY>`` in the launcher's environment, which is how
    sbatch takes them -- ``account`` -> ``SBATCH_ACCOUNT``, ``timelimit`` -> ``SBATCH_TIMELIMIT``.
    Not enumerated, so any option sbatch reads from the environment works without a schema change.
    See ``man sbatch``, INPUT ENVIRONMENT VARIABLES, for the list.

    A launcher exports these only where its own environment does not already set the variable, so
    precedence stays manifest -> script env var -> command line.
    """

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _keys_are_shell_identifiers(self) -> "Sbatch":
        """``cpus-per-task`` would become ``SBATCH_CPUS-PER-TASK``, which export rejects.

        The launcher exports these inside a process substitution, so the failure surfaces as a bare
        "not a valid identifier" with no mention of the manifest.
        """
        for key in self.model_extra or {}:
            if not key.replace("_", "").isalnum():
                raise ValueError(f"sbatch key '{key}' is not a valid shell identifier; use underscores")
        return self

    def env(self) -> Dict[str, str]:
        return {f"SBATCH_{key.upper()}": str(value) for key, value in (self.model_extra or {}).items()}


class Srun(BaseModel):
    """Container settings for the launchers, which shell out to ``srun --container-image``.

    The container belongs with the manifest because it is built *from* it: ``sweep container-config``
    unions every entry's ``config_paths`` and the resulting image has exactly this manifest's
    servers baked in. Running a manifest against an image built for a different one is the failure
    that presents as a hang rather than an error, so naming it here keeps the two together.

    ``MODEL`` deliberately stays a launcher argument: the checkpoint is what you are profiling, not
    part of what the sweep is.
    """

    model_config = ConfigDict(extra="forbid")

    # Reward-profiling sqsh, i.e. CONTAINER.
    container: Optional[str] = None
    # nemo-skills sandbox sqsh, i.e. SANDBOX_CONTAINER. Required by ns_tools and math_formal_lean.
    sandbox_container: Optional[str] = None
    # Passed through to --container-mounts.
    mounts: Optional[str] = None
    # Sandbox tier, i.e. SANDBOX_PORT / SANDBOX_WORKERS. One node only -- sessions pin to a worker
    # by X-Session-ID, so workers is the only axis that scales here.
    sandbox_port: Optional[int] = Field(default=None, ge=1)
    sandbox_workers: Optional[int] = Field(default=None, ge=1)

    def env(self) -> Dict[str, str]:
        pairs = {
            "CONTAINER": self.container,
            "SANDBOX_CONTAINER": self.sandbox_container,
            "MOUNTS": self.mounts,
            "SANDBOX_PORT": self.sandbox_port,
            "SANDBOX_WORKERS": self.sandbox_workers,
        }
        return {key: str(value) for key, value in pairs.items() if value is not None}


class Vllm(BaseModel):
    """What the launcher serves, i.e. ``vllm serve <model>``.

    ``model`` is also passed as ``++policy_model_name``, since the served name and the name Gym
    routes to have to agree. Optional: a manifest that pins a checkpoint is reproducible on its
    own, and leaving it out means the launcher requires MODEL, which is right when the same blend
    is profiled against many checkpoints.
    """

    model_config = ConfigDict(extra="forbid")

    # Served checkpoint path, i.e. MODEL.
    model: Optional[str] = None
    # The vLLM arg script the launcher sources, i.e. VLLM_CONFIG.
    config: Optional[str] = None
    # Prefill/decode split, i.e. NUM_PREFILL_NODES / NUM_DECODE_NODES. One job's nodes are the sum,
    # capped near 16 because --segment needs a topology-contiguous allocation and an NVL72 rack is
    # 18. Decode is the throughput-limiting side, so widen that one first.
    prefill_nodes: Optional[int] = Field(default=None, ge=1)
    decode_nodes: Optional[int] = Field(default=None, ge=1)
    # Concurrent sequences per decode engine, i.e. MAX_NUM_SEQS_PER_DECODE_ENGINE. The launcher
    # derives num_samples_in_parallel from it, so raising one without the other leaves capacity idle.
    max_num_seqs: Optional[int] = Field(default=None, ge=1)

    def env(self) -> Dict[str, str]:
        pairs = {
            "MODEL": self.model,
            "VLLM_CONFIG": self.config,
            "NUM_PREFILL_NODES": self.prefill_nodes,
            "NUM_DECODE_NODES": self.decode_nodes,
            "MAX_NUM_SEQS_PER_DECODE_ENGINE": self.max_num_seqs,
        }
        return {key: str(value) for key, value in pairs.items() if value is not None}


class GymEvalProfile(BaseModel):
    """Settings for ``gym eval profile``, emitted as ``++key=value``.

    Separate from ``gym_eval_run`` because they are different commands: ``allow_partial_rollouts``
    exists only on the profiler (``reward_profile.py:42``), so putting it under ``gym_eval_run``
    sends it to collection, where Hydra force-adds it and nothing reads it.
    """

    model_config = ConfigDict(extra="allow")

    # Keep groups that have some but not all of their repeats. True by default here because a
    # sweep this size is normally profiled before its tail finishes.
    allow_partial_rollouts: bool = True
    # Labels profiled concurrently, i.e. PROFILE_JOBS.
    jobs: Optional[int] = Field(default=None, ge=1)

    def env(self) -> Dict[str, str]:
        out = {"ALLOW_PARTIAL_ROLLOUTS": str(self.allow_partial_rollouts)}
        if self.jobs is not None:
            out["PROFILE_JOBS"] = str(self.jobs)
        return out

    def overrides(self) -> Dict[str, Any]:
        return dict(self.model_extra or {})


class SweepManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identifies this run and scopes every artifact it writes, so profiling the same blend
    # against a different checkpoint never collides.
    nickname: str
    # How many jobs the sweep is split across, i.e. NUM_SHARDS. Nodes = num_shards x
    # (vllm.prefill_nodes + vllm.decode_nodes). Leave unset for the launcher's default.
    num_shards: Optional[int] = Field(default=None, ge=1)

    materialize: Materialize = Field(default_factory=Materialize)
    sbatch: Sbatch = Field(default_factory=Sbatch)
    srun: Srun = Field(default_factory=Srun)
    vllm: Vllm = Field(default_factory=Vllm)
    gym_eval_profile: GymEvalProfile = Field(default_factory=GymEvalProfile)
    gym_env_start: GymEnvStart = Field(default_factory=GymEnvStart)
    gym_eval_run: GymEvalRun = Field(default_factory=GymEvalRun)

    entries: List[SweepEntry] = Field(min_length=1)

    @property
    def num_repeats(self) -> int:
        return self.gym_eval_run.num_repeats

    @model_validator(mode="after")
    def _random_sampling_needs_a_limit(self) -> "SweepManifest":
        """``random`` only selects when there is a limit to select down to.

        Without one it silently returns manifest order, and the counting pass still reads every
        file end to end to satisfy the exact count random would have needed -- 36.9 GB here, for a
        sample that never happens.
        """
        if self.materialize.sample == "random":
            if self.materialize.limit_per_entry is None and not any(e.limit for e in self.entries):
                raise ValueError(
                    "materialize.sample is 'random' but nothing sets a limit, so there is nothing "
                    "to sample down to. Set materialize.limit_per_entry or an entry's limit, or "
                    "drop sample."
                )
        return self

    @model_validator(mode="after")
    def _unique_labels(self) -> "SweepManifest":
        dupes = [label for label, count in Counter(e.label for e in self.entries).items() if count > 1]
        if dupes:
            raise ValueError(f"Duplicate entry labels: {sorted(dupes)}")
        return self

    def config_paths(self) -> List[str]:
        """Union of every entry's configs, de-duplicated, first-seen order preserved."""
        seen: Dict[str, None] = {}
        for entry in self.entries:
            for config in entry.configs:
                seen.setdefault(config, None)
        # Sweep-wide configs go last so they win the merge: several environments ship pointing their
        # judge at policy_model (the model grading itself), and rebinding that is the whole point of
        # a judges config.
        for config in self.gym_env_start.config_paths:
            seen.pop(config, None)
            seen[config] = None
        return list(seen)

    def num_repeats_by_agent(self) -> Dict[str, int]:
        """A ``num_repeats`` mapping for ``gym eval run``.

        Keyed by agent because that is what rollout collection resolves against, so entries
        sharing an agent necessarily share a repeat count. ``resolve_conflicts`` reports those
        collisions rather than silently letting the last entry win.
        """
        repeats: Dict[str, int] = {"_default": self.num_repeats}
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
    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SweepValidationError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SweepValidationError(f"Manifest must be a mapping, got {type(raw).__name__}: {path}")
    try:
        return SweepManifest.model_validate(raw)
    except ValidationError as exc:
        # Otherwise a one-key typo surfaces as a pydantic traceback from a CLI whose other errors
        # are one-liners, and the useful part is buried under a stack.
        problems = "\n".join(
            f"  {'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()
        )
        raise SweepValidationError(f"{path} is not a valid manifest:\n{problems}") from exc


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
        # An entry may list several configs: one declares the agent, the rest declare supporting
        # servers it dispatches to (ns_tools names its verifiers this way). So the agent has to be
        # declared by at least one of them, not by every one.
        declared: set[str] = set()
        found_all_configs = True
        for config in entry.configs:
            config_path = config if Path(config).is_absolute() else repo_root / config
            if not Path(config_path).is_file():
                errors.append(f"[{entry.label}] config not found: {config}")
                found_all_configs = False
                continue
            declared |= _declared_top_level_keys(Path(config_path))
        # effective_agent, not agent: agent_ref_override rewrites every row before dispatch, so
        # the override is the name that has to exist. Checking `agent` passes a typo'd override and
        # fails the whole entry at run time instead.
        if found_all_configs and entry.effective_agent not in declared:
            which = (
                f"agent_ref_override '{entry.agent_ref_override}'"
                if entry.agent_ref_override
                else f"agent '{entry.agent}'"
            )
            errors.append(
                f"[{entry.label}] {which} is not declared by any of its configs "
                f"({', '.join(entry.configs)}). Add the config that defines it, or fix the name."
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
