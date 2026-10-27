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
"""Candidate generation.

The mutation operator is composition-aware: dG tracks base composition, which uniform mutation preserves,
so a uniform operator cannot move a free-energy gate.
"""

from __future__ import annotations

import contextlib
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from ..registry import tool
from ..spec import Cost, EnvironmentFault, field
from ..targets import TARGETS
from ._grnade_shim import _placeholder_pdb


BASES = "ACGU"
WEAK = "AU"  # lowers folding free energy toward the gate
STRONG = "GC"


@dataclass
class MutateIn:
    n: int = field("How many candidates to produce")
    target: str = field("Campaign target name")
    parents: list[str] = field(
        "Sequences to mutate. Empty means start from the wild type; or a selector: @all, @unscored, @top:N.",
        default_factory=list,
    )
    n_mutations: int = field("Mutations per candidate", default=8)
    weak_bias: float = field(
        "Fraction of mutations pushed toward A/U. 0 is uniform; higher lowers GC "
        "and raises free energy toward the gate. 1 only ever introduces A/U.",
        default=0.5,
    )
    seed: int = field("Random seed", default=0)


def _mutable_positions(target) -> list[int]:
    """0-indexed positions a design may change: not the prefix, not conserved."""
    conserved = {p - 1 for p in target.conserved}
    return [i for i in range(len(target.fixed_prefix), target.length) if i not in conserved]


def _random_sequence(target, rng: random.Random) -> str:
    """A random sequence of the target's length, honouring its fixed residues."""
    out = [rng.choice(BASES) for _ in range(target.length)]
    out[: len(target.fixed_prefix)] = list(target.fixed_prefix)
    for position, base in target.conserved.items():
        out[position - 1] = base  # conserved positions are 1-indexed
    return "".join(out)


def _mutate(parent: str, positions: list[int], k: int, weak_bias: float, rng: random.Random) -> str:
    out = list(parent)
    for index in rng.sample(positions, min(k, len(positions))):
        pool = WEAK if rng.random() < weak_bias else STRONG
        choices = [b for b in pool if b != out[index]] or [b for b in BASES if b != out[index]]
        out[index] = rng.choice(choices)
    return "".join(out)


@tool(
    name="generate.mutate",
    summary=(
        "Make candidates by mutating a parent sequence, holding the fixed prefix "
        "and the conserved residues. `weak_bias` steers mutations toward A/U, "
        "which lowers GC and raises EternaFold free energy."
    ),
    cost=Cost.CHEAP,
    model=MutateIn,
)
def generate_mutate(inp: MutateIn) -> dict:
    if inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    if inp.n < 1:
        raise ValueError(f"n must be >= 1, got {inp.n}")
    if not 0.0 <= inp.weak_bias <= 1.0:
        raise ValueError(f"weak_bias must be in [0, 1], got {inp.weak_bias}")

    target = TARGETS[inp.target]
    positions = _mutable_positions(target)
    parents = [p.strip().upper().replace("T", "U") for p in inp.parents]
    if not parents:
        # A de novo puzzle has no wild type, so start from a random sequence of the right length.
        parents = [target.wild_type] if target.wild_type else [_random_sequence(target, random.Random(inp.seed))]
    for parent in parents:
        if len(parent) != target.length:
            raise ValueError(f"parent length {len(parent)} != {target.length}")

    rng = random.Random(inp.seed)
    # Never emit the wild type itself. Keep drawing instead.
    results, seen = [], ({target.wild_type} if target.wild_type else set())
    attempts = 0
    while len(results) < inp.n and attempts < inp.n * 50:
        attempts += 1
        parent = parents[len(results) % len(parents)]
        candidate = _mutate(parent, positions, inp.n_mutations, inp.weak_bias, rng)
        changes = (
            sum(1 for a, b in zip(candidate, target.wild_type, strict=True) if a != b)
            if target.wild_type
            else inp.n_mutations
        )
        if candidate in seen or changes < target.min_changes_from_wt:
            continue
        seen.add(candidate)
        gc = sum(candidate.count(b) for b in STRONG) / len(candidate)
        results.append({"sequence": candidate, "changes_from_wt": changes, "gc": round(gc, 4)})
    if len(results) < inp.n:
        raise RuntimeError(
            f"produced only {len(results)} of {inp.n} distinct candidates in "
            f"{attempts} attempts; raise n_mutations or lower n"
        )
    return {
        "tool": "generate.mutate",
        "target": target.name,
        "n": len(results),
        "weak_bias": inp.weak_bias,
        "results": results,
    }


GRNADE_MAX_N = 5000
GRNADE_MAX_BATCH = 512


@dataclass
class GRNAdeIn:
    target: str = field("Campaign target name")
    structure: str = field("Target 3D structure (PDB). Empty uses $RDS_GRNADE_PDB.", default="")
    n: int = field(
        "How many sequences to draw. Named `n` like every other generator; gRNAde's own flag is total_samples.",
        default=1000,
    )
    batch_size: int = field(
        "How many to sample per forward pass. gRNAde loops "
        "`range(n // batch_size)`, so a batch larger than the draw samples "
        "nothing at all.",
        default=128,
    )
    n_pass: int = field("Stop once this many pass gRNAde's own filter", default=1000)
    temperature_min: float = field("Lowest sampling temperature", default=0.1)
    temperature_max: float = field("Highest sampling temperature", default=1.0)
    seed: int = field("Random seed", default=0)
    mode: str = field(
        "'3d' conditions on coordinates, '2d' on the secondary structure alone. "
        "Leave it empty: 3d is picked when the task supplies a backbone and 2d "
        "when it does not, and asking for 3d on a task that supplies no "
        "coordinates only fails. gRNAde is trained with 3D dropout for "
        "exactly that case.",
        default="",
    )
    filter_metric: str = field(
        "gRNAde's internal filter: openknot_score, sc_score_ribonanzanet, or sc_score_ribonanzanet_ss",
        default="sc_score_ribonanzanet_ss",
    )
    pass_threshold: float = field("Threshold for that filter", default=0.6)
    timeout_s: int = field("Abort a run that exceeds this", default=7200, operator=True)


#: gRNAde's own filters, as design.py names them.
FILTER_METRICS = ("openknot_score", "sc_score_ribonanzanet", "sc_score_ribonanzanet_ss")


def partial_sequence(target) -> str:
    """gRNAde's hard-constraint string: fixed letters, '_' where it may design, enforced during sampling."""
    out = ["_"] * target.length
    for i, base in enumerate(target.fixed_prefix):
        out[i] = base
    for position, base in target.conserved.items():
        out[position - 1] = base
    return "".join(out)


@tool(
    name="generate.grnade",
    summary=(
        "Design sequences for a target RNA 3D structure with gRNAde, a published "
        "geometric inverse-folding model. The campaign's fixed prefix and "
        "conserved residues are passed as hard sampling constraints, so every "
        "returned sequence already satisfies them."
    ),
    cost=Cost.GPU_BATCH,
    model=GRNAdeIn,
)
def generate_grnade(inp: GRNAdeIn) -> dict:
    import csv
    import os
    import subprocess
    import tempfile

    import yaml

    if inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    target = TARGETS[inp.target]
    # Arguments design.py would reject are the model's mistake, not the machine's, so refuse them here.
    if inp.mode not in ("", "2d", "3d"):
        raise ValueError(f"unknown mode {inp.mode!r}; this tool takes: '2d', '3d' or empty")
    if inp.filter_metric not in FILTER_METRICS:
        raise ValueError(f"unknown filter_metric {inp.filter_metric!r}; this tool takes: {', '.join(FILTER_METRICS)}")
    whole = {name: getattr(inp, name) for name in ("n", "batch_size", "n_pass", "seed")}
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in whole.values()):
        raise ValueError(f"n, batch_size, n_pass and seed must be whole numbers, got {whole}")
    if not 0 <= inp.seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {inp.seed}")
    if inp.n < 1 or inp.batch_size < 1:
        raise ValueError(f"n and batch_size must be at least 1, got n={inp.n} and batch_size={inp.batch_size}")
    # A request too large to finish would be masked as a machine fault, so the model's request size is capped.
    if inp.n > GRNADE_MAX_N or min(inp.batch_size, inp.n) > GRNADE_MAX_BATCH:
        raise ValueError(
            f"one call draws at most n={GRNADE_MAX_N} with batch_size<={GRNADE_MAX_BATCH}, "
            f"got n={inp.n} and batch_size={inp.batch_size}"
        )
    if not 0 < inp.temperature_min <= inp.temperature_max:
        raise ValueError(
            f"temperatures need 0 < temperature_min <= temperature_max, got {inp.temperature_min} and {inp.temperature_max}"
        )
    # design.py loops range(total_samples // n_samples), so a batch larger than n samples nothing.
    # Clamp it to n.
    notes = []
    if inp.batch_size > inp.n:
        notes.append(f"batch_size {inp.batch_size} exceeded n {inp.n}; used {inp.n}")
        inp = replace(inp, batch_size=inp.n)

    repo = Path(os.environ.get("GRNADE_REPO", "/opt/geometric-rna-design"))
    if not repo.exists():
        raise EnvironmentFault(f"gRNAde repo not found at {repo}; set GRNADE_REPO")
    work = Path(tempfile.mkdtemp(prefix="grnade_", dir=os.environ.get("RDS_RUN_DIR") or None))  # goes with the run
    # Use a backbone only when the task supplies one. An explicit structure argument still wins.
    # Test the string, not the path: Path("") is ".", which exists.
    disclosed = "backbone_3d" in getattr(target, "provides", frozenset())
    given = inp.structure or (
        getattr(target, "structure_path", "") or os.environ.get("RDS_GRNADE_PDB", "") if disclosed else ""
    )
    structure = Path(given) if given else Path("")
    # A backbone named by the task or the model means 3d, so a missing one fails instead of quietly running 2d.
    mode = inp.mode or ("3d" if given else "2d")
    if not given or not structure.exists():
        if mode != "2d" or inp.structure:
            missing = (
                f"target structure not found at {structure!r}; pass --structure, set RDS_GRNADE_PDB, or use mode='2d'"
            )
            # A path the model passed, or 3d asked of a target that supplies no backbone, is the model's mistake.
            # A backbone the operator should have provided is the machine's.
            raise ValueError(missing) if inp.structure or not disclosed else EnvironmentFault(missing)
        # 2d mode ignores coordinates, but design.py still loads a PDB for the sequence and length.
        structure = _placeholder_pdb(target, work / "placeholder.pdb")
    checkpoint = os.environ.get("RDS_GRNADE_CKPT", "checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5")

    config = {
        "pdb_filepath": str(structure),
        "target_sec_struct": target.target_structure,
        # An empty native_seq breaks gRNAde's featuriser. Use the placeholder's own sequence.
        "native_seq": target.wild_type or "A" * target.length,
        "partial_seq": partial_sequence(target),
        "mode": mode,
        "total_samples": inp.n,
        "n_samples": inp.batch_size,
        "n_pass": inp.n_pass,
        "temperature_min": inp.temperature_min,
        "temperature_max": inp.temperature_max,
        "filter_metric": inp.filter_metric,
        "pass_threshold": inp.pass_threshold,
        "output_dir": str(work / "designs"),
        "model_path": checkpoint,
        "node_in_dim": [15, 4],
        "node_h_dim": [128, 16],
        "edge_in_dim": [132, 3],
        "edge_h_dim": [64, 4],
        "num_layers": 4,
        "drop_rate": 0.5,
        "out_dim": 4,
        "radius": 0.0,
        "top_k": 32,
        "num_rbf": 32,
        "num_posenc": 32,
        "max_num_conformers": 1,
        # design.py requires these four. drop_prob_3d matches the drop3d@0.75 checkpoint.
        "seed": inp.seed,
        "gpu": 0,
        "noise_scale": 0.1,
        "drop_prob_3d": 0.75,
    }
    config_path = work / "design.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    # gRNAde needs PROJECT_PATH, X3DNA and ETERNAFOLD set to import, though only PROJECT_PATH is used.
    env = {**os.environ}
    # gRNAde writes inside its own checkout, which may be read-only, so mirror it as symlinks here.
    mirror = work / "repo"
    mirror.mkdir(parents=True, exist_ok=True)
    for entry in repo.iterdir():
        if entry.name != "checkpoints":
            with contextlib.suppress(OSError):
                (mirror / entry.name).symlink_to(entry)
    checkpoints = mirror / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if (repo / "checkpoints").is_dir():
        for entry in (repo / "checkpoints").iterdir():
            with contextlib.suppress(OSError):
                (checkpoints / entry.name).symlink_to(entry)
    env.setdefault("PROJECT_PATH", str(mirror))
    # Pure PyTorch stand-ins when torch_scatter or torch_cluster has no wheel for this platform.
    if subprocess.run(
        [sys.executable, "-c", "import torch_scatter, torch_cluster"], capture_output=True, env=env
    ).returncode:
        shims = Path(__file__).parent / "pyg_shims"
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(shims), env.get("PYTHONPATH", "")]))
    # design.py always loads RibonanzaNet and RibonanzaNet-SS from checkpoints/. Link mounted weights there.
    tools_weights = Path(os.environ.get("RDS_WEIGHTS", "/weights")) / "grnade_tools"
    for name in ("ribonanzanet", "ribonanzanet_sec_struct"):
        wanted = checkpoints / name
        if not wanted.exists() and (tools_weights / name).is_dir():
            with contextlib.suppress(OSError):
                wanted.symlink_to(tools_weights / name)
    # Let the allocator return reserved but unused blocks.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("X3DNA", str(work))
    env.setdefault("ETERNAFOLD", os.environ.get("RDS_ETERNAFOLD", "/opt/EternaFold"))

    limit = min(inp.timeout_s, float(os.environ.get("RDS_TOOL_TIMEOUT_S", inp.timeout_s)))
    try:
        result = subprocess.run(
            [sys.executable, "design.py", "--config", str(config_path)],
            cwd=str(mirror),
            capture_output=True,
            text=True,
            env=env,
            timeout=limit,
        )
    except subprocess.TimeoutExpired:
        # The model sized this request, so a run that outlasts the limit is its failed call, not a masked fault.
        raise ValueError(
            f"gRNAde did not finish within {limit:.0f} s for n={inp.n}, batch_size={inp.batch_size}; ask for fewer"
        ) from None
    produced = sorted((work / "designs").glob("designs_*.csv"))
    if result.returncode != 0 or not produced:
        raise EnvironmentFault(
            f"gRNAde exited {result.returncode} and wrote {len(produced)} files\n"
            f"--- stdout tail ---\n{result.stdout[-1200:]}\n"
            f"--- stderr tail ---\n{result.stderr[-1200:]}"
        )

    results, seen = [], set()
    with produced[-1].open() as handle:
        for row in csv.DictReader(handle):
            sequence = (row.get("sequence") or row.get("seq") or "").strip().upper()
            if not sequence or sequence in seen:
                continue
            seen.add(sequence)
            # A de novo target has nothing to count changes against.
            changes = (
                sum(1 for a, b in zip(sequence, target.wild_type, strict=True) if a != b) if target.wild_type else None
            )
            entry = {"sequence": sequence, "changes_from_wt": changes}
            for key, value in row.items():
                if key not in ("sequence", "seq"):
                    # Keep the numeric columns only.
                    with contextlib.suppress(TypeError, ValueError):
                        entry[key] = float(value)
            results.append(entry)

    if not results:
        # An empty file means gRNAde's own filter rejected every sample.
        drawn = (inp.n // inp.batch_size) * inp.batch_size
        raise RuntimeError(
            f"gRNAde wrote no designs: {drawn} samples drawn "
            f"({inp.n} // {inp.batch_size} batches of {inp.batch_size}), "
            f"none above {inp.filter_metric} >= {inp.pass_threshold}. The "
            f"filter is gRNAde's own, not this campaign's scoring: "
            f"pass_threshold=0 takes every sample and lets the scorers here "
            f"judge them. Otherwise lower it, raise n, or pick another "
            f"filter_metric."
        )
    return {
        "tool": "generate.grnade",
        "note": "; ".join(notes),
        "target": target.name,
        "mode": mode,
        "structure": str(structure),
        "n": len(results),
        "csv": str(produced[-1]),
        # design.py prints the device it chose, so a run that quietly fell back to CPU is visible in the result.
        "device": next(
            (line.split(":", 1)[1].strip() for line in result.stdout.splitlines() if line.startswith("Device:")), ""
        ),
        "results": results,
    }


#: Pairs a design may use, most stable first. GU wobble pairs are ordinary in measured structures.
PAIRINGS = (("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G"))
COMPLEMENTS = {"A": ("U",), "U": ("A", "G"), "G": ("C", "U"), "C": ("G",)}


@dataclass
class InverseFoldIn:
    n: int = field("How many candidates to produce")
    target: str = field("Campaign target name")
    gc_bias: float = field(
        "Fraction of pairs made G-C or C-G. Higher binds the target structure "
        "harder; lower leaves the design looser and raises free energy.",
        default=0.5,
    )
    seed: int = field("Random seed", default=0)


@tool(
    name="generate.inverse_fold",
    summary=(
        "Design sequences that can form the target's base pairs: every paired "
        "position gets a complementary partner, every free position a random "
        "base. Works on the pair set rather than on bracket nesting, so "
        "pseudoknots are handled like any other pair — which a nested-only "
        "inverse folder cannot do. Being able to form the pairs is not the "
        "same as folding into them, so the result still has to be scored."
    ),
    cost=Cost.CHEAP,
    model=InverseFoldIn,
)
def generate_inverse_fold(inp: InverseFoldIn) -> dict:
    from ..structure import to_pairs

    if inp.target not in TARGETS:
        raise ValueError(f"unknown target {inp.target!r} (have: {', '.join(sorted(TARGETS))})")
    if inp.n < 1:
        raise ValueError(f"n must be >= 1, got {inp.n}")
    if not 0.0 <= inp.gc_bias <= 1.0:
        raise ValueError(f"gc_bias must be in [0, 1], got {inp.gc_bias}")

    target = TARGETS[inp.target]
    if not target.target_structure:
        raise ValueError(f"target {target.name!r} declares no structure to fold into")
    pairs = sorted(to_pairs(target.target_structure))

    # Fixed positions. A fixed base's partner is chosen to complement it.
    fixed = dict(enumerate(target.fixed_prefix))
    fixed.update({position - 1: base for position, base in target.conserved.items()})

    rng = random.Random(inp.seed)
    results, seen = [], set()
    attempts = 0
    while len(results) < inp.n and attempts < inp.n * 50:
        attempts += 1
        out = [""] * target.length
        for i, j in pairs:
            if i in fixed and j in fixed:
                continue  # both ends given, leave them be
            if i in fixed:
                out[j] = rng.choice(COMPLEMENTS[fixed[i]])
            elif j in fixed:
                out[i] = rng.choice(COMPLEMENTS[fixed[j]])
            else:
                strong = PAIRINGS[:2] if rng.random() < inp.gc_bias else PAIRINGS[2:]
                out[i], out[j] = rng.choice(strong)
        for index in range(target.length):
            if index in fixed:
                out[index] = fixed[index]
            elif not out[index]:
                out[index] = rng.choice(BASES)
        candidate = "".join(out)
        if candidate in seen:
            continue
        seen.add(candidate)
        gc = sum(candidate.count(b) for b in STRONG) / len(candidate)
        results.append({"sequence": candidate, "gc": round(gc, 4), "pairs_designed": len(pairs)})
    if len(results) < inp.n:
        raise RuntimeError(
            f"produced only {len(results)} of {inp.n} distinct candidates in "
            f"{attempts} attempts; the target may leave too few free positions"
        )
    return {
        "tool": "generate.inverse_fold",
        "target": target.name,
        "n": len(results),
        "gc_bias": inp.gc_bias,
        "results": results,
    }
