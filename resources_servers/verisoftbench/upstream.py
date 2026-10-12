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

"""Drive the pinned upstream VeriSoftBench harness from NeMo Gym.

VeriSoftBench's verifier is not "call Lean": it is about two thousand lines of
repository-specific preprocessing in ``utils/utils.py``, ``core/lean_interface.py``
and ``core/evaluator.py`` (verification context rebuilt from the source file in the
container, name-conflict renaming, leaked-identifier renaming, ``mutual`` block
handling, per-repository patches) that upstream's published numbers were produced
with. Re-implementing it would build a different instrument, so this module imports
the upstream modules from a checkout pinned to the audited commit and calls them.
Nothing here re-specifies what a proof must do to count.

Upstream is MIT-licensed (utopia-group/VeriSoftBench, ``LICENSE``); its code is not
vendored into this repository but fetched at preparation time by
``scripts/prepare_verisoftbench.py``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import logging
import re
import subprocess
import sys
import threading
import types
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


LOG = logging.getLogger(__name__)

UPSTREAM_REPO_URL = "https://github.com/utopia-group/VeriSoftBench.git"
# HEAD of upstream ``main`` when the availability record was written (2026-09-16,
# "update leaderboard"). Re-verified 2026-09-22.
UPSTREAM_COMMIT = "d066523f2b087f28b793c05eacb29bf2e42c0504"  # pragma: allowlist secret
# ``data/verisoftbench.jsonl`` at that commit; byte-identical to the Hugging Face mirror
# maxRyeery/VeriSoftBench at revision 741c2a14e68a5293b70ef874ce8b7876ce053173.
DATASET_SHA256 = "ea584250021fbfaddc060834be2e05b9b2fae4f720c770a9932c71d039a50961"  # pragma: allowlist secret
DATASET_RELATIVE_PATH = "data/verisoftbench.jsonl"
EXPECTED_TASK_COUNT = 500

# The synthetic example tasks (``data/example.jsonl``) place their theorems in this file,
# which exists in no repository: they are deliberately absent from the repo index, so
# upstream's ``clean_leaked_identifiers`` finds no earlier same-file theorems for them —
# correct for out-of-corpus tasks. Benchmark tasks get no such exception:
# ``require_repo_index`` fails closed when their index data is missing or corrupt.
SYNTHETIC_EXAMPLE_REL_PATH = "GymExample.lean"

# Upstream's Docker layout (``core/lean_interface.py``: DOCKER_LEAN_REPOS_DIR).
DOCKER_LEAN_REPOS_DIR = Path("/workspace/lean_repos")
DOCKER_REPO_INDEX_DIR = "/workspace/repo_index"

# Top-level module names the upstream checkout occupies once it is on ``sys.path``.
UPSTREAM_TOP_LEVEL_MODULES = ("utils", "core", "prompts", "config", "analysis", "extraction")
# ``core/prover_interface.py`` imports these provider SDKs at module import time.
# ``core/evaluator.py`` imports it, and the two functions we need from the evaluator
# (``_clean_thm_stmt`` and ``BenchmarkEvaluator._build_verif_context``) do not touch a
# provider, so absent SDKs are stubbed rather than installed.
_PROVIDER_STUBS = (
    "google",
    "google.genai",
    "google.genai.types",
    "google.oauth2",
    "google.oauth2.service_account",
    "anthropic",
)

CONTEXT_MODE = "filtered_context"


GYM_ROOT = Path(__file__).resolve().parents[2]


def resolve_upstream_dir(value: str | Path) -> Path:
    """A relative ``upstream_dir`` is relative to the NeMo Gym repository root.

    Servers are started with the server directory as working directory, so a config
    value like ``cache/verisoftbench/...`` must not be resolved against the cwd.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else GYM_ROOT / path


class UpstreamError(RuntimeError):
    """The upstream checkout is missing, unpinned or unusable."""


def checkout_head(upstream_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise UpstreamError(f"{upstream_dir} is not a git checkout: {e}") from e


def require_clean_tracked_tree(upstream_dir: Path) -> None:
    """Fail closed when tracked upstream content differs from what its HEAD commits.

    A checkout can sit at the right HEAD with edited tracked files, so the commit check
    alone does not pin the instrument. Untracked and ignored artifacts — the extracted
    ``data/repo_index``, caches — are deliberately generated into the checkout and stay
    allowed; one ``git status`` call keeps this cheap enough to repeat on cached loads.
    """
    try:
        status = subprocess.run(
            ["git", "-C", str(upstream_dir), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise UpstreamError(f"cannot check tracked content of {upstream_dir}: {e}") from e
    if status.strip():
        dirty = ", ".join(line.strip() for line in status.strip().splitlines()[:5])
        raise UpstreamError(
            f"upstream checkout at {upstream_dir} has modified tracked files ({dirty}); "
            f"restore the pinned content of {UPSTREAM_COMMIT} before scoring"
        )


def require_pinned_checkout(upstream_dir: Path) -> None:
    """Fail closed unless ``upstream_dir`` is the audited upstream commit, unmodified."""
    if not (upstream_dir / "evaluate.py").exists():
        raise UpstreamError(f"no upstream checkout at {upstream_dir}; run scripts/prepare_verisoftbench.py first")
    head = checkout_head(upstream_dir)
    if head != UPSTREAM_COMMIT:
        raise UpstreamError(f"upstream checkout at {upstream_dir} is at {head}, expected {UPSTREAM_COMMIT}")
    require_clean_tracked_tree(upstream_dir)


def ensure_checkout(upstream_dir: Path) -> None:
    """Clone upstream at the pinned commit into ``upstream_dir`` if it is not already there."""
    if (upstream_dir / "evaluate.py").exists():
        require_pinned_checkout(upstream_dir)
        return
    upstream_dir.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Cloning %s into %s", UPSTREAM_REPO_URL, upstream_dir)
    subprocess.run(["git", "clone", "--quiet", UPSTREAM_REPO_URL, str(upstream_dir)], check=True)
    subprocess.run(["git", "-C", str(upstream_dir), "checkout", "--quiet", "--detach", UPSTREAM_COMMIT], check=True)
    require_pinned_checkout(upstream_dir)


def require_repo_index(modules: "UpstreamModules") -> str:
    """Fail closed unless the anti-leak repo index covers every benchmark repository.

    Upstream's ``clean_leaked_identifiers`` reads ``data/repo_index/<repo>.json`` and
    silently returns proof and lemmas unchanged when that file is missing, so a lost or
    never-copied index would weaken the acceptance protocol while every verdict still
    looks ordinary. This validates the exact directory upstream reads
    (``utils.REPO_INDEX_DIR``) against the repositories named in the pinned dataset and
    returns a sha256 over the required index files so a run can record which index it
    scored under. The synthetic example tasks (``SYNTHETIC_EXAMPLE_REL_PATH``) stay
    deliberately unindexed *inside* these validated files; nothing is weakened for them.
    """
    index_dir = Path(modules.utils.REPO_INDEX_DIR)
    dataset_path = modules.root / DATASET_RELATIVE_PATH
    try:
        tasks = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        required = sorted({task["lean_root"] for task in tasks})
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        raise UpstreamError(f"cannot read the benchmark's repositories from {dataset_path}: {e}") from e
    problems = []
    digest = hashlib.sha256()
    indexes: dict[str, dict] = {}
    for root in required:
        path = index_dir / f"{root}.json"
        try:
            data = path.read_bytes()
            theorems_by_file = json.loads(data).get("theorems_by_file")
        except OSError:
            problems.append(f"{path.name} is missing")
            continue
        except json.JSONDecodeError:
            problems.append(f"{path.name} is not JSON")
            continue
        if not isinstance(theorems_by_file, dict) or not theorems_by_file:
            problems.append(f"{path.name} has no 'theorems_by_file'")
            continue
        indexes[root] = theorems_by_file
        digest.update(f"{path.name}:{hashlib.sha256(data).hexdigest()}\n".encode())
    # A present, well-formed index can still be the wrong index: upstream's anti-leak
    # pass silently no-ops when the task's target theorem is not in its file's record
    # list, so every benchmark task's (lean_root, rel_path, thm_name) must resolve to a
    # record carrying the fields the pass reads (``name``/``aliases`` for the lookup,
    # ``line`` to filter earlier theorems). Up to five misses are named.
    uncovered = 0
    for task in tasks:
        theorems_by_file = indexes.get(task["lean_root"])
        if theorems_by_file is None:
            continue  # the whole index is already a named problem
        records = theorems_by_file.get(task.get("rel_path"), [])
        target = next(
            (
                record
                for record in records
                if isinstance(record, dict)
                and (record.get("name") == task.get("thm_name") or task.get("thm_name") in record.get("aliases", []))
            ),
            None,
        )
        if target is None or not isinstance(target.get("line"), int):
            uncovered += 1
            if uncovered <= 5:
                problems.append(
                    f"task {task.get('id')} ({task['lean_root']}/{task.get('rel_path')}: {task.get('thm_name')}) "
                    "has no usable index record"
                )
    if uncovered > 5:
        problems.append(f"... and {uncovered - 5} more uncovered tasks")
    if problems:
        raise UpstreamError(
            f"repo index at {index_dir} does not cover the benchmark: {'; '.join(problems)}. "
            "Upstream's leaked-identifier check would silently no-op; copy the generated index "
            "out of the container (README, Setup step 4) before scoring"
        )
    return digest.hexdigest()


def _stub_missing_provider_sdks() -> None:
    """Stub only the provider SDKs that are genuinely absent from the environment.

    ``importlib.util.find_spec`` proves absence without executing the SDK: an installed
    SDK whose own import raises must surface that failure to the caller's rollback, not
    be silently replaced by a stub. Stubs are built from a real ``ModuleSpec`` (as an
    empty package) so spec-based discovery of them does not crash, and parent stubs link
    their stubbed children so attribute access mirrors a real package.
    """
    stubbed: set[str] = set()
    for name in _PROVIDER_STUBS:  # parents precede their children in this tuple
        if name in sys.modules:
            continue
        if importlib.util.find_spec(name) is not None:
            continue
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        module = importlib.util.module_from_spec(spec)
        module.Anthropic = object  # resolves upstream's ``from anthropic import Anthropic``
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent in stubbed:
            setattr(sys.modules[parent], child, module)
        stubbed.add(name)


@dataclass
class UpstreamModules:
    utils: Any
    lean_interface: Any
    evaluator: Any
    prompt_builder: Any
    root: Path


_LOADED: Optional[UpstreamModules] = None


def _module_file_inside(module: Any, upstream_dir: Path) -> bool:
    """Exact path containment of ``module.__file__`` in the checkout, never a string prefix.

    A prefix test accepts a lookalike sibling such as ``<checkout>-evil/utils.py``;
    ``Path.is_relative_to`` only accepts real descendants of ``upstream_dir``.
    """
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    return Path(module_file).resolve().is_relative_to(upstream_dir)


def load_upstream(upstream_dir: Path) -> UpstreamModules:
    """Put the pinned checkout on ``sys.path`` and import the modules the verifier uses."""
    global _LOADED
    upstream_dir = upstream_dir.resolve()
    if _LOADED is not None:
        if _LOADED.root != upstream_dir:
            raise UpstreamError(f"upstream already loaded from {_LOADED.root}; cannot also load {upstream_dir}")
        # Importing modules does not lock Git HEAD: the checkout can be moved to another
        # (clean) commit after the first load, with the loaded Python still from the pin
        # while prompt templates and data are re-read from the moved tree. Recheck the
        # full pin — HEAD and tracked cleanliness — on every cached load.
        require_pinned_checkout(upstream_dir)
        return _LOADED
    require_pinned_checkout(upstream_dir)
    for name in UPSTREAM_TOP_LEVEL_MODULES:
        present = sys.modules.get(name)
        if present is not None and not _module_file_inside(present, upstream_dir):
            raise UpstreamError(f"module name {name!r} is already taken by {present!r}; cannot import upstream")
    # The import is transactional: any failure below — an upstream module, or an
    # installed provider SDK whose own import raises — restores ``sys.path`` and
    # ``sys.modules`` exactly, so no stub or half-imported package outlives a failed load.
    path_before = list(sys.path)
    modules_before = dict(sys.modules)
    try:
        if str(upstream_dir) not in sys.path:
            sys.path.insert(0, str(upstream_dir))
        _stub_missing_provider_sdks()
        utils = importlib.import_module("utils.utils")
        lean_interface = importlib.import_module("core.lean_interface")
        evaluator = importlib.import_module("core.evaluator")
        prompt_builder = importlib.import_module("prompts.prompt_builder")
        for module in (utils, lean_interface, evaluator, prompt_builder):
            if not _module_file_inside(module, upstream_dir):
                raise UpstreamError(
                    f"{module.__name__} was imported from {getattr(module, '__file__', None)}, "
                    f"outside the pinned checkout {upstream_dir}"
                )
    except BaseException:
        sys.path[:] = path_before
        for name in [name for name in sys.modules if name not in modules_before]:
            del sys.modules[name]
        for name, module in modules_before.items():
            if sys.modules.get(name) is not module:
                sys.modules[name] = module
        raise
    _LOADED = UpstreamModules(utils, lean_interface, evaluator, prompt_builder, upstream_dir)
    return _LOADED


class _EvaluatorShim:
    """The attributes ``BenchmarkEvaluator._build_verif_context`` reads from ``self``.

    The evaluator's constructor builds a provider client, so it cannot be instantiated
    without an API key. Its two verification-context methods only use these fields.
    """

    def __init__(self, evaluator_module: Any, lean_repl: Any, docker_container: str, lean_src_dir: Path):
        self.lean_repl = lean_repl
        self.docker_container = docker_container
        self.lean_src_dir = lean_src_dir
        self.logger = LOG
        self._read_source_file = types.MethodType(evaluator_module.BenchmarkEvaluator._read_source_file, self)
        self._build_verif_context = types.MethodType(evaluator_module.BenchmarkEvaluator._build_verif_context, self)


class PromptSource:
    """Upstream's prompt builder without a Lean container.

    The prompt-only face of this boundary: ``scripts/prepare_verisoftbench.py`` builds
    every task's curated-context prompt through it from the row alone, while ``Harness``
    constructs the same ``PromptBuilder`` plus the verification backend.
    """

    def __init__(self, upstream_dir: Path):
        modules = load_upstream(upstream_dir)
        self._builder = modules.prompt_builder.PromptBuilder(modules.root / "prompts" / "templates", mode=CONTEXT_MODE)

    def system_prompt(self) -> str:
        """``prompts/templates/init.txt``; upstream sends it on the initial and every repair call."""
        return self._builder.retrive_sys_prompt()

    def user_prompt(self, row: dict[str, Any]) -> str:
        return self._builder.build_user_prompt(theorem_entry=row, mode=CONTEXT_MODE)


class Harness:
    """One task-independent handle on upstream's prompt builder and Lean interface."""

    def __init__(self, upstream_dir: Path, docker_container: str):
        self.modules = load_upstream(upstream_dir)
        # The anti-leak protocol is part of the instrument: refuse to score without its
        # index, and record which index identity this harness verifies under.
        self.repo_index_digest = require_repo_index(self.modules)
        LOG.info("Repo index validated; sha256 over the required index files: %s", self.repo_index_digest)
        self.docker_container = docker_container
        # In Docker mode upstream uses this path only for path arithmetic.
        self.lean_repl = self.modules.lean_interface.LeanREPL(DOCKER_LEAN_REPOS_DIR, docker_container=docker_container)
        # Per-thread slot for backend faults observed at the docker exec boundary;
        # ``verify_candidate`` runs in worker threads, one compile per thread at a time.
        self._fault_slot = threading.local()
        _observe_transport_faults(self.lean_repl, self._fault_slot)
        self._shim = _EvaluatorShim(self.modules.evaluator, self.lean_repl, docker_container, DOCKER_LEAN_REPOS_DIR)
        self.prompt_builder = self.modules.prompt_builder.PromptBuilder(
            self.modules.root / "prompts" / "templates", mode=CONTEXT_MODE
        )

    # --- prompts ---------------------------------------------------------------

    def system_prompt(self) -> str:
        """``prompts/templates/init.txt``; upstream sends it on the initial and every repair call."""
        return self.prompt_builder.retrive_sys_prompt()

    def user_prompt(self, entry: dict[str, Any]) -> str:
        return self.prompt_builder.build_user_prompt(theorem_entry=entry, mode=CONTEXT_MODE)

    def repair_prompt(self, entry: dict[str, Any], last_proof: str, last_lemmas: str, last_err: str) -> str:
        """The user message of upstream's next repair round (``fix_single_theorem``, top of the loop)."""
        utils = self.modules.utils
        verif_ctx = entry["verif_local_ctxs"]
        add_imports = utils.get_add_imports_needed(entry, last_lemmas, last_proof, verif_ctx)
        ctx_for_prompt = add_imports + "\n" + verif_ctx if add_imports else verif_ctx
        return self.prompt_builder.build_fix_user_prompt(
            CONTEXT_MODE,
            entry,
            entry["thm_stmt"],
            last_err,
            last_proof,
            last_lemmas,
            ctx_for_prompt,
            compiled_file_path=None,
        )

    # --- task preparation ------------------------------------------------------

    def prepare_entry(self, row: dict[str, Any]) -> dict[str, Any]:
        """Upstream's per-task setup: clean the statement and build the verification context.

        Mirrors the top of ``evaluate_single_theorem``. Reads the task's source file from
        the container, so it needs the Lean container running. Upstream's
        ``_build_verif_context`` catches the read failure and silently falls back to
        ``imports + local_ctx`` — correct for a genuinely absent file (the synthetic
        example tasks), wrong for a transport fault: that fallback would be cached and
        every later candidate of the task scored against degraded context. A fault
        recorded during the build therefore refuses the entry instead of returning it.
        """
        entry = dict(row)
        entry["thm_stmt"] = self.modules.evaluator._clean_thm_stmt(
            row["thm_stmt"], gt_proof=row.get("ground_truth_proof", "")
        )
        local_context = entry.get("local_ctxs") or entry.get("local_ctx")
        if local_context is None:
            raise KeyError("Entry must have either 'local_ctxs' or 'local_ctx' key")
        self._fault_slot.fault = None
        entry["verif_local_ctxs"] = self._shim._build_verif_context(
            entry["lean_root"],
            entry["rel_path"],
            entry["imports"],
            local_context,
            entry["thm_stmt"],
            entry["thm_name"],
        )
        fault = self._resolve_fault(entry.get("lean_root"))
        if fault is not None:
            raise UpstreamError(
                f"building the verification context for task {entry.get('id')} hit a backend fault "
                f"({fault}); refusing to cache the degraded fallback context"
            )
        return entry

    # --- model output ----------------------------------------------------------

    def parse_initial(self, text: str) -> tuple[str, str, str]:
        """(proof, aux_lemmas, reasoning) from an initial-round response, as upstream reads them."""
        utils = self.modules.utils
        return (
            utils.get_proof_from_llm_output(text),
            utils.get_lemmas_from_llm_output(text),
            utils.get_reasoning_from_llm_output(text),
        )

    def parse_repair(self, text: str, last_proof: str, last_lemmas: str) -> tuple[str, str]:
        """(proof, aux_lemmas) from a repair-round response; a missing tag keeps the previous round's text."""
        utils = self.modules.utils
        lemmas = utils.get_fixed_lemmas_from_llm_output(text) or last_lemmas
        proof = utils.get_fixed_proof_from_llm_output(text) or last_proof
        return proof, lemmas

    # --- verification ----------------------------------------------------------

    def verify_candidate(
        self, entry: dict[str, Any], proof: str, lemmas: str, proof_id: str, max_conflict_attempts: int
    ) -> "VerificationOutcome":
        """Upstream's verification of one candidate: rename, compile, resolve name clashes, compile again.

        ``max_conflict_attempts`` is 1 on upstream's initial path (``verify_single_proof``)
        and 10 on its repair path (``fix_single_theorem``).
        """
        utils = self.modules.utils
        verif_ctx = entry["verif_local_ctxs"]
        if "verina" in str(entry["lean_root"]) and lemmas.strip():  # upstream keeps this for a repo not in the release
            lemmas = re.sub(r"lemma\b", "theorem", lemmas)
        name_mapping = utils.find_conflicting_names_from_local_context(verif_ctx, lemmas)
        lemmas, proof = utils.apply_name_replacements(lemmas, proof, name_mapping)
        while "axiom " in lemmas:
            lemmas = lemmas.replace("axiom ", "theorem ")
        proof, lemmas = utils.clean_leaked_identifiers(entry, proof, lemmas)
        success, error_msg, fault = self._verify_proof(entry, proof, lemmas, proof_id)
        attempts = 0
        while (
            fault is None
            and not success
            and error_msg
            and "has already been declared" in error_msg
            and attempts < max_conflict_attempts
        ):
            attempts += 1
            name_mapping = utils.find_conflicting_names_from_error(error_msg, lemmas)
            if not name_mapping:
                break
            lemmas, proof = utils.apply_name_replacements(lemmas, proof, name_mapping)
            while "axiom " in lemmas:
                lemmas = lemmas.replace("axiom ", "theorem ")
            proof, lemmas = utils.clean_leaked_identifiers(entry, proof, lemmas)
            success, error_msg, fault = self._verify_proof(entry, proof, lemmas, proof_id)
            if success or "has already been declared" not in error_msg:
                break
        if fault is not None and not success:
            # The backend, not the candidate, failed; carry the typed evidence instead of
            # upstream's formatted message (often reduced to the bare "Compilation failed").
            return VerificationOutcome(False, fault, proof, lemmas, attempts, backend_fault=True)
        return VerificationOutcome(success, None if success else error_msg, proof, lemmas, attempts)

    def _verify_proof(
        self, entry: dict[str, Any], proof: str, lemmas: str, proof_id: str
    ) -> tuple[bool, str, Optional[str]]:
        """Upstream's verify, plus the confirmed backend fault observed at the docker boundary."""
        self._fault_slot.fault = None
        success, error_msg = self.lean_repl.verify_proof(
            thm_name=entry["thm_name"],
            repo_name=entry["lean_root"],
            rel_path=entry["rel_path"],
            local_context=entry["verif_local_ctxs"],
            theorem_stmt=entry["thm_stmt"],
            theorem_proof=proof,
            proof_id=proof_id,
            aux_lemmas=lemmas,
            suffix=entry.get("suffix", ""),
        )
        return success, error_msg, self._resolve_fault(entry.get("lean_root"))

    # --- backend fault confirmation ----------------------------------------------

    def _resolve_fault(self, lean_root: Optional[str]) -> Optional[str]:
        """The calling thread's recorded fault, confirmed candidate-free, else ``None``.

        Compiled source is model-controlled, and ``#eval`` can print any marker-shaped
        line (and even choose the process exit code), so classifier suspicions are not
        evidence by themselves. Direct evidence (``kind == "direct"``: host exceptions,
        the write helper's exit code) stands on its own. A suspicion is confirmed only by
        probes that execute no candidate content: ``docker exec <container> true`` for
        transport, plus a fixed trivial compile in the task's repository for Lake
        re-resolution. An unconfirmed suspicion is dropped and the candidate stays
        charged — the residual risk is a sub-second transient that heals before the
        probe, which is preferred over a maskable-by-content scoring channel.
        """
        recorded = getattr(self._fault_slot, "fault", None)
        if recorded is None:
            return None
        self._fault_slot.fault = None
        kind, message = recorded
        if kind == "direct":
            return message
        transport_error = self._probe_container()
        if transport_error is not None:
            return f"{message} (confirmed candidate-free: {transport_error})"
        if kind == "lake" and lean_root is not None:
            probe_error = self._probe_lake_build(lean_root)
            if probe_error is not None:
                return f"{message} (confirmed candidate-free: {probe_error})"
        LOG.warning(
            "Suspected backend marker was not confirmed by candidate-free probes; charging the candidate: %s", message
        )
        return None

    def _probe_container(self) -> Optional[str]:
        """``None`` when ``docker exec <container> true`` succeeds; the failure otherwise."""
        try:
            result = subprocess.run(
                ["docker", "exec", self.docker_container, "true"], capture_output=True, text=True, timeout=30
            )
        except FileNotFoundError as e:
            return f"docker is not available on the host: {e}"
        except subprocess.TimeoutExpired:
            return "the container probe timed out"
        if result.returncode == 0:
            return None
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return f"container probe failed (exit code {result.returncode}): {detail[0] if detail else ''}"

    _PROBE_SOURCE = "theorem nemo_gym_backend_probe : True := trivial"
    _LEAN_ROOT_RE = re.compile(r"[A-Za-z0-9._-]+")

    def _probe_lake_build(self, lean_root: str) -> Optional[str]:
        """``None`` when a fixed trivial file compiles in the task's repository.

        The probe source is a constant: nothing model-controlled is executed, so a Lake
        dependency re-resolution failure it reproduces is genuine repository state.
        """
        if not self._LEAN_ROOT_RE.fullmatch(lean_root):
            return None  # not a pinned-dataset repository name; nothing to confirm with
        probe_path = f"/tmp/nemo_gym_probe_{uuid.uuid4().hex}.lean"
        script = (
            f"cd {DOCKER_LEAN_REPOS_DIR}/{lean_root} && printf '%s\\n' \"{self._PROBE_SOURCE}\" > {probe_path} && "
            f"lake env lean {probe_path}; rc=$?; rm -f {probe_path}; exit $rc"
        )
        try:
            result = subprocess.run(
                ["docker", "exec", self.docker_container, "bash", "-lc", script],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as e:
            return f"docker is not available on the host: {e}"
        except subprocess.TimeoutExpired:
            return "the trivial-compile probe timed out"
        if result.returncode == 0:
            return None
        detail = f"{result.stdout or ''}\n{result.stderr or ''}".strip().splitlines()
        return f"trivial-compile probe failed in {lean_root} (exit code {result.returncode}): {detail[0] if detail else ''}"

    # --- backend health --------------------------------------------------------

    def container_running(self) -> bool:
        try:
            out = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.docker_container],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return out.returncode == 0 and out.stdout.strip() == "true"


@dataclass
class VerificationOutcome:
    verified: bool
    error_msg: Optional[str]
    proof: str
    aux_lemmas: str
    conflict_resolution_attempts: int
    # The Lean backend, not the candidate, failed: typed evidence captured at the docker
    # exec boundary (``_observe_transport_faults``), never a substring of the formatted
    # compiler message — a diagnostic that merely contains marker-like text stays charged.
    backend_fault: bool = False


# --- model-free validation adapters -------------------------------------------------

# Body-start patterns upstream's ``verify_ground_truth.py`` (inline in ``verify_one``)
# uses to tell a real proof body from a ``ground_truth_proof`` that begins with the tail
# of the statement (rows mis-split at an inner ``let x :=``).
GOLD_BODY_STARTS = (r":=\s+by\b", r":=\s+match\b", r":=\s+calc\b", r":=\s+fun\b", r":=\s+λ\b", r":=\s+begin\b")
# Upstream's trailing-contamination strip for gold proofs, verbatim from
# ``verify_ground_truth.py``: block comments or new top-level declarations accidentally
# included during extraction appear after a blank line at column 0.
_GOLD_CONTAMINATION_RE = re.compile(
    r"\n\n(?=/--|/-|example\b|def\b|theorem\b|lemma\b|#|open\b|namespace\b|section\b).*", re.DOTALL
)


def strip_gold_contamination(ground_truth_proof: str) -> str:
    """Upstream's contamination strip on a shipped ground-truth proof, nothing else."""
    return _GOLD_CONTAMINATION_RE.sub("", ground_truth_proof.strip())


def normalize_gold_body(modules: UpstreamModules, ground_truth_proof: str) -> tuple[str, bool]:
    """Upstream's gold-body normalization: (body at the real separator, was_missplit).

    Validation-only adaptation, explicitly identified: upstream performs these steps
    inline in ``verify_ground_truth.verify_one`` on a whole task entry, so they are not
    importable as functions. This reproduces exactly the two body-only steps the
    model-free sweeps need — the contamination strip and the Case-2a mis-split recovery
    (a ``ground_truth_proof`` that duplicates the statement tail in front of the real
    proof) — and delegates the separator decision to upstream's own
    ``find_decl_body_separator``.
    """
    g = strip_gold_contamination(ground_truth_proof)
    if g.startswith(":=") and not any(re.match(pat, g) for pat in GOLD_BODY_STARTS):
        idx, pat = modules.utils.find_decl_body_separator(g)
        if idx > 0 and pat not in (":=", ":"):
            return g[idx:], True
    return g, False


@contextmanager
def no_rename_override(modules: UpstreamModules) -> Iterator[None]:
    """Replace upstream's ``clean_leaked_identifiers`` with the identity, scoped.

    The oracle and no-rename sweep forms of ``scripts/validate_model_free.py`` measure
    the instrument without the anti-leakage rename. That override is validation-only
    state on the shared upstream module, so it must not outlive the sweep: the original
    function is restored on exit, including on exception, and a later contract-mode
    sweep in the same process sees the pinned rename behavior again.
    """
    utils = modules.utils
    original = utils.clean_leaked_identifiers
    utils.clean_leaked_identifiers = lambda entry, proof, aux_lemmas: (proof, aux_lemmas)
    try:
        yield
    finally:
        utils.clean_leaked_identifiers = original


# --- backend fault provenance ------------------------------------------------------

# ``docker`` writes its own transport failures (unreachable daemon, missing or stopped
# container, exec that never started) to stderr at the start of a line. A compiler
# diagnostic that merely contains such text sits after a ``file.lean:line:col:``
# location, so a line-anchored match on the raw stderr of a nonzero ``docker exec``
# separates infrastructure evidence from elaborator output.
_DOCKER_TRANSPORT_RE = re.compile(
    r"^(?:Error response from daemon|Error: No such container|"
    r"Cannot connect to the Docker daemon|OCI runtime exec failed)",
    re.MULTILINE,
)
# Lake re-resolving a dependency inside the repository during ``lake env lean``: its own
# ``error:`` line with no source location, emitted before the candidate ever reached the
# elaborator.
_LAKE_RESOLUTION_RE = re.compile(r"^error: external command 'git' exited with code \d+", re.MULTILINE)


@dataclass
class TransportSuspicion:
    """Marker-shaped evidence from a ``docker exec`` result, pending confirmation.

    The compiled source is model-controlled and Lean's ``#eval`` can print any line to
    stdout or stderr (and even pick the process exit code), so no text or code in the
    result *proves* that docker or Lake produced it. A suspicion only becomes a backend
    fault after ``Harness._resolve_fault`` confirms it with a candidate-free probe.
    ``kind`` is ``"transport"`` (docker daemon/exec) or ``"lake"`` (dependency
    re-resolution before elaboration).
    """

    message: str
    kind: str


def classify_transport_fault(result: "subprocess.CompletedProcess[str]") -> Optional[TransportSuspicion]:
    """A backend-fault suspicion from a finished ``docker exec`` result, else ``None``.

    Works on the typed result (exit code plus raw stderr/stdout), not on upstream's
    formatted error message: ``compile_file`` keeps only lines that look like Lean
    diagnostics, so by the time it has formatted its output a daemon failure with no
    Lean source location has been reduced to the bare string "Compilation failed".
    A compile timeout is upstream's released protocol and stays charged.
    """
    if result.returncode == 0:
        return None
    stderr = result.stderr or ""
    match = _DOCKER_TRANSPORT_RE.search(stderr)
    if match is not None:
        line = stderr[match.start() :].splitlines()[0]
        return TransportSuspicion(f"docker exec failed (exit code {result.returncode}): {line}", "transport")
    output = f"{result.stdout or ''}\n{stderr}"
    match = _LAKE_RESOLUTION_RE.search(output)
    if match is not None:
        line = output[match.start() :].splitlines()[0]
        return TransportSuspicion(f"Lake re-resolved a dependency before elaboration: {line}", "lake")
    return None


def _observe_transport_faults(repl: Any, slot: threading.local) -> None:
    """Wrap ``repl``'s docker helpers so infrastructure faults keep typed provenance.

    Upstream's ``verify_proof`` ignores the write helper's exit code, maps read failures
    to ``None`` and formats compile output down to Lean-looking lines; a stopped
    container or daemon error would be charged to the candidate. The wrappers record the
    first fault seen by the calling thread into ``slot`` as ``(kind, message)``:
    ``"direct"`` evidence (a host-level exception, or the exit code of a command whose
    output the candidate cannot influence) is final, while ``classify_transport_fault``
    suspicions stay pending until ``Harness._resolve_fault`` confirms them with a
    candidate-free probe. ``Harness`` clears and reads the slot in the same thread
    around task preparation and each ``verify_proof`` call.
    """
    orig_exec = repl._docker_exec
    orig_write = repl._docker_write_file

    def record(kind: str, message: str) -> None:
        if getattr(slot, "fault", None) is None:
            slot.fault = (kind, message)

    def observed_exec(cmd: list, cwd: Path, timeout: int) -> "subprocess.CompletedProcess[str]":
        try:
            result = orig_exec(cmd, cwd, timeout)
        except FileNotFoundError as e:  # the docker binary itself is unavailable
            record("direct", f"docker is not available on the host: {e}")
            raise
        suspicion = classify_transport_fault(result)
        if suspicion is not None:
            record(suspicion.kind, suspicion.message)
        return result

    def observed_write(container_path: str, content: str, timeout: int = 30) -> "subprocess.CompletedProcess[str]":
        result = orig_write(container_path, content, timeout)
        if result.returncode != 0:
            # The candidate file was never materialized; any verdict on it would be about
            # the backend, not the model. The exit code of ``tee`` cannot be produced by
            # candidate content, so this is direct evidence.
            stderr = (result.stderr or "").strip()
            record(
                "direct", f"writing {container_path} in the container failed (exit code {result.returncode}): {stderr}"
            )
        return result

    def observed_read(container_path: str, timeout: int = 30) -> Optional[str]:
        # Upstream's helper returns None for any nonzero ``cat`` and loses the result;
        # run the same command and keep that contract (a genuinely absent file has an
        # upstream fallback), but surface a docker transport failure as a backend fault.
        # Unlike compile output, this command executes no candidate content: daemon/OCI
        # stderr from a failed read is authoritative, so it is DIRECT evidence — a later
        # healthy probe must not discard it, or a healed transient would cache degraded
        # verification context for every later candidate of the task.
        docker_cmd = ["docker", "exec", repl.docker_container, "cat", container_path]
        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            # Upstream's context builder catches this exception and falls back, so the
            # evidence must be recorded before re-raising: a read timeout or a host
            # launch failure is as candidate-free as daemon stderr. (This 30 s transport
            # timeout is distinct from the released 600 s compile timeout, which stays
            # charged.)
            record("direct", f"reading {container_path} failed on the host: {type(e).__name__}: {e}")
            raise
        suspicion = classify_transport_fault(result)
        if suspicion is not None:
            record("direct", f"reading {container_path}: {suspicion.message}")
        return result.stdout if result.returncode == 0 else None

    repl._docker_exec = observed_exec
    repl._docker_write_file = observed_write
    repl._docker_read_file = observed_read
