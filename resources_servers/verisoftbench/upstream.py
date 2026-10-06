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

import importlib
import logging
import re
import subprocess
import sys
import threading
import types
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


def require_pinned_checkout(upstream_dir: Path) -> None:
    """Fail closed unless ``upstream_dir`` is the audited upstream commit."""
    if not (upstream_dir / "evaluate.py").exists():
        raise UpstreamError(f"no upstream checkout at {upstream_dir}; run scripts/prepare_verisoftbench.py first")
    head = checkout_head(upstream_dir)
    if head != UPSTREAM_COMMIT:
        raise UpstreamError(f"upstream checkout at {upstream_dir} is at {head}, expected {UPSTREAM_COMMIT}")


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


def _stub_missing_provider_sdks() -> None:
    for name in _PROVIDER_STUBS:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            # Attributes the upstream ``from x import y`` statements resolve.
            module.genai = types.ModuleType("google.genai")
            module.types = types.ModuleType("google.genai.types")
            module.oauth2 = types.ModuleType("google.oauth2")
            module.service_account = types.ModuleType("google.oauth2.service_account")
            module.Anthropic = object
            sys.modules[name] = module


@dataclass
class UpstreamModules:
    utils: Any
    lean_interface: Any
    evaluator: Any
    prompt_builder: Any
    root: Path


_LOADED: Optional[UpstreamModules] = None


def load_upstream(upstream_dir: Path) -> UpstreamModules:
    """Put the pinned checkout on ``sys.path`` and import the modules the verifier uses."""
    global _LOADED
    upstream_dir = upstream_dir.resolve()
    if _LOADED is not None:
        if _LOADED.root != upstream_dir:
            raise UpstreamError(f"upstream already loaded from {_LOADED.root}; cannot also load {upstream_dir}")
        return _LOADED
    require_pinned_checkout(upstream_dir)
    for name in UPSTREAM_TOP_LEVEL_MODULES:
        present = sys.modules.get(name)
        if present is not None and not str(getattr(present, "__file__", "") or "").startswith(str(upstream_dir)):
            raise UpstreamError(f"module name {name!r} is already taken by {present!r}; cannot import upstream")
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    _stub_missing_provider_sdks()
    utils = importlib.import_module("utils.utils")
    lean_interface = importlib.import_module("core.lean_interface")
    evaluator = importlib.import_module("core.evaluator")
    prompt_builder = importlib.import_module("prompts.prompt_builder")
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


class Harness:
    """One task-independent handle on upstream's prompt builder and Lean interface."""

    def __init__(self, upstream_dir: Path, docker_container: str):
        self.modules = load_upstream(upstream_dir)
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
        the container, so it needs the Lean container running.
        """
        entry = dict(row)
        entry["thm_stmt"] = self.modules.evaluator._clean_thm_stmt(
            row["thm_stmt"], gt_proof=row.get("ground_truth_proof", "")
        )
        local_context = entry.get("local_ctxs") or entry.get("local_ctx")
        if local_context is None:
            raise KeyError("Entry must have either 'local_ctxs' or 'local_ctx' key")
        entry["verif_local_ctxs"] = self._shim._build_verif_context(
            entry["lean_root"],
            entry["rel_path"],
            entry["imports"],
            local_context,
            entry["thm_stmt"],
            entry["thm_name"],
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
        """Upstream's verify, plus the backend fault observed at the docker boundary, if any."""
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
        return success, error_msg, getattr(self._fault_slot, "fault", None)

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


def classify_transport_fault(result: "subprocess.CompletedProcess[str]") -> Optional[str]:
    """A backend-fault description from a finished ``docker exec`` result, else ``None``.

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
        return f"docker exec failed (exit code {result.returncode}): {line}"
    output = f"{result.stdout or ''}\n{stderr}"
    match = _LAKE_RESOLUTION_RE.search(output)
    if match is not None:
        line = output[match.start() :].splitlines()[0]
        return f"Lake re-resolved a dependency before elaboration: {line}"
    return None


def _observe_transport_faults(repl: Any, slot: threading.local) -> None:
    """Wrap ``repl``'s docker helpers so infrastructure faults keep typed provenance.

    Upstream's ``verify_proof`` ignores the write helper's exit code, maps read failures
    to ``None`` and formats compile output down to Lean-looking lines; a stopped
    container or daemon error would be charged to the candidate. The wrappers record the
    first fault seen by the calling thread into ``slot``; ``Harness._verify_proof``
    clears and reads it around each ``verify_proof`` call.
    """
    orig_exec = repl._docker_exec
    orig_write = repl._docker_write_file

    def record(message: str) -> None:
        if getattr(slot, "fault", None) is None:
            slot.fault = message

    def observed_exec(cmd: list, cwd: Path, timeout: int) -> "subprocess.CompletedProcess[str]":
        try:
            result = orig_exec(cmd, cwd, timeout)
        except FileNotFoundError as e:  # the docker binary itself is unavailable
            record(f"docker is not available on the host: {e}")
            raise
        fault = classify_transport_fault(result)
        if fault is not None:
            record(fault)
        return result

    def observed_write(container_path: str, content: str, timeout: int = 30) -> "subprocess.CompletedProcess[str]":
        result = orig_write(container_path, content, timeout)
        if result.returncode != 0:
            # The candidate file was never materialized; any verdict on it would be about
            # the backend, not the model.
            stderr = (result.stderr or "").strip()
            record(f"writing {container_path} in the container failed (exit code {result.returncode}): {stderr}")
        return result

    def observed_read(container_path: str, timeout: int = 30) -> Optional[str]:
        # Upstream's helper returns None for any nonzero ``cat`` and loses the result;
        # run the same command and keep that contract (a genuinely absent file has an
        # upstream fallback), but surface a docker transport failure as a backend fault.
        docker_cmd = ["docker", "exec", repl.docker_container, "cat", container_path]
        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            record(f"docker is not available on the host: {e}")
            raise
        fault = classify_transport_fault(result)
        if fault is not None:
            record(f"reading {container_path}: {fault}")
        return result.stdout if result.returncode == 0 else None

    repl._docker_exec = observed_exec
    repl._docker_write_file = observed_write
    repl._docker_read_file = observed_read
