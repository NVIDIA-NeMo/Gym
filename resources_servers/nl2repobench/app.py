# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NL2RepoBench resources server."""

from __future__ import annotations

import hashlib
import re
import shlex
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from time import monotonic
from traceback import format_exc
from typing import Any, ClassVar, Literal

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, get_first_server_config_dict, is_nemo_gym_fastapi_entrypoint
from resources_servers.nl2repobench.guard_v2 import derive_identifiers, install_guard_v2, network_policy_for_v2
from resources_servers.nl2repobench.task_store import (
    EXPECTED_TASK_COUNT,
    NL2RepoBenchTaskStore,
    Task,
    task_id,
    task_image,
)


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]

# "official": today's behavior - open/restricted egress per enforce_agent_no_network, generic agent
#   image. Must stay byte-identical to reproduce already-completed eval runs.
# "no_internet_preinstalled": agent boots from the pinned per-task grading image (deps pre-installed)
#   with deny-all-except-policy_model egress; the real held-out test files are deleted from /workspace
#   before the agent gets a turn, to use the pre-installed deps without leaking tests.
# "block_target": open egress (so unrelated `pip install` still works), but pip/pip3/git are
#   replaced with wrappers that refuse to install/clone the specific task's own package/repo -
#   mitigates the observed "fetch the real upstream source and submit it" exploit without
#   losing general internet access.
# "block_target_v2": added 2026-10-20 (guard_v2.py). Same open-egress idea but with real package/GitHub identifiers,
#   subcommand-aware pip/git/curl/uv/... wrappers, interpreter (python/node) code inspection, and default-allow egress
#   with deny rules for source-hosting domains. "block_target" above is unchanged (backup of this file before the
#   change: app.py.bak_pre_guard_v2_20261020). See guard_v2.py and sft_runs/lightning_vahid1016/testset/LEAK_AUDIT.md.
AgentNetworkMode = Literal["official", "no_internet_preinstalled", "block_target", "block_target_v2"]

# "block_target" wrapper: installed in place of the real binaries (the real binary is moved
# aside to "<path>.nl2repobench-real" and exec'd from there), rather than relying on a
# PATH-prepend - this guarantees the guard runs regardless of how the sandbox's non-interactive
# shell resolves PATH, instead of depending on shell-specific PATH precedence/expansion rules.
# Substring-matches a normalized (lowercase, "_"->"-") task_id against every argv token, so it
# only catches tasks whose real PyPI/GitHub name is the task_id itself - a known, accepted gap
# (see docs/PIPELINE.md).
#
# Two templates:
# - _GUARD_SCRIPT_TEMPLATE: blanket argv substring match, used for pip/pip3/git/curl/wget. Safe
#   to blanket-match because these tools are essentially never used to run the agent's own tests
#   or legitimate work in a way that would coincidentally mention the target name.
# - _GUARD_SCRIPT_TEMPLATE_MODULE_AWARE: used for python/python3.x binaries and uv, which ARE
#   used constantly for legitimate work (running the agent's own tests, scripts, etc.) that
#   routinely mentions the target name incidentally (e.g. `python -m pytest tests/test_sklearn.py`
#   for task "sklearn" would false-positive-block under a blanket match). Only engages the
#   substring check when the invocation is shaped like a package-fetch: `python -m pip ...`,
#   `python -m uv ...`, or `uv pip ...` - confirmed from live transcripts as the actual bypass
#   shapes used (`python3.12 -m pip download <target>`, `uv pip install <target>.whl` etc.),
#   not a blanket "python touched this name anywhere" check.
_GUARD_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail
TARGET="{target}"
REAL="{real_path}"
norm() {{ echo "$1" | tr 'A-Z_' 'a-z-'; }}
for arg in "$@"; do
  a=$(norm "$arg")
  if [[ "$a" == *"$TARGET"* ]]; then
    echo "nl2repobench guard: blocked '$(basename "$REAL") $*' - matches guarded task target '$TARGET'" >&2
    exit 1
  fi
done
exec "$REAL" "$@"
"""

_GUARD_SCRIPT_TEMPLATE_MODULE_AWARE = """#!/usr/bin/env bash
set -euo pipefail
TARGET="{target}"
REAL="{real_path}"
TOOL_NAME="{tool_name}"
norm() {{ echo "$1" | tr 'A-Z_' 'a-z-'; }}
args=("$@")
is_pkg_fetch=false
for i in "${{!args[@]}}"; do
  a=$(norm "${{args[$i]}}")
  if [[ "$a" == "pip" && $i -gt 0 && "${{args[$((i-1))]}}" == "-m" ]]; then
    is_pkg_fetch=true
    break
  fi
  if [[ "$a" == "pip" && "$TOOL_NAME" == uv ]]; then
    is_pkg_fetch=true
    break
  fi
done
if $is_pkg_fetch; then
  for arg in "${{args[@]}}"; do
    a=$(norm "$arg")
    if [[ "$a" == *"$TARGET"* ]]; then
      echo "nl2repobench guard: blocked '$(basename "$REAL") $*' - matches guarded task target '$TARGET' (package-fetch shape)" >&2
      exit 1
    fi
  done
fi
exec "$REAL" "$@"
"""

# Removes the editable-install metadata of the reference repo from an agent sandbox (see _create_sandbox).
_METADATA_SCRUB_COMMAND = r"""python3 - <<'PYEOF'
import glob, os, re, shutil
IDENTS = set("__IDENTS__".split())
roots = set(glob.glob("/usr/local/lib/python3*/site-packages") + glob.glob("/usr/lib/python3*/dist-packages")
            + glob.glob("/usr/local/lib/python3*/dist-packages") + glob.glob("/usr/local/venv/lib/python3*/site-packages")
            + glob.glob("/openhands/micromamba/envs/*/lib/python3*/site-packages"))
removed = 0
for root in roots:
    for di in glob.glob(root + "/*.dist-info"):
        try:
            data = open(di + "/direct_url.json").read()
        except OSError:
            continue
        if "_reference_repo" not in data:
            continue
        record = di + "/RECORD"
        if os.path.exists(record):
            for line in open(record):
                path = os.path.normpath(os.path.join(root, line.split(",")[0]))
                if os.path.isfile(path) and not path.startswith(di):
                    os.remove(path)
                    removed += 1
        shutil.rmtree(di, ignore_errors=True)
        removed += 1
    for pattern in ("/*.pth", "/__editable__*"):
        for f in glob.glob(root + pattern):
            try:
                if os.path.isfile(f) and "_reference_repo" in open(f, errors="ignore").read():
                    os.remove(f)
                    removed += 1
            except OSError:
                pass
# Rust crates named after the target (e.g. pyca/bcrypt wraps the `bcrypt` crate whose source sits in the image's
# cargo registry) and shell/python history files from the image build
for pattern in ("/root/.cargo/registry/src/*/*", "/root/.cargo/registry/cache/*/*"):
    for path in glob.glob(pattern):
        crate = re.sub(r"-\d.*$", "", os.path.basename(path).lower())
        if re.sub(r"[-_.]", "", crate) in IDENTS:
            shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
            removed += 1
for f in ("/root/.python_history", "/root/.bash_history"):
    if os.path.exists(f):
        os.remove(f)
        removed += 1
print("scrubbed", removed)
PYEOF"""

_WORKSPACE_INSTALL_SCRUB_COMMAND = r"""python3 - <<'PYEOF'
import glob, os, re, shutil
IDENTS = set("__IDENTS__".split())
norm = lambda s: re.sub(r"[-_.]", "", s.lower())
removed = 0
for d in ("/root/.cache/pip", "/root/.cache/uv", "/tmp/pip-cache"):
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True); removed += 1
roots = set(glob.glob("/usr/local/lib/python3*/site-packages") + glob.glob("/usr/lib/python3*/dist-packages")
            + glob.glob("/usr/local/lib/python3*/dist-packages") + glob.glob("/usr/local/venv/lib/python3*/site-packages"))
for root in roots:
    for f in glob.glob(root + "/*.egg-link"):
        if "/workspace" in open(f, errors="ignore").read() or norm(os.path.basename(f)[:-9]) in IDENTS:
            os.remove(f); removed += 1
    for f in glob.glob(root + "/*.pth") + glob.glob(root + "/__editable__*"):
        if not os.path.isfile(f):
            continue
        lines = open(f, errors="ignore").read().splitlines(True)
        keep = [l for l in lines if "/workspace" not in l]
        if len(keep) != len(lines):
            if keep and f.endswith(".pth"):
                open(f, "w").writelines(keep)
            else:
                os.remove(f)
            removed += 1
    for di in glob.glob(root + "/*.dist-info") + glob.glob(root + "/*.egg-info"):
        name = norm(re.sub(r"-\d.*$", "", os.path.basename(di).rsplit(".", 1)[0]))
        try:
            du = open(di + "/direct_url.json").read()
        except OSError:
            du = ""
        if "file:///workspace" in du or name in IDENTS:
            rec = di + "/RECORD"
            if os.path.exists(rec):
                for line in open(rec):
                    p = os.path.normpath(os.path.join(root, line.split(",")[0]))
                    if os.path.isfile(p) and not p.startswith(di):
                        os.remove(p); removed += 1
            shutil.rmtree(di, ignore_errors=True); removed += 1
for b in glob.glob("/usr/local/bin/*"):
    try:
        if not os.path.isfile(b) or os.path.getsize(b) > 8192:
            continue
        t = open(b, errors="ignore").read()
    except OSError:
        continue
    mods = re.findall(r"^\s*(?:from|import)\s+([A-Za-z0-9_]+)", t, re.M) + re.findall(r"__requires__\s*=\s*['\"]([^'\"=<>]+)", t)
    if any(norm(m) in IDENTS for m in mods):
        os.remove(b); removed += 1
print("workspace-install scrub removed", removed)
PYEOF"""

# Timeout for the agent-side workspace collection tar command.
WORKSPACE_COLLECT_TIMEOUT_S = 300.0
# Per-command timeout for install/test commands run in the fresh verifier sandbox.
TEST_COMMAND_TIMEOUT_S = 1800.0
# Combined stdout+stderr for all test commands is capped to keep JSONL rows reasonably sized.
TEST_OUTPUT_MAX_CHARS = 50_000

_WORKSPACE_EXCLUDES = "--exclude=.git --exclude=__pycache__ --exclude=.venv --exclude=node_modules"
# GNU tar treats --exclude as positional once it appears after other
# positional args (-C DIR .) — it must precede them or it's silently ignored
# (older tar) or rejected outright (newer tar, exit code 2).
_COLLECT_COMMAND = f"tar czf /tmp/workspace.tar.gz {_WORKSPACE_EXCLUDES} -C /workspace ."

# Upstream NL2RepoBench runs the agent in a generic, task-agnostic OpenHands
# sandbox image - no pre-installed deps, no existing scaffold, nothing but
# start.md. The paper's grading image (task_image() below) is only ever
# pulled at grading time, layering the agent's finished workspace on top of
# it. Match that here: the agent gets this generic image, the verifier gets
# the real pinned one.
#
# Pulled from ghcr.io, not the upstream docker.all-hands.dev host: the latter
# isn't on this cluster's DNS allowlist (github.com/ghcr.io/Docker Hub all
# resolve; docker.all-hands.dev doesn't), so the OpenSandbox cluster's path to
# it goes through something slow enough that even a single, uncontended pull
# exceeded 30 minutes (confirmed via an isolated concurrency=1 probe).
# ghcr.io/all-hands-ai/runtime:0.56-nikolaik is the same image (~3.1GB,
# confirmed via registry manifest) on infrastructure this cluster already
# treats as fast - the per-task grading images already live there. A probe
# run against this reference completed sandbox creation and reached real
# model calls within ~8 minutes, with zero timeout/retry activity.
_AGENT_SANDBOX_IMAGE = "ghcr.io/all-hands-ai/runtime:0.56-nikolaik"

_PASSED_RE = re.compile(r"(\d+)\s+passed")
_FAILED_RE = re.compile(r"(\d+)\s+failed")
_ERROR_RE = re.compile(r"(\d+)\s+error")

# grading_protocol="upstream": NL2RepoBench's own grader (data/cache/source/openhands/post_processor.py). It deletes these
# agent-written files anywhere in the workspace (remove_package_files) and the agent's files at the task's test_files paths
# (remove_test_files), copies the rest over the grading image's /workspace (which still holds the real packaging and the
# held-out tests), sets PYTHONPATH=/workspace, runs every command without stopping, and sums the "N passed" lines of the
# pytest commands' output (analyze_pytest_results), capped at test_case_count.
_UPSTREAM_PACKAGE_FILES = (
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "tox.ini",
    "pytest.ini",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "environment.yml",
    "conda-env.yaml",
    "manifest.in",
    "MANIFEST.in",
)
_UPSTREAM_STAGE_DIR = "/tmp/nl2repobench_agent_workspace"
_UPSTREAM_PASSED_RE = re.compile(r"(\d+) passed")
_UPSTREAM_FAILED_RE = re.compile(r"(\d+) failed")
_UPSTREAM_ERROR_RE = re.compile(r"(\d+) error")
GradingProtocol = Literal["gym_strict", "upstream"]


def _upstream_restore_command(
    test_files: list[str],
    tar_path: str = "/tmp/workspace.tar.gz",
    workspace: str = "/workspace",
    stage: str = _UPSTREAM_STAGE_DIR,
) -> str:
    """Extract the agent tar to a staging dir, apply upstream's deletions, then overlay it onto the workspace.

    The overlay keeps whatever the grading image already has under /workspace (upstream's Dockerfile does
    `COPY workspace /workspace` on top of the same image), so the real setup.py/pyproject.toml and the held-out tests
    survive wherever the agent's copies were deleted.
    """
    names = " -o ".join(f"-name {shlex.quote(name)}" for name in _UPSTREAM_PACKAGE_FILES)
    test_paths = [
        f"{stage}/{path.strip('/')}"
        for path in test_files
        if path.strip("/") and ".." not in path.strip("/").split("/")
    ]
    remove_tests = f"rm -rf {' '.join(shlex.quote(p) for p in test_paths)}; " if test_paths else ""
    return (
        f"set -e; rm -rf {stage}; mkdir -p {stage}; tar xzf {shlex.quote(tar_path)} -C {stage}; "
        f"find {stage} -type f \\( {names} \\) -delete; {remove_tests}"
        f"tar -C {stage} -cf - . | tar -C {shlex.quote(workspace)} -xf -"
    )


def _parse_pytest_upstream(pytest_outputs: list[str]) -> tuple[int, int, int]:
    """Sum the first "N passed/failed/error" match of every line of every pytest command's output, like upstream."""
    passed = failed = errors = 0
    for output in pytest_outputs:
        for line in output.split("\n"):
            if match := _UPSTREAM_PASSED_RE.search(line):
                passed += int(match.group(1))
            if match := _UPSTREAM_FAILED_RE.search(line):
                failed += int(match.group(1))
            if match := _UPSTREAM_ERROR_RE.search(line):
                errors += int(match.group(1))
    return passed, failed, errors


def _resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (NEMO_GYM_ROOT / expanded).resolve()


class NL2RepoBenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    tasks_dir: Path
    expected_task_count: int = Field(default=EXPECTED_TASK_COUNT, ge=1)

    sandbox_provider: str
    enforce_agent_no_network: bool = True
    sandbox_model_server: ModelServerRef | None = None
    agent_network_mode: AgentNetworkMode = "official"
    # None (default): verify sandbox uses the real pinned per-task image (task_image(task)),
    # exactly like upstream. Set to a generic image (e.g. _AGENT_SANDBOX_IMAGE) for a task set
    # with no pinned, deps-preinstalled grading image of its own - test_commands must then
    # install deps itself (e.g. `pip install -e .`) since the image won't have them pre-baked.
    verify_image_override: str | None = None
    # None (default): task_image(task) resolves against upstream's own
    # ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0 registry. Set to a format string
    # with a {name} placeholder (e.g. "ghcr.io/<org>/nl2repobench-vahid1016/{name}:1.0") for a
    # task set with its OWN pre-built, deps-preinstalled per-task images under a different
    # registry namespace - required to use agent_network_mode="no_internet_preinstalled" with
    # any task set other than upstream's 104 pinned tasks.
    task_image_registry_template: str | None = None
    # no_internet_preinstalled only: also scrub installs that point INTO /workspace (images that ship the reference repo
    # under /workspace with a legacy `setup.py develop` / egg-link or editable install, e.g. task images built from the finished reference repo),
    # their console-script stubs, and pip caches. Off by default so existing eval behavior is byte-identical. Added 2026-10-24.
    scrub_workspace_installs: bool = False
    # no_internet_preinstalled only: grade in verify_image_override instead of the agent's task image. Needed when the task
    # image ships the reference repo under /workspace (the verifier extracts the agent workspace over /workspace without
    # wiping it). Off by default. Added 2026-10-24.
    verify_with_override_image: bool = False
    # "gym_strict" (default, original behavior): grade the agent's workspace as-is - its own setup.py/pyproject.toml and any
    # files it wrote at the held-out test paths stay, and the first failing non-final command zeroes the row.
    # "upstream": grade the way NL2RepoBench itself does (see _UPSTREAM_PACKAGE_FILES). Added 2026-10-27, after finding
    # that about half of all gym_strict rows scored 0 at `pip install -e .` on the agent's own packaging files, which
    # upstream deletes, and that agent-written tests at the held-out paths inflated some rows.
    grading_protocol: GradingProtocol = "gym_strict"
    sandbox_config: dict[str, Any]

    logs_dir: Path = Path("resources_servers/nl2repobench/logs")
    clear_verifier_logs: bool = True
    include_workspace_in_response: bool = False
    workspace_tar_max_bytes: int = 209_715_200
    # None (default): the collected workspace tar is never written to disk - it only exists
    # in-memory for the duration of a single verify() call, between _collect_workspace and
    # _run_verifier, and is discarded once that call returns (confirmed the hard way: every v2
    # eval run's generated code is unrecoverable, only workspace_sha256/workspace_bytes survive
    # in the results JSONL). Set this to persist each rollout's generated workspace as
    # <save_workspace_dir>/<task_id>/<session_id>.tar.gz, so a *future* grading-mechanism change
    # (a different test_commands.json, a different verify image, etc.) can re-verify against the
    # already-generated code without re-running the agent/LLM at all.
    save_workspace_dir: Path | None = None


class NL2RepoBenchInstanceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    image: str
    verifier_metadata: dict[str, Any] | None = None


class NL2RepoBenchSeedSessionRequest(NL2RepoBenchInstanceRequest, BaseSeedSessionRequest):
    pass


class NL2RepoBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]
    workdir: str | None = None


class NL2RepoBenchVerifyRequest(NL2RepoBenchInstanceRequest, BaseVerifyRequest):
    sandbox_handle: str | None = None


class NL2RepoBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    evaluation_completed: bool
    verifier_exit_code: int | None = None
    verifier_error: str | None = None

    tests_passed: int = 0
    tests_failed: int = 0
    tests_error: int = 0
    test_case_count: int = 0
    success_rate: float = 0.0

    workspace_bytes: int | None = None
    workspace_sha256: str | None = None
    workspace_tar: str | None = None
    test_output: str | None = None

    log_dir: str
    workspace_collection_time_s: float
    sandbox_start_time_s: float
    verification_time_s: float


class VerifierResult(BaseModel):
    evaluation_completed: bool
    reward: float
    verifier_exit_code: int | None = None
    verifier_error: str | None = None
    test_output: str | None = None
    tests_passed: int = 0
    tests_failed: int = 0
    tests_error: int = 0
    test_case_count: int = 0
    success_rate: float = 0.0


@dataclass
class AgentSandboxSession:
    task_id: str
    image: str
    sandbox: AsyncSandbox
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]


def _resolve_task_id(body: NL2RepoBenchInstanceRequest) -> str:
    metadata_task_id = (body.verifier_metadata or {}).get("task_id")
    if body.task_id and metadata_task_id and body.task_id != metadata_task_id:
        raise ValueError(
            f"Conflicting NL2RepoBench task IDs: task_id={body.task_id!r}, verifier_metadata.task_id={metadata_task_id!r}"
        )
    current_task_id = body.task_id or metadata_task_id
    if not isinstance(current_task_id, str) or not current_task_id:
        raise ValueError("NL2RepoBench requests must provide verifier_metadata.task_id or task_id")
    return current_task_id


def _resolve_task(
    body: NL2RepoBenchInstanceRequest,
    task_store: NL2RepoBenchTaskStore,
    *,
    registry_template: str | None = None,
) -> Task:
    requested_task_id = _resolve_task_id(body)
    task = task_store.get(requested_task_id)
    if body.image != task_image(task, registry_template=registry_template):
        raise ValueError(f"NL2RepoBench request image does not match the pinned image for task {requested_task_id!r}")
    return task


def _parse_pytest_summary(combined_output: str) -> tuple[int, int, int]:
    """Return (passed, failed, error) from the LAST matching pytest summary occurrences."""

    passed_matches = _PASSED_RE.findall(combined_output)
    failed_matches = _FAILED_RE.findall(combined_output)
    error_matches = _ERROR_RE.findall(combined_output)
    tests_passed = int(passed_matches[-1]) if passed_matches else 0
    tests_failed = int(failed_matches[-1]) if failed_matches else 0
    tests_error = int(error_matches[-1]) if error_matches else 0
    return tests_passed, tests_failed, tests_error


def _compute_reward(tests_passed: int, test_case_count: int) -> float:
    if test_case_count > 0:
        reward = min(tests_passed / test_case_count, 1.0)
    else:
        reward = 0.0
    return max(reward, 0.0)


class NL2RepoBenchResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: NL2RepoBenchResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._task_store = NL2RepoBenchTaskStore(
            _resolve_repo_path(self.config.tasks_dir),
            expected_task_count=self.config.expected_task_count,
        )
        self._agent_sessions: dict[str, AgentSandboxSession] = {}

    def _provider_options(self, *, phase: str) -> dict[str, Any]:
        options = deepcopy(self.config.sandbox_config.get("provider_options", {}))
        mode = self.config.agent_network_mode
        if phase != "agent":
            # Non-agent phases (verifier) never restrict network.
            options.pop("network_policy", None)
            return options
        if mode == "official" and not self.config.enforce_agent_no_network:
            # upstream's agent has real internet access (it must pip install whatever the
            # generic image doesn't already have), so leave provider_options' network_policy
            # untouched/absent rather than forcing a deny-all-plus-one-allow-rule policy
            # regardless of this flag (the previous bug here - the model-egress branch below
            # used to run unconditionally).
            options.pop("network_policy", None)
            return options
        if mode == "block_target_v2":
            # Default-allow egress plus deny rules for source-hosting domains (guard_v2.DENY_EGRESS_TARGETS); the
            # pip/git/curl/python wrappers installed in _create_sandbox do the package-aware part.
            options["network_policy"] = network_policy_for_v2(options.get("network_policy"))
            return options
        if mode == "block_target":
            # Open egress, same as official-with-no-enforcement: the pip/pip3/git guard
            # installed in _create_sandbox() is what actually restricts this mode, not the
            # network layer (OpenSandbox's network_policy is host-based, not package-aware).
            options.pop("network_policy", None)
            return options
        # mode == "no_internet_preinstalled", or mode == "official" with
        # enforce_agent_no_network=True: deny-all except the policy_model host below.

        model_egress_target = self._model_egress_target()
        options.setdefault("network_policy", {"defaultAction": "deny", "egress": []})
        if model_egress_target is not None:
            network_policy = options["network_policy"]
            if not isinstance(network_policy, dict):
                raise TypeError("NL2RepoBench sandbox network_policy must be a mapping")
            egress = network_policy.setdefault("egress", [])
            if not isinstance(egress, list):
                raise TypeError("NL2RepoBench sandbox network_policy.egress must be a list")
            model_rule = {"action": "allow", "target": model_egress_target}
            if model_rule not in egress:
                egress.append(model_rule)

        return options

    def _model_egress_target(self) -> str | None:
        if self.config.sandbox_model_server:
            model_config = get_first_server_config_dict(
                get_global_config_dict(),
                self.config.sandbox_model_server.name,
            )
            target = str(model_config.get("host") or "")
            if not target:
                raise ValueError(f"Model server {self.config.sandbox_model_server.name!r} does not have a host")
        else:
            return None

        if target in {"0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}:
            raise ValueError(
                f"NL2RepoBench task sandboxes cannot reach loopback model host {target!r}; "
                "set NEMO_GYM_SANDBOX_MODEL_BASE_URL or launch Gym with use_absolute_ip=true"
            )
        return target

    async def _create_sandbox(self, task: Task, *, phase: str) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)

        current_task_id = task_id(task)
        mode = self.config.agent_network_mode
        resources = dict(self.config.sandbox_config.get("resources", {}))
        # "no_internet_preinstalled" boots the agent from the pinned per-task grading image too
        # (deps pre-installed), instead of the generic image - the held-out test files it also
        # ships get stripped out below, before the agent gets a turn.
        agent_image = (
            task_image(task, registry_template=self.config.task_image_registry_template)
            if mode == "no_internet_preinstalled"
            else _AGENT_SANDBOX_IMAGE
        )
        # verify_image_override: for task sets with no pinned, deps-preinstalled grading image
        # (e.g. a custom held-out split derived outside upstream NL2RepoBench's 104), point the
        # verifier sandbox at a generic image instead and let its test_commands install deps
        # fresh (pip install -e . etc.) at verify time. None (default) reproduces upstream
        # behavior exactly - the real pinned per-task image, no network needed.
        #
        # mode == "no_internet_preinstalled" always uses the real preinstalled per-task image for
        # verification too, IGNORING verify_image_override - mirrors agent_image's own mode
        # branch above exactly, and is safe to do unconditionally in that branch since
        # test_commands.json's clone+install step (see
        # sft_runs/lightning_vahid1016/testset/make_test_commands_preinstall_aware.py) is
        # idempotent: it detects /tmp/_reference_repo/.git already existing and skips straight to
        # the copy+pytest steps, rather than re-cloning into (and failing against) an
        # already-populated directory. For every OTHER mode, verify_image_override is left as the
        # deciding factor exactly as before - this keeps block_target/official verify behavior
        # byte-identical to already-reported eval baselines, since those don't touch this branch.
        verify_image = (
            task_image(task, registry_template=self.config.task_image_registry_template)
            if mode == "no_internet_preinstalled" and not (self.config.verify_with_override_image and self.config.verify_image_override)
            else self.config.verify_image_override
            or task_image(task, registry_template=self.config.task_image_registry_template)
        )
        spec = SandboxSpec(
            image=agent_image if phase == "agent" else verify_image,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir="/workspace",
            env=dict(self.config.sandbox_config.get("env", {})),
            # NOT using SandboxSpec.files here: it uploads into the container
            # AFTER the image's own entrypoint has already started (AsyncSandbox.start()
            # calls provider.create() first, then uploads), and `workdir` above is only
            # ever used later as an exec cwd - it never creates the directory. So
            # `files` silently drops anything targeting a path whose parent doesn't
            # already exist in the booted image. The per-task pinned images happen to
            # have /workspace pre-created by their own Dockerfile WORKDIR, which made
            # this work before switching the agent sandbox to a generic image whose
            # own WORKDIR is elsewhere (e.g. /openhands/code) and never creates
            # /workspace at all - confirmed via `find / -name start.md` finding
            # nothing in that container. start.md is written explicitly below instead,
            # after an explicit `mkdir -p`, so it doesn't depend on the image's layout.
            files={},
            metadata=provider_metadata
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {
                "benchmark": "nl2repobench-v1",
                "nl2repobench-task": current_task_id[:63],
                "nl2repobench-phase": phase,
                "nemo_gym_agent": self.config.name or "nl2repobench",
            },
            resources=SandboxResources.from_mapping(resources),
            provider_options=self._provider_options(phase=phase),
        )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(spec)
        if phase == "agent":
            # Upstream NL2RepoBench copies start.md into the task workspace at launch
            # (shutil.copy2 onto a host bind-mount); this is our equivalent for the
            # sandbox-provider path. mkdir first (see the comment on `files=` above for
            # why SandboxSpec.files alone silently drops this on images that don't
            # already have /workspace), then upload via the dedicated file-transfer API
            # rather than inlining content into a sandbox.exec() shell command: some
            # start.md specs run up to ~375KB, and the opencode_sandboxed_agent install
            # path hit a real `fork/exec ... argument list too long` failure the last
            # time large NL2RepoBench content was inlined into an exec command string
            # (see benchmarks/nl2repobench/opencode.yaml) - upload() avoids that
            # class of failure entirely rather than relying on staying under some
            # untested ARG_MAX threshold.
            # cwd="/" is required here, not just defensive: exec()'s default cwd is
            # SandboxSpec.workdir ("/workspace" above), and at least one provider
            # (OpenSandbox) validates the cwd exists before running the command at
            # all - so letting this call default to /workspace, the very directory
            # it's trying to create, fails with "working directory does not exist"
            # before mkdir ever runs. "/" is guaranteed to exist in any image.
            mkdir_result = await sandbox.exec(command="mkdir -p /workspace", cwd="/", timeout_s=60)
            if mkdir_result.return_code != 0:
                raise RuntimeError(
                    f"Failed to create /workspace in agent sandbox for {task_id(task)!r}: "
                    f"{mkdir_result.stderr or mkdir_result.stdout or '(no output)'}"
                )
            if mode == "no_internet_preinstalled":
                # agent_image above is task_image(task), which also ships this task's real
                # held-out tests AND the real repo's own root-level packaging/scaffolding files
                # (setup.py, pyproject.toml, requirements*.txt, .pytest_cache/, ...) - confirmed
                # live: deleting only task.test_files.files left setup.py/pyproject.toml/
                # requirements.txt sitting in /workspace across most tasks (e.g. icecream's real
                # setup.py), leaking real dependency versions/package metadata the agent was
                # never supposed to see. Wipe /workspace entirely instead of targeting the known
                # test-file list. Must happen before the agent's session is usable (seed_session
                # can't return until this function returns, so there's no window where the agent
                # could read any of it first).
                #
                # /tmp/_reference_repo ALSO wiped, not just /workspace: our own per-task images
                # (sft_runs/lightning_vahid1016/testset/task_images/) install the reference repo
                # with `pip install -e .` (an EDITABLE install - deliberately matching what
                # test_commands.json itself does at verify time), which leaves the real
                # implementation AND the real held-out tests/ sitting in /tmp/_reference_repo,
                # readable in full by the agent, and importable as the real package via Python's
                # normal import machinery - a complete answer leak, strictly worse than the
                # /workspace case above. Deleting it does NOT break any genuinely-needed
                # pre-installed dependency: only this ONE package's own (editable) install
                # resolves through /tmp/_reference_repo; every other third-party dependency the
                # task needs was installed normally into system site-packages (e.g.
                # /usr/local/lib/python3.*/site-packages), unaffected by this deletion. The agent
                # losing the ability to `import <the task's own package>` is the point, not a
                # side effect to work around - it is never supposed to have the real
                # implementation available at all.
                wipe_result = await sandbox.exec(
                    command="find /workspace -mindepth 1 -delete && rm -rf /tmp/_reference_repo",
                    cwd="/",
                    timeout_s=60,
                )
                if wipe_result.return_code != 0:
                    raise RuntimeError(
                        f"Failed to wipe pre-existing /workspace/_reference_repo contents in agent sandbox for "
                        f"{task_id(task)!r}: {wipe_result.stderr or wipe_result.stdout or '(no output)'}"
                    )
                # Metadata scrub (added 2026-10-21 after a leak scan of the real per-task images, see
                # sft_runs/lightning_vahid1016/testset/LEAK_AUDIT.md): the editable install of the reference repo
                # leaves, outside /tmp/_reference_repo, the package's dist-info (METADATA embeds the README with code
                # examples + version + Home-page URL + dependency list; `pip show` prints them), its console-script
                # stubs in bin/ (reveal the module layout) and the .pth/__editable__ pointer. Remove every dist-info
                # whose direct_url.json points at /tmp/_reference_repo together with the files its RECORD lists, and
                # any leftover .pth/__editable__ file naming it. The verifier uses its own fresh sandbox, unaffected.
                scrub_result = await sandbox.exec(
                    command=_METADATA_SCRUB_COMMAND.replace(
                        "__IDENTS__", " ".join(derive_identifiers(task_id(task), task.test_commands.commands)[0])
                    ),
                    cwd="/",
                    timeout_s=120,
                )
                if scrub_result.return_code != 0:
                    raise RuntimeError(
                        f"Failed to scrub editable-install metadata in agent sandbox for {task_id(task)!r}: "
                        f"{scrub_result.stderr or scrub_result.stdout or '(no output)'}"
                    )
                if self.config.scrub_workspace_installs:
                    ws_scrub = await sandbox.exec(
                        command=_WORKSPACE_INSTALL_SCRUB_COMMAND.replace(
                            "__IDENTS__", " ".join(derive_identifiers(task_id(task), task.test_commands.commands)[0])
                        ),
                        cwd="/",
                        timeout_s=120,
                    )
                    if ws_scrub.return_code != 0:
                        raise RuntimeError(
                            f"Failed to scrub /workspace installs in agent sandbox for {task_id(task)!r}: "
                            f"{ws_scrub.stderr or ws_scrub.stdout or '(no output)'}"
                        )
            elif mode == "block_target":
                await self._install_pip_git_guard(sandbox, task)
            elif mode == "block_target_v2":
                await install_guard_v2(sandbox, task_id(task), task.test_commands.commands)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(task.start_md)
                local_start_md_path = Path(f.name)
            try:
                await sandbox.upload(local_start_md_path, "/workspace/start.md")
            finally:
                local_start_md_path.unlink(missing_ok=True)
        return sandbox

    # Blanket-match tools: never legitimately invoked in a way that would coincidentally mention
    # the target name, so any argv substring hit is a real block. pip3.x name variants added
    # alongside pip/pip3 since some images expose only the versioned name.
    _BLANKET_GUARD_TOOLS = (
        "pip",
        "pip3",
        "pip3.9",
        "pip3.10",
        "pip3.11",
        "pip3.12",
        "pip3.13",
        "git",
        "curl",
        "wget",
    )
    # Module-aware tools: used constantly for legitimate work (running the agent's own tests,
    # scripts) that routinely mentions the target name incidentally - only guarded when invoked
    # in a package-fetch shape (`-m pip ...` / `uv pip ...`), see
    # _GUARD_SCRIPT_TEMPLATE_MODULE_AWARE.
    _MODULE_AWARE_GUARD_TOOLS = (
        "python",
        "python3",
        "python3.9",
        "python3.10",
        "python3.11",
        "python3.12",
        "python3.13",
        "uv",
    )

    async def _install_pip_git_guard(self, sandbox: AsyncSandbox, task: Task) -> None:
        """Replace pip/pip3/git/curl/wget/uv/python(3.x) in place with wrappers that refuse to
        fetch this task's own package/repo by name, while passing every other invocation through
        untouched. Used for agent_network_mode == "block_target", where network access otherwise
        stays open. Covers the bypass vectors confirmed live in transcripts: direct pip/git, bare
        curl/wget to PyPI/GitHub/jsdelivr mirrors, `python -m pip`, and `uv pip install`."""

        current_task_id = task_id(task)
        target = current_task_id.lower().replace("_", "-")
        tool_groups = (
            (self._BLANKET_GUARD_TOOLS, _GUARD_SCRIPT_TEMPLATE),
            (self._MODULE_AWARE_GUARD_TOOLS, _GUARD_SCRIPT_TEMPLATE_MODULE_AWARE),
        )
        for tool_names, template in tool_groups:
            discover_cmd = "; ".join(f"command -v \"{n}\" || echo ''" for n in tool_names)
            discover = await sandbox.exec(command=discover_cmd, cwd="/", timeout_s=30)
            if discover.return_code != 0:
                raise RuntimeError(
                    f"Failed to discover {tool_names!r} locations for guard install on "
                    f"{current_task_id!r}: {discover.stderr or discover.stdout or '(no output)'}"
                )
            real_paths = (discover.stdout or "").splitlines()
            for name, real_path in zip(tool_names, real_paths):
                real_path = real_path.strip()
                if not real_path:
                    continue  # tool isn't installed in this image; nothing to guard
                backup_path = f"{real_path}.nl2repobench-real"
                move_result = await sandbox.exec(
                    command=f"mv {shlex.quote(real_path)} {shlex.quote(backup_path)}", cwd="/", timeout_s=30
                )
                if move_result.return_code != 0:
                    raise RuntimeError(
                        f"Failed to back up real {name!r} for guard install on {current_task_id!r}: "
                        f"{move_result.stderr or move_result.stdout or '(no output)'}"
                    )
                script = template.format(target=target, real_path=backup_path, tool_name=name)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                    f.write(script)
                    local_script_path = Path(f.name)
                try:
                    await sandbox.upload(local_script_path, real_path)
                finally:
                    local_script_path.unlink(missing_ok=True)
                chmod_result = await sandbox.exec(command=f"chmod +x {shlex.quote(real_path)}", cwd="/", timeout_s=30)
                if chmod_result.return_code != 0:
                    raise RuntimeError(
                        f"Failed to chmod guard wrapper for {name!r} on {current_task_id!r}: "
                        f"{chmod_result.stderr or chmod_result.stdout or '(no output)'}"
                    )

    async def _stop_sandbox(self, sandbox: AsyncSandbox, *, task_id: str, phase: str) -> None:
        try:
            await sandbox.stop()
        except Exception:
            print(f"Failed to stop NL2RepoBench {phase} sandbox for {task_id}: {format_exc()}", file=sys.stderr)

    async def seed_session(
        self, request: Request, body: NL2RepoBenchSeedSessionRequest
    ) -> NL2RepoBenchSeedSessionResponse:
        task = _resolve_task(body, self._task_store, registry_template=self.config.task_image_registry_template)
        session_id = str(request.session[SESSION_ID_KEY])
        previous_session = self._agent_sessions.pop(session_id, None)
        if previous_session is not None:
            await self._stop_sandbox(
                previous_session.sandbox,
                task_id=previous_session.task_id,
                phase="replaced-agent",
            )

        sandbox: AsyncSandbox | None = None
        try:
            sandbox = await self._create_sandbox(task, phase="agent")
            current_task_id = task_id(task)
            current_task_image = task_image(task, registry_template=self.config.task_image_registry_template)
            descriptor = await sandbox.serialize()
            sandbox_handle = descriptor.get("sandbox_id") if isinstance(descriptor, dict) else None
            if not isinstance(sandbox_handle, str) or not sandbox_handle:
                raise RuntimeError("NL2RepoBench sandbox provider did not return a sandbox_id")
            sandbox_descriptor = dict(descriptor)
            self._agent_sessions[session_id] = AgentSandboxSession(
                task_id=current_task_id,
                image=current_task_image,
                sandbox=sandbox,
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=sandbox_descriptor,
            )
            return NL2RepoBenchSeedSessionResponse(
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=sandbox_descriptor,
                # OpenCodeSandboxedAgent reconnects to this sandbox by sandbox_handle
                # alone and otherwise defaults to the container's own image-baked
                # WORKDIR (opencode_sandboxed_agent/app.py:496) when running opencode -
                # which is /workspace for the per-task pinned images (their own
                # Dockerfile WORKDIR) but NOT for the generic agent-sandbox image
                # (_AGENT_SANDBOX_IMAGE), whose own WORKDIR is elsewhere. Return it
                # explicitly so opencode actually runs from where start.md was written,
                # regardless of what the booted image's default happens to be. This
                # sandbox is always phase="agent" (see _create_sandbox call above).
                workdir="/workspace",
            )
        except Exception:
            if sandbox is not None:
                await self._stop_sandbox(sandbox, task_id=task_id(task), phase="failed-agent-seed")
            raise

    async def _collect_workspace(self, sandbox: AsyncSandbox) -> tuple[bytes | None, str | None]:
        """Tar up /workspace in the agent sandbox and download it. Returns (tar_bytes, error)."""

        result = await sandbox.exec(_COLLECT_COMMAND, timeout_s=WORKSPACE_COLLECT_TIMEOUT_S)
        if result.return_code != 0:
            details = ((result.stderr or "") + (result.stdout or "")).strip()
            return None, f"Workspace collection exited with code {result.return_code}: {details[-4000:]}"

        with tempfile.TemporaryDirectory(prefix="nemo-gym-nl2repobench-collect-") as temporary_dir:
            local_tar_path = Path(temporary_dir) / "workspace.tar.gz"
            await sandbox.download("/tmp/workspace.tar.gz", local_tar_path)
            tar_bytes = local_tar_path.read_bytes()
        if len(tar_bytes) > self.config.workspace_tar_max_bytes:
            return None, (
                f"Collected workspace tar ({len(tar_bytes)} bytes) exceeds workspace_tar_max_bytes "
                f"({self.config.workspace_tar_max_bytes} bytes)"
            )
        return tar_bytes, None

    async def _run_verifier(self, sandbox: AsyncSandbox, task: Task, workspace_tar: bytes) -> VerifierResult:
        # cwd="/" is required, not just defensive - same reasoning as the agent-phase mkdir
        # (_create_sandbox): exec()'s default cwd is SandboxSpec.workdir ("/workspace"), and at
        # least one provider (OpenSandbox) validates the cwd exists before running the command at
        # all, so without this the call fails with "working directory does not exist: /workspace"
        # before mkdir ever runs. Latent for upstream (whose pinned per-task images always have
        # /workspace pre-created by their own Dockerfile WORKDIR, so this path never executes
        # there) - surfaced by verify_image_override pointing at a generic image instead
        # (confirmed via a real failed run, 100% of verifies failing on this exact error).
        mkdir_result = await sandbox.exec("mkdir -p /workspace", cwd="/", timeout_s=60)
        if mkdir_result.return_code != 0:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_error=f"Failed to create /workspace: {mkdir_result.stderr or ''}",
                test_case_count=task.test_case_count,
            )

        with tempfile.TemporaryDirectory(prefix="nemo-gym-nl2repobench-restore-") as temporary_dir:
            local_tar_path = Path(temporary_dir) / "workspace.tar.gz"
            local_tar_path.write_bytes(workspace_tar)
            await sandbox.upload(local_tar_path, "/tmp/workspace.tar.gz")

        upstream = self.config.grading_protocol == "upstream"
        if upstream:
            extract_result = await sandbox.exec(
                _upstream_restore_command(task.test_files.files), cwd="/", timeout_s=300
            )
        else:
            extract_result = await sandbox.exec("tar xzf /tmp/workspace.tar.gz -C /workspace", timeout_s=300)
        if extract_result.return_code != 0:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=extract_result.return_code,
                verifier_error=f"Failed to extract workspace tar: {extract_result.stderr or ''}",
                test_case_count=task.test_case_count,
            )

        output_parts: list[str] = []
        pytest_outputs: list[str] = []
        verifier_error: str | None = None
        last_exit_code: int | None = None
        commands = task.test_commands.commands
        for index, command in enumerate(commands):
            is_last = index == len(commands) - 1
            run_command = f"export PYTHONPATH=/workspace${{PYTHONPATH:+:$PYTHONPATH}}; {command}" if upstream else command
            command_result = await sandbox.exec(run_command, cwd="/workspace", timeout_s=TEST_COMMAND_TIMEOUT_S)
            last_exit_code = command_result.return_code
            command_output = f"{command_result.stdout or ''}{command_result.stderr or ''}"
            output_parts.append(f"$ {command}\n{command_output}\n[exit code: {command_result.return_code}]\n")
            if upstream:
                # Upstream never stops early: a failed install still leaves the image's own install and PYTHONPATH.
                if "pytest" in command.lower():
                    pytest_outputs.append(command_output)
                continue
            if command_result.return_code != 0 and not is_last:
                verifier_error = f"install command failed: {command}"
                break

        combined_output = ("\n" + "-" * 40 + "\n").join(output_parts)
        if upstream:
            # Parsed before the size cap below, so a long log can't hide a summary line.
            tests_passed, tests_failed, tests_error = _parse_pytest_upstream(pytest_outputs)
            combined_output = f"[grading_protocol: upstream]\n{combined_output}"
        if len(combined_output) > TEST_OUTPUT_MAX_CHARS:
            combined_output = combined_output[-TEST_OUTPUT_MAX_CHARS:]

        if not upstream:
            tests_passed, tests_failed, tests_error = _parse_pytest_summary(combined_output)
        if verifier_error is not None:
            tests_passed = 0
        reward = _compute_reward(tests_passed, task.test_case_count)

        return VerifierResult(
            evaluation_completed=True,
            reward=reward,
            verifier_exit_code=last_exit_code,
            verifier_error=verifier_error,
            test_output=combined_output,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_error=tests_error,
            test_case_count=task.test_case_count,
            success_rate=reward,
        )

    async def verify(self, request: Request, body: NL2RepoBenchVerifyRequest) -> NL2RepoBenchVerifyResponse:
        task = _resolve_task(body, self._task_store, registry_template=self.config.task_image_registry_template)
        current_task_id = task_id(task)
        current_task_image = task_image(task, registry_template=self.config.task_image_registry_template)
        session_id = str(request.session.get(SESSION_ID_KEY, "unknown"))
        sandbox_handle = body.sandbox_handle

        workspace_error: str | None = None
        workspace_tar = b""
        workspace_collection_time_s = 0.0

        agent_session = self._agent_sessions.pop(session_id, None)
        if agent_session is None:
            workspace_error = f"no agent session found for session {session_id!r}"
        else:
            sandbox_handle = agent_session.sandbox_handle
            started = monotonic()
            try:
                if agent_session.task_id != current_task_id:
                    raise RuntimeError(
                        f"NL2RepoBench session task {agent_session.task_id!r} does not match verify task "
                        f"{current_task_id!r}"
                    )
                if agent_session.image != current_task_image:
                    raise RuntimeError(
                        f"NL2RepoBench session image {agent_session.image!r} does not match verify image "
                        f"{current_task_image!r}"
                    )
                collected, collect_error = await self._collect_workspace(agent_session.sandbox)
                if collect_error is not None:
                    workspace_error = collect_error
                else:
                    workspace_tar = collected or b""
            except Exception as error:
                print(
                    f"Failed to collect NL2RepoBench workspace for {current_task_id}: {format_exc()}",
                    file=sys.stderr,
                )
                workspace_error = f"{type(error).__name__}: {error}"
            finally:
                workspace_collection_time_s = monotonic() - started
                await self._stop_sandbox(agent_session.sandbox, task_id=current_task_id, phase="agent")

        workspace_sha256 = hashlib.sha256(workspace_tar).hexdigest()
        workspace_bytes = len(workspace_tar)
        log_dir = _resolve_repo_path(self.config.logs_dir) / current_task_id / session_id

        saved_workspace_path: str | None = None
        if self.config.save_workspace_dir is not None and workspace_error is None and workspace_tar:
            workspace_save_path = (
                _resolve_repo_path(self.config.save_workspace_dir) / current_task_id / f"{session_id}.tar.gz"
            )
            workspace_save_path.parent.mkdir(parents=True, exist_ok=True)
            workspace_save_path.write_bytes(workspace_tar)
            saved_workspace_path = str(workspace_save_path)

        sandbox: AsyncSandbox | None = None
        sandbox_start_time_s = 0.0
        verification_time_s = 0.0
        result = VerifierResult(
            evaluation_completed=False,
            reward=0.0,
            verifier_error=workspace_error,
            test_case_count=task.test_case_count,
        )
        if workspace_error is None:
            try:
                started = monotonic()
                sandbox = await self._create_sandbox(task, phase="verifier")
                sandbox_start_time_s = monotonic() - started
                started = monotonic()
                result = await self._run_verifier(sandbox, task, workspace_tar)
                verification_time_s = monotonic() - started
            except Exception as error:
                print(f"NL2RepoBench verifier failed for {current_task_id}: {format_exc()}", file=sys.stderr)
                result = VerifierResult(
                    evaluation_completed=False,
                    reward=0.0,
                    verifier_error=f"{type(error).__name__}: {error}",
                    test_case_count=task.test_case_count,
                )
            finally:
                if sandbox is not None:
                    await self._stop_sandbox(sandbox, task_id=current_task_id, phase="verifier")

        # test_output is captured in-memory and returned directly, not written under log_dir, so
        # log_dir is a nominal per-session path; clearing is a no-op unless something else wrote
        # into it (e.g. save_workspace_dir, which is a separate directory, unaffected by this).
        if self.config.clear_verifier_logs:
            rmtree(str(log_dir), ignore_errors=True)
            log_dir_str = ""
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_dir_str = str(log_dir)

        return NL2RepoBenchVerifyResponse.model_validate(
            body.model_dump()
            | result.model_dump()
            | {
                "task_id": current_task_id,
                "sandbox_handle": sandbox_handle,
                "workspace_bytes": workspace_bytes,
                "workspace_sha256": workspace_sha256,
                "workspace_tar": saved_workspace_path
                if saved_workspace_path is not None
                else (workspace_tar.hex() if self.config.include_workspace_in_response else None),
                "log_dir": log_dir_str,
                "workspace_collection_time_s": workspace_collection_time_s,
                "sandbox_start_time_s": sandbox_start_time_s,
                "verification_time_s": verification_time_s,
            }
        )


if __name__ == "__main__":
    NL2RepoBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NL2RepoBenchResourcesServer.run_webserver()  # noqa: F401
