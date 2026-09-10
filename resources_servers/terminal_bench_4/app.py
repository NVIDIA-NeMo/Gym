# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal-Bench 4.0 resources server: Harbor's separate-verifier contract on a sandbox provider.

Sequence per rollout (mirrors harbor 0.22.0 ``Trial._run_separate_verifier``):

1. ``seed_session`` starts the AGENT sandbox from the row's ``docker_image`` (the task's prebuilt
   environment image) with the resources the task declares in ``[environment]``.
2. ``verify``: run the task's main-service ``[[verifier.collect]]`` hooks in the agent sandbox; probe
   the declared ``artifacts`` plus the convention dir ``/logs/artifacts``; pack every path that exists
   into ONE gzip tarball rooted at ``/`` and download it; stop the agent sandbox; start the VERIFIER
   sandbox from ``verifier_docker_image`` (the prebuilt ``tests/Dockerfile`` image, ``/tests`` baked in)
   with ``[verifier.environment]`` or, failing that, ``[environment]`` resources; create and
   ``chmod 777`` ``/logs/verifier`` and ``/logs/artifacts``; empty the target of every directory
   artifact and pre-create the parent of every file artifact (Harbor's ``empty_dirs``/``ensure_dirs``);
   upload and extract the tarball with ``--no-same-owner`` (files arrive owned by the verifier user,
   modes preserved, as under Harbor); ``chmod +x /tests/test.sh`` and run
   ``(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1`` under the task's verifier timeout; read
   ``/logs/verifier/reward.json`` first, else ``reward.txt``; always stop the verifier sandbox.

Failure policy: anything that means the task was never graded because of OUR infrastructure (a
sandbox that would not start, an artifact tarball that could not be packed, uploaded or extracted,
a verifier image without ``/tests/test.sh``) raises, so the rollout is invalidated and retried
rather than recorded as a zero. Outcomes that Harbor also records as an ungraded trial (verifier
timeout, missing or unparseable reward file) return ``evaluation_completed = False`` with reward 0
and a ``failure_reason``; they can be caused by what the agent left behind, so they are data.

Golden-patch (oracle) mode: ``is_verifying_golden_patch = true`` makes ``verify`` create the agent
sandbox itself, upload ``<task_folder>/solution`` to ``/solution`` (where Harbor mounts it) and run
``bash /solution/solve.sh`` as the image user before grading. Reference solutions must grade to 1
through this path before any model result is trusted.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path, PurePosixPath
from time import monotonic
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.rollout_observability import SandboxObservation
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.terminal_bench_4.task_manifest import ArtifactEntry, TB4Task, load_task


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]

VERIFIER_DIR = "/logs/verifier"
ARTIFACTS_DIR = "/logs/artifacts"
TESTS_DIR = "/tests"
TEST_SCRIPT = f"{TESTS_DIR}/test.sh"
TEST_STDOUT = f"{VERIFIER_DIR}/test-stdout.txt"
SOLUTION_DIR = "/solution"
REMOTE_TARBALL = "/tmp/nemo_gym_tb4_artifacts.tgz"
REWARD_JSON = "reward.json"
REWARD_TXT = "reward.txt"
VERIFIER_FILES_OF_INTEREST = (REWARD_JSON, REWARD_TXT, "ctrf.json", "test-stdout.txt")

# Exit code the pack/extract shell snippets use when neither tar nor python3 is available.
NO_PACKER_EXIT_CODE = 97
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# Python fallback for images without GNU tar (arguments: tarball, then members relative to /).
_PY_PACK = (
    "import fnmatch,os,sys,tarfile\n"
    "args=sys.argv[1:]\n"
    "out=args.pop(0)\n"
    "excludes=[]\n"
    "while args and args[0]=='-x':\n"
    "    args.pop(0); excludes.append(args.pop(0))\n"
    "if args and args[0]=='--': args.pop(0)\n"
    "def skip(name):\n"
    "    parts=name.split('/')\n"
    "    for pat in excludes:\n"
    "        if '/' in pat:\n"
    "            if name==pat or name.startswith(pat+'/'): return True\n"
    "        elif any(fnmatch.fnmatchcase(part,pat) for part in parts): return True\n"
    "    return False\n"
    "def flt(ti):\n"
    "    return None if skip(ti.name) else ti\n"
    "with tarfile.open(out,'w:gz') as t:\n"
    "    for m in args:\n"
    "        p='/'+m\n"
    "        if os.path.lexists(p): t.add(p,arcname=m,filter=flt)\n"
)
TAR_STDERR = "/tmp/nemo_gym_tb4_tar.err"
_PY_EXTRACT = "import sys,tarfile\nwith tarfile.open(sys.argv[1],'r:gz') as t:\n    t.extractall('/')\n"


class RewardParseError(ValueError):
    """The verifier wrote a reward file that cannot be turned into a number."""


class OracleUnsupportedError(RuntimeError):
    """Golden mode cannot stage /solution in this agent image (non-root image user, no escalation)."""


class TerminalBench4ResourcesServerConfig(BaseResourcesServerConfig):
    # Grading needs the live agent sandbox, so a stored rollout cannot be re-verified.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    is_verifying_golden_patch: bool = False
    golden_patch_timeout_s: float = Field(default=3600.0, gt=0)

    verifier_timeout_multiplier: float = Field(default=1.0, gt=0)
    verifier_timeout_floor_s: float = Field(default=60.0, ge=0)
    verifier_timeout_cap_s: Optional[float] = Field(default=None, gt=0)
    collect_hook_default_timeout_s: float = Field(default=300.0, gt=0)

    artifact_max_bytes: int = Field(default=2 * 1024**3, gt=0)
    allow_compose_tasks: bool = False
    allow_gpu_tasks: bool = False
    stop_agent_sandbox_before_verify: bool = True

    use_task_resources: bool = True
    cpu_multiplier: float = Field(default=1.0, gt=0)
    memory_multiplier: float = Field(default=1.0, gt=0)
    min_cpu: float = Field(default=1.0, gt=0)
    min_memory_mib: int = Field(default=2048, gt=0)
    min_disk_gib: int = Field(default=10, gt=0)

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

    logs_dir: Path = Path("resources_servers/terminal_bench_4/logs")
    max_test_output_chars: int = Field(default=200_000, gt=0)
    # Gym's CPU thread-cap env vars go to the agent sandbox; Harbor sets none in the verifier.
    verifier_derive_cpu_env: bool = False

    debug: bool = False


class TerminalBench4SeedSessionRequest(BaseModel):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")

    task_name: str
    docker_image: str
    verifier_docker_image: str
    task_folder: str


class TerminalBench4SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # Plain string sandbox id; the agent reconnects with AsyncSandbox.connect.


class TerminalBench4VerifyRequest(TerminalBench4SeedSessionRequest, BaseVerifyRequest):
    pass


class TerminalBench4VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_name: str
    evaluation_completed: bool
    verification_time_taken: float

    # Verifier phase
    verifier_exit_code: Optional[int] = None
    verifier_error: Optional[str] = None
    reward_source: Optional[str] = None
    rewards: Dict[str, float] = Field(default_factory=dict)
    test_output: str = ""
    verifier_stderr_tail: str = ""
    verifier_wall_time_s: float = 0.0

    # Artifact phase
    artifact_manifest: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts_packed: int = 0
    artifact_tarball_bytes: int = 0
    artifact_tar_return_code: Optional[int] = None
    artifact_tar_warnings: str = ""
    artifact_excludes: List[str] = Field(default_factory=list)
    collect_hook_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Bookkeeping
    agent_sandbox_stopped: bool = False
    artifact_collection_time_s: float = 0.0
    verifier_sandbox_start_time_s: float = 0.0
    golden_patch_exit_code: Optional[int] = None
    golden_patch_output: Optional[str] = None
    log_dir: str = ""

    verifier_sandbox_observation: Optional[SandboxObservation] = Field(
        default=None, exclude_if=lambda value: value is None
    )


@dataclass
class AgentSession:
    task: TB4Task
    sandbox: AsyncSandbox
    docker_image: str
    started_at: float


@dataclass
class ArtifactProbe:
    source: str
    service: str
    kind: str  # dir | file | missing | unknown
    status: str  # ok | missing | skipped_sidecar
    relative: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "service": self.service,
            "kind": self.kind,
            "status": self.status,
            "relative": self.relative,
        }


@dataclass
class VerifierOutcome:
    evaluation_completed: bool = False
    reward: float = 0.0
    rewards: Dict[str, float] = field(default_factory=dict)
    reward_source: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    test_output: str = ""
    stderr_tail: str = ""
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested directly; every remote command starts with a `: ng-tb4-<step>;` label)
# ---------------------------------------------------------------------------------------------


def labeled(step: str, command: str) -> str:
    """Prefix a shell snippet with a no-op label so execd logs and tests can tell the steps apart."""
    return f": ng-tb4-{step}; {command}"


def build_probe_command(sources: Sequence[str]) -> str:
    quoted = " ".join(shlex.quote(s) for s in sources)
    body = (
        f"for p in {quoted}; do "
        'if [ -d "$p" ]; then k=dir; elif [ -e "$p" ]; then k=file; else k=missing; fi; '
        'printf \'%s\\t%s\\n\' "$k" "$p"; done'
    )
    return labeled("probe", body)


def parse_probe_output(stdout: Optional[str]) -> Dict[str, str]:
    """``kind<TAB>path`` lines -> {path: kind}; unknown lines are ignored."""
    kinds: Dict[str, str] = {}
    for line in (stdout or "").splitlines():
        if "\t" not in line:
            continue
        kind, path = line.split("\t", 1)
        kind = kind.strip()
        if kind in ("dir", "file", "missing"):
            kinds[path] = kind
    return kinds


def translate_excludes(member: str, patterns: Sequence[str]) -> List[str]:
    """Harbor applies ``exclude`` patterns relative to the directory being archived (``tar -C <dir> .``).

    Packing everything from ``/`` in one archive, a path-like pattern (``./megatron_parallel.py``,
    ``sub/dir``) is anchored under its member; a bare name or glob (``__pycache__``, ``*.pyc``,
    ``node_modules``) stays unanchored, which GNU tar matches at any depth (and, as a documented
    over-approximation, in every member of the archive).
    """
    translated: List[str] = []
    for pattern in patterns:
        anchored = pattern.startswith("./")
        cleaned = (pattern[2:] if anchored else pattern).rstrip("/")
        if not cleaned:
            continue
        anchored = anchored or "/" in cleaned
        translated.append(f"{member}/{cleaned}" if anchored else cleaned)
    return translated


def build_pack_command(members: Sequence[str], tarball: str = REMOTE_TARBALL, excludes: Sequence[str] = ()) -> str:
    """Pack ``members`` (paths relative to ``/``) into ``tarball``.

    Prints ``NG_TB4_SIZE=<bytes>`` and ``NG_TB4_TAR_RC=<rc>`` followed by tar's warnings. GNU tar exits 1
    (with a complete archive) when a member changes while it is read, which an agent's leftover
    background process does routinely; Harbor's best-effort collection grades such a trajectory, so
    rc 1 is tolerated and recorded, rc >= 2 is a failure.
    """
    if not members:
        raise ValueError("nothing to pack")
    quoted_members = " ".join(shlex.quote(m) for m in members)
    quoted_tarball = shlex.quote(tarball)
    quoted_err = shlex.quote(TAR_STDERR)
    tar_excludes = "".join(f"--exclude={shlex.quote(p)} " for p in excludes)
    py_excludes = "".join(f"-x {shlex.quote(p)} " for p in excludes)
    body = (
        f"if [ -e {quoted_tarball} ]; then unlink {quoted_tarball}; fi; : > {quoted_err}; "
        "if command -v tar >/dev/null 2>&1; then "
        f"tar -czf {quoted_tarball} -C / --ignore-failed-read {tar_excludes}-- {quoted_members} 2>{quoted_err}; rc=$?; "
        "elif command -v python3 >/dev/null 2>&1; then "
        f"python3 -c {shlex.quote(_PY_PACK)} {quoted_tarball} {py_excludes}-- {quoted_members} 2>{quoted_err}; rc=$?; "
        f"else echo NG_TB4_NO_PACKER >&2; exit {NO_PACKER_EXIT_CODE}; fi; "
        '[ "$rc" -le 1 ] || exit "$rc"; '
        f'echo "NG_TB4_SIZE=$(stat -c %s {quoted_tarball})"; echo "NG_TB4_TAR_RC=$rc"; cat {quoted_err}'
    )
    return labeled("pack", body)


def parse_pack_output(stdout: Optional[str]) -> Tuple[int, int, str]:
    """Return (tarball bytes, tar return code, warning text) from the pack command's stdout."""
    size = rc = None
    warnings: List[str] = []
    for line in (stdout or "").splitlines():
        if line.startswith("NG_TB4_SIZE="):
            size = int(line.split("=", 1)[1].strip())
        elif line.startswith("NG_TB4_TAR_RC="):
            rc = int(line.split("=", 1)[1].strip())
        elif line.strip():
            warnings.append(line.rstrip())
    if size is None or rc is None:
        raise ValueError(f"pack output lacks the size or return-code marker: {stdout!r}")
    return size, rc, "\n".join(warnings)


def build_extract_command(tarball: str = REMOTE_TARBALL) -> str:
    quoted_tarball = shlex.quote(tarball)
    body = (
        "if command -v tar >/dev/null 2>&1; then "
        f"tar -xzpf {quoted_tarball} --no-same-owner -C /; "
        "elif command -v python3 >/dev/null 2>&1; then "
        f"python3 -c {shlex.quote(_PY_EXTRACT)} {quoted_tarball}; "
        f"else echo NG_TB4_NO_PACKER >&2; exit {NO_PACKER_EXIT_CODE}; fi "
        f"&& unlink {quoted_tarball}"
    )
    return labeled("extract", body)


def build_prepare_targets_command(dir_targets: Sequence[str], file_targets: Sequence[str]) -> str:
    """Harbor's verifier-side preparation: empty + chmod 777 directory targets, ensure + chmod 777 file parents."""
    parts: List[str] = [f"mkdir -p {VERIFIER_DIR} {ARTIFACTS_DIR} && chmod 777 {VERIFIER_DIR} {ARTIFACTS_DIR}"]
    for target in dir_targets:
        q = shlex.quote(target)
        # Harbor's empty_dirs guard: a file, dangling link or link-to-file occupying the path is removed first.
        parts.append(
            f"if [ -L {q} ] || {{ [ -e {q} ] && [ ! -d {q} ]; }}; then unlink {q}; fi; "
            f"mkdir -p {q} && find {q} -mindepth 1 -delete && chmod 777 {q}"
        )
    parents = []
    for target in file_targets:
        parent = PurePosixPath(target).parent.as_posix()
        if parent and parent != "/" and parent not in parents:
            parents.append(parent)
    for parent in parents:
        q = shlex.quote(parent)
        parts.append(f"mkdir -p {q} && chmod 777 {q}")
    return labeled("prepare-targets", " && ".join(parts))


def build_verifier_files_probe_command() -> str:
    names = " ".join(VERIFIER_FILES_OF_INTEREST)
    body = (
        f"for f in {names}; do p={VERIFIER_DIR}/$f; "
        'if [ -f "$p" ]; then printf \'%s\\t%s\\n\' "$(stat -c %s "$p")" "$f"; fi; done'
    )
    return labeled("probe-reward", body)


def parse_verifier_files_probe(stdout: Optional[str]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for line in (stdout or "").splitlines():
        if "\t" not in line:
            continue
        size, name = line.split("\t", 1)
        try:
            sizes[name.strip()] = int(size.strip())
        except ValueError:
            continue
    return sizes


def parse_reward_payload(source: str, raw: bytes) -> Tuple[float, Dict[str, float]]:
    """Harbor's reward parsing: ``reward.json`` (``{"reward": x}`` or a single numeric value) or a bare float."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise RewardParseError(f"{source} is empty")
    if source == REWARD_JSON:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise RewardParseError(f"{source} is not valid JSON: {error}") from error
        if isinstance(data, bool):
            raise RewardParseError(f"{source} holds a boolean, not a number")
        if isinstance(data, (int, float)):
            return float(data), {"reward": float(data)}
        if isinstance(data, dict):
            numeric = {
                str(k): float(v) for k, v in data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if "reward" in numeric:
                return numeric["reward"], numeric
            if len(numeric) == 1:
                return next(iter(numeric.values())), numeric
            raise RewardParseError(f"{source} has no `reward` key and is not a single numeric value: {text[:200]}")
        raise RewardParseError(f"{source} has an unsupported JSON shape: {type(data).__name__}")
    try:
        value = float(text)
    except ValueError as error:
        raise RewardParseError(f"{source} is not a number: {text[:200]!r}") from error
    return value, {"reward": value}


def derive_resources(
    environment: Dict[str, Any],
    *,
    base: Dict[str, Any],
    use_task_resources: bool,
    cpu_multiplier: float,
    memory_multiplier: float,
    min_cpu: float,
    min_memory_mib: int,
    min_disk_gib: int,
) -> SandboxResources:
    """Map Harbor's ``cpus`` / ``memory_mb`` / ``storage_mb`` onto a sandbox request, scaled and floored."""
    resources = dict(base)
    if use_task_resources:
        cpus = environment.get("cpus")
        memory_mb = environment.get("memory_mb")
        storage_mb = environment.get("storage_mb")
        if cpus:
            resources["cpu"] = max(float(min_cpu), float(cpus) * cpu_multiplier)
        if memory_mb:
            resources["memory_mib"] = max(int(min_memory_mib), ceil(float(memory_mb) * memory_multiplier))
        if storage_mb:
            resources["disk_gib"] = max(int(min_disk_gib), ceil(float(storage_mb) / 1024))
    return SandboxResources.from_mapping(resources)


def clamp_resource_requests(provider_options: Dict[str, Any], limits: SandboxResources) -> Dict[str, Any]:
    """Cap ``provider_options.resource_requests`` at the sandbox limits.

    Kubernetes rejects a pod whose request exceeds its limit, and on a platform that honours custom
    resources (the ``nemo.nvidia.com/resources: custom`` label) such a sandbox sits in ``Pending``
    until the create call times out. Task-derived limits (e.g. ``storage_mb = 10240`` -> 10 GiB) can be
    smaller than the config's scheduling requests (30 GiB disk), so clamp every request to its limit.
    """
    requests = provider_options.get("resource_requests")
    if not isinstance(requests, dict):
        return provider_options
    clamped = dict(requests)
    for key, limit in (("cpu", limits.cpu), ("memory_mib", limits.memory_mib), ("disk_gib", limits.disk_gib)):
        if limit is not None and key in clamped and clamped[key] is not None and clamped[key] > limit:
            clamped[key] = limit
    provider_options["resource_requests"] = clamped
    return provider_options


def safe_name(value: str) -> str:
    return _SAFE_NAME_RE.sub("_", value).strip("_") or "task"


def tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"...[{len(text) - limit} chars elided]...\n" + text[-limit:]


# ---------------------------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------------------------


class TerminalBench4ResourcesServer(SimpleResourcesServer):
    config: TerminalBench4ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sessions: Dict[str, AgentSession] = dict()
        self._last_pack: Dict[str, Any] = {}

    # -- configuration helpers --------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.config.debug:
            print(f"[terminal_bench_4] {message}", file=sys.stderr, flush=True)

    def _resolve_logs_dir(self) -> Path:
        logs_dir = self.config.logs_dir.expanduser()
        return logs_dir if logs_dir.is_absolute() else (NEMO_GYM_ROOT / logs_dir)

    def _session_log_dir(self, task: TB4Task, session_id: str) -> Path:
        log_dir = self._resolve_logs_dir() / f"{safe_name(task.task_name.rsplit('/', 1)[-1])}__{safe_name(session_id)}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _load_task(self, body: TerminalBench4SeedSessionRequest) -> TB4Task:
        task = load_task(body.task_folder, repo_root=NEMO_GYM_ROOT)
        if not body.docker_image or not body.verifier_docker_image:
            raise ValueError(f"{task.task_name}: rows must carry both docker_image and verifier_docker_image")
        if task.is_compose and not self.config.allow_compose_tasks:
            raise ValueError(
                f"{task.task_name}: multi-service compose task (services {list(task.compose_services)}) cannot run in a "
                "single sandbox; refused (allow_compose_tasks=false)"
            )
        if task.requires_gpu and not self.config.allow_gpu_tasks:
            raise ValueError(f"{task.task_name}: declares gpus > 0; refused (allow_gpu_tasks=false)")
        return task

    def _resources_for(self, environment: Dict[str, Any]) -> SandboxResources:
        return derive_resources(
            environment,
            base=dict(self.config.sandbox_config.get("resources", {})),
            use_task_resources=self.config.use_task_resources,
            cpu_multiplier=self.config.cpu_multiplier,
            memory_multiplier=self.config.memory_multiplier,
            min_cpu=self.config.min_cpu,
            min_memory_mib=self.config.min_memory_mib,
            min_disk_gib=self.config.min_disk_gib,
        )

    def _verifier_timeout_s(self, task: TB4Task) -> float:
        timeout = task.verifier_timeout_sec * self.config.verifier_timeout_multiplier
        timeout = max(timeout, self.config.verifier_timeout_floor_s)
        if self.config.verifier_timeout_cap_s is not None:
            timeout = min(timeout, self.config.verifier_timeout_cap_s)
        return timeout

    # -- sandbox lifecycle ------------------------------------------------------------------

    async def _create_sandbox(
        self, *, image: str, resources: SandboxResources, role: str, task: TB4Task
    ) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        env = dict(self.config.sandbox_config.get("env", {}))
        derive = self.config.sandbox_config.get("derive_cpu_env", True)
        if role == "verifier":
            derive = derive and self.config.verifier_derive_cpu_env
        if derive:
            env = cpu_cap_env(resources.cpu) | env  # explicit keys win over the derived caps

        spec = SandboxSpec(
            image=image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # the image's WORKDIR, as under Harbor
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": task.task_name,
                "tb4_role": role,
            },
            resources=resources,
            entrypoint=None,
            provider_options=clamp_resource_requests(
                deepcopy(self.config.sandbox_config.get("provider_options") or {}), resources
            ),
        )
        sandbox = AsyncSandbox(provider)
        self._log(
            f"starting {role} sandbox for {task.task_name} from {image} with {resources} "
            f"requests={spec.provider_options.get('resource_requests')}"
        )
        await sandbox.start(spec)
        return sandbox

    async def _stop_sandbox(self, sandbox: Optional[AsyncSandbox], *, role: str, task_name: str) -> bool:
        if sandbox is None:
            return False
        try:
            await sandbox.stop()
            return True
        except Exception:
            print(f"Failed to stop TB4 {role} sandbox for {task_name}: {format_exc()}", file=sys.stderr)
            return False

    # -- endpoints --------------------------------------------------------------------------

    async def seed_session(
        self, request: Request, body: TerminalBench4SeedSessionRequest
    ) -> TerminalBench4SeedSessionResponse:
        task = self._load_task(body)
        sandbox = await self._create_sandbox(
            image=body.docker_image, resources=self._resources_for(task.environment), role="agent", task=task
        )
        session_id = request.session[SESSION_ID_KEY]
        previous = self._sessions.pop(session_id, None)
        if previous is not None:
            await self._stop_sandbox(previous.sandbox, role="agent", task_name=previous.task.task_name)
        self._sessions[session_id] = AgentSession(
            task=task, sandbox=sandbox, docker_image=body.docker_image, started_at=monotonic()
        )
        return TerminalBench4SeedSessionResponse(sandbox_handle=sandbox._handle.sandbox_id)

    async def verify(self, request: Request, body: TerminalBench4VerifyRequest) -> TerminalBench4VerifyResponse:
        verify_started_at = monotonic()
        session_id = request.session.get(SESSION_ID_KEY) or f"golden-{uuid4().hex}"
        session = self._sessions.pop(session_id, None)
        try:
            task = self._load_task(body)
            log_dir = self._session_log_dir(task, session_id)
        except Exception:
            # A bad task folder or an unwritable log dir must not orphan the seeded agent sandbox.
            if session is not None:
                await self._stop_sandbox(session.sandbox, role="agent", task_name=body.task_name)
            raise

        golden_output: Optional[str] = None
        golden_exit_code: Optional[int] = None
        if self.config.is_verifying_golden_patch:
            if session is None:
                agent_sandbox = await self._create_sandbox(
                    image=body.docker_image, resources=self._resources_for(task.environment), role="agent", task=task
                )
            else:
                agent_sandbox = session.sandbox
        else:
            if session is None:
                raise RuntimeError(
                    f"{task.task_name}: no agent sandbox for session {session_id!r}; seed_session must precede verify"
                )
            agent_sandbox = session.sandbox

        # -- agent-side phase: oracle, collect hooks, artifacts -------------------------------
        hook_results: List[Dict[str, Any]] = []
        probes: List[ArtifactProbe] = []
        tarball_local: Optional[Path] = None
        tarball_bytes = 0
        artifact_collection_time_s = 0.0
        agent_stopped = False
        temp_dir: Optional[tempfile.TemporaryDirectory] = None
        try:
            # Inside the try on purpose: a full node-local disk raised here once and, with the
            # directory created before the try, the agent sandbox leaked.
            temp_dir = tempfile.TemporaryDirectory(prefix="nemo-gym-tb4-artifacts-")
            if self.config.is_verifying_golden_patch:
                golden_exit_code, golden_output = await self._run_golden_solution(agent_sandbox, task)
            hook_results = await self._run_collect_hooks(agent_sandbox, task)
            collect_started_at = monotonic()
            probes, tarball_local, tarball_bytes = await self._collect_artifacts(
                agent_sandbox, task, Path(temp_dir.name), log_dir
            )
            artifact_collection_time_s = monotonic() - collect_started_at
        except Exception:
            if temp_dir is not None:
                temp_dir.cleanup()
            await self._stop_sandbox(agent_sandbox, role="agent", task_name=task.task_name)
            raise
        if self.config.stop_agent_sandbox_before_verify:
            agent_stopped = await self._stop_sandbox(agent_sandbox, role="agent", task_name=task.task_name)

        # -- verifier-side phase ----------------------------------------------------------------
        verifier_sandbox: Optional[AsyncSandbox] = None
        outcome = VerifierOutcome()
        verifier_started_at = monotonic()
        verifier_sandbox_start_time_s = 0.0
        try:
            verifier_sandbox = await self._create_sandbox(
                image=body.verifier_docker_image,
                resources=self._resources_for(task.effective_verifier_environment()),
                role="verifier",
                task=task,
            )
            verifier_sandbox_start_time_s = monotonic() - verifier_started_at
            await self._stage_verifier(verifier_sandbox, task, probes, tarball_local)
            outcome = await self._run_tests_and_read_reward(verifier_sandbox, task, log_dir)
        except Exception as error:
            outcome.error_type = type(error).__name__
            raise
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
            if not self.config.stop_agent_sandbox_before_verify:
                agent_stopped = await self._stop_sandbox(agent_sandbox, role="agent", task_name=task.task_name)
            verifier_wall_time_s = monotonic() - verifier_started_at
            observation = self._verifier_observation(verifier_sandbox, outcome, verifier_wall_time_s)
            await self._stop_sandbox(verifier_sandbox, role="verifier", task_name=task.task_name)

        failure_reason = None if outcome.evaluation_completed else (outcome.error or "verifier did not complete")
        explicit_only = set(TerminalBench4VerifyResponse.model_fields) - set(TerminalBench4VerifyRequest.model_fields)
        response = TerminalBench4VerifyResponse(
            **body.model_dump(exclude=explicit_only),
            reward=outcome.reward,
            failure_reason=failure_reason,
            evaluation_completed=outcome.evaluation_completed,
            verification_time_taken=monotonic() - verify_started_at,
            verifier_exit_code=outcome.exit_code,
            verifier_error=outcome.error,
            reward_source=outcome.reward_source,
            rewards=outcome.rewards,
            test_output=tail(outcome.test_output, self.config.max_test_output_chars),
            artifact_manifest=[p.as_dict() for p in probes],
            artifacts_packed=sum(1 for p in probes if p.status == "ok"),
            artifact_tarball_bytes=tarball_bytes,
            artifact_tar_return_code=self._last_pack.get("tar_return_code"),
            artifact_tar_warnings=self._last_pack.get("tar_warnings", ""),
            artifact_excludes=list(self._last_pack.get("excludes", [])),
            verifier_stderr_tail=outcome.stderr_tail,
            verifier_wall_time_s=outcome.wall_time_s,
            collect_hook_results=hook_results,
            agent_sandbox_stopped=agent_stopped,
            artifact_collection_time_s=artifact_collection_time_s,
            verifier_sandbox_start_time_s=verifier_sandbox_start_time_s,
            golden_patch_exit_code=golden_exit_code,
            golden_patch_output=golden_output,
            log_dir=str(log_dir),
            verifier_sandbox_observation=observation,
        )
        (log_dir / "verify_summary.json").write_text(
            json.dumps(response.model_dump(exclude={"response", "responses_create_params"}), indent=1, default=str)
        )
        return response

    # -- agent-side steps -------------------------------------------------------------------

    async def _run_golden_solution(self, sandbox: AsyncSandbox, task: TB4Task) -> Tuple[Optional[int], str]:
        solution_dir = task.solution_dir
        solve_sh = solution_dir / "solve.sh"
        if not solve_sh.is_file():
            raise FileNotFoundError(f"{task.task_name}: golden mode needs {solve_sh}")
        files = sorted(p for p in solution_dir.rglob("*") if p.is_file())
        remote_dirs = sorted(
            {f"{SOLUTION_DIR}/{p.parent.relative_to(solution_dir).as_posix()}".rstrip("/.") for p in files}
        )
        mkdir_result = await sandbox.exec(
            labeled("solution-mkdir", "mkdir -p " + " ".join(shlex.quote(d) for d in [SOLUTION_DIR, *remote_dirs])),
            timeout_s=60,
        )
        if mkdir_result.return_code != 0:
            raise OracleUnsupportedError(
                f"{task.task_name}: cannot create {SOLUTION_DIR} as the image user (rc {mkdir_result.return_code}: "
                f"{(mkdir_result.stderr or '')[-500:]}). Harbor places /solution with a daemon-level copy; this provider "
                "has no root escalation, so oracle mode is unavailable on non-root agent images (rollouts are unaffected)"
            )
        for local_path in files:
            remote_path = f"{SOLUTION_DIR}/{local_path.relative_to(solution_dir).as_posix()}"
            await sandbox.upload(local_path, remote_path)
        chmod_result = await sandbox.exec(
            labeled("solution-chmod", f"chmod -R a+rX {SOLUTION_DIR} && chmod +x {SOLUTION_DIR}/solve.sh"),
            timeout_s=60,
        )
        if chmod_result.return_code != 0:
            raise RuntimeError(f"{task.task_name}: chmod on {SOLUTION_DIR} failed: {chmod_result.stderr}")
        self._log(f"running golden solution for {task.task_name}")
        try:
            result = await sandbox.exec(
                labeled("run-solution", f"bash {SOLUTION_DIR}/solve.sh"),
                timeout_s=self.config.golden_patch_timeout_s,
                env={"DEBIAN_FRONTEND": "noninteractive", **task.solution_env},  # what Harbor's OracleAgent sets
            )
        except Exception as error:
            return None, f"golden solution raised {type(error).__name__}: {error}"
        output = (result.stdout or "") + (result.stderr or "")
        if result.error_type:
            output += f"\n[error_type={result.error_type}]"
        return result.return_code, tail(output, self.config.max_test_output_chars)

    async def _run_collect_hooks(self, sandbox: AsyncSandbox, task: TB4Task) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for hook in task.sidecar_collect_hooks:
            results.append({"service": hook.service, "command": hook.command, "status": "skipped_sidecar"})
        for hook in task.main_collect_hooks:
            record: Dict[str, Any] = {"service": hook.service, "command": hook.command}
            try:
                result = await sandbox.exec(
                    labeled("collect-hook", hook.command),
                    timeout_s=hook.timeout_sec or self.config.collect_hook_default_timeout_s,
                    user=hook.user,
                )
                record.update(
                    status="ok" if result.return_code == 0 and not result.error_type else "failed",
                    return_code=result.return_code,
                    error_type=result.error_type,
                    output_tail=tail((result.stdout or "") + (result.stderr or ""), 2000),
                )
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
            results.append(record)
        return results

    async def _collect_artifacts(
        self, sandbox: AsyncSandbox, task: TB4Task, temp_dir: Path, log_dir: Path
    ) -> Tuple[List[ArtifactProbe], Optional[Path], int]:
        probes: List[ArtifactProbe] = []
        for artifact in task.sidecar_artifacts:
            probes.append(ArtifactProbe(artifact.source, artifact.service, "unknown", "skipped_sidecar"))

        main_artifacts: List[ArtifactEntry] = []
        seen: set[str] = set()
        for artifact in task.main_artifacts:
            if artifact.normalized_source in seen:
                continue
            seen.add(artifact.normalized_source)
            main_artifacts.append(artifact)

        probe_result = await sandbox.exec(
            build_probe_command([a.normalized_source for a in main_artifacts]), timeout_s=120
        )
        if probe_result.error_type or probe_result.return_code != 0:
            raise RuntimeError(
                f"{task.task_name}: artifact probe failed (rc {probe_result.return_code}, "
                f"error_type {probe_result.error_type}): {(probe_result.stderr or '')[-500:]}"
            )
        kinds = parse_probe_output(probe_result.stdout)
        members: List[str] = []
        excludes: List[str] = []
        for artifact in main_artifacts:
            kind = kinds.get(artifact.normalized_source, "missing")
            status = "ok" if kind in ("dir", "file") else "missing"
            probes.append(ArtifactProbe(artifact.source, artifact.service, kind, status, artifact.relative_to_root))
            if status == "ok" and artifact.relative_to_root:
                members.append(artifact.relative_to_root)
                if kind == "dir" and artifact.exclude:
                    excludes.extend(translate_excludes(artifact.relative_to_root, artifact.exclude))

        tarball_local: Optional[Path] = None
        tarball_bytes = 0
        self._last_pack = {"tar_return_code": None, "tar_warnings": "", "excludes": excludes}
        if members:
            pack_result = await sandbox.exec(build_pack_command(members, excludes=excludes), timeout_s=1800)
            if pack_result.error_type or pack_result.return_code != 0:
                raise RuntimeError(
                    f"{task.task_name}: packing artifacts failed (rc {pack_result.return_code}, "
                    f"error_type {pack_result.error_type}): {(pack_result.stderr or '')[-800:]}"
                )
            try:
                tarball_bytes, tar_rc, tar_warnings = parse_pack_output(pack_result.stdout)
            except ValueError as error:
                raise RuntimeError(
                    f"{task.task_name}: could not read the pack result: {pack_result.stdout!r}"
                ) from error
            self._last_pack.update(tar_return_code=tar_rc, tar_warnings=tail(tar_warnings, 4000))
            if tar_rc == 1:
                self._log(
                    f"{task.task_name}: tar exited 1 (a member changed while read); archive kept: {tar_warnings[-300:]}"
                )
            if tarball_bytes > self.config.artifact_max_bytes:
                raise RuntimeError(
                    f"{task.task_name}: artifact tarball is {tarball_bytes} bytes, above artifact_max_bytes="
                    f"{self.config.artifact_max_bytes}"
                )
            tarball_local = temp_dir / "artifacts.tgz"
            await sandbox.download(REMOTE_TARBALL, tarball_local)
            local_size = tarball_local.stat().st_size
            if local_size != tarball_bytes:
                raise RuntimeError(
                    f"{task.task_name}: downloaded tarball is {local_size} bytes, sandbox reported {tarball_bytes}"
                )
        self._log(
            f"{task.task_name}: artifacts probed={len(main_artifacts)} packed={len(members)} bytes={tarball_bytes}"
        )
        (log_dir / "artifact_manifest.json").write_text(
            json.dumps(
                {"tarball_bytes": tarball_bytes, "entries": [p.as_dict() for p in probes], **self._last_pack}, indent=1
            )
        )
        return probes, tarball_local, tarball_bytes

    # -- verifier-side steps ----------------------------------------------------------------

    async def _stage_verifier(
        self, sandbox: AsyncSandbox, task: TB4Task, probes: Sequence[ArtifactProbe], tarball_local: Optional[Path]
    ) -> None:
        dir_targets = [p.source.rstrip("/") or "/" for p in probes if p.status == "ok" and p.kind == "dir"]
        file_targets = [p.source for p in probes if p.status == "ok" and p.kind == "file"]
        prepare_result = await sandbox.exec(build_prepare_targets_command(dir_targets, file_targets), timeout_s=300)
        if prepare_result.error_type or prepare_result.return_code != 0:
            raise RuntimeError(
                f"{task.task_name}: preparing verifier directories failed (rc {prepare_result.return_code}): "
                f"{(prepare_result.stderr or '')[-800:]}"
            )
        if tarball_local is not None:
            await sandbox.upload(tarball_local, REMOTE_TARBALL)
            extract_result = await sandbox.exec(build_extract_command(), timeout_s=1800)
            if extract_result.error_type or extract_result.return_code != 0:
                raise RuntimeError(
                    f"{task.task_name}: extracting artifacts in the verifier failed (rc {extract_result.return_code}): "
                    f"{(extract_result.stderr or '')[-800:]}"
                )
        tests_result = await sandbox.exec(
            labeled("check-tests", f"test -f {TEST_SCRIPT} && chmod +x {TEST_SCRIPT}"), timeout_s=60
        )
        if tests_result.error_type or tests_result.return_code != 0:
            raise RuntimeError(
                f"{task.task_name}: verifier image has no executable {TEST_SCRIPT} (rc {tests_result.return_code}); "
                "the prebuilt verifier image must bake tests/ into /tests"
            )

    async def _run_tests_and_read_reward(self, sandbox: AsyncSandbox, task: TB4Task, log_dir: Path) -> VerifierOutcome:
        outcome = VerifierOutcome()
        timeout_s = self._verifier_timeout_s(task)
        self._log(f"{task.task_name}: running {TEST_SCRIPT} with timeout {timeout_s:g}s")
        started_at = monotonic()
        try:
            run_result = await sandbox.exec(
                labeled("run-tests", f"({TEST_SCRIPT}) > {TEST_STDOUT} 2>&1"),
                timeout_s=timeout_s,
                env=task.verifier_env or None,
                user=task.verifier_user,
            )
        except TimeoutError as error:
            # The provider's client-side backstop; a timeout is data (Harbor: VerifierTimeoutError).
            outcome.wall_time_s = monotonic() - started_at
            outcome.error = f"verifier timed out after its {timeout_s:g}s budget ({type(error).__name__}: {error})"
            outcome.error_type = "timeout"
            await self._download_verifier_files(sandbox, log_dir, outcome, reward_required=False)
            return outcome
        # Any other exception (dead backend, transport failure) propagates: invalidated, not scored.
        outcome.wall_time_s = monotonic() - started_at
        outcome.exit_code = run_result.return_code
        outcome.stderr_tail = tail(run_result.stderr or "", 4000)
        if run_result.error_type:
            outcome.error = f"verifier command reported error_type={run_result.error_type}"
            outcome.error_type = run_result.error_type
            await self._download_verifier_files(sandbox, log_dir, outcome, reward_required=False)
            return outcome
        if outcome.wall_time_s >= timeout_s:
            # execd enforced the budget itself and returned an ordinary exit code (opensandbox never
            # reports error_type="timeout"); a reward file written before the kill must not be scored.
            outcome.error = (
                f"verifier exceeded its {timeout_s:g}s budget (wall {outcome.wall_time_s:.1f}s, "
                f"rc {run_result.return_code})"
            )
            outcome.error_type = "timeout"
            await self._download_verifier_files(sandbox, log_dir, outcome, reward_required=False)
            return outcome

        sizes = await self._download_verifier_files(sandbox, log_dir, outcome, reward_required=True)
        source = REWARD_JSON if REWARD_JSON in sizes else REWARD_TXT if REWARD_TXT in sizes else None
        if source is None:
            outcome.error = f"verifier wrote neither {VERIFIER_DIR}/{REWARD_JSON} nor {VERIFIER_DIR}/{REWARD_TXT}"
            return outcome
        try:
            outcome.reward, outcome.rewards = parse_reward_payload(source, (log_dir / source).read_bytes())
        except RewardParseError as error:
            outcome.error = str(error)
            return outcome
        outcome.reward_source = source
        outcome.evaluation_completed = True
        return outcome

    async def _download_verifier_files(
        self, sandbox: AsyncSandbox, log_dir: Path, outcome: VerifierOutcome, *, reward_required: bool
    ) -> Dict[str, int]:
        """Download reward files, ctrf.json and test-stdout.txt when present; return their sizes.

        With ``reward_required`` the probe and the reward-file downloads are load-bearing and raise on
        failure (infrastructure); ctrf.json and the stdout capture are always best-effort.
        """
        try:
            probe = await sandbox.exec(build_verifier_files_probe_command(), timeout_s=60)
        except Exception as error:
            if reward_required:
                raise RuntimeError(
                    f"probing {VERIFIER_DIR} for reward files raised {type(error).__name__}: {error}"
                ) from error
            outcome.error = (outcome.error or "") + f"; probing {VERIFIER_DIR} raised {type(error).__name__}: {error}"
            return {}
        if probe.error_type or probe.return_code != 0:
            message = f"probing {VERIFIER_DIR} failed (rc {probe.return_code}, error_type {probe.error_type})"
            if reward_required:
                raise RuntimeError(message)
            outcome.error = (outcome.error or "") + "; " + message
            return {}
        sizes = parse_verifier_files_probe(probe.stdout)
        for name, size in list(sizes.items()):
            local_path = log_dir / name
            if size <= 0:
                # Present but empty: materialize it so parse_reward_payload reports "is empty" (Harbor: RewardFileEmptyError).
                local_path.write_bytes(b"")
                continue
            try:
                await sandbox.download(f"{VERIFIER_DIR}/{name}", local_path)
            except Exception as error:
                if reward_required and name in (REWARD_JSON, REWARD_TXT):
                    raise RuntimeError(
                        f"downloading {VERIFIER_DIR}/{name} raised {type(error).__name__}: {error}"
                    ) from error
                print(f"Failed to download {VERIFIER_DIR}/{name}: {format_exc()}", file=sys.stderr)
                sizes.pop(name, None)
        stdout_path = log_dir / "test-stdout.txt"
        if stdout_path.exists():
            outcome.test_output = stdout_path.read_text(encoding="utf-8", errors="replace")
        return sizes

    def _verifier_observation(
        self, sandbox: Optional[AsyncSandbox], outcome: VerifierOutcome, wall_time_s: float
    ) -> SandboxObservation:
        handle = getattr(sandbox, "_handle", None) if sandbox is not None else None
        normalized_error = (outcome.error_type or "").lower()
        if "timeout" in normalized_error:
            result = "timeout"
        elif outcome.error_type:
            result = "sandbox_error"
        elif outcome.evaluation_completed:
            result = "completed"
        else:
            result = "failed"
        return SandboxObservation(
            role="verifier",
            provider=getattr(handle, "provider_name", None),
            sandbox_id=getattr(handle, "sandbox_id", None),
            outcome=result,
            exit_code=outcome.exit_code,
            wall_time_s=max(0.0, wall_time_s),
            error_type=outcome.error_type,
        )


if __name__ == "__main__":
    TerminalBench4ResourcesServer.run_webserver()
