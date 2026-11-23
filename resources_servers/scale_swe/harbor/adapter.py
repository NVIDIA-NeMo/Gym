# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Scale-SWE training rows into Harbor task folders, one folder per row.

Adapted from the SWE rows-to-Harbor converter PR (#5789) (``scripts/swe_rows_to_harbor/convert.py``,
``ScaleSWEAdapter``), with its known parity gaps fixed. A converted task runs the episode the Scale-SWE Resources
Server runs under ``single_agent_turn`` (``resources_servers/scale_swe/app.py``, episode-session path):

- **Sandbox**: the row's own image. ``environment/Dockerfile`` is overlay mode (``FROM`` + ``RUN``, nothing is
  built): the first ``RUN`` unpacks a small bundle into ``/var/lib/nemo_gym``, the second runs its ``setup.sh``,
  which repeats ``_seed_session`` step by step with the server's own per-step time limits and tolerance:
  ``pre_commands`` (180 s), a git repository when the image has none, the anti-cheat history scrub (600 s), a
  committer identity (60 s), the base commit and the files already untracked. As at the server, only a base commit
  that ``git rev-parse HEAD`` does not name fails the seed. ``[environment].build_timeout_sec`` covers the sum.
- **Prompt**: ``instruction.md`` is the row's single user message (``responses_create_params.input[0].content``,
  the ``problem_statement``), byte for byte; that is what ``single_agent_turn`` hands ``opencode_agent``.
- **Patch**: a ``[[verifier.collect]]`` hook runs ``capture.py`` in the agent's sandbox, which calls the server's own
  ``capture_model_patch`` (``resources_servers/swebench/patch_capture.py``, vendored) with the server's
  ``patch_capture_mode``, and records whether the capture succeeded, hit a git error or failed for infrastructure.
- **Grading**: ``environment_mode = "separate"`` on the unmodified row image. ``tests/test.sh`` runs
  ``tests/nemo_gym/grade.py``, which calls the server's own ``verification.py`` (vendored unchanged). It exits 75
  (masked) only where no agent code had run: a workdir missing from the image, a reward folder or mapped file it
  cannot protect, a grader error before the tests, or an eval script that stopped before ``TEST_OUTPUT_BEGIN``. Any
  other run without a verdict is a completed 0 (``no_verdict_after_tests``), and a grader failure after the tests
  started exits 1 without a reward, a measured 0. A capture that yielded no patch is a completed 0, as
  everything the hook produced came from the agent's sandbox. The eval script runs as root without the capabilities
  that reach the reward folder, which ``grade.py`` hands to another uid first, so the agent's code cannot create or
  rewrite the reward file. Sandbox resources come from the server's config; the harbor server derives the same
  CPU-count hints (``cpu_cap_env``) from ``cpus`` that the Resources Server adds.
- **Golden patch**: ``solution/solve.sh`` applies the row's ``patch`` with the eval script's ``git apply`` flags, so
  ``gym dataset validate`` checks seed, capture and grading end to end.

Hidden data (the golden patch, ``f2p_script``, ``f2p_patch``, ``FAIL_TO_PASS``, ``PASS_TO_PASS``) lives only in
``solution/`` and ``tests/``, which the agent's sandbox never receives.

    python resources_servers/scale_swe/harbor/adapter.py --rows rows.jsonl --output-dir <dir> \\
        [--ids-file ids.txt] [--native-rows-out native.jsonl] [--id-map-out id_map.json] \\
        [--native-config <scale_swe server config or `gym env resolve` dump>]

Rows are the JSONL ``prepare_scale_swe.py`` writes. The output folder must be empty or absent. Exit 2 for a usage
or input error, including every native server setting the task folders cannot reproduce.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TEMPLATE = HERE / "task-template"
SERVER_CONFIG = REPO_ROOT / "resources_servers/scale_swe/configs/scale_swe.yaml"
SERVER_INSTANCE = "scale_swe_resources_server"
ANTI_CHEAT_MODULE = "resources_servers/swebench/anti_cheat_setup.sh"
ANTI_CHEAT_SCRIPT = REPO_ROOT / ANTI_CHEAT_MODULE
# The server modules a converted task runs unchanged, by their path in this repository.
VERIFICATION_MODULE = "resources_servers/scale_swe/verification.py"
PATCH_CAPTURE_MODULE = "resources_servers/swebench/patch_capture.py"
VENDORED_LICENSE = "Apache-2.0"

STATE_DIR = "/var/lib/nemo_gym"
ARTIFACTS_DIR = "/logs/artifacts/nemo_gym"
# The server's per-step exec limits during the seed (app._seed_session): AsyncSandbox.exec's default (180 s) for
# pre_commands, the git checks, rev-parse and ls-files; 600 s for the anti-cheat scrub; 60 s for the identity.
EXEC_DEFAULT_TIMEOUT_SEC = 180
ANTI_CHEAT_TIMEOUT_SEC = 600
IDENTITY_TIMEOUT_SEC = 60
# Sum of the seed's step limits (5 x 180 + 600 + 60 = 1560) plus room to unpack the bundle.
BUILD_TIMEOUT_SEC = 1800
# capture_model_patch runs at most four git commands, each limited to 300 s.
CAPTURE_TIMEOUT_SEC = 4 * 300 + 60
# Room for grade.py around the server's evaluation_timeout, which bounds the test run itself.
VERIFIER_TIMEOUT_MARGIN_SEC = 300
# opencode_agent's own `timeout` (10800 s), the agent budget single_agent_turn leaves it (no per-row timeout).
DEFAULT_AGENT_TIMEOUT_SEC = 10800
DEFAULT_DATASET_NAME = "nemo-gym/scale-swe"
# The sandbox resources a task folder can state (`cpus`, `memory_mb`, `storage_mb`).
RESOURCE_KEYS = ("cpu", "memory_mib", "disk_gib")
_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_TASK_NAME = re.compile(r"[A-Za-z0-9._-]+")
# The image reference goes raw into the Dockerfile's FROM line and the workdir into setup.sh's messages: no
# whitespace or control characters (a newline would add a Dockerfile line), no shell metacharacters.
_IMAGE_URL = re.compile(r"[^\s\x00-\x1f\x7f]+")
_WORKDIR = re.compile(r"/[A-Za-z0-9._@+/-]*")


class AdapterError(ValueError):
    """A usage or input error, or a native setting the task folders cannot reproduce (exit 2)."""


def _load_repo_module(relative: str) -> ModuleType:
    """Import a module of this checkout by path, so the adapter reads exactly the code it vendors."""
    name = "scale_swe_harbor_" + relative.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str | None:
    """Output of a git command on this checkout, or None outside a git checkout (an installed wheel, a tarball).

    Only provenance comes from git (the commit in the vendored headers); the vendored sha256 identifies the code.
    """
    try:
        result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path) -> dict[str, Any]:
    """A Gym config file with the files its ``config_paths`` names merged under it (repository-relative)."""
    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, dict):
        raise AdapterError(f"{path.name} is not a YAML mapping")
    merged: dict[str, Any] = {}
    for included in document.pop("config_paths", None) or []:
        included_path = Path(included) if Path(included).is_absolute() else REPO_ROOT / included
        merged = _merge(merged, _load_config(included_path))
    return _merge(merged, document)


@dataclass(frozen=True)
class ServerSettings:
    """What the converted tasks take from the ``scale_swe_resources_server`` config."""

    evaluation_timeout: int
    resources: dict[str, Any]
    env: dict[str, str]
    apply_anti_cheating: bool
    patch_capture_mode: str

    @classmethod
    def read(cls, path: Path | None = None) -> ServerSettings:
        """Read the server block of ``configs/scale_swe.yaml``, or of ``--native-config``: a pairing YAML such as
        ``configs/scale_swe_opencode.yaml`` (its ``config_paths`` merged in) or a ``gym env resolve`` dump."""
        config_path = path or SERVER_CONFIG
        try:
            server = _load_config(config_path)[SERVER_INSTANCE]["resources_servers"]["scale_swe"]
        except (KeyError, TypeError) as error:
            raise AdapterError(
                f"no {SERVER_INSTANCE}.resources_servers.scale_swe block in {config_path.name}"
            ) from error
        sandbox_config = server.get("sandbox_config") or {}
        resources = dict(sandbox_config.get("resources") or {})
        cpu = resources.get("cpu")
        refusals = {
            # Golden-patch mode is the task's solution/, run by `gym dataset validate`, not a server setting.
            "is_verifying_golden_patch: true": server.get("is_verifying_golden_patch", False) is True,
            # The agent-sandbox egress block (model host only) has no task-folder equivalent.
            "enforce_agent_no_network: true": server.get("enforce_agent_no_network", False) is True,
            "sandbox_config.provider_options.network_policy": "network_policy"
            in (sandbox_config.get("provider_options") or {}),
            # The harbor server always adds the CPU-count hints for a sandbox with a CPU limit.
            "sandbox_config.derive_cpu_env: false": sandbox_config.get("derive_cpu_env", True) is False,
            # The verifier needs a time limit; the task's [verifier].timeout_sec is derived from it.
            "evaluation_timeout: null": "evaluation_timeout" in server and server["evaluation_timeout"] is None,
            # The server's cpu is a float; a task's `cpus` is a whole number.
            f"sandbox_config.resources.cpu: {cpu} (not a whole number)": cpu is not None
            and float(cpu) != int(float(cpu)),
            **{
                f"sandbox_config.resources.{key}": True
                for key in sorted(resources)
                if key not in RESOURCE_KEYS and resources[key] is not None
            },
        }
        refused = [name for name, on in refusals.items() if on]
        if refused:
            raise AdapterError(f"native settings the task folders cannot reproduce: {', '.join(refused)}")
        return cls(
            evaluation_timeout=int(server.get("evaluation_timeout", 1200)),
            resources=resources,
            env={str(key): str(value) for key, value in (sandbox_config.get("env") or {}).items()},
            apply_anti_cheating=bool(server.get("apply_anti_cheating", True)),
            patch_capture_mode=str(server.get("patch_capture_mode", "worktree")),
        )


class ScaleSWEHarborAdapter:
    """Writes one task folder per Scale-SWE row."""

    def __init__(
        self, *, settings: ServerSettings | None = None, agent_timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC
    ) -> None:
        self.settings = settings or ServerSettings.read()
        self.agent_timeout_sec = agent_timeout_sec
        self.verification = _load_repo_module(VERIFICATION_MODULE)
        patch_capture = _load_repo_module(PATCH_CAPTURE_MODULE)
        if self.settings.patch_capture_mode not in patch_capture.PATCH_CAPTURE_MODES:
            raise AdapterError(f"unknown patch_capture_mode {self.settings.patch_capture_mode!r}")
        self.git_email = patch_capture.GIT_COMMIT_IDENTITY_EMAIL
        self.git_name = patch_capture.GIT_COMMIT_IDENTITY_NAME
        self.gym_commit = _git("rev-parse", "HEAD") or "unknown"
        self.vendored = {
            relative: self._vendored_text(relative) for relative in (VERIFICATION_MODULE, PATCH_CAPTURE_MODULE)
        }

    def _vendored_text(self, relative: str) -> str:
        """The module's text, unchanged, under a one-line header naming its source, commit, sha256 and license."""
        original = (REPO_ROOT / relative).read_text()
        modified = bool(_git("status", "--porcelain", "--", relative) or "")
        header = (
            f"# Vendored unchanged from {relative} at {self.gym_commit}"
            f"{' (uncommitted changes)' if modified else ''}; source sha256 {_sha256(original.encode())};"
            f" license {VENDORED_LICENSE}\n"
        )
        return header + original

    def vendored_digests(self) -> dict[str, str]:
        """sha256 of each file a task carries from elsewhere in this repository, before any header."""
        relatives = (VERIFICATION_MODULE, PATCH_CAPTURE_MODULE, ANTI_CHEAT_MODULE)
        return {relative: _sha256((REPO_ROOT / relative).read_bytes()) for relative in sorted(relatives)}

    # -- seed -----------------------------------------------------------------------------------------------------

    def setup_script(self, row: dict[str, Any]) -> str:
        """``app.ScaleSWEResourcesServer._seed_session``, step by step, run once before the agent starts."""
        workdir = row["workdir"]
        wd = shlex.quote(workdir)
        lines = [
            "#!/bin/bash",
            f"# Seed of resources_servers/scale_swe/app.py (_seed_session) for {row['instance_id']}.",
            "# Generated by resources_servers/scale_swe/harbor/adapter.py."
            " Every step keeps the server's tolerance and",
            "# time limit; only a base commit that `git rev-parse HEAD` does not name fails the seed,"
            " as at the server.",
            f"S={STATE_DIR}",
            'note() { echo "[nemo_gym setup] $*" >&2; }',
        ]
        pre = self.verification.clean_commands(row.get("pre_commands") or "")
        if pre:
            lines += [
                "# pre_commands: checkout of the parent commit and the dataset's own history scrub.",
                f"(cd {wd} && timeout {EXEC_DEFAULT_TIMEOUT_SEC} bash -c {shlex.quote(pre)})"
                " || note 'pre_commands failed (non-fatal, as at the server)'",
            ]
        init = (
            f"cd {wd} && git init -q && git config user.email nemo-gym@nvidia.com && git config user.name nemo-gym "
            "&& git add -A && git commit -q -m 'nemo_gym: initial snapshot' --allow-empty"
        )
        lines += [
            "# _ensure_git_repo: a fresh repository only when the image has none.",
            f"if ! timeout {EXEC_DEFAULT_TIMEOUT_SEC} git -C {wd} rev-parse --git-dir >/dev/null 2>&1; then",
            f"  timeout {EXEC_DEFAULT_TIMEOUT_SEC} bash -c {shlex.quote(init)} || note 'git init failed (non-fatal)'",
            "fi",
        ]
        if self.settings.apply_anti_cheating:
            script_path = f"{workdir.rstrip('/')}/anti_cheat_setup.sh"
            command = (
                f"git reset --hard && WORKING_DIRECTORY={workdir} bash anti_cheat_setup.sh"
                " && rm -f anti_cheat_setup.sh"
            )
            lines += [
                "# apply_anti_cheat_setup (required=False): upload the shared scrub into the workdir, then run it.",
                f'cp "$S/anti_cheat_setup.sh" {shlex.quote(script_path)}'
                f" && (cd {wd} && timeout {ANTI_CHEAT_TIMEOUT_SEC} bash -c {shlex.quote(command)})"
                " || note 'anti-cheat setup failed (non-fatal, as at the server)'",
            ]
        email, name = shlex.quote(self.git_email), shlex.quote(self.git_name)
        identity = (
            f"git config --global user.email {email} && git config --global user.name {name}"
            f" && git config --global --add safe.directory {wd}"
            f" ; git -C {wd} config user.email {email} ; git -C {wd} config user.name {name}"
        )
        lines += [
            "# prepare_git_for_commits: a committer identity, so the agent can commit (never fatal).",
            f"timeout {IDENTITY_TIMEOUT_SEC} bash -c {shlex.quote(identity)} || note 'git identity setup failed'",
            "# The base commit the capture diffs against. The server refuses the seed without one (503, retry the",
            "# episode): without it the agent's patch could never be captured.",
            f"base=$(timeout {EXEC_DEFAULT_TIMEOUT_SEC} git -C {wd} rev-parse HEAD 2>/dev/null | tr -d '[:space:]')",
            "if ! printf '%s' \"$base\" | grep -Eqx '[0-9a-f]{40}([0-9a-f]{24})?'; then",
            f"  echo \"[nemo_gym setup] git rev-parse HEAD did not name a commit in {workdir}: '$base'\" >&2",
            "  exit 1",
            "fi",
            'printf \'%s\\n\' "$base" > "$S/base_commit"',
            "# _pristine_untracked_files: what was untracked before the agent; a failure leaves the list empty.",
            f"timeout {EXEC_DEFAULT_TIMEOUT_SEC} git -C {wd} ls-files --others --exclude-standard"
            ' > "$S/pristine_untracked.txt" 2>/dev/null || : > "$S/pristine_untracked.txt"',
            "exit 0",
        ]
        return "\n".join(lines) + "\n"

    def bundle(self, row: dict[str, Any], setup_script: str) -> bytes:
        """A gzipped tar of what ``/var/lib/nemo_gym`` holds in the agent's sandbox, byte for byte reproducible."""
        files = {
            "setup.sh": setup_script,
            "config.json": json.dumps(
                {"workdir": row["workdir"], "patch_capture_mode": self.settings.patch_capture_mode}, indent=2
            )
            + "\n",
            "anti_cheat_setup.sh": ANTI_CHEAT_SCRIPT.read_text(),
            "capture/capture.py": (TEMPLATE / "environment/capture/capture.py").read_text(),
            "capture/local_sandbox.py": (TEMPLATE / "tests/nemo_gym/local_sandbox.py").read_text(),
            f"capture/{PATCH_CAPTURE_MODULE}": self.vendored[PATCH_CAPTURE_MODULE],
            f"capture/{VERIFICATION_MODULE}": self.vendored[VERIFICATION_MODULE],
        }
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for path in sorted(files):
                data = files[path].encode()
                info = tarfile.TarInfo(path)
                info.size, info.mtime, info.mode = len(data), 0, 0o755 if path.endswith(".sh") else 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                archive.addfile(info, io.BytesIO(data))
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, compresslevel=9) as gz:
            gz.write(raw.getvalue())
        return compressed.getvalue()

    def dockerfile(self, row: dict[str, Any], bundle: bytes) -> str:
        encoded = base64.b64encode(bundle).decode()
        return (
            f"# Generated by resources_servers/scale_swe/harbor/adapter.py for {row['instance_id']}.\n"
            "# Overlay mode: Gym starts the image and runs these RUN lines in the sandbox before the agent;"
            " nothing is\n"
            "# built. The first unpacks the seed bundle (setup.sh, the capture hook and the server modules it calls;\n"
            "# a copy of setup.sh sits beside this file), the second runs the seed.\n"
            f"FROM {row['image_url']}\n"
            f"RUN mkdir -p {STATE_DIR} && printf '%s' '{encoded}' | base64 -d | tar -xzf - -C {STATE_DIR}\n"
            f"RUN bash {STATE_DIR}/setup.sh\n"
        )

    # -- task folder ----------------------------------------------------------------------------------------------

    def task_toml(self, row: dict[str, Any]) -> str:
        sandbox = {"workdir": row["workdir"], **_resource_keys(self.settings.resources)}
        if self.settings.env:
            sandbox["env"] = dict(self.settings.env)
        capture_command = (
            f"rm -rf {ARTIFACTS_DIR}; python3 -I {STATE_DIR}/capture/capture.py;"
            f" test -f {ARTIFACTS_DIR}/capture.json || {{ mkdir -p {ARTIFACTS_DIR} && echo"
            f' \'{{"status": "runner_failed", "error": "python3 could not run capture.py"}}\''
            f" > {ARTIFACTS_DIR}/capture.json; }}"
        )
        return _toml(
            f"Generated by resources_servers/scale_swe/harbor/adapter.py from Scale-SWE row {row['instance_id']}.",
            {"schema_version": "1.4"},
            [
                (
                    "[metadata]",
                    {
                        "source_server": "scale_swe",
                        "instance_id": row["instance_id"],
                        "repo": row.get("repo") or "",
                        "image": row["image_url"],
                    },
                ),
                # No docker_image: the overlay Dockerfile names the image and its RUN lines are the seed.
                ("[environment]", {**sandbox, "build_timeout_sec": BUILD_TIMEOUT_SEC}),
                ("[agent]", {"timeout_sec": self.agent_timeout_sec}),
                (
                    "[verifier]",
                    {
                        "environment_mode": "separate",
                        "timeout_sec": self.settings.evaluation_timeout + VERIFIER_TIMEOUT_MARGIN_SEC,
                    },
                ),
                # The server grades in a fresh sandbox of the plain image: no seed, only its eval script.
                ("[verifier.environment]", {"docker_image": row["image_url"], **sandbox}),
                (
                    "[[verifier.collect]]",
                    {"service": "main", "timeout_sec": CAPTURE_TIMEOUT_SEC, "command": capture_command},
                ),
            ],
        )

    def task_files(self, row: dict[str, Any]) -> dict[str, str | bytes]:
        setup = self.setup_script(row)
        files: dict[str, str | bytes] = {
            "task.toml": self.task_toml(row),
            "instruction.md": _prompt(row),
            "environment/Dockerfile": self.dockerfile(row, self.bundle(row, setup)),
            "environment/setup.sh": setup,
            "tests/test.sh": (TEMPLATE / "tests/test.sh").read_text(),
            "tests/nemo_gym/grade.py": (TEMPLATE / "tests/nemo_gym/grade.py").read_text(),
            "tests/nemo_gym/local_sandbox.py": (TEMPLATE / "tests/nemo_gym/local_sandbox.py").read_text(),
            f"tests/nemo_gym/{VERIFICATION_MODULE}": self.vendored[VERIFICATION_MODULE],
            "tests/nemo_gym/task.json": json.dumps(
                {
                    "source": "scale_swe",
                    "evaluation_timeout": self.settings.evaluation_timeout,
                    "gym_commit": self.gym_commit,
                    "row": {
                        key: row[key]
                        for key in (
                            "instance_id",
                            "workdir",
                            "pre_commands",
                            "f2p_patch",
                            "f2p_script",
                            "FAIL_TO_PASS",
                            "PASS_TO_PASS",
                        )
                        if key in row
                    },
                },
                indent=2,
            )
            + "\n",
        }
        if (row.get("patch") or "").strip():
            files["solution/solve.sh"] = _solve_script(row["workdir"])
            files["solution/golden.patch"] = row["patch"]
        return files

    def write_task(self, row: dict[str, Any], task_dir: Path) -> None:
        for relative, content in self.task_files(row).items():
            path = task_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                # newline="" keeps a CRLF in the row's text as it is; Gym's reader turns it into LF (noted per task).
                with path.open("w", encoding="utf-8", newline="") as handle:
                    handle.write(content)
            path.chmod(0o755 if relative.endswith(".sh") else 0o644)


def _resource_keys(resources: dict[str, Any]) -> dict[str, int]:
    keys = {}
    if resources.get("cpu") is not None:
        keys["cpus"] = int(resources["cpu"])
    if resources.get("memory_mib") is not None:
        keys["memory_mb"] = int(resources["memory_mib"])
    if resources.get("disk_gib") is not None:
        keys["storage_mb"] = int(resources["disk_gib"]) * 1024
    return keys


def _prompt(row: dict[str, Any]) -> str:
    """The row's single user message: what ``single_agent_turn`` hands the agent."""
    params = row.get("responses_create_params") or {}
    messages = params.get("input") if isinstance(params, dict) else None
    if not isinstance(messages, list) or len(messages) != 1:
        raise AdapterError(f"{row.get('instance_id')}: expected responses_create_params.input to hold one message")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise AdapterError(f"{row.get('instance_id')}: expected one user message with string content")
    prompt = message["content"]
    if _CANARY_LINE.match(prompt.split("\n", 1)[0].strip()):
        raise AdapterError(f"{row.get('instance_id')}: the prompt opens with a line Gym drops as a canary marker")
    return prompt


def _solve_script(workdir: str) -> str:
    return (
        "#!/bin/bash\n"
        "# Applies the row's golden patch with the flags the server's eval script applies a patch with\n"
        "# (verification.build_eval_script), for `gym dataset validate`.\n"
        f"cd {shlex.quote(workdir)} || exit 1\n"
        'git apply --reject --recount --ignore-space-change --whitespace=nowarn "$(dirname "$0")/golden.patch"'
        " || true\n"
    )


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    """A TOML value for the few types the task files use; JSON string escapes are valid TOML ones."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value).replace("\x7f", "\\u007f")
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(key)} = {_toml_value(item)}" for key, item in value.items()) + " }"
    raise TypeError(f"no TOML form for {type(value).__name__}")


def _toml(header: str, top: dict[str, Any], tables: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [f"# {header}"]
    lines += [f"{_toml_key(key)} = {_toml_value(value)}" for key, value in top.items()]
    for name, table in tables:
        lines += ["", name] + [f"{_toml_key(key)} = {_toml_value(value)}" for key, value in table.items()]
    return "\n".join(lines) + "\n"


def read_ids(path: Path) -> list[str]:
    """The ``subset`` output: one instance id per line, ``#`` comments, order kept; a duplicate is an error."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AdapterError(f"{path.name} is not valid UTF-8 ({error.reason} at byte {error.start})") from error
    lines = [line.strip() for line in text.splitlines()]
    ids = [line for line in lines if line and not line.startswith("#")]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise AdapterError(f"{path.name}: duplicate id(s) {duplicates[:5]}")
    return ids


def select_rows(path: Path, ids: list[str] | None, limit: int = 0) -> list[tuple[str, int]]:
    """``(instance id, byte offset of its line)`` of the selected rows, in ``ids`` order (file order without ids).

    Streams the file once and checks every selected row (folder name, prompt) before anything is written; the
    caller reads each selected line again by its offset, so only offsets are held in memory.
    """
    wanted = set(ids) if ids is not None else None
    offsets: dict[str, int] = {}
    order: list[str] = []
    offset = 0
    with path.open("rb") as handle:
        for line in handle:
            start, offset = offset, offset + len(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AdapterError(f"line at byte {start} of {path.name} is not a JSON object")
            name = row.get("instance_id")
            if not isinstance(name, str):
                raise AdapterError(f"instance_id {name!r} is not a usable folder name")
            if wanted is not None and name not in wanted:
                continue
            # A name starting with "." could be "." or "..": the output folder itself or its parent.
            if not _TASK_NAME.fullmatch(name) or name.startswith("."):
                raise AdapterError(f"instance_id {name!r} is not a usable folder name")
            if name in offsets:
                raise AdapterError(f"{name} appears twice in {path.name}")
            for key in ("workdir", "image_url"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise AdapterError(f"{name}: {key} is missing or not a non-empty string")
            if not _IMAGE_URL.fullmatch(row["image_url"]):
                raise AdapterError(f"{name}: image_url {row['image_url']!r} holds whitespace or a control character")
            if not _WORKDIR.fullmatch(row["workdir"]):
                raise AdapterError(f"{name}: workdir {row['workdir']!r} is not an absolute path of [A-Za-z0-9._@+/-]")
            _prompt(row)
            try:
                # The task files are written as UTF-8: a lone surrogate (valid JSON) would fail halfway through.
                json.dumps(row, ensure_ascii=False).encode("utf-8")
            except UnicodeEncodeError as error:
                raise AdapterError(f"{name}: the row holds text that is not valid UTF-8 ({error.reason})") from error
            offsets[name] = start
            order.append(name)
    if ids is not None:
        missing = [item for item in ids if item not in offsets]
        if missing:
            raise AdapterError(f"{len(missing)} requested id(s) not in {path.name}: {missing[:5]}")
        order = ids
    if limit:
        order = order[:limit]
    return [(name, offsets[name]) for name in order]


def _read_line(path: Path, offset: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    return line if line.endswith(b"\n") else line + b"\n"


def _dataset_readme(
    *,
    adapter: ScaleSWEHarborAdapter,
    rows_path: Path,
    ids_path: Path | None,
    native_config: Path | None,
    limit: int,
    dataset_name: str,
    notes: list[str],
    count: int,
) -> str:
    """``<output-dir>/README.md``: the inputs by basename, their digests and the command that regenerates the
    folder."""
    rows_sha, rows_lines = hashlib.sha256(), 0
    with rows_path.open("rb") as handle:
        for line in handle:
            rows_sha.update(line)
            rows_lines += line.endswith(b"\n")
    regenerate = ["python resources_servers/scale_swe/harbor/adapter.py", f"--rows {rows_path.name}"]
    regenerate.append("--output-dir <dir>")
    if ids_path:
        regenerate.append(f"--ids-file {ids_path.name}")
    if native_config:
        regenerate.append(f"--native-config {native_config.name}")
    if limit:
        regenerate.append(f"--limit {limit}")
    if dataset_name != DEFAULT_DATASET_NAME:
        regenerate.append(f"--dataset-name {shlex.quote(dataset_name)}")
    if adapter.agent_timeout_sec != DEFAULT_AGENT_TIMEOUT_SEC:
        regenerate.append(f"--agent-timeout-sec {adapter.agent_timeout_sec}")
    ids_line = f" from `{ids_path.name}` (sha256 `{_sha256(ids_path.read_bytes())}`)" if ids_path else ""
    lines = [
        "# Scale-SWE as Harbor task folders",
        "",
        "Generated by `resources_servers/scale_swe/harbor/adapter.py`; that folder's README explains the conversion",
        "and how to run the dataset. Gym does not load this file as a task.",
        "",
        f"- Rows: `{rows_path.name}`, sha256 `{rows_sha.hexdigest()}`, {rows_lines} lines",
        f"- Selected: {count} task(s){ids_line}",
        f"- Gym commit: `{adapter.gym_commit}`",
        f"- Server settings ({native_config.name if native_config else 'configs/scale_swe.yaml'}):"
        f" `{json.dumps(asdict(adapter.settings), sort_keys=True)}`",
        f"- Agent timeout: {adapter.agent_timeout_sec} s",
        '- Separate verification needs no server option: every task sets `environment_mode = "separate"`.',
        "",
        "Vendored files (sha256 of the source file):",
        "",
        *[f"- `{relative}`: `{digest}`" for relative, digest in adapter.vendored_digests().items()],
        "",
        "Regenerate (from the repository root, at the Gym commit above):",
        "",
        "```bash",
        " \\\n    ".join(regenerate),
        "```",
        "",
    ]
    if notes:
        lines += ["Notes:", "", *[f"- {note}" for note in notes], ""]
    return "\n".join(lines)


def convert(
    rows_path: Path,
    output_dir: Path,
    *,
    ids_path: Path | None = None,
    native_rows_out: Path | None = None,
    id_map_out: Path | None = None,
    native_config: Path | None = None,
    limit: int = 0,
    dataset_name: str = DEFAULT_DATASET_NAME,
    agent_timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC,
) -> list[str]:
    """Write one task folder per selected row, plus ``dataset.toml`` and ``README.md``. Returns the task ids.

    With ``native_rows_out``, also write the selected raw lines in the same order (the native arm's rows); with
    ``id_map_out``, ``{line index in native_rows_out: task id}``, which is what native rollouts carry as
    ``_ng_task_index``.
    """
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise AdapterError(f"{output_dir} is not empty")
    if limit < 0:
        raise AdapterError(f"--limit must be 0 (all) or positive, not {limit}")
    for extra_out in (native_rows_out, id_map_out):
        # Checked before any task folder exists, so a bad path cannot leave a partial dataset behind.
        if extra_out is not None and not extra_out.resolve().parent.is_dir():
            raise AdapterError(f"the folder for {extra_out} does not exist")
    adapter = ScaleSWEHarborAdapter(settings=ServerSettings.read(native_config), agent_timeout_sec=agent_timeout_sec)
    ids = read_ids(ids_path) if ids_path else None
    selected = select_rows(rows_path, ids, limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    native_lines = []
    for name, offset in selected:
        line = _read_line(rows_path, offset)
        row = json.loads(line)
        adapter.write_task(row, output_dir / name)
        if "\r" in _prompt(row):
            notes.append(f"{name}: prompt has CR line endings; Gym's instruction.md reader turns them into LF")
        if native_rows_out is not None:
            native_lines.append(line)
    if native_rows_out is not None:
        native_rows_out.write_bytes(b"".join(native_lines))
    task_ids = [name for name, _ in selected]
    if id_map_out is not None:
        id_map = {str(index): task_id for index, task_id in enumerate(task_ids)}
        id_map_out.write_text(json.dumps(id_map, indent=2, sort_keys=True) + "\n")

    (output_dir / "dataset.toml").write_text(
        _toml(
            "Generated by resources_servers/scale_swe/harbor/adapter.py; README.md names the source rows.",
            {},
            [
                ("[dataset]", {"name": dataset_name, "version": "0.1.0"}),
                # As configs/scale_swe_opencode.yaml: training rows, license TBD, OpenCode only.
                (
                    "[gym]",
                    {"type": "train", "license": "TBD", "domain": "coding", "allowed_agents": ["opencode_agent"]},
                ),
            ],
        )
    )
    (output_dir / "README.md").write_text(
        _dataset_readme(
            adapter=adapter,
            rows_path=rows_path,
            ids_path=ids_path,
            native_config=native_config,
            limit=limit,
            dataset_name=dataset_name,
            notes=notes,
            count=len(task_ids),
        )
    )
    return task_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True, help="Scale-SWE rows (JSONL from prepare_scale_swe.py)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New dataset folder (empty or absent)")
    parser.add_argument("--ids-file", type=Path, help="Instance ids to convert, one per line, in this order")
    parser.add_argument("--native-rows-out", type=Path, help="Write the selected raw rows here, in the same order")
    parser.add_argument("--id-map-out", type=Path, help="Write {native line index: task id} here (JSON)")
    parser.add_argument(
        "--native-config",
        type=Path,
        help="The native run's config: a pairing YAML or `gym env resolve` dump (default: configs/scale_swe.yaml)",
    )
    parser.add_argument("--limit", type=int, default=0, help="At most this many of the selected rows (0: all)")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Harbor's org/name")
    parser.add_argument("--agent-timeout-sec", type=int, default=DEFAULT_AGENT_TIMEOUT_SEC)
    args = parser.parse_args(argv)
    try:
        task_ids = convert(
            args.rows,
            args.output_dir,
            ids_path=args.ids_file,
            native_rows_out=args.native_rows_out,
            id_map_out=args.id_map_out,
            native_config=args.native_config,
            limit=args.limit,
            dataset_name=args.dataset_name,
            agent_timeout_sec=args.agent_timeout_sec,
        )
    except (AdapterError, OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        print(f"adapter: {error}", file=sys.stderr)
        return 2
    print(f"converted {len(task_ids)} task(s) into {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
