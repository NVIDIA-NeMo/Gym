# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert SWE-rebench-V2 rows (``resources_servers/swe_rebench``) into Harbor task folders, one folder per row.

    python resources_servers/swe_rebench/harbor/adapter.py --rows rows.jsonl --output-dir /path/to/dataset \\
        [--ids-file ids.txt] [--native-rows-out native.jsonl] [--id-map-out id_map.json] \\
        [--native-config native_resolved.yaml] [--image-digests digests.json] [--parser-repo DIR]

A converted task runs the episode the SWE-rebench Resources Server runs for ``opencode_agent`` under
``single_agent_turn`` (``configs/swe_rebench.yaml``, episode-session path of ``app.py``). Based on the Scale-SWE
adapter (``resources_servers/scale_swe/harbor``), with these swe_rebench specifics:

- **Sandbox**: the row's ``image_name`` (pinned by digest with ``--image-digests``). ``environment/Dockerfile`` is
  overlay mode: the first ``RUN`` unpacks a bundle into ``/var/lib/nemo_gym``, the second runs ``setup.sh``, which
  finds the image's Python 3.10+ or installs a pinned python-build-standalone release (Dart images have no Python),
  then runs ``seed.py``: the server's own ``apply_anti_cheat_setup`` (``required`` as configured), git identity, and
  the pristine untracked listing. Resources and ``GRADLE_USER_HOME`` come from the server config.
- **Prompt**: ``instruction.md`` is the row's single user message, byte for byte.
- **Patch**: a ``[[verifier.collect]]`` hook runs ``capture.py``, the server's ``capture_model_patch`` against the
  row's ``base_commit`` (passed on the hook's command line, never written into the agent's sandbox).
- **Grading**: separate sandbox of the plain row image; ``tests/nemo_gym/grade.py`` runs the server's vendored
  ``verification.py`` and ``log_parsers.py``, with the upstream SWE-rebench-V2 parser tree at the server's pinned
  commit vendored under ``tests/nemo_gym/parser_cache/`` (MIT), and the Maven/Gradle mirror files the server ships.
- **Golden patch**: ``solution/solve.sh`` applies the row's ``patch`` with the eval script's ``git apply`` flags.

Hidden data (``patch``, ``test_patch``, ``FAIL_TO_PASS``, ``PASS_TO_PASS``, ``install_config``) lives only in
``solution/`` and ``tests/``. Exit codes: 0 done, 2 a native setting or input the dataset cannot reproduce.
Output is deterministic: two conversions of the same inputs are byte-identical.
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
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TEMPLATE = HERE / "task-template"
SERVER_CONFIG = REPO_ROOT / "resources_servers/swe_rebench/configs/swe_rebench_legacy_agent.yaml"
SERVER_INSTANCE = "swe_rebench_resources_server"
# Server modules a converted task runs unchanged, by their path in this repository.
VERIFICATION_MODULE = "resources_servers/swe_rebench/verification.py"
LOG_PARSERS_MODULE = "resources_servers/swe_rebench/log_parsers.py"
PATCH_CAPTURE_MODULE = "resources_servers/swebench/patch_capture.py"
ANTI_CHEAT_MODULE = "resources_servers/swebench/anti_cheat.py"
ANTI_CHEAT_SCRIPT = "resources_servers/swebench/anti_cheat_setup.sh"
MIRROR_FILES = (
    "responses_api_agents/swe_agents/maven_mirror/settings.xml",
    "responses_api_agents/swe_agents/maven_mirror/init.gradle",
)
# The upstream parser tree files log_parsers.load_parsers imports (lib.agent.log_parsers needs swe_constants).
UPSTREAM_FILES = (
    "LICENSE",
    "lib/__init__.py",
    "lib/agent/__init__.py",
    "lib/agent/log_parsers.py",
    "lib/agent/swe_constants.py",
)

STATE_DIR = "/var/lib/nemo_gym"
ARTIFACTS_DIR = "/logs/artifacts/nemo_gym"
# app.JVM_MIRROR_ENV: set on every sandbox the server creates while sandbox_config.use_maven_mirror is on (default).
JVM_MIRROR_ENV = {"GRADLE_USER_HOME": "/root/.gradle"}
# The standalone Python download in python.sh: its curl flags (or wget's overall timeout) keep the worst case under it.
PYTHON_DOWNLOAD_BUDGET_SEC = 600
# Seed budget: the standalone Python download and three anti-cheat attempts of 600 s with their 2 s + 4 s delays,
# three git-identity attempts of 60 s, the 180 s untracked listing, and room to unpack.
BUILD_TIMEOUT_SEC = PYTHON_DOWNLOAD_BUDGET_SEC + 3 * 600 + 6 + 3 * 60 + 180 + 114
# capture_model_patch runs at most four git commands, each limited to 300 s.
CAPTURE_TIMEOUT_SEC = 4 * 300 + 60
# Room for test.sh around the server's evaluation_timeout: the standalone Python download and grade.py (300 s).
VERIFIER_TIMEOUT_MARGIN_SEC = PYTHON_DOWNLOAD_BUDGET_SEC + 300
# opencode_agent's own `timeout` (10800 s); single_agent_turn gives no per-row timeout.
DEFAULT_AGENT_TIMEOUT_SEC = 10800
_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_TASK_NAME = re.compile(r"[A-Za-z0-9._-]+")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
# image_name goes into the Dockerfile's FROM line: no whitespace or control character may end that line early.
_IMAGE_NAME = re.compile(r"[^\s\x00-\x1f\x7f]+")
# The server's sandbox_config.resources keys a task folder has a field for.
RESOURCE_KEYS = ("cpu", "memory_mib", "disk_gib")


class AdapterError(ValueError):
    """A native setting or input the converted dataset cannot reproduce (exit 2)."""


def _load_repo_module(relative: str) -> ModuleType:
    """Import a module of this checkout by path, so the adapter reads exactly the code it vendors."""
    name = "swe_rebench_harbor_" + relative.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(*args: str, cwd: Path = REPO_ROOT) -> str | None:
    """Output of a git command, or None outside a git checkout (an installed wheel, a tarball)."""
    try:
        result = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
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
    """What the converted tasks take from the ``swe_rebench_resources_server`` config."""

    evaluation_timeout: int
    resources: dict[str, Any]
    env: dict[str, str]
    apply_anti_cheating: bool
    anti_cheat_required: bool
    patch_capture_mode: str
    overwrite_maven_settings: bool

    @classmethod
    def read(cls, path: Path | None = None) -> ServerSettings:
        """Read the server block from the shipped config, or from ``--native-config``: a ``gym env resolve`` dump or a
        pairing YAML such as ``configs/swe_rebench.yaml`` (its ``config_paths`` merged in)."""
        try:
            server = _load_config(path or SERVER_CONFIG)[SERVER_INSTANCE]["resources_servers"]["swe_rebench"]
        except (KeyError, TypeError) as error:
            raise AdapterError(f"no {SERVER_INSTANCE}.resources_servers.swe_rebench block in {path}") from error
        sandbox_config = server.get("sandbox_config") or {}
        resources = sandbox_config.get("resources") or {}
        cpu = resources.get("cpu")
        refusals = {
            "is_verifying_golden_patch": server.get("is_verifying_golden_patch", False) is True,
            "enforce_agent_no_network": server.get("enforce_agent_no_network", False) is True,
            "sandbox_config.provider_options.network_policy": "network_policy"
            in (sandbox_config.get("provider_options") or {}),
            "sandbox_config.derive_cpu_env: false": sandbox_config.get("derive_cpu_env", True) is False,
            "sandbox_config.legacy_container_setup": bool(sandbox_config.get("legacy_container_setup", False)),
            "sandbox_config.use_maven_mirror: false": sandbox_config.get("use_maven_mirror", True) is False,
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
            # The harbor server always adds the CPU-count hints; the agent-sandbox egress block, a provider network
            # policy, the swe_agents container setup, a mirror-less JVM environment and resources a task has no
            # whole-number field for have no task-folder equivalent here.
            raise AdapterError(f"native settings the task folders cannot reproduce: {', '.join(refused)}")
        mode = str(server.get("patch_capture_mode", "worktree"))
        if mode not in ("worktree", "committed"):
            raise AdapterError(f"unknown patch_capture_mode {mode!r}")
        timeout = server.get("evaluation_timeout", 3600)
        if timeout is None:
            raise AdapterError("evaluation_timeout: null (no limit) has no task-folder equivalent")
        env = {str(key): str(value) for key, value in (sandbox_config.get("env") or {}).items()}
        return cls(
            evaluation_timeout=int(timeout),
            resources=dict(sandbox_config.get("resources") or {}),
            env=JVM_MIRROR_ENV | env,
            apply_anti_cheating=bool(server.get("apply_anti_cheating", True)),
            anti_cheat_required=bool(server.get("anti_cheat_required", False)),
            patch_capture_mode=mode,
            overwrite_maven_settings=bool(server.get("overwrite_maven_settings", True)),
        )


class SWERebenchHarborAdapter:
    """Writes one task folder per SWE-rebench-V2 row."""

    def __init__(
        self,
        *,
        settings: ServerSettings,
        parser_repo: Path,
        image_digests: dict[str, str] | None = None,
        agent_timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC,
    ) -> None:
        self.settings = settings
        self.agent_timeout_sec = agent_timeout_sec
        self.image_digests = dict(image_digests or {})
        self.gym_commit = _git("rev-parse", "HEAD") or "unknown"
        log_parsers = _load_repo_module(LOG_PARSERS_MODULE)
        self.upstream_commit = log_parsers.UPSTREAM_COMMIT
        self.parser_dir_name = log_parsers._repo_dir_name()
        head = _git("rev-parse", "HEAD", cwd=parser_repo)
        if head != self.upstream_commit:
            raise AdapterError(
                f"--parser-repo {parser_repo} is at {head!r}, not the server's pinned {self.upstream_commit}"
            )
        self.upstream = {relative: (parser_repo / relative).read_bytes() for relative in UPSTREAM_FILES}
        self.vendored = {
            relative: self._vendored_text(relative)
            for relative in (VERIFICATION_MODULE, LOG_PARSERS_MODULE, PATCH_CAPTURE_MODULE, ANTI_CHEAT_MODULE)
        }
        # Data files the server reads as they are (written into sandboxes byte for byte): copied without a header.
        self.data_files = {
            relative: (REPO_ROOT / relative).read_text() for relative in (ANTI_CHEAT_SCRIPT, *MIRROR_FILES)
        }

    def _vendored_text(self, relative: str) -> str:
        original = (REPO_ROOT / relative).read_text()
        header = (
            f"# Vendored unchanged from {relative} at Gym commit {self.gym_commit}; source sha256"
            f" {_sha256(original.encode())}; license Apache-2.0\n"
        )
        return header + original

    def _upstream_text(self, relative: str) -> bytes:
        data = self.upstream[relative]
        if not relative.endswith(".py"):
            return data
        header = (
            f"# Vendored unchanged from https://github.com/SWE-rebench/SWE-rebench-V2 {relative} at"
            f" {self.upstream_commit}; source sha256 {_sha256(data)}; license MIT\n"
        )
        return header.encode() + data

    def vendored_digests(self) -> dict[str, str]:
        digests = {relative: _sha256((REPO_ROOT / relative).read_bytes()) for relative in self.vendored}
        digests |= {relative: _sha256(text.encode()) for relative, text in self.data_files.items()}
        digests |= {f"SWE-rebench-V2@{self.upstream_commit}:{r}": _sha256(d) for r, d in self.upstream.items()}
        return dict(sorted(digests.items()))

    def image(self, row: dict[str, Any]) -> str:
        name = row["image_name"]
        digest = self.image_digests.get(name)
        if digest is None:
            return name
        if not _DIGEST.fullmatch(digest):
            raise AdapterError(f"{name}: digest {digest!r} is not sha256:<64 hex>")
        return f"{name}@{digest}"

    # -- seed -----------------------------------------------------------------------------------------------------

    def bundle(self, row: dict[str, Any]) -> bytes:
        """A gzipped tar of what ``/var/lib/nemo_gym`` holds in the agent's sandbox, byte for byte reproducible.

        It holds no hidden data: the row's base commit reaches the capture hook on its command line only.
        """
        config = {
            "instance_id": row["instance_id"],
            "workdir": self._workdir(row),
            "patch_capture_mode": self.settings.patch_capture_mode,
            "apply_anti_cheating": self.settings.apply_anti_cheating,
            "anti_cheat_required": self.settings.anti_cheat_required,
        }
        files = {
            "setup.sh": (TEMPLATE / "environment/setup.sh").read_text(),
            "seed.py": (TEMPLATE / "environment/seed.py").read_text(),
            "capture.py": (TEMPLATE / "environment/capture.py").read_text(),
            "python.sh": (TEMPLATE / "tests/nemo_gym/python.sh").read_text(),
            "local_sandbox.py": (TEMPLATE / "tests/nemo_gym/local_sandbox.py").read_text(),
            "config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
            ANTI_CHEAT_SCRIPT: self.data_files[ANTI_CHEAT_SCRIPT],
            ANTI_CHEAT_MODULE: self.vendored[ANTI_CHEAT_MODULE],
            PATCH_CAPTURE_MODULE: self.vendored[PATCH_CAPTURE_MODULE],
            VERIFICATION_MODULE: self.vendored[VERIFICATION_MODULE],
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
            f"# Generated by resources_servers/swe_rebench/harbor/adapter.py for {row['instance_id']}.\n"
            "# Overlay mode: Gym starts the image and runs these RUN lines in the sandbox before the agent;"
            " nothing is\n"
            "# built. The first unpacks the seed bundle (setup.sh, seed.py, the capture hook and the server modules\n"
            "# they call), the second runs the seed.\n"
            f"FROM {self.image(row)}\n"
            f"RUN mkdir -p {STATE_DIR} && printf '%s' '{encoded}' | base64 -d | tar -xzf - -C {STATE_DIR}\n"
            f"RUN bash {STATE_DIR}/setup.sh\n"
        )

    # -- task folder ----------------------------------------------------------------------------------------------

    @staticmethod
    def _workdir(row: dict[str, Any]) -> str:
        # verification.repo_directory: the image checks the repository out at /<repo name>.
        repo = row["repo"]
        return "/" + (repo.split("/", 1)[1] if "/" in repo else repo)

    def task_toml(self, row: dict[str, Any]) -> str:
        sandbox: dict[str, Any] = {"workdir": self._workdir(row), **_resource_keys(self.settings.resources)}
        sandbox["env"] = dict(sorted(self.settings.env.items()))
        capture_command = (
            f'rm -rf {ARTIFACTS_DIR}; "$(cat {STATE_DIR}/python_bin)" -I {STATE_DIR}/capture.py'
            f" {shlex.quote(row['base_commit'])};"
            f" test -f {ARTIFACTS_DIR}/capture.json || {{ mkdir -p {ARTIFACTS_DIR} && echo"
            f' \'{{"status": "runner_failed", "error": "the capture hook could not run capture.py"}}\''
            f" > {ARTIFACTS_DIR}/capture.json; }}"
        )
        return _toml(
            "Generated by resources_servers/swe_rebench/harbor/adapter.py from SWE-rebench-V2 row"
            f" {row['instance_id']}.",
            {"schema_version": "1.4"},
            [
                (
                    "[metadata]",
                    {
                        "source_server": "swe_rebench",
                        "instance_id": row["instance_id"],
                        "repo": row.get("repo") or "",
                        "language": row.get("language") or "",
                        "image": self.image(row),
                    },
                ),
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
                ("[verifier.environment]", {"docker_image": self.image(row), **sandbox}),
                (
                    "[[verifier.collect]]",
                    {"service": "main", "timeout_sec": CAPTURE_TIMEOUT_SEC, "command": capture_command},
                ),
            ],
        )

    def task_files(self, row: dict[str, Any]) -> dict[str, str | bytes]:
        upstream_root = f"tests/nemo_gym/parser_cache/{self.parser_dir_name}"
        files: dict[str, str | bytes] = {
            "task.toml": self.task_toml(row),
            "instruction.md": _prompt(row),
            "environment/Dockerfile": self.dockerfile(row, self.bundle(row)),
            "tests/test.sh": (TEMPLATE / "tests/test.sh").read_text(),
            "tests/nemo_gym/grade.py": (TEMPLATE / "tests/nemo_gym/grade.py").read_text(),
            "tests/nemo_gym/local_sandbox.py": (TEMPLATE / "tests/nemo_gym/local_sandbox.py").read_text(),
            "tests/nemo_gym/python.sh": (TEMPLATE / "tests/nemo_gym/python.sh").read_text(),
            f"tests/nemo_gym/{VERIFICATION_MODULE}": self.vendored[VERIFICATION_MODULE],
            f"tests/nemo_gym/{LOG_PARSERS_MODULE}": self.vendored[LOG_PARSERS_MODULE],
            **{f"tests/nemo_gym/{relative}": self.data_files[relative] for relative in MIRROR_FILES},
            **{f"{upstream_root}/{relative}": self._upstream_text(relative) for relative in UPSTREAM_FILES},
            "tests/nemo_gym/task.json": json.dumps(
                {
                    "source": "swe_rebench",
                    "evaluation_timeout": self.settings.evaluation_timeout,
                    "overwrite_maven_settings": self.settings.overwrite_maven_settings,
                    "row": {
                        key: row[key]
                        for key in (
                            "instance_id",
                            "repo",
                            "base_commit",
                            "language",
                            "test_patch",
                            "install_config",
                            "FAIL_TO_PASS",
                            "PASS_TO_PASS",
                        )
                        if key in row
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        }
        if (row.get("patch") or "").strip():
            files["solution/solve.sh"] = _solve_script(self._workdir(row))
            files["solution/golden.patch"] = row["patch"]
        return files

    def write_task(self, row: dict[str, Any], task_dir: Path) -> None:
        for relative, content in self.task_files(row).items():
            path = task_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                # newline="" keeps a CRLF in the row's text as it is; Gym's reader turns it into LF.
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
    if not isinstance(params, dict):
        raise AdapterError(f"{row.get('instance_id')}: expected responses_create_params.input to hold one message")
    # A request control the row sets (a sampling value such as temperature 0 or top_p 0.4, tools, instructions, a
    # token limit) has no task-folder form: the agent's run configuration sets the request. Checked by presence, not
    # truthiness, so a zero or false value is refused too; null, absent and an empty tool list convert.
    for key, value in sorted(params.items()):
        if key != "input" and value is not None and not (key == "tools" and value == []):
            raise AdapterError(f"{row.get('instance_id')}: responses_create_params.{key} has no task-folder form")
    messages = params.get("input")
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
        # json.dumps writes a character outside the Basic Multilingual Plane as a surrogate pair, which TOML
        # rejects: write that pair as one \UXXXXXXXX escape. An escaped backslash is matched first, as a unit.
        return re.sub(
            r"\\\\|\\ud[89ab][0-9a-f]{2}\\ud[c-f][0-9a-f]{2}",
            lambda match: match[0] if match[0] == "\\\\" else "\\U{:08x}".format(ord(json.loads(f'"{match[0]}"'))),
            json.dumps(value).replace("\x7f", "\\u007f"),
        )
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
    """The ``subset`` output: one id per line, ``#`` comments, order kept; a duplicate is an error."""
    stripped = [line.strip() for line in path.read_text().splitlines()]
    ids = [line for line in stripped if line and not line.startswith("#")]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise AdapterError(f"{path}: duplicate id(s) {duplicates[:5]}")
    return ids


def select_rows(path: Path, ids: list[str] | None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream ``(raw line, row)``; with ``ids``, only those, in ``ids`` order (only the selected lines are kept)."""
    wanted = set(ids) if ids is not None else None
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if wanted is None:
                yield line, json.loads(line)
                continue
            row = json.loads(line)
            if row["instance_id"] in wanted:
                if row["instance_id"] in found:
                    raise AdapterError(f"{row['instance_id']} appears twice in {path}")
                found[row["instance_id"]] = line
    if ids is None:
        return
    missing = [item for item in ids if item not in found]
    if missing:
        raise AdapterError(f"{len(missing)} requested id(s) not in {path.name}: {missing[:5]}")
    for item in ids:
        yield found[item], json.loads(found[item])


def convert(
    rows_path: Path,
    output_dir: Path,
    *,
    parser_repo: Path,
    ids_path: Path | None = None,
    native_rows_out: Path | None = None,
    id_map_out: Path | None = None,
    native_config: Path | None = None,
    image_digests_path: Path | None = None,
    agent_timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC,
) -> list[str]:
    """Write one task folder per selected row, plus ``dataset.toml`` and ``README.md``. Returns the task ids."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AdapterError(f"{output_dir} is not empty")
    settings = ServerSettings.read(native_config)
    digests = json.loads(image_digests_path.read_text()) if image_digests_path else None
    adapter = SWERebenchHarborAdapter(
        settings=settings, parser_repo=parser_repo, image_digests=digests, agent_timeout_sec=agent_timeout_sec
    )
    ids = read_ids(ids_path) if ids_path else None
    selected = list(select_rows(rows_path, ids))
    selected_ids = [row["instance_id"] for _, row in selected]
    repeated = sorted({name for name in selected_ids if selected_ids.count(name) > 1})
    if repeated:
        raise AdapterError(f"instance_id(s) {repeated[:5]} appear more than once in {rows_path.name}")
    for _, row in selected:
        name = row["instance_id"]
        # A name starting with "." could be "." or "..": the output folder itself or its parent.
        if not _TASK_NAME.fullmatch(name) or name.startswith("."):
            raise AdapterError(f"instance_id {name!r} is not a usable folder name")
        if not isinstance(row.get("image_name"), str) or not _IMAGE_NAME.fullmatch(row["image_name"]):
            raise AdapterError(
                f"{name}: image_name {row.get('image_name')!r} is empty or holds whitespace or a control character"
            )
        if digests is not None and row["image_name"] not in digests:
            raise AdapterError(f"{name}: no digest for {row['image_name']} in {image_digests_path.name}")
        adapter.image(row)  # refuses a digest that is not sha256:<64 hex>
        _prompt(row)
        try:
            # The task files are written as UTF-8: a lone surrogate (valid JSON) would fail halfway through.
            json.dumps(row, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError as error:
            raise AdapterError(f"{name}: the row holds text that is not valid UTF-8 ({error.reason})") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = []
    for _, row in selected:
        adapter.write_task(row, output_dir / row["instance_id"])
        task_ids.append(row["instance_id"])
    if native_rows_out is not None:
        native_rows_out.write_text("".join(line if line.endswith("\n") else line + "\n" for line, _ in selected))
    if id_map_out is not None:
        id_map = {str(index): task_id for index, task_id in enumerate(task_ids)}
        id_map_out.write_text(json.dumps(id_map, indent=2, sort_keys=True) + "\n")

    (output_dir / "dataset.toml").write_text(
        _toml(
            "Generated by resources_servers/swe_rebench/harbor/adapter.py; README.md names the source rows.",
            {},
            [
                ("[dataset]", {"name": "nemo-gym/swe-rebench-v2", "version": "0.1.0"}),
                # As configs/swe_rebench_opencode.yaml: training rows, nebius/SWE-rebench-V2's cc-by-4.0, OpenCode.
                (
                    "[gym]",
                    {
                        "type": "train",
                        "license": "Creative Commons Attribution 4.0 International",
                        "domain": "coding",
                        "allowed_agents": ["opencode_agent"],
                    },
                ),
            ],
        )
    )
    rows_bytes = rows_path.read_bytes()
    rows_lines = rows_bytes.count(b"\n")
    regen = [
        "python resources_servers/swe_rebench/harbor/adapter.py",
        f"--rows {rows_path.name}",
        "--output-dir <dir>",
    ]
    if ids_path:
        regen.append(f"--ids-file {ids_path.name}")
    if native_config:
        regen.append(f"--native-config {native_config.name}")
    if image_digests_path:
        regen.append(f"--image-digests {image_digests_path.name}")
    regen.append("--parser-repo <SWE-rebench-V2 checkout at the pinned commit>")
    readme = [
        "# SWE-rebench-V2 as Harbor task folders",
        "",
        "Generated by `resources_servers/swe_rebench/harbor/adapter.py`; see that folder's README.",
        "",
        f"- Rows: `{rows_path.name}`, sha256 `{_sha256(rows_bytes)}`, {rows_lines} lines",
        f"- Selected: {len(task_ids)} task(s)"
        + (f" from `{ids_path.name}` (sha256 `{_sha256(ids_path.read_bytes())}`)" if ids_path else ""),
        f"- Gym commit: `{adapter.gym_commit}`",
        f"- Server settings: `{json.dumps(asdict(settings), sort_keys=True)}`",
        f"- Images pinned by digest: {'yes' if digests is not None else 'no (tags as the rows name them)'}",
        '- Separate verification needs no server option: every task sets `environment_mode = "separate"`.',
        "",
        "Vendored files (sha256 of the source):",
        "",
        *[f"- `{name}`: `{digest}`" for name, digest in adapter.vendored_digests().items()],
        "",
        "Regenerate:",
        "",
        "```bash",
        " \\\n    ".join(regen),
        "```",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme))
    return task_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True, help="SWE-rebench-V2 rows (prepare_swe_rebench.py JSONL)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New dataset folder (empty or absent)")
    parser.add_argument("--ids-file", type=Path, help="Instance ids to convert, in this order (subset output)")
    parser.add_argument("--native-rows-out", type=Path, help="Write the selected raw rows here, in --ids-file order")
    parser.add_argument("--id-map-out", type=Path, help="Write {native line index: task id} here")
    parser.add_argument(
        "--native-config",
        type=Path,
        help="A `gym env resolve` dump of the native run config, or a pairing YAML (its config_paths are merged)",
    )
    parser.add_argument("--image-digests", type=Path, help="JSON {image_name: sha256:<hex>} to pin each FROM")
    parser.add_argument(
        "--parser-repo",
        type=Path,
        required=True,
        help="A SWE-rebench-V2 checkout at log_parsers.UPSTREAM_COMMIT (the server's cache clone works)",
    )
    parser.add_argument("--agent-timeout-sec", type=int, default=DEFAULT_AGENT_TIMEOUT_SEC)
    args = parser.parse_args(argv)
    try:
        task_ids = convert(
            args.rows,
            args.output_dir,
            parser_repo=args.parser_repo,
            ids_path=args.ids_file,
            native_rows_out=args.native_rows_out,
            id_map_out=args.id_map_out,
            native_config=args.native_config,
            image_digests_path=args.image_digests,
            agent_timeout_sec=args.agent_timeout_sec,
        )
    except AdapterError as error:
        print(f"adapter: {error}", file=sys.stderr)
        return 2
    print(f"converted {len(task_ids)} task(s) into {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
