# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in sandbox diagnostics helpers."""

from __future__ import annotations

import shlex
from typing import Any, NotRequired, TypedDict


class AperfDiagnosticConfig(TypedDict):
    """Explicit APerf diagnostic settings.

    The sandbox observability module never starts APerf automatically. Callers
    can opt in through ``SandboxSpec.extensions`` or by building this command
    and executing it with the public ``Sandbox``/``AsyncSandbox`` API.
    """

    enabled: bool
    run_name: str
    interval_s: NotRequired[int | float]
    period_s: NotRequired[int | float]
    tmp_dir: NotRequired[str | None]
    collect_only: NotRequired[list[str]]
    dont_collect: NotRequired[list[str]]
    profile: NotRequired[bool]
    extra_args: NotRequired[list[str]]
    output_dir: NotRequired[str]
    local_output_dir: NotRequired[str]
    install_url: NotRequired[str]


APERF_EXTENSION_PREFIX = "observability.aperf."
DEFAULT_APERF_DIR = "/tmp/nemo-gym-aperf"


def aperf_record_command(config: AperfDiagnosticConfig | None) -> str | None:
    """Return an ``aperf record`` command when diagnostics are explicitly enabled."""
    if not config or not config.get("enabled", False):
        return None

    args = ["aperf", "record", "-r", str(config["run_name"])]
    if config.get("interval_s") is not None:
        args.extend(["-i", _number_arg(config["interval_s"])])
    if config.get("period_s") is not None:
        args.extend(["-p", _number_arg(config["period_s"])])
    if config.get("tmp_dir"):
        args.extend(["--tmp-dir", str(config["tmp_dir"])])
    if config.get("collect_only"):
        args.extend(["--collect-only", ",".join(config["collect_only"])])
    if config.get("dont_collect"):
        args.extend(["--dont-collect", ",".join(config["dont_collect"])])
    if config.get("profile"):
        args.append("--profile")
    args.extend(str(arg) for arg in config.get("extra_args", []))
    return shlex.join(args)


def aperf_config_from_extensions(
    extensions: dict[str, str],
    *,
    metadata: dict[str, str] | None = None,
    sandbox_id: str | None = None,
    timeout_s: int | None = None,
) -> AperfDiagnosticConfig | None:
    """Build an APerf diagnostic config from provider-neutral sandbox extensions."""
    enabled = _bool_extension(extensions.get(f"{APERF_EXTENSION_PREFIX}enabled"))
    if not enabled:
        return None

    metadata = metadata or {}
    run_name = (
        extensions.get(f"{APERF_EXTENSION_PREFIX}run_name")
        or metadata.get("trajectory_id")
        or metadata.get("instance_id")
        or sandbox_id
        or "sandbox"
    )
    config: AperfDiagnosticConfig = {
        "enabled": True,
        "run_name": _safe_run_name(run_name),
        "output_dir": extensions.get(f"{APERF_EXTENSION_PREFIX}output_dir") or DEFAULT_APERF_DIR,
    }
    if extensions.get(f"{APERF_EXTENSION_PREFIX}local_output_dir"):
        config["local_output_dir"] = extensions[f"{APERF_EXTENSION_PREFIX}local_output_dir"]
    if extensions.get(f"{APERF_EXTENSION_PREFIX}install_url"):
        config["install_url"] = extensions[f"{APERF_EXTENSION_PREFIX}install_url"]
    if extensions.get(f"{APERF_EXTENSION_PREFIX}tmp_dir"):
        config["tmp_dir"] = extensions[f"{APERF_EXTENSION_PREFIX}tmp_dir"]
    else:
        config["tmp_dir"] = f"{config['output_dir'].rstrip('/')}/tmp"
    if extensions.get(f"{APERF_EXTENSION_PREFIX}interval_s"):
        config["interval_s"] = _number_value(extensions[f"{APERF_EXTENSION_PREFIX}interval_s"])
    if extensions.get(f"{APERF_EXTENSION_PREFIX}period_s"):
        config["period_s"] = _number_value(extensions[f"{APERF_EXTENSION_PREFIX}period_s"])
    elif timeout_s:
        config["period_s"] = timeout_s
    else:
        config["period_s"] = 24 * 60 * 60
    if extensions.get(f"{APERF_EXTENSION_PREFIX}collect_only"):
        config["collect_only"] = _csv_value(extensions[f"{APERF_EXTENSION_PREFIX}collect_only"])
    if extensions.get(f"{APERF_EXTENSION_PREFIX}dont_collect"):
        config["dont_collect"] = _csv_value(extensions[f"{APERF_EXTENSION_PREFIX}dont_collect"])
    if _bool_extension(extensions.get(f"{APERF_EXTENSION_PREFIX}profile")):
        config["profile"] = True
    if extensions.get(f"{APERF_EXTENSION_PREFIX}extra_args"):
        config["extra_args"] = shlex.split(extensions[f"{APERF_EXTENSION_PREFIX}extra_args"])
    return config


def aperf_start_command(config: AperfDiagnosticConfig) -> str:
    """Return a shell command that starts APerf recording in the background."""
    record_command = aperf_record_command(config)
    if record_command is None:
        raise ValueError("APerf start requires an enabled diagnostic config")

    output_dir = str(config.get("output_dir") or DEFAULT_APERF_DIR)
    install_url = config.get("install_url")
    install_block = _install_block(install_url)
    return "\n".join(
        [
            "set -euo pipefail",
            f"base_dir={shlex.quote(output_dir)}",
            'bin_dir="$base_dir/bin"',
            'mkdir -p "$bin_dir" "$base_dir/output" "$base_dir/tmp"',
            'export PATH="$bin_dir:$PATH"',
            install_block,
            'if [ -f "$base_dir/aperf.pid" ] && kill -0 "$(cat "$base_dir/aperf.pid")" 2>/dev/null; then',
            "  exit 0",
            "fi",
            'cd "$base_dir/output"',
            f"nohup {record_command} > \"$base_dir/aperf_record.log\" 2>&1 &",
            'echo "$!" > "$base_dir/aperf.pid"',
        ]
    )


def aperf_stop_command(config: AperfDiagnosticConfig) -> str:
    """Return a shell command that stops APerf and packages its artifacts."""
    output_dir = str(config.get("output_dir") or DEFAULT_APERF_DIR)
    archive_path = aperf_archive_path(config)
    return "\n".join(
        [
            "set -euo pipefail",
            f"base_dir={shlex.quote(output_dir)}",
            f"archive_path={shlex.quote(archive_path)}",
            'if [ -f "$base_dir/aperf.pid" ]; then',
            '  pid="$(cat "$base_dir/aperf.pid")"',
            '  if kill -0 "$pid" 2>/dev/null; then',
            '    kill -INT "$pid" 2>/dev/null || true',
            "    for _ in $(seq 1 20); do",
            '      kill -0 "$pid" 2>/dev/null || break',
            "      sleep 1",
            "    done",
            '    kill -TERM "$pid" 2>/dev/null || true',
            "  fi",
            "fi",
            'find "$base_dir" -maxdepth 6 -type f -print > "$base_dir/file_list.txt" || true',
            'tar -czf "$archive_path" -C "$base_dir" . || true',
        ]
    )


def aperf_archive_path(config: AperfDiagnosticConfig) -> str:
    """Remote sandbox path for the packaged APerf artifact."""
    output_dir = str(config.get("output_dir") or DEFAULT_APERF_DIR).rstrip("/")
    return f"{output_dir}.tgz"


def _number_arg(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _bool_extension(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _number_value(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def _csv_value(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_run_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)[:120]


def _install_block(install_url: str | None) -> str:
    if not install_url:
        return "command -v aperf >/dev/null 2>&1"
    quoted_url = shlex.quote(install_url)
    return "\n".join(
        [
            "if ! command -v aperf >/dev/null 2>&1; then",
            f"  python -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' {quoted_url} \"$base_dir/aperf.tgz\"",
            '  tar -xzf "$base_dir/aperf.tgz" -C "$base_dir"',
            '  aperf_bin="$(find "$base_dir" -type f -name aperf -perm -111 | head -n 1)"',
            '  test -n "$aperf_bin"',
            '  install "$aperf_bin" "$bin_dir/aperf"',
            "fi",
        ]
    )
