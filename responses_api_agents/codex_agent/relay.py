# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional Relay configuration for Codex inside a Gym task sandbox."""

import json
import shlex
from pathlib import PurePosixPath
from textwrap import dedent
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class CodexRelayConfig(BaseModel):
    """Record the sandboxed Codex run through a pinned Relay CLI.

    Gym installs the CLI separately from the task's Python environment unless
    ``executable`` names a preinstalled binary of exactly ``version``.
    ``output_dir`` must be included in the task owner's artifact collection.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    version: Literal["0.11.0"] = "0.11.0"
    executable: str | None = None
    output_dir: str = "/logs/artifacts/relay"

    @field_validator("output_dir")
    @classmethod
    def absolute_output_directory(cls, value: str) -> str:
        """Keep recording destinations explicit and independent of the task cwd."""
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or not path.name or "\x00" in value:
            raise ValueError("relay.output_dir must be an absolute sandbox directory other than /")
        return str(path)

    @field_validator("executable")
    @classmethod
    def nonempty_executable(cls, value: str | None) -> str | None:
        """An override is one executable, not a shell command."""
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("relay.executable must name an executable")
        return value


def relay_files(settings: CodexRelayConfig, *, session_dir: str, codex_command: list[str]) -> dict[str, str]:
    """Render session-private launcher/config files without changing Codex's HOME.

    Relay needs one executable for version probing; the installed Codex runtime
    has a Node executable and a JS entrypoint. Its private shell launcher keeps
    those two arguments intact and forwards the CLI arguments verbatim.
    """
    directory = f"{session_dir}/relay"
    output_directory = str(PurePosixPath(settings.output_dir) / PurePosixPath(session_dir).name)
    # These fixed schemas need only quoted strings; JSON escaping is valid for
    # TOML basic strings and keeps Relay independent of optional Python writers.
    return {
        "relay/codex": "#!/bin/sh\nexec " + shlex.join(codex_command) + ' "$@"\n',
        "relay/config.toml": dedent(f"""\
            version = 1
            [agents.codex]
            command = {json.dumps(f"{directory}/codex", ensure_ascii=False)}
            """),
        "relay/plugins.toml": dedent(f"""\
            version = 1
            [[components]]
            kind = "observability"
            enabled = true
            [components.config]
            version = 4
            enable_full_payloads = true
            [components.config.atof]
            enabled = true
            [[components.config.atof.sinks]]
            type = "file"
            output_directory = {json.dumps(output_directory, ensure_ascii=False)}
            filename = "events.jsonl"
            mode = "append"
            [components.config.atif]
            enabled = true
            output_directory = {json.dumps(output_directory, ensure_ascii=False)}
            filename_template = "trajectory-{{session_id}}.json"
            """),
    }
