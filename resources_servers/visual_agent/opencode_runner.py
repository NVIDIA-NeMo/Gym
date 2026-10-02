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
"""Run OpenCode as the agentic judge inside a grader sandbox."""

import json
import logging
import tarfile
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from nemo_gym.sandbox import AsyncSandbox


LOG = logging.getLogger(__name__)


class JudgeOpenCodeConfig(BaseModel):
    # Cached OpenCode binaries reachable inside the grader sandbox (e.g. on a mounted volume), installed
    # with a script taking --glibc-binary/--musl-binary (or --binary). Unset: `version` is downloaded
    # from opencode.ai in every grader sandbox.
    install_script_path: Optional[str] = None
    binary_path: Optional[str] = None
    musl_binary_path: Optional[str] = None
    version: str = "1.17.11"
    max_context_window: int = 262144
    timeout_s: float = 1800
    # Declares image input so screenshots the judge reads are sent to the model.
    enable_image_input: bool = True
    # Merged over the defaults below (permissions, tools, compaction).
    opencode_config: Dict[str, Any] = Field(default_factory=dict)


class OpenCodeRunResult(BaseModel):
    finished: bool
    return_code: Optional[int] = None
    error_type: Optional[str] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    num_assistant_messages: int = 0
    num_tool_calls: int = 0
    num_images_read: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    export_local_path: Optional[str] = None


def build_opencode_config(model_base_url: str, config: JudgeOpenCodeConfig) -> Dict[str, Any]:
    model: Dict[str, Any] = {
        "limit": {
            "context": config.max_context_window,
            "input": config.max_context_window,
            "output": config.max_context_window,
        }
    }
    if config.enable_image_input:
        model |= {"modalities": {"input": ["text", "image"], "output": ["text"]}, "attachment": True}
    base = {
        "$schema": "https://opencode.ai/config.json",
        "model": "nemo_gym_judge/judge_model",
        "provider": {
            "nemo_gym_judge": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": model_base_url,
                    "apiKey": "dummy_key",  # pragma: allowlist secret
                    "timeout": False,
                    "chunkTimeout": 600000,
                },
                "models": {"judge_model": model},
            }
        },
        # The grader sandbox is disposable and holds only copies, so the judge may run anything in it.
        "permission": {"*": "allow"},
        "tools": {"webfetch": False, "websearch": False, "task": False, "skill": False, "todowrite": False},
        "compaction": {"auto": True},
    }
    return base | config.opencode_config


def summarize_export(export: Dict[str, Any]) -> Dict[str, int]:
    """Counts from an `opencode export` session used as judge-effort metrics."""
    counts = {
        "num_assistant_messages": 0,
        "num_tool_calls": 0,
        "num_images_read": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    for message in export.get("messages") or []:
        info = message.get("info") or {}
        if info.get("role") != "assistant":
            continue
        counts["num_assistant_messages"] += 1
        tokens = info.get("tokens") or {}
        counts["input_tokens"] += int(tokens.get("input") or 0)
        counts["output_tokens"] += int(tokens.get("output") or 0)
        for part in message.get("parts") or []:
            if part.get("type") != "tool":
                continue
            counts["num_tool_calls"] += 1
            attachments = (part.get("state") or {}).get("attachments") or []
            counts["num_images_read"] += sum(1 for a in attachments if str(a.get("mime", "")).startswith("image/"))
    return counts


def opencode_install_command(config: JudgeOpenCodeConfig) -> str:
    """Shell command that installs OpenCode in the grader sandbox (same options as the OpenCode agent)."""
    if config.install_script_path and config.binary_path:
        if config.musl_binary_path:
            binaries = f"--glibc-binary {quote(config.binary_path)} --musl-binary {quote(config.musl_binary_path)}"
        else:
            binaries = f"--binary {quote(config.binary_path)}"
        return f"bash {quote(config.install_script_path)} {binaries}"
    return (
        'installer=$(mktemp) && curl -fsSL -o "$installer" https://opencode.ai/install '
        f'&& VERSION={quote(config.version)} bash "$installer"'
    )


async def run_opencode_judge(
    sandbox: AsyncSandbox,
    *,
    prompt: str,
    model_base_url: str,
    config: JudgeOpenCodeConfig,
    workdir: str,
    export_local_path: Path,
    title: str = "visual-agent-judge",
) -> OpenCodeRunResult:
    """Run one OpenCode session in `workdir` with `prompt`, and download its export."""
    data_home = f"{workdir}/.opencode-data"
    prompt_remote = f"{workdir}/.judge_prompt.md"
    with TemporaryDirectory() as tmp:
        prompt_local = Path(tmp) / "prompt.md"
        prompt_local.write_text(prompt)
        await sandbox.upload(prompt_local, prompt_remote)

    install = opencode_install_command(config)
    opencode_config = json.dumps(build_opencode_config(model_base_url, config))
    # The prompt goes in on stdin: positional `opencode run` arguments are re-quoted by OpenCode,
    # which would wrap the whole prompt in quotes and escape every inner double quote.
    command = (
        f"cd {quote(workdir)} && {install} > /dev/null && export PATH=$HOME/.opencode/bin:$PATH && "
        f"OPENCODE_CONFIG_CONTENT={quote(opencode_config)} OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000 "
        f"XDG_DATA_HOME={quote(data_home)} opencode run --title {quote(title)} < {quote(prompt_remote)} "
        '&& echo "OpenCode judge finished"'
    )
    result = None
    error_type = None
    try:
        result = await sandbox.exec(command, timeout_s=config.timeout_s)
    except Exception as exc:  # Timeouts and transport errors: still try to salvage the session export.
        error_type = type(exc).__name__
        LOG.warning("OpenCode judge exec failed: %s", error_type)

    stdout = (getattr(result, "stdout", "") or "") if result is not None else ""
    run = OpenCodeRunResult(
        finished="OpenCode judge finished" in stdout,
        return_code=getattr(result, "return_code", None),
        error_type=error_type or getattr(result, "error_type", None),
        stdout_tail=stdout[-3000:],
        stderr_tail=((getattr(result, "stderr", "") or "") if result is not None else "")[-3000:],
    )

    export_remote = f"{workdir}/.judge_export.json"
    try:
        list_result = await sandbox.exec(
            f"export PATH=$HOME/.opencode/bin:$PATH && XDG_DATA_HOME={quote(data_home)} "
            "opencode session list --format json",
            timeout_s=120,
        )
        sessions = json.loads(list_result.stdout or "[]")
        if sessions:
            session_id = sessions[0]["id"]
            export_result = await sandbox.exec(
                f"export PATH=$HOME/.opencode/bin:$PATH && XDG_DATA_HOME={quote(data_home)} "
                f"opencode export {quote(session_id)} > {quote(export_remote)}",
                timeout_s=300,
            )
            if export_result.return_code == 0:
                export_local_path.parent.mkdir(parents=True, exist_ok=True)
                await sandbox.download(export_remote, export_local_path)
                export = json.loads(export_local_path.read_text() or "{}")
                run = run.model_copy(update={**summarize_export(export), "export_local_path": str(export_local_path)})
    except Exception:
        LOG.warning("Could not export the OpenCode judge session", exc_info=True)
    return run


async def exec_checked(sandbox: AsyncSandbox, command: str, *, timeout_s: float = 600, what: str = "") -> str:
    """Run a command that must succeed; returns stdout."""
    result = await sandbox.exec(command, timeout_s=timeout_s)
    if result.return_code != 0 or result.error_type is not None:
        raise RuntimeError(
            f"{what or command[:80]} failed (rc={result.return_code}, error={result.error_type}): "
            f"{(result.stderr or '')[-1500:]} {(result.stdout or '')[-1500:]}"
        )
    return result.stdout or ""


async def upload_tree(sandbox: AsyncSandbox, files: Dict[str, Path], remote_dir: str) -> None:
    """Upload {relative path: local file} as one tarball and unpack it under `remote_dir`."""
    if not files:
        await exec_checked(sandbox, f"mkdir -p {quote(remote_dir)}", what=f"mkdir {remote_dir}")
        return
    with TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for rel, local in sorted(files.items()):
                tar.add(str(local), arcname=rel)
        remote_archive = f"/tmp/upload_{uuid4().hex}.tar.gz"
        await sandbox.upload(archive, remote_archive)
    await exec_checked(
        sandbox,
        f"mkdir -p {quote(remote_dir)} && tar xzf {quote(remote_archive)} -C {quote(remote_dir)} && rm -f {quote(remote_archive)}",
        what=f"unpack into {remote_dir}",
    )


def list_files(root: Path, prefix: str = "") -> Dict[str, Path]:
    return {f"{prefix}{p.relative_to(root)}": p for p in sorted(root.rglob("*")) if p.is_file()}
