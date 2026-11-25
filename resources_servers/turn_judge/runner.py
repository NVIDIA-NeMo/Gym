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
"""Run one pair judgment in its own sandbox (mirrors GAR's group grader runner, resources_servers/swebench/group_grader
on sdevare's branch): start a sandbox from the task image, upload the pair workspace to /judge as one tarball, install
the S3-cached OpenCode, run `opencode run` with the instructions on stdin against the judge model's OpenAI-compatible
endpoint, read back /judge/out/pair.json, notes.md and the session export, stop the sandbox."""

import asyncio
import io
import json
import shlex
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec

FINISHED = "TURN_JUDGE_OPENCODE_FINISHED"
# Stops of sandboxes whose creation was still in flight when their run_pair was cancelled (kept referenced until done).
_BACKGROUND_STOPS: set[asyncio.Task] = set()


@dataclass
class OpenCodeSettings:
    install_script_path: str = "/mnt/s3-data/data/harness/opencode/install.sh"
    binary_path: str = "/mnt/s3-data/data/harness/opencode/linux-x64/1.17.11/opencode"
    # Set for a libc-aware cached installer (--glibc-binary / --musl-binary), as the rollout agent's
    # remote_opencode_musl_binary_path; empty = the installer takes --binary only.
    musl_binary_path: str = ""
    version: str = "1.17.11"
    max_context_window: int = 262144
    max_output_tokens: int = 32768
    temperature: float | None = 0.3
    top_p: float | None = 0.95


@dataclass
class PairRunSettings:
    judge_base_url: str
    judge_model: str
    opencode: OpenCodeSettings = field(default_factory=OpenCodeSettings)
    session_timeout_s: float = 2400.0
    sandbox_ttl_s: int = 4 * 3600
    ready_timeout_s: int = 1200
    cpu: float = 1.0
    memory_mib: int = 4096
    metadata: dict[str, str] = field(default_factory=dict)


def opencode_config(s: PairRunSettings) -> dict[str, Any]:
    oc = s.opencode
    agent_opts = {k: v for k, v in (("temperature", oc.temperature), ("top_p", oc.top_p)) if v is not None}
    cfg: dict[str, Any] = {
        "model": f"judge/{s.judge_model}",
        "provider": {
            "judge": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": s.judge_base_url,
                    "apiKey": "dummy_key",
                    "timeout": False,
                    "chunkTimeout": 600000,
                },
                "models": {
                    s.judge_model: {
                        "limit": {
                            "context": oc.max_context_window,
                            "input": oc.max_context_window,
                            "output": oc.max_output_tokens,
                        }
                    }
                },
            }
        },
        "permission": {"*": "allow", "webfetch": "deny", "websearch": "deny"},
        "tools": {"webfetch": False, "websearch": False, "task": False, "skill": False, "todowrite": False},
        "compaction": {"auto": True},
    }
    if agent_opts:
        cfg["agent"] = {"build": agent_opts}
    return cfg


def tarball(files: Mapping[str, bytes | str]) -> bytes:
    """In-memory tar.gz of {relative path: content} (no local files: keeps metadata load off shared storage)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, content in files.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


async def _exec(sb: AsyncSandbox, cmd: str, timeout_s: float) -> tuple[int | None, str, str]:
    r = await sb.exec(cmd, timeout_s=timeout_s)
    return getattr(r, "return_code", None), getattr(r, "stdout", "") or "", getattr(r, "stderr", "") or ""


async def _read(sb: AsyncSandbox, path: str, limit: int = 4_000_000) -> str | None:
    rc, out, _ = await _exec(sb, f"test -f {shlex.quote(path)} && head -c {limit} {shlex.quote(path)}", 120)
    return out if rc == 0 else None


async def _stop_when_created(sb: AsyncSandbox, start: asyncio.Future) -> None:
    try:
        await start
    except BaseException:  # creation failed or was cancelled: nothing to stop (the provider cleans up failed creates)
        return
    try:
        await sb.stop()
    except Exception:  # the sandbox TTL / the run-id reap are the backstop
        pass


def background_stops() -> set[asyncio.Task]:
    """Pending stops of sandboxes created after their session was cancelled (await them before reaping at exit)."""
    return set(_BACKGROUND_STOPS)


async def run_pair(
    provider: Mapping[str, Any],
    image: str,
    files: Mapping[str, bytes | str],
    instructions: str,
    s: PairRunSettings,
    required: str = "out/pair.json",
    extra_reads: tuple[str, ...] = ("out/decisions.json",),
) -> dict[str, Any]:
    """One judge session (a pair, or a group stage); never raises. Returns status (ok when ``required`` exists),
    pair_json (= the required file's text), notes, outputs {extra path: text}, export, timings, errors."""
    res: dict[str, Any] = {
        "status": "error",
        "image": image,
        "pair_json": None,
        "notes": None,
        "export": None,
        "timings": {},
        "error": "",
    }
    t0 = time.time()
    sb = AsyncSandbox(dict(provider))
    spec = SandboxSpec(
        image=image,
        ttl_s=s.sandbox_ttl_s,
        ready_timeout_s=s.ready_timeout_s,
        workdir="/",
        env={"OPENCODE_VERSION": s.opencode.version},
        metadata=dict(s.metadata),
        resources={"cpu": s.cpu, "memory_mib": s.memory_mib},
    )
    started = False
    try:
        # A cancel (deadline, DELETE, shutdown) that lands while the sandbox is being created would lose its handle
        # and leak the sandbox the server is still creating until its TTL: shield the create, and on a cancel stop the
        # sandbox in the background once the create returns.
        start = asyncio.ensure_future(sb.start(spec))
        try:
            await asyncio.shield(start)
        except asyncio.CancelledError:
            if start.done():
                started = not start.cancelled() and start.exception() is None
            else:
                task = asyncio.ensure_future(_stop_when_created(sb, start))
                _BACKGROUND_STOPS.add(task)
                task.add_done_callback(_BACKGROUND_STOPS.discard)
            raise
        started = True
        res["timings"]["sandbox_start_s"] = round(time.time() - t0, 1)
        payload = dict(files)
        payload["INSTRUCTIONS.md"] = instructions
        data = await asyncio.to_thread(tarball, payload)  # gzip of a whole group workspace: keep it off the event loop
        with tempfile.TemporaryDirectory(prefix="turn-judge-") as tmp:
            local = Path(tmp) / "judge.tar.gz"
            local.write_bytes(data)
            await sb.upload(local, "/tmp/judge.tar.gz")
        rc, out, err = await _exec(
            sb, "mkdir -p /judge/out && tar xzf /tmp/judge.tar.gz -C /judge && rm -f /tmp/judge.tar.gz && echo ok", 300
        )
        if rc != 0:
            raise RuntimeError(f"upload/extract failed rc={rc}: {err[-500:]}")
        oc = s.opencode
        if oc.musl_binary_path:
            install = (
                f"bash {shlex.quote(oc.install_script_path)} --glibc-binary {shlex.quote(oc.binary_path)} "
                f"--musl-binary {shlex.quote(oc.musl_binary_path)}"
            )
        else:
            install = f"bash {shlex.quote(oc.install_script_path)} --binary {shlex.quote(oc.binary_path)}"
        cfg = shlex.quote(json.dumps(opencode_config(s)))
        cmd = (
            f"cd /judge && {install} > /judge/.install.log 2>&1 && export PATH=$HOME/.opencode/bin:$PATH && "
            f"OPENCODE_CONFIG_CONTENT={cfg} OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000 "
            f"XDG_DATA_HOME=/judge/.opencode-data opencode run --title turn-judge < /judge/INSTRUCTIONS.md "
            f"> /judge/.opencode.out 2> /judge/.opencode.err; echo {FINISHED} rc=$?"
        )
        t1 = time.time()
        try:
            rc, out, err = await _exec(sb, cmd, s.session_timeout_s)
            res["opencode_finished"] = FINISHED in out
            res["opencode_tail"] = out[-300:]
        except Exception as e:  # timeout / transport: still try to salvage the outputs
            res["opencode_finished"] = False
            res["error"] = f"session: {type(e).__name__}: {e}"[:1000]
        res["timings"]["session_s"] = round(time.time() - t1, 1)
        res["pair_json"] = await _read(sb, f"/judge/{required}")
        res["outputs"] = {p: await _read(sb, f"/judge/{p}") for p in extra_reads}
        res["self_check"] = (
            (await _exec(sb, "cd /judge && python3 bin/check_pair.py 2>&1 | head -40", 120))[1]
            if required == "out/pair.json"
            else None
        )
        res["notes"] = await _read(sb, "/judge/out/notes.md")
        res["opencode_stderr_tail"] = (await _read(sb, "/judge/.opencode.err", 20000) or "")[-3000:]
        rc, out, _ = await _exec(
            sb,
            "cd /judge && export PATH=$HOME/.opencode/bin:$PATH && XDG_DATA_HOME=/judge/"
            ".opencode-data opencode session list --format json 2>/dev/null",
            120,
        )
        try:
            sid = json.loads(out)[0]["id"]
            rc, out, _ = await _exec(
                sb,
                f"cd /judge && export PATH=$HOME/.opencode/bin:$PATH && XDG_DATA_HOME="
                f"/judge/.opencode-data opencode export {shlex.quote(sid)} 2>/dev/null",
                300,
            )
            res["export"] = out
        except Exception as e:
            res["export_error"] = f"{type(e).__name__}: {e}"[:300]
        res["status"] = "ok" if res["pair_json"] else ("no_output" if res.get("opencode_finished") else "timeout")
    except Exception as e:
        res["error"] = (res["error"] + f" | {type(e).__name__}: {e}")[:2000]
    finally:
        if started:
            try:
                await sb.stop()
            except Exception as e:  # the sandbox TTL is the backstop
                res["stop_error"] = f"{type(e).__name__}: {e}"[:300]
        res["timings"]["total_s"] = round(time.time() - t0, 1)
    return res
