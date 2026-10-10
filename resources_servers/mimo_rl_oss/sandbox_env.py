# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import SandboxSpec, create_provider
from nemo_gym.sandbox.api import _AsyncLoopRunner


@dataclass
class GymSandboxEnvironmentConfig:
    image: str
    cwd: str = "/"
    timeout: int = 300
    env: dict[str, str] = field(default_factory=dict)
    provider: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    answer_leak_blocklist: list[str] | None = None


class GymSandboxEnvironment:
    def __init__(self, *, config_class: type = GymSandboxEnvironmentConfig, **kwargs: Any) -> None:
        self.config = config_class(**kwargs)
        self.instance_id: str | None = None
        self.handle = None
        provider = copy.deepcopy(self.config.provider)
        for block in provider.values():
            # The global aiohttp client belongs to the server's loop, not ours.
            if isinstance(block, dict) and isinstance(block.get("connection"), dict):
                block["connection"]["transport_backend"] = "httpx"
        self._runner = _AsyncLoopRunner(wait_timeout_s=7 * 24 * 3600)
        self._provider = self._runner.call("create_provider", lambda: create_provider(provider))

    def _run(self, name: str, factory):
        return self._runner.run(name, factory)

    def start(self) -> None:
        spec = dict(self.config.spec)
        metadata = dict(spec.pop("metadata", {}))
        if self.instance_id:
            metadata["instance_id"] = self.instance_id[:63]
        sandbox_spec = SandboxSpec(
            image=self.config.image,
            workdir=self.config.cwd,
            env=dict(self.config.env),
            metadata=metadata,
            **spec,
        )
        self.handle = self._run("create", lambda: self._provider.create(sandbox_spec))

    def descriptor(self) -> dict[str, Any]:
        return self._run("serialize", lambda: self._provider.serialize_handle(self.handle))

    def execute(self, command: str, cwd: str = "", timeout: int | None = None, **_: Any) -> dict[str, Any]:
        timeout = timeout or self.config.timeout
        try:
            result = self._run(
                "exec",
                lambda: self._provider.exec(
                    self.handle, command, cwd=cwd or self.config.cwd, timeout_s=timeout, env=self.config.env or None
                ),
            )
        except Exception as e:
            return {"output": f"{type(e).__name__}: {e}", "returncode": None, "reason": "transport_error"}
        # OpenSandbox's execd reports a non-zero exit by appending this line to stderr.
        stderr = re.sub(rf"(?:^|\n)CommandExecError: {result.return_code}\s*$", "", result.stderr or "")
        output = "".join(part for part in (result.stdout, stderr) if part)
        return {"output": output, "returncode": result.return_code, "reason": result.error_type or "ok"}

    def execute_detached(self, command: str, cwd: str = "", timeout: int | None = None, **_: Any) -> dict[str, Any]:
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path: str, dest_path: str, **_: Any) -> None:
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(src_path)
        parent = os.path.dirname(dest_path.rstrip("/")) or "/"
        self.execute(f"mkdir -p '{parent}'", cwd="/")
        if src.is_file():
            self._run("upload", lambda: self._provider.upload_file(self.handle, src, dest_path))
            return
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "dir.tar"
            with tarfile.open(archive, "w", dereference=True) as tar:
                tar.add(src, arcname=".")
            remote = f"/tmp/_mimo_copy_{os.getpid()}_{id(archive)}.tar"
            self._run("upload", lambda: self._provider.upload_file(self.handle, archive, remote))
        res = self.execute(f"mkdir -p '{dest_path}' && tar xf {remote} -C '{dest_path}' && rm -f {remote}", cwd="/")
        if res["returncode"] != 0:
            raise RuntimeError(f"copy_to {dest_path} failed: {res['output'][:500]}")

    def copy_out(self, src_path: str, dest_path: str, **_: Any) -> None:
        self._run("download", lambda: self._provider.download_file(self.handle, src_path, Path(dest_path)))

    def get_template_vars(self) -> dict[str, Any]:
        return {"cwd": self.config.cwd, "image": self.config.image}

    def cleanup(self) -> None:
        if self.handle is not None:
            handle, self.handle = self.handle, None
            try:
                self._run("close", lambda: self._provider.close(handle))
            except Exception:
                pass
        self._runner.close()
