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
"""Small Docker operations for the pinned Harbor execution profile.

Harbor owns creation and deletion. This helper accepts the exact main-container
ID resolved by Harbor, not an arbitrary hostname or a network-sidecar ID.
"""

import asyncio
import json
import re
from pathlib import Path


async def command(*args: str, input_bytes: bytes | None = None) -> bytes:
    """Run a bounded host command, preserving failures rather than ignoring them."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(30):
            stdout, stderr = await process.communicate(input_bytes)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({process.returncode}): {stderr.decode(errors='replace')[-2000:]}")
    return stdout


class TrialContainer:
    """Operate only within one Harbor-owned Docker container."""

    def __init__(self, container_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", container_id):
            raise ValueError("Expected the exact main-container ID from Docker Compose")
        self.id = container_id
        self._protected: dict[str, str] | None = None

    async def execute(self, *args: str, root: bool = False, input_bytes: bytes | None = None) -> bytes:
        return await command(
            "docker", "exec", "-i", "-u", "0" if root else "1000", self.id, *args, input_bytes=input_bytes
        )

    async def validate(self) -> dict[str, object]:
        """Check the historical compute/user limits before starting a trial."""
        info = json.loads(await command("docker", "inspect", self.id))[0]
        host = info["HostConfig"]
        uid = (await command("docker", "exec", self.id, "id", "-u")).decode().strip()
        if uid != "1000" or host["NanoCpus"] != 500_000_000 or host["Memory"] != 768 * 1024**2:
            raise RuntimeError("Container user/CPU/memory differs from the frozen profile")
        if (
            host["PidsLimit"] != 256
            or host["CapDrop"] != ["ALL"]
            or not any(option.startswith("no-new-privileges") for option in host["SecurityOpt"] or [])
        ):
            raise RuntimeError("Container capability/PID settings differ from the frozen profile")
        return {
            "image": info["Image"],
            "uid": uid,
            "cpu_nano": host["NanoCpus"],
            "memory": host["Memory"],
            "network_mode": host["NetworkMode"],
            "cap_drop": host["CapDrop"],
            "security_opt": host["SecurityOpt"],
            "pids_limit": host["PidsLimit"],
        }

    async def isolate_catalog(self) -> None:
        """Copy up all files so parallel trials never watch the same image inode."""
        script = """import pathlib,shutil
source=pathlib.Path('/skills')
target=pathlib.Path('/skills-private')
if target.exists():raise RuntimeError('Catalog already copied')
shutil.copytree(source,target,symlinks=False)
for p in source.rglob('*'):
 if p.is_dir():p.chmod(0o755)
source.chmod(0o755)
shutil.rmtree(source)
target.rename(source)
for p in [source,*source.rglob('*')]:
 p.chmod(p.stat().st_mode & ~0o222)
"""
        await self.execute("python3", "-c", script, root=True)

    async def catalog_hashes(self) -> dict[str, str]:
        """Hash the staged directories with the historical mounted-tree ordering."""
        script = """import hashlib,json,pathlib
result={}
for d in sorted(pathlib.Path('/skills').iterdir()):
 if not d.is_dir() or not (d/'SKILL.md').is_file():raise RuntimeError('Unexpected catalog member')
 entries=[]
 for p in sorted(d.rglob('*')):
  if p.is_symlink():raise RuntimeError('Catalog contains symlink')
  if p.is_file():
   local=p.relative_to(d)
   if set(local.parts)&{'evals','.git','__pycache__'}:raise RuntimeError('Hidden labels in mounted catalog')
   entries.append((str(local),hashlib.sha256(p.read_bytes()).hexdigest()))
 result[d.name]=hashlib.sha256(json.dumps(entries,separators=(',',':')).encode()).hexdigest()
print(json.dumps(result))
"""
        return json.loads(await self.execute("python3", "-c", script, root=True))

    async def preserve_initial_processes(self) -> None:
        """Protect Harbor's initial processes using PID plus process start time."""
        if self._protected is not None:
            raise RuntimeError("Process boundary already recorded")
        script = """import json,pathlib
out={}
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try:out[p.name]=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
  except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(out))
"""
        self._protected = json.loads(await self.execute("python3", "-c", script, root=True))

    async def stop_work(self) -> None:
        """Terminate all newly created agent-user processes and confirm none remain."""
        if self._protected is None:
            raise RuntimeError("Missing initial process boundary")
        script = """import json,os,pathlib,signal,sys
protected=json.loads(sys.argv[1]);sig=int(sys.argv[2]);alive=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  if p.stat().st_uid!=1000 or fields[0]=='Z' or protected.get(p.name)==fields[19]:continue
  alive.append(p.name)
  if sig:os.kill(int(p.name),sig)
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(alive))
"""
        for sig in (15, 9):
            await self.execute("python3", "-c", script, json.dumps(self._protected), str(sig))
            for _ in range(10):
                alive = json.loads(await self.execute("python3", "-c", script, json.dumps(self._protected), "0"))
                if not alive:
                    return
                await asyncio.sleep(0.05)
        raise RuntimeError("Agent work is still running after termination")

    async def start_observer(self, *, paths: dict[str, str]) -> asyncio.subprocess.Process:
        """Start the independent root observer with its own captured stdout pipe."""
        source = Path(__file__).with_name("observe.py").read_text()
        return await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-u",
            "0",
            self.id,
            "python3",
            "-u",
            "-c",
            source,
            json.dumps(paths),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def stop_observer(self, *, pid: int) -> None:
        """Signal the root observer without relying on an installed kill binary."""
        if pid <= 1:
            raise ValueError("Invalid observer PID")
        await self.execute(
            "python3", "-c", "import os,signal,sys; os.kill(int(sys.argv[1]),signal.SIGTERM)", str(pid), root=True
        )
