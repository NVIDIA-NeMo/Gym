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

"""File transfer in and out of an IdeGYM sandbox, over its bash tool.

IdeGYM's filesystem API is unusable here: its read endpoint streams raw bytes while the
orchestrator forwards requests as JSON text, and its typed file endpoints only write
UTF-8. So bytes travel base64-encoded through the bash tool, chunked both ways — uploads
to stay inside :data:`~nemo_gym.sandbox.providers.idegym.config.MAX_COMMAND_BYTES`,
downloads because each chunk's stdout is persisted as an async-operation result.

Fine for the config files, patches, and logs a benchmark moves around, but not a bulk
channel — have the sandbox fetch large inputs itself.
"""

import asyncio
import base64
import binascii
import logging
import posixpath
from collections.abc import Awaitable, Callable
from pathlib import Path

from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.providers.idegym.config import IdeGymFilesConfig
from nemo_gym.sandbox.providers.idegym.errors import IdeGymTransferError
from nemo_gym.sandbox.providers.idegym.shell import quote


LOGGER = logging.getLogger(__name__)

# Runs one command in the sandbox; the provider binds it to a handle.
ExecRunner = Callable[..., Awaitable[SandboxExecResult]]


def _describe(result: SandboxExecResult) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    return f"exit code {result.return_code}" + (f": {detail}" if detail else "")


class Base64BashFileTransfer:
    """Moves single files between the local filesystem and an IdeGYM sandbox."""

    def __init__(self, config: IdeGymFilesConfig, run: ExecRunner) -> None:
        self._config = config
        self._run = run

    async def upload(self, source_path: Path, target_path: str) -> None:
        """Upload one local file, creating its parent directory in the sandbox."""
        data = await asyncio.to_thread(Path(source_path).read_bytes)
        chunk_size = self._config.upload_chunk_bytes
        # An empty file still needs one (empty) chunk, so that it gets created.
        chunks = [data[offset : offset + chunk_size] for offset in range(0, len(data), chunk_size)] or [b""]
        for index, chunk in enumerate(chunks):
            result = await self._exec(self._write_chunk_script(target_path, chunk, first=index == 0))
            if result.return_code != 0:
                raise IdeGymTransferError(
                    f"Uploading {source_path} to {target_path!r} failed on chunk "
                    f"{index + 1}/{len(chunks)} ({_describe(result)})"
                )

    async def download(self, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to ``target_path``."""
        size = await self._remote_size(source_path)
        cap = self._config.max_download_bytes
        if cap is not None and size > cap:
            raise IdeGymTransferError(
                f"Refusing to download {source_path!r}: {size} bytes exceeds files.max_download_bytes "
                f"({cap}). Archive and split it in the sandbox, or raise the limit."
            )
        chunk_size = self._config.download_chunk_bytes
        data = bytearray()
        # Each chunk asserts its own decoded length, so the concatenation needs no
        # separate check and a file that shrinks mid-read fails on the short chunk.
        for offset in range(0, size, chunk_size):
            data.extend(await self._read_chunk(source_path, offset, min(chunk_size, size - offset)))
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, bytes(data))

    async def _exec(self, script: str) -> SandboxExecResult:
        return await self._run(script, timeout_s=self._config.timeout_s)

    # --- script construction ----------------------------------------------

    def _write_chunk_script(self, target_path: str, chunk: bytes, *, first: bool) -> str:
        """Decode one base64 chunk into ``target_path``, appending unless ``first``.

        Chunks are encoded independently, so appending the decoded bytes rebuilds the
        file exactly, whatever the chunk alignment.
        """
        lines = []
        if first and (parent := posixpath.dirname(target_path)):
            lines.append(f"mkdir -p {quote(parent)}")
        # The pipeline's exit code is `base64 -d`'s, so a decode or write error surfaces.
        encoded = quote(base64.b64encode(chunk).decode("ascii"))
        lines.append(f"printf '%s' {encoded} | base64 -d {'>' if first else '>>'} {quote(target_path)}")
        return "\n".join(lines)

    def _size_script(self, source_path: str) -> str:
        path = quote(source_path)
        # `-f` rejects directories and devices, whose read would never end; the redirect
        # keeps a leading-dash path from being taken as a `wc` option.
        return f"[ -f {path} ] || {{ printf 'not a regular file\\n' >&2; exit 1; }}\nwc -c < {path}"

    def _read_chunk_script(self, source_path: str, offset: int, length: int) -> str:
        # No `pipefail`: `head -c` closes the pipe once it has its bytes, so `tail` dies
        # of SIGPIPE and would make a good chunk look failed. `_read_chunk` checks the
        # decoded length instead.
        return f"tail -c +{offset + 1} < {quote(source_path)} | head -c {length} | base64"

    # --- sandbox round trips ----------------------------------------------

    async def _remote_size(self, source_path: str) -> int:
        result = await self._exec(self._size_script(source_path))
        if result.return_code != 0:
            raise IdeGymTransferError(f"Cannot read {source_path!r} from the sandbox ({_describe(result)})")
        text = (result.stdout or "").strip()
        if not text.isdigit():
            raise IdeGymTransferError(f"The sandbox reported an unparsable size for {source_path!r}: {text[:200]!r}")
        return int(text)

    async def _read_chunk(self, source_path: str, offset: int, length: int) -> bytes:
        what = f"{length} bytes at offset {offset} of {source_path!r}"
        result = await self._exec(self._read_chunk_script(source_path, offset, length))
        if result.return_code != 0:
            raise IdeGymTransferError(f"Reading {what} failed ({_describe(result)})")
        try:
            chunk = base64.b64decode("".join((result.stdout or "").split()), validate=True)
        except (binascii.Error, ValueError) as e:
            raise IdeGymTransferError(f"The sandbox returned invalid base64 for {source_path!r}: {e}") from e
        if len(chunk) != length:
            raise IdeGymTransferError(f"Expected {what} but got {len(chunk)}")
        return chunk
