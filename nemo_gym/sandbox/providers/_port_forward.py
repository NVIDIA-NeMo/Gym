# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone loopback TCP relay executed inside a sandbox."""

import asyncio
import errno
import socket
import sys
from pathlib import Path


async def main():
    target, ready_file, *ports = sys.argv[1:]

    async def relay(reader, writer):
        upstream = None
        try:
            remote_reader, upstream = await asyncio.open_connection(target, writer.get_extra_info("sockname")[1])

            async def copy(source, destination):
                while data := await source.read(65536):
                    destination.write(data)
                    await destination.drain()
                if destination.can_write_eof():
                    destination.write_eof()

            await asyncio.gather(copy(reader, upstream), copy(remote_reader, writer))
        except (OSError, ConnectionError):
            pass
        finally:
            writer.close()
            if upstream is not None:
                upstream.close()

    servers = []
    try:
        for port in ports:
            servers.append(await asyncio.start_server(relay, "127.0.0.1", int(port)))
            try:
                servers.append(await asyncio.start_server(relay, "::1", int(port), family=socket.AF_INET6))
            except OSError as error:
                if error.errno not in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                    raise
        Path(ready_file).write_text("ready\n")
        await asyncio.Event().wait()
    finally:
        for server in servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in servers))


if __name__ == "__main__":
    asyncio.run(main())
