# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward loopback ports to the same ports on a host alias.

Services on a workstation bind loopback, which a container cannot reach, and a host firewall may drop
connections to the host's other addresses. This relay listens on the container's loopback for each port
and forwards to the same port on a host alias that reaches the host's loopback (colima:
host.lima.internal; Docker Desktop: host.docker.internal). Standard library only.

    python3 relay.py <host alias> <port> [<port> ...]
"""

import asyncio
import sys


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError):
        pass


async def serve(host: str, port: int) -> asyncio.Server:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
        except OSError:
            writer.close()
            return
        try:
            await asyncio.gather(pipe(reader, upstream_writer), pipe(upstream_reader, writer))
        finally:
            upstream_writer.close()
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", port)


async def main(host: str, ports: list[int]) -> None:
    servers = [await serve(host, port) for port in ports]
    print(f"relay ready: {' '.join(map(str, ports))} -> {host}", flush=True)
    await asyncio.gather(*(server.serve_forever() for server in servers))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], [int(port) for port in sys.argv[2:]]))
