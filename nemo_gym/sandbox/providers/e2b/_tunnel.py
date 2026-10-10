# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TCP relay run inside a sandbox with operator-provided aiohttp and pyroute2 installations."""

import asyncio
import json
import sys
from pathlib import Path

from aiohttp import ClientError, ClientSession, WSMsgType, web


async def bridge(reader, writer, websocket):
    """Preserve binary data and TCP half-close over an authenticated WebSocket."""

    async def to_websocket():
        while data := await reader.read(65536):
            await websocket.send_bytes(data)
        await websocket.send_str("eof")

    async def to_tcp():
        async for message in websocket:
            if message.type == WSMsgType.BINARY:
                writer.write(message.data)
                await writer.drain()
            elif message.type == WSMsgType.TEXT and message.data == "eof":
                if writer.can_write_eof():
                    writer.write_eof()
                return
            else:
                raise ConnectionError("TCP tunnel closed unexpectedly")
        raise ConnectionError("TCP tunnel closed before EOF")

    tasks = [asyncio.create_task(to_websocket()), asyncio.create_task(to_tcp())]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        await websocket.close()


async def main(config):
    if config.get("local_addresses"):
        from pyroute2 import IPRoute

        # Namespace-local addresses preserve peer identity without advertising
        # routes or changing the host/cluster network.
        def configure_addresses():
            with IPRoute() as routes:
                interface = routes.link_lookup(ifname="lo")[0]
                for address in config["local_addresses"]:
                    routes.addr("add", index=interface, address=address, prefixlen=32)

        await asyncio.to_thread(configure_addresses)

    async def incoming(request):
        if request.headers.get("X-Gym-Tunnel-Token") != config["token"]:
            raise web.HTTPForbidden()
        try:
            port = int(request.query["port"])
        except (KeyError, ValueError):
            raise web.HTTPBadRequest() from None
        if port not in config["ports"]:
            raise web.HTTPForbidden()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        websocket = web.WebSocketResponse(heartbeat=30)
        await websocket.prepare(request)
        try:
            await bridge(reader, writer, websocket)
        except (OSError, ConnectionError):
            pass
        return websocket

    async with ClientSession() as session:

        async def outgoing(reader, writer, peer, port):
            try:
                async with session.ws_connect(
                    peer["url"], params={"port": port}, headers=peer["headers"], heartbeat=30
                ) as websocket:
                    await bridge(reader, writer, websocket)
            except (OSError, ClientError, ConnectionError):
                # A failed upstream connection is a failed client socket; it must
                # not bring down unrelated services or leave their writers open.
                writer.close()

        servers = []
        runner = None
        try:
            if "token" in config:
                app = web.Application()
                app.router.add_get("/{path:.*}", incoming)
                runner = web.AppRunner(app, access_log=None)
                await runner.setup()
                await web.TCPSite(runner, "0.0.0.0", config["tunnel_port"]).start()
            for peer in config["peers"]:
                for address in peer["addresses"]:
                    for port in peer["ports"]:

                        async def relay(reader, writer, peer=peer, port=port):
                            await outgoing(reader, writer, peer, port)

                        servers.append(await asyncio.start_server(relay, address, port))
            Path(config["ready_file"]).write_text("ready\n")
            await asyncio.Event().wait()
        finally:
            for server in servers:
                server.close()
            await asyncio.gather(*(server.wait_closed() for server in servers))
            if runner:
                await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main(json.loads(Path(sys.argv[1]).read_text())))
