# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linux network wrapper; requires user/network/mount/PID namespaces.

The outer process owns network connections. The harness has only loopback and a
Unix-socket bridge to an origin allowlist. This is a network boundary, not a
replacement for the caller's filesystem sandbox. Do not expose host sockets or
credentials through that sandbox. Runtime installation happens before this wrapper.
"""

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import signal
import socket
import struct
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


SEARCH_ORIGIN = ("mcp.exa.ai", 443)


def origin(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Expected an HTTP(S) model URL without embedded credentials")
    return parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)


async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        with contextlib.suppress(OSError, RuntimeError):
            writer.write_eof()


async def splice(
    left_reader: asyncio.StreamReader,
    left_writer: asyncio.StreamWriter,
    right_reader: asyncio.StreamReader,
    right_writer: asyncio.StreamWriter,
) -> None:
    tasks = [
        asyncio.create_task(relay(left_reader, right_writer)),
        asyncio.create_task(relay(right_reader, left_writer)),
    ]
    try:
        # Stop when either peer closes. No background tool may keep the proxy alive after the harness exits.
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        right_writer.close()


async def proxy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, allowed: set[tuple[str, int]]) -> None:
    remote_writer = None
    try:
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15)
        lines = raw.decode("iso-8859-1").split("\r\n")
        method, target, version = lines[0].split(" ")
        if version not in {"HTTP/1.0", "HTTP/1.1"}:
            raise ValueError("Unsupported HTTP version")
        parsed = urlsplit("https://" + target if method == "CONNECT" else target)
        destination = origin(parsed.geturl())
        if destination not in allowed or (method != "CONNECT" and parsed.scheme != "http"):
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return
        remote_reader, remote_writer = await asyncio.wait_for(asyncio.open_connection(*destination), 15)
        if method == "CONNECT":
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
        else:
            # Ignore Host supplied by the client; authority always comes from the validated URL.
            headers = [
                line
                for line in lines[1:]
                if line
                and line.split(":", 1)[0].lower()
                not in {"host", "proxy-authorization", "proxy-connection", "connection"}
            ]
            headers += ["Host: " + parsed.netloc, "Connection: close"]
            path = (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
            remote_writer.write(
                (f"{method} {path} {version}\r\n" + "\r\n".join(headers) + "\r\n\r\n").encode("iso-8859-1")
            )
            await remote_writer.drain()
        await splice(reader, writer, remote_reader, remote_writer)
    except (ValueError, OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
        with contextlib.suppress(OSError):
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
    finally:
        if remote_writer is not None:
            remote_writer.close()
        writer.close()


async def bridge(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, path: str) -> None:
    try:
        remote_reader, remote_writer = await asyncio.open_unix_connection(path)
        await splice(reader, writer, remote_reader, remote_writer)
    except OSError:
        pass
    finally:
        writer.close()


def loopback_up() -> None:
    # SIOCGIFFLAGS/SIOCSIFFLAGS: only this new network namespace is changed.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        flags = fcntl.ioctl(sock, 0x8913, struct.pack("16sH14x", b"lo", 0))
        current = struct.unpack("16sH14x", flags)[1]
        fcntl.ioctl(sock, 0x8914, struct.pack("16sH14x", b"lo", current | 1))


async def run_child(command: list[str], env: dict[str, str] | None = None) -> int:
    process = await asyncio.create_subprocess_exec(*command, env=env)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: process.send_signal(s) if process.returncode is None else None)
    try:
        return await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def run(args: argparse.Namespace) -> int:
    if args.mode == "inherit":
        os.execvp(args.command[0], args.command)
    if args.inside:
        loopback_up()
        server = await asyncio.start_server(lambda r, w: bridge(r, w, args.inside), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        env = {
            k: v
            for k, v in os.environ.items()
            if k.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
        }
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[key] = f"http://127.0.0.1:{port}"
        env["NO_PROXY"] = env["no_proxy"] = ""
        # Pin the stock search provider; no arbitrary pages are reachable through the proxy.
        env["OPENCODE_WEBSEARCH_PROVIDER"] = "exa"
        env["OPENCODE_ENABLE_EXA"] = (
            os.environ.get("OPENCODE_ENABLE_EXA", "1") if args.mode == "model_and_websearch" else "0"
        )
        env["OPENCODE_ENABLE_PARALLEL"] = "0"
        if args.bubblewrap:
            # Bubblewrap clears the inherited environment; forward only this wrapper's network settings.
            position = args.command.index("--clearenv") + 1
            settings = []
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "NO_PROXY",
                "no_proxy",
                "OPENCODE_WEBSEARCH_PROVIDER",
                "OPENCODE_ENABLE_EXA",
                "OPENCODE_ENABLE_PARALLEL",
            ):
                settings.extend(["--setenv", key, env[key]])
            args.command[position:position] = settings
        async with server:
            return await run_child(args.command, env)
    allowed = {origin(args.model_url)}
    if args.mode == "model_and_websearch":
        allowed.add(SEARCH_ORIGIN)
    # /tmp keeps Unix socket addresses below Linux's length limit.
    with tempfile.TemporaryDirectory(prefix="gym-net-") as directory:
        path = str(Path(directory) / "proxy.sock")
        server = await asyncio.start_unix_server(lambda r, w: proxy(r, w, allowed), path)
        os.chmod(path, 0o600)
        async with server:
            return await run_child(
                [
                    "unshare",
                    "--user",
                    "--map-root-user",
                    "--net",
                    "--mount",
                    "--pid",
                    "--fork",
                    "--kill-child",
                    "--mount-proc",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mode",
                    args.mode,
                    "--inside",
                    path,
                    *(["--bubblewrap"] if args.bubblewrap else []),
                    "--",
                    *args.command,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("model_only", "model_and_websearch", "inherit", "none", "websearch", "all"), required=True
    )
    parser.add_argument("--model-url")
    parser.add_argument("--inside", help=argparse.SUPPRESS)
    parser.add_argument("--bubblewrap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.mode = {"none": "model_only", "websearch": "model_and_websearch", "all": "inherit"}.get(args.mode, args.mode)
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command:
        parser.error("A command is required")
    if args.mode != "inherit" and not args.inside and not args.model_url:
        config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT") or Path("opencode.json").read_text())
        args.model_url = config["provider"]["nemo"]["options"]["baseURL"]
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
