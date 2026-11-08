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
import ctypes
import fcntl
import ipaddress
import json
import os
import signal
import socket
import struct
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


def origin(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Expected an HTTP(S) model URL without embedded credentials")
    return parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)


def local_addresses() -> set[str]:
    """Read all local interface addresses before entering the task namespace."""

    class Ifaddrs(ctypes.Structure):
        pass

    Ifaddrs._fields_ = [
        ("next", ctypes.POINTER(Ifaddrs)),
        ("name", ctypes.c_char_p),
        ("flags", ctypes.c_uint),
        ("address", ctypes.c_void_p),
        ("netmask", ctypes.c_void_p),
        ("destination", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
    ]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.getifaddrs.argtypes = [ctypes.POINTER(ctypes.POINTER(Ifaddrs))]
    libc.freeifaddrs.argtypes = [ctypes.POINTER(Ifaddrs)]
    head = ctypes.POINTER(Ifaddrs)()
    if libc.getifaddrs(ctypes.byref(head)):
        raise OSError(ctypes.get_errno(), "Cannot enumerate interfaces for public-web isolation")
    addresses = set()
    current = head
    try:
        while current:
            address = current.contents.address
            if address:
                family = ctypes.cast(address, ctypes.POINTER(ctypes.c_ushort)).contents.value
                if family == socket.AF_INET:
                    addresses.add(str(ipaddress.ip_address(ctypes.string_at(address, 16)[4:8])))
                elif family == socket.AF_INET6:
                    addresses.add(str(ipaddress.ip_address(ctypes.string_at(address, 28)[8:24])))
            current = current.contents.next
    finally:
        libc.freeifaddrs(head)
    if not addresses:
        raise ValueError("Cannot establish local addresses for public-web isolation")
    return addresses


async def resolved_addresses(destination: tuple[str, int]) -> list[tuple[int, str]]:
    records = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(*destination, type=socket.SOCK_STREAM), 10)
    return list(dict.fromkeys((family, str(ipaddress.ip_address(address[0]))) for family, _, _, _, address in records))


def public_address(address: str) -> bool:
    parsed = ipaddress.ip_address(address)
    if not parsed.is_global or parsed.is_multicast:
        return False
    if isinstance(parsed, ipaddress.IPv6Address):
        # Refuse mapped/transition forms that can embed a different destination,
        # including NAT64, 6to4 and Teredo. Native public IPv6 remains supported.
        return parsed in ipaddress.ip_network("2000::/3") and parsed.sixtofour is None and parsed.teredo is None
    return True


class PublicWebPolicy:
    """Resolve and check each connection before opening a numeric destination."""

    def __init__(
        self,
        allowed: set[tuple[str, int]],
        protected_addresses: set[str],
        *,
        protected_origins: set[tuple[str, int]] | None = None,
        allow_public: bool = True,
    ):
        self.allowed = allowed
        self.protected_addresses = set(protected_addresses)
        self.protected_origins = protected_origins or set()
        self.protected_endpoints: set[tuple[str, int]] = set()
        self.allow_public = allow_public

    async def refresh(self) -> dict[tuple[str, int], list[tuple[int, str]]]:
        """Refresh protected DNS before every connection; retain earlier addresses too.

        DNS failure is an error, never permission to bypass a protected origin.
        Explicit grants may use a control host on a different port, but cannot
        share a control endpoint even under a different hostname.
        """
        if self.allowed & self.protected_origins:
            raise PermissionError("Protected control service shares an explicitly allowed origin")
        self.protected_addresses.update(local_addresses())
        origins = sorted(self.allowed | self.protected_origins)
        records = dict(
            zip(origins, await asyncio.gather(*(resolved_addresses(item) for item in origins)), strict=True)
        )
        if any(not addresses for addresses in records.values()):
            raise OSError("Cannot resolve all model/tool/control origins")
        for endpoint, addresses in records.items():
            self.protected_addresses.update(address for _, address in addresses)
            if endpoint in self.protected_origins:
                self.protected_endpoints.update((address, endpoint[1]) for _, address in addresses)
        for endpoint in self.allowed:
            if any((address, endpoint[1]) in self.protected_endpoints for _, address in records[endpoint]):
                raise PermissionError("Model/tool grant resolves to a protected control endpoint")
        return records

    async def connect(self, destination: tuple[str, int]) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        records = await self.refresh()
        addresses = records.get(destination)
        if addresses is None:
            addresses = await resolved_addresses(destination)
        if not addresses:
            raise OSError("Destination has no usable addresses")
        if destination not in self.allowed and (
            not self.allow_public
            or any(not public_address(address) or address in self.protected_addresses for _, address in addresses)
        ):
            raise PermissionError("Destination is not an admitted public-web origin")
        # Connect to a validated numeric address, never resolve the hostname again.
        # Validate every answer before connecting so mixed private/public DNS fails closed.
        for family, address in addresses:
            try:
                return await asyncio.wait_for(
                    asyncio.open_connection(address, destination[1], family=family, flags=socket.AI_NUMERICHOST), 15
                )
            except OSError:
                continue
        raise OSError("No validated destination address was reachable")


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


async def proxy(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    allowed: set[tuple[str, int]],
    public_web: PublicWebPolicy | None = None,
) -> None:
    remote_writer = None
    try:
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15)
        lines = raw.decode("iso-8859-1").split("\r\n")
        method, target, version = lines[0].split(" ")
        if version not in {"HTTP/1.0", "HTTP/1.1"}:
            raise ValueError("Unsupported HTTP version")
        parsed = urlsplit("https://" + target if method == "CONNECT" else target)
        destination = origin(parsed.geturl())
        if (public_web is None and destination not in allowed) or (method != "CONNECT" and parsed.scheme != "http"):
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return
        if public_web is None:
            remote_reader, remote_writer = await asyncio.wait_for(asyncio.open_connection(*destination), 15)
        else:
            try:
                remote_reader, remote_writer = await public_web.connect(destination)
            except PermissionError:
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return
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
    process = await asyncio.create_subprocess_exec(*command, env=env, start_new_session=True)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: os.killpg(process.pid, s) if process.returncode is None else None)
    try:
        return await process.wait()
    finally:
        # Kill the attempt's process group, including children that outlive the
        # immediate command. Never signal the caller's own process group.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.returncode is None:
            await asyncio.wait_for(process.wait(), 5)
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


async def run(args: argparse.Namespace) -> int:
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
                "OPENCODE_ENABLE_PARALLEL",
            ):
                settings.extend(["--setenv", key, env[key]])
            args.command[position:position] = settings
        async with server:
            return await run_child(args.command, env)
    allowed = {origin(args.model_url)}
    if args.mode in {"model_and_tools", "inherit"}:
        allowed.update(origin(url) for url in args.tool_url)
    public_web = PublicWebPolicy(
        allowed,
        local_addresses(),
        protected_origins={origin(url) for url in args.protect_url},
        allow_public=args.mode == "inherit",
    )
    await public_web.refresh()
    # /tmp keeps Unix socket addresses below Linux's length limit.
    with tempfile.TemporaryDirectory(prefix="gym-net-", dir="/tmp") as directory:
        path = str(Path(directory) / "proxy.sock")
        server = await asyncio.start_unix_server(lambda r, w: proxy(r, w, allowed, public_web), path)
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
                    "-I",
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
        "--mode",
        choices=("model_only", "model_and_tools", "inherit", "none", "all"),
        required=True,
    )
    parser.add_argument("--model-url")
    parser.add_argument("--protect-url", action="append", default=[], help="Control-service origin to deny")
    parser.add_argument("--tool-url", action="append", default=[], help="Accepted MCP grant URL (repeatable)")
    parser.add_argument("--inside", help=argparse.SUPPRESS)
    parser.add_argument("--bubblewrap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.mode = {
        "none": "model_only",
        "all": "inherit",
    }.get(args.mode, args.mode)
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command:
        parser.error("A command is required")
    if not args.inside and not args.model_url:
        config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT") or Path("opencode.json").read_text())
        args.model_url = config["provider"]["nemo"]["options"]["baseURL"]
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
