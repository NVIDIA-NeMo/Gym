# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE-Together dependency proxy, outside all task sandboxes.

The controller registers immutable package restrictions using an admin secret.
Workloads receive only an opaque lease token and the public CA certificate.
Registry TLS is inspected; other reference-allowed hosts use ordinary CONNECT.
"""

import argparse
import base64
import hmac
import http.client
import json
import logging
import os
import re
import select
import shutil
import socket
import ssl
import subprocess
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from environment_servers.swe_together.egress_policy import (
    REGISTRY_HOSTS,
    DenyPackages,
    allowed_host,
    registry_decision,
    validate_task_allow,
)


LOGGER = logging.getLogger(__name__)
TOKEN = re.compile(r"[a-f0-9]{64}\Z")
HOP_HEADERS = {
    "connection",
    "proxy-connection",
    "proxy-authorization",
    "proxy-authenticate",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class RegistryCA:
    """Per-service CA and fixed registry leaves; private material stays on the host."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        self.certificate = directory / "ca.pem"
        key = directory / "ca.key"
        openssl = shutil.which("openssl")
        if openssl is None:
            raise RuntimeError("SWE-Together dependency proxy requires OpenSSL on its trusted host")

        def run(*args):
            subprocess.run([openssl, *args], check=True, capture_output=True, timeout=30)

        if not self.certificate.exists():
            run(
                "req",
                "-x509",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:P-256",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(self.certificate),
                "-days",
                "30",
                "-subj",
                "/CN=SWE-Together dependency CA",
                "-addext",
                "basicConstraints=critical,CA:TRUE",
                "-addext",
                "keyUsage=critical,keyCertSign,cRLSign",
            )
        self.contexts = {}
        for host in REGISTRY_HOSTS:
            cert, leaf_key, csr = (directory / (host + suffix) for suffix in (".pem", ".key", ".csr"))
            extensions = directory / (host + ".ext")
            extensions.write_text(
                f"subjectAltName=DNS:{host}\nbasicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\n"
            )
            run(
                "req",
                "-new",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:P-256",
                "-nodes",
                "-keyout",
                str(leaf_key),
                "-out",
                str(csr),
                "-subj",
                f"/CN={host}",
            )
            run(
                "x509",
                "-req",
                "-in",
                str(csr),
                "-CA",
                str(self.certificate),
                "-CAkey",
                str(key),
                "-set_serial",
                str(int.from_bytes(os.urandom(16), "big")),
                "-out",
                str(cert),
                "-days",
                "7",
                "-extfile",
                str(extensions),
            )
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(cert, leaf_key)
            self.contexts[host] = context
        for path in directory.glob("*.key"):
            path.chmod(0o600)


@dataclass(frozen=True)
class Lease:
    packages: DenyPackages
    task_allow: tuple[str, ...]
    expires: float
    record: dict


class DependencyProxy(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, *, admin_key: str, ca: RegistryCA):
        if len(admin_key) < 32:
            raise ValueError("Dependency proxy admin key must contain at least 32 characters")
        self.admin_key = admin_key
        self.ca = ca
        self.leases: dict[str, Lease] = {}
        self.lock = threading.Lock()
        super().__init__(address, ProxyRequest)

    def lease(self, token: str) -> Lease | None:
        with self.lock:
            now = time.monotonic()
            for expired in [key for key, value in self.leases.items() if value.expires <= now]:
                del self.leases[expired]
            return self.leases.get(token)


class ProxyRequest(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 60

    def log_message(self, *_):
        # Request paths and Proxy-Authorization can contain workload secrets.
        pass

    def reply(self, status: int, body: bytes = b"", content_type: str = "text/plain"):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", content_type)
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def authorized_lease(self) -> Lease | None:
        try:
            scheme, encoded = self.headers.get("Proxy-Authorization", "").split(" ", 1)
            if scheme.lower() != "basic":
                return None
            username, separator, token = base64.b64decode(encoded, validate=True).decode().partition(":")
            if not separator or username != "swet" or not TOKEN.fullmatch(token):
                return None
            return self.server.lease(token)
        except (ValueError, UnicodeError):
            return None

    def body(self, limit: int) -> bytes:
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked request bodies are unsupported")
        values = self.headers.get_all("Content-Length", [])
        if len(values) > 1:
            raise ValueError("Ambiguous request length")
        size = int(values[0]) if values else 0
        if not 0 <= size <= limit:
            raise ValueError("Request body too large")
        value = self.rfile.read(size)
        if len(value) != size:
            raise ValueError("Incomplete request body")
        return value

    def control(self) -> bool:
        if not self.path.startswith("/"):
            return False
        if self.path == "/ca" and self.command == "GET":
            self.reply(200, self.server.ca.certificate.read_bytes(), "application/x-pem-file")
            return True
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.server.admin_key):
            self.reply(403)
            return True
        token = self.path.removeprefix("/leases/")
        if not self.path.startswith("/leases/") or not TOKEN.fullmatch(token):
            self.reply(404)
            return True
        try:
            with self.server.lock:
                if self.command == "DELETE":
                    self.server.leases.pop(token, None)
                elif self.command == "PUT":
                    record = json.loads(self.body(256 * 1024))
                    if set(record) != {"packages", "task_allow", "ttl_seconds"}:
                        raise ValueError("Unexpected lease fields")
                    ttl = record["ttl_seconds"]
                    if type(ttl) not in (int, float) or not 0 < ttl <= 86400:
                        raise ValueError("Invalid lease lifetime")
                    packages = record["packages"]
                    if set(packages) != {"npm", "pypi", "crates"} or any(
                        not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                        for values in packages.values()
                    ):
                        raise ValueError("Invalid package names")
                    extra = tuple(validate_task_allow(record["task_allow"]))
                    previous = self.server.leases.get(token)
                    if previous and previous.record != record:
                        self.reply(409, b"Lease policy is immutable")
                        return True
                    if previous is None:
                        self.server.leases[token] = Lease(
                            DenyPackages.from_dict(packages), extra, time.monotonic() + ttl, record
                        )
                else:
                    self.reply(405)
                    return True
            self.reply(204)
        except (ValueError, TypeError, KeyError):
            self.reply(400, b"Invalid lease")
        return True

    def do_CONNECT(self):
        lease = self.authorized_lease()
        if lease is None:
            self.reply(407)
            return
        try:
            target = urlsplit("//" + self.path)
            host = (target.hostname or "").lower().rstrip(".")
            if (
                target.username
                or target.password
                or target.path
                or target.query
                or target.fragment
                or target.port != 443
            ):
                raise ValueError("Only HTTPS CONNECT is supported")
            if not allowed_host(host, lease.task_allow):
                self.reply(403, b"Host denied")
                return
            if host in REGISTRY_HOSTS:
                self.send_response(200, "Connection Established")
                self.end_headers()
                self.wfile.flush()
                self.connection = self.server.ca.contexts[host].wrap_socket(self.connection, server_side=True)
                self.rfile = self.connection.makefile("rb")
                self.wfile = self.connection.makefile("wb")
                self.registry_host = host
                self.registry_lease = lease
                # One inspected request per connection. Never blindly tunnel a
                # second request after validating the first one's package path.
                self.handle_one_request()
                self.close_connection = True
            else:
                with socket.create_connection((host, 443), timeout=30) as upstream:
                    self.send_response(200, "Connection Established")
                    self.end_headers()
                    self.wfile.flush()
                    while True:
                        ready, _, _ = select.select([self.connection, upstream], [], [], 60)
                        if not ready or time.monotonic() >= lease.expires:
                            break
                        for source in ready:
                            data = source.recv(65536)
                            if not data:
                                return
                            (upstream if source is self.connection else self.connection).sendall(data)
        except (OSError, ValueError):
            self.close_connection = True

    def forward(self):
        try:
            if hasattr(self, "registry_host"):
                host, port, scheme, path = self.registry_host, 443, "https", self.path
                lease = self.registry_lease
                if not path.startswith("/") or path.startswith("//"):
                    raise ValueError("Expected an origin-form registry request")
            else:
                if self.control():
                    return
                lease = self.authorized_lease()
                if lease is None:
                    self.reply(407)
                    return
                target = urlsplit(self.path)
                host, scheme = (target.hostname or "").lower().rstrip("."), target.scheme
                port = target.port or 80
                if scheme != "http" or port != 80 or target.username or target.password or target.fragment:
                    raise ValueError("Expected an absolute HTTP request")
                path = (target.path or "/") + ("?" + target.query if target.query else "")
            if time.monotonic() >= lease.expires or not allowed_host(host, lease.task_allow):
                self.reply(403, b"Host or lease denied")
                return
            reason = registry_decision(host, self.command, path, lease.packages)
            if reason:
                self.reply(403, reason.encode())
                return
            body = self.body(8 * 1024 * 1024)
            connection_headers = {value.strip().lower() for value in self.headers.get("Connection", "").split(",")}
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_HEADERS | connection_headers | {"host", "authorization", "content-length"}
            }
            headers.update({"Host": host, "Connection": "close", "Content-Length": str(len(body))})
            cls = HTTPSConnection if scheme == "https" else HTTPConnection
            with closing(cls(host, port, timeout=60)) as upstream:
                upstream.request(self.command, path, body=body, headers=headers)
                response = upstream.getresponse()
                self.send_response(response.status)
                # Bun's registry client needs explicit response framing for
                # streamed manifests. Reframe decoded upstream chunks without
                # buffering potentially large package downloads in memory.
                chunked = (
                    self.command != "HEAD"
                    and response.status not in (204, 304)
                    and response.getheader("Content-Length") is None
                )
                for key, value in response.getheaders():
                    if key.lower() not in HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                if self.command != "HEAD":
                    while chunk := response.read(65536):
                        if chunked:
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        else:
                            self.wfile.write(chunk)
                    if chunked:
                        self.wfile.write(b"0\r\n\r\n")
            self.close_connection = True
        except (ValueError, http.client.HTTPException):
            self.reply(400)
        except OSError:
            self.close_connection = True

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_PATCH = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3128)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ["SWE_TOGETHER_DEPENDENCY_ADMIN_KEY"]
    with DependencyProxy((args.host, args.port), admin_key=key, ca=RegistryCA(args.state_dir)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
