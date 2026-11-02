# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import http.client
import json
import shutil
import ssl
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiohttp import ClientSession

from environment_servers.swe_together import egress, egress_proxy
from environment_servers.swe_together.egress_policy import (
    DenyPackages,
    allowed_host,
    parse_workspace_packages,
    registry_decision,
    workspace_packages_script,
)


@pytest.mark.parametrize(
    ("host", "path", "allowed"),
    [
        ("registry.npmjs.org", "/@biomejs%2fbiome", True),
        ("registry.npmjs.org", "/vitest/-/vitest-3.0.0.tgz", True),
        ("registry.npmjs.org", "/-/v1/search?text=project", False),
        ("registry.npmjs.org", "/@example%2fproject", False),
        ("registry.npmjs.org", "/%40example%2fproject", False),
        ("registry.npmjs.org", "/%2d/v1/search?text=project", False),
        ("registry.yarnpkg.com", "/@example/project/-/project-1.0.tgz", False),
        ("pypi.org", "/simple/pytest/", True),
        ("pypi.org", "/pypi/own_package/json", False),
        ("pypi.org", "/pypi/%6fwn_package/json", False),
        ("pypi.org", "/search/?q=project", False),
        ("files.pythonhosted.org", "/packages/aa/bb/cc/own_package-1.0.whl", False),
        ("files.pythonhosted.org", "/packages/aa/bb/cc/pytest-8.0.whl", True),
        ("index.crates.io", "/se/rd/serde", True),
        ("index.crates.io", "/ow/n_/own_crate", False),
        ("crates.io", "/api/v1/crates?q=foo", False),
        ("static.crates.io", "/crates/own-crate/own-crate-1.0.crate", False),
    ],
)
def test_reference_package_boundaries(host, path, allowed):
    deny = DenyPackages(npm=["@example/project"], pypi=["own-package"], crates=["own_crate"])
    assert (registry_decision(host, "GET", path, deny) is None) == allowed


@pytest.mark.parametrize(
    ("filename", "project", "allowed"),
    [
        ("own-package-1.0.tar.gz", "own-package", False),
        ("own_package-1.0.zip", "own-package", False),
        ("Own.Package-1.0.tar.bz2", "own-package", False),
        ("own-package-1.0.tar.xz", "own-package", False),
        ("own-package-1.0.tgz", "own-package", False),
        ("own_package-1.0.whl", "own-package", False),
        ("own_package-1.0-py3-none-any.whl", "own-package", False),
        ("own_package-1.0-1-py3-none-any.whl", "own-package", False),
        ("own_package-1.0-py3-none-any.whl.metadata", "own-package", False),
        ("own-package-1.0.tar.gz.metadata", "own-package", False),
        ("own-2-package-1.0.tar.gz", "own-2-package", False),
        ("own-2-package-1.0.tar.gz", "own", True),
        ("own-package-1.0-1.tar.gz", "own-package", False),
        ("own-package-1.0-rc1.tar.gz", "own-package", False),
        ("%6fwn%2dpackage-1.0.tar.gz", "own-package", False),
        ("pytest-8.0.whl", "own-package", True),
        ("pytest-8.0-py3-none-any.whl", "own-package", True),
        ("pytest-8.0-py3-none-any.whl.metadata", "own-package", True),
        ("pytest-8.0.tar.gz", "own-package", True),
        ("pytest-8.0.tar.gz.metadata", "own-package", True),
        ("own-package.tar.gz", "own-package", False),
        ("own_package-1.0-invalid.whl", "own-package", False),
    ],
)
def test_pypi_distribution_package_boundaries(filename, project, allowed):
    reason = registry_decision(
        "files.pythonhosted.org", "GET", "/packages/aa/bb/cc/" + filename, DenyPackages(pypi=[project])
    )
    assert (reason is None) == allowed


@pytest.mark.parametrize(
    "host",
    [
        "github.com",
        "raw.githubusercontent.com",
        "huggingface.co",
        "proxy.golang.org",
        "127.0.0.1",
        "api.anthropic.com",
        "example.org",
    ],
)
def test_dependency_proxy_does_not_open_other_routes(host):
    assert not allowed_host(host)


def test_reference_packages_are_scanned_before_execution(tmp_path):
    import subprocess

    (tmp_path / "package.json").write_text('{"name":"@example/project", "dependencies":{"name":"external"}}')
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "own_package"\n')
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules/package.json").write_text('{"name":"dependency"}')
    result = subprocess.run(
        ["sh", "-c", workspace_packages_script(str(tmp_path))], check=True, capture_output=True, text=True
    )
    assert "SWT_PKGSCAN_DONE" in result.stdout
    packages = parse_workspace_packages(result.stdout)
    assert "@example/project" in packages["npm"] and "dependency" not in packages["npm"]
    assert "own_package" in packages["pypi"]


@pytest.fixture(scope="module")
def ca(tmp_path_factory):
    if shutil.which("openssl") is None:
        pytest.skip("OpenSSL required for registry TLS test")
    return egress_proxy.RegistryCA(tmp_path_factory.mktemp("dependency-ca"))


@pytest.fixture
def proxy(ca, monkeypatch):
    requested = []

    class Origin(BaseHTTPRequestHandler):
        def do_GET(self):
            requested.append(self.path)
            body = b"package bytes"
            self.send_response(200)
            if self.path == "/chunked-manifest":
                body = gzip.compress(body * 10000)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for offset in range(0, len(body), 31):
                    chunk = body[offset : offset + 31]
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    origin_thread = threading.Thread(target=origin.serve_forever)
    origin_thread.start()
    real_https = http.client.HTTPSConnection
    # The client-to-proxy connection uses real TLS and CA validation. Only the
    # fixed external registry origin is replaced with a local package fixture.
    monkeypatch.setattr(
        egress_proxy,
        "HTTPSConnection",
        lambda host, port, **kwargs: http.client.HTTPConnection(*origin.server_address, **kwargs),
    )
    server = egress_proxy.DependencyProxy(("127.0.0.1", 0), admin_key="admin" * 16, ca=ca)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        yield server, requested, real_https
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
        origin.shutdown()
        origin.server_close()
        origin_thread.join()


def control(server, method="PUT", *, key=None, policy=None):
    body = json.dumps(
        policy
        or {"packages": {"npm": ["own-package"], "pypi": [], "crates": []}, "task_allow": [], "ttl_seconds": 600}
    )
    with closing(http.client.HTTPConnection(*server.server_address, timeout=10)) as conn:
        conn.request(
            method, "/leases/" + "a" * 64, body=body, headers={"Authorization": "Bearer " + (key or server.admin_key)}
        )
        response = conn.getresponse()
        response.read()
        return response.status


def fetch(proxy, path, *, token="a" * 64, host="registry.npmjs.org", headers=None):
    server, _, https = proxy
    context = ssl.create_default_context(cafile=str(server.ca.certificate))
    auth = "Basic " + base64.b64encode(("swet:" + token).encode()).decode()
    with closing(https(*server.server_address, context=context, timeout=10)) as conn:
        conn.set_tunnel(host, 443, headers={"Proxy-Authorization": auth})
        conn.request("GET", path)
        response = conn.getresponse()
        if headers is not None:
            headers.update(response.getheaders())
        return response.status, response.read()


def test_tls_allows_dependencies_and_denies_search_and_own_package(proxy):
    assert control(proxy[0]) == 204
    assert fetch(proxy, "/vitest")[0] == 200
    assert fetch(proxy, "/@biomejs%2fbiome") == (200, b"package bytes")
    assert fetch(proxy, "/-/v1/search?text=foo")[0] == 403
    assert fetch(proxy, "/own-package/-/own-package-2.tgz")[0] == 403
    assert fetch(proxy, "/%6fwn-package")[0] == 403
    assert proxy[1] == ["/vitest", "/@biomejs%2fbiome"]


def test_tls_denies_hyphenated_pypi_source_before_forwarding(proxy):
    policy = {"packages": {"npm": [], "pypi": ["own-package"], "crates": []}, "task_allow": [], "ttl_seconds": 600}
    assert control(proxy[0], policy=policy) == 204
    path = "/packages/aa/bb/cc/own-package-1.0.tar.gz"
    assert fetch(proxy, path, host="files.pythonhosted.org")[0] == 403
    assert not proxy[1]
    external = "/packages/aa/bb/cc/pytest-8.0.tar.gz"
    assert fetch(proxy, external, host="files.pythonhosted.org") == (200, b"package bytes")
    assert proxy[1] == [external]


def test_streamed_registry_manifest_has_explicit_framing(proxy):
    assert control(proxy[0]) == 204
    headers = {}
    status, body = fetch(proxy, "/chunked-manifest", headers=headers)
    assert status == 200
    assert headers["Transfer-Encoding"] == "chunked"
    assert headers["Content-Encoding"] == "gzip"
    assert "Content-Length" not in headers
    assert gzip.decompress(body) == b"package bytes" * 10000


def test_unknown_revoked_and_unauthorized_leases_fail_closed(proxy):
    with pytest.raises(OSError, match="407"):
        fetch(proxy, "/vitest")
    assert control(proxy[0], key="wrong") == 403
    assert control(proxy[0]) == 204
    assert control(proxy[0], "DELETE") == 204
    with pytest.raises(OSError, match="407"):
        fetch(proxy, "/vitest")
    assert not proxy[1]


def test_lease_cannot_be_relaxed_after_candidate_edits(proxy):
    assert control(proxy[0]) == 204
    changed = {"packages": {"npm": [], "pypi": [], "crates": []}, "task_allow": [], "ttl_seconds": 600}
    assert control(proxy[0], policy=changed) == 409
    assert fetch(proxy, "/own-package")[0] == 403


def test_proxy_rejects_nonreference_connect_hosts(proxy):
    assert control(proxy[0]) == 204
    with pytest.raises(OSError, match="403"):
        fetch(proxy, "/anything", host="github.com")
    assert not proxy[1]


def test_workload_only_receives_role_route_and_dependency_proxy():
    access = egress.DependencyAccess("http://10.0.0.3:3128")
    original = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "10.0.0.1"}]}
    policy, env, files = access.sandbox_settings("a" * 64, original, "public CA")
    assert policy["egress"] == [*original["egress"], {"action": "allow", "target": "10.0.0.3"}]
    assert len(original["egress"]) == 1
    assert env["NO_PROXY"] == "127.0.0.1,localhost,10.0.0.1"
    assert env["HTTPS_PROXY"] == "http://swet:" + "a" * 64 + "@10.0.0.3:3128"
    assert "registry.npmjs.org" not in json.dumps(policy)
    assert "ADMIN" not in json.dumps(env)
    assert files == {egress.CA_PATH: "public CA"}
    with pytest.raises(ValueError, match="dedicated"):
        egress.DependencyAccess("http://10.0.0.1:3128").sandbox_settings("a" * 64, original, "CA")


@pytest.mark.asyncio
async def test_failed_package_scan_does_not_register_a_policy(tmp_path, monkeypatch):
    request = AsyncMock()
    monkeypatch.setattr(egress, "http_request", request)
    from types import SimpleNamespace

    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="partial"))
    )
    with pytest.raises(RuntimeError, match="freeze"):
        await egress.DependencyAccess("http://10.0.0.3:3128").register(
            "a" * 64, sandbox, workdir="/workspace", task_dir=tmp_path, artifact_dir=tmp_path, ttl=600
        )
    request.assert_not_called()


@pytest_asyncio.fixture
async def controller_http_client(monkeypatch):
    async with ClientSession() as client:
        monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", lambda: client)
        yield client


@pytest.mark.asyncio
async def test_controller_registers_frozen_policy_and_revokes_it(proxy, tmp_path, monkeypatch, controller_http_client):
    from types import SimpleNamespace

    server = proxy[0]
    monkeypatch.setenv("SWE_TOGETHER_DEPENDENCY_ADMIN_KEY", server.admin_key)
    access = egress.DependencyAccess("http://%s:%d" % server.server_address)
    assert await access.certificate() == server.ca.certificate.read_text()
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, error_type=None, stdout="npm\town-package\nSWT_PKGSCAN_DONE\n")
        )
    )
    await access.register("a" * 64, sandbox, workdir="/workspace", task_dir=tmp_path, artifact_dir=tmp_path, ttl=600)
    record = json.loads((tmp_path / "dependency-policy.json").read_text())
    assert record["packages"]["npm"] == ["own-package"]
    assert server.admin_key not in json.dumps(record)
    assert fetch(proxy, "/vitest")[0] == 200
    assert fetch(proxy, "/own-package")[0] == 403
    await access.revoke("a" * 64)
    assert server.lease("a" * 64) is None
