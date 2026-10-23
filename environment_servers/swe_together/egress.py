# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-side dependency access; task clients cannot change their policy."""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from environment_servers.swe_together.egress_policy import (
    parse_workspace_packages,
    task_allow_from_toml,
    workspace_packages_script,
)
from environment_servers.swe_together.runtime import PYTHON
from nemo_gym.server_utils import raise_for_status
from nemo_gym.server_utils import request as http_request


CA_PATH = "/tmp/gym-swet-dependency-ca.pem"
BUNDLE_PATH = "/tmp/gym-swet-dependency-bundle.pem"


class DependencyAccess:
    def __init__(self, url: str):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("dependency_proxy_url must be an HTTP origin on a dedicated trusted endpoint")
        self.url = url.rstrip("/")
        self.host = parsed.hostname

    async def certificate(self) -> str:
        async with await http_request("GET", self.url + "/ca") as response:
            await raise_for_status(response)
            certificate = await response.text()
        if "-----BEGIN CERTIFICATE-----" not in certificate:
            raise ValueError("Dependency proxy did not return its CA certificate")
        return certificate

    async def register(self, token: str, sandbox, *, workdir: str, task_dir: Path, artifact_dir: Path, ttl: float):
        result = await sandbox.exec(workspace_packages_script(workdir), cwd="/", timeout_s=120)
        if result.return_code or result.error_type or "SWT_PKGSCAN_DONE" not in (result.stdout or ""):
            raise RuntimeError("Could not freeze the task's package denylist before execution")
        policy = {
            "packages": parse_workspace_packages(result.stdout),
            "task_allow": task_allow_from_toml(task_dir),
            "ttl_seconds": ttl,
        }
        async with await http_request(
            "PUT", self.url + "/leases/" + token, headers=self._headers(), json=policy
        ) as response:
            await raise_for_status(response)
        (artifact_dir / "dependency-policy.json").write_text(json.dumps(policy, indent=2) + "\n")

    async def revoke(self, token: str):
        async with await http_request("DELETE", self.url + "/leases/" + token, headers=self._headers()) as response:
            await raise_for_status(response)

    def _headers(self):
        # Only the controller process sees this credential; never serialize it in
        # resolved Gym configuration, sandbox env, proxy URLs, or task artifacts.
        return {"Authorization": "Bearer " + os.environ["SWE_TOGETHER_DEPENDENCY_ADMIN_KEY"]}

    def sandbox_settings(self, token: str, policy: dict, certificate: str) -> tuple[dict, dict, dict]:
        from copy import deepcopy

        policy = deepcopy(policy)
        targets = [rule.get("target") for rule in policy.get("egress", []) if rule.get("action") == "allow"]
        if policy.get("defaultAction", policy.get("default_action")) != "deny" or not targets:
            raise ValueError("Dependency access requires default-deny egress and a role-scoped model endpoint")
        if self.host in targets:
            raise ValueError("Dependency proxy must have a dedicated endpoint separate from model routes")
        if any(not isinstance(target, str) or any(char in target for char in "*/") for target in targets):
            raise ValueError("Model egress targets must be exact hosts or IPs")
        policy["egress"].append({"action": "allow", "target": self.host})
        proxy = self.url.replace("http://", "http://swet:" + token + "@", 1)
        bypass = ",".join(["127.0.0.1", "localhost", *targets])
        env = {
            name: proxy
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
        }
        env.update(
            {
                "NO_PROXY": bypass,
                "no_proxy": bypass,
                "GOPROXY": "off",
                "GOFLAGS": "-mod=mod",
                "GONOSUMDB": "*",
                "GONOSUMCHECK": "1",
                "GOSUMDB": "off",
                "CARGO_NET_GIT_FETCH_WITH_CLI": "true",
                "NODE_EXTRA_CA_CERTS": CA_PATH,
            }
        )
        env.update(
            {
                name: BUNDLE_PATH
                for name in (
                    "SSL_CERT_FILE",
                    "REQUESTS_CA_BUNDLE",
                    "CURL_CA_BUNDLE",
                    "PIP_CERT",
                    "GIT_SSL_CAINFO",
                    "CARGO_HTTP_CAINFO",
                    "NIX_SSL_CERT_FILE",
                )
            }
        )
        return policy, env, {CA_PATH: certificate}


async def install_trust_bundle(sandbox):
    # Preserve the image's public roots and add only this service's public CA.
    # No private key or model/control-plane credential enters the sandbox.
    script = "\n".join(
        [
            "from pathlib import Path",
            "import ssl",
            "roots = next((Path(p) for p in ('/etc/ssl/certs/ca-certificates.crt', '/etc/pki/tls/certs/ca-bundle.crt') if Path(p).is_file()), None)",
            f"Path({BUNDLE_PATH!r}).write_bytes((roots.read_bytes() if roots else b'') + b'\\n' + Path({CA_PATH!r}).read_bytes())",
            f"ssl.create_default_context(cafile={BUNDLE_PATH!r})",
        ]
    )
    from shlex import quote

    result = await sandbox.exec(PYTHON + " -I -c " + quote(script), cwd="/", timeout_s=30)
    if result.return_code or result.error_type:
        raise RuntimeError("Could not install dependency proxy CA trust")
