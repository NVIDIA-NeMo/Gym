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
"""Optional TLS trust for an E2B backend that uses a PRIVATE CA or a SELF-SIGNED certificate (e.g. a
self-hosted, non-public E2B-compatible gateway). Public E2B needs none of this.

Default behavior: do nothing -- the system trust store is used. Opt in via env (nothing hardcoded):

  CYBERGYM_E2B_CA_CERT=/path/to/ca.pem   Trust this CA/cert bundle (PEM) for the E2B backend.
  CYBERGYM_E2B_TLS_PIN=1                  Pin the E2B_API_URL server's own cert as the sole trust
                                          anchor (for a self-signed backend). REQUIRES
                                          CYBERGYM_E2B_CERT_SHA256; the first run prints the
                                          fingerprint and stops so you can verify + set it.
  CYBERGYM_E2B_CERT_SHA256=<hex>          The expected cert SHA-256 (required with PIN).

Call install() before creating any sandbox or building any template. It trusts the chosen cert for the
E2B SDK's transports (control plane + the envd data plane) instead of disabling verification.
"""

import hashlib
import os
import ssl
from urllib.parse import urlsplit


def install():
    ca_path = os.environ.get("CYBERGYM_E2B_CA_CERT")
    pin = os.environ.get("CYBERGYM_E2B_TLS_PIN") == "1"
    if not ca_path and not pin:
        return  # default: system trust store (correct for public E2B)
    if ca_path and pin:
        raise SystemExit("set only ONE of CYBERGYM_E2B_CA_CERT or CYBERGYM_E2B_TLS_PIN, not both "
                         "(a requested pin must not be silently overridden by a CA bundle)")

    if ca_path:
        pem = open(ca_path, encoding="utf-8").read()
        src = f"CA bundle {ca_path}"
    else:
        url = os.environ.get("E2B_API_URL")
        if not url:
            raise SystemExit("CYBERGYM_E2B_TLS_PIN=1 but E2B_API_URL is not set")
        sp = urlsplit(url)
        host, port = sp.hostname, sp.port or 443  # honor a non-default port in the URL
        pem = ssl.get_server_certificate((host, port))
        fp = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()
        expected = os.environ.get("CYBERGYM_E2B_CERT_SHA256", "").lower().replace(":", "")
        # Require a known fingerprint: fetching + trusting whatever cert is presented on every run is
        # trust-on-first-use with no protection against a swapped cert. On the first run we print the
        # fingerprint and stop, so it can be verified out-of-band and pinned via the env var.
        if not expected:
            raise SystemExit(
                f"[e2b-tls] {host}:{port} presented cert sha256={fp}\n"
                f"Verify this out-of-band, then set CYBERGYM_E2B_CERT_SHA256={fp} to pin it.")
        if expected != fp:
            raise SystemExit(f"E2B cert fingerprint mismatch: got {fp}, expected {expected}")
        print(f"[e2b-tls] pinned {host}:{port} cert sha256={fp}")
        src = f"pinned cert from {host}:{port}"

    # Trust `pem` for the E2B SDK's pyqwest/httpx transports (both control-plane template/sandbox calls
    # and the envd data plane). Patch the transport classes so each new client is constructed with it.
    patched = 0
    for modname, clsname in (("e2b.api.client_async", "HTTPTransport"),
                             ("e2b.api.client_sync", "SyncHTTPTransport"),
                             ("e2b.template_async.build_api", "HTTPTransport")):
        try:
            mod = __import__(modname, fromlist=[clsname])
            orig = getattr(mod, clsname)
        except Exception:
            continue

        def make(*args, _orig=orig, **kwargs):
            kwargs["tls_ca_cert"] = pem.encode()
            kwargs["tls_include_system_certs"] = False
            return _orig(*args, **kwargs)

        setattr(mod, clsname, make)
        patched += 1
    if not patched:
        raise SystemExit("could not patch any E2B transport for custom TLS (SDK layout changed?)")
    print(f"[e2b-tls] trusting {src} for the E2B backend ({patched} transports)")
