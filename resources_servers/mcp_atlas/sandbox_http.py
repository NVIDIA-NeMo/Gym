# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One HTTP request to the MCP server inside its sandbox; uses only the standard library."""

from __future__ import annotations

import http.client
import json
import math
import os


REQUEST_ENV = "MCP_ATLAS_HTTP_REQUEST"
PORT = 1984
ALLOWED_ROUTES = {("GET", "/health"), ("POST", "/list-tools"), ("POST", "/call-tool")}


def main(*, port: int = PORT) -> None:
    """Print a response envelope. The executable always uses sandbox loopback port 1984.

    ``port`` lets local subprocess tests supply an ephemeral HTTP server without changing the request contract.
    Neither request data nor process environment can override the HTTP destination.
    """
    try:
        request = json.loads(os.environ[REQUEST_ENV])
        if not isinstance(request, dict) or set(request) != {"method", "path", "timeout_s", "json"}:
            raise ValueError("invalid request fields")
        method, path, timeout = request["method"], request["path"], request["timeout_s"]
        if not isinstance(method, str) or not isinstance(path, str) or (method, path) not in ALLOWED_ROUTES:
            raise ValueError("unsupported method or path")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number")
        body = None if request["json"] is None else json.dumps(request["json"], allow_nan=False).encode("utf-8")
    except (KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"error": "invalid_request", "message": str(exc)}))
        return

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        text = response.read().decode("utf-8", errors="replace")
        try:
            result = json.loads(text)
        except ValueError:
            result = text
        envelope = {"status": response.status, "body": result}
    except TimeoutError:
        envelope = {"error": "timeout", "message": "sandbox HTTP request timed out"}
    except (OSError, http.client.HTTPException) as exc:
        envelope = {"error": "transport", "message": type(exc).__name__}
    finally:
        connection.close()
    print(json.dumps(envelope, allow_nan=False))


if __name__ == "__main__":
    main()
