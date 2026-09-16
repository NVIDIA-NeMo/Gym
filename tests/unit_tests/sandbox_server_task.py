# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a background HTTP server, then probe it from a later sandbox command."""

import subprocess
import sys
import time
import urllib.request


URL = "http://127.0.0.1:5000/"


def check_server() -> None:
    with urllib.request.urlopen(URL, timeout=3) as response:
        assert response.status == 200
        print(response.status)


def start_server() -> None:
    with open("/tmp/http.log", "wb") as log:
        subprocess.Popen(
            [sys.executable, "-m", "http.server", "5000", "--bind", "127.0.0.1"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
    for _ in range(100):
        try:
            check_server()
            return
        except OSError:
            # Establish readiness before exiting, so a later failure measures survival.
            time.sleep(0.1)
    raise RuntimeError("HTTP server did not become ready")


if __name__ == "__main__":
    if sys.argv[1] == "start":
        start_server()
    elif sys.argv[1] == "check":
        check_server()
    else:
        raise ValueError("Expected start or check")
