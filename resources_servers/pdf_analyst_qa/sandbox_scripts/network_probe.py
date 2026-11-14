# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check model connectivity and blocked public network access inside a sandbox."""

import argparse
import socket
import urllib.parse


def check(model_url: str) -> None:
    """Require a reachable model server and blocked access to the public probe."""
    url = urllib.parse.urlsplit(model_url)
    if not url.hostname:
        raise RuntimeError("invalid model URL")
    with socket.create_connection((url.hostname, url.port or (443 if url.scheme == "https" else 80)), timeout=5):
        pass
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=2):
            raise RuntimeError("unrelated egress is allowed")
    except OSError:
        pass
    print("model reachable; unrelated egress blocked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_url")
    check(parser.parse_args().model_url)
