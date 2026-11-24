# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_opencode_archive_is_pinned_and_verified_before_extraction():
    dockerfile = (Path(__file__).resolve().parents[1] / "sandbox_image" / "Dockerfile").read_text()
    assert "ARG OPENCODE_VERSION=1.17.11" in dockerfile
    assert "ARG OPENCODE_SHA256=36064327d5a3fba662264f5726f517eafa26d1aefb0d59bc95e66bddccf3497b" in dockerfile
    start = dockerfile.index("ARG OPENCODE_VERSION=")
    download = dockerfile.index("-o /tmp/opencode.tar.gz", start)
    verify = dockerfile.index('echo "${OPENCODE_SHA256}  /tmp/opencode.tar.gz" | sha256sum -c -', download)
    extract = dockerfile.index("tar -xzf /tmp/opencode.tar.gz", verify)
    assert start < download < verify < extract
