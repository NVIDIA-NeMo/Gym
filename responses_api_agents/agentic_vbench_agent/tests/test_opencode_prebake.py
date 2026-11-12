# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from responses_api_agents.agentic_vbench_agent.remote_environment import _with_opencode_base, opencode_base_tag


@pytest.mark.parametrize(
    ("dockerfile", "expected"),
    [
        (
            b"FROM localhost/avb-sdg-base:1\nCOPY materials/ /baked/\n",
            b"FROM localhost/avb-sdg-base-opencode:1\nCOPY materials/ /baked/\n",
        ),
        (
            b"FROM localhost/avb-sdg-repair-dns-denoise:1 AS build\n",
            b"FROM localhost/avb-sdg-repair-dns-denoise-opencode:1 AS build\n",
        ),
        # Registry images (repurpose and older assembly tasks) get a derivative too.
        (b"FROM python:3.11-slim\nRUN true\n", b"FROM localhost/python-3.11-slim-opencode:1\nRUN true\n"),
        (
            b"FROM --platform=linux/amd64 python:3.12-slim AS b\nFROM b\n",
            b"FROM --platform=linux/amd64 localhost/python-3.12-slim-opencode:1 AS b\nFROM b\n",
        ),
        # Already rewritten: idempotent.
        (b"FROM localhost/avb-sdg-base-opencode:1\n", b"FROM localhost/avb-sdg-base-opencode:1\n"),
        (b"FROM scratch\n", b"FROM scratch\n"),
        (b"FROM ${BASE}\n", b"FROM ${BASE}\n"),
    ],
)
def test_with_opencode_base(dockerfile: bytes, expected: bytes) -> None:
    assert _with_opencode_base(dockerfile) == expected


def test_opencode_base_tag() -> None:
    assert opencode_base_tag("localhost/avb-sdg-base:1") == "localhost/avb-sdg-base-opencode:1"
    assert opencode_base_tag("python:3.11-slim") == "localhost/python-3.11-slim-opencode:1"
    assert opencode_base_tag("build") is None
    assert opencode_base_tag("localhost/x-opencode:1") is None


def test_job_uses_prebaked_agent() -> None:
    from responses_api_agents.agentic_vbench_agent import harbor_runner

    src = open(harbor_runner.__file__).read()
    assert "opencode_agent:PrebakedOpenCode" in src
    assert '"name": "opencode"' not in src
