# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools and grading remain usable independently, including after the tool process exits."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    exclude_domains_digest,
)
from resources_servers.browsecomp_advanced_harness.tools_sidecar import BrowsecompToolsServer


EXCLUSIONS = Path(__file__).with_name("dummy_exclude_domains_file.json")


def config(**kwargs):
    return TavilySearchResourcesServerConfig(
        name="tools",
        host="127.0.0.1",
        port=8080,
        entrypoint="tools_sidecar.py",
        tavily_api_key="test-key",
        exclude_domains_file_path=kwargs.pop("exclude_domains_file_path", str(EXCLUSIONS)),
        use_judge=False,
        **kwargs,
    )


def verification_body(metrics):
    return {
        "question": "What is the capital of France?",
        "ground_truth": "Paris",
        "responses_create_params": {"input": []},
        "response": {
            "id": "r",
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "m",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Answer: Paris\nConfidence: 100", "annotations": []}],
                }
            ],
            "browsecomp_tool_metrics": metrics,
        },
    }


def test_tool_surface_search_pages_shell_and_external_verification(tmp_path):
    outside_client = MagicMock(spec=ServerClient)
    tools = BrowsecompToolsServer(
        config=config(workspace="per_session", workspace_root=str(tmp_path), allow_host_execution=True),
        server_client=outside_client,
    )
    tools._async_tavily_clients[0].search = AsyncMock(
        return_value={
            "results": [
                {"url": "https://example.org/paris", "title": "Paris", "content": "Paris is the capital of France."}
            ]
        }
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("BROWSECOMP_TOOL_TOKEN", "unit-grant")
        app = tools.setup_webserver()
    with TestClient(app) as client:
        assert client.get("/_metrics").status_code == 403
        assert client.post("/search", json={"queries": ["test"]}).status_code == 403
        client.headers["Authorization"] = "Bearer unit-grant"
        assert not {"/verify", "/seed_session", "/openapi.json"}.intersection(r.path for r in client.app.routes)
        result = client.post("/search", json={"queries": ["France capital"]})
        assert result.status_code == 200, result.text
        result = client.post("/bash_command", json={"keystrokes": "cat pages/*"})
        assert result.status_code == 200 and "capital of France" in result.json()["results_string"]
        blocked = client.post("/bash_command", json={"keystrokes": "curl http://resources:9000/verify"})
        assert "[exit_code=-3]" in blocked.json()["results_string"]
        metrics = client.get("/_metrics").json()
        assert metrics["async_tavily_calls"][0]["function"] == "search"
        assert metrics["exclude_domains_sha256"] == exclude_domains_digest(tools._exclude_domains)
        assert tools._shell_sessions == {}, "The tools container must not create another sandbox"
        assert not outside_client.mock_calls
    # A fresh outside verifier consumes only the saved answer and metrics, with no tools service.
    resources = TavilySearchResourcesServer(config=config(use_tool_sidecar_metrics=True), server_client=outside_client)
    with TestClient(resources.setup_webserver()) as client:
        result = client.post("/verify", json=verification_body(metrics))
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1
        assert result.json()["metrics"] == {"async_tavily_calls": metrics["async_tavily_calls"]}


@pytest.mark.parametrize(
    "metrics,detail",
    [
        (None, "needs browsecomp_tool_metrics"),
        ({}, "needs browsecomp_tool_metrics"),
        ({"async_tavily_calls": "not-a-list"}, "Malformed browsecomp_tool_metrics"),
        ({"async_tavily_calls": []}, "different exclusion list"),
        ({"async_tavily_calls": [], "exclude_domains_sha256": exclude_domains_digest(["other.org"])}, "different"),
    ],
    ids=["missing", "empty", "malformed", "no-digest", "other-list"],
)
def test_sidecar_metrics_are_required_before_grading(monkeypatch, metrics, detail):
    grade = MagicMock()
    monkeypatch.setattr(TavilySearchResourcesServer, "_verify_answer_with_regex", grade)
    resources = TavilySearchResourcesServer(
        config=config(use_tool_sidecar_metrics=True), server_client=MagicMock(spec=ServerClient)
    )
    with TestClient(resources.setup_webserver()) as client:
        result = client.post("/verify", json=verification_body(metrics))
    assert result.status_code == 422 and detail in result.text
    grade.assert_not_called()


def _json_request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    with urlopen(
        Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": "Bearer process-test"}),
        timeout=3,
    ) as response:
        return json.load(response)


@pytest.mark.parametrize("container", [False, True], ids=["process", "image"])
def test_real_tools_process_stops_before_outside_verify(tmp_path, container):
    image = os.getenv("NEMO_GYM_TEST_TOOLS_IMAGE")
    if container and (not image or not shutil.which("docker")):
        pytest.skip("Set NEMO_GYM_TEST_TOOLS_IMAGE to exercise a built tools image")
    exclusions = os.environ.get("NEMO_GYM_TEST_TOOLS_EXCLUSIONS", str(EXCLUSIONS)) if container else str(EXCLUSIONS)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "BROWSECOMP_TOOL_TOKEN": "process-test",
        "BROWSECOMP_TOOL_SETTINGS": json.dumps(
            {
                "tavily_api_key": "test-key",
                "workspace": "per_session",
                "port": port,
                "exclude_domains_file_path": "/opt/gym/exclude-domains.json" if container else str(EXCLUSIONS),
                "workspace_root": "/tmp/browsecomp-pages" if container else str(tmp_path / "pages"),
            }
        ),
    }
    name = f"gym-tool-sidecar-test-{uuid4().hex[:12]}"
    argv = [sys.executable, "-m", "resources_servers.browsecomp_advanced_harness.tools_sidecar"]
    if container:
        inspected = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                image,
                "/opt/gym/.venv/bin/python",
                "-c",
                "from pathlib import Path; root=Path('/opt/gym/resources_servers'); "
                "assert not list(root.rglob('*.jsonl')); "
                "assert not list(root.rglob('configs')); assert not list(root.rglob('tests'))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert inspected.returncode == 0, inspected.stderr
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:{port}",
            "-e",
            "BROWSECOMP_TOOL_TOKEN",
            "-e",
            "BROWSECOMP_TOOL_SETTINGS",
            image,
            "/opt/gym/.venv/bin/python",
            "-m",
            "resources_servers.browsecomp_advanced_harness.tools_sidecar",
        ]
    with (tmp_path / "tools.log").open("w+") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    pytest.fail(log.read())
                try:
                    _json_request(f"http://127.0.0.1:{port}/health")
                    break
                except (URLError, OSError):
                    time.sleep(0.1)
            for route in ("verify", "seed_session", "aggregate_metrics", "openapi.json"):
                with pytest.raises(HTTPError) as error:
                    _json_request(f"http://127.0.0.1:{port}/{route}", {})
                assert error.value.code == 404
            output = _json_request(f"http://127.0.0.1:{port}/bash_command", {"keystrokes": "ls"})
            assert "[exit_code=0]" in output["results_string"]
            metrics = _json_request(f"http://127.0.0.1:{port}/_metrics")
        finally:
            if container:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        assert process.poll() is not None
    resources = TavilySearchResourcesServer(
        config=config(
            use_tool_sidecar_metrics=True,
            exclude_domains_file_path=exclusions,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    with TestClient(resources.setup_webserver()) as client:
        result = client.post("/verify", json=verification_body(metrics))
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1
