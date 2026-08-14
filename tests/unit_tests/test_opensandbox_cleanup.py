# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers.opensandbox import cleanup_sandboxes


SCRIPT = Path(cleanup_sandboxes.__file__)
SBATCH_SCRIPT = Path("benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh")
TEST_ACCESS_KEY = "fixture-access-key"  # pragma: allowlist secret


class Response(io.BytesIO):
    def __init__(self, payload: object = "", status: int = 200) -> None:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        super().__init__(body.encode())
        self.status = status

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class RequestSequence:
    def __init__(self, *responses: object) -> None:
        self.responses = iter(responses)
        self.requests: list[tuple[urllib.request.Request, int]] = []

    def __call__(self, request: urllib.request.Request, timeout: int) -> Response:
        self.requests.append((request, timeout))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if not isinstance(response, Response):
            raise TypeError(f"invalid test response: {response!r}")
        return response


def sandbox(sandbox_id: str, *, run_id: str = "job-7", user: str = "alice") -> dict[str, object]:
    return {
        "id": sandbox_id,
        "metadata": {
            "nemo-gym.nvidia.com/run": run_id,
            "nemo-gym.nvidia.com/user": user,
        },
    }


def page(items: list[object], *, has_next_page: bool = False) -> Response:
    return Response({"items": items, "pagination": {"hasNextPage": has_next_page}})


def http_error(url: str, status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, status, "failure", {}, io.BytesIO())


def run_cleanup(
    *,
    domain: str = "https://sandbox.example",
    protocol: str = "http",
    run_id: str = "job-7",
    user: str = "alice",
    reap: bool = True,
) -> int:
    return cleanup_sandboxes.cleanup_sandboxes(
        domain=domain,
        protocol=protocol,
        access_key=TEST_ACCESS_KEY,
        run_id=run_id,
        user=user,
        reap=reap,
    )


def test_cleanup_paginates_then_deletes_only_exact_run_and_user(monkeypatch: pytest.MonkeyPatch) -> None:
    opener = RequestSequence(
        page(
            [
                sandbox("sandbox-a"),
                sandbox("wrong-run", run_id="job-8"),
                sandbox("wrong-user", user="bob"),
                {"id": "missing-metadata", "metadata": None},
            ],
            has_next_page=True,
        ),
        page([sandbox("sandbox/b")]),
        Response(status=204),
        Response(status=204),
    )
    monkeypatch.setattr(cleanup_sandboxes.urllib.request, "urlopen", opener)

    assert run_cleanup(domain="sandbox.example/", protocol="https") == 0
    requests = [request for request, _timeout in opener.requests]
    assert [request.get_method() for request in requests] == ["GET", "GET", "DELETE", "DELETE"]
    assert [request.full_url for request in requests] == [
        "https://sandbox.example/v1/sandboxes?page=1&pageSize=100",
        "https://sandbox.example/v1/sandboxes?page=2&pageSize=100",
        "https://sandbox.example/v1/sandboxes/sandbox-a",
        "https://sandbox.example/v1/sandboxes/sandbox%2Fb",
    ]
    assert all(timeout == cleanup_sandboxes.REQUEST_TIMEOUT_SECONDS for _request, timeout in opener.requests)
    assert all(dict(request.header_items())["Open-sandbox-api-key"] == TEST_ACCESS_KEY for request in requests)


def test_audit_does_not_delete_matches(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    opener = RequestSequence(page([sandbox("sandbox-a")]))
    monkeypatch.setattr(cleanup_sandboxes.urllib.request, "urlopen", opener)

    assert run_cleanup(reap=False) == 0
    assert len(opener.requests) == 1
    assert "Would delete 1 OpenSandbox sandbox" in capsys.readouterr().out


def test_cleanup_normalizes_scope_like_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    normalized_run = "run_7" + "x" * 58
    opener = RequestSequence(
        page([sandbox("sandbox-a", run_id=normalized_run, user="alice_team")]),
        Response(status=204),
    )
    monkeypatch.setattr(cleanup_sandboxes.urllib.request, "urlopen", opener)

    assert run_cleanup(run_id=f" run 7{'x' * 70} ", user="alice team") == 0
    assert [request.get_method() for request, _timeout in opener.requests] == ["GET", "DELETE"]


def test_delete_404_is_idempotent(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    opener = RequestSequence(
        page([sandbox("gone")]),
        http_error("https://sandbox.example/v1/sandboxes/gone", 404),
    )
    monkeypatch.setattr(cleanup_sandboxes.urllib.request, "urlopen", opener)

    assert run_cleanup() == 0
    assert "Sandbox gone was already gone" in capsys.readouterr().out


def test_delete_continues_after_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opener = RequestSequence(
        page([sandbox("failed"), sandbox("deleted"), sandbox("disconnected")]),
        http_error("https://sandbox.example/v1/sandboxes/failed", 500),
        Response(status=204),
        urllib.error.URLError("disconnected"),
    )
    monkeypatch.setattr(cleanup_sandboxes.urllib.request, "urlopen", opener)

    assert run_cleanup() == 1
    assert [request.get_method() for request, _timeout in opener.requests] == [
        "GET",
        "DELETE",
        "DELETE",
        "DELETE",
    ]
    output = capsys.readouterr()
    assert "Failed to delete failed -> HTTP 500" in output.err
    assert "Failed to delete disconnected" in output.err
    assert TEST_ACCESS_KEY not in output.out + output.err


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be an object"),
        ({"pagination": {"hasNextPage": False}}, "missing items or pagination"),
        ({"items": [], "pagination": {}}, "missing pagination.hasNextPage"),
        ({"items": [None], "pagination": {"hasNextPage": False}}, "invalid sandbox"),
        ({"items": [{"metadata": ["invalid"]}], "pagination": {"hasNextPage": False}}, "metadata must be an object"),
        (
            {"items": [{"metadata": sandbox("unused")["metadata"]}], "pagination": {"hasNextPage": False}},
            "without an id",
        ),
    ],
)
def test_cleanup_rejects_malformed_list_responses(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    monkeypatch.setattr(cleanup_sandboxes.urllib.request, "urlopen", RequestSequence(Response(payload)))

    with pytest.raises(ValueError, match=message):
        run_cleanup(reap=False)


def test_cleanup_rejects_invalid_domain() -> None:
    with pytest.raises(ValueError, match="invalid OpenSandbox domain"):
        run_cleanup(domain="file:///tmp/sandboxes", reap=False)


@pytest.mark.parametrize("missing", ["OPENSANDBOX_API_KEY", "NEMO_GYM_RUN_ID", "NEMO_GYM_USER"])
def test_cli_requires_credentials_and_exact_scope_before_network(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    monkeypatch.setenv("OPENSANDBOX_API_KEY", TEST_ACCESS_KEY)
    monkeypatch.setenv("NEMO_GYM_RUN_ID", "job-7")
    monkeypatch.setenv("NEMO_GYM_USER", "alice")
    monkeypatch.delenv(missing)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_USER", raising=False)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setattr(
        cleanup_sandboxes.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network request must not be made"),
    )

    with pytest.raises(SystemExit, match="2"):
        cleanup_sandboxes.main(["--domain", "sandbox.example"])


def test_cli_forwards_environment_and_return_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.example")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", TEST_ACCESS_KEY)
    monkeypatch.setenv("NEMO_GYM_RUN_ID", "job-7")
    monkeypatch.setenv("NEMO_GYM_USER", "alice")
    calls = []

    def record_cleanup(**kwargs: object) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cleanup_sandboxes, "cleanup_sandboxes", record_cleanup)
    assert cleanup_sandboxes.main(["--protocol", "https", "--reap"]) == 0
    assert calls == [
        {
            "domain": "sandbox.example",
            "protocol": "https",
            "access_key": TEST_ACCESS_KEY,
            "run_id": "job-7",
            "user": "alice",
            "reap": True,
        }
    ]

    monkeypatch.setattr(cleanup_sandboxes, "cleanup_sandboxes", lambda **_kwargs: 1)
    assert cleanup_sandboxes.main([]) == 1

    def raise_cleanup_error(**_kwargs: object) -> int:
        raise OSError("down")

    monkeypatch.setattr(cleanup_sandboxes, "cleanup_sandboxes", raise_cleanup_error)
    assert cleanup_sandboxes.main([]) == 1
    assert "OpenSandbox cleanup failed: down" in capsys.readouterr().err


def test_script_help_uses_only_standard_library() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_slurm_batch_command_wires_cleanup_epilogue() -> None:
    render = subprocess.run(
        [
            "bash",
            "-c",
            'sbatch() { printf "%s\\n" "$VLLM_PD_BATCH_COMMAND"; }; source "$1" --config benchmark.yaml',
            "bash",
            str(SBATCH_SCRIPT),
        ],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "NUM_PREFILL_NODES": "1",
            "NUM_DECODE_NODES": "1",
            "MODEL": "model",
            "CONTAINER": "container",
            "MOUNTS": "mounts",
            "VLLM_CONFIG": "config",
            "EXPERIMENT_NAME": "experiment",
            "USER": "test-user",
        },
        text=True,
    )
    assert render.returncode == 0, render.stderr
    assert "trap cleanup_job EXIT" in render.stdout
    assert "cleanup_server" in render.stdout
    assert "cleanup_sandboxes.py" in render.stdout
    assert '--run-id "$SLURM_JOB_ID"' in render.stdout
    assert '--user "${NEMO_GYM_USER:-$SLURM_JOB_USER}"' in render.stdout

    syntax = subprocess.run(["bash", "-n"], input=render.stdout, check=False, capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
