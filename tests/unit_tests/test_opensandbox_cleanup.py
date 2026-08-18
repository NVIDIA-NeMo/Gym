# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from nemo_gym.sandbox.providers.opensandbox import cleanup_sandboxes


SCRIPT = Path(cleanup_sandboxes.__file__)
SBATCH_SCRIPT = Path("benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh")
TEST_ACCESS_KEY = "fixture-access-key"  # pragma: allowlist secret


class Response:
    def __init__(
        self,
        payload: object = "",
        status: int = 200,
        *,
        enter: Any = None,
        exit: Any = None,
        error: BaseException | None = None,
    ) -> None:
        self.payload = payload
        self.status = status
        self.enter = enter
        self.exit = exit
        self.error = error

    async def __aenter__(self) -> "Response":
        if self.error:
            raise self.error
        if self.enter:
            await self.enter()
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self.exit:
            await self.exit()

    async def json(self, *, content_type: None) -> object:
        assert content_type is None
        return self.payload

    async def read(self) -> bytes:
        return b""


class Session:
    def __init__(
        self,
        *get_responses: Response,
        delete_responses: dict[str, Response] | None = None,
    ) -> None:
        self.get_responses = iter(get_responses)
        self.delete_responses = delete_responses or {}
        self.requests: list[tuple[str, str, dict[str, object]]] = []
        self.closed = False

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True

    def get(self, url: str, **kwargs: object) -> Response:
        self.requests.append(("GET", url, kwargs))
        return next(self.get_responses)

    def delete(self, url: str, **kwargs: object) -> Response:
        self.requests.append(("DELETE", url, kwargs))
        return self.delete_responses[url]


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


def install_session(
    monkeypatch: pytest.MonkeyPatch, session: Session
) -> tuple[list[dict[str, int]], list[dict[str, object]], object]:
    connector_calls: list[dict[str, int]] = []
    session_calls: list[dict[str, object]] = []
    connector = object()

    def make_connector(**kwargs: int) -> object:
        connector_calls.append(kwargs)
        return connector

    def make_session(**kwargs: object) -> Session:
        session_calls.append(kwargs)
        return session

    monkeypatch.setattr(cleanup_sandboxes.aiohttp, "TCPConnector", make_connector)
    monkeypatch.setattr(cleanup_sandboxes.aiohttp, "ClientSession", make_session)
    return connector_calls, session_calls, connector


def run_cleanup(
    *,
    domain: str = "https://sandbox.example",
    protocol: str = "http",
    run_id: str = "job-7",
    user: str = "alice",
    reap: bool = True,
) -> int:
    return asyncio.run(
        cleanup_sandboxes.cleanup_sandboxes(
            domain=domain,
            protocol=protocol,
            access_key=TEST_ACCESS_KEY,
            run_id=run_id,
            user=user,
            reap=reap,
        )
    )


def test_cleanup_uses_one_pool_and_deletes_only_exact_run_and_user(monkeypatch: pytest.MonkeyPatch) -> None:
    delete_responses = {
        "https://sandbox.example/v1/sandboxes/sandbox-a": Response(status=204),
        "https://sandbox.example/v1/sandboxes/sandbox%2Fb": Response(status=204),
    }
    session = Session(
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
        page([]),  # the confirming re-list after a successful sweep
        delete_responses=delete_responses,
    )
    connector_calls, session_calls, connector = install_session(monkeypatch, session)

    assert run_cleanup(domain="sandbox.example/", protocol="https") == 0
    assert session.closed
    assert connector_calls == [
        {
            "limit": cleanup_sandboxes.REAP_CONCURRENCY,
            "limit_per_host": cleanup_sandboxes.REAP_CONCURRENCY,
        }
    ]
    assert len(session_calls) == 1
    assert session_calls[0]["connector"] is connector
    assert session_calls[0]["headers"] == {"OPEN-SANDBOX-API-KEY": TEST_ACCESS_KEY}
    assert session_calls[0]["timeout"].total == cleanup_sandboxes.REQUEST_TIMEOUT_SECONDS
    assert session.requests[:2] == [
        (
            "GET",
            "https://sandbox.example/v1/sandboxes",
            {"allow_redirects": False, "params": {"page": 1, "pageSize": 100}},
        ),
        (
            "GET",
            "https://sandbox.example/v1/sandboxes",
            {"allow_redirects": False, "params": {"page": 2, "pageSize": 100}},
        ),
    ]
    assert {url for method, url, _kwargs in session.requests if method == "DELETE"} == set(delete_responses)
    assert all(kwargs == {"allow_redirects": False} for method, _url, kwargs in session.requests if method == "DELETE")


async def test_reap_limits_concurrent_deletes(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    maximum = 0
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def enter() -> None:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == cleanup_sandboxes.REAP_CONCURRENCY:
            all_started.set()
        await release.wait()

    async def exit() -> None:
        nonlocal active
        active -= 1

    total = cleanup_sandboxes.REAP_CONCURRENCY + 1
    matches = [sandbox(f"sandbox-{index}") for index in range(total)]
    delete_responses = {
        f"https://sandbox.example/v1/sandboxes/sandbox-{index}": Response(
            status=204,
            enter=enter,
            exit=exit,
        )
        for index in range(total)
    }
    install_session(monkeypatch, Session(page(matches), page([]), delete_responses=delete_responses))
    task = asyncio.create_task(
        cleanup_sandboxes.cleanup_sandboxes(
            domain="https://sandbox.example",
            protocol="http",
            access_key=TEST_ACCESS_KEY,
            run_id="job-7",
            user="alice",
            reap=True,
        )
    )
    try:
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert maximum == cleanup_sandboxes.REAP_CONCURRENCY
    finally:
        release.set()
        result = await task
    assert result == 0


def test_audit_does_not_delete_matches(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    session = Session(page([sandbox("sandbox-a")]))
    install_session(monkeypatch, session)

    assert run_cleanup(reap=False) == 0
    assert [method for method, _url, _kwargs in session.requests] == ["GET"]
    assert "Would delete 1 OpenSandbox sandbox" in capsys.readouterr().out


def test_cleanup_normalizes_scope_like_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    normalized_run = "run_7" + "x" * 58
    url = "https://sandbox.example/v1/sandboxes/sandbox-a"
    session = Session(
        page([sandbox("sandbox-a", run_id=normalized_run, user="alice_team")]),
        page([]),
        delete_responses={url: Response(status=204)},
    )
    install_session(monkeypatch, session)

    assert run_cleanup(run_id=f" run 7{'x' * 70} ", user="alice team") == 0
    assert [method for method, _url, _kwargs in session.requests] == ["GET", "DELETE", "GET"]


def test_delete_404_is_idempotent(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    url = "https://sandbox.example/v1/sandboxes/gone"
    session = Session(page([sandbox("gone")]), page([]), delete_responses={url: Response(status=404)})
    install_session(monkeypatch, session)

    assert run_cleanup() == 0
    assert "Sandbox gone was already gone" in capsys.readouterr().out


def test_redirects_are_not_followed(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    url = "https://sandbox.example/v1/sandboxes/redirected"
    session = Session(page([sandbox("redirected")]), delete_responses={url: Response(status=302)})
    install_session(monkeypatch, session)

    assert run_cleanup() == 1
    assert session.requests[-1] == ("DELETE", url, {"allow_redirects": False})
    assert "Failed to delete redirected -> HTTP 302" in capsys.readouterr().err

    session = Session(Response(status=302))
    install_session(monkeypatch, session)
    with pytest.raises(ValueError, match="list request failed -> HTTP 302"):
        run_cleanup(reap=False)
    assert session.requests == [
        (
            "GET",
            "https://sandbox.example/v1/sandboxes",
            {"allow_redirects": False, "params": {"page": 1, "pageSize": 100}},
        )
    ]


def test_delete_continues_after_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = "https://sandbox.example/v1/sandboxes"
    session = Session(
        page([sandbox("failed"), sandbox("deleted"), sandbox("disconnected")]),
        page([sandbox("failed"), sandbox("disconnected")]),  # survivors re-listed
        delete_responses={
            f"{base}/failed": Response(status=500),
            f"{base}/deleted": Response(status=204),
            f"{base}/disconnected": Response(error=aiohttp.ClientConnectionError("disconnected")),
        },
    )
    install_session(monkeypatch, session)

    assert run_cleanup() == 1
    # first sweep deletes all three; the retry sweep re-attempts the two failures
    assert [method for method, _url, _kwargs in session.requests].count("DELETE") == 5
    output = capsys.readouterr()
    assert "Failed to delete failed -> HTTP 500" in output.err
    assert "Failed to delete disconnected -> disconnected" in output.err
    assert TEST_ACCESS_KEY not in output.out + output.err


def test_reap_sweeps_catch_list_stragglers(monkeypatch: pytest.MonkeyPatch) -> None:
    # A list taken while the cancelled workload still mutates the set can skip
    # entries across page boundaries; the re-list sweep must catch them.
    base = "https://sandbox.example/v1/sandboxes"
    session = Session(
        page([sandbox("first")]),
        page([sandbox("straggler")]),
        page([]),
        delete_responses={
            f"{base}/first": Response(status=204),
            f"{base}/straggler": Response(status=204),
        },
    )
    install_session(monkeypatch, session)

    assert run_cleanup() == 0
    deletes = [url for method, url, _kwargs in session.requests if method == "DELETE"]
    assert deletes == [f"{base}/first", f"{base}/straggler"]


def test_reap_gives_up_after_bounded_sweeps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = "https://sandbox.example/v1/sandboxes"
    lists = [page([sandbox(f"s{index}")]) for index in range(cleanup_sandboxes.REAP_SWEEPS)]
    lists.append(page([sandbox("left-behind")]))
    session = Session(
        *lists,
        delete_responses={f"{base}/s{index}": Response(status=204) for index in range(cleanup_sandboxes.REAP_SWEEPS)},
    )
    install_session(monkeypatch, session)

    assert run_cleanup() == 1
    assert "1 OpenSandbox sandbox(es) were not reaped" in capsys.readouterr().err


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
    install_session(monkeypatch, Session(Response(payload)))

    with pytest.raises(ValueError, match=message):
        run_cleanup(reap=False)


def test_cleanup_rejects_invalid_domain() -> None:
    with pytest.raises(ValueError, match="invalid OpenSandbox domain"):
        run_cleanup(domain="file:///tmp/sandboxes", reap=False)


@pytest.mark.parametrize(
    "argv",
    [
        ["--run-id", "job-7", "--user", "alice"],
        ["--connection-config", "env.yaml", "--user", "alice"],
        ["--connection-config", "env.yaml", "--run-id", "job-7"],
        ["--connection-config", "env.yaml", "--run-id", "", "--user", "alice"],
        ["--connection-config", "env.yaml", "--run-id", "job-7", "--user", " "],
        ["--connection-config", "env.yaml", "--run-id", "job-7", "--user", "alice", "--unknown"],
    ],
)
def test_cli_requires_connection_config_and_exact_scope(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        cleanup_sandboxes.main(argv)


@pytest.mark.parametrize(("configured_protocol", "expected_protocol"), [(None, "http"), ("https", "https")])
def test_cli_uses_standalone_connection_config_and_forwards_return_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    configured_protocol: str | None,
    expected_protocol: str,
) -> None:
    calls = []
    config = tmp_path / "env.yaml"
    protocol = f"      protocol: {configured_protocol}\n" if configured_protocol else ""
    config.write_text(
        "decoy:\n"
        "  domain: wrong.example\n"
        "sandbox:\n"
        "  opensandbox:\n"
        "    connection:\n"
        "      domain: sandbox.example\n"
        f"      api_key: {TEST_ACCESS_KEY}\n"
        f"{protocol}"
    )
    argv = [
        "--connection-config",
        str(config),
        "--run-id",
        "job-7",
        "--user",
        "alice",
        "--reap",
    ]

    async def record_cleanup(**kwargs: object) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cleanup_sandboxes, "cleanup_sandboxes", record_cleanup)
    assert cleanup_sandboxes.main(argv) == 0
    assert calls == [
        {
            "domain": "sandbox.example",
            "protocol": expected_protocol,
            "access_key": TEST_ACCESS_KEY,
            "run_id": "job-7",
            "user": "alice",
            "reap": True,
        }
    ]

    async def failed_cleanup(**_kwargs: object) -> int:
        return 1

    monkeypatch.setattr(cleanup_sandboxes, "cleanup_sandboxes", failed_cleanup)
    assert cleanup_sandboxes.main(argv) == 1

    async def raise_cleanup_error(**_kwargs: object) -> int:
        raise OSError("down")

    monkeypatch.setattr(cleanup_sandboxes, "cleanup_sandboxes", raise_cleanup_error)
    assert cleanup_sandboxes.main(argv) == 1
    assert "OpenSandbox cleanup failed: down" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("[]\n", "must contain a YAML object"),
        ("other: {}\n", "config 'sandbox' is required"),
        ("sandbox:\n  docker: {}\n", "config 'sandbox.opensandbox' is required"),
        ("sandbox:\n  opensandbox: {}\n", "connection' is required"),
        (
            f"sandbox:\n  opensandbox:\n    connection:\n      api_key: {TEST_ACCESS_KEY}\n",
            "connection.domain' is required",
        ),
        (
            "sandbox:\n  opensandbox:\n    connection:\n      domain: sandbox.example\n",
            "connection.api_key' is required",
        ),
        (
            "sandbox:\n"
            "  opensandbox:\n"
            "    connection:\n"
            "      domain: sandbox.example\n"
            f"      api_key: {TEST_ACCESS_KEY}\n"
            "      protocol: ftp\n",
            "connection.protocol' must be http or https",
        ),
        (
            "sandbox:\n  opensandbox:\n    connection:\n      api_key: [fixture-access-key\n",
            "invalid YAML connection config",
        ),
    ],
)
def test_cli_rejects_invalid_connection_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    config: str,
    message: str,
) -> None:
    config_path = tmp_path / "env.yaml"
    config_path.write_text(config)

    async def fail_cleanup(**_kwargs: object) -> int:
        pytest.fail("network request must not be made")

    monkeypatch.setattr(cleanup_sandboxes, "cleanup_sandboxes", fail_cleanup)
    assert (
        cleanup_sandboxes.main(["--connection-config", str(config_path), "--run-id", "job-7", "--user", "alice"]) == 1
    )
    stderr = capsys.readouterr().err
    assert message in stderr
    assert TEST_ACCESS_KEY not in stderr


def test_script_help_runs_by_direct_path() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("first_step", "eval_status", "expected_status"),
    [("eval", 37, 37), ("eval", 143, 143), ("server", 0, 41)],
)
def test_slurm_batch_command_wires_cleanup_epilogue(
    tmp_path: Path, first_step: str, eval_status: int, expected_status: int
) -> None:
    render = subprocess.run(
        [
            "bash",
            "-c",
            'sbatch() { printf "%s\\n__BATCH_COMMAND__\\n%s\\n" "$eval_command" "$batch_command"; }; '
            'source "$1" "${@:2}"',
            "bash",
            str(SBATCH_SCRIPT),
            "--config",
            "benchmark.yaml",
            "--run-id",
            "attacker",
            "--user",
            "attacker",
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
    eval_command, batch_command = render.stdout.split("\n__BATCH_COMMAND__\n", maxsplit=1)

    assert 'export NEMO_GYM_RUN_ID="$SLURM_JOB_ID"' in eval_command
    assert 'export NEMO_GYM_USER="${NEMO_GYM_USER:-$SLURM_JOB_USER}"' in eval_command
    assert "cleanup_sandboxes.py" not in eval_command
    assert "trap cleanup_sandboxes" not in eval_command

    assert "trap cleanup_job EXIT" in batch_command
    assert batch_command.count("cleanup_sandboxes.py") == 1
    cleanup_function = batch_command[batch_command.index("cleanup_job()") : batch_command.index("trap cleanup_job")]
    assert cleanup_function.index("job_status=$?") < cleanup_function.index("trap - EXIT INT TERM")
    assert cleanup_function.index("trap - EXIT INT TERM") < cleanup_function.index("set +e")
    cleanup_command = 'python3 "$SLURM_SUBMIT_DIR/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py"'
    assert cleanup_function.index("set +e") < cleanup_function.index(cleanup_command)
    assert cleanup_function.index(cleanup_command) < cleanup_function.index('kill "$server_step"')
    assert cleanup_function.index('kill "$server_step"') < cleanup_function.index('exit "$job_status"')
    assert 'wait "$server_step"' not in cleanup_function
    assert '--connection-config "$SLURM_SUBMIT_DIR/env.yaml"' in cleanup_function
    assert '--run-id "$SLURM_JOB_ID"' in cleanup_function
    assert '--user "${NEMO_GYM_USER:-$SLURM_JOB_USER}"' in cleanup_function
    assert 'exit "$job_status"' in cleanup_function
    assert "attacker" not in batch_command

    for command in (eval_command, batch_command):
        syntax = subprocess.run(["bash", "-n"], input=command, check=False, capture_output=True, text=True)
        assert syntax.returncode == 0, syntax.stderr

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    events = tmp_path / "events"
    stubs = {
        "scontrol": "#!/bin/bash\nprintf 'node-a\\nnode-b\\n'\n",
        "python3": (
            "#!/bin/bash\n"
            'if [[ " $* " != *" --reap "* ]]; then\n'
            '    echo "unexpected $*" >> "$EVENTS"\n'
            "    exit 10\n"
            "fi\n"
            'echo "reap $*" >> "$EVENTS"\n'
            "exit 9\n"
        ),
        "srun": (
            "#!/bin/bash\n"
            'if [[ " $* " == *eval-container-on-node* ]]; then\n'
            '    while [[ ! -f "$SERVER_READY" ]]; do sleep 0.01; done\n'
            '    if [[ "$FIRST_STEP" == eval ]]; then\n'
            '        if [[ "$EVAL_STATUS" == 143 ]]; then kill -TERM "$PPID"; fi\n'
            '        exit "$EVAL_STATUS"\n'
            "    fi\n"
            "    trap 'exit 0' TERM\n"
            "    while :; do sleep 0.1; done\n"
            "fi\n"
            'touch "$SERVER_READY"\n'
            'if [[ "$FIRST_STEP" == server ]]; then exit "$SERVER_STATUS"; fi\n'
            "trap 'echo server-stop >> \"$EVENTS\"; exit 0' TERM\n"
            "while :; do sleep 0.1; done\n"
        ),
    }
    for name, contents in stubs.items():
        stub = stub_dir / name
        stub.write_text(contents)
        stub.chmod(0o755)

    result = subprocess.run(
        ["bash", "-c", batch_command],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{stub_dir}:{os.environ['PATH']}",
            "EVENTS": str(events),
            "EVAL_STATUS": str(eval_status),
            "FIRST_STEP": first_step,
            "NEMO_GYM_USER": "synthetic-user",
            "SERVER_STATUS": "41",
            "SERVER_READY": str(tmp_path / "server-ready"),
            "SLURM_CPUS_ON_NODE": "4",
            "SLURM_JOB_ID": "job-7",
            "SLURM_JOB_NODELIST": "nodes",
            "SLURM_JOB_USER": "slurm-user",
            "SLURM_SUBMIT_DIR": str(tmp_path),
            "eval_command": "eval-command",
            "vllm_command": "server-command",
        },
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_status
    assert "OpenSandbox cleanup failed with status 9" in result.stderr
    event_lines = events.read_text().splitlines()
    assert event_lines[0].startswith("reap ")
    assert "--run-id job-7 --user synthetic-user --reap" in event_lines[0]
    assert event_lines[1:] == (["server-stop"] if first_step == "eval" else [])
