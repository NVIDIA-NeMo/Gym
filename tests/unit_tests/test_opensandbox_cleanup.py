# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import subprocess
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


def test_reap_succeeds_when_final_sweep_removes_last_straggler(monkeypatch: pytest.MonkeyPatch) -> None:
    base = "https://sandbox.example/v1/sandboxes"
    sandbox_ids = [f"straggler-{index}" for index in range(cleanup_sandboxes.REAP_SWEEPS)]
    session = Session(
        *(page([sandbox(sandbox_id)]) for sandbox_id in sandbox_ids),
        page([]),
        delete_responses={f"{base}/{sandbox_id}": Response(status=204) for sandbox_id in sandbox_ids},
    )
    install_session(monkeypatch, session)

    assert run_cleanup() == 0
    deletes = [url for method, url, _kwargs in session.requests if method == "DELETE"]
    assert deletes == [f"{base}/{sandbox_id}" for sandbox_id in sandbox_ids]


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
        [str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def install_sbatch_stub(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    calls = tmp_path / "sbatch-calls"
    count = tmp_path / "sbatch-count"
    eval_capture = tmp_path / "eval-command"
    batch_capture = tmp_path / "batch-command"
    stub = stub_dir / "sbatch"
    stub.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "call_count=0\n"
        'if [[ -f "$SBATCH_COUNT" ]]; then read -r call_count < "$SBATCH_COUNT"; fi\n'
        "call_count=$((call_count + 1))\n"
        'printf "%s\\n" "$call_count" > "$SBATCH_COUNT"\n'
        "{\n"
        "    printf 'CALL\\0'\n"
        "    printf '%s\\0' \"$@\"\n"
        "    printf 'END\\0'\n"
        '} >> "$SBATCH_CALLS"\n'
        "if (( call_count == 1 )); then\n"
        '    printf "%s" "${eval_command:-}" > "$EVAL_CAPTURE"\n'
        '    printf "%s" "${batch_command:-}" > "$BATCH_CAPTURE"\n'
        "    printf '7001;hsg\\n'\n"
        "    exit 0\n"
        "fi\n"
        'if [[ "${FAIL_CLEANUP:-0}" == 1 ]]; then\n'
        "    echo 'simulated cleanup submission failure' >&2\n"
        "    exit 9\n"
        "fi\n"
        "printf '7002;hsg\\n'\n"
    )
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "SBATCH_CALLS": str(calls),
        "SBATCH_COUNT": str(count),
        "EVAL_CAPTURE": str(eval_capture),
        "BATCH_CAPTURE": str(batch_capture),
        "NUM_PREFILL_NODES": "1",
        "NUM_DECODE_NODES": "1",
        "MODEL": "model",
        "CONTAINER": "container",
        "MOUNTS": "mounts",
        "VLLM_CONFIG": "config",
        "EXPERIMENT_NAME": "experiment",
        "USER": "test-user",
        "NEMO_GYM_USER": "synthetic-user",
        "SBATCH_GRES": "gpu:4",
        "SBATCH_PARTITION": "batch",
        "SBATCH_QOS": "interactive",
    }
    return calls, env


def read_sbatch_calls(path: Path) -> list[list[str]]:
    calls: list[list[str]] = []
    current: list[str] | None = None
    for raw_token in path.read_bytes().split(b"\0"):
        if raw_token == b"CALL":
            current = []
        elif raw_token == b"END":
            assert current is not None
            calls.append(current)
            current = None
        elif raw_token:
            assert current is not None
            current.append(raw_token.decode())
    assert current is None
    return calls


def test_slurm_launcher_submits_one_dependent_cpu_cleanup_job(tmp_path: Path) -> None:
    calls_path, env = install_sbatch_stub(tmp_path)
    result = subprocess.run(
        ["bash", str(SBATCH_SCRIPT), "--config", "benchmark.yaml", "--run-id", "attacker"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "Submitted batch job 7001",
        "Submitted cleanup job 7002 for batch job 7001",
    ]
    main_call, cleanup_call = read_sbatch_calls(calls_path)
    assert "--parsable" in main_call

    submit_dir = str(Path.cwd().resolve())
    cleanup_script = f"{submit_dir}/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py"
    assert cleanup_call == [
        "--parsable",
        "--dependency=afterany:7001",
        "--partition=cpu",
        "--qos=cpu-short",
        "--gres=none",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=256M",
        "--time=00:30:00",
        "--job-name=gym-cleanup-7001",
        f"--output={submit_dir}/slurm-logs/%j-gym-cleanup-7001.log",
        cleanup_script,
        "--connection-config",
        f"{submit_dir}/env.yaml",
        "--run-id",
        "7001",
        "--user",
        "synthetic-user",
        "--reap",
    ]

    eval_command = (tmp_path / "eval-command").read_text()
    batch_command = (tmp_path / "batch-command").read_text()
    assert 'export NEMO_GYM_RUN_ID="$SLURM_JOB_ID"' in eval_command
    assert 'export NEMO_GYM_USER="${NEMO_GYM_USER:-$SLURM_JOB_USER}"' in eval_command
    assert "cleanup_sandboxes.py" not in eval_command
    assert "cleanup_sandboxes.py" not in batch_command
    assert "cleanup_server()" in batch_command
    assert "attacker" not in batch_command


def test_slurm_launcher_skips_cleanup_job_without_eval_args(tmp_path: Path) -> None:
    calls_path, env = install_sbatch_stub(tmp_path)
    result = subprocess.run(
        ["bash", str(SBATCH_SCRIPT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["Submitted batch job 7001"]
    assert len(read_sbatch_calls(calls_path)) == 1


def test_slurm_launcher_reports_cleanup_submission_failure(tmp_path: Path) -> None:
    calls_path, env = install_sbatch_stub(tmp_path)
    env["FAIL_CLEANUP"] = "1"
    result = subprocess.run(
        ["bash", str(SBATCH_SCRIPT), "--config", "benchmark.yaml"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Failed to submit cleanup job for batch job 7001; the batch job is still active" in result.stderr
    assert len(read_sbatch_calls(calls_path)) == 2


@pytest.mark.parametrize(
    ("first_step", "eval_status", "expected_status"),
    [("eval", 37, 37), ("eval", 143, 143), ("server", 0, 41)],
)
def test_slurm_batch_command_preserves_status_and_stops_server(
    tmp_path: Path, first_step: str, eval_status: int, expected_status: int
) -> None:
    bash_version = subprocess.run(
        ["bash", "-c", 'printf "%s %s" "${BASH_VERSINFO[0]}" "${BASH_VERSINFO[1]}"'],
        check=True,
        capture_output=True,
        text=True,
    )
    if tuple(map(int, bash_version.stdout.split())) < (5, 1):
        pytest.skip("the launcher uses wait -n -p from Bash 5.1")

    _calls_path, env = install_sbatch_stub(tmp_path)
    launch = subprocess.run(
        ["bash", str(SBATCH_SCRIPT), "--config", "benchmark.yaml"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert launch.returncode == 0, launch.stderr
    batch_command = (tmp_path / "batch-command").read_text()

    stub_dir = Path(env["PATH"].split(":", maxsplit=1)[0])
    events = tmp_path / "events"
    events.touch()
    stubs = {
        "scontrol": "#!/bin/bash\nprintf 'node-a\\nnode-b\\n'\n",
        "python3": '#!/bin/bash\necho unexpected-parent-python >> "$EVENTS"\nexit 10\n',
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
            'if [[ "$FIRST_STEP" == server ]]; then\n'
            '    echo server-exit >> "$EVENTS"\n'
            '    exit "$SERVER_STATUS"\n'
            "fi\n"
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
            **env,
            "EVENTS": str(events),
            "EVAL_STATUS": str(eval_status),
            "FIRST_STEP": first_step,
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
    assert events.read_text().splitlines() == (["server-stop"] if first_step == "eval" else ["server-exit"])
