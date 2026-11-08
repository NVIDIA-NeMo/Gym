# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline contract tests for the control runner; these never contact a sandbox provider."""

import copy
import gzip
import json
import sys
from pathlib import Path

import pytest
from scripts import prepare_swe_verifier_controls as prepare
from scripts import run_swe_verifier_controls as runner

from tests.unit_tests.tooling.test_swe_rows_to_harbor import REBENCH_ROW, parser_cache  # noqa: F401


@pytest.fixture
def prepared(tmp_path, monkeypatch, parser_cache):
    row = copy.deepcopy(REBENCH_ROW)
    raw = (json.dumps(row) + "\n").encode()
    source, roster, pins = [tmp_path / name for name in ("source.gz", "roster.json", "pins.json")]
    source.write_bytes(gzip.compress(raw))
    roster.write_text(
        json.dumps(
            {
                "source_uncompressed_sha256": prepare.digest(raw),
                "tasks": [
                    {
                        "instance_id": row["instance_id"],
                        "source_line_1based": 1,
                        "payload_sha256": prepare.payload_digest(row),
                    }
                ],
            }
        )
    )
    pins.write_text(
        json.dumps(
            {
                "source_pool_sha256": prepare.digest(raw),
                "records": [
                    {
                        "instance_id": row["instance_id"],
                        "original_image": row["image_name"],
                        "status": "resolved",
                        "image": "docker.io/test/image@sha256:" + "a" * 64,
                        "config_digest": "sha256:" + "b" * 64,
                        "os": "linux",
                        "architecture": "amd64",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(prepare, "clean_revision", lambda root: "test-source")
    output = tmp_path / "prepared"
    manifest = prepare.prepare(source, roster, pins, parser_cache, output, count=1)
    monkeypatch.setattr(
        runner, "clean_revision", lambda root: "test-source" if root == runner.ROOT else manifest["parser_commit"]
    )
    return output


def test_preparer_retains_tests_and_records_every_fixture(prepared):
    manifest = runner.validate_prepared(prepared)
    gold, empty, timeout = [json.loads((prepared / case / "rows.jsonl").read_text()) for case in prepare.CASES]
    assert gold["image_name"].endswith("@sha256:" + "a" * 64)
    assert empty == gold | {"patch": ""}
    assert timeout == gold | {
        "install_config": gold["install_config"]
        | {
            "install": [],
            "test_cmd": [
                f"printf '{prepare.FORCED_TIMEOUT_MARKER}\\n'; sleep 60",
            ],
        }
    }
    assert manifest["tasks"][0]["original_payload_sha256"] == prepare.payload_digest(REBENCH_ROW)
    assert manifest["tasks"][0]["pinned_payload_sha256"] == prepare.payload_digest(gold)
    task = runner.task_folder_name("swe_rebench", gold["instance_id"])
    assert (prepared / "empty" / "harbor" / task / "solution" / "solve.sh").read_text().endswith("exit 0\n")
    assert [
        json.loads((prepared / case / "source-settings.json").read_text())["evaluation_timeout"]
        for case in prepare.CASES
    ] == [180, 180, 30]


@pytest.mark.parametrize("change", ["fixture", "source", "parser"])
def test_mutated_inputs_are_rejected_before_execution(prepared, monkeypatch, change):
    if change == "fixture":
        (prepared / "gold" / "rows.jsonl").write_text("{}\n")
    else:
        previous = runner.clean_revision
        monkeypatch.setattr(
            runner,
            "clean_revision",
            lambda path: "changed" if (path == runner.ROOT) == (change == "source") else previous(path),
        )
    with pytest.raises(ValueError):
        runner.validate_prepared(prepared)


@pytest.mark.parametrize("secret", ["1", 'quote"back\\slash'])
def test_redaction_preserves_json_and_numeric_token_evidence(tmp_path, monkeypatch, secret):
    monkeypatch.setenv("SWE_CONTROL_TEST_SECRET", secret)
    path = tmp_path / "receipt.json"
    runner.write_json(
        path, {"error": "SDK said " + secret, "return_code": 1, "token_ids": [1, 2], "logprobs": [-1.0], "ok": True}
    )
    assert json.loads(path.read_text()) == {
        "error": "SDK said [REDACTED]",
        "return_code": 1,
        "token_ids": [1, 2],
        "logprobs": [-1.0],
        "ok": True,
    }


def timeout_result(root, *, source=False):
    attempts = []
    for number in (1, 2):
        logs = root / f"attempt-{number}"
        logs.mkdir(parents=True)
        (logs / "test_output.log").write_text(prepare.FORCED_TIMEOUT_MARKER)
        (logs / ("verification.json" if source else "grading.json")).write_text(
            json.dumps(
                {
                    "error": (
                        "evaluation ended without complete test output (exit -1; cause unknown)"
                        if source
                        else "evaluation timed out"
                    ),
                    "sandbox_id": f"verifier-{number}",
                }
            )
        )
        (logs / "exec_result.json").write_text(
            json.dumps(
                {
                    "return_code": -1 if source else 124,
                    "error_type": None if source else "timeout",
                    "stderr": "signal: killed" if source else "",
                    "timeout_s": 30,
                    "exec_elapsed_s": 30.1,
                    "test_started": True,
                    "test_completed": False,
                }
            )
        )
        (logs / "verifier-cleanup.json").write_text(
            json.dumps({"sandbox_id": f"verifier-{number}", "teardown_confirmed": True})
        )
        attempts.append({"verifier_logs_dir": str(logs)})
    return {
        "reward": 0,
        "mask_sample": True,
        "evaluation_completed": False,
        "verification_attempts": 2 if source else attempts,
    }


@pytest.mark.parametrize("source", [True, False])
@pytest.mark.parametrize(
    "fault",
    [None, "first-marker-missing", "wrong-cause", "one-attempt", "measured-zero", "missing-exec", "same-sandbox"],
)
def test_timeout_requires_both_real_test_attempts(tmp_path, source, fault):
    result = timeout_result(tmp_path, source=source)
    if fault == "first-marker-missing":
        (tmp_path / "attempt-1" / "test_output.log").write_text("setup never reached test command")
    elif fault == "wrong-cause":
        (tmp_path / "attempt-1" / ("verification.json" if source else "grading.json")).write_text(
            '{"error":"provider denied"}'
        )
    elif fault == "one-attempt":
        result["verification_attempts"] = 1 if source else result["verification_attempts"][:1]
    elif fault == "measured-zero":
        result["evaluation_completed"] = True
    elif fault == "missing-exec":
        (tmp_path / "attempt-1" / "exec_result.json").unlink()
    elif fault == "same-sandbox":
        path = tmp_path / "attempt-2" / ("verification.json" if source else "verifier-cleanup.json")
        value = json.loads(path.read_text())
        value["sandbox_id"] = "verifier-1"
        path.write_text(json.dumps(value))
    assessment = runner.assess_control(result, "timeout", source_logs=tmp_path if source else None)
    assert assessment["passed"] == (fault is None)
    if fault is None:
        assert all(
            item["deadline_cause"] == ("unavailable" if source else "confirmed")
            and item["observed_classification"] == ("interrupted_test_output" if source else "typed_timeout")
            for item in assessment["attempt_evidence"]
        )


@pytest.mark.parametrize("route", ["flat", "typed"])
def test_source_runner_uses_real_http_routes_with_offline_sandboxes(tmp_path, monkeypatch, route):
    from resources_servers.swe_rebench.app import SWERebenchResourcesServer
    from resources_servers.swe_rebench.tests.test_sessions import _VERIFY, _server

    server, sandboxes = _server(monkeypatch)
    original = SWERebenchResourcesServer._create_sandbox

    async def create(*args, **kwargs):
        sandbox = await original(*args, **kwargs)
        sandbox.teardown_confirmed = True
        sandbox.cleanup_errors = []
        return sandbox

    monkeypatch.setattr(SWERebenchResourcesServer, "_create_sandbox", create)
    case = tmp_path / "empty"
    case.mkdir()
    (case / "rows.jsonl").write_text(json.dumps(_VERIFY | {"patch": ""}) + "\n")
    settings = server.config.model_dump(mode="json")
    (case / "source-settings.json").write_text(json.dumps(settings))
    provider = tmp_path / "provider.yaml"
    provider.write_text("sandbox:\n  docker: {}\n")
    output = tmp_path / "output"
    output.mkdir()
    runner.source_control(case, provider, route, 0, output)
    receipt = json.loads((output / "result.json").read_text())
    assert "error" not in receipt, receipt
    assert "close_error" not in receipt, receipt
    assert [call["path"] for call in receipt["calls"]] == ["/seed_session", "/verify", "/close_session"]
    assert receipt["cleanup"]["owned_task_sandboxes"] == receipt["cleanup"]["owned_verifier_sandboxes"] == 0
    assert len(sandboxes) == 2
    assert all(sandbox.stop.await_count == 1 for sandbox in sandboxes)
    assert (
        json.loads((output / "verifier" / _VERIFY["instance_id"] / "attempt-1" / "verification.json").read_text())[
            "completed"
        ]
        is True
    )


@pytest.mark.parametrize("fault", [None, "duplicate", "extra", "missing", "cleanup", "outer-timeout", "spawn-failure"])
def test_full_matrix_keeps_denominator_and_requires_cleanup(prepared, tmp_path, monkeypatch, fault):
    commands = []
    manifest = runner.validate_prepared(prepared)
    instance = manifest["tasks"][0]["instance_id"]
    task = runner.task_folder_name("swe_rebench", instance)

    def run(command, output, **kwargs):
        commands.append(command)
        output.mkdir(parents=True)
        source = "--source-case" in command
        case = Path(command[command.index("--source-case") + 1]).name if source else output.name
        if fault == "spawn-failure":
            raise OSError("offline simulated subprocess failure")
        root = output / "verifier" / instance if source else output / "logs"
        result = (
            timeout_result(root, source=source)
            if case == "timeout"
            else {
                "reward": 1 if case == "gold" else 0,
                "mask_sample": False,
                "evaluation_completed": True,
                "verification_attempts": 1 if source else [{"verifier_logs_dir": str(root)}],
            }
        )
        if source:
            runner.write_json(
                output / "result.json",
                {
                    "result": result,
                    "cleanup": {
                        "task_teardown_confirmed": True,
                        "owned_task_sandboxes": 0,
                        "owned_verifier_sandboxes": 0,
                    },
                },
            )
            return 0
        for number, attempt in enumerate(result["verification_attempts"], 1):
            for role in ("agent", "verifier"):
                runner.write_json(
                    Path(attempt["verifier_logs_dir"]) / f"{role}-cleanup.json",
                    {"teardown_confirmed": fault != "cleanup", "sandbox_id": f"{role}-{number}"},
                )
        rows = [result | {"_ng_task_id": {"taskset": "controls", "task_id": task}}]
        if fault == "duplicate":
            rows *= 2
        elif fault == "extra":
            rows.append(result | {"task_id": {"task_id": "unexpected"}})
        elif fault == "missing":
            rows = []
        (output / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return 124 if fault == "outer-timeout" else (0 if case == "gold" else 1)

    monkeypatch.setattr(runner, "run_command", run)
    provider = tmp_path / "provider.yaml"
    provider.write_text("sandbox:\n  opensandbox: {}\n")
    summary = runner.run_controls(prepared, provider, tmp_path / "runs", concurrency=2)
    assert summary["expected_controls"] == len(summary["controls"]) == 9
    assert (summary["verdict"] == "PASS_WITH_UNCONFIRMED_DEADLINE_CAUSE") == (fault is None)
    if fault is None:
        assert summary["unconfirmed_deadline_causes"] == 4
    assert len(commands) == 9
    harbor_commands = [command for command in commands if "--source-case" not in command]
    assert all(command[1:3] == ["dataset", "validate"] for command in harbor_commands)
    assert all(
        "++harbor_environment_server.environment_servers.harbor.inconclusive_verification_retries=1" in command
        for command in harbor_commands
    )


def test_bounded_subprocess_preserves_return_code_and_output(tmp_path):
    code = runner.run_command(
        [sys.executable, "-c", "print('control output'); raise SystemExit(3)"],
        tmp_path / "run",
        run_id="unit",
        timeout_s=5,
    )
    assert code == 3
    assert (tmp_path / "run" / "stdout.log").read_text() == "control output\n"
    assert json.loads((tmp_path / "run" / "process.json").read_text())["return_code"] == 3


def test_harbor_cli_resolves_matched_settings_without_starting_servers(prepared, tmp_path, monkeypatch):
    import nemo_gym.cli.main as cli
    import nemo_gym.config.global_config as global_config
    from environment_servers.harbor.tests.conftest import write_image_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "https://example.invalid")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "offline-test-placeholder")
    image = json.loads((prepared / "gold" / "rows.jsonl").read_text())["image_name"]
    write_image_config(prepared / "gold" / "harbor", image)
    target = tmp_path / "run"
    target.mkdir()
    provider = runner.ROOT / "examples/swe_verifier_controls/opensandbox.yaml"
    command = runner.harbor_command(prepared, provider, target, tmp_path / "envs", case="gold", concurrency=2)
    observed = []

    def collect(name, tokens):
        assert name == "nemo_gym.cli.eval:e2e_rollout_collection"
        monkeypatch.setattr(sys, "argv", ["test", *tokens])
        config = global_config.get_global_config_dict()
        observed.append(config)
        server_name = config.environment_server_routes["harbor"]
        settings = config[server_name].environment_servers.harbor
        assert settings.verifier_completion_key == "evaluation_completed"
        assert settings.inconclusive_verification_retries == 1
        assert settings.sandbox_provider_options.resource_requests.memory_mib == 4096
        assert settings.sandbox_provider_options.platform.arch == "amd64"
        assert config.num_samples_in_parallel == 2 and config.num_repeats == 1
        assert config.resume_from_cache is False and config.skip_venv_if_present is True
        assert config.sandbox.opensandbox.connection.domain == "https://example.invalid"
        assert config.sandbox.opensandbox.connection.transport_backend == "aiohttp"
        assert Path(config.output_jsonl_fpath) == target / "rollouts.jsonl"
        assert Path(config.results_dir) == target / "results"

    monkeypatch.setattr(cli, "dispatch", collect)
    monkeypatch.setattr(sys, "argv", command)
    # No result was produced: validation must fail, after the normal config has been resolved.
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert len(observed) == 1
