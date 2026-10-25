# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym eval prepare <target>``: the preparation of a Harbor run on its own, and its reuse by a run."""

import argparse
import json
import sys
from pathlib import Path

import pytest

import nemo_gym.cli.main as cli_main
import nemo_gym.tasks.harbor.cli as harbor_cli
from nemo_gym.tasks.harbor import image_configs
from tests.unit_tests.test_harbor_image_configs import MAIN_IMAGE, FakeClient, write_task


class RegistryCalls:
    """Stands in for ``RegistryClient`` in the recorder: every construction is counted, and a ``FakeClient``
    answers the lookups so no network is touched."""

    def __init__(self):
        self.constructed = 0
        self.resolved: list[str] = []

    def __call__(self, *args, **kwargs) -> FakeClient:
        self.constructed += 1
        client = FakeClient()
        client.resolved = self.resolved
        return client


@pytest.fixture
def registry(monkeypatch) -> RegistryCalls:
    calls = RegistryCalls()
    monkeypatch.setattr(image_configs, "RegistryClient", calls)
    return calls


def write_dataset(root: Path) -> Path:
    folder = root / "ds"
    folder.mkdir()
    write_task(folder, "pull", image=None, dockerfile="FROM alpha:1\nWORKDIR /app\n")
    write_task(folder, "overlay", image=None, dockerfile="FROM beta:2\nRUN true\n")
    write_task(folder, "compose", compose="services:\n  main:\n  cache:\n    image: redis:7-alpine\n")
    return folder


def run_gym(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["gym", *argv])
    cli_main.main()


class TestEvalPrepareRouting:
    def test_a_target_routes_to_the_harbor_prepare_with_the_selection(self, monkeypatch, tmp_path):
        captured: list[tuple[str, str | None, str | None, list[str]]] = []
        monkeypatch.setattr(
            harbor_cli,
            "prepare_only",
            lambda args, overrides: captured.append((args.target, args.only_tasks, args.exclude_tasks, overrides)),
        )
        run_gym(monkeypatch, ["eval", "prepare", str(tmp_path), "--only-tasks", "a*,b", "+builds.enabled=false"])
        run_gym(
            monkeypatch, ["eval", "prepare", "harbor:terminal-bench/terminal-bench@4.0.0", "--exclude-tasks", "gpu-*"]
        )
        assert captured == [
            (str(tmp_path), "a*,b", None, ["+builds.enabled=false"]),
            ("harbor:terminal-bench/terminal-bench@4.0.0", None, "gpu-*", []),
        ]

    def test_a_bare_hydra_override_in_the_target_slot_stays_an_override(self, monkeypatch):
        dispatched: list[tuple[str, list[str]]] = []
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: dispatched.append((target, overrides)))
        run_gym(monkeypatch, ["eval", "prepare", "+use_cached_prepared_benchmarks=true", "--benchmark", "gsm8k"])
        ((target, overrides),) = dispatched
        assert target == "nemo_gym.cli.eval:prepare_benchmark"
        assert "+use_cached_prepared_benchmarks=true" in overrides
        assert any(token.startswith("+config_paths=[") and "gsm8k" in token for token in overrides)

    def test_a_selection_without_a_target_is_refused(self, monkeypatch, capsys):
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: pytest.fail("dispatched"))
        with pytest.raises(SystemExit):
            run_gym(monkeypatch, ["eval", "prepare", "--benchmark", "gsm8k", "--only-tasks", "a"])
        assert "--only-tasks and --exclude-tasks apply to a task TARGET" in capsys.readouterr().err


class TestEvalPrepareOfAFolder:
    def test_records_builds_nothing_writes_rows_and_a_second_run_asks_the_registry_for_nothing(
        self, monkeypatch, tmp_path, registry, capsys
    ):
        folder = write_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)
        run_gym(monkeypatch, ["eval", "prepare", str(folder), "+builds.enabled=false"])
        out = capsys.readouterr().out
        # The base images were recorded before the tasks loaded (one client), then the Compose images of the loaded tasks (another).
        recorded = json.loads((folder / "compose-images.json").read_text())
        assert sorted(recorded) == sorted(["alpha:1", "beta:2", "redis:7-alpine", MAIN_IMAGE])
        assert registry.constructed == 2 and registry.resolved[:2] == ["beta:2", "alpha:1"]
        rows_path = tmp_path / "results" / "harbor" / "ds" / "tasks.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        assert sorted(row["task_id"]["task_id"] for row in rows) == ["compose", "overlay", "pull"]
        assert json.loads((tmp_path / "results" / "harbor" / "ds" / "skipped.json").read_text()) == {}
        assert (tmp_path / "results" / "harbor" / "ds" / "images.json").is_file()
        assert (
            f"Base image configurations recorded in {folder / 'compose-images.json'} (2 resolved from the registry now"
            in out
        )
        assert "Compose image configurations recorded in" in out and "resolved from the registry now" in out
        assert "Prepared ds: 3 task(s) in results/harbor/ds/tasks.jsonl" in out
        assert "Next: gym eval run" in out and "no registry request" in out
        assert not (tmp_path / "results" / "harbor" / "ds" / "rollouts.jsonl").exists(), "no run happened"

        # The second invocation finds every image recorded: the registry client is not even constructed.
        registry.constructed = 0
        del registry.resolved[:]
        run_gym(monkeypatch, ["eval", "prepare", str(folder), "+builds.enabled=false"])
        out = capsys.readouterr().out
        assert registry.constructed == 0 and registry.resolved == []
        assert "Base image configurations recorded in" in out and "(all reused, no registry request)" in out
        assert sorted(json.loads((folder / "compose-images.json").read_text())) == sorted(recorded)

    def test_only_the_selected_tasks_images_are_recorded(self, monkeypatch, tmp_path, registry):
        folder = write_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)
        run_gym(monkeypatch, ["eval", "prepare", str(folder), "--only-tasks", "pull", "+builds.enabled=false"])
        assert sorted(json.loads((folder / "compose-images.json").read_text())) == ["alpha:1"]
        rows = (tmp_path / "results" / "harbor" / "ds" / "tasks.jsonl").read_text().splitlines()
        assert len(rows) == 1

    def test_a_target_that_is_not_a_dataset_is_a_usage_error_not_a_traceback(self, monkeypatch, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as info:
            run_gym(monkeypatch, ["eval", "prepare", str(tmp_path / "missing"), "+builds.enabled=false"])
        assert info.value.code == 2
        assert "missing" in capsys.readouterr().err

    def test_a_registry_failure_is_a_usage_error_naming_the_image(self, monkeypatch, tmp_path, capsys):
        folder = write_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)

        class RateLimited(FakeClient):
            def image_config(self, reference: str) -> dict:
                if reference == "alpha:1":
                    raise image_configs.ImageConfigError(
                        "library/alpha: registry returned HTTP 429 for manifests/1 (gave up after 8 attempts)"
                    )
                return super().image_config(reference)

        monkeypatch.setattr(image_configs, "RegistryClient", lambda *args, **kwargs: RateLimited())
        with pytest.raises(SystemExit):
            run_gym(monkeypatch, ["eval", "prepare", str(folder), "+builds.enabled=false"])
        err = capsys.readouterr().err
        assert "base image 'alpha:1' (used by task(s) pull)" in err and "HTTP 429" in err
        # The image resolved before the failure is on disk for the next attempt.
        assert sorted(json.loads((folder / "compose-images.json").read_text())) == ["beta:2"]


class TestARunAfterAPrepare:
    def test_gym_eval_run_reuses_the_complete_record_and_contacts_no_registry(self, monkeypatch, tmp_path, registry):
        folder = write_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)
        run_gym(monkeypatch, ["eval", "prepare", str(folder), "+builds.enabled=false"])
        assert registry.constructed == 2
        registry.constructed = 0

        # The run's own machinery after preparation is not under test: it is stubbed past the point where it
        # would start servers, and the preparation it does is the real one.
        dispatched: list[str] = []
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: dispatched.append(target))
        monkeypatch.setattr(harbor_cli, "build_run", lambda prepared, agent, **kwargs: (tmp_path / "run.yaml", []))
        monkeypatch.setattr(harbor_cli, "_resolved_collection", lambda tokens, prepared: object())
        monkeypatch.setattr(harbor_cli, "_resumable_cache", lambda collection: None)

        def fail(message):
            raise SystemExit(message)

        args = argparse.Namespace(
            target=str(folder), agent="hermes_agent", sandbox=None, _parser=argparse.Namespace(error=fail)
        )
        harbor_cli.run_target(args, ["+builds.enabled=false"])
        assert dispatched == ["nemo_gym.cli.eval:e2e_rollout_collection"]
        assert registry.constructed == 0, "a complete record means no registry client at all"
        # The run scopes its rows to the agent; the prepare's rows stay beside them.
        rows_dir = tmp_path / "results" / "harbor" / "ds"
        assert (rows_dir / "tasks.jsonl").is_file() and (rows_dir / "tasks_hermes_agent.jsonl").is_file()

    def test_prepare_target_itself_reuses_a_complete_record(self, monkeypatch, tmp_path, registry):
        """What ``gym eval run`` and ``gym dataset validate`` share: ``prepare_target`` on a prepared folder."""
        from nemo_gym.tasks.harbor.build_options import BuildOptions

        folder = write_dataset(tmp_path)
        harbor_cli.prepare_target(str(folder), output_root=tmp_path / "out", build=BuildOptions(enabled=False))
        assert registry.constructed == 2
        registry.constructed = 0
        prepared = harbor_cli.prepare_target(
            str(folder), output_root=tmp_path / "validation", build=BuildOptions(enabled=False), rows_name="x.jsonl"
        )
        assert registry.constructed == 0
        assert [task.task_id for task in prepared.tasks] == ["compose", "overlay", "pull"]
