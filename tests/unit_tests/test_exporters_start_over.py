# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A restart that starts its output over must leave the earlier job's tracking run as that job left it.

The exporters open through `setup_exporters`, which logs the config at once; collection decides only later whether the
reopened run continues or the output starts over (`start_over`)."""

from asyncio import Future
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import orjson
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig
from pytest import MonkeyPatch

import nemo_gym.artifacts.exporters.wandb as wandb_module
import nemo_gym.rollout_collection as collection_module
import nemo_gym.rollout_reverification as reverification_module
from nemo_gym.artifacts.exporters import export_rollouts, get_exporters, setup_exporters, teardown_exporters
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _CompletedRollout
from nemo_gym.rollout_reverification import OutputPaths


@pytest.fixture(autouse=True)
def _no_exporters_left_open():
    yield
    teardown_exporters()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: MonkeyPatch) -> DictConfig:
    """A run that resumes `tmp_path / "rollouts.jsonl"`, tracked in a real local MLflow file store."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return DictConfig(
        {
            "mlflow_tracking_uri": f"file://{tmp_path / 'mlruns'}",
            "mlflow_tracking_token": None,
            "mlflow_experiment_name": "gym",
            "mlflow_run_name": "run",
            "output_jsonl_fpath": str(tmp_path / "rollouts.jsonl"),
            "resume_from_cache": True,
        }
    )


def _rollout(task: int, reward: float) -> dict[str, Any]:
    return {"_ng_task_index": task, "_ng_rollout_index": 0, "reward": reward}


def _table(client: MlflowClient, run_id: str) -> list[dict[str, Any]]:
    table = orjson.loads(Path(client.download_artifacts(run_id, "rollouts.json")).read_bytes())
    return [orjson.loads(row[table["columns"].index("Rollout")]) for row in table["data"]]


def _snapshot(client: MlflowClient, run_id: str) -> dict[str, Any]:
    """Everything a later job could change on a run: artifacts with their bytes, params, tags, status, metrics."""
    root = Path(client.download_artifacts(run_id, ""))
    run = client.get_run(run_id)
    return {
        "artifacts": {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()},
        "params": dict(run.data.params),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        "status": run.info.status,
        "metrics": dict(run.data.metrics),
    }


def _first_job(store: DictConfig) -> tuple[MlflowClient, str]:
    """The earlier job: opened through the dispatcher, logs one rollout, closes."""
    (exporter,) = setup_exporters(store)
    export_rollouts([_rollout(0, 1.0)])
    client, run_id = exporter.client, exporter.run_id
    teardown_exporters()
    return client, run_id


def _marker(store: DictConfig) -> Path:
    output = Path(store["output_jsonl_fpath"])
    return output.with_name(f"{output.stem}_mlflow_run_id")


class TestMLflowStartOverThroughTheDispatcher:
    def test_starting_over_leaves_the_old_run_untouched_and_logs_the_config_once_to_the_new_one(
        self, store: DictConfig
    ) -> None:
        client, old_run = _first_job(store)
        before = _snapshot(client, old_run)

        store["num_repeats"] = 4  # the restarted job's config differs, as a later segment's does
        (exporter,) = setup_exporters(store)
        assert exporter.run_id == old_run
        for each in get_exporters():  # what collection does when it finds nothing to resume
            each.start_over()
        export_rollouts([_rollout(0, 0.0)])
        new_run = exporter.run_id
        teardown_exporters()

        assert new_run != old_run
        assert _snapshot(client, old_run) == before
        assert _marker(store).read_text() == new_run
        assert [row["reward"] for row in _table(client, new_run)] == [0.0]
        new_artifacts = sorted(_snapshot(client, new_run)["artifacts"])
        assert [name for name in new_artifacts if "config" in name] == ["global_config.json"]
        assert client.get_run(new_run).data.params["num_repeats"] == "4"

    @pytest.mark.parametrize("logs", ["rollouts", "nothing"])
    def test_a_continuing_run_gets_its_config_once(self, store: DictConfig, logs: str) -> None:
        client, old_run = _first_job(store)
        store["num_repeats"] = 4

        (exporter,) = setup_exporters(store)
        if logs == "rollouts":
            export_rollouts([_rollout(1, 1.0)])
        teardown_exporters()

        segments = [n for n in _snapshot(client, old_run)["artifacts"] if n.startswith("config/segment_")]
        assert len(segments) == 1
        assert client.get_run(old_run).data.params["num_repeats"] == "4"
        assert client.get_run(old_run).data.tags["gym_resume_count"] == "1"

    def test_a_failed_new_run_does_not_leave_the_next_resume_reopening_the_old_one(
        self, store: DictConfig, monkeypatch: MonkeyPatch, caplog
    ) -> None:
        client, old_run = _first_job(store)
        before = _snapshot(client, old_run)

        (exporter,) = setup_exporters(store)
        monkeypatch.setattr(MlflowClient, "create_run", MagicMock(side_effect=MlflowException("server down")))
        exporter.start_over()
        monkeypatch.undo()
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        teardown_exporters()

        assert "failed to log a new run" in caplog.text
        assert not _marker(store).exists()
        # The output was started over, so the next resume has rows the old run never saw: they must reach a new run.
        (resumed,) = setup_exporters(store)
        export_rollouts([_rollout(0, 0.0)])
        replacement = resumed.run_id
        teardown_exporters()
        assert replacement != old_run
        assert [row["reward"] for row in _table(client, replacement)] == [0.0]
        assert _snapshot(client, old_run) == before

    def test_a_marker_naming_another_run_is_left_alone(self, store: DictConfig, monkeypatch: MonkeyPatch) -> None:
        client, old_run = _first_job(store)
        (exporter,) = setup_exporters(store)
        _marker(store).write_text("another-run")  # rewritten by someone else since this job read it
        monkeypatch.setattr(MlflowClient, "create_run", MagicMock(side_effect=MlflowException("server down")))
        exporter.start_over()
        assert _marker(store).read_text() == "another-run"


@pytest.fixture
def wandb_store(tmp_path: Path, monkeypatch: MonkeyPatch) -> DictConfig:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return DictConfig(
        {
            "recording_enabled": True,
            "results_dir": str(tmp_path / "results"),
            "wandb_project": "proj",
            "wandb_name": "run",
            "wandb_api_key": "secret-key",  # pragma: allowlist secret
            "output_jsonl_fpath": str(tmp_path / "rollouts.jsonl"),
            "resume_from_cache": True,
        }
    )


class TestWandbStartOverThroughTheDispatcher:
    """W&B is mocked at `wandb.init`: these check the calls the exporter makes, not W&B itself."""

    def test_starting_over_updates_only_the_new_runs_config(
        self, wandb_store: DictConfig, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "rollouts_wandb_run_id").write_text("old-run")
        old, new = MagicMock(id="old-run"), MagicMock(id="new-run")
        init = MagicMock(side_effect=[old, new])
        monkeypatch.setattr(wandb_module.wandb, "init", init)

        setup_exporters(wandb_store)
        for each in get_exporters():
            each.start_over()
        teardown_exporters()

        reopen, fresh = init.call_args_list
        assert (reopen.kwargs["id"], "id" in fresh.kwargs) == ("old-run", False)
        old.config.update.assert_not_called()
        new.config.update.assert_called_once()
        assert (tmp_path / "rollouts_wandb_run_id").read_text() == "new-run"

    def test_a_continuing_run_updates_its_config_once(
        self, wandb_store: DictConfig, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "rollouts_wandb_run_id").write_text("old-run")
        old = MagicMock(id="old-run")
        monkeypatch.setattr(wandb_module.wandb, "init", MagicMock(return_value=old))

        setup_exporters(wandb_store)
        teardown_exporters()

        old.config.update.assert_called_once()
        old.finish.assert_called_once_with()

    @pytest.mark.parametrize("fails", ["finish", "init"])
    def test_a_failure_while_starting_over_never_reaches_the_old_run(
        self, wandb_store: DictConfig, monkeypatch: MonkeyPatch, tmp_path: Path, fails: str
    ) -> None:
        (tmp_path / "rollouts_wandb_run_id").write_text("old-run")
        old, new = MagicMock(id="old-run"), MagicMock(id="new-run")
        if fails == "finish":
            old.finish.side_effect = RuntimeError("service unavailable")
            init = MagicMock(side_effect=[old, new])
        else:
            init = MagicMock(side_effect=[old, RuntimeError("service unavailable")])
        monkeypatch.setattr(wandb_module.wandb, "init", init)

        (exporter,) = setup_exporters(wandb_store)
        exporter.start_over()
        export_rollouts([_rollout(0, 0.0)])
        teardown_exporters()

        old.config.update.assert_not_called()
        old.log.assert_not_called()
        if fails == "finish":
            assert (tmp_path / "rollouts_wandb_run_id").read_text() == "new-run"
            new.log.assert_called_once()
        else:
            assert not (tmp_path / "rollouts_wandb_run_id").exists()


class TestReverificationStartOver:
    """The reverification decision with the effective (prefixed) output paths and saved answers on disk."""

    @pytest.mark.parametrize(
        "on_disk, recovery, starts_over",
        [
            ([], {}, True),
            (["failures"], {}, False),
            (["output"], {}, False),
            ([], {"judge_failed_only": True, "append": True}, False),
        ],
        ids=["nothing to resume", "sidecar only", "output", "append recovery"],
    )
    def test_a_resume_starts_the_tracking_run_over_only_when_nothing_is_cached(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, on_disk: list, recovery: dict, starts_over: bool
    ) -> None:
        paths = OutputPaths(output=tmp_path / "unsafe_out.jsonl", failures=tmp_path / "unsafe_out_failures.jsonl")
        for name in on_disk:
            getattr(paths, name).write_text('{"_ng_task_index": 0, "_ng_rollout_index": 0}\n')
        before = {p: p.read_bytes() for p in (paths.output, paths.failures) if p.exists()}
        exporter = MagicMock()
        monkeypatch.setattr("nemo_gym.rollout_reverification.get_exporters", lambda: [exporter])
        config = reverification_module.RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath="in.jsonl",
            rollouts_jsonl_fpath="rollouts.jsonl",
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            resume_from_cache=True,
            **recovery,
        )

        reverification_module._start_over_without_cache(config, paths)

        assert exporter.start_over.called == starts_over
        assert {p: p.read_bytes() for p in (paths.output, paths.failures) if p.exists()} == before


class TestReverificationThroughTheHelper:
    """run_from_config with real payload preparation from saved answers on disk and a real local MLflow store.

    Faked: the resources-server mode check and the verify call, which asserts it receives the saved answer. Any other
    outbound request fails the test, so no policy model can be reached."""

    async def test_a_resume_with_no_output_tracks_the_fresh_rows_in_a_new_run(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, store: DictConfig
    ) -> None:
        answer = {
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "B"}]}]
        }
        inputs = tmp_path / "inputs.jsonl"
        inputs.write_bytes(
            orjson.dumps(
                {
                    "_ng_task_index": 0,
                    "_ng_rollout_index": 0,
                    "agent_ref": {"name": "agent"},
                    "responses_create_params": {"input": "Which letter?"},
                    "verifier_metadata": {"expected_answer": "B"},
                }
            )
            + b"\n"
        )
        saved = tmp_path / "rollouts.jsonl"
        saved.write_bytes(
            orjson.dumps({**_rollout(0, 1.0), "agent_ref": {"name": "agent"}, "response": answer}) + b"\n"
        )
        saved_bytes = {p: p.read_bytes() for p in (inputs, saved)}
        output = tmp_path / "reverified.jsonl"
        store["output_jsonl_fpath"] = str(output)
        client, old_run = _first_job(store)  # an earlier reverification into the same output
        before = _snapshot(client, old_run)
        assert not output.exists()  # its output is gone: nothing to resume

        received: list[dict] = []

        async def verify(payload: dict) -> tuple[dict, dict]:
            received.append(payload)
            return payload, _rollout(payload["_ng_task_index"], 0.0)

        def no_outbound_request(*_args, **_kwargs):
            raise AssertionError("reverification must not send any request here")

        monkeypatch.setattr("nemo_gym.client.server_client.request", no_outbound_request)
        monkeypatch.setattr(
            reverification_module, "_run_verification_payloads", lambda payloads, **_kw: [verify(p) for p in payloads]
        )
        monkeypatch.setattr(reverification_module, "_guard_reverify_mode", MagicMock(return_value=_none()))
        (exporter,) = setup_exporters(store)
        assert exporter.run_id == old_run

        await reverification_module.RolloutReverificationHelper().run_from_config(
            reverification_module.RolloutReverificationConfig(
                materialized_inputs_jsonl_fpath=str(inputs),
                rollouts_jsonl_fpath=str(saved),
                output_jsonl_fpath=str(output),
                resume_from_cache=True,
                upload_rollouts=True,
                disable_aggregation=True,
            )
        )
        new_run = exporter.run_id
        teardown_exporters()

        assert [payload["response"] for payload in received] == [answer]
        assert {p: p.read_bytes() for p in (inputs, saved)} == saved_bytes
        assert new_run != old_run and _marker(store).read_text() == new_run
        assert [row["reward"] for row in _table(client, new_run)] == [0.0]
        assert _snapshot(client, old_run) == before


class TestBestEffortLimits:
    def test_when_the_marker_cannot_be_removed_and_the_new_run_fails_nothing_reaches_the_old_run(
        self, store: DictConfig, monkeypatch: MonkeyPatch, caplog
    ) -> None:
        client, old_run = _first_job(store)
        before = _snapshot(client, old_run)
        (exporter,) = setup_exporters(store)

        def refuse(self, *args, **kwargs):
            raise PermissionError("read-only")

        monkeypatch.setattr(Path, "unlink", refuse)
        monkeypatch.setattr(MlflowClient, "create_run", MagicMock(side_effect=MlflowException("server down")))
        exporter.start_over()
        export_rollouts([_rollout(0, 0.0)])
        exporter.export_metrics({"reward": 0.0})
        monkeypatch.undo()
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        teardown_exporters()

        assert "the next resume reopens that run" in caplog.text
        assert _snapshot(client, old_run) == before
        # The documented limit: the marker still names the old run.
        assert _marker(store).read_text() == old_run

    def test_a_held_config_is_masked_when_the_run_continues(self, store: DictConfig) -> None:
        client, old_run = _first_job(store)
        store["policy_api_key"] = "held-secret-value"  # pragma: allowlist secret

        setup_exporters(store)
        teardown_exporters()

        artifacts = _snapshot(client, old_run)["artifacts"]
        (segment,) = [name for name in artifacts if name.startswith("config/segment_")]
        assert b"held-secret-value" not in artifacts[segment]
        assert client.get_run(old_run).data.params["policy_api_key"] != "held-secret-value"


async def _none() -> None:
    return None


class TestStringFlags:
    """The exporters read a string flag, as an `${oc.env:...}` interpolation yields, as the run config does."""

    @pytest.mark.parametrize("flag", ["false", "0", "no"])
    async def test_a_collection_that_starts_fresh_never_logs_to_the_earlier_run(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, store: DictConfig, flag: str
    ) -> None:
        client, old_run = _first_job(store)
        before = _snapshot(client, old_run)
        store["resume_from_cache"] = flag
        (exporter,) = setup_exporters(store)
        assert exporter.run_id != old_run and not exporter.reopened

        monkeypatch.setattr(collection_module, "get_global_config_dict", MagicMock(return_value={}))
        inputs = tmp_path / "input.jsonl"
        inputs.write_bytes(
            orjson.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}}) + b"\n"
        )

        class Helper(RolloutCollectionHelper):
            def _run_examples_with_metadata(self, examples, *args, **kwargs):
                futures = []
                for example in examples:
                    future = Future()
                    future.set_result(_CompletedRollout(row=example, result={"reward": 0.0}, rollout_latency_ms=None))
                    futures.append(future)
                return futures

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(inputs),
            output_jsonl_fpath=store["output_jsonl_fpath"],
            resume_from_cache=flag,
            disable_aggregation=True,
        )
        assert config.resume_from_cache is False
        await Helper().run_from_config(config)  # collection logs its rollout itself
        new_run = exporter.run_id
        teardown_exporters()

        assert new_run != old_run
        assert _snapshot(client, old_run) == before
        assert _marker(store).read_text() == new_run
        assert [row["reward"] for row in _table(client, new_run)] == [0.0]

    @pytest.mark.parametrize("flag", ["true", "1", "yes"])
    def test_a_string_true_flag_still_reopens_the_recorded_run(self, store: DictConfig, flag: str) -> None:
        _, old_run = _first_job(store)
        store["resume_from_cache"] = flag
        (exporter,) = setup_exporters(store)
        assert exporter.run_id == old_run and exporter.reopened
