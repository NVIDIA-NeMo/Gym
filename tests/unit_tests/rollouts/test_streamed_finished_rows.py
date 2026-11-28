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
"""Collection without retained results holds finished rows one at a time and writes the same artifacts."""

import shutil
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from omegaconf import OmegaConf

import nemo_gym.rollouts.rollout_collection as rollout_collection
import nemo_gym.rollouts.rollout_recovery as rollout_recovery
from nemo_gym.artifacts.rollout_rows import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY
from nemo_gym.config.global_config import AGENT_REF_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.rollouts.rollout_collection import (
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    RolloutStore,
    _aggregation_entry,
    _CompletedRollout,
)
from nemo_gym.rollouts.rollout_records import RolloutRecord


class _Reply:
    status = 200
    ok = True
    headers: dict = {}

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def read(self) -> bytes:
        return self._body

    def release(self) -> None:
        pass


def _install_aggregating_server(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Answer /aggregate_metrics as an environment server would: from the body the collector sends.

    Returns the request bodies, as the collector serialized them.
    """
    bodies: list[bytes] = []

    async def post(*, server_name: str, url_path: str, json, **kwargs):
        assert (server_name, url_path) == ("my_environment_server", "/aggregate_metrics")
        body = orjson.dumps(json.model_dump(exclude_unset=True))
        bodies.append(body)
        metrics = compute_aggregate_metrics(orjson.loads(body)["verify_responses"])
        return _Reply(orjson.dumps(metrics.model_dump()))

    client = SimpleNamespace(
        post=post,
        global_config_dict=OmegaConf.create(
            {
                "my_agent": {"responses_api_agents": {"impl": {}}},
                "my_environment_server": {
                    "environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}
                },
            }
        ),
    )
    monkeypatch.setattr(rollout_collection, "setup_server_client_utils", lambda *args, **kwargs: client)
    monkeypatch.setattr(rollout_collection, "get_global_config_dict", lambda: {})
    return bodies


def _result_for(row: dict) -> dict:
    """A /run result with `size` characters of generated content, built when the rollout completes."""
    task, size = row["task"], row["size"]
    response = {
        "id": f"resp_{task}",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"{task:04d}" * (size // 4)}],
            }
        ],
        "usage": {"input_tokens": 100 + task, "output_tokens": 7 * task, "total_tokens": 100 + 8 * task},
        "incomplete_details": {"reason": "max_output_tokens"} if task % 4 == 1 else None,
        "metrics": {"ttft_s": 0.25 * task},
    }
    result = {"response": response, "reward": float(task % 3 == 0), "judge_score": task / 10}
    if row["case"] == "counted failure":
        result[NG_FAILURE_CLASS_KEY] = "verify_failed"
        result["reward"] = 0.0
    elif row["case"] == "no persist":
        result = {NG_NO_PERSIST_KEY: True, "reward": 0.0}
    return result


class _ScriptedHelper(RolloutCollectionHelper):
    """Dispatch replaced by results built as each rollout is awaited, so no result exists before it completes."""

    def _run_examples_with_metadata(self, examples, *args, **kwargs):
        async def complete(row: dict) -> _CompletedRollout:
            return _CompletedRollout(row=row, result=_result_for(row), rollout_latency_ms=None)

        return (complete(row) for row in examples)


def _write_inputs(path: Path, cases: list[str], size: int) -> None:
    path.write_bytes(
        b"".join(
            orjson.dumps(
                {
                    "responses_create_params": {"input": [{"role": "user", "content": f"task {task}"}]},
                    AGENT_REF_KEY_NAME: {"name": "my_agent"},
                    "task": task,
                    "case": case,
                    "size": size,
                }
            )
            + b"\n"
            for task, case in enumerate(cases)
        )
    )


def _config(tmp_path: Path, output: Path, *, retain: bool) -> RolloutCollectionConfig:
    return RolloutCollectionConfig(
        input_jsonl_fpath=str(tmp_path / "input.jsonl"),
        output_jsonl_fpath=str(output),
        retain_results_in_memory=retain,
        route_failures_to_sidecar=True,
        count_failure_classes_as_zero=["verify_failed"],
        health_check_workers=1,
    )


_CASES = ["success"] * 6 + ["counted failure", "no persist", "success", "counted failure"]


async def test_non_retaining_collection_writes_what_retaining_collection_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every artifact, and the body sent to /aggregate_metrics, is byte-identical whether results are retained."""
    # One run id for both runs, so the rows they save can be compared byte for byte.
    monkeypatch.setattr(rollout_recovery, "uuid4", lambda: SimpleNamespace(hex="0" * 32))
    bodies = _install_aggregating_server(monkeypatch)
    _write_inputs(tmp_path / "input.jsonl", _CASES, size=4096)
    output = tmp_path / "run" / "rollouts.jsonl"

    artifacts = {}
    for retain in (True, False):
        shutil.rmtree(output.parent, ignore_errors=True)
        returned = await _ScriptedHelper().run_from_config(_config(tmp_path, output, retain=retain))
        assert len(returned) == (len(_CASES) if retain else 0)
        artifacts[retain] = {
            path.name: path.read_bytes()
            for path in sorted(output.parent.iterdir())
            if path.is_file() and path.name != "rollouts_run.lock"
        }

    # The status file stamps when it was written; nothing else in any artifact may differ.
    for saved in artifacts.values():
        status = orjson.loads(saved["rollouts_batch_status.json"])
        assert status.pop("updated_at")
        saved["rollouts_batch_status.json"] = orjson.dumps(status)
    assert artifacts[False] == artifacts[True]
    names = set(artifacts[False])
    for name in (
        "rollouts.jsonl",
        "rollouts_failures.jsonl",
        "rollouts_aggregate_metrics.json",
        "rollouts_coverage.json",
        "quality_summary.json",
        "rollout_verdicts.jsonl",
    ):
        assert name in names
    assert len(artifacts[False]["rollouts.jsonl"].splitlines()) == _CASES.count("success")
    # The scored rows and the counted failures reach the aggregation, without their generated content.
    retaining_body, non_retaining_body = bodies
    assert non_retaining_body == retaining_body
    sent = orjson.loads(non_retaining_body)["verify_responses"]
    assert len(sent) == _CASES.count("success") + _CASES.count("counted failure")
    assert all(set(entry.get("response", {})) <= {"usage", "incomplete_details", "metrics"} for entry in sent)
    assert [entry[TASK_INDEX_KEY_NAME] for entry in sent] == sorted(entry[TASK_INDEX_KEY_NAME] for entry in sent)


@pytest.mark.parametrize("retain", [False, True], ids=["finished rows dropped", "finished rows retained"])
async def test_collection_memory_follows_the_largest_row_without_retained_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retain: bool
) -> None:
    """Without retained results the collector holds a few rows at a time, whatever the run's size.

    Live traced memory is sampled as each outcome is saved, as each saved row is read back after collection, and as
    the aggregation starts. (Not the traced peak: orjson reserves several times a row's size while it serializes
    one, address space the process never touches.) The retaining run is the control: the same samples see it hold
    every row it saved.
    """
    _install_aggregating_server(monkeypatch)
    row_bytes, rows = 1 << 20, 40
    _write_inputs(tmp_path / "input.jsonl", ["success"] * (rows - 2) + ["counted failure"] * 2, size=row_bytes)
    config = _config(tmp_path, tmp_path / "run" / "rollouts.jsonl", retain=retain)

    samples: dict[str, int] = {}

    def sample(point: str) -> None:
        samples[point] = max(samples.get(point, 0), tracemalloc.get_traced_memory()[0] - before)

    record_outcome = RolloutStore.record_outcome

    def recording(store, result, **kwargs):
        sample("outcome saved")
        return record_outcome(store, result, **kwargs)

    read = RolloutRecord._read

    def reading(record, file):
        row = read(record, file)
        sample("row read back")
        return row

    class Helper(_ScriptedHelper):
        async def _call_aggregate_metrics(self, results, rows, output_fpath, **kwargs):
            sample("aggregation")
            return await super()._call_aggregate_metrics(results, rows, output_fpath, **kwargs)

    monkeypatch.setattr(RolloutStore, "record_outcome", recording)
    monkeypatch.setattr(RolloutRecord, "_read", reading)
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        returned = await Helper().run_from_config(config)
    finally:
        tracemalloc.stop()

    saved = (tmp_path / "run" / "rollouts.jsonl").stat().st_size
    assert saved > (rows - 2) * row_bytes
    if retain:
        assert len(returned) == rows
        assert samples["outcome saved"] > (rows - 3) * row_bytes
        assert samples["aggregation"] > saved
    else:
        assert returned == []
        assert set(samples) == {"outcome saved", "row read back", "aggregation"}
        assert max(samples.values()) < 4 * row_bytes, {k: round(v / row_bytes, 1) for k, v in samples.items()}


async def test_load_from_cache_reads_no_saved_rows_for_a_config_that_retains_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A driver counting what a resume would run (`len(_load_from_cache(config)[0])`) reads no saved row back when
    its config retains no results. A retaining config, or an explicit argument, still loads them."""
    _install_aggregating_server(monkeypatch)
    _write_inputs(tmp_path / "input.jsonl", _CASES, size=1024)
    output = tmp_path / "run" / "rollouts.jsonl"
    await _ScriptedHelper().run_from_config(_config(tmp_path, output, retain=False))
    resume = _config(tmp_path, output, retain=False).model_copy(update={"resume_from_cache": True})

    loaded = []
    selected = RolloutStore.selected

    def record_selected(store, disposition):
        loaded.append(disposition)
        return selected(store, disposition)

    monkeypatch.setattr(RolloutStore, "selected", record_selected)
    pending, rows, results, lines = RolloutCollectionHelper()._load_from_cache(resume)
    assert (rows, results, lines) == ([], [], [])
    assert "success" not in loaded

    retaining = resume.model_copy(update={"retain_results_in_memory": True})
    for config, kwargs in ((retaining, {}), (resume, {"retain_results_in_memory": True})):
        again, rows, results, lines = RolloutCollectionHelper()._load_from_cache(config, **kwargs)
        assert again == pending
        assert len(results) == len(rows) == len(lines) == _CASES.count("success")
        assert [orjson.loads(line) for [line] in lines] == results


async def test_store_streams_the_rows_it_selects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_aggregating_server(monkeypatch)
    _write_inputs(tmp_path / "input.jsonl", _CASES, size=1024)
    output = tmp_path / "run" / "rollouts.jsonl"
    await _ScriptedHelper().run_from_config(_config(tmp_path, output, retain=False))

    store = RolloutStore.read(output)
    assert list(store.iter_selected("success")) == store.selected("success")
    assert len(store.selected("success")) == _CASES.count("success")
    expected_failures = [
        row for row in store.selected("failure") + store.selected("omitted") if not row.get("_ng_omitted")
    ]
    assert list(store.iter_failures()) == store.failures() == expected_failures
    assert len(expected_failures) == _CASES.count("counted failure")


def test_aggregation_entry_is_stable_when_applied_again() -> None:
    """Collection may hand `_call_aggregate_metrics` entries already reduced; reducing them again changes nothing."""
    row = {
        "responses_create_params": {"input": []},
        "response": {"output": [{"type": "message"}], "usage": {"total_tokens": 3}, "metrics": {"ttft_s": 1.0}},
        "ng_trajectory": {"calls": []},
        "ng_agent_observations": [],
        "ng_model_call_capture": {"calls": []},
        "reward": 1.0,
        "_ng_task_index": 0,
    }
    entry = _aggregation_entry(row)
    assert entry == {
        "reward": 1.0,
        "_ng_task_index": 0,
        "response": {"usage": {"total_tokens": 3}, "metrics": {"ttft_s": 1.0}},
    }
    again = _aggregation_entry(entry)
    assert again == entry and list(again) == list(entry)
    assert _aggregation_entry({"reward": 0.0, "response": {"output": []}}) == {"reward": 0.0}
