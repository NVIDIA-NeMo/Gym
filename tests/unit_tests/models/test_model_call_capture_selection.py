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
"""Which model calls the evaluation capture store records: the server allowlist, per-rollout sampling,
the free-space floor, and the calling server named on each record."""

import logging
import os
import zlib
from types import SimpleNamespace

import pytest
from fastapi import Body, FastAPI
from fastapi.testclient import TestClient

import nemo_gym.models.model_call_capture as model_call_capture
from nemo_gym.core.episode_context import CALLER_SERVER_HEADER
from nemo_gym.core.episode_types import EpisodeId
from nemo_gym.models.model_call_capture import (
    CaptureStore,
    ModelCallCaptureConfig,
    _CaptureGate,
    install_model_call_capture,
)
from nemo_gym.rollouts.model_calls import read_model_call_records


def _app(tmp_path, *, model_server_name: str = "srv", **config) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/responses")
    async def _responses(body: dict = Body()) -> dict:
        return {"id": "resp", "object": "response", "output": [], "echo": body}

    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path, **config),
        model_server_name=model_server_name,
    )
    return app


def _group_admitted(group: str, sample_every: int) -> bool:
    return zlib.crc32(group.encode()) % sample_every == 0


def test_defaults_keep_every_call_of_every_server(tmp_path) -> None:
    config = ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path)
    assert config.model_call_capture_servers is None
    assert config.model_call_capture_sample_every == 1
    assert config.model_call_capture_min_free_gib == 0.0
    assert config.records_server("policy_model") and config.records_server(None)

    client = TestClient(_app(tmp_path, model_server_name="policy_model"))
    for rollout_id in ("a_g0", "b_g1", "c"):
        assert client.post(f"/ng-rollout/{rollout_id}/v1/responses", json={"input": "x"}).status_code == 200
        assert len(CaptureStore(tmp_path).read(rollout_id)) == 1


def test_default_gate_never_checks_free_space(tmp_path, monkeypatch) -> None:
    def _no_statvfs(_path):
        raise AssertionError("statvfs must not be called without a free-space floor")

    monkeypatch.setattr(model_call_capture.os, "statvfs", _no_statvfs)
    assert _CaptureGate(tmp_path, sample_every=1, min_free_gib=0.0).has_space()


def test_server_allowlist_records_only_listed_servers(tmp_path) -> None:
    config = {"model_call_capture_servers": ["judge_model"]}
    policy = TestClient(_app(tmp_path / "policy", model_server_name="policy_model", **config))
    judge = TestClient(_app(tmp_path / "judge", model_server_name="judge_model", **config))

    # The unlisted server still answers on the rollout-prefixed path; it just records nothing.
    response = policy.post("/ng-rollout/r1_g0/v1/responses", json={"input": "x"})
    assert response.status_code == 200 and response.json()["echo"] == {"input": "x"}
    assert CaptureStore(tmp_path / "policy").read("r1_g0") == []

    assert judge.post("/ng-rollout/r1_g0/v1/responses", json={"input": "x"}).status_code == 200
    [call] = read_model_call_records(CaptureStore(tmp_path / "judge"), "r1_g0")
    assert call.model_ref is not None and call.model_ref.name == "judge_model"


@pytest.mark.parametrize("extra_names", [[], ["typo", "env_a", "observability_enabled"]])
def test_server_allowlist_warns_about_names_that_are_not_local_model_servers(tmp_path, caplog, extra_names) -> None:
    config = ModelCallCaptureConfig(
        observability_enabled=True,
        model_call_capture_dir=tmp_path,
        model_call_capture_servers=["judge_model", *extra_names],
    )
    global_config = {
        "judge_model": {"responses_api_models": {"vllm_model": {}}},
        "env_a": {"resources_servers": {"env": {}}},
        "observability_enabled": True,
    }
    with caplog.at_level(logging.WARNING, logger=model_call_capture.logger.name):
        install_model_call_capture(
            FastAPI(), config, model_server_name="judge_model", global_config_dict=global_config
        )
    warnings = [
        record.getMessage() for record in caplog.records if "model_call_capture_servers" in record.getMessage()
    ]
    if extra_names:
        [warning] = warnings
        assert all(name in warning for name in extra_names)
        assert "judge_model" not in warning
    else:
        assert warnings == []


@pytest.mark.parametrize(
    "capture_key, rollout_id",
    [
        ("abc", "abc"),
        ("abc_g0", "abc_g0"),
        ("abc_g3-a1", "abc_g3"),
        ("foo-a0", "foo-a0"),
        ("task-a01-a1", "task-a01"),
        ("x-a1-a2", "x-a1-a2"),  # Opaque key outside Gym's episode ID grammar.
        # Rollout collection's own keys are `<task>-<rollout>`, and a re-dispatch appends `-a<n>`.
        ("3-0", "3-0"),
        ("3-0-a1", "3-0"),
        ("3-0-a12", "3-0"),
    ],
)
def test_capture_group_uses_the_logical_rollout_id(capture_key: str, rollout_id: str) -> None:
    assert _CaptureGate.group_of(capture_key) == rollout_id


def test_sample_every_keeps_or_drops_all_gym_attempts_of_a_rollout(tmp_path) -> None:
    sample_every = 4
    rollout_ids = [
        "3-0",
        "3-1",
        "foo-a0",
        "task-a01",
        # NeMo RL sends each generation's own ID, including its attempt UUID for token capture.
        "prompt-g0-t0",
        "prompt-g3-t0",
        f"prompt_g0_a{0:032x}",
        f"prompt_g1_a{1:032x}",
    ]
    assert {_group_admitted(rollout_id, sample_every) for rollout_id in rollout_ids} == {True, False}
    client = TestClient(_app(tmp_path, model_server_name="judge_model", model_call_capture_sample_every=sample_every))
    store = CaptureStore(tmp_path)
    for rollout_id in rollout_ids:
        expected_records = 1 if _group_admitted(rollout_id, sample_every) else 0
        for attempt in (0, 1, 2):
            key = EpisodeId(rollout_id=rollout_id, attempt=attempt).capture_key
            assert client.post(f"/ng-rollout/{key}/v1/responses", json={"input": "x"}).status_code == 200
            assert len(store.read(key)) == expected_records, key


def test_sample_every_keeps_about_one_rollout_in_n(tmp_path) -> None:
    gate = _CaptureGate(tmp_path, sample_every=8, min_free_gib=0.0)
    kept = sum(gate.samples(f"prompt-{i}_g0") for i in range(4000))
    assert 400 < kept < 600


def test_sample_every_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError):
        ModelCallCaptureConfig(
            observability_enabled=True, model_call_capture_dir=tmp_path, model_call_capture_sample_every=0
        )


def _statvfs(free_bytes: int):
    return lambda _path: SimpleNamespace(f_bavail=free_bytes, f_frsize=1)


def test_free_space_floor_stops_and_resumes_recording(tmp_path, monkeypatch, caplog) -> None:
    gib = 2**30
    monkeypatch.setattr(_CaptureGate, "_SPACE_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(model_call_capture.os, "statvfs", _statvfs(1 * gib))
    client = TestClient(_app(tmp_path, model_server_name="judge_model", model_call_capture_min_free_gib=5))

    with caplog.at_level(logging.WARNING, logger=model_call_capture.logger.name):
        response = client.post("/ng-rollout/low_g0/v1/responses", json={"input": "x"})
    assert response.status_code == 200  # the call itself is unaffected
    assert CaptureStore(tmp_path).read("low_g0") == []
    assert CaptureStore(tmp_path).is_incomplete("low_g0")
    assert any("paused" in record.getMessage() for record in caplog.records)

    caplog.clear()
    monkeypatch.setattr(model_call_capture.os, "statvfs", _statvfs(10 * gib))
    with caplog.at_level(logging.WARNING, logger=model_call_capture.logger.name):
        client.post("/ng-rollout/ok_g0/v1/responses", json={"input": "x"})
    assert len(CaptureStore(tmp_path).read("ok_g0")) == 1
    assert not CaptureStore(tmp_path).is_incomplete("ok_g0")
    assert any("resumed" in record.getMessage() for record in caplog.records)
    assert client.post("/ng-rollout/low_g0/v1/responses", json={"input": "x"}).status_code == 200
    assert len(CaptureStore(tmp_path).read("low_g0")) == 1
    assert CaptureStore(tmp_path).is_incomplete("low_g0")


def test_free_space_check_is_rate_limited(tmp_path, monkeypatch) -> None:
    calls = []

    def _counting_statvfs(path):
        calls.append(path)
        return SimpleNamespace(f_bavail=100 * 2**30, f_frsize=1)

    monkeypatch.setattr(model_call_capture.os, "statvfs", _counting_statvfs)
    gate = _CaptureGate(tmp_path, sample_every=1, min_free_gib=1.0)
    assert all(gate.has_space() for _ in range(50))
    assert len(calls) == 1


def test_unreadable_capture_filesystem_counts_as_full(tmp_path, monkeypatch) -> None:
    def _failing_statvfs(_path):
        raise OSError("gone")

    monkeypatch.setattr(model_call_capture.os, "statvfs", _failing_statvfs)
    assert not _CaptureGate(tmp_path, sample_every=1, min_free_gib=1.0).has_space()


def test_caller_server_is_recorded_when_the_request_names_it(tmp_path) -> None:
    client = TestClient(_app(tmp_path, model_server_name="judge_model"))

    client.post("/ng-rollout/named_g0/v1/responses", json={"input": "x"}, headers={CALLER_SERVER_HEADER: "env_a"})
    client.post("/ng-rollout/anon_g0/v1/responses", json={"input": "x"})

    [named] = CaptureStore(tmp_path).read("named_g0")
    [anon] = CaptureStore(tmp_path).read("anon_g0")
    assert named["caller_server"] == "env_a"
    assert "caller_server" not in anon


def test_disabled_capture_ignores_the_new_settings(tmp_path) -> None:
    app = FastAPI()

    @app.post("/v1/responses")
    async def _responses(body: dict = Body()) -> dict:
        return {"ok": True}

    install_model_call_capture(
        app,
        ModelCallCaptureConfig(
            model_call_capture_dir=tmp_path,
            model_call_capture_servers=["judge_model"],
            model_call_capture_sample_every=3,
        ),
        model_server_name="judge_model",
    )
    assert _group_admitted("a", 3)
    assert TestClient(app).post("/ng-rollout/a/v1/responses", json={"input": "x"}).json() == {"ok": True}
    assert not any(os.scandir(tmp_path))
