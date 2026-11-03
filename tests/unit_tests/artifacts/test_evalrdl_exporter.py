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
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym.artifacts.exporters.isolated as isolated
from nemo_gym.artifacts.exporters import load_run_exporter_class
from nemo_gym.artifacts.exporters.base import ExportState
from nemo_gym.artifacts.exporters.evalrdl import EvalRDLExportConfig, EvalRDLExporter
from nemo_gym.artifacts.exporters.isolated import ensure_exporter_environment
from nemo_gym.artifacts.run_artifacts import RunArtifacts


# A stand-in for the `evalrdl` SDK with the same names and signatures, driven by environment variables. The worker
# runs as a real subprocess against it, so the harness, the payload and the result protocol are all exercised.
FAKE_SDK = """
import json
import os
from pathlib import Path


class RdlError(Exception):
    pass


class ClientTooOld(RdlError):
    pass


class AuthError(RdlError):
    def __init__(self, message, reason):
        super().__init__(message)
        self.reason = reason


class IntakeRefused(RdlError):
    def __init__(self, reason, stage, upload_token):
        super().__init__(reason)
        self.reason, self.stage, self.upload_token = reason, stage, upload_token


class UploadResult:
    def __init__(self, state):
        self.upload_token = "tok-1"
        self.state = state
        self.run_id = "run-9"
        self.benchmark = "gpqa"
        self.n_rollouts = 2
        self.needs_binding = os.environ.get("FAKE_EVALRDL_NEEDS_BINDING") == "1"


class _Uploads:
    def __init__(self, url):
        self.url = url

    def upload(self, path, *, label=None, meta=None, tags=(), parallel=4, resume=None, wait=True, timeout=1800,
               on_progress=None):
        staged = Path(path)
        Path(os.environ["FAKE_EVALRDL_LOG"]).write_text(json.dumps({
            "path": str(staged), "url": self.url, "label": label, "meta": meta, "tags": list(tags), "parallel": parallel, "wait": wait,
            "timeout": timeout, "files": sorted(p.name for p in staged.iterdir()),
            "symlinks": [p.name for p in staged.iterdir() if p.is_symlink()],
        }))
        mode = os.environ["FAKE_EVALRDL_MODE"]
        if mode == "refused":
            raise IntakeRefused("no gym markers", "decode", "tok-1")
        if mode == "rdl_error":
            raise RdlError("service unavailable")
        if mode == "too_old":
            raise ClientTooOld("client 0.0.1 is older than 0.0.300; run: pip install -U evalrdl")
        if mode == "anonymous":
            raise AuthError("no token was found", "anonymous")
        if not wait:
            return [UploadResult("accepted")]
        if mode == "two_results":
            return [UploadResult("committed"), UploadResult("committed")]
        return [UploadResult(mode)]


class Client:
    def __init__(self, url=None, token=None, **kwargs):
        self.uploads = _Uploads(url)
"""


@pytest.fixture
def fake_sdk(tmp_path: Path, monkeypatch: MonkeyPatch) -> dict[str, Any]:
    """Run the exporter's worker for real against the fake SDK, without building a virtual environment."""
    sdk_root = tmp_path / "fake_sdk"
    (sdk_root / "evalrdl" / "sdk").mkdir(parents=True)
    (sdk_root / "evalrdl" / "__init__.py").write_text("")
    (sdk_root / "evalrdl" / "sdk" / "__init__.py").write_text("")
    (sdk_root / "evalrdl" / "sdk" / "api.py").write_text(textwrap.dedent(FAKE_SDK))
    log = tmp_path / "sdk_call.json"
    requested: list[tuple[str, ...]] = []

    def fake_environment(venv_root: Path, name: str, requirements: tuple[str, ...]) -> Path:
        requested.append(requirements)
        return Path(sys.executable)

    monkeypatch.setattr(isolated, "ensure_exporter_environment", fake_environment)
    monkeypatch.setattr(
        isolated,
        "_worker_environment",
        lambda python: {**os.environ, "PYTHONPATH": str(sdk_root), "FAKE_EVALRDL_LOG": str(log)},
    )
    monkeypatch.setenv("FAKE_EVALRDL_MODE", "committed")
    return {"log": log, "requested": requested}


def _make_run(tmp_path: Path) -> RunArtifacts:
    rollouts = tmp_path / "gpqa_rollouts.jsonl"
    rollouts.write_text('{"reward": 1.0}\n')
    artifacts = RunArtifacts(rollouts)
    artifacts.aggregate_metrics_json_fpath.write_text("[]")
    artifacts.resolved_config_yaml_fpath.write_text("model: m\n")
    artifacts.failures_jsonl_fpath.write_text("")
    artifacts.materialized_inputs_jsonl_fpath.write_text('{"x": 1}\n')
    stamp = rollouts.stat().st_mtime + 10
    os.utime(artifacts.aggregate_metrics_json_fpath, (stamp, stamp))
    return artifacts


def _exporter(artifacts: RunArtifacts, tmp_path: Path, **settings: Any) -> EvalRDLExporter:
    config = {
        "rollouts_jsonl_fpath": str(artifacts.rollouts_jsonl_fpath),
        "uv_venv_dir": str(tmp_path / "v"),
        **settings,
    }
    return EvalRDLExporter.from_global_config(OmegaConf.create(config))


def _sdk_call(fake_sdk: dict[str, Any]) -> dict[str, Any]:
    return json.loads(fake_sdk["log"].read_text())


class TestConfig:
    def test_defaults(self) -> None:
        config = EvalRDLExportConfig(rollouts_jsonl_fpath=Path("r.jsonl"))

        assert (config.label, config.tags, config.meta, config.url) == (None, [], {}, None)
        assert (config.parallel, config.wait, config.timeout, config.requirement) == (4, True, 1800, "evalrdl")

    def test_command_line_containers_become_plain_python(self) -> None:
        overrides = OmegaConf.from_dotlist(["rollouts_jsonl_fpath=r.jsonl", "meta.k=v", "meta.n=3"])
        overrides.tags = ["a", "b"]

        config = EvalRDLExportConfig.model_validate(overrides)

        assert config.tags == ["a", "b"] and type(config.tags) is list
        assert config.meta == {"k": "v", "n": 3} and type(config.meta) is dict
        json.dumps({"tags": config.tags, "meta": config.meta})

    @pytest.mark.parametrize("settings", [{"parallel": 0}, {"timeout": 0}])
    def test_nonpositive_numbers_are_rejected(self, settings: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            EvalRDLExportConfig(rollouts_jsonl_fpath=Path("r.jsonl"), **settings)

    def test_is_a_registered_exporter_that_installs_the_configured_requirement(self, tmp_path: Path) -> None:
        assert load_run_exporter_class("evalrdl") is EvalRDLExporter
        artifacts = _make_run(tmp_path)

        assert _exporter(artifacts, tmp_path).required_packages() == ("evalrdl",)
        assert _exporter(artifacts, tmp_path, requirement="evalrdl==1.2.3").required_packages() == ("evalrdl==1.2.3",)


class TestRunDirectory:
    def test_the_directory_the_rollouts_are_in_is_uploaded_as_it_is(
        self, tmp_path: Path, fake_sdk: dict[str, Any]
    ) -> None:
        artifacts = _make_run(tmp_path)
        (tmp_path / "quality_summary.json").write_text("{}")
        (tmp_path / "model-calls").mkdir()
        (tmp_path / "model-calls" / "0-0.capture.jsonl").write_text("{}")
        before = sorted(p.name for p in tmp_path.iterdir())

        _exporter(artifacts, tmp_path).export(artifacts)

        call = _sdk_call(fake_sdk)
        assert call["path"] == str(tmp_path.resolve())
        assert call["files"] == before
        assert {"quality_summary.json", "model-calls", "gpqa_rollouts_resolved_config.yaml"} <= set(call["files"])
        assert sorted(p.name for p in tmp_path.iterdir() if p.name != fake_sdk["log"].name) == before

    def test_a_relative_rollouts_path_is_uploaded_by_its_absolute_directory(
        self, tmp_path: Path, fake_sdk: dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        _make_run(tmp_path)
        monkeypatch.chdir(tmp_path)
        relative = RunArtifacts(Path("gpqa_rollouts.jsonl"))

        _exporter(relative, tmp_path).export(relative)

        assert _sdk_call(fake_sdk)["path"] == str(tmp_path.resolve())

    def test_a_symlinked_rollouts_file_is_uploaded_from_the_directory_it_is_linked_in(
        self, tmp_path: Path, fake_sdk: dict[str, Any]
    ) -> None:
        target = tmp_path / "elsewhere"
        target.mkdir()
        real = _make_run(target)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        link = run_dir / "gpqa_rollouts.jsonl"
        link.symlink_to(real.rollouts_jsonl_fpath)
        for sidecar in ("aggregate_metrics_json_fpath", "resolved_config_yaml_fpath"):
            getattr(RunArtifacts(link), sidecar).write_bytes(getattr(real, sidecar).read_bytes())
        linked = RunArtifacts(link)
        stamp = real.rollouts_jsonl_fpath.stat().st_mtime + 10
        os.utime(linked.aggregate_metrics_json_fpath, (stamp, stamp))

        _exporter(linked, tmp_path).export(linked)

        assert _sdk_call(fake_sdk)["path"] == str(run_dir.resolve())

    @pytest.mark.parametrize("other", ["math_rollouts.jsonl", "rollouts.jsonl"])
    def test_a_directory_holding_another_runs_rollouts_is_refused_before_any_environment_is_built(
        self, tmp_path: Path, fake_sdk: dict[str, Any], other: str
    ) -> None:
        artifacts = _make_run(tmp_path)
        (tmp_path / other).write_text("{}\n")

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert result.state is ExportState.FAILED
        assert f"also holds {other}" in result.error and "directory of its own" in result.error
        assert fake_sdk["requested"] == []
        assert not fake_sdk["log"].exists()

    def test_the_runs_own_sidecars_are_not_another_runs_rollouts(
        self, tmp_path: Path, fake_sdk: dict[str, Any]
    ) -> None:
        artifacts = _make_run(tmp_path)

        assert _exporter(artifacts, tmp_path).export(artifacts).state is ExportState.SUCCEEDED

    @pytest.mark.parametrize(("wait", "expected"), [(True, 1800 + 3600), (False, 3600)])
    def test_the_worker_may_run_for_the_upload_allowance_plus_the_wait_for_a_verdict(
        self, tmp_path: Path, wait: bool, expected: float
    ) -> None:
        artifacts = _make_run(tmp_path)

        assert _exporter(artifacts, tmp_path, wait=wait).worker_timeout_s({}) == expected

    @pytest.mark.parametrize(
        ("remove", "message"),
        [
            ("aggregate_metrics_json_fpath", "Aggregate metrics not found"),
            ("resolved_config_yaml_fpath", "Saved run config not found"),
            ("rollouts_jsonl_fpath", "Rollouts file not found"),
        ],
    )
    def test_a_run_missing_what_the_reader_needs_fails_before_any_environment_is_built(
        self, tmp_path: Path, fake_sdk: dict[str, Any], remove: str, message: str
    ) -> None:
        artifacts = _make_run(tmp_path)
        getattr(artifacts, remove).unlink()

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert result.state is ExportState.FAILED
        assert message in result.error
        assert fake_sdk["requested"] == []
        assert not fake_sdk["log"].exists()

    def test_an_aggregate_file_with_a_failed_agent_is_refused_before_any_environment_is_built(
        self, tmp_path: Path, fake_sdk: dict[str, Any]
    ) -> None:
        artifacts = _make_run(tmp_path)
        artifacts.aggregate_metrics_json_fpath.write_text('[{"agent_ref": {"name": "bad"}, "aggregation_error": {}}]')

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert result.state is ExportState.FAILED
        assert "Aggregation failed for agents: bad" in result.error
        assert fake_sdk["requested"] == []


class TestUpload:
    def test_every_setting_reaches_the_sdk(self, tmp_path: Path, fake_sdk: dict[str, Any]) -> None:
        artifacts = _make_run(tmp_path)

        _exporter(
            artifacts,
            tmp_path,
            label="nightly",
            tags=["a", "b"],
            meta={"k": "v"},
            parallel=2,
            wait=False,
            timeout=60,
            url="stg",
        ).export(artifacts)

        call = _sdk_call(fake_sdk)
        assert {key: call[key] for key in ("url", "label", "tags", "meta", "parallel", "wait", "timeout")} == {
            "url": "stg",
            "label": "nightly",
            "tags": ["a", "b"],
            "meta": {"k": "v"},
            "parallel": 2,
            "wait": False,
            "timeout": 60,
        }

    def test_unset_settings_leave_the_sdk_to_its_own_defaults(self, tmp_path: Path, fake_sdk: dict[str, Any]) -> None:
        artifacts = _make_run(tmp_path)

        _exporter(artifacts, tmp_path).export(artifacts)

        call = _sdk_call(fake_sdk)
        assert (call["url"], call["label"], call["meta"], call["tags"]) == (None, None, None, [])
        assert (call["parallel"], call["wait"], call["timeout"]) == (4, True, 1800)
        assert fake_sdk["requested"] == [("evalrdl",)]

    @pytest.mark.parametrize(
        ("mode", "state", "exit_code"),
        [
            ("committed", ExportState.SUCCEEDED, 0),
            ("deduped", ExportState.SKIPPED, 0),
            ("accepted", ExportState.FAILED, 3),
            ("refused", ExportState.FAILED, 4),
            ("rdl_error", ExportState.FAILED, 1),
            ("two_results", ExportState.FAILED, 1),
        ],
    )
    def test_each_upload_state_maps_to_a_result_and_exit_code(
        self, tmp_path: Path, fake_sdk: dict[str, Any], monkeypatch: MonkeyPatch, mode: str, state, exit_code: int
    ) -> None:
        monkeypatch.setenv("FAKE_EVALRDL_MODE", mode)
        artifacts = _make_run(tmp_path)

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert (result.state, result.exit_code) == (state, exit_code)

    def test_a_committed_upload_reports_the_run_it_created(self, tmp_path: Path, fake_sdk: dict[str, Any]) -> None:
        artifacts = _make_run(tmp_path)

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert result.detail == {
            "evalrdl_state": "committed",
            "upload_token": "tok-1",
            "run_id": "run-9",
            "benchmark": "gpqa",
            "n_rollouts": 2,
            "needs_binding": False,
        }

    def test_a_refusal_names_the_stage_and_reason(
        self, tmp_path: Path, fake_sdk: dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_EVALRDL_MODE", "refused")
        artifacts = _make_run(tmp_path)

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert result.error == "EvalRDL refused the upload at stage 'decode': no gym markers"
        assert (result.detail["evalrdl_state"], result.detail["stage"]) == ("refused", "decode")

    def test_without_waiting_an_accepted_upload_is_the_expected_outcome(
        self, tmp_path: Path, fake_sdk: dict[str, Any]
    ) -> None:
        artifacts = _make_run(tmp_path)

        result = _exporter(artifacts, tmp_path, wait=False).export(artifacts)

        assert result.state is ExportState.SUCCEEDED, result.error
        assert result.detail["upload_token"] == "tok-1"

    def test_a_committed_upload_that_needs_binding_says_so(
        self, tmp_path: Path, fake_sdk: dict[str, Any], monkeypatch: MonkeyPatch, capfd: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("FAKE_EVALRDL_NEEDS_BINDING", "1")
        artifacts = _make_run(tmp_path)

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert result.state is ExportState.SUCCEEDED
        assert "needs_binding=true (upload token tok-1)" in capfd.readouterr().err

    def test_a_client_that_is_too_old_names_the_environment_to_rebuild(
        self, tmp_path: Path, fake_sdk: dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_EVALRDL_MODE", "too_old")
        artifacts = _make_run(tmp_path)

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert result.state is ExportState.FAILED
        assert result.error.startswith(
            "ClientTooOld: client 0.0.1 is older than 0.0.300; run: pip install -U evalrdl."
        )
        assert "delete that directory to rebuild it" in result.error and "+requirement=evalrdl==" in result.error

    def test_a_missing_token_names_the_environment_variables(
        self, tmp_path: Path, fake_sdk: dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_EVALRDL_MODE", "anonymous")
        artifacts = _make_run(tmp_path)

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert "Set EVALRDL_TOKEN (or GITLAB_TOKEN or GITLAB_PRIVATE_TOKEN)" in result.error

    def test_an_accepted_upload_points_at_its_token(
        self, tmp_path: Path, fake_sdk: dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_EVALRDL_MODE", "accepted")
        artifacts = _make_run(tmp_path)

        result = _exporter(artifacts, tmp_path).export(artifacts)

        assert "no verdict before the timeout" in result.error
        assert result.detail["upload_token"] == "tok-1"


@pytest.mark.with_evalrdl_sdk
def test_the_real_sdk_still_has_the_interface_the_worker_calls(tmp_path: Path) -> None:
    """Installs the real SDK into the exporter environment and checks the names and arguments the worker uses.

    It uploads nothing and needs no token. The package index that serves the SDK comes from your own uv
    configuration (for example `UV_EXTRA_INDEX_URL`), never from this repository. Run it with
    `pytest --with_evalrdl_sdk`.
    """
    python = ensure_exporter_environment(
        tmp_path, "evalrdl", (EvalRDLExportConfig.model_fields["requirement"].default,)
    )
    check = """
import inspect
from evalrdl.sdk.api import Client, IntakeRefused, RdlError
upload = inspect.signature(Client(url="stg", token=False).uploads.upload)
wanted = {"label", "tags", "meta", "parallel", "wait", "timeout"}
missing = wanted - set(upload.parameters)
assert not missing, f"upload() no longer takes {missing}"
assert issubclass(IntakeRefused, RdlError)
"""
    completed = subprocess.run([str(python), "-c", check], capture_output=True, text=True, check=False, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
