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
import argparse

import pytest
import yaml
from hydra.errors import ConfigCompositionException
from pydantic import ValidationError
from pytest import MonkeyPatch

import nemo_gym.orchestration.submit as submit_module
from nemo_gym.cli.main import _eval_submit


COMPUTE = {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}}
SERVICE = {"container": "gym:latest", "type": "vllm", "model": "org/model"}
DRIVER = {"container": "gym:latest", "benchmarks": {"gsm8k": {}}}
JOB = {"output_path": "/tmp/gym-jobs"}


def _args(config_path, *, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(config=str(config_path), dry_run=dry_run)


def _capture_submit(monkeypatch: MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_submit(config, *, dry_run: bool = False) -> None:
        captured["config"] = config
        captured["dry_run"] = dry_run

    monkeypatch.setattr(submit_module, "submit", fake_submit)
    return captured


class TestEvalSubmitFlatConfig:
    def test_flat_config_validates_and_submits(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        captured = _capture_submit(monkeypatch)
        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump({"services": {"svc": SERVICE}, "compute": COMPUTE, "driver": DRIVER, "job": JOB})
        )

        _eval_submit(_args(config_path), overrides=[])

        assert captured["config"].job.output_path == "/tmp/gym-jobs"
        assert captured["dry_run"] is False

    def test_dry_run_flag_is_forwarded(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        captured = _capture_submit(monkeypatch)
        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump({"services": {"svc": SERVICE}, "compute": COMPUTE, "driver": DRIVER, "job": JOB})
        )

        _eval_submit(_args(config_path, dry_run=True), overrides=[])

        assert captured["dry_run"] is True

    def test_bare_override_replaces_existing_key(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        captured = _capture_submit(monkeypatch)
        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump({"services": {"svc": SERVICE}, "compute": COMPUTE, "driver": DRIVER, "job": JOB})
        )

        _eval_submit(_args(config_path), overrides=["job.output_path=/tmp/other"])

        assert captured["config"].job.output_path == "/tmp/other"

    def test_plus_prefix_rejects_override_of_existing_key(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        _capture_submit(monkeypatch)
        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump({"services": {"svc": SERVICE}, "compute": COMPUTE, "driver": DRIVER, "job": JOB})
        )

        with pytest.raises(ConfigCompositionException):
            _eval_submit(_args(config_path), overrides=["+job.output_path=/tmp/other"])

    def test_plus_prefix_adds_new_key(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        captured = _capture_submit(monkeypatch)
        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump({"services": {"svc": SERVICE}, "compute": COMPUTE, "driver": DRIVER, "job": JOB})
        )

        _eval_submit(_args(config_path), overrides=["+driver.env.FOO=bar"])

        assert captured["config"].driver.env == {"FOO": "bar"}


class TestEvalSubmitScratchNamespace:
    """Root-level keys prefixed with `_` are scratch namespaces: not part of SubmitConfig's schema, only
    present so other fields can interpolate into them. They must be fully defined in the config file
    itself (only pre-existing leaves may be overridden, and only with a bare `key=value`), and get
    resolved-then-stripped before validation instead of tripping SubmitConfig's `extra="forbid"`."""

    def _write_config(self, tmp_path, **extra):
        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump({"services": {"svc": SERVICE}, "compute": COMPUTE, "driver": DRIVER, "job": JOB, **extra})
        )
        return config_path

    def test_scratch_namespace_is_stripped_before_validation(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        captured = _capture_submit(monkeypatch)
        config_path = self._write_config(tmp_path, _my_env_space={"tag": "default-tag"})

        _eval_submit(_args(config_path), overrides=[])

        assert not hasattr(captured["config"], "_my_env_space")

    def test_scratch_namespace_is_resolved_before_being_stripped(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        """A field can interpolate into the scratch namespace; the interpolated value must survive even
        though the scratch namespace itself gets dropped afterwards."""
        captured = _capture_submit(monkeypatch)
        config_path = self._write_config(
            tmp_path,
            _my_env_space={"tag": "default-tag"},
            driver={**DRIVER, "container": "gym:${_my_env_space.tag}"},
        )

        _eval_submit(_args(config_path), overrides=[])

        assert captured["config"].driver.container == "gym:default-tag"
        assert not hasattr(captured["config"], "_my_env_space")

    def test_bare_override_of_existing_scratch_leaf(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        captured = _capture_submit(monkeypatch)
        config_path = self._write_config(
            tmp_path,
            _my_env_space={"tag": "default-tag"},
            driver={**DRIVER, "container": "gym:${_my_env_space.tag}"},
        )

        _eval_submit(_args(config_path), overrides=["_my_env_space.tag=nightly"])

        assert captured["config"].driver.container == "gym:nightly"

    def test_bare_override_of_typo_scratch_leaf_fails(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        """Hydra itself rejects a bare override of a key that doesn't already exist, for free."""
        config_path = self._write_config(tmp_path, _my_env_space={"tag": "default-tag"})

        with pytest.raises(ConfigCompositionException):
            _eval_submit(_args(config_path), overrides=["_my_env_space.tagg=nightly"])

    def test_plus_override_into_scratch_namespace_is_rejected(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        """`+`/`++` could silently create an unused, typo'd field in a scratch namespace; refuse it
        outright instead of letting it through."""
        config_path = self._write_config(tmp_path, _my_env_space={"tag": "default-tag"})

        with pytest.raises(ValueError, match="scratch namespace"):
            _eval_submit(_args(config_path), overrides=["+_my_env_space.tagg=nightly"])

    def test_typo_in_top_level_key_is_rejected(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        """A root key that isn't `_`-prefixed and isn't a SubmitConfig field is a real typo, not a scratch
        namespace — it must still hit SubmitConfig's strict validation instead of being silently dropped."""
        config_path = self._write_config(tmp_path, drivver=DRIVER)

        with pytest.raises(ValidationError):
            _eval_submit(_args(config_path), overrides=[])

    def test_without_scratch_namespace_still_validates(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        captured = _capture_submit(monkeypatch)
        config_path = self._write_config(tmp_path)

        _eval_submit(_args(config_path), overrides=[])

        assert captured["config"].job.output_path == "/tmp/gym-jobs"


class TestEvalSubmitConfigGroupComposition:
    def test_defaults_list_pulls_in_config_group(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        """A `defaults:` list should compose config-group files the same way real Hydra does."""
        captured = _capture_submit(monkeypatch)

        compute_dir = tmp_path / "compute"
        compute_dir.mkdir()
        (compute_dir / "slurm.yaml").write_text(yaml.dump(COMPUTE))

        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "defaults": [{"compute": "slurm"}, "_self_"],
                    "services": {"svc": SERVICE},
                    "driver": DRIVER,
                    "job": JOB,
                }
            )
        )

        _eval_submit(_args(config_path), overrides=[])

        assert captured["config"].compute["cluster"].account == "my-account"
        assert captured["config"].compute["cluster"].hostname == "foo"

    def test_config_group_can_be_overridden_from_cli(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        captured = _capture_submit(monkeypatch)

        compute_dir = tmp_path / "compute"
        compute_dir.mkdir()
        (compute_dir / "slurm.yaml").write_text(yaml.dump(COMPUTE))
        other_compute = {"cluster": {"type": "slurm", "account": "other-account", "hostname": "bar"}}
        (compute_dir / "other.yaml").write_text(yaml.dump(other_compute))

        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "defaults": [{"compute": "slurm"}, "_self_"],
                    "services": {"svc": SERVICE},
                    "driver": DRIVER,
                    "job": JOB,
                }
            )
        )

        _eval_submit(_args(config_path), overrides=["compute=other"])

        assert captured["config"].compute["cluster"].account == "other-account"
        assert captured["config"].compute["cluster"].hostname == "bar"

    def test_repeated_calls_do_not_leak_global_hydra_state(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        """GlobalHydra must be reset between calls or the second `initialize_config_dir` raises."""
        captured = _capture_submit(monkeypatch)
        config_path = tmp_path / "submit.yaml"
        config_path.write_text(
            yaml.dump({"services": {"svc": SERVICE}, "compute": COMPUTE, "driver": DRIVER, "job": JOB})
        )

        _eval_submit(_args(config_path), overrides=[])
        _eval_submit(_args(config_path), overrides=[])

        assert captured["config"].job.output_path == "/tmp/gym-jobs"
