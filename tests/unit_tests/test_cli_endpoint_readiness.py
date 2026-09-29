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
from unittest.mock import MagicMock

import requests
from omegaconf import DictConfig
from pytest import MonkeyPatch, raises

import nemo_gym.cli.env
from nemo_gym.cli.env import (
    _ENDPOINT_POLL_INTERVAL_SEC,
    RunHelper,
    _collect_model_endpoints,
    _model_endpoint_timeout_seconds,
    _probe_endpoint,
    _wait_for_model_endpoints,
)
from nemo_gym.config_types import ConfigError


class _FakeClock:
    """Stand-in for `time` and `sleep` so the wait loop runs without real waiting."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += seconds


def _config(**model_servers) -> DictConfig:
    return DictConfig(
        {
            "dry_run": False,
            "head_server": {"host": "127.0.0.1", "port": 11000},
            **model_servers,
        }
    )


class TestEndpointsFromConfigShapes:
    """Driven from the model server config classes so a new server or a changed field type fails
    here rather than quietly shrinking what the check covers."""

    def test_reads_a_string_base_url(self) -> None:
        config = _config(
            policy_model={
                "responses_api_models": {
                    "openai_model": {"entrypoint": "app.py", "openai_base_url": "https://api.openai.com/v1"}
                }
            }
        )
        assert [("openai_base_url", "https://api.openai.com/v1")] == _collect_model_endpoints(config)

    def test_reads_a_list_base_url(self) -> None:
        """vllm_model and the local vLLM servers type base_url as Union[str, List[str]] for load
        balancing, which is the shape a training run is most likely to use."""
        config = _config(
            policy_model={
                "responses_api_models": {
                    "vllm_model": {
                        "entrypoint": "app.py",
                        "base_url": ["http://replica-a:8000/v1", "http://replica-b:8000/v1"],
                    }
                }
            }
        )
        assert [
            ("base_url", "http://replica-a:8000/v1"),
            ("base_url", "http://replica-b:8000/v1"),
        ] == _collect_model_endpoints(config)

    def test_skips_an_empty_list(self) -> None:
        """local_vllm_model carries [] until Gym launches vLLM and fills it in."""
        config = _config(
            policy_model={"responses_api_models": {"local_vllm_model": {"entrypoint": "app.py", "base_url": []}}}
        )
        assert [] == _collect_model_endpoints(config)

    def test_skips_null(self) -> None:
        config = _config(
            judge={"responses_api_models": {"openai_model": {"entrypoint": "app.py", "openai_base_url": None}}}
        )
        assert [] == _collect_model_endpoints(config)

    def test_carries_the_config_key_for_the_report(self) -> None:
        config = _config(
            judge={
                "responses_api_models": {
                    "openai_model": {"entrypoint": "app.py", "openai_base_url": "http://judge:8000/v1"}
                }
            }
        )
        assert [("openai_base_url", "http://judge:8000/v1")] == _collect_model_endpoints(config)


class TestProbeClassification:
    def test_any_http_answer_is_reachable(self, monkeypatch: MonkeyPatch) -> None:
        for status_code in (200, 401, 404, 500):
            monkeypatch.setattr(
                nemo_gym.cli.env.requests, "get", MagicMock(return_value=MagicMock(status_code=status_code))
            )
            assert nemo_gym.cli.env._ENDPOINT_ANSWERING == _probe_endpoint("http://x:8000/v1")

    def test_untrusted_certificate_is_reachable(self, monkeypatch: MonkeyPatch) -> None:
        """A completed TLS handshake proves something is listening. SSLError subclasses
        ConnectionError, so deciding on the parent class would reject it."""
        monkeypatch.setattr(
            nemo_gym.cli.env.requests, "get", MagicMock(side_effect=requests.exceptions.SSLError("bad cert"))
        )
        assert nemo_gym.cli.env._ENDPOINT_ANSWERING == _probe_endpoint("https://x:8000/v1")

    def test_refused_is_worth_waiting_for(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            nemo_gym.cli.env.requests, "get", MagicMock(side_effect=requests.exceptions.ConnectionError("refused"))
        )
        assert nemo_gym.cli.env._ENDPOINT_REFUSED == _probe_endpoint("http://x:8000/v1")

    def test_an_unresolvable_name_is_not_worth_waiting_for(self, monkeypatch: MonkeyPatch) -> None:
        """Waiting cannot create a DNS record, so a placeholder like unset.local is reported once
        instead of costing the whole timeout."""
        dns_failure = requests.exceptions.ConnectionError(
            requests.packages.urllib3.exceptions.NameResolutionError("unset.local", None, Exception("no such host"))
        )
        monkeypatch.setattr(nemo_gym.cli.env.requests, "get", MagicMock(side_effect=dns_failure))
        assert nemo_gym.cli.env._ENDPOINT_UNRESOLVABLE == _probe_endpoint("http://unset.local/v1")


class TestCheckStopsStartupCleanly:
    def test_unresolvable_endpoints_do_not_block(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            nemo_gym.cli.env, "_probe_endpoint", MagicMock(return_value=nemo_gym.cli.env._ENDPOINT_UNRESOLVABLE)
        )
        sleep_mock = MagicMock()
        unreachable = _wait_for_model_endpoints(
            [("openai_base_url", "http://unset.local/v1")], timeout_seconds=600, sleep_fn=sleep_mock
        )

        assert [] == unreachable
        sleep_mock.assert_not_called()

    def test_servers_are_shut_down_before_the_error(self, monkeypatch: MonkeyPatch) -> None:
        """The Popens have no process group and no atexit handler, and every caller reaches
        shutdown() only after start() returns."""
        monkeypatch.setattr(
            nemo_gym.cli.env, "_wait_for_model_endpoints", MagicMock(return_value=[("openai_base_url", "http://x/v1")])
        )
        helper = RunHelper.__new__(RunHelper)
        shutdown_mock = MagicMock()
        helper.shutdown = shutdown_mock

        with raises(ConfigError):
            RunHelper.wait_for_model_endpoints(helper, _config(model_endpoint_readiness_timeout_seconds=1))

        shutdown_mock.assert_called_once()

    def test_failure_is_a_config_error_not_a_system_exit(self, monkeypatch: MonkeyPatch) -> None:
        """NeMo-RL imports RunHelper, so a library method must not exit the process."""
        monkeypatch.setattr(
            nemo_gym.cli.env, "_wait_for_model_endpoints", MagicMock(return_value=[("openai_base_url", "http://x/v1")])
        )
        helper = RunHelper.__new__(RunHelper)
        helper.shutdown = MagicMock()

        with raises(ConfigError) as exc_info:
            RunHelper.wait_for_model_endpoints(helper, _config(model_endpoint_readiness_timeout_seconds=1))

        assert not isinstance(exc_info.value, SystemExit)
        # The report names the config key, since the endpoint may not be the policy model.
        assert "openai_base_url" in str(exc_info.value)


class TestTimeoutFromConfig:
    def test_absent_key_uses_the_same_default_as_the_config_parser(self) -> None:
        """A caller that builds a config directly should not silently skip the check."""
        assert 600.0 == _model_endpoint_timeout_seconds(_config())

    def test_zero_and_negative_skip_the_check(self) -> None:
        assert 0.0 == _model_endpoint_timeout_seconds(_config(model_endpoint_readiness_timeout_seconds=0))
        assert -1.0 == _model_endpoint_timeout_seconds(_config(model_endpoint_readiness_timeout_seconds=-1))

    def test_a_non_numeric_value_is_reported_as_a_config_mistake(self) -> None:
        with raises(ConfigError) as exc_info:
            _model_endpoint_timeout_seconds(_config(model_endpoint_readiness_timeout_seconds="10m"))

        assert "model_endpoint_readiness_timeout_seconds" in str(exc_info.value)
        assert "10m" in str(exc_info.value)

    def test_null_uses_the_default_rather_than_skipping(self) -> None:
        assert 600.0 == _model_endpoint_timeout_seconds(_config(model_endpoint_readiness_timeout_seconds=None))

    def test_the_bound_covers_the_first_probe_round(self) -> None:
        """An unresolvable name costs a full resolver timeout in the first round. If the clock
        started after that round, several of them would push the real wall time past the bound."""
        clock = _FakeClock()

        def slow_probe(url: str, timeout_seconds: float = 5.0) -> str:
            clock.sleep(5.0)
            return nemo_gym.cli.env._ENDPOINT_UNRESOLVABLE if "unset" in url else nemo_gym.cli.env._ENDPOINT_REFUSED

        with MonkeyPatch.context() as mp:
            mp.setattr(nemo_gym.cli.env, "_probe_endpoint", slow_probe)
            _wait_for_model_endpoints(
                [("a", "http://unset-1/v1"), ("b", "http://unset-2/v1"), ("c", "http://down/v1")],
                timeout_seconds=12,
                monotonic=clock,
                sleep_fn=clock.sleep,
            )

        # Three probes at 5s each already exceed the 12s bound, so the wait loop must not add more.
        assert clock() <= 12 + _ENDPOINT_POLL_INTERVAL_SEC + 5.0, f"overran the bound: {clock():.0f}s"
