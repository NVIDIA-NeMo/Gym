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
import os
import stat
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.h2_ping_sidecar import launcher
from nemo_gym.h2_ping_sidecar.config import H2PingSidecarConfig, is_loopback_host, parse_duration_seconds
from nemo_gym.h2_ping_sidecar.launcher import (
    H2PingSidecarManager,
    NodeSidecars,
    SidecarError,
    resolve_binary,
    resolve_instances,
    rewrite_base_urls,
    select_ray_nodes,
    sidecar_command,
    sidecar_env,
    start_h2_ping_sidecar,
)


NVCF = "https://abc123.invocation.api.nvcf.nvidia.com"


def _cfg(**kwargs) -> H2PingSidecarConfig:
    return H2PingSidecarConfig.model_validate({"enabled": True, **kwargs})


def _fake_binary(tmp_path: Path, script: str) -> Path:
    path = tmp_path / "h2-ping-sidecar"
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(script))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# A stand-in for the Go binary: writes the ready file named by -ready-file, then idles until SIGTERM.
READY_THEN_IDLE = """
    import signal, sys, time
    args = sys.argv[1:]
    open(args[args.index("-ready-file") + 1], "w").write("1\\n")
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    time.sleep(60)
"""


class TestConfig:
    def test_disabled_by_default(self):
        assert H2PingSidecarConfig().enabled is False

    @pytest.mark.parametrize(
        "text,seconds",
        [("60s", 60), ("15m", 900), ("1m30s", 90), ("1.5s", 1.5), ("250ms", 0.25), ("1h", 3600)],
    )
    def test_parse_duration(self, text, seconds):
        assert parse_duration_seconds(text) == pytest.approx(seconds)

    @pytest.mark.parametrize("text", ["", "60", "s", "1d", "1m 30s", "-5s"])
    def test_parse_duration_rejects(self, text):
        with pytest.raises(ValueError):
            parse_duration_seconds(text)

    def test_numbers_are_seconds(self):
        assert _cfg(ping_interval=30).ping_interval == "30s"

    @pytest.mark.parametrize("interval", ["340s", "6m", "1h", "0s"])
    def test_ping_interval_must_stay_below_global_accelerator_limit(self, interval):
        with pytest.raises(ValidationError, match="ping_interval"):
            _cfg(ping_interval=interval)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            _cfg(memory="1GiB")

    def test_upstream_must_be_https_and_drops_path(self):
        with pytest.raises(ValidationError, match="https"):
            _cfg(instances=[{"upstream": "http://example.com"}])
        instance = _cfg(instances=[{"upstream": f"{NVCF.upper()}/v1/"}]).instances[0]
        assert instance.upstream == NVCF

    @pytest.mark.parametrize("listen", ["1250", "127.0.0.1", "127.0.0.1:0", "127.0.0.1:99999", "127.0.0.1:x"])
    def test_listen_must_be_host_port(self, listen):
        with pytest.raises(ValidationError, match="listen"):
            _cfg(instances=[{"upstream": NVCF, "listen": listen}])

    def test_instances_must_be_distinct(self):
        with pytest.raises(ValidationError, match="unique"):
            _cfg(
                instances=[
                    {"name": "a", "upstream": NVCF},
                    {"name": "a", "upstream": NVCF, "listen": "127.0.0.1:1251"},
                ]
            )
        with pytest.raises(ValidationError, match="different addresses"):
            _cfg(instances=[{"name": "a", "upstream": NVCF}, {"name": "b", "upstream": NVCF}])

    @pytest.mark.parametrize("value", ["1GiB", "512MiB", "1048576", "2TiB"])
    def test_gomemlimit_accepts(self, value):
        assert _cfg(gomemlimit=value).gomemlimit == value

    @pytest.mark.parametrize("value", ["1GB", "lots", "-1", "1 GiB"])
    def test_gomemlimit_rejects(self, value):
        with pytest.raises(ValidationError, match="gomemlimit"):
            _cfg(gomemlimit=value)

    def test_nodes(self):
        assert _cfg().nodes == "local"
        assert _cfg(nodes="all").nodes == "all"
        assert _cfg(nodes=["10.0.0.1"]).nodes == ["10.0.0.1"]
        for bad in ("some", []):
            with pytest.raises(ValidationError):
                _cfg(nodes=bad)

    def test_retry_body_limit(self):
        assert _cfg().retry_body_limit == 16 * 1024 * 1024
        assert _cfg(retry_body_limit=0).retry_body_limit == 0
        with pytest.raises(ValidationError):
            _cfg(retry_body_limit=-1)

    def test_max_conn_age(self):
        assert _cfg().max_conn_age == "50m"
        assert _cfg(max_conn_age="0s").max_conn_age == "0s"
        assert _cfg(max_conn_age=0).max_conn_age == "0s"
        assert _cfg(max_conn_age="59m").max_conn_age == "59m"
        for too_long in ("60m", "1h", "2h", 3600):
            with pytest.raises(ValidationError, match="max_conn_age"):
                _cfg(max_conn_age=too_long)
        with pytest.raises(ValidationError):
            _cfg(max_conn_age="soon")

    def test_gomaxprocs_must_be_positive(self):
        with pytest.raises(ValidationError):
            _cfg(gomaxprocs=0)


class TestFromGlobalConfig:
    def test_missing_block_is_disabled(self):
        assert launcher.h2_ping_sidecar_config_from_global_config(OmegaConf.create({})).enabled is False

    def test_block_with_interpolation(self):
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "sidecar": {"enabled": True, "instances": [{"upstream": "${policy_base_url}"}]},
            }
        )
        config = launcher.h2_ping_sidecar_config_from_global_config(gcd)
        assert config.instances[0].upstream == NVCF


class TestResolveInstances:
    def test_configured(self):
        config = _cfg(instances=[{"name": "judge", "upstream": NVCF}])
        assert resolve_instances(config, OmegaConf.create({}))[0].name == "judge"

    def test_derived_from_policy_base_url(self):
        instances = resolve_instances(_cfg(), OmegaConf.create({"policy_base_url": f"{NVCF}/v1"}))
        assert [(i.name, i.upstream, i.listen) for i in instances] == [("policy", NVCF, "127.0.0.1:1250")]

    def test_nothing_to_derive_from(self):
        with pytest.raises(SidecarError, match="nothing to forward to"):
            resolve_instances(_cfg(), OmegaConf.create({}))

    def test_policy_base_url_not_https(self):
        with pytest.raises(SidecarError, match="https"):
            resolve_instances(_cfg(), OmegaConf.create({"policy_base_url": "http://localhost:8000/v1"}))


class TestResolveBinary:
    def test_configured_binary(self, tmp_path):
        binary = _fake_binary(tmp_path, "pass")
        assert resolve_binary(_cfg(binary=str(binary)), OmegaConf.create({})) == binary

    def test_missing_binary(self, tmp_path):
        with pytest.raises(SidecarError, match="build_if_missing"):
            resolve_binary(_cfg(binary=str(tmp_path / "nope"), build_if_missing=False), OmegaConf.create({}))

    def test_default_path_is_under_cache_dir(self, tmp_path):
        gcd = OmegaConf.create({"cache_dir": str(tmp_path)})
        with pytest.raises(SidecarError, match=str(tmp_path / "h2-ping-sidecar" / "h2-ping-sidecar")):
            resolve_binary(_cfg(build_if_missing=False), gcd)

    def test_builds_by_default_when_missing(self, tmp_path):
        gcd = OmegaConf.create({"cache_dir": str(tmp_path)})
        expected = tmp_path / "h2-ping-sidecar" / "h2-ping-sidecar"

        def fake_build(path):
            path.parent.mkdir(parents=True)
            path.write_text("#!/bin/sh\n")
            path.chmod(0o755)

        assert _cfg().build_if_missing is True
        with patch.object(launcher, "build_sidecar", side_effect=fake_build) as build:
            assert resolve_binary(_cfg(), gcd) == expected
        build.assert_called_once_with(expected)

    def test_existing_binary_is_not_rebuilt(self, tmp_path):
        binary = _fake_binary(tmp_path, "pass")
        with patch.object(launcher, "build_sidecar") as build:
            assert resolve_binary(_cfg(binary=str(binary)), OmegaConf.create({})) == binary
        build.assert_not_called()

    def test_builds_when_missing(self, tmp_path):
        target = tmp_path / "out" / "h2-ping-sidecar"

        def fake_build(path):
            path.parent.mkdir(parents=True)
            path.write_text("#!/bin/sh\n")
            path.chmod(0o755)

        with patch.object(launcher, "build_sidecar", side_effect=fake_build) as build:
            assert resolve_binary(_cfg(binary=str(target), build_if_missing=True), OmegaConf.create({})) == target
        build.assert_called_once_with(target)

    def test_build_without_go(self, tmp_path):
        with patch.object(launcher.shutil, "which", return_value=None):
            with pytest.raises(SidecarError, match="`go` is not on PATH"):
                launcher.build_sidecar(tmp_path / "x")


class TestCommandAndEnv:
    def test_command(self):
        config = _cfg(ping_interval="45s", ping_timeout="10s", shutdown_grace="2m", insecure_skip_verify=True)
        instance = config.model_copy().instances or [launcher.SidecarInstanceConfig(upstream=NVCF)]
        command = sidecar_command("/bin/sidecar", config, instance[0], "/tmp/ready")
        assert command == [
            "/bin/sidecar",
            "-listen", "127.0.0.1:1250",
            "-upstream", NVCF,
            "-ping-interval", "45s",
            "-ping-timeout", "10s",
            "-shutdown-grace", "2m",
            "-retry-body-limit", "16777216",
            "-max-conn-age", "50m",
            "-ready-file", "/tmp/ready",
            "-insecure-skip-verify",
        ]  # fmt: skip

    def test_env_carries_go_runtime_limits(self, monkeypatch):
        monkeypatch.delenv("GOMEMLIMIT", raising=False)
        monkeypatch.delenv("GOMAXPROCS", raising=False)
        env = sidecar_env(_cfg(gomemlimit="1GiB", gomaxprocs=2))
        assert (env["GOMEMLIMIT"], env["GOMAXPROCS"]) == ("1GiB", "2")
        env = sidecar_env(_cfg())
        assert "GOMEMLIMIT" not in env and "GOMAXPROCS" not in env


class TestRewriteBaseUrls:
    def _instances(self):
        return [launcher.SidecarInstanceConfig(name="nvcf", upstream=NVCF)]

    def test_rewrites_policy_and_inline_urls_and_leaves_the_rest(self):
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "policy_model": {
                    "responses_api_models": {
                        "openai_model": {"openai_base_url": "${policy_base_url}", "entrypoint": "app.py"}
                    }
                },
                "judge": {
                    "responses_api_models": {
                        "genrm_model": {"base_url": [f"{NVCF}/v1", "https://other.example.com/v1"]},
                        "inline": {"openai_base_url": f"{NVCF}/v1?x=1"},
                        "direct": {"openai_base_url": "https://integrate.api.nvidia.com/v1"},
                    }
                },
                "sidecar": {"enabled": True, "instances": [{"upstream": f"{NVCF}"}]},
            }
        )
        changes = rewrite_base_urls(gcd, self._instances())

        assert gcd.policy_base_url == "http://127.0.0.1:1250/v1"
        # An interpolation follows the key it points at instead of being overwritten.
        raw = OmegaConf.to_container(gcd)
        assert raw["policy_model"]["responses_api_models"]["openai_model"]["openai_base_url"] == "${policy_base_url}"
        assert gcd.policy_model.responses_api_models.openai_model.openai_base_url == "http://127.0.0.1:1250/v1"
        assert list(gcd.judge.responses_api_models.genrm_model.base_url) == [
            "http://127.0.0.1:1250/v1",
            "https://other.example.com/v1",
        ]
        assert gcd.judge.responses_api_models.inline.openai_base_url == "http://127.0.0.1:1250/v1?x=1"
        assert gcd.judge.responses_api_models.direct.openai_base_url == "https://integrate.api.nvidia.com/v1"
        assert len(changes) == 3

    def test_no_match_changes_nothing(self):
        gcd = OmegaConf.create({"policy_base_url": "https://integrate.api.nvidia.com/v1"})
        assert rewrite_base_urls(gcd, self._instances()) == []
        assert gcd.policy_base_url == "https://integrate.api.nvidia.com/v1"

    def test_host_match_is_case_insensitive_and_exact(self):
        gcd = OmegaConf.create({"policy_base_url": f"{NVCF.upper().replace('HTTPS', 'https')}/v1"})
        assert len(rewrite_base_urls(gcd, self._instances())) == 1
        gcd = OmegaConf.create({"policy_base_url": f"{NVCF}.evil.example/v1"})
        assert rewrite_base_urls(gcd, self._instances()) == []


class TestNodeSidecars:
    def _start(self, tmp_path, script, **config):
        binary = _fake_binary(tmp_path, script)
        node = NodeSidecars(
            _cfg(startup_timeout_seconds=5, shutdown_grace="1s", **config),
            [launcher.SidecarInstanceConfig(name="nvcf", upstream=NVCF)],
            str(binary),
            str(tmp_path / "logs"),
        )
        return node

    def test_start_check_stop(self, tmp_path):
        node = self._start(tmp_path, READY_THEN_IDLE)
        (summary,) = node.start()
        assert "nvcf listening on 127.0.0.1:1250" in summary
        assert node.failures() == []
        ready_files = list((tmp_path / "logs").glob("h2ping-nvcf-*.ready"))
        assert len(ready_files) == 1
        node.stop()
        assert not ready_files[0].exists()
        assert (tmp_path / "logs").glob("h2ping-nvcf-*.log")

    def test_exit_before_ready_reports_log_tail(self, tmp_path):
        node = self._start(
            tmp_path, "print('cannot bind 127.0.0.1:1250: address already in use'); raise SystemExit(1)"
        )
        with pytest.raises(SidecarError, match="already in use"):
            node.start()

    def test_stale_ready_file_is_not_readiness(self, tmp_path):
        node = self._start(tmp_path, "import time; time.sleep(60)")
        node._config = node._config.model_copy(update={"startup_timeout_seconds": 0.5})
        logs = tmp_path / "logs"
        logs.mkdir()
        stale = Path(node._ready_path(node._instances[0]))
        stale.write_text("999\n")
        with pytest.raises(SidecarError, match="not ready after"):
            node.start()
        assert not stale.exists()

    def test_owners_never_share_a_ready_marker(self, tmp_path):
        """Concurrent runs with the same instance names in one log dir must not touch each other's marker."""
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)

        def owner(port):
            return NodeSidecars(
                _cfg(startup_timeout_seconds=5, shutdown_grace="1s"),
                [launcher.SidecarInstanceConfig(name="nvcf", upstream=NVCF, listen=f"127.0.0.1:{port}")],
                str(binary),
                str(tmp_path / "logs"),
            )

        a, b = owner(1250), owner(1251)
        assert a._ready_path(a._instances[0]) != b._ready_path(b._instances[0])

        (tmp_path / "logs").mkdir()
        a._spawn(a._instances[0])  # A has bound and written its marker, but is not waiting on it yet
        b.start()
        b.stop()  # B's cleanup runs while A is between spawn and wait
        try:
            a._wait_ready(a._instances[0])
            assert Path(a._ready_path(a._instances[0])).exists()
        finally:
            a.stop()

    def test_crash_after_start_is_reported(self, tmp_path):
        node = self._start(
            tmp_path,
            """
            import sys, time
            args = sys.argv[1:]
            open(args[args.index("-ready-file") + 1], "w").write("1\\n")
            time.sleep(0.3)
            raise SystemExit(3)
            """,
        )
        node.start()
        deadline = __import__("time").monotonic() + 5
        while not node.failures() and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.05)
        (message,) = node.failures()
        assert "exited with code 3" in message
        node.stop()

    def test_non_executable_binary(self, tmp_path):
        node = self._start(tmp_path, "pass")
        os.chmod(node._binary, 0o644)
        with pytest.raises(SidecarError, match="same path on every node"):
            node.start()

    def test_go_limits_reach_the_process(self, tmp_path):
        out = tmp_path / "env.txt"
        script = f"""
            import os, sys
            args = sys.argv[1:]
            open({str(out)!r}, "w").write(os.environ["GOMEMLIMIT"] + " " + os.environ["GOMAXPROCS"])
            open(args[args.index("-ready-file") + 1], "w").write("1\\n")
            import time; time.sleep(60)
        """
        node = self._start(tmp_path, script, gomemlimit="256MiB", gomaxprocs=2)
        node.start()
        node.stop()
        assert out.read_text() == "256MiB 2"


class TestSelectRayNodes:
    NODES = [
        {"NodeID": "a" * 56, "NodeManagerAddress": "10.0.0.1", "Alive": True},
        {"NodeID": "b" * 56, "NodeManagerAddress": "10.0.0.2", "Alive": True},
    ]

    def test_all(self):
        assert select_ray_nodes(_cfg(nodes="all"), self.NODES) == self.NODES

    def test_all_with_no_nodes(self):
        with pytest.raises(SidecarError, match="no alive Ray nodes"):
            select_ray_nodes(_cfg(nodes="all"), [])

    def test_by_ip(self):
        assert select_ray_nodes(_cfg(nodes=["10.0.0.2"]), self.NODES) == [self.NODES[1]]

    def test_unknown_ip(self):
        with pytest.raises(SidecarError, match="10.9.9.9"):
            select_ray_nodes(_cfg(nodes=["10.9.9.9"]), self.NODES)


class TestManager:
    def test_local_start_check_stop(self, tmp_path):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        config = _cfg(startup_timeout_seconds=5, shutdown_grace="1s")
        manager = H2PingSidecarManager(
            config, [launcher.SidecarInstanceConfig(name="nvcf", upstream=NVCF)], binary, str(tmp_path / "logs")
        )
        manager.start()
        manager.check()
        manager.stop()
        manager.stop()  # idempotent

    def test_failed_start_cleans_up(self, tmp_path):
        binary = _fake_binary(tmp_path, "raise SystemExit(1)")
        manager = H2PingSidecarManager(
            _cfg(startup_timeout_seconds=5),
            [launcher.SidecarInstanceConfig(name="nvcf", upstream=NVCF)],
            binary,
            str(tmp_path / "logs"),
        )
        with pytest.raises(SidecarError):
            manager.start()
        assert manager._local is None

    def test_ray_nodes_need_ray(self, tmp_path):
        ray = MagicMock()
        ray.is_initialized.return_value = False
        manager = H2PingSidecarManager(
            _cfg(nodes="all"), [launcher.SidecarInstanceConfig(upstream=NVCF)], Path("/x"), "/l"
        )
        with patch.object(launcher, "_get_ray", return_value=ray):
            with pytest.raises(SidecarError, match="Ray is not initialized"):
                manager.start()

    def test_one_actor_per_selected_node(self):
        ray = MagicMock()
        ray.is_initialized.return_value = True
        ray.nodes.return_value = [
            {"NodeID": "a" * 56, "NodeManagerAddress": "10.0.0.1", "Alive": True},
            {"NodeID": "b" * 56, "NodeManagerAddress": "10.0.0.2", "Alive": True},
            {"NodeID": "c" * 56, "NodeManagerAddress": "10.0.0.3", "Alive": False},
        ]
        ray.get.return_value = ["10.0.0.1: nvcf listening"]
        manager = H2PingSidecarManager(
            _cfg(nodes="all"), [launcher.SidecarInstanceConfig(upstream=NVCF)], Path("/x"), "/l"
        )
        with patch.object(launcher, "_get_ray", return_value=ray):
            manager.start()
            assert sorted(manager._actors) == ["10.0.0.1", "10.0.0.2"]
            manager.stop()
        assert ray.kill.call_count == 2
        assert manager._actors == {}

    def test_actor_failure_aborts_start(self):
        ray = MagicMock()
        ray.is_initialized.return_value = True
        ray.nodes.return_value = [{"NodeID": "a" * 56, "NodeManagerAddress": "10.0.0.1", "Alive": True}]
        ray.get.side_effect = RuntimeError("binary not executable here")
        manager = H2PingSidecarManager(
            _cfg(nodes="all"), [launcher.SidecarInstanceConfig(upstream=NVCF)], Path("/x"), "/l"
        )
        with patch.object(launcher, "_get_ray", return_value=ray):
            with pytest.raises(SidecarError, match="10.0.0.1: binary not executable here"):
                manager.start()


class TestStartH2PingSidecar:
    def test_disabled_is_a_no_op(self):
        gcd = OmegaConf.create({"policy_base_url": f"{NVCF}/v1"})
        assert start_h2_ping_sidecar(gcd) is None
        assert gcd.policy_base_url == f"{NVCF}/v1"

        gcd = OmegaConf.create({"policy_base_url": f"{NVCF}/v1", "sidecar": {"enabled": False}})
        assert start_h2_ping_sidecar(gcd) is None
        assert gcd.policy_base_url == f"{NVCF}/v1"

    def test_enabled_starts_and_rewrites(self, tmp_path):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "sidecar": {
                    "enabled": True,
                    "binary": str(binary),
                    "log_dir": str(tmp_path / "logs"),
                    "shutdown_grace": "1s",
                },
            }
        )
        manager = start_h2_ping_sidecar(gcd)
        try:
            assert gcd.policy_base_url == "http://127.0.0.1:1250/v1"
            manager.check()
        finally:
            manager.stop()

    def test_rewrite_can_be_disabled(self, tmp_path):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "sidecar": {
                    "enabled": True,
                    "binary": str(binary),
                    "log_dir": str(tmp_path / "logs"),
                    "shutdown_grace": "1s",
                    "rewrite_base_urls": False,
                },
            }
        )
        manager = start_h2_ping_sidecar(gcd)
        try:
            assert gcd.policy_base_url == f"{NVCF}/v1"
        finally:
            manager.stop()

    def test_failure_leaves_urls_untouched(self, tmp_path):
        binary = _fake_binary(tmp_path, "raise SystemExit(1)")
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "sidecar": {"enabled": True, "binary": str(binary), "log_dir": str(tmp_path / "logs")},
            }
        )
        with pytest.raises(SidecarError):
            start_h2_ping_sidecar(gcd)
        assert gcd.policy_base_url == f"{NVCF}/v1"

    def test_block_is_not_a_server_instance(self):
        from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS

        assert "sidecar" in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


class TestResolverBackedUrls:
    """URLs that come from a resolver are rewritten too; aliases stay aliases."""

    def _instances(self):
        return [launcher.SidecarInstanceConfig(name="nvcf", upstream=NVCF)]

    def test_env_backed_policy_url_routes_policy_and_its_alias(self, monkeypatch):
        monkeypatch.setenv("POLICY_URL", f"{NVCF}/v1")
        gcd = OmegaConf.create(
            {
                "policy_base_url": "${oc.env:POLICY_URL}",
                "policy_model": {"responses_api_models": {"openai_model": {"openai_base_url": "${policy_base_url}"}}},
            }
        )

        changes = rewrite_base_urls(gcd, self._instances())

        assert gcd.policy_base_url == "http://127.0.0.1:1250/v1"
        assert gcd.policy_model.responses_api_models.openai_model.openai_base_url == "http://127.0.0.1:1250/v1"
        raw = OmegaConf.to_container(gcd)
        assert raw["policy_model"]["responses_api_models"]["openai_model"]["openai_base_url"] == "${policy_base_url}"
        assert len(changes) == 1

    def test_env_backed_judge_scalar(self, monkeypatch):
        monkeypatch.setenv("JUDGE_URL", f"{NVCF}/v1")
        gcd = OmegaConf.create({"judge": {"responses_api_models": {"m": {"openai_base_url": "${oc.env:JUDGE_URL}"}}}})
        changes = rewrite_base_urls(gcd, self._instances())
        assert gcd.judge.responses_api_models.m.openai_base_url == "http://127.0.0.1:1250/v1"
        assert [c.location for c in changes] == ["judge.responses_api_models.m.openai_base_url"]

    def test_list_alias_rewrites_a_copy_and_leaves_the_pool_alone(self):
        pool = [f"{NVCF}/v1", "https://other.example/v1"]
        gcd = OmegaConf.create({"pool": pool, "judge": {"responses_api_models": {"m": {"base_url": "${pool}"}}}})

        changes = rewrite_base_urls(gcd, self._instances())

        assert list(gcd.judge.responses_api_models.m.base_url) == [
            "http://127.0.0.1:1250/v1",
            "https://other.example/v1",
        ]
        assert list(gcd.pool) == pool
        assert len(changes) == 1

    def test_list_alias_with_no_match_stays_an_alias(self):
        gcd = OmegaConf.create(
            {"pool": ["https://other.example/v1"], "judge": {"responses_api_models": {"m": {"base_url": "${pool}"}}}}
        )
        assert rewrite_base_urls(gcd, self._instances()) == []
        assert OmegaConf.to_container(gcd)["judge"]["responses_api_models"]["m"]["base_url"] == "${pool}"

    def test_element_alias_inside_a_literal_list_is_kept(self):
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "judge": {"responses_api_models": {"m": {"base_url": ["${policy_base_url}"]}}},
            }
        )
        rewrite_base_urls(gcd, self._instances())
        assert OmegaConf.to_container(gcd)["judge"]["responses_api_models"]["m"]["base_url"] == ["${policy_base_url}"]
        assert list(gcd.judge.responses_api_models.m.base_url) == ["http://127.0.0.1:1250/v1"]

    def test_unresolvable_value_is_left_for_the_server_to_report(self, monkeypatch):
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        gcd = OmegaConf.create({"policy_base_url": "${oc.env:NOT_SET_ANYWHERE}"})
        assert rewrite_base_urls(gcd, self._instances()) == []

    def test_malformed_url_names_its_location(self):
        gcd = OmegaConf.create({"judge": {"responses_api_models": {"m": {"openai_base_url": "https://[invalid/v1"}}}})
        with pytest.raises(SidecarError, match=r"judge\.responses_api_models\.m\.openai_base_url"):
            rewrite_base_urls(gcd, self._instances())


class TestStartIsTransactional:
    def _gcd(self, tmp_path, binary, judge_url=None):
        gcd = {
            "policy_base_url": f"{NVCF}/v1",
            "sidecar": {
                "enabled": True,
                "binary": str(binary),
                "log_dir": str(tmp_path / "logs"),
                "shutdown_grace": "1s",
            },
        }
        if judge_url is not None:
            gcd["judge"] = {"responses_api_models": {"m": {"openai_base_url": judge_url}}}
        return OmegaConf.create(gcd)

    def test_malformed_url_fails_before_any_process_starts(self, tmp_path):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        gcd = self._gcd(tmp_path, binary, judge_url="https://[invalid/v1")

        with patch.object(H2PingSidecarManager, "start") as start:
            with pytest.raises(SidecarError, match="not a valid URL"):
                start_h2_ping_sidecar(gcd)

        start.assert_not_called()
        assert gcd.policy_base_url == f"{NVCF}/v1"
        assert not (tmp_path / "logs").exists()

    def test_failure_after_start_stops_the_proxy(self, tmp_path):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        gcd = self._gcd(tmp_path, binary)
        real = launcher.rewrite_base_urls
        calls = []

        def flaky(config, instances):
            calls.append(1)
            if len(calls) == 1:  # the dry run before anything starts
                return real(config, instances)
            raise RuntimeError("boom after start")

        with patch.object(launcher, "rewrite_base_urls", side_effect=flaky):
            with patch.object(
                H2PingSidecarManager, "stop", autospec=True, side_effect=H2PingSidecarManager.stop
            ) as stop:
                with pytest.raises(RuntimeError, match="boom after start"):
                    start_h2_ping_sidecar(gcd)

        stop.assert_called_once()
        assert list((tmp_path / "logs").glob("*.ready")) == []

    def test_interrupt_after_start_stops_the_proxy(self, tmp_path):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        gcd = self._gcd(tmp_path, binary)
        real = launcher.rewrite_base_urls
        calls = []

        def interrupted(config, instances):
            calls.append(1)
            if len(calls) == 1:
                return real(config, instances)
            raise KeyboardInterrupt

        with patch.object(launcher, "rewrite_base_urls", side_effect=interrupted):
            with pytest.raises(KeyboardInterrupt):
                start_h2_ping_sidecar(gcd)

        assert list((tmp_path / "logs").glob("*.ready")) == []


class TestLoopbackWarning:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.1.2.3", "localhost", "LOCALHOST", "::1", "[::1]"])
    def test_loopback(self, host):
        assert is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "example.com", "", "192.168.1.2"])
    def test_not_loopback(self, host):
        assert not is_loopback_host(host)

    def test_non_loopback_listen_is_accepted_but_warned_about(self, tmp_path, capsys):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "sidecar": {
                    "enabled": True,
                    "binary": str(binary),
                    "log_dir": str(tmp_path / "logs"),
                    "shutdown_grace": "1s",
                    "instances": [{"name": "nvcf", "upstream": NVCF, "listen": "0.0.0.0:1250"}],
                },
            }
        )
        manager = start_h2_ping_sidecar(gcd)
        try:
            assert "not a loopback address" in capsys.readouterr().out
        finally:
            manager.stop()

    def test_loopback_listen_is_not_warned_about(self, tmp_path, capsys):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "sidecar": {
                    "enabled": True,
                    "binary": str(binary),
                    "log_dir": str(tmp_path / "logs"),
                    "shutdown_grace": "1s",
                },
            }
        )
        manager = start_h2_ping_sidecar(gcd)
        try:
            assert "not a loopback address" not in capsys.readouterr().out
        finally:
            manager.stop()
