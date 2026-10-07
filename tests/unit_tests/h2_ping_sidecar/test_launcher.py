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
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.h2_ping_sidecar import launcher
from nemo_gym.h2_ping_sidecar.config import H2PingSidecarConfig, is_loopback_host, parse_duration_seconds
from nemo_gym.h2_ping_sidecar.launcher import (
    H2PingSidecarManager,
    NodeSidecars,
    SidecarError,
    build_sidecar,
    resolve_binary,
    rewrite_base_urls,
    sidecar_command,
    sidecar_env,
    start_h2_ping_sidecar,
)


NVCF = "https://abc123.invocation.api.nvcf.nvidia.com"
INSTANCE = {"name": "nvcf", "upstream": NVCF}


def _cfg(**kwargs) -> H2PingSidecarConfig:
    return H2PingSidecarConfig.model_validate({"enabled": True, "instances": [INSTANCE], **kwargs})


def _instance(**kwargs) -> launcher.SidecarInstanceConfig:
    return launcher.SidecarInstanceConfig(**{**INSTANCE, **kwargs})


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


def _fake_go(tmp_path: Path, build_script: str, version: str = "go1.25.0") -> Path:
    """A `go` on PATH: `go env GOVERSION` prints a version, `go build -o X .` runs `build_script` with X."""
    path = tmp_path / "fakebin" / "go"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "env" ]; then echo {version}; exit 0; fi\n'
        'while [ "$#" -gt 0 ]; do if [ "$1" = "-o" ]; then OUT="$2"; fi; shift; done\n' + textwrap.dedent(build_script)
    )
    path.chmod(0o755)
    return path


GOOD_BUILD = 'printf "#!/bin/sh\\nexit 0\\n" > "$OUT"; chmod 755 "$OUT"\n'


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

    def test_nodes_is_not_a_setting(self):
        """The sidecar always runs next to the servers RunHelper starts."""
        with pytest.raises(ValidationError):
            _cfg(nodes="all")

    def test_instances_are_required_when_enabled(self):
        with pytest.raises(ValidationError, match="instances is required"):
            H2PingSidecarConfig.model_validate({"enabled": True})
        # Nothing is needed while the sidecar is off.
        assert H2PingSidecarConfig.model_validate({"enabled": False}).instances == []

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
                "judge_url": f"{NVCF}/v1",
                "sidecar": {"enabled": True, "instances": [{"upstream": "${judge_url}"}]},
            }
        )
        config = launcher.h2_ping_sidecar_config_from_global_config(gcd)
        assert config.instances[0].upstream == NVCF


class TestResolveBinary:
    def test_configured_binary(self, tmp_path):
        binary = _fake_binary(tmp_path, "pass")
        assert resolve_binary(_cfg(binary=str(binary)), OmegaConf.create({})) == binary

    def test_missing_binary(self, tmp_path):
        with pytest.raises(SidecarError, match="build_if_missing"):
            resolve_binary(_cfg(binary=str(tmp_path / "nope"), build_if_missing=False), OmegaConf.create({}))

    def test_build_without_go(self, tmp_path):
        with patch.object(launcher.shutil, "which", return_value=None):
            with pytest.raises(SidecarError, match="`go` is not on PATH"):
                launcher.build_sidecar(tmp_path / "x")

    def test_builds_when_missing_and_reuses(self, tmp_path):
        go = _fake_go(tmp_path, GOOD_BUILD)
        gcd = OmegaConf.create({"cache_dir": str(tmp_path / "cache")})
        with patch.object(launcher.shutil, "which", return_value=str(go)):
            path = resolve_binary(_cfg(), gcd)
            assert path.is_file() and os.access(path, os.X_OK)
            assert path.parent.parent.parent == (tmp_path / "cache" / "h2-ping-sidecar").resolve()
            mtime = path.stat().st_mtime_ns
            assert resolve_binary(_cfg(), gcd) == path
        assert path.stat().st_mtime_ns == mtime  # not rebuilt

    def test_build_can_be_disabled(self, tmp_path):
        gcd = OmegaConf.create({"cache_dir": str(tmp_path / "cache")})
        with patch.object(launcher.shutil, "which", return_value=None):
            with pytest.raises(SidecarError, match="does not exist"):
                resolve_binary(_cfg(build_if_missing=False), gcd)

    def test_failed_build_leaves_nothing_behind(self, tmp_path):
        go = _fake_go(tmp_path, 'echo "boom" >&2; : > "$OUT"; exit 1\n')
        target = tmp_path / "out" / "h2-ping-sidecar"
        with patch.object(launcher.shutil, "which", return_value=str(go)):
            with pytest.raises(SidecarError, match="boom"):
                build_sidecar(target)
        assert list((tmp_path / "out").iterdir()) == []

    def test_concurrent_builds_all_succeed_and_leave_no_partial_file(self, tmp_path):
        go = _fake_go(tmp_path, "sleep 0.2\n" + GOOD_BUILD)
        target = tmp_path / "out" / "h2-ping-sidecar"
        errors = []

        def build():
            try:
                build_sidecar(target)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        with patch.object(launcher.shutil, "which", return_value=str(go)):
            threads = [threading.Thread(target=build) for _ in range(4)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        assert errors == []
        assert [p.name for p in (tmp_path / "out").iterdir()] == ["h2-ping-sidecar"]
        assert os.access(target, os.X_OK)


class TestBinaryCacheKey:
    """A cached binary is only reused for the same sources, and for the same Go release when Go is present."""

    def _sources(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.go").write_text("package main\n")
        (src / "rotate.go").write_text("package main\n")
        (src / "go.mod").write_text("module x\n\ngo 1.25\n")
        (src / "main_test.go").write_text("package main\n")
        return src

    def test_digest_changes_with_sources_but_not_tests(self, tmp_path):
        src = self._sources(tmp_path)
        with patch.object(launcher, "SIDECAR_SOURCE_DIR", src):
            base = launcher._source_digest()
            assert launcher._source_digest() == base
            assert len(base) == 12 and set(base) <= set("0123456789abcdef")

            (src / "main_test.go").write_text("package main // changed\n")
            assert launcher._source_digest() == base  # tests are not part of the binary

            (src / "main.go").write_text("package main // changed\n")
            changed = launcher._source_digest()
            assert changed != base
            (src / "go.mod").write_text("module x\n\ngo 1.26\n")
            assert launcher._source_digest() != changed

    def test_binary_built_from_older_sources_is_not_reused(self, tmp_path):
        src = self._sources(tmp_path)
        go = _fake_go(tmp_path, GOOD_BUILD)
        gcd = OmegaConf.create({"cache_dir": str(tmp_path / "cache")})
        with (
            patch.object(launcher, "SIDECAR_SOURCE_DIR", src),
            patch.object(launcher.shutil, "which", return_value=str(go)),
        ):
            old = resolve_binary(_cfg(), gcd)
            (src / "main.go").write_text("package main // a later fix\n")
            new = resolve_binary(_cfg(), gcd)
        assert new != old
        assert old.is_file() and new.is_file()

    def test_a_go_upgrade_rebuilds(self, tmp_path):
        src = self._sources(tmp_path)
        gcd = OmegaConf.create({"cache_dir": str(tmp_path / "cache")})
        paths = []
        for version in ("go1.25.0", "go1.25.1"):
            go = _fake_go(tmp_path, GOOD_BUILD, version=version)
            with (
                patch.object(launcher, "SIDECAR_SOURCE_DIR", src),
                patch.object(launcher.shutil, "which", return_value=str(go)),
            ):
                paths.append(resolve_binary(_cfg(), gcd))
        assert paths[0] != paths[1]
        assert paths[0].parent.parent == paths[1].parent.parent  # same sources, different Go
        assert [p.parent.name for p in paths] == ["go1.25.0", "go1.25.1"]
        assert all(p.is_file() for p in paths)

    def test_without_go_the_newest_binary_for_these_sources_is_used(self, tmp_path):
        """A node with no Go can run a binary that was built (by anyone) from the current sources."""
        src = self._sources(tmp_path)
        gcd = OmegaConf.create({"cache_dir": str(tmp_path / "cache")})
        built = []
        for index, version in enumerate(("go1.25.0", "go1.25.1")):
            go = _fake_go(tmp_path, GOOD_BUILD, version=version)
            with (
                patch.object(launcher, "SIDECAR_SOURCE_DIR", src),
                patch.object(launcher.shutil, "which", return_value=str(go)),
            ):
                path = resolve_binary(_cfg(), gcd)
            os.utime(path, (1_000_000 + index, 1_000_000 + index))
            built.append(path)

        with (
            patch.object(launcher, "SIDECAR_SOURCE_DIR", src),
            patch.object(launcher.shutil, "which", return_value=None),
        ):
            assert resolve_binary(_cfg(), gcd) == built[-1]

            # Sources that nobody built have no binary to fall back on.
            (src / "main.go").write_text("package main // not built by anyone\n")
            with pytest.raises(SidecarError, match="`go` is not on PATH"):
                resolve_binary(_cfg(), gcd)

    def test_missing_binary_message_names_the_searched_path(self, tmp_path):
        gcd = OmegaConf.create({"cache_dir": str(tmp_path / "cache")})
        with patch.object(launcher.shutil, "which", return_value=None):
            with pytest.raises(SidecarError) as excinfo:
                resolve_binary(_cfg(), gcd)
        assert str(tmp_path / "cache" / "h2-ping-sidecar") in str(excinfo.value)


class TestCommandAndEnv:
    def test_command(self):
        config = _cfg(ping_interval="45s", ping_timeout="10s", shutdown_grace="2m", insecure_skip_verify=True)
        command = sidecar_command("/bin/sidecar", config, config.instances[0], "/tmp/ready")
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
        return [_instance()]

    def test_rewrites_model_urls_and_leaves_the_rest(self):
        gcd = OmegaConf.create(
            {
                "policy_base_url": "https://policy.example.com/v1",
                "judge": {
                    "responses_api_models": {
                        "genrm_model": {"base_url": [f"{NVCF}/v1", "https://other.example.com/v1"]},
                        "inline": {"openai_base_url": f"{NVCF}/v1?x=1"},
                        "direct": {"openai_base_url": "https://integrate.api.nvidia.com/v1"},
                        "local": {"base_url": ["http://127.0.0.1:8000/v1"]},
                    }
                },
                "sidecar": {"enabled": True, "instances": [{"upstream": NVCF}]},
            }
        )
        changes = rewrite_base_urls(gcd, self._instances())

        servers = gcd.judge.responses_api_models
        assert list(servers.genrm_model.base_url) == ["http://127.0.0.1:1250/v1", "https://other.example.com/v1"]
        assert servers.inline.openai_base_url == "http://127.0.0.1:1250/v1?x=1"
        assert servers.direct.openai_base_url == "https://integrate.api.nvidia.com/v1"
        assert list(servers["local"].base_url) == ["http://127.0.0.1:8000/v1"]
        assert gcd.policy_base_url == "https://policy.example.com/v1"
        assert len(changes) == 2

    def test_top_level_policy_base_url_is_never_rewritten(self):
        """Only model-server fields are routed; the top-level key stays as the user wrote it."""
        gcd = OmegaConf.create({"policy_base_url": f"{NVCF}/v1"})
        assert rewrite_base_urls(gcd, self._instances()) == []
        assert gcd.policy_base_url == f"{NVCF}/v1"

    def test_alias_of_the_top_level_url_is_routed_for_that_field_only(self):
        gcd = OmegaConf.create(
            {
                "policy_base_url": f"{NVCF}/v1",
                "policy_model": {"responses_api_models": {"openai_model": {"openai_base_url": "${policy_base_url}"}}},
            }
        )
        changes = rewrite_base_urls(gcd, self._instances())
        assert gcd.policy_model.responses_api_models.openai_model.openai_base_url == "http://127.0.0.1:1250/v1"
        assert gcd.policy_base_url == f"{NVCF}/v1"
        assert len(changes) == 1

    def test_no_match_changes_nothing(self):
        gcd = OmegaConf.create({"m": {"responses_api_models": {"s": {"openai_base_url": "https://x.example.com/v1"}}}})
        assert rewrite_base_urls(gcd, self._instances()) == []
        assert gcd.m.responses_api_models.s.openai_base_url == "https://x.example.com/v1"

    def test_host_match_is_case_insensitive_and_exact(self):
        upper = NVCF.upper().replace("HTTPS", "https")
        gcd = OmegaConf.create({"m": {"responses_api_models": {"s": {"openai_base_url": f"{upper}/v1"}}}})
        assert len(rewrite_base_urls(gcd, self._instances())) == 1
        gcd = OmegaConf.create({"m": {"responses_api_models": {"s": {"openai_base_url": f"{NVCF}.evil.example/v1"}}}})
        assert rewrite_base_urls(gcd, self._instances()) == []


class TestResolverBackedUrls:
    """URLs that come from a resolver are rewritten too."""

    def _instances(self):
        return [_instance()]

    def test_env_backed_judge_url(self, monkeypatch):
        monkeypatch.setenv("JUDGE_URL", f"{NVCF}/v1")
        gcd = OmegaConf.create({"judge": {"responses_api_models": {"m": {"base_url": ["${oc.env:JUDGE_URL}"]}}}})
        changes = rewrite_base_urls(gcd, self._instances())
        assert list(gcd.judge.responses_api_models.m.base_url) == ["http://127.0.0.1:1250/v1"]
        assert [c.location for c in changes] == ["judge.responses_api_models.m.base_url[0]"]

    def test_env_backed_scalar(self, monkeypatch):
        monkeypatch.setenv("JUDGE_URL", f"{NVCF}/v1")
        gcd = OmegaConf.create({"judge": {"responses_api_models": {"m": {"openai_base_url": "${oc.env:JUDGE_URL}"}}}})
        rewrite_base_urls(gcd, self._instances())
        assert gcd.judge.responses_api_models.m.openai_base_url == "http://127.0.0.1:1250/v1"

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

    def test_unresolvable_value_is_left_for_the_server_to_report(self, monkeypatch):
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        gcd = OmegaConf.create(
            {
                "policy_base_url": "${oc.env:NOT_SET_ANYWHERE}",
                "j": {"responses_api_models": {"m": {"base_url": "${oc.env:NOT_SET_ANYWHERE}"}}},
            }
        )
        assert rewrite_base_urls(gcd, self._instances()) == []

    def test_malformed_url_names_its_location(self):
        gcd = OmegaConf.create({"judge": {"responses_api_models": {"m": {"openai_base_url": "https://[invalid/v1"}}}})
        with pytest.raises(SidecarError, match=r"judge\.responses_api_models\.m\.openai_base_url"):
            rewrite_base_urls(gcd, self._instances())


class TestNodeSidecars:
    def _node(self, tmp_path, script, **config):
        binary = _fake_binary(tmp_path, script)
        return NodeSidecars(
            _cfg(startup_timeout_seconds=5, shutdown_grace="1s", **config),
            [_instance()],
            str(binary),
            str(tmp_path / "logs"),
        )

    def test_start_check_stop(self, tmp_path):
        node = self._node(tmp_path, READY_THEN_IDLE)
        (summary,) = node.start()
        assert "nvcf listening on 127.0.0.1:1250" in summary
        assert node.failures() == []
        ready_files = list((tmp_path / "logs").glob("h2ping-nvcf-*.ready"))
        assert len(ready_files) == 1
        node.stop()
        assert not ready_files[0].exists()
        assert list((tmp_path / "logs").glob("h2ping-nvcf-*.log"))

    def test_exit_before_ready_reports_log_tail_and_a_busy_port_hint(self, tmp_path):
        node = self._node(tmp_path, "print('cannot bind 127.0.0.1:1250: address already in use'); raise SystemExit(1)")
        with pytest.raises(SidecarError, match=r"already in use.*\n.*cannot bind"):
            node.start()

    def test_other_exits_do_not_blame_the_port(self, tmp_path):
        node = self._node(tmp_path, "print('flag provided but not defined: -max-conn-age'); raise SystemExit(2)")
        with pytest.raises(SidecarError) as excinfo:
            node.start()
        message = str(excinfo.value)
        assert "exited with code 2" in message and "flag provided but not defined" in message
        assert "already in use" not in message

    def test_stale_ready_file_is_not_readiness(self, tmp_path):
        node = self._node(tmp_path, "import time; time.sleep(60)")
        node._config = node._config.model_copy(update={"startup_timeout_seconds": 0.5})
        (tmp_path / "logs").mkdir()
        stale = Path(node._ready_path(node._instances[0]))
        stale.write_text("999\n")
        with pytest.raises(SidecarError, match="not ready after"):
            node.start()
        assert not stale.exists()

    def test_owners_never_share_a_ready_marker_or_a_log(self, tmp_path):
        """Concurrent runs with the same instance names in one log dir must not touch each other's files."""
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)

        def owner(port):
            return NodeSidecars(
                _cfg(startup_timeout_seconds=5, shutdown_grace="1s"),
                [_instance(listen=f"127.0.0.1:{port}")],
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
            assert a._log_paths["nvcf"] != b._log_paths["nvcf"]
        finally:
            a.stop()

    def test_crash_after_start_is_reported(self, tmp_path):
        node = self._node(
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
        node = self._node(tmp_path, "pass")
        os.chmod(node._binary, 0o644)
        with pytest.raises(SidecarError, match="not executable"):
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
        node = self._node(tmp_path, script, gomemlimit="256MiB", gomaxprocs=2)
        node.start()
        node.stop()
        assert out.read_text() == "256MiB 2"


class TestManager:
    def test_start_check_stop(self, tmp_path):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        manager = H2PingSidecarManager(
            _cfg(startup_timeout_seconds=5, shutdown_grace="1s"), [_instance()], binary, str(tmp_path / "logs")
        )
        manager.start()
        manager.check()
        manager.stop()
        manager.stop()  # idempotent

    def test_failed_start_cleans_up(self, tmp_path):
        binary = _fake_binary(tmp_path, "raise SystemExit(1)")
        manager = H2PingSidecarManager(_cfg(startup_timeout_seconds=5), [_instance()], binary, str(tmp_path / "logs"))
        with pytest.raises(SidecarError):
            manager.start()
        assert manager._sidecars is None

    def test_check_raises_when_a_sidecar_died(self, tmp_path):
        binary = _fake_binary(
            tmp_path,
            """
            import sys, time
            args = sys.argv[1:]
            open(args[args.index("-ready-file") + 1], "w").write("1\\n")
            time.sleep(0.2)
            raise SystemExit(4)
            """,
        )
        manager = H2PingSidecarManager(_cfg(startup_timeout_seconds=5), [_instance()], binary, str(tmp_path / "logs"))
        manager.start()
        try:
            deadline = __import__("time").monotonic() + 5
            with pytest.raises(SidecarError, match="stopped unexpectedly"):
                while True:
                    manager.check()
                    if __import__("time").monotonic() > deadline:
                        break
                    __import__("time").sleep(0.05)
        finally:
            manager.stop()


class TestStartH2PingSidecar:
    def _gcd(self, tmp_path, binary, **sidecar):
        return OmegaConf.create(
            {
                "policy_base_url": "http://127.0.0.1:8000/v1",
                "judge": {"responses_api_models": {"genrm_model": {"base_url": [f"{NVCF}/v1"]}}},
                "sidecar": {
                    "enabled": True,
                    "instances": [INSTANCE],
                    "binary": str(binary),
                    "log_dir": str(tmp_path / "logs"),
                    "shutdown_grace": "1s",
                    **sidecar,
                },
            }
        )

    def test_disabled_is_a_no_op(self):
        gcd = OmegaConf.create({"judge": {"responses_api_models": {"m": {"base_url": [f"{NVCF}/v1"]}}}})
        assert start_h2_ping_sidecar(gcd) is None
        gcd["sidecar"] = {"enabled": False}
        assert start_h2_ping_sidecar(gcd) is None
        assert list(gcd.judge.responses_api_models.m.base_url) == [f"{NVCF}/v1"]

    def test_enabled_without_instances_is_an_error(self, tmp_path):
        gcd = self._gcd(tmp_path, _fake_binary(tmp_path, READY_THEN_IDLE))
        del gcd.sidecar["instances"]
        with pytest.raises(ValidationError, match="instances is required"):
            start_h2_ping_sidecar(gcd)

    def test_enabled_starts_and_rewrites_the_judge_url(self, tmp_path):
        gcd = self._gcd(tmp_path, _fake_binary(tmp_path, READY_THEN_IDLE))
        manager = start_h2_ping_sidecar(gcd)
        try:
            assert list(gcd.judge.responses_api_models.genrm_model.base_url) == ["http://127.0.0.1:1250/v1"]
            assert gcd.policy_base_url == "http://127.0.0.1:8000/v1"  # a local policy never needs the sidecar
            manager.check()
        finally:
            manager.stop()

    def test_rewrite_can_be_disabled(self, tmp_path):
        gcd = self._gcd(tmp_path, _fake_binary(tmp_path, READY_THEN_IDLE), rewrite_base_urls=False)
        manager = start_h2_ping_sidecar(gcd)
        try:
            assert list(gcd.judge.responses_api_models.genrm_model.base_url) == [f"{NVCF}/v1"]
        finally:
            manager.stop()

    def test_failure_leaves_urls_untouched(self, tmp_path):
        gcd = self._gcd(tmp_path, _fake_binary(tmp_path, "raise SystemExit(1)"))
        with pytest.raises(SidecarError, match="exited with code 1"):
            start_h2_ping_sidecar(gcd)
        assert list(gcd.judge.responses_api_models.genrm_model.base_url) == [f"{NVCF}/v1"]

    def test_block_is_not_a_server_instance(self):
        from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS

        assert "sidecar" in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


class TestStartIsTransactional:
    def _gcd(self, tmp_path, binary, bad_url=None):
        gcd = {
            "judge": {"responses_api_models": {"m": {"base_url": [f"{NVCF}/v1"]}}},
            "sidecar": {
                "enabled": True,
                "instances": [INSTANCE],
                "binary": str(binary),
                "log_dir": str(tmp_path / "logs"),
                "shutdown_grace": "1s",
            },
        }
        if bad_url is not None:
            gcd["other"] = {"responses_api_models": {"m": {"openai_base_url": bad_url}}}
        return OmegaConf.create(gcd)

    def test_malformed_url_fails_before_any_process_starts(self, tmp_path):
        gcd = self._gcd(tmp_path, _fake_binary(tmp_path, READY_THEN_IDLE), bad_url="https://[invalid/v1")

        with patch.object(H2PingSidecarManager, "start") as start:
            with pytest.raises(SidecarError, match="not a valid URL"):
                start_h2_ping_sidecar(gcd)

        start.assert_not_called()
        assert list(gcd.judge.responses_api_models.m.base_url) == [f"{NVCF}/v1"]
        assert not (tmp_path / "logs").exists()

    @pytest.mark.parametrize("failure", [RuntimeError("boom after start"), KeyboardInterrupt()])
    def test_failure_after_start_stops_the_proxy(self, tmp_path, failure):
        gcd = self._gcd(tmp_path, _fake_binary(tmp_path, READY_THEN_IDLE))
        real = launcher.rewrite_base_urls
        calls = []

        def flaky(config, instances):
            calls.append(1)
            if len(calls) == 1:  # the dry run before anything starts
                return real(config, instances)
            raise failure

        with patch.object(launcher, "rewrite_base_urls", side_effect=flaky):
            with patch.object(
                H2PingSidecarManager, "stop", autospec=True, side_effect=H2PingSidecarManager.stop
            ) as stop:
                with pytest.raises(type(failure)):
                    start_h2_ping_sidecar(gcd)

        stop.assert_called_once()
        assert list((tmp_path / "logs").glob("*.ready")) == []


class TestLoopbackWarning:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.1.2.3", "localhost", "LOCALHOST", "::1", "[::1]"])
    def test_loopback(self, host):
        assert is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "example.com", "", "192.168.1.2"])
    def test_not_loopback(self, host):
        assert not is_loopback_host(host)

    def _run(self, tmp_path, listen, capsys):
        binary = _fake_binary(tmp_path, READY_THEN_IDLE)
        gcd = OmegaConf.create(
            {
                "judge": {"responses_api_models": {"m": {"base_url": [f"{NVCF}/v1"]}}},
                "sidecar": {
                    "enabled": True,
                    "instances": [{"name": "nvcf", "upstream": NVCF, "listen": listen}],
                    "binary": str(binary),
                    "log_dir": str(tmp_path / "logs"),
                    "shutdown_grace": "1s",
                },
            }
        )
        manager = start_h2_ping_sidecar(gcd)
        try:
            return capsys.readouterr().out
        finally:
            manager.stop()

    def test_non_loopback_listen_is_accepted_but_warned_about(self, tmp_path, capsys):
        assert "not a loopback address" in self._run(tmp_path, "0.0.0.0:1250", capsys)

    def test_loopback_listen_is_not_warned_about(self, tmp_path, capsys):
        assert "not a loopback address" not in self._run(tmp_path, "127.0.0.1:1250", capsys)
