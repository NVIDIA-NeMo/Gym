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
"""The `telemetry:` config block, and its translation into the child-process environment.

Gym spawns each server with `Popen`, so `os.environ` is the only channel a YAML setting
has to reach a server process. These tests cover that translation and, importantly, its
precedence: a raw env var set by the user must beat YAML, never the other way round.
"""

import os

import pytest
from omegaconf import OmegaConf

from nemo_gym.telemetry.config import TelemetryConfig
from nemo_gym.telemetry.setup import (
    configure_telemetry_env,
    is_telemetry_env_enabled,
    telemetry_config_from_global_config,
)
from tests.unit_tests.telemetry.conftest import no_lens


def test_defaults_are_off_and_gym_shaped():
    config = TelemetryConfig()
    assert config.enabled is False
    assert config.service_name == "nemo-gym"
    assert config.span_groups == "default"


def test_export_strategy_defaults_to_all_ranks():
    """Every Gym server process must export.

    NeMo-RL and Megatron-LM default to `single_rank` because they run one process tree
    where rank 0 is representative. Every Gym server is rank 0 of its own world of 1, so
    `single_rank` would be meaningless here and any silenced process is a hole in the
    middle of a distributed trace.
    """
    assert TelemetryConfig().export_strategy == "all_ranks"


def test_unknown_keys_are_preserved_not_rejected():
    """`extra="allow"` keeps a forward-compatible key from breaking an existing config."""
    config = TelemetryConfig.model_validate({"enabled": True, "some_future_knob": 7})
    assert config.enabled is True
    assert config.some_future_knob == 7


def test_config_reads_the_telemetry_block_from_a_gym_config(clean_otel_env):
    global_config = OmegaConf.create(
        {
            "telemetry": {"enabled": True, "span_groups": "per_rollout", "exporter": "console"},
            "some_server": {"resources_servers": {"x": {"entrypoint": "app.py"}}},
        }
    )
    config = telemetry_config_from_global_config(global_config)
    assert config.enabled is True
    assert config.span_groups == "per_rollout"
    assert config.exporter == "console"


def test_missing_telemetry_block_yields_a_disabled_config():
    """A config that never mentions telemetry must resolve to 'off', not raise."""
    config = telemetry_config_from_global_config(OmegaConf.create({"a_server": {}}))
    assert config.enabled is False


def test_none_global_config_yields_a_disabled_config():
    assert telemetry_config_from_global_config(None).enabled is False


def test_telemetry_is_a_reserved_top_level_key():
    """Otherwise Gym's config parser reads `telemetry:` as a server instance definition."""
    from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
    from nemo_gym.telemetry.setup import TELEMETRY_KEY_NAME

    assert TELEMETRY_KEY_NAME in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


def test_config_is_translated_into_env_for_spawned_servers(clean_otel_env):
    """The YAML block must land in os.environ, which is what Popen hands to each server."""
    configure_telemetry_env(
        TelemetryConfig(enabled=True, span_groups="all", exporter="console", service_name="my-gym")
    )
    assert os.environ["NEMO_GYM_OTEL_ENABLED"] == "1"
    assert os.environ["NEMO_GYM_OTEL_SPAN_GROUPS"] == "all"
    assert os.environ["NEMO_GYM_OTEL_EXPORTER"] == "console"
    assert os.environ["OTEL_SERVICE_NAME"] == "my-gym"


def test_booleans_are_translated_as_1_and_0(clean_otel_env):
    configure_telemetry_env(TelemetryConfig(enabled=True, logs_enabled=False, metrics_enabled=True))
    assert os.environ["NEMO_GYM_OTEL_ENABLED"] == "1"
    assert os.environ["NEMO_GYM_OTEL_LOGS_ENABLED"] == "0"
    assert os.environ["NEMO_GYM_OTEL_METRICS_ENABLED"] == "1"


def test_env_wins_over_yaml(clean_otel_env):
    """Raw env vars override the config block — the documented precedence.

    Uses setdefault, so an operator can flip telemetry off (or repoint the exporter) on a
    single run without editing a shared config file.
    """
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "0")
    clean_otel_env.setenv("NEMO_GYM_OTEL_SPAN_GROUPS", "verify")
    clean_otel_env.setenv("OTEL_SERVICE_NAME", "set-by-hand")

    configure_telemetry_env(TelemetryConfig(enabled=True, span_groups="all", service_name="from-yaml"))

    assert os.environ["NEMO_GYM_OTEL_ENABLED"] == "0", "YAML must not overwrite an explicit env var"
    assert os.environ["NEMO_GYM_OTEL_SPAN_GROUPS"] == "verify"
    assert os.environ["OTEL_SERVICE_NAME"] == "set-by-hand"


def test_disabled_config_produces_no_run_id(clean_otel_env):
    assert configure_telemetry_env(TelemetryConfig(enabled=False)) is None
    assert "NEMO_GYM_OTEL_RUN_ID" not in os.environ


def test_a_single_run_id_is_minted_for_the_whole_fleet(clean_otel_env):
    """Every server process must report the same nemo.run.id or they cannot be grouped."""
    run_id = configure_telemetry_env(TelemetryConfig(enabled=True))
    assert run_id
    assert os.environ["NEMO_GYM_OTEL_RUN_ID"] == run_id


def test_explicit_run_id_is_respected(clean_otel_env):
    clean_otel_env.setenv("NEMO_GYM_OTEL_RUN_ID", "my-run")
    assert configure_telemetry_env(TelemetryConfig(enabled=True)) == "my-run"


def test_slurm_job_id_becomes_the_run_id(clean_otel_env):
    """On a cluster the job id is the natural correlation key across every process."""
    clean_otel_env.setenv("SLURM_JOB_ID", "123456")
    assert configure_telemetry_env(TelemetryConfig(enabled=True)) == "123456"


def test_configure_with_no_block_is_a_no_op(clean_otel_env):
    assert configure_telemetry_env(None) is None
    assert "NEMO_GYM_OTEL_ENABLED" not in os.environ


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("garbage", False),
    ],
)
def test_env_enabled_parsing(clean_otel_env, value, expected):
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", value)
    assert is_telemetry_env_enabled() is expected


def test_nemo_lens_prefix_is_honoured_as_a_fallback(clean_otel_env):
    """`NEMO_LENS_ENABLED=1` is the documented one-liner for switching telemetry on."""
    clean_otel_env.setenv("NEMO_LENS_ENABLED", "1")
    assert is_telemetry_env_enabled() is True


def test_gym_prefix_takes_precedence_over_the_lens_fallback(clean_otel_env):
    clean_otel_env.setenv("NEMO_LENS_ENABLED", "1")
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "0")
    assert is_telemetry_env_enabled() is False


# --------------------------------------------------------------------------- #
# Reaching the per-server venvs
# --------------------------------------------------------------------------- #


def test_no_server_requirements_when_telemetry_is_off(clean_otel_env):
    """A normal run's server venvs must be byte-for-byte what they are today."""
    from nemo_gym.telemetry.setup import server_venv_requirements

    assert server_venv_requirements() == []


def test_server_requirements_pin_the_same_lens_this_process_runs(clean_otel_env):
    """Server venvs install `nemo-gym[dev]`, not `nemo-gym[telemetry]`.

    Without an explicit requirement, telemetry would be on in the orchestrator and absent
    in every server — a trace with a hole in the middle. And a bare `nemo-lens[sdk]` is
    not enough: `[tool.uv.sources]` only governs dependencies resolved *through* the local
    project, so a requirement named on the uv command line resolves from PyPI instead
    (lens 0.1.0) while the orchestrator runs the pinned commit.

    Deriving the requirement from this process's own installed distribution makes that
    skew impossible rather than relying on two pins staying in sync.
    """
    pytest.importorskip("nemo.lens")
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")

    from nemo_gym.telemetry.setup import server_venv_requirements

    requirements = server_venv_requirements()
    lens_requirement = next(req for req in requirements if req.startswith("nemo-lens"))

    assert lens_requirement.startswith("nemo-lens[sdk]"), "the server needs the SDK, not just the API"
    assert "==" in lens_requirement or "@" in lens_requirement, (
        f"nemo-lens must be pinned for the server venvs, got {lens_requirement!r}"
    )
    if "@" in lens_requirement:
        # Installed from git: the requirement must name the exact commit, not a branch.
        commit = lens_requirement.rsplit("@", 1)[1]
        assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), (
            f"expected a full commit sha, got {commit!r}"
        )


def test_server_requirements_include_the_instrumentation_packages(clean_otel_env):
    """nemo-lens imports these lazily and raises ImportError without them, so a server
    venv that lacks them loses inbound context extraction and every SERVER span."""
    pytest.importorskip("nemo.lens")
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")

    from nemo_gym.telemetry.setup import server_venv_requirements

    requirements = " ".join(server_venv_requirements())
    assert "opentelemetry-instrumentation-fastapi" in requirements
    assert "opentelemetry-instrumentation-aiohttp-client" in requirements


def test_missing_lens_yields_no_requirements_rather_than_an_unpinned_one(clean_otel_env):
    """Telemetry enabled but lens absent must not put a floating `nemo-lens` in the venvs."""
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")

    from nemo_gym.telemetry import setup as telemetry_setup

    with no_lens():
        original = telemetry_setup._installed_requirement
        telemetry_setup._installed_requirement = lambda *args: None
        try:
            assert telemetry_setup.server_venv_requirements() == []
        finally:
            telemetry_setup._installed_requirement = original


def test_head_server_deps_carry_telemetry_into_server_venvs(clean_otel_env):
    """The end of the chain: what `uv pip install` actually receives for each server."""
    pytest.importorskip("nemo.lens")
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "1")

    from nemo_gym.telemetry.setup import server_venv_requirements

    head_server_deps = ["ray[default]==2.56.1", *server_venv_requirements()]
    assert any(dep.startswith("nemo-lens[sdk]") for dep in head_server_deps)


# --------------------------------------------------------------------------- #
# Enablement precedence
# --------------------------------------------------------------------------- #


def test_lens_env_var_alone_enables_telemetry(clean_otel_env):
    """`NEMO_LENS_ENABLED=1` with no `telemetry:` block must actually enable telemetry.

    A config with no telemetry block resolves to `enabled=False`. Exporting that value
    would write `NEMO_GYM_OTEL_ENABLED=0`, which shadows `NEMO_LENS_ENABLED=1` because the
    Gym-prefixed variable deliberately beats the lens fallback — leaving the documented
    one-liner with no effect. `configure_telemetry_env` therefore exports the *effective*
    enablement rather than the config's.
    """
    clean_otel_env.setenv("NEMO_LENS_ENABLED", "1")

    run_id = configure_telemetry_env(TelemetryConfig())

    assert is_telemetry_env_enabled() is True, "the config default overwrote the enabling env var"
    assert os.environ["NEMO_GYM_OTEL_ENABLED"] == "1"
    assert run_id, "an enabled run must still mint a run id"


def test_a_disabled_run_writes_no_telemetry_env_at_all(clean_otel_env):
    """With telemetry off, `os.environ` must be left exactly as it was.

    Server processes inherit this environment. Writing `NEMO_GYM_OTEL_*=0` into every run
    is both noise and, as the regression above showed, actively harmful.
    """
    before = {key for key in os.environ if key.startswith(("NEMO_GYM_OTEL_", "OTEL_"))}

    assert configure_telemetry_env(TelemetryConfig()) is None

    after = {key for key in os.environ if key.startswith(("NEMO_GYM_OTEL_", "OTEL_"))}
    assert after == before


def test_explicit_env_off_beats_an_enabled_config_block(clean_otel_env):
    """An operator must be able to switch telemetry off for one run without editing YAML."""
    clean_otel_env.setenv("NEMO_GYM_OTEL_ENABLED", "0")

    configure_telemetry_env(TelemetryConfig(enabled=True, span_groups="all"))

    assert os.environ["NEMO_GYM_OTEL_ENABLED"] == "0"
    assert is_telemetry_env_enabled() is False


def test_enabling_via_lens_env_still_carries_yaml_settings_to_servers(clean_otel_env):
    """Enabling through the env must not discard the rest of the config block."""
    clean_otel_env.setenv("NEMO_LENS_ENABLED", "1")

    configure_telemetry_env(TelemetryConfig(span_groups="per_rollout", exporter="console"))

    assert os.environ["NEMO_GYM_OTEL_SPAN_GROUPS"] == "per_rollout"
    assert os.environ["NEMO_GYM_OTEL_EXPORTER"] == "console"
