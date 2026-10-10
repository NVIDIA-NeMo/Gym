# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CCC directory env vars: the CCC_-prefixed names win, and the unprefixed names remain fallbacks.

The IOI benchmarks inherit these fields from the CCC config, so they are resolved through the real config
parser as well: a benchmark-level override would silently keep reading only the old variable names.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATHS = {
    "competitive_coding_challenges_resources_server": REPO_ROOT
    / "resources_servers/competitive_coding_challenges/configs/competitive_coding_challenges.yaml",
    "ioi_resources_server": REPO_ROOT / "benchmarks/ioi/config.yaml",
    "ioi26_resources_server": REPO_ROOT / "benchmarks/ioi26/config.yaml",
}
DIR_ENV_VARS = ("CCC_SHARED_TEMP_DIR", "SHARED_TEMP_DIR", "CCC_LOCAL_COMPILE_DIR", "LOCAL_COMPILE_DIR")
LEGACY_ENV = {"SHARED_TEMP_DIR": "/legacy/shared", "LOCAL_COMPILE_DIR": "/legacy/compile"}
PREFIXED_ENV = {"CCC_SHARED_TEMP_DIR": "/ccc/shared", "CCC_LOCAL_COMPILE_DIR": "/ccc/compile"}


def _resolved_dirs(server_instance: str) -> tuple[str, str]:
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                OmegaConf.load(CONFIG_PATHS[server_instance]),
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    server_config = resolved[server_instance].resources_servers.competitive_coding_challenges
    return server_config.shared_dir, server_config.local_compile_dir


@pytest.mark.parametrize("server_instance", list(CONFIG_PATHS))
@pytest.mark.parametrize(
    ("env", "expected"),
    [
        pytest.param({}, ("/tmp", "/tmp/nemo-gym-compile"), id="unset"),
        pytest.param(LEGACY_ENV, ("/legacy/shared", "/legacy/compile"), id="legacy-only"),
        pytest.param(PREFIXED_ENV, ("/ccc/shared", "/ccc/compile"), id="prefixed-only"),
        pytest.param({**LEGACY_ENV, **PREFIXED_ENV}, ("/ccc/shared", "/ccc/compile"), id="prefixed-wins"),
    ],
)
def test_dir_env_vars_prefer_ccc_names_with_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch, server_instance: str, env: dict[str, str], expected: tuple[str, str]
) -> None:
    for name in DIR_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert _resolved_dirs(server_instance) == expected
