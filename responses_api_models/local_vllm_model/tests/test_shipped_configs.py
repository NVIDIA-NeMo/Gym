# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The shipped configs must satisfy the assertions app.py makes about them.

`_configure_vllm_serve` asserts `VLLM_RAY_DP_PACK_STRATEGY` is present in the
env vars it builds from `vllm_serve_env_vars`. Every per-model config sets it;
the generic `local_vllm_model.yaml` -- the config used for any `--model` without
a dedicated file -- shipped `vllm_serve_env_vars: {}`, so it could never satisfy
that assertion:

    gym env start --environment blackjack --model-type local_vllm_model \
        ++policy_model_name=Qwen/Qwen3-8B

    AssertionError: Please provide a value for `VLLM_RAY_DP_PACK_STRATEGY`
                    for `policy_model`

These tests read the YAML that actually ships rather than restating a value
inline, so a config that cannot start is caught here instead of at run time.
"""

from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
GENERIC_CONFIG = CONFIG_DIR / "local_vllm_model.yaml"

# Asserted by app.py::_configure_vllm_serve on every startup path.
REQUIRED_ENV_VARS = ("VLLM_RAY_DP_PACK_STRATEGY",)


def _serve_env_vars(config_path: Path) -> dict:
    loaded = yaml.safe_load(config_path.read_text())
    return loaded["policy_model"]["responses_api_models"]["local_vllm_model"]["vllm_serve_env_vars"]


class TestShippedConfigsSatisfyStartupAssertions:
    def test_generic_config_sets_the_env_vars_app_py_requires(self) -> None:
        """The fallback config must be startable as shipped."""
        env_vars = _serve_env_vars(GENERIC_CONFIG)
        for name in REQUIRED_ENV_VARS:
            assert name in env_vars, (
                f"{GENERIC_CONFIG.name} omits {name}, which "
                f"app.py::_configure_vllm_serve asserts on every startup. Any "
                f"model without a dedicated config would fail to start."
            )

    def test_every_shipped_config_sets_the_env_vars_app_py_requires(self) -> None:
        """No config may ship in a state that cannot pass its own assertion."""
        offenders = []
        for config_path in sorted(CONFIG_DIR.rglob("*.yaml")):
            loaded = yaml.safe_load(config_path.read_text())
            model = (loaded or {}).get("policy_model", {}).get("responses_api_models", {}).get("local_vllm_model")
            if model is None or "vllm_serve_env_vars" not in model:
                continue
            missing = [n for n in REQUIRED_ENV_VARS if n not in model["vllm_serve_env_vars"]]
            if missing:
                offenders.append(f"{config_path.relative_to(CONFIG_DIR)}: missing {missing}")

        assert not offenders, "configs that cannot satisfy app.py's assertions:\n" + "\n".join(offenders)
