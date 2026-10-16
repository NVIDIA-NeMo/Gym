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
"""Guards the ``nemo_gym`` public API surface.

Environments are expected to import base classes and core types from ``nemo_gym``.
These tests pin that contract.
Every advertised name resolves to the same object as its defining module.
Importing ``nemo_gym`` does not eagerly import the server stack.
"""

import importlib
import inspect
import subprocess
import sys

import pytest

import nemo_gym
from nemo_gym import _LAZY_EXPORTS


EXPECTED_LAZY_EXPORTS = {
    "AggregateMetrics",
    "AggregateMetricsRequest",
    "BaseMultiRewardVerifyResponse",
    "BaseResourcesServerConfig",
    "BaseResponsesAPIAgentConfig",
    "BaseResponsesAPIModelConfig",
    "BaseRunRequest",
    "BaseRunServerInstanceConfig",
    "BaseSeedSessionRequest",
    "BaseSeedSessionResponse",
    "BaseServerConfig",
    "BaseVerifyRequest",
    "BaseVerifyResponse",
    "Domain",
    "ModelServerRef",
    "NeMoGymAsyncOpenAI",
    "NeMoGymChatCompletion",
    "NeMoGymChatCompletionCreateParamsNonStreaming",
    "NeMoGymResponse",
    "NeMoGymResponseCreateParamsNonStreaming",
    "ReverifyMode",
    "SESSION_ID_KEY",
    "ServerClient",
    "SimpleResourcesServer",
    "SimpleResponsesAPIAgent",
    "SimpleResponsesAPIModel",
    "get_response_json",
    "raise_for_status",
}

EXPECTED_EAGER_EXPORTS = {
    "CACHE_DIR",
    "PARENT_DIR",
    "RESULTS_DIR",
    "ROOT_DIR",
    "WORKING_DIR",
    "__package_name__",
    "__version__",
}

_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_KEYWORD = inspect.Parameter.KEYWORD_ONLY
_REQUIRED = inspect.Parameter.empty

# nemo_gym.in_process: plan a benchmark in one call, then run single tasks against running servers.
# Changes must be additive: a new parameter is optional and keyword-only. Removing or renaming a parameter,
# or changing its default, needs a deprecation first; update this list in the same change.
EXPECTED_IN_PROCESS_TASK_SIGNATURES = {
    "plan_benchmark": [
        ("benchmark", _POSITIONAL, _REQUIRED),
        ("task_ids", _KEYWORD, None),
        ("overrides", _KEYWORD, None),
        ("work_dir", _KEYWORD, None),
    ],
    "TaskRunner.__init__": [
        ("self", _POSITIONAL, _REQUIRED),
        ("rows", _POSITIONAL, _REQUIRED),
        ("head_server", _KEYWORD, _REQUIRED),
    ],
    "TaskRunner.run_task": [
        ("self", _POSITIONAL, _REQUIRED),
        ("task_id", _POSITIONAL, _REQUIRED),
        ("num_repeats", _KEYWORD, 1),
        ("rollout_id", _KEYWORD, None),
    ],
    "TaskRunner.task_ids": [("self", _POSITIONAL, _REQUIRED)],
}


@pytest.mark.parametrize("qualified_name", sorted(EXPECTED_IN_PROCESS_TASK_SIGNATURES))
def test_in_process_task_api_keeps_its_signature(qualified_name: str):
    from nemo_gym import in_process

    target = in_process
    for part in qualified_name.split("."):
        target = getattr(target, part)
    parameters = inspect.signature(target).parameters.values()
    actual = [(parameter.name, parameter.kind, parameter.default) for parameter in parameters]
    assert actual == EXPECTED_IN_PROCESS_TASK_SIGNATURES[qualified_name]


def test_public_api_contract_is_explicit():
    assert set(_LAZY_EXPORTS) == EXPECTED_LAZY_EXPORTS
    assert set(nemo_gym.__all__) == EXPECTED_EAGER_EXPORTS | EXPECTED_LAZY_EXPORTS


@pytest.mark.parametrize("name", sorted(_LAZY_EXPORTS))
def test_public_symbol_is_accessible(name: str):
    assert getattr(nemo_gym, name) is not None


@pytest.mark.parametrize("name, module_name", sorted(_LAZY_EXPORTS.items()))
def test_public_symbol_matches_deep_import(name: str, module_name: str):
    """The top-level re-export must be the identical object exposed by the internal module.

    This lets downstream code migrate without behavior changes.
    It preserves ``isinstance`` and subclass checks across both import styles.
    """
    module = importlib.import_module(f"nemo_gym.{module_name}")
    assert getattr(nemo_gym, name) is getattr(module, name)


def test_all_is_sorted():
    assert nemo_gym.__all__ == sorted(nemo_gym.__all__)


def test_dir_includes_lazy_exports():
    listed = dir(nemo_gym)
    assert set(_LAZY_EXPORTS).issubset(listed)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = nemo_gym.DefinitelyNotARealSymbol


def test_top_level_import_is_lazy():
    """Importing ``nemo_gym`` must not eagerly import the heavy submodules.

    Run in a fresh interpreter to avoid modules loaded by other tests.
    Accessing a symbol should import its backing module on demand.
    """
    script = (
        "import sys\n"
        "import nemo_gym\n"
        "assert 'nemo_gym.base_resources_server' not in sys.modules, 'import was not lazy'\n"
        "assert nemo_gym.SimpleResourcesServer is not None\n"
        "assert 'nemo_gym.base_resources_server' in sys.modules, 'access did not import module'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
