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

import re
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
OSWORLD_AGENT_REQUIREMENTS = ROOT / "responses_api_agents/osworld_agent/requirements.txt"
OSWORLD_AGENT_OVERRIDES = ROOT / "responses_api_agents/osworld_agent/uv-overrides.txt"
OSWORLD_AGENT_UV_CONFIG = ROOT / "responses_api_agents/osworld_agent/uv.toml"
OSWORLD_AGENT_PUBLIC_OVERRIDES = ROOT / "responses_api_agents/osworld_agent/overrides.txt"
OSWORLD_RESOURCES_PROJECT = ROOT / "resources_servers/osworld/pyproject.toml"
OSWORLD_AGENT_README = ROOT / "responses_api_agents/osworld_agent/README.md"
OSWORLD_BENCHMARK_README = ROOT / "benchmarks/osworld/README.md"
VLLM_MODEL_PYTHON = ROOT / "responses_api_models/vllm_model/uv-python-version.txt"
VLLM_MODEL_MANAGED_PYTHON = ROOT / "responses_api_models/vllm_model/uv-managed-python.txt"
OSWORLD_AGENT_PYTHON = ROOT / "responses_api_agents/osworld_agent/uv-python-version.txt"
OSWORLD_AGENT_MANAGED_PYTHON = ROOT / "responses_api_agents/osworld_agent/uv-managed-python.txt"
OSWORLD_UNSUPPORTED_VM_PROVIDER_DEPENDENCIES = {
    "alibabacloud-ecs20140526",
    "alibabacloud-tea-openapi",
    "alibabacloud-tea-util",
    "azure-identity",
    "azure-mgmt-compute",
    "azure-mgmt-network",
    "volcengine-python-sdk",
}


def _uv_config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["tool"]["uv"]


def test_osworld_agent_uv_config_mirrors_project_resolver_policy() -> None:
    with OSWORLD_AGENT_UV_CONFIG.open("rb") as f:
        server_config = tomllib.load(f)
    project_config = _uv_config()

    for key in ("constraint-dependencies", "override-dependencies"):
        assert server_config[key] == project_config[key]
    project_exclusions = set(project_config["exclude-dependencies"])
    server_exclusions = set(server_config["exclude-dependencies"])
    assert project_exclusions <= server_exclusions
    assert server_exclusions - project_exclusions == OSWORLD_UNSUPPORTED_VM_PROVIDER_DEPENDENCIES
    required_version = SpecifierSet(server_config["required-version"])
    assert Version("0.11.24") not in required_version
    assert Version("0.11.25") in required_version
    assert "managed" not in server_config


def test_osworld_runtime_consumers_share_one_pinned_revision() -> None:
    revision_pattern = re.compile(r"OSWorld/archive/([0-9a-f]{40})\.tar\.gz")
    agent_match = revision_pattern.search(OSWORLD_AGENT_REQUIREMENTS.read_text(encoding="utf-8"))
    resources_match = revision_pattern.search(OSWORLD_RESOURCES_PROJECT.read_text(encoding="utf-8"))

    assert agent_match, "The OSWorld agent must pin an immutable OSWorld archive revision"
    assert resources_match, "The OSWorld Resources Server must pin an immutable OSWorld archive revision"
    assert agent_match.group(1) == resources_match.group(1)
    revision = agent_match.group(1)
    assert f"commit `{revision}`" in OSWORLD_AGENT_README.read_text(encoding="utf-8")
    benchmark_readme = OSWORLD_BENCHMARK_README.read_text(encoding="utf-8")
    assert f"pinned to `{revision}`" in benchmark_readme
    assert f"git checkout {revision}" in benchmark_readme


def test_osworld_resources_server_avoids_unsatisfiable_torchvision_resolution() -> None:
    with OSWORLD_RESOURCES_PROJECT.open("rb") as f:
        resource_uv = tomllib.load(f)["tool"]["uv"]

    assert "torchvision; sys_platform == 'never'" in resource_uv["override-dependencies"]


def test_osworld_agent_dependency_overrides() -> None:
    requirements = OSWORLD_AGENT_REQUIREMENTS.read_text(encoding="utf-8")
    agent_overrides = OSWORLD_AGENT_OVERRIDES.read_text(encoding="utf-8")
    public_overrides = OSWORLD_AGENT_PUBLIC_OVERRIDES.read_text(encoding="utf-8")

    assert "grpcio-status==1.71.2" in agent_overrides
    assert "protobuf==5.29.6" in agent_overrides
    assert "numpy>=2.1,<2.5" in agent_overrides
    assert "torch==2.11.0" in public_overrides
    assert "numpy==2.5.1" not in agent_overrides
    assert "opencv-python-headless==5.0.0.93" not in agent_overrides
    assert "numpy>=2.1,<2.5" in requirements


def test_server_ray_version_is_owned_by_parent_process() -> None:
    # global_config.py injects the parent process's exact Ray version into
    # every server installation. Static overrides must not drag a current
    # source checkout back to whichever Ray happened to ship in a base image.
    ray_override = re.compile(r"(?m)^\s*ray(?:\[default\])?\s*[<>=!~]")
    managed_overrides = [
        *ROOT.glob("responses_api_agents/*/uv-overrides.txt"),
        *ROOT.glob("responses_api_models/*/uv-overrides.txt"),
        *ROOT.glob("resources_servers/*/uv-overrides.txt"),
    ]
    for override_path in managed_overrides:
        assert ray_override.search(override_path.read_text(encoding="utf-8")) is None
    assert not (ROOT / "responses_api_models/vllm_model/uv-overrides.txt").exists()


def test_nemo_rl_servers_use_the_project_python_floor() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        python_floor = tomllib.load(f)["project"]["requires-python"].removeprefix(">=")

    assert OSWORLD_AGENT_PYTHON.read_text(encoding="utf-8").strip() == python_floor
    assert VLLM_MODEL_PYTHON.read_text(encoding="utf-8").strip() == python_floor
    assert OSWORLD_AGENT_MANAGED_PYTHON.read_text(encoding="utf-8").strip() == "true"
    assert VLLM_MODEL_MANAGED_PYTHON.read_text(encoding="utf-8").strip() == "true"
