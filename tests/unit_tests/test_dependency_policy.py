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

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
OSWORLD_AGENT_REQUIREMENTS = ROOT / "responses_api_agents/osworld_agent/requirements.txt"
OSWORLD_AGENT_UV_CONFIG = ROOT / "responses_api_agents/osworld_agent/uv.toml"
OSWORLD_AGENT_PUBLIC_OVERRIDES = ROOT / "responses_api_agents/osworld_agent/overrides.txt"
OSWORLD_RESOURCES_PROJECT = ROOT / "resources_servers/osworld/pyproject.toml"
OSWORLD_RESOURCES_PYTHON = ROOT / "resources_servers/osworld/.python-version"
OSWORLD_AGENT_README = ROOT / "responses_api_agents/osworld_agent/README.md"
OSWORLD_BENCHMARK_README = ROOT / "benchmarks/osworld/README.md"
VLLM_MODEL_PYTHON = ROOT / "responses_api_models/vllm_model/.python-version"
OSWORLD_AGENT_PYTHON = ROOT / "responses_api_agents/osworld_agent/.python-version"
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

    assert server_config["constraint-dependencies"] == project_config["constraint-dependencies"]
    project_overrides = set(project_config["override-dependencies"])
    server_overrides = set(server_config["override-dependencies"])
    assert project_overrides <= server_overrides
    assert server_overrides - project_overrides == {
        "grpcio-status==1.71.2",
        "protobuf==5.29.6",
        "numpy>=2.1,<2.5",
    }
    project_exclusions = set(project_config["exclude-dependencies"])
    server_exclusions = set(server_config["exclude-dependencies"])
    assert project_exclusions <= server_exclusions
    assert server_exclusions - project_exclusions == OSWORLD_UNSUPPORTED_VM_PROVIDER_DEPENDENCIES
    required_version = SpecifierSet(server_config["required-version"])
    assert Version("0.11.24") not in required_version
    assert Version("0.11.25") in required_version
    assert "managed" not in server_config
    assert "python-preference" not in server_config
    assert server_config["pip"]["torch-backend"] == "cpu"


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
        resources_project = tomllib.load(f)
    resource_uv = resources_project["tool"]["uv"]

    assert "torchvision; sys_platform == 'never'" in resource_uv["override-dependencies"]


def test_osworld_resources_server_owns_a_python_313_wheel_compatible_runtime() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        parent_python = tomllib.load(f)["project"]["requires-python"]
    with OSWORLD_RESOURCES_PROJECT.open("rb") as f:
        resources_project = tomllib.load(f)

    assert resources_project["project"]["requires-python"] == parent_python
    resource_python = SpecifierSet(OSWORLD_RESOURCES_PYTHON.read_text(encoding="utf-8").strip())
    assert Version(parent_python.removeprefix(">=")) in resource_python
    assert Version("3.14") not in resource_python
    assert resources_project["tool"]["uv"]["pip"]["torch-backend"] == "cpu"

    direct = {Requirement(value).name: Requirement(value) for value in resources_project["project"]["dependencies"]}
    overrides = {
        Requirement(value).name: Requirement(value)
        for value in resources_project["tool"]["uv"]["override-dependencies"]
    }
    for package, rejected, admitted in (
        ("numpy", "1.26.4", "2.4.6"),
        ("opencv-python-headless", "4.8.1.78", "4.10.0.84"),
        ("Pillow", "11.0.0", "12.3.0"),
        ("matplotlib", "3.7.5", "3.10.6"),
    ):
        assert Version(rejected) not in direct[package].specifier
        assert Version(admitted) in direct[package].specifier
        assert Version(rejected) not in overrides[package].specifier
        assert Version(admitted) in overrides[package].specifier

    assert Version("2.5.1") not in overrides["torch"].specifier
    assert Version("2.11.0") in overrides["torch"].specifier


def test_osworld_agent_dependency_overrides() -> None:
    requirements = OSWORLD_AGENT_REQUIREMENTS.read_text(encoding="utf-8")
    with OSWORLD_AGENT_UV_CONFIG.open("rb") as config_file:
        agent_overrides = tomllib.load(config_file)["override-dependencies"]
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
    ray_override = re.compile(r"^ray(?:\[default\])?\s*[<>=!~]", re.IGNORECASE)
    with OSWORLD_AGENT_UV_CONFIG.open("rb") as config_file:
        agent_overrides = tomllib.load(config_file)["override-dependencies"]
    with OSWORLD_RESOURCES_PROJECT.open("rb") as config_file:
        resource_overrides = tomllib.load(config_file)["tool"]["uv"]["override-dependencies"]
    for requirement in (*agent_overrides, *resource_overrides):
        assert ray_override.search(requirement) is None


def test_role_runtime_policy_uses_standard_uv_files() -> None:
    legacy_markers = {
        "uv-managed-python.txt",
        "uv-overrides.txt",
        "uv-python-version.txt",
        "uv-torch-backend.txt",
    }
    tracked_role_dirs = (
        ROOT / "responses_api_agents/osworld_agent",
        ROOT / "resources_servers/osworld",
        ROOT / "responses_api_models/vllm_model",
    )

    assert not [path for role_dir in tracked_role_dirs for path in role_dir.iterdir() if path.name in legacy_markers]


def test_nemo_rl_servers_use_a_compatible_python_range() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        python_floor = tomllib.load(f)["project"]["requires-python"].removeprefix(">=")

    for python_policy in (OSWORLD_AGENT_PYTHON, VLLM_MODEL_PYTHON):
        supported = SpecifierSet(python_policy.read_text(encoding="utf-8").strip())
        assert Version(python_floor) in supported
        assert Version("3.14") not in supported
