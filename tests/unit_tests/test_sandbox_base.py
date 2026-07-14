# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Contract tests for the provider-neutral public sandbox API.

These cover the base types in ``nemo_gym.sandbox`` (spec, resources, handle,
exec result, status, errors, registry, config resolution) and must keep
passing without the optional ``sandbox`` extra installed.
"""

import dataclasses

import pytest

import nemo_gym.sandbox as sandbox_pkg
from nemo_gym.sandbox import (
    ExecResult,
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
    get_provider_class,
    list_providers,
    resolve_provider_config,
    resolve_provider_metadata,
)


def test_public_api_exports_are_importable() -> None:
    for name in sandbox_pkg.__all__:
        assert getattr(sandbox_pkg, name) is not None


def test_sandbox_status_is_a_string_enum() -> None:
    assert [status.value for status in SandboxStatus] == ["starting", "running", "stopped", "error", "unknown"]
    assert SandboxStatus("running") is SandboxStatus.RUNNING
    assert isinstance(SandboxStatus.RUNNING, str)


def test_sandbox_resources_from_mapping_defaults_and_coercion() -> None:
    assert SandboxResources.from_mapping(None) == SandboxResources()
    assert SandboxResources.from_mapping({}) == SandboxResources()
    assert SandboxResources.from_mapping({"cpu": None, "gpu": None}) == SandboxResources()

    resources = SandboxResources.from_mapping(
        {"cpu": "2", "memory_mib": "1024", "disk_gib": 10.0, "gpu": 1, "gpu_type": "H100"}
    )
    assert resources == SandboxResources(cpu=2.0, memory_mib=1024, disk_gib=10, gpu=1, gpu_type="H100")
    assert isinstance(resources.cpu, float)
    assert isinstance(resources.memory_mib, int)

    with pytest.raises(ValueError, match="Unknown sandbox resource keys: vram"):
        SandboxResources.from_mapping({"vram": 1})


def test_sandbox_resources_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SandboxResources().cpu = 1.0  # type: ignore[misc]


def test_sandbox_spec_defaults_and_resource_coercion() -> None:
    spec = SandboxSpec()
    assert spec.image is None
    assert spec.env == {}
    assert spec.files == {}
    assert spec.metadata == {}
    assert spec.provider_options == {}
    assert spec.entrypoint is None
    assert spec.resources == SandboxResources()

    # Default containers are per-instance, not shared across specs.
    assert spec.env is not SandboxSpec().env

    coerced = SandboxSpec(image="image:tag", resources={"cpu": 1})
    assert coerced.resources == SandboxResources(cpu=1.0)

    explicit = SandboxResources(gpu=1)
    assert SandboxSpec(resources=explicit).resources is explicit

    with pytest.raises(ValueError, match="Unknown sandbox resource keys"):
        SandboxSpec(resources={"bogus": 1})

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.image = "other"  # type: ignore[misc]


def test_sandbox_exec_result_contract_and_alias() -> None:
    assert ExecResult is SandboxExecResult

    result = SandboxExecResult(stdout="out", stderr=None, return_code=0)
    assert result.error_type is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.return_code = 1  # type: ignore[misc]


def test_sandbox_handle_carries_opaque_provider_state() -> None:
    raw = object()
    handle = SandboxHandle(sandbox_id="sandbox-1", provider_name="fake", raw=raw)
    assert handle.raw is raw

    # Mutable by design: providers may refresh raw state on an existing handle.
    handle.raw = None
    assert handle.raw is None


def test_sandbox_create_error_hierarchy() -> None:
    assert issubclass(SandboxCreateError, RuntimeError)
    assert issubclass(SandboxCreateVerificationError, SandboxCreateError)


def test_builtin_provider_classes_load_without_optional_extras() -> None:
    for provider_name in ("apptainer", "daytona", "docker", "ecs_fargate", "opensandbox"):
        assert provider_name in list_providers()
        provider_class = get_provider_class(provider_name)
        assert provider_class.name == provider_name


def test_resolve_provider_config_rejects_non_mapping_blocks() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_provider_config("sandbox", {"sandbox": ["not", "a", "mapping"]})

    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_provider_metadata("sandbox", {"sandbox": ["not", "a", "mapping"]})


def test_resolve_provider_config_reports_no_candidates_when_config_is_empty() -> None:
    with pytest.raises(ValueError, match=r"Available sandbox configs: \(none\)"):
        resolve_provider_config("missing", {})
