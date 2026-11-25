# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.task_environment import resolve_task_environment


IMAGE = "registry.example/tasks/parser@sha256:" + "a" * 64


def test_direct_image_resolves_to_sandbox_spec():
    resolved = resolve_task_environment(
        {"task_environment": {"task_id": "parser-001", "image": IMAGE, "workdir": "/testbed"}}
    )
    spec = resolved.sandbox_spec()
    assert spec.image == IMAGE
    assert spec.workdir == "/testbed"
    assert spec.metadata["task_id"] == "parser-001"
    assert spec.metadata["task_image"] == IMAGE


def test_manifest_task_id_resolves():
    resolved = resolve_task_environment(
        {"task_environment": {"task_id": "parser-001"}},
        manifest={"parser-001": {"image": IMAGE, "workdir": "/testbed"}},
    )
    assert resolved.image == IMAGE
    assert resolved.workdir == "/testbed"


def test_direct_image_and_manifest_must_match():
    other = "registry.example/tasks/parser@sha256:" + "b" * 64
    with pytest.raises(ValueError, match="does not match the manifest"):
        resolve_task_environment(
            {"task_environment": {"task_id": "parser-001", "image": other}},
            manifest={"parser-001": IMAGE},
        )


def test_dataset_cannot_set_operator_fields():
    with pytest.raises(ValueError, match="extra"):
        resolve_task_environment(
            {
                "task_environment": {
                    "image": IMAGE,
                    "provider_options": {"privileged": True},
                }
            }
        )


def test_operator_fields_are_preserved():
    operator = SandboxSpec(
        ttl_s=300,
        env={"MODE": "eval"},
        provider_options={"network": "isolated"},
        ports=[8080],
    )
    spec = resolve_task_environment(
        {"task_environment": {"image": IMAGE}},
        operator_spec=operator,
    ).sandbox_spec(operator)
    assert spec.ttl_s == 300
    assert spec.env == {"MODE": "eval"}
    assert spec.provider_options == {"network": "isolated"}
    assert spec.ports == (8080,)


def test_mutable_image_is_rejected():
    with pytest.raises(ValueError, match="immutable OCI digest"):
        resolve_task_environment(
            {"task_environment": {"image": "registry.example/tasks/parser:latest"}}
        )
