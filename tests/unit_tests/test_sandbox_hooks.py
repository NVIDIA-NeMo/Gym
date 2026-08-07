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


"""Tests for the sandbox_task hook mechanism (spec_resolver / exec_wrapper).

These cover `nemo_gym.sandbox.hooks` on its own, without the CLI that consumes it.
"""

import json

import pytest

from nemo_gym.sandbox import SandboxSpec
from nemo_gym.sandbox.hooks import (
    SandboxHookError,
    load_hook,
    resolve_spec,
    task_id_for_row,
    wrap_command,
)


def test_load_hook_imports_callable() -> None:
    assert load_hook("json:dumps") is json.dumps


@pytest.mark.parametrize(
    "reference, match",
    [
        ("json.dumps", "must be of the form"),
        ("nemo_gym_missing_module_xyz:thing", "could not be imported"),
        ("json:not_a_real_attribute", "not found"),
        ("json:__doc__", "not callable"),
    ],
)
def test_load_hook_rejects_bad_references(reference: str, match: str) -> None:
    with pytest.raises(SandboxHookError, match=match):
        load_hook(reference)


########################################
# hooks: id + spec resolution
########################################


def test_task_id_falls_back_through_conventional_fields() -> None:
    assert task_id_for_row({"instance_id": "a"}) == "a"
    assert task_id_for_row({"task_id": "b"}) == "b"
    assert task_id_for_row({"uuid": "c"}) == "c"
    assert task_id_for_row({"nope": "d"}) is None


def test_task_id_supports_nested_field() -> None:
    """Several servers keep the id under verifier_metadata."""
    row = {"verifier_metadata": {"task_id": "nested"}}
    assert task_id_for_row(row, id_from_row="verifier_metadata.task_id") == "nested"


def test_resolve_spec_prefers_resolver_over_config() -> None:
    """A server that computes its spec in Python is the authority on it."""
    spec, source = resolve_spec(
        sandbox_spec={"image": "from-config"},
        sandbox_task={"spec_resolver": "tests.unit_tests.test_sandbox_hooks:_resolver"},
        row={"instance_id": "x"},
        server_config={},
    )
    assert spec.image == "resolved-x"
    assert "spec_resolver" in source


def _resolver(row, server_config):
    return SandboxSpec(image=f"resolved-{(row or {}).get('instance_id', 'none')}")


def test_resolve_spec_uses_row_image_when_config_has_none() -> None:
    spec, source = resolve_spec(
        sandbox_spec={"ttl_s": 5},
        sandbox_task={"image_from_row": "image_name"},
        row={"image_name": "from-row"},
        server_config={},
    )
    assert spec.image == "from-row" and source == "row field"


def test_resolve_spec_row_image_goes_through_rewrites() -> None:
    """A derived image must reach the mirror too, not just a configured one."""
    spec, _ = resolve_spec(
        sandbox_spec={"image_rewrites": [{"from": "docker.io/", "to": "mirror/"}]},
        sandbox_task={"image_from_row": "image_name"},
        row={"image_name": "docker.io/x:1"},
        server_config={},
    )
    assert spec.image == "mirror/x:1"


def test_resolve_spec_surfaces_hook_failures() -> None:
    with pytest.raises(SandboxHookError, match="spec_resolver .* failed"):
        resolve_spec(
            sandbox_spec=None,
            sandbox_task={"spec_resolver": "tests.unit_tests.test_sandbox_hooks:_boom"},
            row={},
            server_config={},
        )


def _boom(row, server_config):
    raise ValueError("nope")


########################################
# hooks: command wrapping
########################################


def _upper_wrapper(command, **kwargs):
    return f"WRAPPED({command})"


def test_wrap_command_applies_and_reports() -> None:
    command, wrapped = wrap_command("ls", sandbox_task={"exec_wrapper": f"{__name__}:_upper_wrapper"})
    assert command == "WRAPPED(ls)" and wrapped is True


def test_wrap_command_bare_skips_wrapper() -> None:
    """--bare exists so you can debug around a wrapper that is itself suspect."""
    command, wrapped = wrap_command("ls", sandbox_task={"exec_wrapper": f"{__name__}:_upper_wrapper"}, bare=True)
    assert command == "ls" and wrapped is False


def test_wrap_command_without_wrapper_is_identity() -> None:
    assert wrap_command("ls", sandbox_task={}) == ("ls", False)


########################################
# Server discovery
########################################
