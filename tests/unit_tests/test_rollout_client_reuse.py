# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the production helper's client lifecycle without serving processes."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
from pydantic import BaseModel, Field, PrivateAttr


class _Head(BaseModel):
    host: str
    port: int


@pytest.fixture
def client_factory():
    # Keep the production Pydantic fields and method verbatim; fake only the
    # config-fetch boundary, which would otherwise require a live Gym server.
    path = Path(__file__).resolve().parents[2] / "nemo_gym/rollout_collection.py"
    module = ast.parse(path.read_text())
    original = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RolloutCollectionHelper"
    )
    fields = {"reuse_server_client", "_cached_server_client", "_cached_head_server_config"}
    nodes = [
        node
        for node in original.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in fields
        or isinstance(node, ast.FunctionDef)
        and node.name == "setup_server_client"
    ]
    calls = []
    version = [0]
    fail = [False]

    def load(head=None):
        calls.append(head.model_copy(deep=True) if head is not None else None)
        if fail[0]:
            raise ConnectionError("head unavailable")
        return SimpleNamespace(head_server_config=head, global_config_dict={"version": version[0]})

    namespace = {
        "__name__": __name__,
        "BaseModel": BaseModel,
        "Field": Field,
        "PrivateAttr": PrivateAttr,
        "Optional": Optional,
        "ServerClient": SimpleNamespace,
        "BaseServerConfig": _Head,
        "setup_server_client_utils": load,
    }
    cls = ast.ClassDef(
        name="RolloutCollectionHelper",
        bases=[ast.Name(id="BaseModel", ctx=ast.Load())],
        keywords=[],
        body=nodes,
        decorator_list=[],
    )
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return SimpleNamespace(helper=namespace["RolloutCollectionHelper"], calls=calls, version=version, fail=fail)


def test_default_keeps_refreshing(client_factory):
    factory = client_factory
    helper = factory.helper()
    first = helper.setup_server_client(_Head(host="one", port=1000))
    second = helper.setup_server_client(_Head(host="one", port=1000))
    assert first is not second
    assert len(factory.calls) == 2


def test_opt_in_reuses_client_and_resolved_config_for_many_prompts(client_factory):
    factory = client_factory
    helper = factory.helper(reuse_server_client=True)
    first = helper.setup_server_client(_Head(host="one", port=1000))
    for _ in range(100):
        reused = helper.setup_server_client(_Head(host="one", port=1000))
        assert reused is first
        assert reused.global_config_dict is first.global_config_dict
    assert len(factory.calls) == 1


def test_explicit_refresh_observes_changed_config(client_factory):
    factory = client_factory
    helper = factory.helper(reuse_server_client=True)
    head = _Head(host="one", port=1000)
    first = helper.setup_server_client(head)
    factory.version[0] = 1
    assert helper.setup_server_client(head).global_config_dict["version"] == 0
    refreshed = helper.setup_server_client(head, refresh=True)
    assert refreshed is not first
    assert refreshed.global_config_dict["version"] == 1
    assert helper.setup_server_client(head) is refreshed
    assert len(factory.calls) == 2


@pytest.mark.parametrize("field,value", [("host", "two"), ("port", 2000)])
def test_mutating_head_config_invalidates_cache(client_factory, field, value):
    factory = client_factory
    helper = factory.helper(reuse_server_client=True)
    head = _Head(host="one", port=1000)
    first = helper.setup_server_client(head)
    setattr(head, field, value)
    assert helper.setup_server_client(head) is not first
    assert len(factory.calls) == 2


def test_helpers_do_not_share_cached_clients(client_factory):
    factory = client_factory
    one = factory.helper(reuse_server_client=True)
    two = factory.helper(reuse_server_client=True)
    assert one.setup_server_client(_Head(host="one", port=1000)) is not two.setup_server_client(
        _Head(host="one", port=1000)
    )
    assert len(factory.calls) == 2


def test_switching_heads_and_explicit_versus_default_never_reuses_wrong_client(client_factory):
    factory = client_factory
    helper = factory.helper(reuse_server_client=True)
    default = helper.setup_server_client()
    assert helper.setup_server_client() is default
    one = helper.setup_server_client(_Head(host="one", port=1000))
    assert one is not default
    two = helper.setup_server_client(_Head(host="two", port=1000))
    assert two is not one
    assert helper.setup_server_client() is not two
    assert len(factory.calls) == 4


def test_failed_refresh_raises_without_poisoning_cached_client(client_factory):
    factory = client_factory
    helper = factory.helper(reuse_server_client=True)
    head = _Head(host="one", port=1000)
    first = helper.setup_server_client(head)
    factory.fail[0] = True
    with pytest.raises(ConnectionError, match="head unavailable"):
        helper.setup_server_client(head, refresh=True)
    assert helper.setup_server_client(head) is first
    factory.fail[0] = False
    factory.version[0] = 2
    assert helper.setup_server_client(head, refresh=True).global_config_dict["version"] == 2


def test_client_cache_is_not_serialized(client_factory):
    helper = client_factory.helper(reuse_server_client=True)
    helper.setup_server_client(_Head(host="one", port=1000))
    assert helper.model_dump() == {"reuse_server_client": True}
