# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for helper-local resolved client/config reuse."""

from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.rollout_collection import RolloutCollectionHelper


@pytest.fixture
def load_client(monkeypatch):
    def load(head=None):
        return ServerClient(
            head_server_config=head or BaseServerConfig(host="localhost", port=11000),
            global_config_dict=OmegaConf.create({"version": 0}),
        )

    loader = Mock(side_effect=load)
    monkeypatch.setattr("nemo_gym.rollout_collection.setup_server_client_utils", loader)
    return loader


def test_default_keeps_reloading(load_client):
    helper = RolloutCollectionHelper()
    head = BaseServerConfig(host="localhost", port=11000)
    first = helper.setup_server_client(head)
    assert helper.setup_server_client(head) is not first
    assert load_client.call_count == 2


def test_opt_in_reuses_client_and_resolved_config(load_client):
    helper = RolloutCollectionHelper(reuse_server_client=True)
    first = helper.setup_server_client(BaseServerConfig(host="localhost", port=11000))
    second = helper.setup_server_client(BaseServerConfig(host="localhost", port=11000))
    assert second is first
    assert second.global_config_dict is first.global_config_dict
    load_client.assert_called_once()


def test_default_head_is_also_cached_when_opted_in(load_client):
    helper = RolloutCollectionHelper(reuse_server_client=True)
    first = helper.setup_server_client()
    assert helper.setup_server_client() is first
    load_client.assert_called_once_with(None)


@pytest.mark.parametrize("field,value", [("host", "other-host"), ("port", 12000)])
def test_mutating_head_config_invalidates_cache(load_client, field, value):
    helper = RolloutCollectionHelper(reuse_server_client=True)
    head = BaseServerConfig(host="localhost", port=11000)
    first = helper.setup_server_client(head)
    setattr(head, field, value)
    assert helper.setup_server_client(head) is not first
    assert load_client.call_count == 2


def test_helpers_do_not_share_cached_clients(load_client):
    head = BaseServerConfig(host="localhost", port=11000)
    one = RolloutCollectionHelper(reuse_server_client=True)
    two = RolloutCollectionHelper(reuse_server_client=True)
    assert one.setup_server_client(head) is not two.setup_server_client(head)
    assert load_client.call_count == 2


def test_switching_heads_and_explicit_versus_default_invalidates_cache(load_client):
    helper = RolloutCollectionHelper(reuse_server_client=True)
    default = helper.setup_server_client()
    one = helper.setup_server_client(BaseServerConfig(host="localhost", port=11000))
    two = helper.setup_server_client(BaseServerConfig(host="other-host", port=11000))
    assert default is not one
    assert one is not two
    assert helper.setup_server_client() is not two
    assert load_client.call_count == 4


def test_failed_initial_load_is_retried(load_client):
    helper = RolloutCollectionHelper(reuse_server_client=True)
    client = ServerClient(
        head_server_config=BaseServerConfig(host="localhost", port=11000),
        global_config_dict=OmegaConf.create({}),
    )
    load_client.side_effect = [ConnectionError("head unavailable"), client]
    with pytest.raises(ConnectionError, match="head unavailable"):
        helper.setup_server_client()
    assert helper.setup_server_client() is client
    assert helper.setup_server_client() is client
    assert load_client.call_count == 2


def test_failed_head_change_never_returns_previous_heads_client(load_client):
    helper = RolloutCollectionHelper(reuse_server_client=True)
    first_head = BaseServerConfig(host="localhost", port=11000)
    first = helper.setup_server_client(first_head)
    load_client.side_effect = ConnectionError("other head unavailable")
    other_head = BaseServerConfig(host="other-host", port=11000)
    for _ in range(2):
        with pytest.raises(ConnectionError, match="other head unavailable"):
            helper.setup_server_client(other_head)
    assert helper.setup_server_client(first_head) is first
    assert load_client.call_count == 3


def test_client_cache_is_not_serialized(load_client):
    helper = RolloutCollectionHelper(reuse_server_client=True)
    helper.setup_server_client()
    dumped = helper.model_dump()
    assert dumped["reuse_server_client"] is True
    assert not {"_cached_server_client", "_cached_head_server_config"} & dumped.keys()
