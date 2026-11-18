# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A provider block's ``owner_concurrency``: how many sandboxes one owner starts and stops at once."""

import pytest

from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_owner_concurrency


class TestOwnerConcurrency:
    def test_a_block_without_it_is_unbounded(self) -> None:
        concurrency = resolve_provider_owner_concurrency({"docker": {}})

        assert (concurrency.starts, concurrency.stops) == (None, None)

    def test_a_named_block_sets_starts_and_stops_apart(self) -> None:
        named = {"docker_sandbox": {"docker": {}, "owner_concurrency": {"starts": 64, "stops": 16}}}

        concurrency = resolve_provider_owner_concurrency("docker_sandbox", named)

        assert (concurrency.starts, concurrency.stops) == (64, 16)
        assert resolve_provider_config("docker_sandbox", named) == {"docker": {}}, "it is not a provider key"

    @pytest.mark.parametrize("value", [{"starts": 0}, {"stops": -1}, {"start": 4}])
    def test_an_invalid_value_is_refused(self, value: dict) -> None:
        with pytest.raises(ValueError, match="owner_concurrency"):
            resolve_provider_owner_concurrency({"docker": {}, "owner_concurrency": value})
