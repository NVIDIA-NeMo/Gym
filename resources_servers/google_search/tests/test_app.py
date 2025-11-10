# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.google_search.app import (
    GoogleSearchResourcesServer,
    GoogleSearchResourcesServerConfig,
    box_parser,
)


class TestApp:
    def test_sanity(self) -> None:
        config = GoogleSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            google_api_key="dummy_key",  # pragma: allowlist secret
            google_cx="dummy_cx",
        )
        GoogleSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_box_parser_valid_content(self) -> None:
        """Test box_parser with valid boxed content"""
        # Test basic boxed content
        result = box_parser("The answer is \\boxed{42}")
        assert result == "42"

        # Test with complex content
        result = box_parser("After calculation: \\boxed{x + y = 10}")
        assert result == "x + y = 10"

        # Test with no boxed content
        result = box_parser("No boxed content here")
        assert result is None

        # Test with empty string
        result = box_parser("")
        assert result is None
