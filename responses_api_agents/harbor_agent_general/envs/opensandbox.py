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

import ssl
from typing import Any

from harbor.environments.opensandbox import OpenSandboxEnvironment as HarborOpenSandboxEnvironment


class OpenSandboxEnvironment(HarborOpenSandboxEnvironment):
    """
    Harbor environment backed by the OpenSandbox Python SDK.

    Allows turning off TLS verification, while preserving all other behavior.
    """

    def __init__(self, *args: Any, tls_verify: bool = True, **kwargs: Any) -> None:
        self._tls_verify = tls_verify
        super().__init__(*args, **kwargs)

    def _build_connection_config(self, sdk: dict[str, Any]) -> Any:
        config = super()._build_connection_config(sdk)
        if not self._tls_verify:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            config._ssl_context = context
        return config
