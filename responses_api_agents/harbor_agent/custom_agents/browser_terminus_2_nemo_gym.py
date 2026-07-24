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
"""DOM-only Terminus-2 baseline for packaged Worldsims browser tasks."""

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .terminus_2_nemo_gym import Terminus2NemoGym


_BROWSER_PREAMBLE = """\
This is a browser-use task. A packaged Chromium browser is available through:

  browser_open 'https://example/path'

The command loads the URL in Chromium at 1440x900 through the task's packaged
website proxy and prints the rendered DOM. Each invocation starts a new browser
session, so it cannot click, type, or preserve interactive state. Use it for
every read-only page that supports your answer. Do not use curl, wget, Python
HTTP clients, search engines, local website files, hidden/debug endpoints, or
internal addresses as evidence. Do not inspect the implementation of
browser_open. Follow the task's source and output contract exactly, and write
the requested JSON under /app.

"""


class BrowserTerminus2NemoGym(Terminus2NemoGym):
    """Give Terminus-2 a stateless rendered-DOM browser command."""

    @staticmethod
    def name() -> str:
        return "browser-terminus-2-nemo-gym"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await super().run(_BROWSER_PREAMBLE + instruction, environment, context)
