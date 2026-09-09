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

from nooa import Agent, CodeActStrategy, strategy


class GymResourceAgent(Agent):
    """Solve tasks using resource methods supplied by Gym at runtime."""

    @strategy(CodeActStrategy())
    async def answer(self, task: str) -> str:
        """Solve `task`.

        Inspect and call the available resource methods when needed, then
        return a concise answer grounded in their results.
        """

        ...
