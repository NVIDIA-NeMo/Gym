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


class WeatherAgent(Agent):
    """Answer weather questions using only Gym-provided environment tools."""

    @strategy(CodeActStrategy())
    async def answer(self, question: str) -> str:
        """Answer `question`.

        Extract the requested city, call `self.gym_tools.get_weather(city=...)`, and
        return a concise answer grounded in the tool result.
        """

        ...
