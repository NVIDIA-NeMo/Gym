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
"""Framework-agnostic core of the Indian banking environment.

- banking_tools: the 33 deterministic banking tools (in-process, no external tool server).
- engine: per-episode world state (seed_world / apply_tool).
- reward: deterministic partial-credit reward (ACTION x DB x COMMUNICATE).
- judge: prompt construction and verdict parsing for the NL-assertion judge.
- user_sim: system prompt for the role-swapped user simulator.
- tool_schemas: OpenAI function-calling schemas for the tools.
"""
