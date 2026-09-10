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
"""The nemo-gym session (one rollout) currently executing a tool call.

``app.py`` sets this around ``ToolManager.execute_tool`` / ``cleanup_request`` so the
sandbox backends can key sandboxes by ROLLOUT even though nemo_skills hands them only
its own IPython ``session_id`` (``DirectPythonTool.requests_to_sessions`` maps
request_id -> IPython session; the mapping is private to the tool).
"""

from contextvars import ContextVar


current_session_id: ContextVar[str | None] = ContextVar("ns_tools_current_session_id", default=None)
