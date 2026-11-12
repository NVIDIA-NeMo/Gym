# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.opencode_multimodal_benchmarks.tests import tool_adapter_checks


class TestAdapter(tool_adapter_checks.AdapterChecks):
    component = "opencode_mcqa_tool_rewards"
    parent = "mcqa"
    invalid_fields = {"grading_mode": "invalid", "options": "A"}
