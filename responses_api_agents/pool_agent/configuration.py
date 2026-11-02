# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import JsonValue


def pool_config(*, base_url: str, model: str, context_window: int, timeout: float = 300) -> dict[str, JsonValue]:
    """Configure Pool tools, context compaction, and Gym model routing."""
    # `pool exec --agent-config-file` does not merge with pool's defaults, so this mirrors them.
    config = {
        "max_steps": 0,
        "enabled_tools": [
            "todo_action",
            "read",
            "edit",
            "write",
            "get_diagnostics",
            "question",
            "shell",
            "shell_list",
            "shell_tail",
            "shell_status",
            "shell_kill",
            "shell_send",
            "shell_wait",
            "skill",
            "subagent",
            "switch_mode",
            "list_secrets",
            "exit",
        ],
        "http_timeout": f"{timeout}s",
        "enable_no_tool_call_retry": False,
        "enable_truncation_recovery": True,
        "enable_multi_tool_calls": True,
        "enable_tool_call_loop_detection": True,
        "enable_todo_reminders": True,
        "parallel_tool_calls": {"tools": ["subagent"]},
        "tools": {
            "edit": {"fuzzy_match_threshold": 0.05},
            "read": {"max_lines": 500, "line_num_separator": "|◊|"},
            "write": {"line_num_separator": "|◊|"},
            "shell": {"foreground_timeout": "120s"},
        },
        "model": {
            "provider": {
                "openai": {
                    "base_url": base_url,
                    "api_key": "dummy_key",  # pragma: allowlist secret
                    "model_id": model,
                    "use_streaming": False,
                }
            },
            "prompt": {"hidden_tool_names": ["exit"]},
            "max_completion_retries": 2,
            "max_completion_retries_transient": 3,
            "exit_tool_on_stop": True,
            "streaming_events_interval": "100ms",
        },
        "memory": {
            "compact": {
                "TriggerCompressionTokenCount": int(context_window * 0.8),
                "MaxSummarizeTokenCount": int(context_window * 0.7),
                "MinRetainedSteps": 5,
            }
        },
    }
    return config
