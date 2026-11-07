# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Original Harvey LAB portions remain under MIT; NVIDIA modifications follow.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LAB tool schemas adapted from the pinned upstream source.

https://github.com/harveyai/harvey-labs/blob/
9e9fa7fe126103de1fceb2227b7c70bf59b67f73/lab_core/harness/tools.py
"""

CONTAINER_TOOL_RUNNER_PATH = "/opt/legal-agent-bench/container_tool_runner.py"

TOOL_DEFINITIONS = [
    {
        "name": "bash",
        "description": "Execute a bash command and return its output. Use for running scripts, installing "
        "packages, file manipulation, and any shell operation. The working directory persists "
        "between calls.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
    {
        "name": "read",
        "description": "Read a file from the input directory or workspace. Handles .docx, .xlsx, .pptx, .pdf, and "
        "plain text — extraction is automatic; use this rather than a skill just to read. Use "
        "offset and limit for large files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path (resolved against workspace then input directory) or absolute path",
                },
                "offset": {"type": "integer", "description": "Line number to start reading from (0-based). Optional."},
                "limit": {"type": "integer", "description": "Maximum number of lines to return. Optional."},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write",
        "description": "Write a plain markdown file (typically `response.md`) to the output directory. For binary "
        "deliverables (.docx, .xlsx, .pptx), use the file-type skill manuals — do not write raw "
        "markdown to a binary extension. Creates parent directories if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path under the output directory (e.g., 'response.md')",
                },
                "content": {"type": "string", "description": "Markdown content to write"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit",
        "description": "Perform exact string replacement in a file you have already created or read. The "
        "old_string must appear exactly once unless replace_all is true. Use for incremental "
        "refinement, not first-time writes.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to modify"},
                "old_string": {"type": "string", "description": "The exact text to find and replace"},
                "new_string": {"type": "string", "description": "The replacement text"},
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace all occurrences. Default false.",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern, sorted by modification time. Defaults to searching the "
        "input directory. Prefer this over `bash find` or `bash ls` for file discovery.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., '**/*.docx', 'src/**/*.py')",
                },
                "path": {"type": "string", "description": "Directory to search in. Defaults to the input directory."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search file contents using regex patterns. Defaults to searching the input directory. "
        "Returns matching file paths or matching lines with context.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to the input directory.",
                },
                "glob": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py', '*.docx')"},
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output format. 'content' shows matching "
                    "lines, 'files_with_matches' shows file "
                    "paths, 'count' shows match counts. Default: "
                    "'files_with_matches'.",
                },
            },
            "required": ["pattern"],
        },
    },
]

FINISH_TOOL_DEFINITION = {
    "name": "finish",
    "description": "Signal that the task is complete after all required deliverables have been created in the "
    "output directory. Do not call this until there is no more work to do.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief summary of the completed work."},
            "deliverables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths of the deliverable files you produced, "
                "relative to the output directory (e.g. "
                "['response.md', 'memo.docx']).",
            },
        },
        "required": ["summary"],
    },
}


def get_all_tool_definitions(*, enable_finish: bool = True) -> list[dict]:
    """Return the six workspace tools and the optional explicit completion tool."""
    return [*TOOL_DEFINITIONS, *([FINISH_TOOL_DEFINITION] if enable_finish else [])]
