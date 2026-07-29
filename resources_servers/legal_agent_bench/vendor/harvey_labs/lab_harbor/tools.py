# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Agent-visible tool schemas used by the Harbor agent."""

# The container-side implementation lives in lab_harbor/container_tool_runner.py
# and is copied into generated Harbor task environments.

MAX_READ_LINES = 200
CONTAINER_TOOL_RUNNER_PATH = "/opt/legal-agent-bench/container_tool_runner.py"


TOOL_DEFINITIONS = [
    {
        "name": "bash",
        "description": (
            "Execute a bash command and return its output. Use for running "
            "scripts, file manipulation, and shell operations that need the "
            "preinstalled runtime tools. The working directory persists between "
            "calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "read",
        "description": (
            "Read a file from the filesystem. Supports all common formats: "
            ".docx (converted to markdown), .xlsx (converted to text tables), "
            ".pptx (converted to markdown), .pdf (extracted text and tables), "
            "and plain text files. Use line_count_only first for large files, "
            "then use offset and limit to read the file in chunks. A read call "
            f"returns at most {MAX_READ_LINES} parsed text lines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Filename or relative path. The Harbor tool runtime checks the "
                        "workspace and the VDR. Avoid absolute paths."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of lines to return. Optional. Values "
                        f"above {MAX_READ_LINES} are capped. Set limit to 0 "
                        "to intentionally read from offset to the end of the "
                        "file without the page cap; explicit null is also "
                        "accepted by the Harbor tool runtime."
                    ),
                },
                "line_count_only": {
                    "type": "boolean",
                    "description": (
                        "If true, return only the number of parsed text lines "
                        "in the file. Use before reading large files."
                    ),
                    "default": False,
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write",
        "description": (
            "Write content to a file. Creates parent directories if needed. "
            "Use for producing plain-text deliverables and file output. For "
            "Word documents, use write_docx instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Relative filename to write under the output "
                        "directory. The Harbor tool runtime routes relative paths to the "
                        "output dir automatically. Do not use absolute paths."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The content to write",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "write_docx",
        "description": (
            "Create a valid Microsoft Word .docx deliverable from Markdown. "
            "Use this for ordinary Word document outputs instead of write or bash. "
            "Supports headings, paragraphs, bold/italic text, bullet and "
            "numbered lists, and Markdown tables. For more complex Word "
            "operations such as redlines, comments, custom styles, exact "
            "template formatting, or advanced layout, use bash with the Word "
            "docx skill and the preinstalled document libraries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Relative .docx filename to write under the output "
                        "directory. If no extension is provided, .docx is "
                        "appended automatically. Do not use absolute paths."
                    ),
                },
                "markdown": {
                    "type": "string",
                    "description": "Markdown content to convert into a Word document.",
                },
            },
            "required": ["file_path", "markdown"],
        },
    },
    {
        "name": "edit",
        "description": (
            "Perform exact string replacement in a file. The old_string must "
            "appear exactly once in the file (unless replace_all is true). "
            "Use for targeted modifications without rewriting the entire file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to modify",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text",
                },
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
        "description": (
            "Find files matching a glob pattern. Returns matching file paths "
            "sorted by modification time. Use for targeted file discovery."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., '**/*.docx', 'src/**/*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to working directory.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search file contents using regex patterns. Returns matching file paths or matching lines with context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to working directory.",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py', '*.docx')",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "Output format. 'content' shows matching lines, "
                        "'files_with_matches' shows file paths, 'count' shows "
                        "match counts. Default: 'files_with_matches'."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
]


def get_all_tool_definitions() -> list[dict]:
    """Return the canonical Harbor agent tool schema."""
    return list(TOOL_DEFINITIONS)
