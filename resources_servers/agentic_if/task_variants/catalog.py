# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool contracts of the pinned OpenCode binary, shared by row validation and the Environment Server.

`data/opencode_builtin_schemas.json` is captured from the stock pinned binary, not authored: start it with
`serve --pure` in an isolated directory and config, and keep the `parameters` schemas that its loopback
`/experimental/tool?provider=<provider>&model=<model>` endpoint returns for the tools in BUILTIN_TOOLS. An
OpenAI-compatible test provider with the model IDs `dummy_model` and `gpt-5` covers them all (the latter exposes
`apply_patch` instead of `edit` and `write`); no inference call is needed. Record the binary's `--version` and
SHA-256, which a test ties to the configured pin. The snapshot is the binary's contract, not a promise that every tool
is offered on every request: the Environment Server presents the schemas each request actually carries.
"""

import json
from pathlib import Path


# The built-in tools the pinned OpenCode binary exposes.
BUILTIN_TOOLS = frozenset({"bash", "read", "write", "edit", "glob", "grep", "apply_patch", "task", "todowrite"})
BUILTIN_SCHEMA_SNAPSHOT = json.loads(
    Path(__file__).with_name("data").joinpath("opencode_builtin_schemas.json").read_text()
)
# {tool: (required arguments, optional arguments)}: row validation checks event argument names against these.
BUILTIN_ARGUMENTS = {
    name: (frozenset(schema.get("required", [])), frozenset(schema["properties"]) - set(schema.get("required", [])))
    for name, schema in BUILTIN_SCHEMA_SNAPSHOT["schemas"].items()
}
