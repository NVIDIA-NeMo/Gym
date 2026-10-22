# Simple Strands Agent

The pinned Simple Strands Agent dependency is installed with the agent server. The native system
prompt is retained and additional system prompts are appended.

Set `repo_dir` to an existing checkout to run tools there. With `HarnessAgent`,
`repo_dir: "."` uses the task's working directory. Run artifacts stay under
`workspace_root` and cleanup preserves the checkout and its edits.
