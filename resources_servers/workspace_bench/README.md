# Workspace-Bench resources server

Gives each task a sandbox with its input files in `/workspace/input`. After the agent runs, a separate judge sandbox
runs the pinned upstream agent-as-a-judge on `/workspace/output`. Reward is passed rubrics over total rubrics.

- Inputs are snapshotted before the agent runs, so the judge compares against the original files.
- Judge API failures go to the failures file instead of scoring zero.
- Set `artifact_root` to save each trial. Passing a saved `artifact_id` to `/verify` regrades it without rerunning
  the agent.
