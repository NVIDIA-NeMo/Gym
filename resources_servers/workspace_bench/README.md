# Workspace-Bench resources server

Creates a sandbox per task with the task's input files in `/workspace/input`, keeps rubrics on the server, and grades
`/workspace/output` by running the pinned upstream agent-as-a-judge in a separate sandbox. Task reward is passed rubrics
divided by total rubrics, the upstream rubric pass rate.

Input files and metadata are snapshotted before the agent runs, so the judge compares outputs against the original
references even if the agent modifies its inputs. Judge API failures go to the failures file instead of being scored
as zero. A received but unparseable verdict fails its rubrics, as upstream does.

## Optional artifact retention

Set `artifact_root` to save each trial's inputs, metadata, workspace, request, and result. Passing a saved
`artifact_id` to `/verify` regrades that trial without rerunning the agent.
