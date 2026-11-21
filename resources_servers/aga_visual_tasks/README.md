# AGA original training resources

Lends the original AGA Code-assisted VQA / Agent-X task an isolated image workspace, seeded with the original public
images under `/images`, and grades the final answer with the original math / MCQA / string graders. See
[the task contract](../../environments/aga_visual_tasks/README.md) and `configs/multiharness.yaml`, which runs the task
under OpenCode, Pi, Codex, Claude Code and OpenClaw.

`source-provenance.json` pins the original files that this server adapts.
