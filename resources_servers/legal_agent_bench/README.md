# Legal Agent Bench

Harvey LAB legal-document tasks run through NeMo Gym's Harbor Environment Server.
LAB Native is the default agent; Hermes, Claude Code, and Codex use the same
prepared tasks and verifier.

See the [LAB evaluation guide](../../fern/versions/latest/pages/evaluation/legal-agent-bench.mdx)
for preparation, judges, harness and sandbox selection, and score interpretation.
The package keeps LAB-specific preparation, tools, and grading here; the shared
Environment Server owns sandbox setup, verification, artifacts, and cleanup.

Harvey LAB is MIT-licensed. [License and attribution](vendor/harvey_labs/README.md)
identify adapted source files and revisions. Task documents and generated indexes
are downloaded during preparation and are not committed.
