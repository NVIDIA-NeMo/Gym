# AgentVista verifier

This Resources Server checks the original task/image binding, invokes the pinned
AgentVista answer judge in the separate native worker runtime, and aggregates
scores. It is stateless: the verify request contains native worker evidence and
hash-bound task metadata, with no live sandbox or tool state to own. Gym's base
seed/close implementation therefore acknowledges stateless sessions.

Use the complete pairings in `benchmarks/agentvista_default/config.yaml` and
`benchmarks/agentvista_opencode/config.yaml`. Configure judge credentials only in
the service environment (`VERIFIER_API_KEY`, `VERIFIER_END_POINT`, and
`VERIFIER_MODEL_NAME`). The agent does not receive them. Provider, parse, binding,
and worker failures stay masked; measured wrong answers count as zero. Verification
has a bounded subprocess deadline and kills its worker on cancellation. Artifacts
and library caches remain under `output_root`.

Run focused tests with `pytest resources_servers/agentvista/tests`. This server
uses the same separately pinned AgentVista source and native runtime as the agent;
see its README for provisioning and source/license provenance.

The manifest-native `VERIFIER_FIXTURE` exercises the production judge adapter with
only the external provider/module replaced. It requires no private environment,
dataset downloads or model calls. Run the full CI path, including fixtures, with
`gym env test +entrypoint=resources_servers/agentvista +should_validate_data=true`.
Prepared metadata is validated by `task_data.py`; the benchmark's agent reference
binds that schema during dataset collation.
