# GDPVal resources server

Pairs with `responses_api_agents/gdpval_sandboxed_agent` over Gym's standard
session-cookie mechanism, following the SWE/Terminal Bench division of roles.

- `/seed_session`: look up `responses_create_params.metadata.task_id`, create
  the configured sandbox, stage public inputs, and return `sandbox_handle`.
- `/verify`: accept the agent's trajectory and task ID, download artifacts and
  traces, stop the sandbox, run the existing structured GDPVal rubric judge, and
  return its reward and metadata. Empty submissions and supported policy limits
  retain their existing zero-reward behavior; infrastructure errors fail the run.
- `/cleanup_session`: idempotently stop the session's sandbox and remove state.

Rubrics, judge credentials and host artifact paths stay on the resources side.
The agent owns OpenHands installation and execution. Configuration and deployment
commands live in the package's root README and `configs/gdpval_sc_mimo.yaml`.

Session state lives in this process, as in the upstream reference. A process
restart loses active sessions; the sandbox TTL bounds orphan lifetime. The
implementation does not claim stateless reverification support.
