# NOOA Agent

The NOOA agent adapter provides typed configuration for constructing NOOA agents and invoking them through
asynchronous Python adapters that receive validated NeMo Gym Responses requests.

## Embedded CodeAct scope

Embedded CodeAct persists for the duration of one invocation. `execute_python` is not computation-only: it can
read and modify files, start processes, use the network, and change other node state. Embedded mode provides no
sandbox protection. Only injected resource methods are Gym/verifier-observable and authoritative; direct CodeAct
side effects belong to NOOA and are not reliably visible to verification or replay. The trajectory distinguishes
the outer `execute_python` call from nested resource calls.

## Sandboxed execution

`nooa.execution_mode: sandboxed` runs the complete NOOA agent in the task sandbox supplied by Resources.
NOOA stages its own runtime and closes its worker before disconnecting; Resources owns sandbox creation and teardown.
Native sessions use the NOOA-specific `nooa_single_agent_turn` environment.
