# Claude Code Sandboxed Agent

Runs Claude Code inside a sandbox created by the benchmark resources server. The resources server
returns a portable sandbox handle from `/seed_session`. The agent reconnects to that sandbox, runs
Claude Code, and sends the captured response to the resources server for verification.

The harness contains no benchmark setup or scoring logic.

This first implementation requires a connectable sandbox provider such as OpenSandbox. It mirrors
the current OpenCode prototype. A sandbox-server-backed provider can use the same handle contract
when that shared service is available.
