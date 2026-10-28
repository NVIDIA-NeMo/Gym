# Terminal-Bench 4 session environment

Run the existing pinned TB4 tasks and verifier with OpenCode:

```sh
gym eval prepare --benchmark terminal_bench_4/opencode
gym eval run --benchmark terminal_bench_4/opencode \
  --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/tb4-opencode/rollouts.jsonl \
  ++use_absolute_ip=true
```

Configure the sandbox deployment as described in
[the TB4 recipe](../../benchmarks/terminal_bench_4/README.md).
The Gym model server must be reachable from inside the task container.

This server accepts TB4's existing prepared rows. Resources provisions the task,
returns its instruction and sandbox access, and retains ownership of task and
verifier containers. The environment opens an agent session, executes within the
task deadline, requires a confirmed agent close, and then asks Resources to grade.
Failed setup or unconfirmed process termination leaves the task ungraded and
closes Resources. Duplicate requests with the same owner and rollout share the
same in-flight execution; conflicting requests fail. Retries after a process
restart do not resume an agent conversation. Provider TTL handles abandoned
containers after a process crash.

Task-local MCP servers and skills are supplied through the shared
`/tmp/.nemo-gym/task.json` convention. The environment requires the agent to
advertise support when a task declares them. OpenCode enables native MCP tools
and explicit skill paths, and checks MCP connection readiness before its first
model call.

The mini-SWE recipe keeps its existing execution path. OpenCode's prompts, tools,
compaction and token limits differ, so the profiles do not establish score parity.
