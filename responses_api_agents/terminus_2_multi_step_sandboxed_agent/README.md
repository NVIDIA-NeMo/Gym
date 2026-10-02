# Terminus 2 over Harbor multi-step tasks

Runs Gym's sandboxed Terminus 2 loop (`responses_api_agents/terminus_2_sandboxed_agent`) once
per Harbor `[[steps]]` entry inside one task container, the way Harbor's `MultiStepTrial` does:
prepare the step, run the agent with the step's own budget and a fresh conversation, verify the
step, stop when a step misses its `min_reward` gate. The model loop is inherited unchanged.

The resources server owns the task files and the container and must expose:

| Endpoint | Returns |
| --- | --- |
| `POST /seed_session` | `sandbox_handle`, `steps: [{name, agent_timeout_s}]` |
| `POST /prepare_step {step_index}` | `instruction`, `agent_timeout_s`, `setup_ok` |
| `POST /verify_step {step_index}` | `stop` |
| `POST /verify` | the aggregated verify response |

Step 0 uses the row's `responses_create_params.input`; later steps use the instruction returned
by `/prepare_step`. `resources_servers/oragentbench` is the first server on this protocol. Configure
it exactly like `terminus_2_sandboxed_agent`; see
`resources_servers/oragentbench/configs/oragentbench.yaml` for a complete example.
