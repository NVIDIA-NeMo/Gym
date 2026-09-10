# Description
This is a resources server for executing NeMo Skills tools (e.g., stateful Python code execution) with math verification via math_with_judge.

It integrates with the NeMo Skills ToolManager to dynamically load and execute tools, maintaining stateful sessions across multiple tool calls within a rollout.

# Example usage

## Running servers
The following are example commands for running this resources server with the simple agent and a vLLM model:
```bash
gym env start \
    --resources-server ns_tools \
    --resources-server math_with_judge \
    --model-type vllm_model \
    --model-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-8B \
    ++math_with_judge.resources_servers.math_with_judge.should_use_judge=false \
    ++math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
```

Then, rollouts can be collected using a command such as the following:
```bash
gym eval run --no-serve \
    --agent ns_tools_simple_agent \
    --input resources_servers/ns_tools/data/example.jsonl \
    --output data/ns_tools_rollouts.jsonl \
    --limit 5
```

## Disaggregated execution with OpenSandbox

Use `resources_servers/ns_tools/configs/ns_tools_sandbox.yaml` to opt in. Set
`OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, and `NS_SANDBOX_IMAGE` for your
service and a NeMo Skills sandbox image containing `/start-with-nginx.sh` and
`curl`:

```bash
gym env start \
  --config resources_servers/ns_tools/configs/ns_tools_sandbox.yaml \
  --resources-server math_with_judge \
  --model-type vllm_model
```

The config composes Gym's shipped OpenSandbox provider settings, including its
TLS, 502 retry and background polling behavior. Service requests run through
`sandbox.exec` as localhost HTTP requests inside the sandbox; ns_tools does not
open a second HTTP client or resolve externally exposed service endpoints.
The sandbox sets `EXECD_API_GRACE_SHUTDOWN=50ms` for short command responses.

Eight direct-created sandboxes are shared by default. Sessions stay on one
sandbox for stateful Python execution; unhealthy sandboxes are replaced and
session restoration follows `disable_session_restore`. Use
`NS_SANDBOX_POOL_SIZE` for capacity and `NS_SANDBOX_TTL_S` for lifetime. An empty
`NS_SANDBOX_POOL_REF` selects direct creation; set a name to claim a prewarmed
server-side pool. `NS_SANDBOX_POOL_FALLBACK=false` disables direct fallback.
A prewarmed template must already run the NeMo Skills service and set the same
execution grace. The ordinary `ns_tools.yaml` keeps colocated execution.

### One sandbox per rollout session

`resources_servers/ns_tools/configs/ns_tools_session_sandbox.yaml` selects
`sandbox_type: sandbox_per_session` (`session_sandboxes.py`): nothing is shared
between rollouts. The first tool call of a rollout creates its own sandbox, every
later call of that rollout reuses it, and the sandbox is deleted as soon as the
rollout is verified (`/verify` -> nemo_skills `cleanup_request` -> `end_session`).
Rollouts that never reach `/verify` are deleted after
`NS_SANDBOX_SESSION_IDLE_TIMEOUT_S` (default 3600) of inactivity, every live
sandbox is deleted at server shutdown, and `NS_SANDBOX_SESSION_TTL_S` (default
7200) is the cluster-side backstop. Sandboxes are keyed by the nemo-gym session
of the rollout (published through `session_context.current_session_id`), so an
IPython session restarted by an execution timeout stays on the same sandbox.

Size the pod for ONE Python session: limits `NS_SANDBOX_SESSION_CPU_LIMIT` /
`NS_SANDBOX_SESSION_MEM_LIMIT_MIB` / `NS_SANDBOX_SESSION_DISK_LIMIT_GIB` (4 / 8192 /
10) and requests `NS_SANDBOX_SESSION_CPU_REQUEST` / `..._MEM_REQUEST_MIB` /
`..._DISK_REQUEST_GIB` (1 / 2048 / 5; one IPython session is one Python process). The
pod runs a single HTTP worker (`NUM_WORKERS=1`) with OMP/BLAS thread caps derived from
the cpu limit. Every sandbox carries the pod labels
`nemo.nvidia.com/resources: custom` and `purpose: ns-tools-per-session` (`metadata`,
extendable in the yaml; the provider adds its `nemo-gym.nvidia.com/*` attribution
labels); the first one exempts the
pod from the request-clamping admission policy on the NeMo cells so the requests
are honoured. `NS_SANDBOX_CREATE_CONCURRENCY` (64) bounds concurrent creates at a
batch start; `NS_SANDBOX_POOL_SIZE` is rejected in this mode. The first tool call of a
rollout also creates its sandbox: `NS_SANDBOX_CREATE_WAIT_TIMEOUT_S` (270) bounds how
long that call waits for the create (keep it below the agent-to-server HTTP timeout,
300 s by default; on expiry the model gets an ordinary tool timeout while the create
finishes and the next call reuses it) and `NS_SANDBOX_FIRST_REQUEST_TIMEOUT_S` (60)
is the HTTP timeout of the first request on a fresh sandbox (IPython kernel spawn). `NS_SANDBOX_POOL_REF`
may still name a prewarmed pool to claim (one claim per session).

`NS_SANDBOX_TRANSPORT` selects how tool requests reach the sandbox's NeMo-Skills
server: `exec` (default; `curl` through the sandbox exec API) or `http` (direct
requests to the service port through the OpenSandbox endpoint proxy, one round trip
per call — roughly 3x lower per-call latency than exec).

## Sample data format
Each sample requires:
- `question`: The math question being asked
- `expected_answer`: The expected answer (verified by math-verify)
- `responses_create_params`: The request parameters for the model, including tool definitions

Example with Python tool:
```json
{
  "question": "What is 2 + 2?",
  "expected_answer": "4",
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a helpful assistant. Put your final answer in \\boxed{}."},
      {"role": "user", "content": "What is 2 + 2? Use Python to calculate this."}
    ],
    "tools": [{
      "type": "function",
      "name": "stateful_python_code_exec",
      "description": "Execute Python code in a stateful environment.",
      "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"]
      }
    }]
  }
}
```

## Preparing datasets

The `prepare_dataset.py` script transforms source datasets into the JSONL format required by nemo-gym.

### Regenerating compmath_prepared.jsonl

The `data/compmath_prepared.jsonl` file is generated from the nemo-skills comp-math-24-25 dataset. To regenerate it:

```bash
python prepare_dataset.py \
    --input /path/to/nemo_skills/dataset/comp-math-24-25/test.txt \
    --output data/compmath_prepared.jsonl \
    --prompt_config generic/math \
    --tools nemo_skills.mcp.servers.python_tool::DirectPythonTool \
    --verifier_type math_with_judge
```

### Validating example data

To validate the example data and regenerate metrics:
```bash
gym dataset collate \
    --resources-server ns_tools \
    --config responses_api_models/openai_model/configs/openai_model.yaml \
    --output-dir data/ns_tools \
    --mode example_validation
```

# Licensing information
Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- nemo-skills-tools: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)
