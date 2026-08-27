# Evaluate agents securely — NeMo Gym + OpenShell

A 3-minute live demo. Hermes Agent, driven by **Nemotron 3.5 Lightning**, solves a real
SWE-bench Verified task — inside an OpenShell sandbox whose network egress is denied by
default at the gateway.

The argument the demo makes, in one line: **evaluating an agent means running
model-authored code you have not read, and the isolation for that should be one config
flag, not a rewrite of your harness.**

## Why this wiring

`anyswe_agent` does something most harnesses don't: it runs the *agent itself* inside the
task sandbox, not just the agent's shell commands. `agent_server_module` points at
`responses_api_agents.hermes_agent.app`, so Hermes boots in the container, and its
terminal and file tools never touch your host. The only thing crossing the boundary is
the model call to Nemotron 3.5 Lightning.

That means the blast radius of a bad rollout is the sandbox, and the sandbox's egress is
whatever the OpenShell gateway policy says it is.

The detail that makes the policy tight: the in-sandbox agent calls the **Gym model
server** on your host, and the Gym model server calls Lightning. The sandbox never needs
a route to the model provider — so the allowlist is one internal host, plus whatever the
task itself genuinely needs.

| Layer | Component | Config |
|---|---|---|
| Model | Nemotron 3.5 Lightning 30B-A3B | your own `env.yaml` ([T-30 prep](#t-30-prep)) |
| Agent harness | Hermes Agent, embedded in the task container | [`anyswe_hermes.yaml`](../../responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml) |
| Isolation | OpenShell gateway (container/MicroVM + egress policy) | [`openshell.yaml`](../../nemo_gym/sandbox/providers/openshell/configs/openshell.yaml) |
| Verifier | SWE-bench `get_eval_report`, in a fresh sandbox | `anyswe_agent` |

## T-30 prep

Do all of this well before you present. Nothing here is worth watching.

```bash
uv sync --extra openshell
```

Start an OpenShell gateway. The Docker compose deployment is fine for a laptop demo —
plaintext control plane on `localhost:8080`, health on `8081`:

```bash
git clone https://github.com/NVIDIA/OpenShell && cd OpenShell/deploy/docker && docker compose up -d
```

Point Gym at Lightning. If your root `env.yaml` already resolves `policy_model_name` to a
Lightning id, you are done — skip this.

Otherwise add one of the two blocks below to your **own** `env.yaml` at the Gym repo root.
This demo deliberately ships no env file: `env.yaml` is gitignored repo-wide
([.gitignore:229](../../.gitignore:229)) because it holds live keys, and a committed
template is just an invitation to fill one in and push it. Keep it out of the repo.

Fastest path — a hosted gateway:

```yaml
policy_base_url: https://inference-api.nvidia.com
policy_api_key: ${oc.env:POLICY_API_KEY}
policy_model_name: nvidia/nvidia/nemotron-3.5-lightning   # gateway routing id, not the HF id
responses_create_params:
  temperature: 1.0
  top_p: 0.95
```

Or serve it yourself — required if you want the published Lightning numbers, since the
score-determining flags can only be set this way:

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
responses_create_params:
  temperature: 1.0
  top_p: 0.95
```

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --port 8000 \
  --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 --max-model-len 262144
```

`--reasoning-parser nemotron_v3` is mandatory for a tool-calling harness. Without it
`<think>` stays inline, Hermes replays a different context every turn, and the agent looks
erratic on stage for a reason that has nothing to do with the model.

Note the endpoint above is **not** what the sandbox talks to. The in-sandbox Hermes agent
calls the Gym model server on your host, and the Gym model server calls this endpoint — so
the sandbox needs no route to any model provider. That is what keeps
[`egress-policy.yaml`](egress-policy.yaml) as short as it is.

Then, from the Gym repo root:

```bash
export ANYSWE_CONTAINER_FORMATTER='docker.io/swebench/sweb.eval.x86_64.{instance_id}'
export NEMO_GYM_SANDBOX_MODEL_BASE_URL=http://host.docker.internal:13909
demos/openshell-secure-eval/demo.sh prep
```

`NEMO_GYM_SANDBOX_MODEL_BASE_URL` is the **Gym model server** as addressed from inside a
sandbox — not your vLLM box and not the hosted gateway. It has to match the allow rule in
[`egress-policy.yaml`](egress-policy.yaml), which is literal (that file is read with a
plain `yaml.safe_load`, so `${oc.env:...}` will *not* interpolate there — edit the values
directly).

`prep` checks the gateway, builds a 2-task dataset, and pulls the SWE-bench task images.
That image pull is several minutes of dead air if you leave it for the live run.

**Where `hermes-agent` has to live.** Not on your host. `anyswe_agent` never imports the
agent module — it passes the module and class names into the sandbox as strings
(`NGSWE_AGENT_MODULE`, [app.py:346](../../responses_api_agents/anyswe_agent/app.py:346)),
and [agent_runner.py:93](../../responses_api_agents/anyswe_agent/agent_runner.py:93) does
the `importlib.import_module` *inside* the container against
`/agent_deps_mount/bin/python`. Your host venv can be entirely innocent of `model_tools`.

What matters is the sandbox runtime, and `agent_runtime_source` controls it:

| Value | Behavior |
|---|---|
| `baked` (config default) | Expects the task image to already ship `/agent_deps_mount/bin/python` with hermes-agent in it. Stock `swebench/sweb.eval.*` images do **not** — the run dies with `task image does not contain /agent_deps_mount/bin/python`. |
| `auto` (what `demo.sh up` sets) | Builds the runtime once via `setup_scripts/hermes_agent_deps.sh` — portable python + NeMo Gym + the pinned hermes-agent — and uploads it per sandbox. |

`demo.sh up` passes `agent_runtime_source=auto` for exactly this reason. Override with
`RUNTIME_SOURCE=baked` if you have your own baked images. The first `auto` build is slow
and cached, so trigger it during prep rather than on stage:

```bash
demos/openshell-secure-eval/demo.sh up   # let it build the runtime once, then Ctrl-C
```

**Rehearse `probe` once against your gateway.** The policy shape in
[`egress-policy.yaml`](egress-policy.yaml) is checked against `openshell` 0.0.92 — it
parses cleanly into `sandbox_pb2.SandboxPolicy` (the file header has the one-line
command). What that check does *not* cover is field *values*: `protocol`, `enforcement`
and `access` are plain proto strings validated by the gateway, not by protobuf. A
rehearsal against your actual gateway is the only thing that settles those.

## The 3 minutes

### 0:00–0:30 — the problem

> "To evaluate a coding agent, you hand a model a shell and let it write and run code you
> haven't reviewed. Everyone does this. Most people do it on a laptop or a shared CI box
> with full network access. Here's what it looks like when you don't."

Have [`anyswe_hermes.yaml`](../../responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml)
on screen. Point at `agent_server_module: responses_api_agents.hermes_agent.app`.

> "This is Hermes Agent. NeMo Gym runs it *inside* the task container — the agent, not
> just its commands. And this line —" `sandbox_provider: sandbox` "— is the whole
> isolation contract. It names a provider. It doesn't pick one."

### 0:30–1:00 — the swap

```bash
demos/openshell-secure-eval/demo.sh up
```

While it starts, show the command it runs:

```bash
gym env start \
  --config responses_api_agents/anyswe_agent/configs/anyswe_hermes.yaml \
  --config nemo_gym/sandbox/providers/openshell/configs/openshell.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

> "Three configs: the agent, the sandbox provider, the model. Every shipped provider
> binds the same name, so moving from Docker to OpenShell to a MicroVM is swapping the
> middle line. The agent config doesn't change. Nemotron 3.5 Lightning is on the third."

### 1:00–2:00 — the run

```bash
demos/openshell-secure-eval/demo.sh eval
```

Narrate the log as OpenShell provisions the sandbox and Hermes starts working. What to
point at, in order:

1. The gateway creating the sandbox and the readiness probe passing.
2. Hermes exploring `/testbed`, reproducing the bug, editing, running tests — all of it
   gateway-side.
3. Your host: nothing was installed, nothing was written outside the sandbox.

> "Lightning is a 30B-A3B model — 3 billion active parameters. It's fast enough that the
> agent loop is the interesting cost, not the tokens. That matters when a real eval is
> 500 tasks times 5 repeats."

### 2:00–2:40 — the money shot

```bash
demos/openshell-secure-eval/demo.sh probe
```

```
https://pypi.org/simple/            rc=0    ALLOWED
https://pastebin.com                rc=1    DENIED BY GATEWAY POLICY
```

> "Same sandbox, same policy, no agent in the loop — so this is the gateway refusing, not
> a model politely declining. There's no deny-list here: OpenShell's model is allowlist-
> by-construction, so anything I didn't name has no route out. pypi is on the list because
> the task genuinely needs it. If the agent tries to exfiltrate the repo, or a transitive
> dependency phones home during `pip install`, it fails at the gateway. That's the
> difference between a policy and a hope."

Then show [`egress-policy.yaml`](egress-policy.yaml) and land the point that makes it
tight:

> "Notice what's *not* on this allowlist: the model. The agent inside the sandbox talks to
> the Gym model server on my host, and my host talks to Lightning. The box running
> unreviewed model-authored code has no route to any inference endpoint at all."

### 2:40–3:00 — the close

Show the score in `results/openshell-secure-eval/rollouts.jsonl`.

> "Real patch, real SWE-bench grading, in a box that could not have reached the internet
> if it wanted to. Isolation was one config line — so the secure setup is the *convenient*
> one, and that's the only kind anybody actually keeps using."

## If it breaks

| Symptom | Fix |
|---|---|
| `task image does not contain /agent_deps_mount/bin/python` | `agent_runtime_source` is `baked` against a stock image; use `auto` |
| `No module named 'model_tools'` *inside the sandbox* | The uploaded runtime is stale or incomplete; delete `responses_api_agents/anyswe_agent/.anyswe_agent_deps*.tar.gz` and let `auto` rebuild. Harmless on the host — nothing there imports it |
| `OpenShell SDK is required` | `uv sync --extra openshell` |
| `Failed to parse version field` | `version` is a proto int — `1`, not `"v1"` |
| `no field named "network"` | The field is `network_policies` (a map of named rules), not `network` |
| Gateway connection fails | `OPENSHELL_GATEWAY_ENDPOINT` is the gRPC port (8080), not the health port (8081) |
| Policy rejected at create | Gateway `SandboxPolicy` proto differs — see the note in [`egress-policy.yaml`](egress-policy.yaml) |
| `${oc.env:...}` appears literally in a policy error | The policy file is loaded with plain `yaml.safe_load`; use literal values or inline the policy as a mapping in the Hydra config |
| Agent can't reach the model | `NEMO_GYM_SANDBOX_MODEL_BASE_URL` must resolve *from inside* the sandbox, and the policy's allow rule must match that host and port |
| Agent looks incoherent turn to turn | Model served without `--reasoning-parser nemotron_v3` |
| `sandbox_provider requires a container image` | `ANYSWE_CONTAINER_FORMATTER` points at a `.sif`; OpenShell takes registry images |

**Fallback:** if the gateway is down at showtime, run `up`/`eval` with
`--config nemo_gym/sandbox/providers/docker/configs/docker.yaml` instead. The demo's
point — that the provider is one swappable line — survives; you lose the egress probe,
which is the beat worth protecting. Rehearse `probe` (see T-30 prep).
