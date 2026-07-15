# Pain point 1: Verifier hyperparameters live in YAML, but iteration feels like "new server instance per change" (issue #1415)

## 1. TL;DR

When you iterate on verifier design — adjusting weights in a partial-credit reward system, tuning
thresholds, or tweaking reward hyperparameters — every change today requires standing up what is
effectively a new resources server instance: kill the foreground `gym env start` process, edit YAML
(or add a `++` CLI override), and restart the whole stack (Ray, head server, per-server venv
activation, every server subprocess). That is because each server's config is snapshotted into an
environment variable at `Popen` time and validated into a frozen Pydantic object exactly once at
startup; there is no hot-reload, no config-mutation endpoint, and no per-request override field in
the verify schema. Worse, because verification runs *inside* the agent's `/run` request, re-scoring
also means re-paying all model inference — there is no re-verify command that replays saved
rollouts through `/verify`. The one-line takeaway: **verifier hyperparameters are launch-time
constants, so tuning them costs a full stack restart plus a full re-rollout per change — a
hyperparameter sweep priced like an infrastructure deployment.**

## 2. Experience it yourself

We use `resources_servers/example_tool_call_multireward` (a partial-credit environment with three
{0,1} reward components summed into `reward`) and the hosted policy model config
[eccn-llama-3.1-8b.yaml](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml). That
config points at an OpenAI-compatible endpoint and reads the `INFERENCE_KEY` env var
([eccn-llama-3.1-8b.yaml:6](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml#L6)).
The model is an ECCN-classifier fine-tune, so its answer quality is irrelevant here — the
*workflow* is the point, and any model config can be substituted.

One wiring note: agent configs reference a model instance named `policy_model`
([example_tool_call_multireward.yaml:16-18](../resources_servers/example_tool_call_multireward/configs/example_tool_call_multireward.yaml#L16)),
but the eccn config's top-level key is `eccn-llama-3.1-8b-instruct`
([eccn-llama-3.1-8b.yaml:1](../responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml#L1)),
so we override the agent's `model_server.name` (verified: without the override, config parsing
fails with `ServerRefNotFoundError`, raised at
[global_config.py:342-345](../nemo_gym/global_config.py#L342)).

### (a) Baseline run

Terminal 1 — start the servers (stays in the foreground):

```bash
cd /path/to/Gym
export INFERENCE_KEY=<your key>

uv run gym env start \
  --config resources_servers/example_tool_call_multireward/configs/example_tool_call_multireward.yaml \
  --config responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml \
  "++example_tool_call_multireward_simple_agent.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct"
```

First start is slow: it creates a `.venv` per server and pip-installs requirements
([setup_command.py:112-170](../nemo_gym/cli/setup_command.py#L112)). Subsequent starts can skip
that with `+skip_venv_if_present=true` (opt-in, defaults to false —
[global_config.py:695-696](../nemo_gym/global_config.py#L695)).

Terminal 2 — collect a small rollout set (5 tasks x 2 repeats = 10 rollouts):

```bash
uv run gym eval run --no-serve \
  -a example_tool_call_multireward_simple_agent \
  --input resources_servers/example_tool_call_multireward/data/example.jsonl \
  --output results/multireward_baseline.jsonl \
  --num-repeats 2 --concurrency 5
```

(`--no-serve` collects against the already-running servers —
[main.py:484-490](../nemo_gym/cli/main.py#L484); `-a/--input/--output/--num-repeats/--concurrency`
are all real flags — [main.py:492-498](../nemo_gym/cli/main.py#L492).)

Expected artifacts, written next to each other:

- `results/multireward_baseline.jsonl` — one row per rollout. Each row carries
  `responses_create_params`, the full `response`, the scalar `reward`, and — because this
  environment surfaces them — `reward_components`, `correctness`, `schema_valid`, `format`,
  `predicted_calls` (fields declared at
  [app.py:61-71](../resources_servers/example_tool_call_multireward/app.py#L61)), plus
  `_ng_task_index` / `_ng_rollout_index` / `agent_ref` stamps
  ([rollout_collection.py:513-517](../nemo_gym/rollout_collection.py#L513)).
- `results/multireward_baseline_materialized_inputs.jsonl` — the exact expanded inputs
  ([rollout_collection.py:492-494](../nemo_gym/rollout_collection.py#L492)).
- `results/multireward_baseline_aggregate_metrics.json` — per-component means etc.
  ([rollout_collection.py:577](../nemo_gym/rollout_collection.py#L577)).

Look at one row:

```bash
python -c "
import json
row = json.loads(open('results/multireward_baseline.jsonl').readline())
print(row['reward'], row['reward_components'])
"
# e.g. 3.0 {'correctness': 1.0, 'schema_valid': 1.0, 'format': 1.0}
```

The ECCN fine-tune may well score 0 on some components — that's fine; you now have a baseline
reward distribution to iterate against.

### (b) Now try to change one reward weight

Here is the first surprise: **this environment has zero tunable hyperparameters.** Its config class
is empty:

```python
# resources_servers/example_tool_call_multireward/app.py:51-52
class ToolCallMultiRewardResourcesServerConfig(BaseResourcesServerConfig):
    pass
```

([app.py:51-52](../resources_servers/example_tool_call_multireward/app.py#L51)). The weights are
hard-coded literals inside `verify()` — the components at
[app.py:126-137](../resources_servers/example_tool_call_multireward/app.py#L126) and the unweighted
sum at [app.py:147](../resources_servers/example_tool_call_multireward/app.py#L147):

```python
return ToolCallMultiRewardVerifyResponse(
    **body.model_dump(),
    reward=sum(reward_components.values()),
    ...
)
```

So "change the format weight to 0.25" means **editing code**. And after editing code, the server
process is still running the old code — nothing watches files. Your only options:

**Path 1 — code edit + restart.** Edit `verify()`, then go to Terminal 1 and press Ctrl-C. There is
no `gym env stop` and no reload command — the only shutdown path is `KeyboardInterrupt` on the
foreground process ([env.py:399-404](../nemo_gym/cli/env.py#L399)), which SIGINTs every child
([env.py:344-347](../nemo_gym/cli/env.py#L344)). Then re-run the full `gym env start` command, wait
for Ray + head server + health polling, and re-run `gym eval run` — which re-pays *all* model
inference, because the reward is computed inside the same `/run` request that generated the rollout
(see step 8 of the call chain below).

**Path 2 — promote the weight to config (one-time code change), then tune via YAML or `++`.**
Suppose you add `format_weight: float = 1.0` to the config class and use it in `verify()`. Now you
can set it two ways:

```yaml
# resources_servers/example_tool_call_multireward/configs/example_tool_call_multireward.yaml
example_tool_call_multireward:
  resources_servers:
    example_tool_call_multireward:
      entrypoint: app.py
      format_weight: 0.25        # <- new
```

or, without touching the file, as a Hydra override appended to `gym env start` (CLI overrides merge
last and beat every config file — [global_config.py:586-587](../nemo_gym/global_config.py#L586)):

```bash
uv run gym env start \
  --config resources_servers/example_tool_call_multireward/configs/example_tool_call_multireward.yaml \
  --config responses_api_models/vllm_model/configs/eccn-llama-3.1-8b.yaml \
  "++example_tool_call_multireward_simple_agent.responses_api_agents.simple_agent.model_server.name=eccn-llama-3.1-8b-instruct" \
  "++example_tool_call_multireward.resources_servers.example_tool_call_multireward.format_weight=0.25"
```

You can preview exactly how that override lands with `gym env resolve` (same `--config` flags plus
the `++` tokens; it prints the merged YAML and exits —
[env.py:865-885](../nemo_gym/cli/env.py#L865)). I verified the override above merges cleanly into
the resources server block.

**But — and this is the pain point — both paths require the same restart.** Editing the YAML on
disk while the server runs does nothing: the child process's entire view of the config is the
`NEMO_GYM_CONFIG_DICT` env var serialized at `Popen` time
([env.py:198-201](../nemo_gym/cli/env.py#L198)), and the `++` override only exists in the CLI
process's argv at launch. Either way: Ctrl-C, restart, re-collect.

### (c) The cost of one iteration

Per hyperparameter value you want to try:

| Step | What happens | Cost |
|---|---|---|
| Ctrl-C `gym env start` | SIGINT to all children, join head server ([env.py:344-386](../nemo_gym/cli/env.py#L344)) | seconds |
| Restart `gym env start` | New Ray cluster ([env.py:158](../nemo_gym/cli/env.py#L158)), head server, per-server bash + venv activation + `python app.py` ([env.py:198-203](../nemo_gym/cli/env.py#L198)), health polling every 3s ([env.py:241-252](../nemo_gym/cli/env.py#L241)) | tens of seconds (minutes on first run while venvs build) |
| Re-run `gym eval run --no-serve` | Every task re-runs the *full* agent loop: seed_session → model inference → verify | the entire model-inference bill, again |

And the offline tool does not help: `gym eval profile --inputs ... --rollouts ...`
([main.py:527-538](../nemo_gym/cli/main.py#L527)) only re-aggregates the numeric fields already
frozen into the rollouts file ([reward_profile.py:229-233](../nemo_gym/reward_profile.py#L229),
driven from [eval.py:421-457](../nemo_gym/cli/eval.py#L421)) — it never calls `/verify`, so your
new weight is invisible to it. `gym eval run --resume` (`+resume_from_cache`,
[main.py:491](../nemo_gym/cli/main.py#L491)) only *skips* already-collected `(task, rollout)` keys
([rollout_collection.py:467-473](../nemo_gym/rollout_collection.py#L467)) — it reuses the old
rewards rather than recomputing them. Ten candidate weights = ten restarts = ten full re-rollouts.

## 3. Architecture & code flow

How a YAML value becomes `self.config.<field>` inside `verify()`, end to end:

1. **CLI flags become Hydra tokens.** `--config PATH` is translated to `+config_paths=[...]`
   ([main.py:104-112](../nemo_gym/cli/main.py#L104)); the `env start` command's full flag set is
   registered at [main.py:453-467](../nemo_gym/cli/main.py#L453). Any bare `+key=value` /
   `++key=value` token you type passes straight through. `dispatch()` then rewrites
   `sys.argv = [argv0, *overrides]` and calls the target function
   ([main.py:73-78](../nemo_gym/cli/main.py#L73)) — everything downstream is Hydra, not argparse.

2. **Hydra parses argv into a DictConfig.** There is no primary Hydra config file; a monkeypatched
   `@hydra.main(config_path=None, ...)` wrapper captures the config built purely from CLI tokens
   ([global_config.py:219-223](../nemo_gym/global_config.py#L219)) — which is why new keys need
   `+`/`++`.

3. **YAML files merge under the CLI dict.** `parse()` loads each `config_paths` entry (transitively
   — a config can list more `config_paths`,
   [global_config.py:265-267](../nemo_gym/global_config.py#L265)), then merges with the CLI dict
   last so command-line overrides beat everything
   ([global_config.py:585-587](../nemo_gym/global_config.py#L585)):

   ```python
   # Merge config dicts
   # global_config_dict is the last config arg here since we want command line args to override everything else.
   global_config_dict = OmegaConf.merge(*extra_configs, global_config_dict)
   ```

   Cross-references between servers are validated and hosts/ports defaulted in
   `validate_and_populate_defaults`
   ([global_config.py:325-357](../nemo_gym/global_config.py#L325)). The result is cached in the
   module global `_GLOBAL_CONFIG_DICT` ([global_config.py:57](../nemo_gym/global_config.py#L57),
   returned early on every later call at
   [global_config.py:789-791](../nemo_gym/global_config.py#L789)) — "resolved once and only once".

4. **`RunHelper.start()` snapshots the config into each child's environment.** The *entire* merged
   config is serialized to YAML once ([env.py:161](../nemo_gym/cli/env.py#L161)) and baked into the
   bash command for every server subprocess
   ([env.py:198-201](../nemo_gym/cli/env.py#L198)):

   ```python
   command = f"""{setup_env_command(dir_path, global_config_dict, top_level_path)} \\
   && {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME}={escaped_config_dict_yaml_str} \\
   {NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}={shlex.quote(top_level_path)} \\
   python {str(entrypoint_fpath)}"""
   ```

   launched via `Popen(..., executable="/bin/bash", shell=True)`
   ([setup_command.py:205-212](../nemo_gym/cli/setup_command.py#L205)). **This line is the freeze
   point**: after `Popen`, no YAML edit or override can reach the child.

5. **The child rebuilds the config from the env var.** `app.py`'s `__main__` calls
   `run_webserver()`
   ([app.py:158-159](../resources_servers/example_tool_call_multireward/app.py#L158)), which calls
   `get_global_config_dict()`; in a child, that reads `NEMO_GYM_CONFIG_DICT` directly, "with no
   additional validation", and caches it
   ([global_config.py:793-800](../nemo_gym/global_config.py#L793)).

6. **Three-level indexing + Pydantic validation produce the frozen server config.**
   `load_config_from_global_config` reads `NEMO_GYM_CONFIG_PATH` (the top-level key), indexes
   `<instance> -> <server_type> -> <impl>` via `get_first_server_config_dict`
   ([global_config.py:829-835](../nemo_gym/global_config.py#L829)), and validates against the
   annotation of the server class's `config` field
   ([server_utils.py:368-377](../nemo_gym/server_utils.py#L368)):

   ```python
   server_config_cls: Type[BaseRunServerInstanceConfig] = cls.model_fields["config"].annotation
   server_config = server_config_cls.model_validate(
       OmegaConf.to_container(server_config_dict, resolve=True) | {"name": config_path_str}
   )
   ```

   For our demo server that annotation is `ToolCallMultiRewardResourcesServerConfig`
   ([app.py:74-75](../resources_servers/example_tool_call_multireward/app.py#L74)), which adds
   nothing to the base fields `host`/`port`/`num_workers`/`entrypoint`/`domain`/`name`
   ([config_types.py:563-575](../nemo_gym/config_types.py#L563)). Undeclared YAML keys (like the
   `verified`/`description` metadata, or a `format_weight` you haven't declared) are **silently
   dropped** — Pydantic's default is to ignore extras.

7. **The server object is constructed once, then uvicorn runs without reload.**
   `server = cls(config=server_config, server_client=server_client)`
   ([server_utils.py:624-629](../nemo_gym/server_utils.py#L624)), then
   `uvicorn.run(**uvicorn_kwargs)` with no `reload=` and no file watcher
   ([server_utils.py:669-698](../nemo_gym/server_utils.py#L669)). `self.config` is immutable in
   practice from here on.

8. **`verify()` is called per rollout, inside the agent's `/run`.** The route is registered at
   [base_resources_server.py:138](../nemo_gym/base_resources_server.py#L138)
   (`app.post("/verify")(self.verify)`; abstract method at
   [base_resources_server.py:146-148](../nemo_gym/base_resources_server.py#L146)). The rollout
   collector POSTs each task row to the agent's `/run`
   ([rollout_collection.py:689](../nemo_gym/rollout_collection.py#L689)); the simple agent runs the
   model loop, then builds the verify request from the original row plus the final response and
   POSTs `/verify` ([simple_agent/app.py:197-206](../responses_api_agents/simple_agent/app.py#L197)):

   ```python
   verify_request = SimpleAgentVerifyRequest.model_validate(
       body.model_dump() | {"response": await get_response_json(response)}
   )
   verify_response = await self.server_client.post(
       server_name=self.config.resources_server.name,
       url_path="/verify", ...
   )
   ```

   Inside `verify()`, hyperparameters would be read from `self.config` — the object frozen in step
   6. In this environment they are instead literals at
   [app.py:126](../resources_servers/example_tool_call_multireward/app.py#L126) and
   [app.py:147](../resources_servers/example_tool_call_multireward/app.py#L147).

## 4. Why it hurts / gap analysis

**What exists:**

- A clean launch-time override mechanism: any nested server field can be set from the CLI with
  `++<instance>.<type>.<impl>.<field>=<value>` because the CLI dict merges last
  ([global_config.py:586-587](../nemo_gym/global_config.py#L586)), previewable via `gym env
  resolve` ([env.py:865-885](../nemo_gym/cli/env.py#L865)).
- Per-*task* data flows to the verifier: dataset-row extras (like `expected_call`,
  [app.py:55-58](../resources_servers/example_tool_call_multireward/app.py#L55)) ride through the
  agent because `SimpleAgentRunRequest`/`SimpleAgentVerifyRequest` are `extra="allow"`
  ([simple_agent/app.py:51-56](../responses_api_agents/simple_agent/app.py#L51)).
- Offline *statistics* recomputation: `gym eval profile`
  ([eval.py:421-457](../nemo_gym/cli/eval.py#L421)).

**What's missing (all verified absent):**

1. **No hot-reload or config-mutation endpoint.** `uvicorn.run` is invoked without `reload`
   ([server_utils.py:669-698](../nemo_gym/server_utils.py#L669)); `SimpleResourcesServer` exposes
   exactly `/seed_session`, `/verify`, `/aggregate_metrics`
   ([base_resources_server.py:132-141](../nemo_gym/base_resources_server.py#L132)) — no `/config`,
   `/reload`, or `/refresh` anywhere in `nemo_gym/`.
2. **No per-request hyperparameter override.** `BaseVerifyRequest` is exactly
   `{responses_create_params, response}`
   ([base_resources_server.py:87-88](../nemo_gym/base_resources_server.py#L87)); there is no
   `verifier_config`/`overrides` field, so a request can carry per-task ground truth but not
   per-run knob settings.
3. **No re-verify CLI.** The full command registry
   ([main.py:313-540](../nemo_gym/cli/main.py#L313)) has `eval prepare|run|aggregate|profile` and
   nothing that replays saved rollouts through `/verify`. `--resume` skips completed keys
   ([rollout_collection.py:467-473](../nemo_gym/rollout_collection.py#L467)); without it the output
   is deleted and fully recollected
   ([rollout_collection.py:483](../nemo_gym/rollout_collection.py#L483),
   [:496](../nemo_gym/rollout_collection.py#L496)).
4. **Config is a launch-time snapshot** ([env.py:198-201](../nemo_gym/cli/env.py#L198),
   [global_config.py:793-800](../nemo_gym/global_config.py#L793)), cached in a module global
   ([global_config.py:789-791](../nemo_gym/global_config.py#L789)) and materialized into a frozen
   Pydantic object ([server_utils.py:624-629](../nemo_gym/server_utils.py#L624)). Some servers
   deepen the freeze by deriving state at init — e.g. `FrontierScienceJudgeServer` loads its judge
   prompt from disk once in `model_post_init`
   ([frontierscience_judge/app.py:239-242](../resources_servers/frontierscience_judge/app.py#L239)).

Why this forces the expensive path: hyperparameters can only enter through the config object, the
config object only exists at process construction, and the reward is only computed inside the live
`/run` pipeline. So the *minimum* unit of iteration is (restart the whole stack) + (re-run all
model inference), even when the change is a single float. The demo environment makes it worse by
not exposing the floats at all — but even a fully config-driven verifier (e.g.
`rubric_pass_score_threshold` at
[frontierscience_judge/app.py:185-188](../resources_servers/frontierscience_judge/app.py#L185)) pays
the identical restart + re-rollout price per value.

## 5. Where a fix would hook in

Concrete pointers for implementing a lighter-weight loop, roughly in ascending order of scope:

**(a) Expose the weights as config in the demo env (table stakes, per-env).** Replace the empty
config class at [app.py:51-52](../resources_servers/example_tool_call_multireward/app.py#L51):

```python
class ToolCallMultiRewardResourcesServerConfig(BaseResourcesServerConfig):
    reward_weights: Dict[str, float] = Field(
        default_factory=lambda: {"correctness": 1.0, "schema_valid": 1.0, "format": 1.0}
    )
```

and in `verify()` change [app.py:147](../resources_servers/example_tool_call_multireward/app.py#L147)
to `reward=sum(self.config.reward_weights[k] * v for k, v in reward_components.items())`. This
makes the knob `++`-tunable but does **not** remove the restart — it's a prerequisite for (b)-(d).

**(b) Per-request verifier overrides (smallest core change with the biggest payoff).** Add an
optional field to the verify request schema at
[base_resources_server.py:87-88](../nemo_gym/base_resources_server.py#L87):

```python
class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse
    verifier_overrides: Optional[Dict[str, Any]] = None  # new
```

and have `verify()` implementations consult `body.verifier_overrides` (e.g. via a helper on
`SimpleResourcesServer` that returns `self.config` re-validated with the overrides merged). The
plumbing already exists: the collector POSTs the whole dataset row to `/run`
([rollout_collection.py:689](../nemo_gym/rollout_collection.py#L689)) and the simple agent forwards
`body.model_dump() | {"response": ...}` to `/verify`
([simple_agent/app.py:197-204](../responses_api_agents/simple_agent/app.py#L197)) through
`extra="allow"` models ([simple_agent/app.py:51-56](../responses_api_agents/simple_agent/app.py#L51))
— so a top-level `verifier_overrides` key stamped onto every materialized input row would arrive at
the verifier with zero agent changes. The natural stamping point is
`_preprocess_rows_from_config`, mirroring how `responses_create_params` overrides are merged per
row today (`row[...] = row[...] | responses_create_params_overrides`,
[rollout_collection.py:344-347](../nemo_gym/rollout_collection.py#L344)), fed from a new
`verifier_overrides` field on `RolloutCollectionConfig` (fields start at
[rollout_collection.py:165](../nemo_gym/rollout_collection.py#L165)) and a matching `gym eval run`
flag in the registry at [main.py:474-507](../nemo_gym/cli/main.py#L474).

**(c) A config-mutation endpoint on the resources server.** Register it alongside the existing
routes in `SimpleResourcesServer.setup_webserver`
([base_resources_server.py:132-141](../nemo_gym/base_resources_server.py#L132)):

```python
app.post("/update_config")(self.update_config)

async def update_config(self, body: Dict[str, Any]) -> BaseResourcesServerConfig:
    new_config = type(self.config).model_validate(self.config.model_dump() | body)
    self.config = new_config   # BaseServer is a pydantic model; config is a mutable field
    return self.config
```

Caveats an implementer must handle: (1) servers that derive state in
`model_post_init`/`__init__` (e.g. prompt loading at
[frontierscience_judge/app.py:239-242](../resources_servers/frontierscience_judge/app.py#L239))
need a re-init hook; (2) with `num_workers > 1` uvicorn runs multiple worker processes
([server_utils.py:680-693](../nemo_gym/server_utils.py#L680)) and a POST mutates only the worker
that served it — either restrict to single-worker servers or broadcast.

**(d) An offline re-verify command (pairs with (b) or (c); removes the re-rollout cost).** A new
`gym eval verify` entry in the command registry
([main.py:313-540](../nemo_gym/cli/main.py#L313)) plus a helper in
[eval.py](../nemo_gym/cli/eval.py) modeled on `reward_profile`
([eval.py:421-457](../nemo_gym/cli/eval.py#L421)) that reads a rollouts JSONL and POSTs each row
back to the resources server's `/verify`. This works because `BaseVerifyResponse` extends
`BaseVerifyRequest` ([base_resources_server.py:87-92](../nemo_gym/base_resources_server.py#L87)) and
this server echoes the whole request back (`**body.model_dump()`,
[app.py:145-146](../resources_servers/example_tool_call_multireward/app.py#L145)) — so every stored
rollout row is already a valid verify request superset (confirm per-server: stateful verifiers that
read session state can't be replayed). Combined with (a)+(c), the iteration loop becomes: collect
rollouts once, then per weight candidate: `POST /update_config` + `gym eval verify` — seconds
instead of a restart plus a full inference bill.

The cheapest coherent MVP for issue #1415 is (a) + (d): make weights config, keep the server up via
`gym env start`, and re-score saved rollouts against a restarted-or-mutated verifier without ever
re-running the model.
