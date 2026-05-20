---
name: integrate-benchmark
description: >
  Guide for adding a new benchmark or training environment to NeMo-Gym. This should
  only be used when a benchmark or training environment ALREADY exists but is not in 
  NeMo Gym yet. You can also use this when wrapping an existing 3rd-party benchmark
  library. 
  If the benchmark/training environment doesn't already exist, for example a brand
  new benchmark or environment that they are defining for the first time, use the
  `create-benchmark` skill instead. 
  Triggered by: "integrate benchmark", "wrap benchmark", 
  "port benchmark", "add existing benchmark", "integrate X into Gym", "wrap X library", 
  "add X benchmark to Gym"
---

# Add Benchmark to NeMo-Gym


## Ask a few simple questions for documentation and code structure:

- Environment name
- Source repo/location of benchmark or environment
- Paper/reference (if applicable)
- License (I don't know is fine if not known)
- Brief description: What does this environment evaluate? (e.g. web navigation, code generation, tool use)

If any of the questions the user is not sure about then you can skip over it. 
Try to figure out any information you're sure of from looking at the benchmark/environment they supply, then
you can fill in information yourself. 

## Information Gathering and Implementation Planning

To help figure out what kind of environment/benchmark this is it can be helpful to ask the user questions 
to learn how the agent interacts with the environment and the other dependencies for the environment. 

You can refer to "Background Information about Benchmarks" in this file for additional context.

Use the information already supplied by the user like paper, reference, source repo, etc, to answer the
below as much as possible. After you have filled in all the information you can ask the user too and use
these two sources of information to find discrepancies to clarify the environment/benchmark. 

Ask the user to define how the agent interacts with the environment - here are
some common things to think about and challenge the user on. 
- Does the agent receive a natural language prompt and return an answer?
- Does the model use tools (function calling, code execution, web browsing)?
- Is it single-turn or multi-turn (does the model get feedback and retry)?

Then, ask the user about how verification works. 
What's the reward signal? Is it binary pass/fail, a score, or multiple
metrics? How is correctness determined? (exact match, test cases, judge model, human eval)?

Ask about external dependencies.
Does this environment require external tools, specific runtimes, or sandboxes (e.g. compilers, browsers, Docker, VMs)?
If so, list them and note whether they can be auto-installed on server startup. 

Ask about data.
- Dataset source (e.g. HuggingFace, custom):
- Approximate size (number of tasks):
- Splits available (train/validation/test):
If they didn't already provide paper/reference/source repo then ask for this.
We're looking for published or known results to use as a reference.
Link to leaderboards, papers, or repos with reported numbers.

Lastly, note anything an engineer should know about running this environment:
- Does it need specific hardware (GPUs, large memory)?
- Does it require network access, Docker, or a VM?
- Are there known limitations on parallelism or throughput?
- Any OS or platform restrictions?

## Build

Use the information from information gathering from both the user
and the benchmark/environment source to properly design implementation
according to the guidelines below: 

No matter what kind of external benchmark/environment you are integrating,
you will integrate at the agent server level and not in resources server.
In short, you will wrap the benchmark in the agent server's `/run` endpoint. 

- Integrate at the agent server level (not resources server) 
- Agent's `/run` endpoint wraps the external library 
- Pre-process from Gym schema to library input, post-process back to `BaseVerifyResponse`
- Reproduce publicly reported numbers with the original repo first, then reproduce again after Gym integration
- Add the dependency in `requirements.txt`

## Workflow
TODO: worried this is overfit to tau2. 

### Step 1: Scaffold the agent server
TODO: it'd be great if we had a cli command to scaffold - do we have this? 
Follow the structure of `responses_api_agents/tau2` to start.

  Required structure:

  responses_api_agents/<name>/
  ├── app.py
  ├── configs/<name>_agent.yaml
  ├── data/example.jsonl          # 5 entries, committed to git
  ├── tests/__init__.py
  ├── tests/test_app.py
  ├── requirements.txt
  └── README.md

requirements.txt content:

  -e nemo-gym[dev] @ ../../
  <upstream-library>==<pin>     # or: git+https://... @ <sha>

Per the docs, this is the only place upstream dependencies are declared — do not vendor them into nemo_gym/

### Step 2: Define request/response schemas
Subclass BaseRunRequest with any extra fields your library's task runner needs (task spec, seed, run config, etc.). Subclass BaseVerifyResponse for the agent's reply — include
both reward and any per-task metrics you want logged downstream (duration, step counts, token usage, finish reasons).

Reference: `responses_api_agents/tau2/app.py`

### Step 3: Implement `/run` - wrap the upstream library 
Subclass SimpleResponsesAPIAgent. Per the docs, leave responses() as raise NotImplementedError — external integrations only need /run.

In run():

1. Preprocess — translate BaseRunRequest + responses_create_params into whatever shape your library's entrypoint expects. (See responses_api_agents/tau2/app.py:126-152.)
2. Point the upstream LLM client at Gym model servers. For each model role the library needs, expose a ModelServerRef field in the agent config (model_server for policy, plus
extras like user_model_server for simulators). At runtime, set the library's api_base = f"{get_server_url(self.config.<ref>.name)}/v1" and a dummy API key. Tau2 does this for
both policy and user-sim models at app.py:131-148.
3. Await the library's task entrypoint. Example: result = await run_single_task(**body_dict) (app.py:152).
4. Postprocess the trajectory for RL. Convert the library's message list → OpenAI responses items via VLLMConverter.chat_completions_messages_to_responses_items, then split with
  split_responses_input_output_items. This is what makes the trajectory consumable by Gym's training loop. (app.py:154-169.)
5. Return your *VerifyResponse with reward set from the library's result object plus any metrics you computed.

### Step 4: Auto-install external tools (if applicable)

If the benchmark requires an external tool (compiler, runtime, etc.), auto-install it on server startup so users don't need manual setup. See `references/patterns.md` § "External Tool Auto-Install Pattern".

Key points:
- Create `setup_<tool>.py` with `ensure_<tool>()` — checks PATH, forks on `sys.platform` (brew on macOS, build from source on Linux)
- Call it in `model_post_init()` before semaphore init
- Build scripts should be idempotent and install into a local gitignored prefix
- Add a `pytest_configure` hook in `tests/conftest.py` that calls `ensure_<tool>()` before collection

### Step 5: Write YAML configs

1. Agent config — `responses_api_agents/<name>/configs/<name>_agent.yaml`

Declares the agent server: entrypoint, every ModelServerRef the library needs, library-specific settings (max steps, concurrency knobs, debug flags), and an
example dataset. Reference: `responses_api_agents/tau2/configs/tau2_agent.yaml`.

2. Benchmark config — `benchmarks/<name>/config.yaml`

Chains to the agent config via config_paths and uses _inherit_from to override per-variant knobs (which model serves the agent, which model serves the simulator, num_repeats,
dataset path). This is what isolates one benchmark variant from another so the agent config stays generic. Reference: benchmarks/tau2/config.yaml.

### Step 6: Test

```bash
# Run server tests (creates isolated .venv, slow on first run)
ng_test +entrypoint=resources_servers/my_benchmark

# Run core library tests to check nothing broke
pytest tests/unit_tests/ -x
```

Test coverage must be >= 95%. Write tests for: verify pass, verify fail (wrong output), verify fail (no code extracted), verify fail (compilation error if applicable), verify timeout.

### Step 6: Smoke test end-to-end

```bash
# Start servers
ng_run "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# Quick test with example data
ng_collect_rollouts +agent_name=my_benchmark_simple_agent \
  +input_jsonl_fpath=resources_servers/my_benchmark/data/example.jsonl \
  +output_jsonl_fpath=results/example_rollouts.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Inspect results
```

### Step 7: Baseline (reward profiling)

Run against multiple models to validate correctness. Recommended suite:
- Your policy model of interest
- At least one open-source instruct model (e.g. Qwen 3 30B A3B Instruct)
- At least one open-source thinking model (e.g. Qwen 3 30B A3B Thinking)
- At least one closed-source model (e.g. GPT-5 Nano or GPT-5)

```bash
# Collect rollouts
ng_collect_rollouts +agent_name=my_benchmark_simple_agent \
  +input_jsonl_fpath=resources_servers/my_benchmark/data/my_dataset.jsonl \
  +output_jsonl_fpath=results/rollouts.jsonl \
  +num_repeats=5 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Compute per-task pass rates
ng_reward_profile +input_jsonl_fpath=resources_servers/my_benchmark/data/my_dataset.jsonl \
  +rollouts_jsonl_fpath=results/rollouts.jsonl \
  +output_jsonl_fpath=results/profiled.jsonl \
  +pass_threshold=1.0

# Aggregate metrics (pass@1 = avg_reward, pass@k from max_reward)
python scripts/print_aggregate_results.py +jsonl_fpath=results/profiled.jsonl
```

Increase `num_repeats` until variance < 1% across runs on the same model.

Closed-source models should score at or above open-source models. If not, investigate for bugs. Inspect actual failure cases in the rollout JSONL, not just aggregate numbers.

For external benchmarks: reproduce the original repo's published numbers first. Then reproduce after Gym integration. Scores should match.

### Step 8: Pre-commit and PR

Use `.github/ISSUE_TEMPLATE/environment-integration.md` to make sure and issue is created for the integrated environment. 

```bash
pre-commit run --all-files
```

First run may fail as hooks auto-modify files (`verified: false` flag, README table). Stage changes and run again.

Set `verified: true` in YAML config after successful baselining. Include W&B links and screenshots of results in the PR description.

To avoid committing unrelated auto-fixes from other servers, scope pre-commit to your files:
```bash
pre-commit run --files resources_servers/my_benchmark/**/*
```
If hooks modify files in other directories, discard those changes:
```bash
git checkout -- resources_servers/other_server/
```

## Constraints

- Use NeMo Gym's OpenAI client (`nemo_gym/openai_utils.py`), not LiteLLM/Anthropic/other
- **Use aiohttp, not httpx, for async HTTP.** All async HTTP calls must go through `nemo_gym.server_utils.request()` (aiohttp). httpx has O(n^2) connection pooling that hangs at high concurrency. When wrapping external libraries that use httpx internally, replace their HTTP transport with an aiohttp adapter — see `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) for the pattern and `docs/infrastructure/engineering-notes/aiohttp-vs-httpx.md` for the rationale.
- Pass configuration through Gym config (YAML), not environment variables
- Code must run on Linux
- `/run` endpoint must be async
- Errors from tool execution or bad model output must return error responses, not crash
- All commits require DCO sign-off (`-s`) and cryptographic signature (`-S`)
- Issue for the integrated environment is created from `.github/ISSUE_TEMPLATE/environment-integration.md`

## Reference

For detailed code patterns, schemas, and examples: see [references/patterns.md](references/patterns.md).

## Background Information about Benchmarks 
TODO: we could tell the agent to go read fern/versions/latest/pages/about/architecture.mdx 
to learn about architecture of environments. 
Benchmarks are fundamentally synonymous with environments, so understanding how
environments work will help you understand how benchmarks also work. 

There are generally 3 kinds of external benchmarks/environment structure.
This is based off the information comes "with" the benchmark that's going to be integrated:

1) Benchmarks/environments that define tasks and verifier. These notably don't have an agent harness. 
A good example of this is MMLU. Users will define the model that they want to improve on this benchmark. 
TODO: do these have action and state? I don't think so probably? 

2) Benchmarks/environments that define tasks, the agent/agent harness, and the state, action, and verifier. 
A good example of this kind of environment is tau2. There is a specific tau2 agent harness, which is used
for doing tool calling for this benchmark. Users will define the model that they want to improve on this benchmark. 

3) Benchmarks/environments that define tasks, the verifier, the state, and actions. 
A good example of this kind of environment is SWEBench. Users can define the model 
and/or agent that they want to improve on this benchmark. 
TODO: include state and actions? 
