# DeepSearchQA

[DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa) evaluates
multi-step web research questions with single-answer and set-answer targets.
This benchmark runs a Gym agent harness in a sandbox with Exa MCP search and
uses the [`deepsearchqa`](../../resources_servers/deepsearchqa/README.md)
resources server to verify the final answer. 

## Prepare the data

The preparation script downloads the full DeepSearchQA CSV from Hugging Face
and writes approximately 900 tasks to `data/deepsearchqa_benchmark.jsonl`.

```bash
gym eval prepare --benchmark deepsearchqa
```

Five prepared tasks and their example rollouts are checked in under the
[`deepsearchqa` resources server](../../resources_servers/deepsearchqa/data/).

## Run

The benchmark config defaults to Claude Code using your configured policy model.
Export `EXA_API_KEY`; the environment reads it automatically. Agent and judge
defaults ship with the environment, so no environment-specific `env.yaml` block
is needed. The judge uses the same model as the policy unless overridden.
Use `--limit 1` for a quick end-to-end check before running the full benchmark.

Complete long-run config copies for all supported harnesses are in
[`configs`](configs). Each file includes the DeepSearchQA verifier, agent
settings, sandbox settings, prompt, and dataset. The files do not inherit from
a shared agent config. There are two kinds:

- **Session harnesses** (Codex, Pi, Hermes, OpenCode, OpenClaw, Kilo Code) run
  under `single_agent_turn_legacy`. Each episode, the agent creates its own
  sandbox and gets the `web_search`, `find_in_page` and `scroll_page` tools over
  MCP from the `exa_search` tool server (`tool_servers`): a
  [`tavily_search`](../../resources_servers/tavily_search/README.md) server with
  `search_provider: exa`. Its exclusion policy,
  [`search_exclusions.json`](search_exclusions.json), keeps the dataset's
  Hugging Face and Kaggle pages, and any URL containing `deepsearchqa` (also with
  `-` or `_` between the words), out of
  the results and page reads of these three tools; it
  does not restrict the sandbox's own network access, so a shell or a harness's
  own fetch tool can still reach them. The sandbox calls the model
  server and `exa_search` on the Gym host, so these configs set
  `use_absolute_ip: true`: the sandbox must be able to reach that host. These
  copies use different search tools, result formats and exclusions than their
  earlier `harness_exa_search` versions, so their scores are not directly
  comparable with runs of those versions.
- **Claude Code, Cline, Prime Agent, Simple Strands, and NeMo Fabric
  DeepAgents** still run through the deprecated `harness_exa_search` agent
  behind `legacy_agent`, which starts the harness and its Exa integration in a
  sandbox and relays model calls itself. Claude Code takes no MCP tool grants
  in a session, and the other four have no session path yet. Their `agent`
  field selects a name from Gym's shared `harness_agent` registry, and
  `agent_kwargs` contains that agent's settings.

Export `EXA_API_KEY` for either kind, and pass a sandbox provider config next
to the config copy.

The Claude Code configs install Gym and the Claude CLI in a fresh sandbox
using `setup_command`. No cluster runtime archive is required.

For example, select the Codex copy directly:

```bash
gym eval run \
  --benchmark deepsearchqa/configs/deepsearchqa_codex \
  --model-type <model-type> \
  --model <model> \
  --limit 1
```

The available
copies are Claude Code, Codex, OpenCode, Pi, Hermes, OpenClaw, Kilocode, Cline,
Prime Agent, Simple Strands, and NeMo Fabric DeepAgents.

## Verification

The verifier compares the submitted answer with the reference answer using the
official DeepSearchQA [`judge_prompt.txt`](../../resources_servers/deepsearchqa/judge_prompt.txt).
It supports both `Single Answer` and `Set Answer` examples and rejects missing
required answers or excessive extra answers.
