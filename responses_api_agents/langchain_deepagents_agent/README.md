# LangChain DeepAgents Agent

Runs a [LangChain DeepAgents](https://python.langchain.com/docs/experimental/deep_agents/) agent as a
native NeMo Gym agent server.

`app.py`'s `DeepAgentsAgent` is the generic base that should be used by your LangChain DeepAgent. 
This base class does not define a `deepagents` graph by design, since every developer's DeepAgent 
is different. 

Use `reasoning_search_agent.py` as a concrete
subclass example and follow a similar pattern to run
run a different `deepagents` graph. Subclass `DeepAgentsAgent` the same way rather than modifying
`reasoning_search_agent.py`.

## Quick start

### env.yaml

Create `env.yaml` in the repo root if you don't already have one (see [Quickstart](https://docs.nvidia.com/nemo/gym/main/get-started/quickstart)),
and add the following keys. `policy_*` drives the deepagents agent itself — required for both benchmarks below:

```yaml
policy_api_key: sk-or-...
policy_model_name: nvidia/nemotron-3-ultra-550b-a55b:free
tavily_api_key: <YOUR_KEY>
```

**Running the `tavily_search` benchmark only** (skip this for `reasoning_gym`): add these keys to the
same `env.yaml` too. They configure an LLM judge that the *resources server* uses to grade the agent's
final answer against ground truth at verification time — a separate model from `policy_*` above, which
the agent itself never talks to:

```yaml
exclude_domains_file_path: resources_servers/tavily_search/tests/dummy_exclude_domains_file.json
search_judge_model_base_url: https://openrouter.ai/api/v1
search_judge_model_api_key: sk-or-...
search_judge_model_name: anthropic/claude-sonnet-4.5
```

`exclude_domains_file_path` points at a JSON list of domains to exclude from search results; the path
above is a test fixture usable as-is for a quick run — swap in your own list for real evals.

### Launch

This agent always composes with `--model-type`, since it has no provider credentials of its own — it
reaches `model_server`, not a provider directly. Two ready-to-run combo configs bundle a resources server
with this agent:

```bash
# reasoning_gym
gym env start --resources-server reasoning_gym/reasoning_gym_langchain_deepagents_agent_model_server \
    --model-type inference_provider/openrouter

# tavily_search
gym env start --resources-server tavily_search/tavily_search_langchain_deepagents_agent_model_server \
    --model-type inference_provider/openrouter
```

To attach this agent to a different resources server, add the generic
[`configs/langchain_deepagents_agent.yaml`](configs/langchain_deepagents_agent.yaml) alongside your
resources server's config and override its `resources_server.name` (`???` by default):

```bash
gym env start --resources-server <your_server> --model-type inference_provider/openrouter \
    --config responses_api_agents/langchain_deepagents_agent/configs/langchain_deepagents_agent.yaml \
    +langchain_deepagents_agent.responses_api_agents.langchain_deepagents_agent.resources_server.name=<your_server>_resources_server
```

### Run the agent

```bash
gym eval run --no-serve --agent reasoning_gym_langchain_deepagents_agent_model_server \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output /tmp/langchain_deepagents_agent_rg.jsonl
```

Expect `mean/reward: 1.0` over those 5 rows — `reasoning_gym` scores deterministically by extracting the
`<answer>` tag, so there's no judge noise in the number.

## Tool ownership: agent vs. resources server

In this implementation, tools belong to the agent in the agent server, not the resources server. A subclass
of `DeepAgentsAgent` (e.g. `reasoning_search_agent.py`'s `ReasoningSearchDeepAgent`) declares its tools by
passing them into `create_deep_agent(tools=[...])` inside `build_agent()` — `reasoning_search_agent.py`
wires `TavilySearch` this way, calling the Tavily API directly with `tavily_api_key`. The `tavily_search`
resources server it can pair with does define its own `web_search` endpoint, but this agent never calls it. Only its `verify()` endpoint is used for verification. In short, the agent works with any resource-server for
task data and verification functionality, but ignores other functionality of the resource server like tools. 

This is the default shape because tool definitions (schemas, call handling, framework-specific bindings
like `langchain_tavily.TavilySearch`) are inherently agent-harness concerns. They travel with the
framework you're wrapping. Keeping them in the agent server also lets the same
resources server pair with agent servers that expose the same capability differently (e.g. a different
search tool implementation, or a framework with its own native tool-calling conventions).

## LangChain ↔ Gym Responses conversion contract

Gym uses the OpenAI Responses API to describe what the agent server exchanges with the resources server and the model server. 
For this implementation there are 2 complexities:
1. `deepagents`/LangGraph knows nothing about that format: it runs on its own `BaseMessage` types (`HumanMessage`, `AIMessage`, `ToolMessage`, ...) held
as graph state
2. `deepagents` agent server is implemented with the DeepAgent "owning" the agent loop. In short, it drives its own internal tool/summarization/subagent etc loop. One `/v1/responses` call into this agent server
can trigger many internal LLM calls before the graph is done. But at the same time, all these calls need to be logged in the Gym rollout! 

Wrapping an unmodified `deepagents` graph
inside a Gym agent server means converting between these two message shapes every time one side talks to
the other. And since so much occurs in the internal agentic loop of the DeepAgent, it's not acceptable to merely track information once per request. Every time there is an internal model turn it must be tracked, too. The functions below describe this conversion layer.

One rollout crosses these representations:

```text
NeMoGym Responses input
  |
  |  to_langchain()             once per /v1/responses call
  v
LangChain message state
  |
  |  to_gym_input()             every internal model turn
  v
NeMoGym Responses request to the Model Server
  |
  |  to_langchain_ai_message()  every internal model turn
  v
LangChain AIMessage / ToolMessage state
  |
  |  to_responses()             once, after the graph completes
  v
Final NeMoGymResponse
```

Why each representation exists:

- `to_langchain()`: deepagents' graph only operates on LangChain's own `BaseMessage` types. So the incoming conversation is converted once, up front. Now the query can be sent to Langchain `.ainvoke` function.
- `to_gym_input()`: `GymResponsesChatModel` (deepagents' plugged-in "model") talks to the Model Server
  exclusively over the Responses API, and deepagents calls its model repeatedly inside one `/v1/responses`
  call as it runs its own internal tool loop — so the *current* message list is re-serialized on every one
  of those internal calls, not just the outer one.
- `to_langchain_ai_message()`: the graph can only keep looping on LangChain messages. It has no way to
  interpret a raw Responses response, so each Model Server reply is converted back before deepagents can
  decide its next tool call.
- `to_responses()`: the outer Gym protocol expects this agent server to hand back one Responses-shaped
  result, not a LangChain state object, so the whole accumulated message history is flattened back into a
  single `NeMoGymResponse` at the end.

### Data stored in representation shifts

For each seam, "preserved" means the data survives the hop unchanged; "lost" means it's dropped or replaced
with a synthesized value.

This occurs because there is not a perfect 1:1 mapping between Langchain message format and Responses API.

| seam | preserved | lost |
| --- | --- | --- |
| `to_langchain()` | `message`, `function_call`, `function_call_output`, `call_id` | unrecognized item types (silently); per-item `id`; `developer` vs `system` distinction |
| `to_gym_input()` | text content, tool calls, reasoning items (round-tripped) | `AIMessage.id`, `response_metadata`, non-text content parts |
| `to_langchain_ai_message()` | text, tool calls, reasoning items (full item dict), response `id` | `usage_metadata` |
| `to_responses()` | full `function_call`/`function_call_output` trace, reasoning | per-turn response ids; response-level fields are synthesized |

### Known lossy points - will need to fixed/hardened in follow-up PRs
1. to_langchain() drops unknown item types silently, while to_langchain_ai_message() raises on them. Asymmetric and the input side should fail loudly too.
2. usage_metadata is not populated, so LangChain-side token accounting sees nothing. Forces deepagents onto its approximate counter, which is not the best solution. Note that rollouts do get total token usage, but
`usage_metadata` not being populated/lost can be a bug for deepagents and/or Langsmith. 
3. developer and system both map to SystemMessage and both come back as system. Responses API differentiates between `developer` from `system` but LangChain doesn't have this differentiation, so it 
gets lost. 
4. Response-level fields on the final NeMoGymResponse (id, tools, tool_choice) are synthesized, not
carried from the model server responses because they're not relevant since the schema assumes one request 
to response from the agent server. But each internal loop in the agent server generates these so there
is nothing to report at the end. This is a bug that should be handled more elegantly. 


## Context summarization and offloaded history
DeepAgents have summarization middleware that is automatically built in with `create_deep_agent()` function.
Evicted history goes to `/conversation_history/{thread_id}.md` and `thread_id` is the same as `rollout_id`.
Without explicitly providing a `thread_id`, deepagents creates a fresh session per event so the 
conversation is considered a fresh thread each time. Worth noting the agent can and does read_file that path mid-rollout to recover context.

### Max input tokens
Max input tokens is set by using the DeepAgents model profile and is left to be implemented by 
the DeepAgent you subclass. 

### Token accounting is approximate
The missing `usage_metadata` bug makes it so LangChain deepagents summarizationmiddleware uses 
`count_tokens_approximately()` which only looks at `messages`. This can drastically undercount token
usage, making summarization trigger far too late according to what `max_input_tokens` is set to. 

### Logical vs. effective history
The final `NeMoGymResponse` shows the full conversation becasue it has the message hsitory in the
graph state. However, when summarization is called the model server likely did not have the same
request - it gets the compacted version. 

## Open gaps/limitations

- **No trajectory/observability capture (yet).** This agent does not build a `TrajectoryRecord`
  (per-tool-call/model-call observability) — `responses()` always takes the plain `graph.ainvoke()` path.
  Reward/pass-fail scoring is unaffected either way (`verify` only reads `body.response.output_text`);
  what's missing is only the rich per-call detail Gym's own rollout-collection pipeline can attach to a
  trajectory.

- **Context-overflow errors don't reach deepagents' own retry path.** When the Model Server rejects a
  request for exceeding the model's real context window, `raise_for_status()` raises an aiohttp
  `ClientResponseError` — not LangChain's `ContextOverflowError` — so `deepagents`' built-in overflow-retry
  fallback (which catches that specific exception and retries with more aggressive compaction) never fires;
  a raw HTTP error propagates instead. In practice this is now mostly avoided in the first place:
  `max_input_tokens` is a required config field, and `deepagents`' `SummarizationMiddleware` triggers
  compaction at 85% of that threshold, so a well-configured rollout should compact well before hitting the
  provider's real limit — `max_input_tokens` wasn't defined for this agent at all when this gap was first
  reported, so that alone accounts for most of the risk. That's considered safe enough to ship as-is; the
  exception-type mismatch itself is still unfixed, so if compaction ever lags (e.g. `max_input_tokens` set
  too high relative to the real limit, compounded by the approximate token counting above), the failure
  mode is still an opaque `ClientResponseError` rather than a graceful retry. 

- **Known lossy points in the LangChain ↔ Responses translation layer** — see the table and list above:
  unrecognized input item types are dropped silently on one side but raise on the other, `usage_metadata`
  isn't populated, `developer` and `system` roles collapse into one, and several fields on the final
  `NeMoGymResponse` are synthesized rather than carried from any real model call. These are tracked as
  follow-up hardening, not blockers.
