# Agentic GenRM

Trains a Generative Reward Model that judges inside an agent harness. The task and the reward are
[`generative_reward_model`](../generative_reward_model/README.md)'s: a judging prompt with a conversation,
two candidate responses and rubrics goes in; one JSON verdict with per-rubric and overall scores and a
1-6 ranking comes out; the reward is the negative weighted L1 error against human labels (0 is perfect,
`parse_failure_penalty` when no single valid verdict parses).

The difference is where the judge works. It runs as OpenCode (`opencode_sandboxed_agent`) in a
sandbox, so before it commits to a verdict it can:

- run the responses' code, which is already saved as files (Python 3.12, gcc/g++);
- recompute numbers;
- search the web and read pages through `search` / `browse`, the Exa tools of the search RL harness
  ([`browsecomp_advanced_harness`](../browsecomp_advanced_harness)), served from the Gym host over MCP.

## Servers

| Instance | Entrypoint | Role |
|---|---|---|
| `agentic_genrm` | `app.py` | Verifier. `/seed_session` returns both responses' fenced code blocks as per-row sandbox files under `/workspace/genrm/` (the prose stays in the prompt only). `/verify` scores the final assistant message only. |
| `genrm_search` | `search_app.py` | Tool server: the harness's `search` and `browse`, exposed over Gym MCP. Provider keys, exclusions, the contamination filter and the cache stay on the host. |
| `agentic_genrm_opencode_agent` | `opencode_sandboxed_agent` | OpenCode with `network_access: model_and_tools`: the sandbox reaches only the policy model and `genrm_search`. Sub-agents, todo tools and OpenCode's own web tools are denied. |

### Verify

- **Final message only.** An agent trajectory carries several assistant messages, and the strict
  parser fails on more than one JSON dict. The text after the last tool call is scored; a run that
  ended on a tool call has no final message and is a parse failure.
- **Truncation fails.** A response that stopped on the token limit is a parse failure even if a
  verdict made it out, as a truncated single-turn verdict is.
- **`execution_failure_reward_zero` stays off.** A GenRM reward of 0 is a perfect verdict; zeroing a
  failed run would hand it the best reward (and the agent's empty-response check rejects this
  verifier anyway). Infrastructure failures are masked by the trainer instead.

Extra metrics: `final_message_found`, `response_incomplete`, `num_tool_calls`, `num_bash_calls`,
`num_search_calls`, `num_browse_calls`, `num_other_tool_calls`, `used_tools`, `used_web`.

### Search leakage

HelpSteer3 and the arena preference sets are public together with their labels, so a judge that
finds its own row reads the answer. On top of the harness's filters (domain exclusions passed to Exa
and checked on browse; results dropped when their URL is a Hugging Face dataset page or their text names
the benchmark), `genrm_search` adds:

- `extra_exclude_domains` (with `data/search_exclusions.json`: `lmarena.ai`, `lmsys.org`);
- `extra_contaminated_url_substrings` (arena GitHub/Spaces, Kaggle datasets and competitions);
- `extra_contamination_patterns` (`helpsteer`, `arena-human-preference`, ...), checked against the whole
  serialized result.

Drops are logged as `[browsecomp][contamination] ...` lines. For training, point
`exclude_domains_file_path` at the data-registry opt-out list; the extra domains are added to it.

## Running

Needs `OPENSANDBOX_DOMAIN` / `OPENSANDBOX_API_KEY` and `BC_EXA_API_KEYS_JSON` (a JSON list of Exa keys).
The sandbox cannot reach opencode.ai or GitHub, so set `remote_opencode_install_script_path`,
`remote_opencode_binary_path`, `remote_opencode_musl_binary_path` and `local_ripgrep_binary_path` on the
agent, as the SWE OpenCode runs do.

```bash
gym env start \
    --config resources_servers/agentic_genrm/configs/agentic_genrm_opencode.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

Training rows are the GenRM blend's rows with `agent_ref` pointing at `agentic_genrm_opencode_agent`.

## Tests

```bash
pytest resources_servers/agentic_genrm/tests/ -v
```
