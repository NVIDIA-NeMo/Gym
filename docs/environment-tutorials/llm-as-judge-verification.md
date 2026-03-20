(llm-as-judge-verification)=

# LLM-as-a-judge in verification

Use a **second language model** inside your resources server’s `verify()` when rewards depend on semantic equivalence, rubrics, or other judgments that are expensive or awkward to encode in deterministic code. This tutorial explains where that fits in NeMo Gym, how to configure the judge endpoint, and where to copy working patterns from the tree.

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

< Back to Building Environments
:::

---

## Prerequisites

- {ref}`task-verification` — especially *What is LLM-as-a-judge?*
- {ref}`core-components` — resources server vs model server roles
- {ref}`configuration-concepts` — Hydra composition and server references

---

## When to use an LLM judge (and when not to)

**Prefer deterministic verification** if you can score the rollout with exact string match, multiple choice, matching a known tool trace, running code or SQL and comparing outputs, or a small equivalence library (for example for math). It is usually faster, cheaper, and easier to operate at high throughput than an LLM judge.

**Use an LLM judge** when:

- Success is **rubric- or criteria-based** (instruction following, safety nuance, “does this satisfy the spec?”).
- A formal checker is **impractical** but you can describe what “good” means in a prompt.

**Tradeoffs:** extra latency and cost, non-determinism unless you tune temperature and parsing, and possible **positional bias** (judge favors text in a fixed slot). Some servers mitigate bias with a second pass that **swaps** gold vs prediction (see [`equivalence_llm_judge`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/equivalence_llm_judge)).

---

## Architecture: where the judge runs

During rollout collection, the **agent** calls the **policy model**; when the episode ends, the **resources server** runs `verify()`. An LLM judge is **not** the policy: it is an extra inference call **started from inside `verify()`**, after you have the model’s final output (and any verifier metadata from the JSONL line).

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
  subgraph rollout[Rollout]
    A[Agent server] --> M[Policy model server]
    M --> A
    A --> R[Resources server verify]
  end
  R --> J[Judge model server]
  J --> R
```

**Typical in-repo pattern (Gym-internal):** `verify()` uses `self.server_client.post(..., url_path="/v1/responses", ...)` to call a **named model server** declared in the same Hydra config. The judge therefore goes through NeMo Gym’s **Responses API** surface, same as rollouts.

**Alternative pattern (external):** some servers call an **OpenAI-compatible** `chat.completions` client pointed at URLs you supply (e.g. HPC or a separate cluster). [`proof_verification`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/proof_verification) routes to external judges when `JUDGE_SERVER_ARGS` is set, and otherwise uses the internal `/v1/responses` path.

For how NeMo Gym sits next to GPUs and training frameworks, see {doc}`/infrastructure/deployment-topology`.

---

## Configuration: wiring the judge in YAML

Most LLM-judge servers expose fields along these lines (exact names vary by server; always check that server’s `configs/*.yaml` and `README.md`):

| Idea | Typical config shape |
|------|----------------------|
| Which model server to call | `judge_model_server: { type: responses_api_models, name: <server_key> }` |
| Generation settings for the judge | `judge_responses_create_params` (e.g. `max_output_tokens`, `temperature`, `top_p`; `input` often filled in code) |
| Prompting | Inline `judge_prompt_template` / `judge_system_message`, or paths like `judge_prompt_template_fpath` |
| Load control | Fields such as `judge_endpoint_max_concurrency` where implemented |

**Same server as policy:** set `name:` to the policy model’s key (e.g. `policy_model`). **Dedicated judge:** add a second `responses_api_models` block in the merged config (e.g. `judge_model`) and set `judge_model_server.name: judge_model`. [`multichallenge`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/multichallenge) documents this split in its YAML comments.

**Example fragment** (adapted from [`equivalence_llm_judge`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/equivalence_llm_judge)):

```yaml
equivalence_llm_judge:
  resources_servers:
    equivalence_llm_judge:
      judge_model_server:
        type: responses_api_models
        name: policy_model
      judge_responses_create_params:
        input: []
      judge_prompt_template_fpath: prompt_templates/equivalence_llm_judge.txt
      judge_equal_label: "[[A=B]]"
      judge_not_equal_label: "[[A!=B]]"
      judge_endpoint_max_concurrency: 64
```

Model URLs, API keys, and model IDs for hosted backends belong in your **merged Gym config** (e.g. `env.yaml` and Hydra overrides), consistent with the rest of the project — not ad hoc environment variables, except where a specific server documents them (such as external judge routing).

---

## See also

- {ref}`task-verification` — verification patterns and reward design
- {doc}`/resources-server/index` — role of `verify()`
- {doc}`/infrastructure/deployment-topology` — cluster layout and GPUs
- {doc}`/contribute/environments/new-environment` — scaffolding a new resources server