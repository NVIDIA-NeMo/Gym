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

## See also

- {ref}`task-verification` — verification patterns and reward design
- {doc}`/resources-server/index` — role of `verify()`
- {doc}`/infrastructure/deployment-topology` — cluster layout and GPUs
- {doc}`/contribute/environments/new-environment` — scaffolding a new resources server