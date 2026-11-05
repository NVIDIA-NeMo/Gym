# VIALS: Visual Interpretation of Artifacts in the Life Sciences

The VIALS benchmark evaluates how accurately frontier models interpret visual artifacts
from professional life sciences workflows. This configuration implements direct
evaluation: questions and images go to a vision-capable policy model without tools.

- Integration profile: `custom-gym-verifier`
- Agent: Gym's existing `simple_agent`
- Scorer: `resources_servers/vials`


Tool-assisted evaluation can be added as a separate configuration later.

## Sources

- Code: https://github.com/Handshake-AI-Research/VIALS
- Dataset: https://huggingface.co/datasets/handshake-ai-research/VIALS
- Direct pipeline: vials/pipeline.py

Pinned versions:
- VIALS code commit: 002a0d25031f02a760da5d14f2c29bbd26eb06e4
- Hugging Face dataset commit: 9657bf84c6f6e470e1c31123afe3f9152c3655bd
- Source configuration: `tasks`; source split: `test`; no task filtering.

## Task contract

The full benchmark contains 161 tasks. Each task provides a question
and one or more images.

The agent receives:
- The question.
- All task images embedded in the model input, preserving multi-panel order.
- The direct-evaluation system prompt and answer instructions from upstream VIALS.

The model returns text with a final line:
ANSWER: <final answer>

Ground-truth answers and numeric bounds remain outside the model input. There
are no solver tools, filesystem access, or task sandboxes in this configuration.
The policy endpoint must support image inputs.

## Scoring

Use the upstream VIALS answer extraction and LLM judge.
Each attempt receives a binary reward: 1 for correct, 0 otherwise.

Judge model: `gpt-5-mini`, configured separately through Gym's model-server adapter.
Image-assisted regrading is enabled: unresolved label mappings from the text-only
judge are regraded with all original panels. Numeric bounds are passed unchanged
to the upstream judge. Judge usage combines both passes.
Aggregate metric: mean correctness across task attempts.

Missing/incorrect answers count as zero. Judge/API failures are reported and
masked from aggregation, unlike the upstream pipeline's zero-score treatment of
exhausted judge failures. Report masked counts alongside accuracy.

## Evaluation configuration

- Agent: `simple_agent`, one model turn, no tools.
- Policy model: selected through the Gym model-server configuration.
- Upstream reproduction settings: temperature `1.0`, output budget `32768`,
  three attempts per task, provider-default reasoning effort.
- The checked-in dataset defaults to one attempt for initial evaluation. Use the
  explicit evaluation flags below for upstream sampling settings.
- Judge concurrency defaults to `4`. Transport retries use Gym's model-server
  client; received but malformed judge JSON is retried up to three times and then
  scores zero with a diagnostic flag.
- Record the policy model/version, endpoint, runtime limits, and settings with results.

## Preparation and execution

Run from the Gym repository root in its development environment:

```bash
gym eval prepare --benchmark vials
gym env validate vials
gym env test vials
gym env test --resources-server vials
```

Preparation writes `data/vials.jsonl`, a five-task `data/example.jsonl`, and
original images under `data/images/`. Inputs are self-contained base64 images;
`prompt.yaml` is applied during preparation, not again at rollout time. Rerun
preparation after changing prompts.

Merge the fields from `env.example.yaml` into the repository-root `env.yaml`,
providing your vision policy endpoint and a separate judge key. Credentials stay
in local Gym configuration. The default judge uses OpenAI's `gpt-5-mini`; its
endpoint and model are independently configurable. A one-task smoke evaluation is:

```bash
gym eval run --benchmark vials --model-type openai_model \
    --split example --limit 1 --num-repeats 1 \
    --temperature 1.0 --max-output-tokens 32768 \
    --output results/vials_smoke.jsonl
```

For a full evaluation, select `--split benchmark`, remove `--limit 1`, and use
`--num-repeats 3`.
Inspect policy responses, judge diagnostics, and masked samples before interpreting
accuracy. Offline verifier fixtures do not establish agreement with live judges.

## Licenses and intended use

Upstream code: MIT (see `UPSTREAM_LICENSE` and
`resources_servers/vials/UPSTREAM_LICENSE`). Dataset: CC-BY-4.0.

Evaluation only: upstream states that benchmark data must not appear in training
corpora. Attribute individual images using the upstream dataset's attribution records.

Of the 161 images, 152 are already CC BY 4.0, and the remaining 9 are CC BY 4.0-compatible (CC BY 2.0, CC BY 3.0, CC0, CC BY 4.0 (US), or dual-licensed). Internal lab images originally produced by Handshake AI are released under CC BY 4.0 as part of this dataset. When reusing individual images, please cite the source listed for that task in [attributions.csv](./attributions.csv).
