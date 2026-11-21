# Indic Arena Proxy v3 (chat400)

400 matched conversations per language in English, Bengali, Gujarati, Hindi,
Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Tamil, Telugu, and Urdu.
Each language has 320 single-turn and 80 multi-turn conversations. The task is to
generate **one next assistant reply given the complete conversation history**.

This release contains 5,200 source prompt records and 5,200 frozen GPT-5.4-mini
reference-answer records. It preserves the supplied source files byte for byte.
`checksums.json` records SHA-256 checksums, row counts, and translation exclusions.
The single Tamil `hard_error` translation is retained for provenance but excluded
from evaluation, leaving 5,199 scored tasks across all languages.

## Files and loading

- `lmarena_v3_chat400_<lang>.jsonl`: original conversation, question ID, and source metadata.
- `references/gpt-5.4-mini/<lang>.jsonl`: frozen reference reply and generation metadata.
- `checksums.json`: file integrity and language counts.

The Hugging Face configurations `prompts_<lang>` and `references_<lang>` have
separate schemas and each expose a `test` split. For faithful source loading:

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="anushakamathofficial/indic-arena-proxy-v3",
    repo_type="dataset",
    filename="lmarena_v3_chat400_hi.jsonl",
    revision="<immutable commit SHA>",
)
with open(path, encoding="utf-8") as stream:
    prompts = [json.loads(line) for line in stream]
```

Join references using `question_id`, never row position. The prompt records also
retain upstream `baseline_answer` and `other_answer` fields from LMArena; these
are **not** this benchmark's GPT-5.4-mini reference. Evaluation must use
`references/gpt-5.4-mini/<lang>.jsonl` and its `generation` field.

## Evaluation protocol

The associated NeMo Gym integration is `benchmarks/indic/arena_proxy_v3` in
Indic Gym. It uses the existing `simple_agent`, model server, environment server,
and `arena_judge` resources server. Generation defaults to temperature 1.0 and
top-p 0.95; record any changes such as temperature 0.6 with the run.

The supplied setup uses Gemini 3.8 Flash (`gcp/google/gemini-3.8-flash`) through
the NVIDIA inference endpoint, temperature 0, reasoning effort low, and a 32,768
token judge limit. These are source-protocol identifiers, not a claim about
current endpoint availability. Each task receives two pairwise comparisons
against GPT-5.4-mini with candidate/reference positions swapped. Creative
writing / role-play uses the creative template; other groups use the default
conversation-aware template. Thinking text is removed before judging.

Scores use Arena-Elo aggregation with strong verdicts weighted by 3 and 100
bootstrap rounds. The reported Indic headline is the unweighted mean of
`all.score` over the 12 Indic languages. English is separate. Report all,
single-turn, multi-turn, and the seven topic groups per language. Missing
generations or judge failures must be reported as incomplete.

## Provenance and terms

The English conversations were sampled from the LMArena proxy v3 data used by
NeMo Gym, distributed through the NVIDIA GitLab dataset registry as
`lmarena_v3` version `0.0.1`. The supplied chat400 export adds sampling metadata
and Indic translations produced with `indic-llm-eval-suite`. This release does
not independently resample, translate, or regenerate any answers.

The source NeMo Gym `lmarena_v3/manifest.yaml` lists dataset licensing as
`unknown`. No new license is granted here, and the software's Apache-2.0 license
does not relicense the conversations, translations, or model outputs. Preserve
the applicable upstream terms and attribution.

Translations include quality warnings and may contain errors or mixed scripts.
This is a translated proxy benchmark, not a comprehensive assessment of each
language. The selected conversations and model outputs can contain user-generated
content. Do not use this held-out benchmark as training data for evaluated models.
