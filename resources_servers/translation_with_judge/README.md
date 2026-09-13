# translation_with_judge

Machine-translation verifier scored by an LLM judge instead of a fixed
metric. `verify()` sends the source segment, a reference translation, and
the policy model's candidate translation to a judge model (any
OpenAI-Responses-compatible endpoint) and asks it to rate adequacy +
fluency on a 0-100 scale. That score, normalized to `[0, 1]`, is the RL
reward.

Corpus-level sentence-BLEU and chrF (via `sacrebleu`) are also computed per
row as cheap, judge-independent diagnostics — they do **not** affect the
reward, only `compute_metrics()`'s aggregate output.

Includes dedicated tokenizer support for Indic languages: sentence-BLEU
uses sacrebleu's `flores200` SentencePiece tokenizer (instead of the
default whitespace/punctuation tokenizer, which doesn't segment Brahmic
scripts meaningfully) for Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati,
Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu, Nepali, Sanskrit, Dogri,
Konkani, Sindhi, Bodo, Kashmiri, Maithili, and Manipuri — see
`_TOKENIZER_BY_FLORES_SUBTAG` in `app.py`.

## Row schema

Each row in `data/*.jsonl` (see `TranslationWithJudgeRunRequest` in `app.py`):

| Field | Meaning |
| --- | --- |
| `prompt` | `[{"role": "user", "content": "..."}]` — the instruction-wrapped source segment |
| `responses_create_params` | `{"input": prompt}` — required by nemo_gym, sent to the policy model |
| `solution` | Reference translation |
| `src_lang` / `tgt_lang` | FLORES-200 codes (e.g. `eng_Latn`, `kan_Knda`) for `prompt`'s language / `solution`'s language |
| `direction` | `src2tgt` \| `tgt2src` — provenance only, not read by `verify()` |
| `prompt_style` | Which instruction template wrapped the segment — provenance only |
| `agent_ref` | `{"type": "responses_api_agents", "name": "translation_with_judge_simple_agent"}` — required by nemo_gym, routes the row to the agent |

`dataset_type` also rides along for provenance but isn't read by `verify()`.

## Metric outputs

`compute_metrics()` groups rollouts by `(src_lang, tgt_lang)` (FLORES-200
codes, e.g. `eng_Latn`, `kan_Knda`) and emits, per pair and as cross-pair
aggregates (`xx->xx`, `<src>->xx`, `xx-><tgt>`):

- `<src>-><tgt>/judge_score` (+ `_std_dev_across_runs`) — mean LLM-judge score, 0-100
- `<src>-><tgt>/bleu`, `<src>-><tgt>/chrf` (+ `_std_dev_across_runs`) — diagnostic only

`get_key_metrics()` returns the headline aggregates: `xx->xx/judge_score`,
`eng_Latn->xx/judge_score`, plus `xx->xx/bleu` and `xx->xx/chrf` for context.

## Judge model

Point `translation_judge_model` at any server implementing the OpenAI
Responses API. The default config assumes a self-hosted vLLM server:

```bash
docker run -it --rm \
  --name gpt-oss-120b-judge \
  --gpus all \
  --network host \
  --ipc host \
  -v /fsx2:/fsx2 \
  -v /fsx2/opensource-models:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest

# inside the container:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve openai/gpt-oss-120b \
  --port 8005 \
  --async-scheduling \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss
```

Any other OpenAI-Responses-compatible server (vLLM, SGLang, a hosted API,
etc.) works too — just point the three config values below at it.
`configs/translation_with_judge.yaml` reads the judge endpoint from three
overridable top-level values (defaults shown):

```yaml
translation_judge_base_url: http://localhost:8005/v1
translation_judge_api_key: EMPTY
translation_judge_model_name: openai/gpt-oss-120b
```

Override on the CLI if the judge runs elsewhere, e.g.
`++translation_judge_base_url=http://<host>:<port>/v1`.

## Example usage

`--model-type vllm_model` is a *client* to an already-running OpenAI-Responses-compatible
server -- it needs `--model-url`/`--model-api-key`/`--model` set, or `gym env start` fails
with a missing-model-url error. For a smoke test you can point it at the same judge server
(or any other server already serving an instruct model); for a real eval, swap in your
actual policy checkpoint's server. To have `gym` launch a policy vLLM server itself instead,
use `--model-type local_vllm_model --model <HF id or path>`.

```bash
# Running servers
gym env start \
    --resources-server translation_with_judge \
    --model-type vllm_model \
    --model openai/gpt-oss-120b \
    --model-url http://<policy-host>:<port>/v1 \
    --model-api-key EMPTY \
    ++translation_judge_base_url=http://<judge-host>:<port>/v1

# Collecting rollouts (5-example smoke test), in another shell once all 4 servers report ready
gym eval run --no-serve \
    --agent translation_with_judge_simple_agent \
    --input resources_servers/translation_with_judge/data/example.jsonl \
    --output results/translation_with_judge_rollouts.jsonl \
    --num-repeats 1
```

`data/example_rollouts.jsonl` was generated this way against a live
`openai/gpt-oss-120b` judge+policy endpoint (SGLang), scoring 94-100/100 on
the 5 example rows.

## Config

| Key | Default | Meaning |
| --- | --- | --- |
| `judge_model_server` | `translation_judge_model` | `ModelServerRef` to the judge's `responses_api_models` instance |
| `judge_responses_create_params` | `{input: [], max_output_tokens: 1024}` | Extra params merged into each judge call; `input` is overwritten per-call |
| `strip_reasoning` | `true` | Drop a `<think>...</think>` preamble from the policy model's output before scoring |

## Licensing information

- Code: Apache 2.0
- Data: `data/example.jsonl` (and the rows it was drawn from in
  `data/example_rollouts.jsonl`) are 5 sentence pairs taken directly from
  the [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
  `devtest` split (English/Hindi/Bengali/Telugu/Tamil/Kannada), licensed
  **CC-BY-SA 4.0** by Meta AI. This environment currently ships example
  data only — no `train`/`validation` split is included, since a larger
  Indic-language training curriculum this was developed against blends
  several corpora (Samanantar, NLLB, AI4Bharat's BPCC, ILCI, and others)
  whose individual licenses have not yet been verified for redistribution.

Dependencies
- nemo_gym: Apache 2.0
- sacrebleu: Apache 2.0
- sentencepiece: Apache 2.0 (used by sacrebleu's `flores200` tokenizer for Indic-language sentence-BLEU)
