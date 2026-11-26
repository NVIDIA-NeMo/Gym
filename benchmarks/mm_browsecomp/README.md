# MM-BrowseComp

Multimodal BrowseComp ([paper](https://arxiv.org/abs/2508.13186),
[data](https://github.com/MMBrowseComp/MM-BrowseComp)): 400 multi-hop web-research questions whose
solution needs image or video content; 243 carry one or two question images. It runs on the BrowseComp
harness and judge unchanged: only the dataset and the excluded domains differ from `benchmarks/browsecomp`.

## Provenance

- **Source**: `data/MMBrowseComp_400.jsonl` of github.com/MMBrowseComp/MM-BrowseComp, plus the 251
  question images the rows name (`MMBC_images/`), both read at commit `cf82a239` (`REPO_REVISION` in
  `prepare.py`). The rows link their images on `main`; `prepare.py` takes only the file name from them.
- **License**: MIT, as the upstream README's badge states (the repository has no LICENSE file).
- **Split and count**: the single 400-question set, no filtering.
- **Encryption**: question, answer and checklist are XOR-encrypted with each row's canary;
  `prepare.py` decrypts them with BrowseComp's `decrypt`. The prepared file holds the decrypted rows:
  it is git-ignored and must not be published.

## Task contract

- **System prompt and tools**: BrowseComp's (`SYSTEM_PROMPT` + `WORKSPACE_SYSTEM_ADDENDUM`, `TOOLS`).
- **User turn**: the question, with the official "provide your problem-solving roadmap" prefix replaced
  by BrowseComp's `Explanation / Exact Answer / Confidence` suffix (`QUERY_SUFFIX`). A question with
  images gets `input_text` plus one `input_image` data URL per image, in the upstream order; the MIME
  type follows the image bytes, since some upstream files carry the wrong extension. Questions without
  images stay plain text. The policy model must accept images.
- **Harness**: `browsecomp_agent` keeps the user turn, image included, in every model call and across
  context resets. Search and browse return text only, so images found on the web are not visible.
- **Excluded domains**: BrowseComp's TDM opt-out list plus `huggingface.co` and `hf.co`
  (`exclusions.json`), which host copies of the dataset.
- **Reference answer**: kept out of the prompt, in `ground_truth`. Each row also keeps `id`,
  `category`, `subtask`, `level`, `num_images` and the decrypted `checklist`, for per-slice scoring and
  the official checklist judge.

## Scoring and run conditions

Reward is 1 when the BrowseComp judge grades the answer `correct: yes` against `ground_truth`, else 0;
the aggregate is `mean/reward` over the 400 questions. Judge, step budget (400), context resets (50k
tokens) and sampling are those of `benchmarks/browsecomp/config.yaml`. The official MM-BrowseComp
checklist judge is not part of the reward; it can be applied offline to the saved rollouts.

## Prepare

```bash
gym eval prepare --benchmark mm_browsecomp
# offline: a directory holding MMBrowseComp_400.jsonl and images/
MM_BROWSECOMP_DIR=/path/to/MM-BrowseComp gym eval prepare --benchmark mm_browsecomp
```

Without `MM_BROWSECOMP_DIR`, the jsonl and the images are downloaded into `data/cache`. The prepared
file embeds the images (about 260 MB).

## Run

Against a vision-capable vLLM endpoint, with the Exa key for search and the inference API key for the
judge. `--benchmark mm_browsecomp` loads this config, which already includes `benchmarks/browsecomp`'s,
so do not also pass the BrowseComp config.

```bash
gym eval run --benchmark mm_browsecomp --model-type vllm_model \
    --model-url "$POLICY_BASE_URL" --model-api-key "$POLICY_API_KEY" --model "$POLICY_MODEL_NAME" \
    --split benchmark --output results/mm_browsecomp.jsonl \
    +exa_api_key="$EXA_API_KEY" +inference_api_key="$INFERENCE_API_KEY"
```
