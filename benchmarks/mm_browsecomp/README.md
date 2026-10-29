# MM-BrowseComp

[MM-BrowseComp](https://arxiv.org/abs/2508.13186) ([repo](https://github.com/MMBrowseComp/MM-BrowseComp)) asks
hard multi-hop questions whose answers need image or video content on the web. Upstream ships questions and a
judge but no agent, so the harness here (BrowseComp's search agent with a `view_image` tool) defines the score.

## Data

`prepare.py` reads upstream at commit `cf82a2390aa7a7ed3d96f72969d22548eff3e101`:

- `data/MMBrowseComp_400.jsonl`: the 400 questions of the 2026-01 update, the set this benchmark scores.
- `data/MMBrowseComp.jsonl`: the 224 questions of the paper, all among the 400; their rows carry `in_paper_set`.
- `MMBC_images/`: the 251 prompt images of 243 questions, inlined as data URLs in their original bytes, typed
  by file signature (three upstream files have the wrong extension).

It decrypts the question, answer and checklist (upstream's XOR scheme, the canary as password) and drops the
`source` evidence URLs, which point at the answer. The output holds plaintext answers and is gitignored.
146 questions have a video checklist item (`checklist_property` code 2); the policy sees no video, so
`no_video/reward` reports the rest.

## Run

```bash
config_paths="benchmarks/mm_browsecomp/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
gym eval prepare "+config_paths=[${config_paths}]"
gym eval run "+config_paths=[${config_paths}]" \
    ++mm_browsecomp_benchmark_resources_server.resources_servers.mm_browsecomp.exa_api_key='[<EXA_KEY>]' \
    ++mm_browsecomp_benchmark_resources_server.resources_servers.mm_browsecomp.exclude_domains_file_path=<file> \
    ++judge_model.responses_api_models.vllm_model.api_key=<INFERENCE_API_KEY>
```

The judge is `nvidia/zai-org/glm-5.1` on inference-api.nvidia.com with upstream's prompt (upstream used
gpt-4o-2024-11-20, which that endpoint does not serve as a pinned snapshot). Headline metric: `mean/reward`
(overall accuracy); `strict_accuracy` and `checklist_score` follow the paper's SA and AVG CS. The judging
differences from upstream are listed in `resources_servers/mm_browsecomp/README.md`.

## Tool-free variant

`config_no_tools.yaml` (prepare: `prepare_no_tools.py`) runs the paper's 224 questions the way upstream ran
its tool-free models: one call with the question and its images, no system prompt, no tools, max 16,384
output tokens, temperature 1.0, top_p 1.0. The same checklist judge grades it. It checks the data and the
judge against paper Table 1 (for example GPT-4.1: 7.59 OA); it does not measure the search harness.
