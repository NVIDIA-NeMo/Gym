# MM-BrowseComp resources server

Multimodal web research for [MM-BrowseComp](https://arxiv.org/abs/2508.13186): hard multi-hop questions whose
answers need images found on the web. The server extends `browsecomp_advanced_harness` and keeps its tools,
its contamination guard and its excluded domains unchanged.

## Tools

- `search`, `browse`, `bash_command`: as in BrowseComp (Exa by default). With `exa_browse_image_links > 0`,
  `browse` lists each page's images (URL and alt text) under `[Images]`.
- `view_image(url)`: downloads one image and returns it as a JPEG data URL (long edge at most
  `view_image_max_side`, 1,536 px by default). It refuses non-HTTP URLs, excluded domains, benchmark mirrors,
  hosts that resolve to non-public addresses (also after a redirect) and images over `view_image_max_bytes`.

Use it with `browsecomp_agent` and `tool_images: true`: the agent puts the returned images in a user message
after the round's tool outputs, because a chat tool message carries text only. It keeps the newest
`max_tool_images_in_context` tool images and replaces older ones with a text marker.

## Grading

The judge prompt is upstream's `construct_judge_prompt` (src/eval.py at `cf82a239`, MIT), byte for byte: the
question, the reference answer, the checklist and the last 25,000 characters of the final answer. The judge
answers `CHECKLIST_SCORE`, `CHECKLIST_RESULT` and `OVERALL_CORRECTNESS`.

- `reward` is overall accuracy (OA): `OVERALL_CORRECTNESS: YES`.
- `strict_accuracy` (SA) also needs a full `CHECKLIST_SCORE`; `checklist_score` is its mean fraction.
- Both, and `reward`, are also reported for `paper_224/` (the paper's 224 questions) and `no_video/` (questions
  without a video checklist item).

Differences from upstream:

- The last verdict line is parsed, not the first, so an echoed example line cannot decide the grade.
- A reply with no `OVERALL_CORRECTNESS` is sampled again, up to 15 times (the BrowseComp judge's policy);
  upstream counts it as wrong.
- A rollout that times out or fails counts as wrong (0) in every metric; upstream drops failed generations.
  As upstream, `checklist_score` leaves out judged rollouts whose `CHECKLIST_SCORE` did not parse.

## Run

```bash
config_paths="resources_servers/mm_browsecomp/configs/mm_browsecomp.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
gym env start "+config_paths=[${config_paths}]"
```

The benchmark is `benchmarks/mm_browsecomp`.

## Licensing information

Code: Apache 2.0. Data: MM-BrowseComp, MIT (README badge and dataset card); the prepared rows hold decrypted
answers, so never publish them.
