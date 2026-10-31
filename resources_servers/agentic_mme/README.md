# Agentic-MME (atomic tools)

Native Gym implementation of the public Agentic-MME atomic interface:
14 image tools plus 3 optional retrieval tools. The verifier reports **Track B,
task-only accuracy**, not the paper's full S/V process scores. No reward is
given for using a tool. No benchmark tasks are configured as training data.

Sources:

- [Paper](https://arxiv.org/abs/2604.03016), especially appendices A, B, E.
- [Reference interface](https://github.com/ChoS3nE11ven/Agentic-MME/tree/5e5dfbdc7b40b53dea17d5ce754c14034acb185b).
- [Official data](https://huggingface.co/datasets/Crystal1047/Agentic-MME/tree/b9ea9d3f68ff896d83fec666b6143da594c054a8).

## Tools

Image tools: crop, rotate, flip, resize, enhance, grayscale, autocontrast,
blur, sharpen, denoise, edge_detect, invert, equalize, threshold.

Retrieval: google_search (Serper), google_lens_search (Serper + ImgBB),
fetch_webpage (Jina Reader). The paper describes download_image, but the released
atomic harness disables it; this integration likewise does not expose it.
Free-form Python/Gen mode is not implemented.

Images use zero-based indices. All original images are indexed first; each
successful visual operation appends one image. Coordinates are normalized to
0–1000. Positive rotations are counterclockwise. New images are actually included
in the next model input, not just returned as paths. Original images must be
PNG/JPEG/WebP data URLs. Local paths and arbitrary remote URLs are not opened.

Tools enforce per-image/episode pixel limits, argument validation and bounded
rollout concurrency. OpenCV is required for consistent denoise/Canny/Sobel
semantics; there is no silent Pillow approximation.

## Scoring and traces

Task metadata uses the released shape:

    verifier_metadata:
      task_id: ...
      golden_answer:
        value: "44.6 million"
        match_type: contains
      process_evaluation:
        efficiency:
          reference_tool_calls: 2

Accepted match modes are exact (case-sensitive), contains (case-insensitive,
with the released short-number boundary rule), and numeric (absolute tolerance).
The default is contains, matching the released evaluator, although the paper
describes normalized exact match. Empty/missing/nonfinite targets are rejected.
This preserves the released contains behavior, including its limitations; it is
not a hardened training verifier. Final answers use <answer>...</answer>, with
plain terminal text as a fallback; reasoning blocks are excluded.

The response includes tool_trace, counts, overthink and process_scores_available=false.
Overthink counts successful observable interactions relative to reference_tool_calls;
failed requests are counted separately. It never changes the reward.
Full S/V, V-tool and V-true checkpoint judging and published-score reproduction
remain unimplemented. Tool traces retain arguments, outputs and image artifacts
for future process evaluation. Do not report these runs as official Track A.

The default budget is 15 interaction rounds / 15 attempted tools, plus at most
one final-answer-only model call. Invalid tool calls consume the budget.
All prior messages and model token metadata are preserved between turns.

## Data

The five checked-in examples are small **synthetic smoke tasks**, not official
benchmark examples or a baseline. The official dataset can be converted with
the companion prepare_gym.py script in the separate Agentic-MME source checkout:

    uv run python /path/to/Agentic-MME-source/prepare_gym.py \
      --dataset-root /path/to/downloaded/Agentic-MME \
      --output resources_servers/agentic_mme/data/validation.jsonl

The converter retains task IDs, gold answers, process checkpoints and difficulty
in verifier_metadata, never in the model prompt. It embeds input images and
does not expose reference crops or search answers. The released HF split is
called train, but it is an evaluation benchmark; do not train on it by default.
Full converted datasets are ignored by git. No registry upload is performed.

## Run

From the Gym checkout, install the agent's requirements in its isolated environment.
Provide policy_base_url, policy_api_key and policy_model_name in private Gym config.

    gym env start \
      --config resources_servers/agentic_mme/configs/agentic_mme.yaml \
      --model-type openai_model

    gym eval run --no-serve --agent agentic_mme_agent \
      --input resources_servers/agentic_mme/data/example.jsonl \
      --output /tmp/agentic-mme-smoke.jsonl --num-repeats 1 \
      --max-output-tokens 4096 --temperature 0

Visual-only is an ablation, not full benchmark parity. To enable all 17 tools,
merge configs/agentic_mme_search.yaml after the base config and supply
agentic_mme_serper_api_key, agentic_mme_imgbb_api_key and agentic_mme_jina_api_key
(the Jina key may be an empty string). This explicitly allows images to be sent
to ImgBB/Serper. Only use it with data approved for those services.

For deterministic offline retrieval, set retrieval.mode=replay and put an ordered
retrieval_replay list on each input row; entries contain tool_name, arguments and
output from a previous tool_trace. Replay must match tool and normalized arguments.
Mismatches fail visibly and never fall back to the network.

Live model/provider smoke testing and baselining are required before marking
verified=true. Unit tests are not evidence of benchmark performance.

## Licensing

New Gym code: Apache-2.0. The official dataset card declares Apache-2.0.
No reference-repository implementation or paper text is vendored: the reference
code checkout has no declared code license. Tool interfaces were independently
implemented using Pillow (HPND), NumPy (BSD) and OpenCV (Apache-2.0).
