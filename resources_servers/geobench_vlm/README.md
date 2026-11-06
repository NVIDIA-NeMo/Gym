# GEOBench-VLM NemoGym

NemoGym implementation of
[GEOBench-VLM](https://github.com/The-AI-Alliance/GEO-Bench-VLM), the
vision-language benchmark from "GEOBench-VLM: Benchmarking Vision-Language
Models for Geospatial Tasks"
([arXiv:2411.19325](https://arxiv.org/abs/2411.19325)).

This is a single-turn image benchmark. It is separate from the GeoBenchX
tool-calling Gym.

## Scope

- Official multiple-choice prompt: answer with one letter, A through E.
- Single-image and before/after temporal items.
- Reward `1` when the parsed letter matches `ground_truth_option`, otherwise `0`.
- `geobench_vlm_first_letter_correct` records the released temporal script's
  rule, which keeps only the first character.
- Headline paper metric: MCQ accuracy. LLaVA-OneVision is reported at 41.7%.
  Referring-expression IoU and segmentation masks are not scored here.

The committed dataset is `data/example.jsonl` (six rows from five synthetic
questions, because one question has two prompts). The 3,211-question Hugging
Face split and its images are not vendored, and `.gitignore` keeps locally built
rows out of git. See `DATA-ATTESTATION.md`.

## Data use

GEOBench-VLM is for evaluation and testing only; do not train on it. The
datasets approved for training are BigEarthNet v2, EuroSAT, USDA Cropland
Data Layer 2020–2023, NAIP, Microsoft Global ML Building Footprints,
OpenStreetMap, SpaceNet 2, and SpaceNet 3. `training_sources.py` records each
one's use, license, and required attribution. This server does not download
them, and `initiate` refuses.

## Build a local dataset

Place the official tree so `Single/qa.json` and, if you have it,
`Temporal/qa.json` exist, then:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from resources_servers.geobench_vlm.task_data import load_qa_tree, materialize
rows = load_qa_tree(Path("/path/to/GEOBench-VLM"))
materialize(Path("resources_servers/geobench_vlm/data/benchmark.jsonl"), rows)
print(len(rows))
PY
```

Point the dataset entry in `configs/geobench_vlm.yaml` at that file.

### Perception subset

`materialize_perception` builds 1,557 questions from eight classification
and counting tasks (`PERCEPTION_TASKS`) in the Hugging Face `Single.parquet`:
one row per question, text before the image, temperature 0, and
8,192 output tokens. It also writes a blind copy with no image, which
measures how much the options alone give away. The builder checks the
parquet's SHA-256 and the per-task counts (`PERCEPTION_COUNTS`) against
revision `HF_REVISION`, and stops before writing if either differs:

```bash
huggingface-cli download aialliance/GEOBench-VLM Single.parquet --repo-type dataset \
  --revision b3485fe4971213ee13f91960638ddc5c4a098e70 --local-dir /path/to
.venv/bin/python -c "
from pathlib import Path
from resources_servers.geobench_vlm.task_data import materialize_perception
materialize_perception(Path('/path/to/Single.parquet'), Path('resources_servers/geobench_vlm/data'))"
```

Pair those files with `configs/perception_protocol_reasoning_off.yaml` or
`_on.yaml` as the policy model config. `nemotron_omni_reasoning_{off,on}.yaml`
use the Nemotron 3 Nano Omni model-card sampling instead. Image
files are read and sent as data URLs when the row is built.

## Run the example

```bash
.venv/bin/gym env start \
  "+config_paths=[responses_api_models/inference_provider/configs/inference_provider.yaml,resources_servers/geobench_vlm/configs/geobench_vlm.yaml]"

.venv/bin/gym eval run --no-serve \
  --agent geobench_vlm_simple_agent \
  --input resources_servers/geobench_vlm/data/example.jsonl \
  --output resources_servers/geobench_vlm/data/example_rollouts.jsonl \
  --limit 6 --num-repeats 1 --concurrency 1
```

`env.yaml` needs `policy_base_url`, `policy_api_key`, and `policy_model_name`
for a vision-capable model. Do not commit that file.
