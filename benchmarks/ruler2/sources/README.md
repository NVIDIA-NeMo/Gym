# RULER v2 generator scripts

Vendored verbatim from
[`nemo_skills/dataset/ruler2/`](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/dataset/ruler2)
to keep the Gym benchmark self-contained.

| File | Source |
| --- | --- |
| `prepare_niah.py` | [`nemo_skills/dataset/ruler2/prepare_niah.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/prepare_niah.py) |
| `prepare_mmlu.py` | [`nemo_skills/dataset/ruler2/prepare_mmlu.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/prepare_mmlu.py) |
| `prepare_qa.py` | [`nemo_skills/dataset/ruler2/prepare_qa.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/prepare_qa.py) |
| `tokenizer.py` | [`nemo_skills/dataset/ruler2/tokenizer.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/tokenizer.py) |

These scripts are invoked from `benchmarks/ruler2/prepare.py` via a private
venv. The orchestrator concatenates their per-task outputs into a single
Gym benchmark JSONL with per-row `task`, `eval_type`, and `match_type`.

License: Apache 2.0 (matches upstream `nemo_skills`).
