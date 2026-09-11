# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare HLE-Verified evaluation data for NeMo Gym.

HLE-Verified (``skylenage/HLE-Verified``) is a re-annotation of Humanity's Last
Exam in which every question was re-checked by domain experts and sorted into
three subsets:

  * **Gold** — the question and its reference answer were confirmed correct as-is.
  * **Revision** — the reference answer was wrong or under-specified upstream and
    has been corrected here.
  * **Uncertain** — the annotators could not confirm the answer; excluded from the
    default eval split because grading against it is unreliable.

Mirrors NeMo Skills' ``nemo_skills/dataset/hle_verified/prepare.py``: the default
``text`` subset is the image-free rows of Gold + Revision, matching Skills'
``EVAL_SPLIT = "text"``. Rows are emitted in the same shape as
``benchmarks/hle/prepare.py`` so the ``equivalence_llm_judge`` resources server and
the official HLE judge prompt can be reused unchanged.

Field renames vs Skills (to match Gym's HLE rows):
  - Skills' ``problem``  -> Gym ``question``
  - Skills' ``id``       -> Gym ``uuid``
  - Skills' ``subset_for_metrics`` -> Gym ``category``

``--include-vision`` additionally keeps the image questions and materializes every
row's ``responses_create_params.input``, mirroring ``benchmarks/hle``'s vision
variant. Subset and modality are independent: ``--subset`` still selects verified
classes, and ``--include-vision`` decides whether image questions come along. Skills
has no vision counterpart, so that split is Gym-only and not comparable to it.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT = DATA_DIR / "hle_verified_benchmark.jsonl"
DEFAULT_OUTPUT_VISION = DATA_DIR / "hle_verified_benchmark_vision.jsonl"

# Applied at prepare time in vision mode, and at rollout time via `prompt_config` in
# text mode — the same file either way, so the two modes prompt identically.
PROMPT_CONFIG_FPATH = BENCHMARK_DIR / "prompts" / "default.yaml"

REPO_ID = "skylenage/HLE-Verified"

# Verified-class labels as they appear in the dataset's `Verified_Classes` column,
# mapped to the short subset names used by `--subset`. Copied from Skills'
# HLE_VERIFIED_CLASSES_MAP so the two pipelines select identical row sets.
VERIFIED_CLASSES_MAP = {
    "Gold subset": "gold",
    "Revision subset": "revision",
    "Uncertain subset": "uncertain",
}
VERIFIED_CLASSES_REVERSE_MAP = {v: k for k, v in VERIFIED_CLASSES_MAP.items()}

# The default eval subset: text-only rows from Gold + Revision (Uncertain dropped).
DEFAULT_SUBSET = "text"
SUBSETS = (DEFAULT_SUBSET, "all") + tuple(VERIFIED_CLASSES_MAP.values())

# Fields the upstream repo folds into a single JSON-encoded `json` column rather than
# exposing as top-level columns. Read back out here so rows carry them individually.
_PACKED_FIELDS = ("author_name", "rationale", "answer_type", "canary", "image")


def _unpack(row: dict) -> dict:
    """Return ``row`` with the fields packed into its ``json`` column hoisted to the top level.

    ``skylenage/HLE-Verified`` stores ``author_name`` / ``rationale`` / ``answer_type`` /
    ``canary`` / ``image`` as a JSON string in a ``json`` column. Top-level columns win if
    the repo ever promotes them, so this stays correct across either schema.
    """
    packed: dict = {}
    raw = row.get("json")
    if isinstance(raw, str) and raw:
        try:
            packed = json.loads(raw)
        except json.JSONDecodeError:
            packed = {}

    unpacked = dict(row)
    for field in _PACKED_FIELDS:
        if unpacked.get(field) is None:
            unpacked[field] = packed.get(field)
    return unpacked


def keep_row(row: dict, subset: str, include_vision: bool = False) -> bool:
    """Whether an (already unpacked) row belongs in ``subset``.

    Modality is orthogonal to ``subset``: image questions are dropped unless
    ``include_vision`` is set, because the text-mode prompt sends only the question
    text and an image question graded on its caption alone is unanswerable.
    """
    if row.get("image") and not include_vision:
        return False

    verified_class = row.get("Verified_Classes")
    if subset == "all":
        return True
    if subset == DEFAULT_SUBSET:
        # Gold + Revision; Uncertain answers are not reliable enough to grade against.
        return verified_class != VERIFIED_CLASSES_REVERSE_MAP["uncertain"]
    return verified_class == VERIFIED_CLASSES_REVERSE_MAP[subset]


@lru_cache(maxsize=1)
def _hle_build_input():
    """``benchmarks/hle``'s ``_build_input``, loaded from the sibling file BY PATH.

    Deliberately not ``from benchmarks.hle.prepare import _build_input``. ``benchmarks/``
    has no ``__init__.py``, so it is a namespace package: its ``__path__`` merges *every*
    ``benchmarks/`` directory on ``sys.path``, and a submodule comes from whichever
    portion holds it first. The eval container ships its own portion at
    ``/opt/nemo-gym/benchmarks`` ahead of the uploaded checkout, and that copy has
    ``hle/`` but not ``hle_verified/`` -- so the two halves of this function used to come
    from *different* checkouts:

        benchmarks.hle          -> /opt/nemo-gym/benchmarks/hle       (older, no _build_input)
        benchmarks.hle_verified -> the uploaded checkout              (this file)

    which failed on the cluster, at prepare time, after the allocation:
    ``ImportError: cannot import name '_build_input' from 'benchmarks.hle.prepare'``.
    Anchoring on ``__file__`` instead makes both halves come from this checkout, whatever
    else is on ``sys.path``.
    """
    path = BENCHMARK_DIR.parent / "hle" / "prepare.py"
    spec = importlib.util.spec_from_file_location("_hle_prepare_for_hle_verified", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmarks/hle's prepare.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_input


def format_entry(row: dict, prompt_config: Any = None) -> dict:
    """Map an unpacked HLE-Verified row to a Gym JSONL row.

    ``question`` / ``expected_answer`` are what the prompt template and the judge read;
    everything else is carried through for per-category analysis of the rollouts.

    Passing ``prompt_config`` switches to vision mode: the prompt is applied here rather
    than at rollout time and the row carries a materialized ``responses_create_params.input``,
    with image questions gaining an ``input_image`` block. Such rows are self-contained and
    must be used with ``prompt_config: null``, which is mutually exclusive with a
    pre-populated input.
    """
    entry = {
        "question": row["question"],
        "expected_answer": row["answer"],
        # Not used for grading — the judge compares free-form text — but useful for analysis.
        "answer_type": row.get("answer_type"),
        "uuid": row["id"],
        "category": row.get("category"),
        "raw_subject": row.get("raw_subject"),
        "verified_class": VERIFIED_CLASSES_MAP.get(row.get("Verified_Classes"), row.get("Verified_Classes")),
    }
    if prompt_config is not None:
        # Taken from benchmarks/hle for the same reason the judge prompt is shared:
        # hle_verified exists to be compared against hle, so the two must build their
        # multimodal inputs identically or the comparison measures the prompt instead.
        build_input = _hle_build_input()

        entry["has_image"] = bool(row.get("image"))
        entry["responses_create_params"] = {
            "input": build_input(prompt_config, row["question"], row.get("image") or "")
        }
    return entry


def prepare(
    subset: str = DEFAULT_SUBSET,
    output_fpath: Optional[Union[str, Path]] = None,
    include_vision: bool = False,
) -> Path:
    """Download HLE-Verified and convert to Gym JSONL format.

    Args:
        subset: Which verified classes to keep. ``"text"`` (default) is Gold +
            Revision, matching Skills' ``EVAL_SPLIT``. ``"gold"`` / ``"revision"`` /
            ``"uncertain"`` select a single class, and ``"all"`` keeps every class.
            Orthogonal to ``include_vision``, which decides modality.
        include_vision: When ``False`` (default), image questions are dropped and rows
            carry raw fields to be templated at rollout time via ``prompt_config``. When
            ``True``, image questions are kept and every row is materialized with
            ``responses_create_params.input``; use with ``prompt_config: null`` and a
            vision-capable policy model.
        output_fpath: Where to write. Defaults to
            ``benchmarks/hle_verified/data/hle_verified_benchmark.jsonl`` (or the
            ``_vision`` variant), which is the path ``config.yaml`` / ``config_vision.yaml``
            point at — override it only when preparing a non-default ``subset``, so the
            two don't clobber each other.

    Returns:
        Path to the written JSONL file.
    """
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}")

    from datasets import load_dataset

    from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict

    print(f"Downloading {REPO_ID} from HuggingFace...")
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    ds = load_dataset(REPO_ID, split="train", token=hf_token)

    prompt_config = None
    if include_vision:
        from nemo_gym.prompt import load_prompt_config

        prompt_config = load_prompt_config(str(PROMPT_CONFIG_FPATH))

    if output_fpath is not None:
        output_fpath = Path(output_fpath)
    else:
        output_fpath = DEFAULT_OUTPUT_VISION if include_vision else DEFAULT_OUTPUT
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    n_skipped = 0
    n_image = 0
    for raw_row in ds:
        row: dict[str, Any] = _unpack(raw_row)
        if not keep_row(row, subset, include_vision=include_vision):
            n_skipped += 1
            continue
        n_image += bool(row.get("image"))
        lines.append(json.dumps(format_entry(row, prompt_config), ensure_ascii=False) + "\n")

    with open(output_fpath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    modality = f", including {n_image} image questions" if include_vision else ""
    print(f"Wrote {len(lines)} problems to {output_fpath} (skipped {n_skipped} outside subset {subset!r}{modality})")
    return output_fpath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare HLE-Verified benchmark data.")
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        choices=SUBSETS,
        help="Which verified classes to keep (default: text = Gold + Revision).",
    )
    parser.add_argument(
        "--include-vision",
        action="store_true",
        help="Keep image questions and materialize inputs (writes hle_verified_benchmark_vision.jsonl).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}, or the _vision variant).",
    )
    args = parser.parse_args()
    prepare(subset=args.subset, output_fpath=args.output, include_vision=args.include_vision)
