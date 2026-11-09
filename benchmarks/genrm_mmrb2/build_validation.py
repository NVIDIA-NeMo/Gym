# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the MMRB2 reasoning Parquet split into self-contained Gym judge inputs.

The published reasoning prompt is downloaded from a pinned upstream repository revision.
"""

import argparse
import base64
import json
import re
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow.parquet as pq
from PIL import Image

from resources_servers.genrm_eval.upstream_prompt import download_prompt


UPSTREAM_REVISION = "7ff0552cf298441af5ebbabea634b682ef04afee"  # pragma: allowlist secret


def load_prompt() -> str:
    return download_prompt(
        repo="facebookresearch/MMRB2",
        revision=UPSTREAM_REVISION,
        filename="evaluate/generate_judgements/evaluator_apis/llm_judges/evaluation_prompts.py",
        name="get_reasoning_prompt",
        cache_dir=Path(__file__).parent / ".cache",
    )


def text_part(text):
    return {"type": "input_text", "text": text}


def interleave(text, images):
    """Each text field indexes its own image list; never move images to the end."""
    parts, used = [], set()
    for segment in re.split(r"(<image_\d+>)", text or ""):
        match = re.fullmatch(r"<image_(\d+)>", segment)
        if match:
            index = int(match[1])
            if index >= len(images):
                raise ValueError(f"Image placeholder {segment} has no corresponding image")
            image = images[index]
            data = image["bytes"] or Path(image["path"]).read_bytes()
            with Image.open(BytesIO(data)) as decoded:
                mime = Image.MIME[decoded.format]
            parts.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{base64.b64encode(data).decode()}",
                    "detail": "auto",
                }
            )
            used.add(index)
        elif segment:
            parts.append(text_part(segment))
    if used != set(range(len(images))):
        raise ValueError("Images without text placeholders would be lost")
    return parts


def convert(row, genrm=False, *, judge_prompt=None):
    if row["response_a_images"] or row["response_b_images"]:
        raise ValueError("Only text-only candidate responses are supported")
    chosen = row["chosen"].upper()
    if chosen not in ("A", "B"):
        raise ValueError(f"Invalid human preference for {row['pair_id']}")
    prompt = interleave(row["prompt_text"], row["prompt_images"])
    candidates = [interleave(row[f"response_{side}_text"], row[f"response_{side}_images"]) for side in ("a", "b")]
    for order in ("forward", "reverse"):
        first, second = candidates if order == "forward" else candidates[::-1]
        if genrm:
            # The checkpoint's chat template supplies judging instructions; no MMRB2 wrapper.
            params = {
                "input": [{"role": "user", "content": prompt}],
                "metadata": {
                    "response_1": row["response_a_text"] if order == "forward" else row["response_b_text"],
                    "response_2": row["response_b_text"] if order == "forward" else row["response_a_text"],
                },
            }
        else:
            if judge_prompt is None:
                judge_prompt = load_prompt()
            content = [text_part(judge_prompt), text_part("[ORIGINAL PROMPT TO MODEL:]")]
            content += prompt + [text_part("[RESPONSE A:]")] + first + [text_part("[RESPONSE B:]")] + second
            params = {"input": [{"role": "user", "content": content}]}
        yield {
            "id": f"{row['pair_id']}:{order}",
            "responses_create_params": params,
            "verifier_metadata": {
                "pair_id": row["pair_id"],
                "expected_label": chosen if order == "forward" else ("B" if chosen == "A" else "A"),
                "order": order,
                "source": row["prompt_source"],
                "pair_type": "chosen_text__rejected_text",
                "task": "reasoning",
                "response_modality": "text_only",
                "judge_mode": "genrm" if genrm else "llm_as_a_judge",
                "ground_truth_label": 3 if (chosen == "A") == (order == "forward") else 4,
                "data_type": row["prompt_source"],
            },
        }


def prepare(source_dir, output, genrm=False, *, task="reasoning", response_modality="text_only"):
    if task != "reasoning" or response_modality != "text_only":
        raise ValueError("Only task=reasoning with response_modality=text_only is supported")
    files = sorted(source_dir.glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No test Parquet files in {source_dir}")
    judge_prompt = "" if genrm else load_prompt()
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    selected = 0
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for file in files:
                for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
                    row = batch.to_pylist()[0]
                    if row["pair_id"] in seen:
                        raise ValueError(f"Duplicate pair: {row['pair_id']}")
                    seen.add(row["pair_id"])
                    if row["response_a_images"] or row["response_b_images"]:
                        continue
                    selected += 1
                    for task in convert(row, genrm=genrm, judge_prompt=judge_prompt):
                        stream.write(json.dumps(task, ensure_ascii=False) + "\n")
            if len(seen) != 1000:
                raise ValueError(f"Expected the full 1,000-pair reasoning split, got {len(seen)}")
            if selected != 350:
                raise ValueError(f"Expected 350 text-only candidate pairs, got {selected}")
            stream.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(
        f"task: reasoning; response_modality: text_only; pairs: {selected}; judge requests: {2 * selected}; output: {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Downloaded reasoning Parquet directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-mode", choices=["genrm", "llm_as_a_judge"], default="genrm")
    parser.add_argument("--task", choices=["reasoning"], default="reasoning")
    parser.add_argument("--response-modality", choices=["text_only"], default="text_only")
    args = parser.parse_args()
    prepare(
        args.source_dir,
        args.output,
        genrm=args.judge_mode == "genrm",
        task=args.task,
        response_modality=args.response_modality,
    )
