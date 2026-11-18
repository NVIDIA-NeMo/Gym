# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the completed MMRB v1 download to self-contained Gym judge inputs.

The published judging prompt is downloaded from the same revision as the upstream data downloader.
"""

import argparse
import base64
import json
from collections import Counter
from mimetypes import guess_type
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from resources_servers.genrm_eval.upstream_prompt import download_prompt


def load_prompt(config: Path | None = None) -> str:
    if config is not None:
        return yaml.safe_load(config.read_text())["judge_prompt"]
    from benchmarks.genrm_mmrb1.prepare import UPSTREAM_REVISION

    return download_prompt(
        repo="facebookresearch/multimodal_rewardbench",
        revision=UPSTREAM_REVISION,
        filename="scripts/1_run_model_as_judge_gpt4o.py",
        name="judge_prompt",
        cache_dir=Path(__file__).parent / ".cache",
    )


def subcategory(row: dict) -> str:
    category = row["Category"]
    if category == "safety":
        return "safety/bias" if row["ID"].lower().startswith("pairs") else "safety/toxicity"
    if category == "reasoning":
        return "reasoning/math" if row["ID"].lower().startswith("math") else "reasoning/coding"
    return category


def convert(row: dict, data_dir: Path, judge_prompt: str, *, genrm: bool = False, record_index: int = 0) -> dict:
    if row["Better"] not in ("Output1", "Output2"):
        raise ValueError(f"Unknown label for {row['ID']}: {row['Better']}")
    if not row["Text"].strip():
        raise ValueError(f"Missing prepared question for {row['ID']}")
    image = (data_dir / row["Image"]).resolve()
    if not image.is_relative_to(data_dir.resolve()):
        raise ValueError(f"Image must be inside data directory: {row['Image']}")
    mime = guess_type(image.name)[0] or "application/octet-stream"
    image_part = {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode()}",
        "detail": "auto",
    }
    question = (
        row["Text"]
        if genrm
        else judge_prompt.format(question=row["Text"], answer_a=row["Output1"], answer_b=row["Output2"])
    )
    params = {"input": [{"role": "user", "content": [{"type": "input_text", "text": question}, image_part]}]}
    expected = "A" if row["Better"] == "Output1" else "B"
    if genrm:
        params["metadata"] = {"response_1": row["Output1"], "response_2": row["Output2"]}
    # The release reuses question IDs for distinct comparisons; retain every record.
    record_id = f"{row['ID']}:{record_index}"
    return {
        "id": record_id,
        "responses_create_params": params,
        "verifier_metadata": {
            "pair_id": record_id,
            "original_id": row["ID"],
            "expected_label": expected,
            "order": "forward",
            "source": row["ID"].split("_")[0],
            "pair_type": "text_text",
            "category": row["Category"],
            "subcategory": subcategory(row),
            "ground_truth_label": 3 if expected == "A" else 4,
            "data_type": subcategory(row),
            "judge_mode": "genrm" if genrm else "llm_as_a_judge",
        },
    }


def prepare(
    source_dir: Path,
    output: Path,
    *,
    genrm: bool = False,
    prompt_config: Path | None = None,
) -> None:
    source = source_dir / "data/all_data.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is missing. Finish scripts/0_download_data.py first; do not use all_data_release.json."
        )
    rows = json.loads(source.read_text())
    if not rows:
        raise ValueError("The prepared dataset is empty")
    ids = [row["ID"] for row in rows]
    release = source_dir / "data/all_data_release.json"
    if release.is_file() and Counter(ids) != Counter(row["ID"] for row in json.loads(release.read_text())):
        raise ValueError("Prepared IDs do not match the full public release")
    prompt = "" if genrm else load_prompt(prompt_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for index, row in enumerate(rows):
                task = convert(row, source.parent, prompt, genrm=genrm, record_index=index)
                stream.write(json.dumps(task, ensure_ascii=False) + "\n")
            stream.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"pairs/requests: {len(rows)}; images per request: 1; output: {output}")
    print("categories:", dict(Counter(row["Category"] for row in rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Downloaded upstream repository")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-mode", choices=["genrm", "llm_as_a_judge"], default="genrm")
    parser.add_argument(
        "--judge-prompt-config",
        type=Path,
        help="Optional YAML prompt override; otherwise download the pinned upstream prompt",
    )
    args = parser.parse_args()
    prepare(args.source_dir, args.output, genrm=args.judge_mode == "genrm", prompt_config=args.judge_prompt_config)
