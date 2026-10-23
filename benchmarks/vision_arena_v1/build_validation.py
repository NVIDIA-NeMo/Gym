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
"""Build self-contained VisionArena rows from Parquet prompts and baseline generations."""

import argparse
import base64
import json
import mimetypes
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO


def build_validation(*, questions: Path, responses: Path, output: Path, allow_partial: bool = False) -> int:
    """Embed prompts, images, and complete baseline answers; require all 500 unless explicitly relaxed."""
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    prompts = pq.read_table(questions).to_pylist()
    ids = {row["question_id"] for row in prompts}
    if len(prompts) != 500 or len(ids) != 500:
        raise ValueError("Expected exactly 500 distinct VisionArena questions")

    # Keep the last usable answer per ID; failed retries must not erase successful ones.
    baselines = {}
    with responses.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row["question_id"]
            if question_id not in ids:
                raise ValueError(f"Baseline question ID is absent from prompts: {question_id}")
            if "baseline_answer" in row:
                # Answers-only artifacts retain upstream IDs, but no prompts or images.
                answer = row["baseline_answer"]
                if question_id in baselines:
                    raise ValueError(f"Duplicate baseline question ID: {question_id}")
                if not isinstance(answer, str) or not answer.strip() or not row.get("baseline_model"):
                    raise ValueError(f"Invalid baseline answer: {question_id}")
                # Answers-only artifacts must follow the same final-answer contract as raw generations.
                answer = re.sub(r"<(think|thinking)>.*?</\1>", "", answer, flags=re.DOTALL).strip()
                if not answer or re.search(r"<(?:think|thinking)>", answer):
                    raise ValueError(f"Invalid baseline final answer: {question_id}")
                baselines[question_id] = (
                    {"model": row["baseline_model"], "response": {"id": row.get("baseline_response_id")}},
                    answer,
                )
                continue
            if row.get("error"):
                continue
            choices = (row.get("response") or {}).get("choices") or []
            if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                continue
            answer = (choices[0].get("message") or {}).get("content")
            if not isinstance(answer, str):
                continue
            # Baseline reasoning must never appear in the answer sent to the judge.
            answer = re.sub(r"<(think|thinking)>.*?</\1>", "", answer, flags=re.DOTALL).strip()
            if not answer or re.search(r"<(?:think|thinking)>", answer):
                continue
            baselines[question_id] = (row, answer)

    # Require full coverage unless a partial diagnostic dataset was explicitly requested.
    missing = ids - baselines.keys()
    if missing and not allow_partial:
        raise ValueError(
            f"Missing usable baseline answers for {len(missing)}/500 prompts. "
            "Finish/resume generation first. Example IDs: " + ", ".join(sorted(missing)[:5])
        )
    prompts = [prompt for prompt in prompts if prompt["question_id"] in baselines]
    models = {row["model"] for row, _ in baselines.values()}
    if len(models) != 1:
        raise ValueError(f"Expected one baseline model, got {sorted(models)}")

    # Validate image and prompt structure before opening the output.
    for prompt in prompts:
        if len(prompt["turns"]) != 1 or len(prompt["images"]) != 1:
            raise ValueError(f"Expected one turn and one image: {prompt['question_id']}")
        image = prompt["images"][0]
        mime = mimetypes.guess_type(image["path"])[0]
        if not mime or not mime.startswith("image/") or not image["bytes"]:
            raise ValueError(f"Invalid image: {prompt['question_id']}")

    # Write beside the destination so publishing is atomic on the same filesystem.
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as destination:
        temporary = Path(destination.name)
        try:
            _write_rows(prompts, baselines, destination)
            destination.close()
            # Publish only a complete dataset; do not overwrite an existing destination.
            output.hardlink_to(temporary)
        finally:
            temporary.unlink(missing_ok=True)
    return len(prompts)


def _write_rows(prompts: list[dict], baselines: dict, destination: TextIO) -> None:
    """Serialize validated comparisons to the temporary dataset."""
    for prompt in prompts:
        baseline, answer = baselines[prompt["question_id"]]
        question = "\n".join(part["content"] for part in prompt["turns"][0])
        image = prompt["images"][0]
        mime = mimetypes.guess_type(image["path"])[0]
        # Embed image bytes so evaluation does not depend on local paths or URLs.
        encoded = base64.b64encode(image["bytes"]).decode()
        row = {
            "question_id": prompt["question_id"],
            "question": question,
            "category": "vision_arena_v1",
            "baseline_answer": answer,
            "baseline_model": baseline["model"],
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": question},
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime};base64,{encoded}",
                                "detail": "auto",
                            },
                        ],
                    }
                ],
            },
            "metadata": {
                "source": "lmarena-ai/vision-arena-bench-v0.1",
                "cluster_name": prompt["cluster_name"],
                "baseline_response_id": baseline["response"].get("id"),
                "baseline_request_params": baseline.get("request_params"),
            },
        }
        destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Merge VisionArena prompts and baseline answers into an evaluation dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-partial", action="store_true", help="Include only prompts with usable baseline answers"
    )
    args = parser.parse_args()
    count = build_validation(
        questions=args.questions, responses=args.responses, output=args.output, allow_partial=args.allow_partial
    )
    print(f"Written {count} prompts to {args.output}")


if __name__ == "__main__":
    main()
