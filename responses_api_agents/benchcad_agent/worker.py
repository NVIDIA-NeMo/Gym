# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted upstream adapter, executed with BenchCAD's Python, without importing Gym.

Prediction programs are NEVER executed here. The Gym agent executes them in a
fresh sandbox, then passes only the resulting STEP to this scorer.
"""

import argparse
import functools
import importlib.util
import json
import math
import re
import runpy
import sys
from pathlib import Path
from types import ModuleType


DATASET_REVISION = "5919f578ab09ec283603a082fab07c7639ab56eb"


def download(root: Path, output: Path, task: str, limit: int | None) -> None:
    """Run the original data converter against a pinned Hugging Face snapshot."""
    import huggingface_hub

    # The upstream CLI has no revision argument. Bind its two download APIs in
    # this short-lived worker so the original conversion code remains unchanged.
    huggingface_hub.hf_hub_download = functools.partial(
        huggingface_hub.hf_hub_download,
        revision=DATASET_REVISION,
    )
    huggingface_hub.snapshot_download = functools.partial(
        huggingface_hub.snapshot_download,
        revision=DATASET_REVISION,
    )
    script = {
        "vision2code": "Vision2Code/tools/download_codegen_bench.py",
        "codeedit": "CodeEdit/tools/download_edit_bench.py",
        "qa": "QA/tools/download_qa_img.py",
    }[task]
    sys.argv = [str(root / script), "--out", str(output)]
    if limit is not None:
        sys.argv += ["--limit", str(limit)]
    if task == "vision2code" and limit is not None:
        # A smoke needs only the first shard; do not download all 17,900 renders.
        sys.argv += ["--max-shards", "1"]
    runpy.run_path(str(root / script), run_name="__main__")


def load_module(root: Path, relative: str) -> ModuleType:
    """Load a task module without colliding with another task's `pipeline` package."""
    spec = importlib.util.spec_from_file_location(relative.replace("/", "_"), root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_text(raw: str) -> str:
    """Remove complete reasoning blocks before applying upstream answer parsing."""
    return re.sub(r"<(think|thinking)>.*?</\1>", "", raw, flags=re.DOTALL).strip()


def score_qa(root: Path, raw: str, pairs: list[dict]) -> dict:
    """Preserve upstream's per-question rounding, type semantics and part mean."""
    scorer = load_module(root, "QA/scoring/qa_score.py")
    answers = scorer.parse_json_numbers(final_text(raw), len(pairs))
    if answers is None or not all(math.isfinite(value) for value in answers):
        return {"reward": 0.0, "status": "parse_fail", "question_scores": []}
    scores = [
        scorer.qa_score_single(answer, pair["answer"], pair.get("type", "dim"))
        for answer, pair in zip(answers, pairs, strict=True)
    ]
    return {"reward": scorer.qa_score(answers, pairs), "status": "ok", "question_scores": scores}


def export_records(root: Path, *, source: Path, output: Path, task: str, limit: int | None) -> Path:
    """Adapt upstream records and prompts; keep reference data out of agent inputs."""
    import shutil

    from benchcad_core.scoring.exec_cq import execute_cq_to_step

    records = [json.loads(line) for line in (source / "records.jsonl").read_text().splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]
    module_path = {
        "vision2code": "Vision2Code/pipeline/prompt.py",
        "codeedit": "CodeEdit/pipeline/modes.py",
        "code_qa": "QA/pipeline/prompt.py",
        "vision_qa": "QA/pipeline/prompt.py",
    }[task]
    prompt_module = load_module(root, module_path)
    rows = []
    seen = set()
    for record in records:
        rid = record["record_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", rid) or rid in {".", ".."} or rid in seen:
            raise ValueError(f"Unsafe or duplicate BenchCAD record_id: {rid!r}")
        seen.add(rid)
        directory = output / task / rid
        directory.mkdir(parents=True, exist_ok=True)
        if task == "vision2code":
            system, user, images = prompt_module.build(record, source)
        else:
            mode = {"codeedit": "instruction", "code_qa": "code", "vision_qa": "img"}[task]
            system, user, images = prompt_module.build(record, source, mode)
        image_names = []
        for number, image in enumerate(images):
            name = f"view_{number}.png"
            shutil.copyfile(image, directory / name)
            image_names.append(name)
        metadata = {"task": task, "record_id": rid, "family": record.get("family", ""), "images": image_names}
        if task in {"vision_qa", "code_qa"}:
            metadata["qa_pairs"] = record["qa_pairs"]
        else:
            key = "gt_step_path" if task == "codeedit" else "step_path"
            reference = source / record[key]
            if not reference.is_file():
                # Upstream's committed CodeEdit fixtures contain code, but no STEP assets.
                code_key = "gt_code_path" if task == "codeedit" else "code_path"
                execute_cq_to_step((source / record[code_key]).read_text(), reference)
            shutil.copyfile(reference, directory / "reference.step")
            if task == "codeedit":
                metadata["baseline_iou"] = float(record["iou"])
        (directory / "task.json").write_text(json.dumps(metadata, allow_nan=False))
        image_instruction = "".join(f"\nUse the read tool to inspect /workspace/{name}." for name in image_names)
        prompt = system + "\n\n" + user + image_instruction
        prompt += (
            "\n\nYou may use the terminal and Python to work on this task. "
            "Your final response must contain the complete answer in the format above, "
            "even if you also save files. CadQuery is available through `python`."
        )
        rows.append(
            {
                "task": task,
                "record_id": rid,
                "family": metadata["family"],
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
            }
        )
    destination = output / f"{task}.jsonl"
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return destination


def main() -> None:
    """Expose preparation, code extraction and scoring to the Gym process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--task", choices=["vision2code", "codeedit", "qa"], required=True)
    fetch.add_argument("--limit", type=int)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--task", choices=["vision2code", "codeedit", "vision_qa", "code_qa"], required=True)
    export.add_argument("--limit", type=int)
    patch = sub.add_parser("patch")
    patch.add_argument("--answer", type=Path, required=True)
    patch.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--task-dir", type=Path, required=True)
    score.add_argument("--answer", type=Path, required=True)
    score.add_argument("--step", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    if args.command == "download":
        download(args.root, args.output, args.task, args.limit)
    elif args.command == "export":
        print(export_records(args.root, source=args.source, output=args.output, task=args.task, limit=args.limit))
    elif args.command == "patch":
        from benchcad_core.scoring.exec_cq import _patch_export, extract_code

        code = extract_code(final_text(args.answer.read_text()))
        if code:
            args.output.write_text(_patch_export(code, Path("/workspace/prediction.step")))
        print(json.dumps({"has_code": bool(code)}))
    else:
        metadata = json.loads((args.task_dir / "task.json").read_text())
        if metadata["task"] in {"vision_qa", "code_qa"}:
            result = score_qa(args.root, args.answer.read_text(), metadata["qa_pairs"])
        else:
            from benchcad_core.scoring.iou import iou_step_vs_step, norm_iou

            iou = iou_step_vs_step(args.step, args.task_dir / "reference.step")
            reward = norm_iou(iou, metadata["baseline_iou"]) if metadata["task"] == "codeedit" else iou
            result = {"reward": round(reward, 4), "iou": round(iou, 4), "status": "ok"}
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
