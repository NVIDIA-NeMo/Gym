#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage a Gym GDPVal row for the pending OpenHands integration.

Requires locally materialized reference files. No network or model calls.
Mount only task_dir/agent into the task sandbox at /workspace; keep judge.json
and provenance.json outside that mount. This does not run OpenHands.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath


def parsed(value, expected, field):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, expected):
        raise ValueError(f"{field} must be {expected.__name__}")
    return value


def reference_name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Reference filename must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError(f"Unsafe reference filename: {value!r}")
    return path


def stage(row, references, destination, dataset_identity):
    """Stage a fresh task directory; READY marks completed publication."""
    task_id = row.get("task_id")
    prompt = row.get("prompt")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a nonempty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if not dataset_identity.strip():
        raise ValueError("A dataset identity including version and split is required")
    names = parsed(row.get("reference_files", []), list, "reference_files")
    names = [reference_name(name) for name in names]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate reference filenames")
    rubric = row.get("rubric_json", {})
    if isinstance(rubric, str):
        rubric = json.loads(rubric)
    if not isinstance(rubric, (dict, list)):
        raise ValueError("rubric_json must be a JSON object or array")
    pretty = row.get("rubric_pretty", "")
    if not isinstance(pretty, str):
        raise ValueError("rubric_pretty must be a string")
    if not rubric and not pretty.strip():
        raise ValueError("A rubric is required for a judged rollout")
    references = Path(references).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    sources = []
    for name in names:
        source = references.joinpath(*name.parts)
        for parent in [source, *source.parents]:
            if parent == references:
                break
            if parent.is_symlink():
                raise ValueError(f"Symlink reference is not accepted: {name}")
        source.resolve(strict=True).relative_to(references)
        if not source.is_file():
            raise ValueError(f"Reference must be a regular file: {name}")
        sources.append(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=destination.parent))
    try:
        agent = temporary / "agent"
        inputs = agent / "reference_files"
        inputs.mkdir(parents=True)
        (agent / "deliverables").mkdir()
        manifest = []
        for name, source in zip(names, sources):
            target = inputs.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest.append({"path": str(name), "bytes": target.stat().st_size, "sha256": digest.hexdigest()})
        input_paths = [f"/workspace/reference_files/{name}" for name in names]
        agent_prompt = (prompt + "\n\nTask input files (JSON list):\n" + json.dumps(input_paths)
                        + "\nSave final deliverables under /workspace/deliverables. "
                        + "Identify the files you produced in your final response.\n")
        agent_prompt += (
            "\nDocument tools available in this environment: pdftotext, pdftoppm, tesseract, "
            "and Python modules fitz (PyMuPDF), pypdf, pytesseract, and reportlab. "
            "Use pypdf rather than the unavailable legacy PyPDF2 module. "
            "If PDF text extraction is empty, check form fields and use OCR for scanned pages: "
            "render pages with pdftoppm or fitz, then run tesseract on the images. "
            "An empty text extraction does not mean the document is blank. "
            "Keep temporary OCR files under /workspace, outside /workspace/deliverables.\n"
        )
        (agent / "prompt.txt").write_text(agent_prompt, encoding="utf-8")
        judge = {"task_id": task_id, "prompt": prompt, "rubric_json": rubric, "rubric_pretty": pretty,
                 "sector": row.get("sector", ""), "occupation": row.get("occupation", ""),
                 "deliverables_dir": str(destination / "agent" / "deliverables")}
        (temporary / "judge.json").write_text(json.dumps(judge, indent=2) + "\n", encoding="utf-8")
        provenance = {"task_id": task_id, "dataset_identity": dataset_identity,
                      "source_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                      "inputs": manifest, "status": "staged_not_executed"}
        (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        # Unique destinations are required; never overwrite an earlier rollout.
        destination.mkdir(mode=0o700)
        try:
            for child in temporary.iterdir():
                child.rename(destination / child.name)
            (destination / "READY").write_text("staged_not_executed\n", encoding="utf-8")
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(temporary)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset-identity", required=True, help="Dataset version and split")
    args = parser.parse_args()
    matches = []
    with args.dataset.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("task_id") == args.task_id:
                    matches.append(row)
    if len(matches) != 1:
        parser.error(f"Expected exactly one matching task, found {len(matches)}")
    print(stage(matches[0], args.references, args.destination, args.dataset_identity))


if __name__ == "__main__":
    main()
