#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert official CVDP agentic-heavy benchmark rows to NeMo-Gym JSONL.

The public CVDP release is hosted on Hugging Face as
``nvidia/cvdp-benchmark-dataset``. Agentic-heavy rows use this shape:

    {
      "id": "cvdp_agentic_heavy_<repo>_<n>",
      "categories": ["cid016", "hard"],
      "prompt": "...",
      "system_message": "...",
      "context": {"repo": "cvdp_agentic_heavy_<repo>", "commit": "<sanitized sha>"},
      "harness": {"docker-compose.yml": "...", ...},
      "patch": {...},
      "origin": {...}
    }

The corresponding public source contexts are Git bundles under the dataset
folder ``cvdp_v1.1.0_agentic_heavy_code_generation_public``. This script can
use a local bundle directory, or download bundles from Hugging Face on demand.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from typing import Iterable, Optional


DEFAULT_HF_DATASET = "nvidia/cvdp-benchmark-dataset"
DEFAULT_HF_REVISION = "main"
DEFAULT_HF_INPUT = "cvdp_v1.1.0_agentic_heavy_code_generation.jsonl"
DEFAULT_PUBLIC_CONTEXT_DIR = "cvdp_v1.1.0_agentic_heavy_code_generation_public"


AGENTIC_SYSTEM_PROMPT = """\
You are an expert hardware design engineer working in a sandbox at /code.

TOOLS - these are the ONLY tools. Any other call, such as bash, find, sed, or grep, will fail:
  ls(path)                           - list a directory
  cat(filename)                      - read a file
  echo(content, filename)            - write or overwrite a whole file
  edit(filename, old_text, new_text) - replace one unique span in an existing file
  iverilog(args)                     - compile Verilog/SystemVerilog with Icarus Verilog
  vvp(filename)                      - run a compiled simulation
  pwd()                              - print the current working directory

GRADING: a hidden harness checks files in /code after you finish. Text-only answers are ignored.
You must write your solution with echo or edit.

Recommended flow:
1. Use ls to discover the project layout. Do not guess paths.
2. Use cat to inspect the task-relevant source, spec, and testbench files.
3. Write the change with echo for new files or edit for existing files.
4. Compile with iverilog. If compilation fails, fix the first actionable error and compile again.
5. Run vvp after every successful compile. Passing compilation alone is not enough.
6. Iterate until the visible simulation passes or there is no actionable local failure left.

Rules:
- Do not modify hidden harness files.
- Do not include cocotb or pytest files in iverilog commands.
- Prefer edit over echo when changing an existing multi-line file.
- Use exact paths discovered from ls/cat output.
"""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "ls",
        "description": "List files and directories at the given path.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path to list."}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "cat",
        "description": "Read and return the contents of a file.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string", "description": "Path to the file to read."}},
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "echo",
        "description": "Write content to a file, creating it if needed. Overwrites the full file if it exists.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to write."},
                "filename": {"type": "string", "description": "Path to write."},
            },
            "required": ["content", "filename"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "edit",
        "description": (
            "Replace one unique span in an existing file. Prefer this over echo for modifying "
            "existing RTL or testbench files. old_text must appear exactly once."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Existing file to modify."},
                "old_text": {"type": "string", "description": "Exact unique text currently in the file."},
                "new_text": {"type": "string", "description": "Replacement text."},
            },
            "required": ["filename", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "iverilog",
        "description": "Compile Verilog/SystemVerilog files with Icarus Verilog.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"args": {"type": "string", "description": "Arguments to pass to iverilog."}},
            "required": ["args"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "vvp",
        "description": "Run a compiled Icarus Verilog simulation.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string", "description": "Compiled simulation file to run."}},
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "pwd",
        "description": "Return the current working directory path.",
        "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
]


VERILOG_EXT_SWAPS = {".sv": ".v", ".v": ".sv", ".svh": ".vh", ".vh": ".svh"}
TB_PATTERNS = re.compile(
    r"(^|/)(tb_\w+\.(v|sv|vh|svh)|\w+_tb\.(v|sv|vh|svh)|test_\w+\.(v|sv|py)|\w+_test\.(v|sv|py))$",
    re.IGNORECASE,
)


def _is_git_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "git@", "ssh://")) or value.endswith(".git")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "repo"


def _hf_resolve_url(dataset: str, revision: str, path: str) -> str:
    quoted_path = urllib.parse.quote(path, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/datasets/{dataset}/resolve/{quoted_revision}/{quoted_path}"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as fout:
        shutil.copyfileobj(response, fout)
    tmp.replace(destination)


def _bundle_path_for_repo(
    repo_name: str,
    bundle_dir: Optional[Path],
    *,
    download_bundles: bool,
    hf_dataset: str,
    hf_revision: str,
    public_context_dir: str,
) -> Path:
    if bundle_dir is None:
        raise ValueError(
            f"context.repo={repo_name!r} is not a git URL. Provide --bundle-dir, or pass --download-bundles."
        )
    bundle = bundle_dir / f"{repo_name}.bundle"
    if bundle.exists():
        return bundle
    if not download_bundles:
        raise FileNotFoundError(f"Missing bundle for {repo_name!r}: {bundle}")
    url = _hf_resolve_url(hf_dataset, hf_revision, f"{public_context_dir}/{repo_name}.bundle")
    _download(url, bundle)
    return bundle


def _has_checked_out_files(path: Path) -> bool:
    if not path.exists():
        return False
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != ".git"]
        if files:
            return True
    return False


def _clone_checkout(source: str | Path, commit: str, destination: Path) -> None:
    subprocess.run(["git", "clone", "--no-checkout", str(source), str(destination)], check=True)
    subprocess.run(["git", "checkout", commit], cwd=destination, check=True)


def _read_context_files(repo_dir: Path) -> dict[str, str]:
    has_external = (repo_dir / "external").is_dir()
    files: dict[str, str] = {}
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for filename in filenames:
            path = Path(root) / filename
            rel = path.relative_to(repo_dir).as_posix()
            if has_external and rel.startswith("external/"):
                rel = rel[len("external/"):]
            try:
                files[rel] = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
    return files


def resolve_context_files(
    context: dict,
    *,
    cache_dir: Path,
    bundle_dir: Optional[Path],
    download_bundles: bool,
    hf_dataset: str,
    hf_revision: str,
    public_context_dir: str,
) -> dict[str, str]:
    repo = str(context.get("repo") or "")
    commit = str(context.get("commit") or "")
    if not repo or not commit:
        raise ValueError("context must contain repo and commit")

    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = cache_dir / f"{_safe_name(repo)}_{commit[:12]}"
    if repo_dir.exists() and not _has_checked_out_files(repo_dir):
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        if _is_git_url(repo):
            source: str | Path = repo
        else:
            source = _bundle_path_for_repo(
                repo,
                bundle_dir,
                download_bundles=download_bundles,
                hf_dataset=hf_dataset,
                hf_revision=hf_revision,
                public_context_dir=public_context_dir,
            )
        print(f"  resolving {repo} @ {commit[:12]}")
        _clone_checkout(source, commit, repo_dir)
    else:
        print(f"  using cached context {repo_dir}")

    return _read_context_files(repo_dir)


def fix_prompt_paths(prompt: str, context_files: dict[str, str], task_id: str) -> str:
    context_set = set(context_files)

    def _fix_one(match: re.Match) -> str:
        full = match.group(0)
        rel = full.replace("/code/", "", 1)
        if rel in context_set:
            return full
        stem, ext = os.path.splitext(rel)
        swapped_ext = VERILOG_EXT_SWAPS.get(ext)
        if swapped_ext:
            candidate = stem + swapped_ext
            if candidate in context_set:
                fixed = "/code/" + candidate
                print(f"  [{task_id}] path fix: {full} -> {fixed}")
                return fixed
        for prefix in ["external/", "src/", "rtl/"]:
            candidate = prefix + rel
            if candidate in context_set:
                fixed = "/code/" + candidate
                print(f"  [{task_id}] path fix: {full} -> {fixed}")
                return fixed
            if swapped_ext:
                candidate2 = prefix + stem + swapped_ext
                if candidate2 in context_set:
                    fixed = "/code/" + candidate2
                    print(f"  [{task_id}] path fix: {full} -> {fixed}")
                    return fixed
        return full

    return re.sub(r"/code/[\w/.+\-]+\.\w+", _fix_one, prompt)


def find_testbenches(context_files: dict[str, str]) -> list[str]:
    return sorted(path for path in context_files if TB_PATTERNS.search(path))


def enrich_prompt_with_testbenches(prompt: str, testbenches: list[str], task_id: str) -> str:
    if not testbenches:
        return prompt
    shown = testbenches[:8]
    tb_list = "\n".join(f"  - {path}" for path in shown)
    more = f" and {len(testbenches) - len(shown)} more" if len(testbenches) > len(shown) else ""
    hint = (
        f"AVAILABLE TESTBENCHES in this project ({len(testbenches)} found{more}):\n{tb_list}\n\n"
        "Use relevant visible testbenches for local iverilog/vvp checks when possible.\n\n---\n\n"
    )
    print(f"  [{task_id}] found {len(testbenches)} visible testbench candidates")
    return hint + prompt



def _is_commercial_eda_service(service_name: str, service_config: dict) -> bool:
    image = str(service_config.get("image") or "")
    command = service_config.get("command") or ""
    entrypoint = service_config.get("entrypoint") or ""
    text = " ".join([
        service_name,
        image,
        command if isinstance(command, str) else " ".join(map(str, command)),
        entrypoint if isinstance(entrypoint, str) else " ".join(map(str, entrypoint)),
    ]).lower()
    return "__verif_eda_image__" in text or "xrun" in text or "xcelium" in text


def sanitize_harness_files(harness_files: dict[str, str], *, include_commercial_eda: bool, task_id: str) -> Optional[dict[str, str]]:
    if include_commercial_eda:
        return dict(harness_files)

    sanitized = dict(harness_files)
    compose_key = next((key for key in sanitized if key.endswith("docker-compose.yml")), "")
    if not compose_key:
        return sanitized

    try:
        compose_data = yaml.safe_load(sanitized[compose_key]) or {}
    except Exception as exc:
        print(f"  [{task_id}] skipping: could not parse docker-compose.yml: {exc}", file=sys.stderr)
        return None

    services = compose_data.get("services") or {}
    if not isinstance(services, dict):
        print(f"  [{task_id}] skipping: docker-compose.yml has no services", file=sys.stderr)
        return None

    removed = []
    for name, service in list(services.items()):
        if isinstance(service, dict) and _is_commercial_eda_service(name, service):
            removed.append(name)
            del services[name]
    if removed:
        print(f"  [{task_id}] removed commercial EDA service(s): {', '.join(removed)}")

    if not services:
        print(f"  [{task_id}] skipping: only commercial EDA harness services were available", file=sys.stderr)
        return None

    networks = compose_data.get("networks")
    if isinstance(networks, dict):
        for name in list(networks):
            net = networks[name]
            net_text = f"{name} {net}".lower()
            if "licnetwork" in net_text:
                del networks[name]
        if not networks:
            compose_data.pop("networks", None)

    sanitized[compose_key] = yaml.safe_dump(compose_data, sort_keys=False)
    return sanitized


def build_developer_prompt(system_message: str) -> str:
    system_message = (system_message or "").strip()
    if not system_message:
        return AGENTIC_SYSTEM_PROMPT
    return f"{AGENTIC_SYSTEM_PROMPT}\n\nAdditional benchmark instructions:\n{system_message}"


def convert_entry(entry: dict, args: argparse.Namespace) -> Optional[dict]:
    task_id = str(entry.get("id") or "")
    categories = list(entry.get("categories") or [])
    difficulty = str(entry.get("difficulty") or (categories[1] if len(categories) > 1 else ""))
    prompt = str(entry.get("prompt") or "")

    if args.skip_resolve:
        context_files = {"_unresolved_context.json": json.dumps(entry.get("context", {}), sort_keys=True)}
    else:
        try:
            context_files = resolve_context_files(
                dict(entry.get("context") or {}),
                cache_dir=Path(args.repos_cache),
                bundle_dir=Path(args.bundle_dir) if args.bundle_dir else None,
                download_bundles=args.download_bundles,
                hf_dataset=args.hf_dataset,
                hf_revision=args.hf_revision,
                public_context_dir=args.public_context_dir,
            )
        except Exception as exc:
            print(f"  skipping {task_id}: failed to resolve context: {exc}", file=sys.stderr)
            return None
        prompt = fix_prompt_paths(prompt, context_files, task_id)
        if not args.no_testbench_hints:
            prompt = enrich_prompt_with_testbenches(prompt, find_testbenches(context_files), task_id)

    harness_files = sanitize_harness_files(
        dict(entry.get("harness") or {}),
        include_commercial_eda=args.include_commercial_eda,
        task_id=task_id,
    )
    if harness_files is None:
        return None

    row = {
        "responses_create_params": {
            "input": [
                {"role": "developer", "content": build_developer_prompt(str(entry.get("system_message") or ""))},
                {"role": "user", "content": prompt},
            ],
            "tools": TOOL_DEFINITIONS,
        },
        "verifier_metadata": {
            "task_id": task_id,
            "categories": categories,
            "difficulty": difficulty,
            "context_files": context_files,
            "harness_files": harness_files,
            "origin": entry.get("origin") or {},
        },
    }
    if args.include_solution_metadata:
        row["verifier_metadata"]["patch"] = entry.get("patch") or {}
    return row


def _iter_lines(input_spec: str, *, hf_dataset: str, hf_revision: str) -> Iterable[str]:
    if input_spec.startswith("hf://"):
        relpath = input_spec[len("hf://"):]
        url = _hf_resolve_url(hf_dataset, hf_revision, relpath)
        with urllib.request.urlopen(url) as response:
            for raw in response:
                yield raw.decode("utf-8")
    elif input_spec.startswith(("http://", "https://")):
        with urllib.request.urlopen(input_spec) as response:
            for raw in response:
                yield raw.decode("utf-8")
    else:
        with open(input_spec, encoding="utf-8") as fin:
            yield from fin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=f"hf://{DEFAULT_HF_INPUT}",
        help="Input JSONL path, URL, or hf://relative/path. Defaults to the official v1.1.0 agentic-heavy JSONL.",
    )
    parser.add_argument("--output", required=True, help="Output NeMo-Gym JSONL path")
    parser.add_argument(
        "--repos-cache",
        default=os.environ.get("CVDP_REPOS_CACHE", ".cache/cvdp_repos_cache"),
        help="Directory for checked-out context repositories. Defaults to CVDP_REPOS_CACHE or .cache/cvdp_repos_cache.",
    )
    parser.add_argument("--bundle-dir", default="", help="Directory containing official CVDP public .bundle files")
    parser.add_argument("--download-bundles", action="store_true", help="Download missing public bundles from Hugging Face")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET, help="Hugging Face dataset id")
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION, help="Hugging Face dataset revision")
    parser.add_argument("--public-context-dir", default=DEFAULT_PUBLIC_CONTEXT_DIR, help="HF subdirectory containing public bundles")
    parser.add_argument("--skip-resolve", action="store_true", help="Do not resolve context repositories; emit unresolved context metadata")
    parser.add_argument("--no-testbench-hints", action="store_true", help="Do not prepend visible-testbench hints to prompts")
    parser.add_argument(
        "--include-commercial-eda",
        action="store_true",
        help="Keep harness services that require commercial EDA/Xcelium. By default these services are removed, and EDA-only rows are skipped.",
    )
    parser.add_argument(
        "--include-solution-metadata",
        action="store_true",
        help="Include official patch/reference metadata in verifier_metadata. Disabled by default to reduce leakage risk.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Convert at most this many rows")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.download_bundles and not args.bundle_dir:
        args.bundle_dir = str(Path(args.repos_cache) / "bundles")

    written = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(_iter_lines(args.input, hf_dataset=args.hf_dataset, hf_revision=args.hf_revision), 1):
            if args.limit and written >= args.limit:
                break
            if not line.strip():
                continue
            entry = json.loads(line)
            task_id = entry.get("id", f"line_{line_num}")
            print(f"[{line_num}] processing {task_id}")
            row = convert_entry(entry, args)
            if row is None:
                skipped += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} rows to {output}")
    if skipped:
        print(f"Skipped {skipped} rows")


if __name__ == "__main__":
    main()
