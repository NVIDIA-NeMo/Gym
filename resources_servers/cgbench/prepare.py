# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared preparation logic for all six CG-Bench (full / mini / hard x single_turn / agentic) benchmarks."""

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Literal

from resources_servers.cgbench.dataset import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SOURCE,
    SLICES,
    CGBenchBenchmark,
    CGBenchCohort,
    CGBenchSlice,
    load_cgbench,
    normalize_video_uid,
)
from resources_servers.cgbench.scoring import (
    CGBENCH_POLICIES,
    CGBENCH_RULE_POLICY,
    CGBenchAnswerPolicy,
    prompt_text,
)
from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.task import Setting, prepare_agentic, prepare_single_turn


CGBenchPromptStyle = Literal["official", "boxed"]
OFFICIAL_VIDEO_SUBDIR = "cg_videos_720p"


def _safe_sandbox_path(path: str | None) -> str:
    if path is None:
        raise ValueError("Agentic projection requires an explicit source and safe sandbox path")
    parsed = PurePosixPath(path)
    if not parsed.parts or parsed.is_absolute() or ".." in parsed.parts or any(char in path for char in "\r\n\\"):
        raise ValueError("Agentic projection requires an explicit source and safe sandbox path")
    return parsed.as_posix()


def _safe_sandbox_root(path: str) -> PurePosixPath:
    parsed = PurePosixPath(path)
    if not parsed.is_absolute() or ".." in parsed.parts or any(char in path for char in "\r\n\\"):
        raise ValueError("sandbox_root must be an absolute POSIX directory")
    return parsed


def prepare_cgbench_dataset(
    records: Iterable[Mapping[str, object]],
    *,
    benchmark: CGBenchBenchmark,
    dataset_revision: str = DATASET_REVISION,
    media: Mapping[str, VideoSource],
    setting: Setting,
    answer_policy: CGBenchAnswerPolicy = CGBENCH_RULE_POLICY,
    prompt: CGBenchPromptStyle = "official",
    num_repeats: int = 1,
    sandbox_root: str = "/testbed",
) -> tuple[list[dict[str, object]], CGBenchCohort]:
    """Build evaluation-only CG-Bench long_acc rows and their frozen cohort manifest."""
    if benchmark not in SLICES or setting not in ("single_turn", "agentic"):
        raise ValueError("Unknown benchmark or setting")
    if answer_policy not in CGBENCH_POLICIES:
        raise ValueError(f"Unknown answer_policy {answer_policy!r}")
    if prompt not in ("official", "boxed"):
        raise ValueError(f"Unknown prompt style {prompt!r}")
    root = _safe_sandbox_root(sandbox_root) if setting == "agentic" else None
    record_list = list(records)
    items = load_cgbench(record_list, benchmark=benchmark, dataset_revision=dataset_revision, media=media)
    rows: list[dict[str, object]] = []
    for item in items:
        video_uid = item.question.video_id
        binding = media[video_uid]
        rendered_prompt = prompt_text(item.question.question, item.question.options) if prompt == "official" else None
        if setting == "single_turn":
            if binding.video_url is None:
                raise ValueError(f"Missing video_url for video {video_uid!r}")
            row: dict[str, object] = dict(
                prepare_single_turn(
                    item.question,
                    expected_answer=item.expected_answer,
                    video_url=binding.video_url,
                    prompt_text=rendered_prompt,
                )
            )
        else:
            if binding.source is None:
                raise ValueError("Agentic projection requires an explicit source and safe sandbox path")
            sandbox_path = _safe_sandbox_path(binding.sandbox_path)
            assert root is not None
            row = dict(
                prepare_agentic(
                    item.question,
                    expected_answer=item.expected_answer,
                    video_path=str(root / sandbox_path),
                    prompt_text=rendered_prompt,
                )
            )
            row["video_binding"] = {
                "source": binding.source,
                "sandbox_path": sandbox_path,
                "sha256": binding.sha256,
            }
        if item.annotation_warnings:
            row["annotation_warnings"] = list(item.annotation_warnings)
        rows.append(row)
    cohort = CGBenchCohort(
        benchmark=benchmark,
        dataset_source=DATASET_SOURCE,
        dataset_revision=dataset_revision,
        setting=setting,
        answer_policy=answer_policy,
        task_ids=tuple(item.question.task_id for item in items),
        task_subsets={item.question.task_id: item.subsets for item in items},
        num_repeats=num_repeats,
    )
    return rows, cohort


def file_sha256(path: Path) -> str:
    """Stream SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    """Parse a ``FILES.sha256`` file (``<sha256>  [./]<path>``)."""
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        digest, name = stripped.split(maxsplit=1)
        checksums[name.removeprefix("./")] = digest
    return checksums


def load_annotations_file(path: Path, *, slice_spec: CGBenchSlice) -> list[dict[str, object]]:
    """Read a CG-Bench slice JSON after checking its pinned SHA-256 and question count."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != slice_spec.json_sha256:
        raise ValueError(
            f"{path} has SHA-256 {actual_sha256}, expected {slice_spec.json_sha256} "
            f"({DATASET_REPO_ID}@{DATASET_REVISION}:{slice_spec.json_filename})"
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list of CG-Bench question records")
    if len(records) != slice_spec.num_questions:
        raise ValueError(f"{path} has {len(records)} questions, expected {slice_spec.num_questions}")
    return records


def _resolve_annotations_path(
    slice_spec: CGBenchSlice,
    *,
    source_dir: Path,
    annotations_path: str | None,
    download: bool,
) -> Path:
    if annotations_path:
        return Path(annotations_path)
    env_override = os.environ.get(slice_spec.json_env_var)
    if env_override:
        return Path(env_override)
    local_path = source_dir / slice_spec.json_filename
    if local_path.is_file():
        return local_path
    if not download:
        raise ValueError(
            f"Missing {local_path}. Accept the gated terms at {DATASET_SOURCE}, set HF_TOKEN, and download "
            f"{slice_spec.json_filename} at revision {DATASET_REVISION} into {source_dir} "
            f"(or set {slice_spec.json_env_var} / pass download=True)."
        )
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                DATASET_REPO_ID,
                slice_spec.json_filename,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=str(source_dir),
                token=os.environ.get("HF_TOKEN") or True,
            )
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to download {slice_spec.json_filename} from gated dataset {DATASET_REPO_ID}@{DATASET_REVISION}. "
            f"Accept the dataset terms on Hugging Face ({DATASET_SOURCE}) and set HF_TOKEN (or run "
            f"`huggingface-cli login`)."
        ) from exc


def _format_video_url(video_url_dir: str, filename: str) -> str:
    if video_url_dir.startswith(("http://", "https://", "file://")):
        return f"{video_url_dir.rstrip('/')}/{filename}"
    path = Path(video_url_dir)
    resolved = path if path.is_absolute() else path.resolve()
    return (resolved / filename).as_uri()


def _lookup_video_checksum(checksums: Mapping[str, str], video_uid: str) -> str | None:
    filename = f"{video_uid}.mp4"
    for candidate in (f"{OFFICIAL_VIDEO_SUBDIR}/{filename}", f"videos/{filename}", filename):
        digest = checksums.get(candidate)
        if digest is not None:
            return digest
    return None


def resolve_video_checksums(
    video_uids: list[str],
    *,
    source_dir: Path,
    video_dir: Path,
    workers: int | None = None,
) -> dict[str, str]:
    """Return ``{video_uid: sha256}`` from ``FILES.sha256`` or by hashing local ``<video_dir>/<uid>.mp4`` files."""
    checksum_file = source_dir / "FILES.sha256"
    if checksum_file.is_file():
        checksums = read_checksums(checksum_file)
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for uid in video_uids:
            digest = _lookup_video_checksum(checksums, uid)
            if digest is None:
                missing.append(uid)
            else:
                resolved[uid] = digest
        if missing:
            raise ValueError(f"{checksum_file} has no SHA-256 for {OFFICIAL_VIDEO_SUBDIR}/{missing[0]}.mp4")
        return resolved

    missing_files = [uid for uid in video_uids if not (video_dir / f"{uid}.mp4").is_file()]
    if missing_files:
        raise ValueError(
            f"Missing {len(missing_files)} video(s) under {video_dir} (e.g. {missing_files[0]}.mp4). "
            f"Accept the gated terms at {DATASET_SOURCE}, set HF_TOKEN, download video_chunk_*.zip from "
            f"{DATASET_REPO_ID}@{DATASET_REVISION}, and unpack them into {OFFICIAL_VIDEO_SUBDIR}/."
        )

    max_workers = workers or min(8, os.cpu_count() or 1)

    def hash_one(uid: str) -> tuple[str, str]:
        return uid, file_sha256(video_dir / f"{uid}.mp4")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(hash_one, video_uids))


def prepare_cgbench_benchmark(
    *,
    benchmark: CGBenchBenchmark,
    setting: Setting,
    output_fpath: Path,
    cohort_fpath: Path,
    source_dir: str | None = None,
    video_dir: str | None = None,
    video_url_dir: str | None = None,
    annotations_path: str | None = None,
    answer_policy: CGBenchAnswerPolicy = CGBENCH_RULE_POLICY,
    prompt: CGBenchPromptStyle = "official",
    num_repeats: int = 1,
    sandbox_root: str = "/testbed",
    download: bool = False,
) -> Path:
    """Materialize benchmark JSONL rows and frozen cohort JSON for one CG-Bench benchmark entry."""
    if source_dir is None:
        raise ValueError(
            f"Set source_dir to the directory holding {SLICES[benchmark].json_filename} and "
            f"{OFFICIAL_VIDEO_SUBDIR}/ from {DATASET_REPO_ID}@{DATASET_REVISION} "
            f"(gated: auto on Hugging Face; accept terms and set HF_TOKEN)."
        )
    slice_spec = SLICES[benchmark]
    source = Path(source_dir).resolve()
    videos = (Path(video_dir) if video_dir is not None else source / OFFICIAL_VIDEO_SUBDIR).resolve()
    json_path = _resolve_annotations_path(
        slice_spec, source_dir=source, annotations_path=annotations_path, download=download
    )
    records = load_annotations_file(json_path, slice_spec=slice_spec)
    unique_uids = list(dict.fromkeys(normalize_video_uid(record.get("video_uid")) for record in records))
    checksums = resolve_video_checksums(unique_uids, source_dir=source, video_dir=videos)

    media: dict[str, VideoSource] = {}
    url_base = video_url_dir if video_url_dir is not None else str(videos)
    for uid in unique_uids:
        filename = f"{uid}.mp4"
        source_rel = f"{OFFICIAL_VIDEO_SUBDIR}/{filename}"
        sha256 = checksums[uid]
        if setting == "single_turn":
            media[uid] = VideoSource(
                source_url=source_rel,
                sha256=sha256,
                video_url=_format_video_url(url_base, filename),
            )
        else:
            media[uid] = VideoSource(
                source_url=source_rel,
                sha256=sha256,
                source=str(videos / filename),
                sandbox_path=f"videos/{filename}",
            )

    rows, cohort = prepare_cgbench_dataset(
        records,
        benchmark=benchmark,
        dataset_revision=DATASET_REVISION,
        media=media,
        setting=setting,
        answer_policy=answer_policy,
        prompt=prompt,
        num_repeats=num_repeats,
        sandbox_root=sandbox_root,
    )
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cohort_fpath.parent.mkdir(parents=True, exist_ok=True)
    cohort_fpath.write_text(cohort.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return output_fpath
