# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Video-Index v1.0 / v1.1 single-turn and agentic JSONL rows and frozen cohorts."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Literal

from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.task import Setting, prepare_agentic, prepare_single_turn
from resources_servers.videoindex.dataset import (
    RELEASES,
    VideoIndexBenchmark,
    VideoIndexCohort,
    VideoIndexRelease,
    load_videoindex,
    normalize_video_relpath,
)
from resources_servers.videoindex.scoring import (
    VIDEOINDEX_POLICIES,
    VIDEOINDEX_RULE_POLICY,
    VideoIndexAnswerPolicy,
    prompt_text,
)


VideoIndexPromptStyle = Literal["official", "boxed"]


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


def prepare_videoindex_dataset(
    records: Iterable[Mapping[str, object]],
    *,
    benchmark: VideoIndexBenchmark,
    dataset_revision: str,
    media: Mapping[str, VideoSource],
    setting: Setting,
    answer_policy: VideoIndexAnswerPolicy = VIDEOINDEX_RULE_POLICY,
    prompt: VideoIndexPromptStyle = "official",
    num_repeats: int = 1,
    sandbox_root: str = "/testbed",
) -> tuple[list[dict[str, object]], VideoIndexCohort]:
    """Build evaluation-only Video-Index rows and their frozen cohort manifest."""
    if benchmark not in RELEASES or setting not in ("single_turn", "agentic"):
        raise ValueError("Unknown benchmark or setting")
    if answer_policy not in VIDEOINDEX_POLICIES:
        raise ValueError(f"Unknown answer_policy {answer_policy!r}")
    if prompt not in ("official", "boxed"):
        raise ValueError(f"Unknown prompt style {prompt!r}")
    root = _safe_sandbox_root(sandbox_root) if setting == "agentic" else None
    record_list = list(records)
    items = load_videoindex(record_list, benchmark=benchmark, dataset_revision=dataset_revision, media=media)
    rows: list[dict[str, object]] = []
    for record, item in zip(record_list, items):
        video_rel = normalize_video_relpath(record.get("video"))
        binding = media[video_rel]
        rendered_prompt = prompt_text(item.question.question, item.question.options) if prompt == "official" else None
        if setting == "single_turn":
            if binding.video_url is None:
                raise ValueError(f"Missing video_url for {video_rel!r}")
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
    cohort = VideoIndexCohort(
        benchmark=benchmark,
        dataset_source=RELEASES[benchmark].dataset_source,
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


def load_items_file(path: Path, *, release: VideoIndexRelease) -> list[dict[str, object]]:
    """Read ``items/test.jsonl`` after checking its pinned SHA-256 and question count."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != release.items_sha256:
        raise ValueError(
            f"{path} has SHA-256 {actual_sha256}, expected {release.items_sha256} "
            f"({release.repo_id}@{release.dataset_revision})"
        )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != release.num_questions:
        raise ValueError(f"{path} has {len(records)} questions, expected {release.num_questions}")
    return records


def _resolve_items_path(release: VideoIndexRelease, *, source_dir: Path, download: bool) -> Path:
    env_override = os.environ.get(release.items_env_var)
    if env_override:
        return Path(env_override)
    local_path = source_dir / release.items_filename
    if local_path.is_file():
        return local_path
    if not download:
        raise ValueError(
            f"Missing {local_path}. Download {release.repo_id}@{release.dataset_revision} into {source_dir} "
            f"(or set {release.items_env_var} / pass download=True)."
        )
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            release.repo_id,
            release.items_filename,
            repo_type="dataset",
            revision=release.dataset_revision,
            local_dir=str(source_dir),
        )
    )


def _hub_lfs_metadata(release: VideoIndexRelease, paths: list[str]) -> dict[str, tuple[int, str]]:
    """Fetch ``{path: (size, sha256)}`` for video LFS objects at the pinned revision."""
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(
        release.repo_id, paths, revision=release.dataset_revision, repo_type="dataset", expand=True
    )
    found = {info.path: info for info in infos}
    missing = [path for path in paths if path not in found or found[path].lfs is None]
    if missing:
        raise ValueError(
            f"{release.repo_id}@{release.dataset_revision}: {len(missing)} videos are not LFS files, e.g. {missing[0]}"
        )
    return {path: (found[path].size, found[path].lfs.sha256) for path in paths}


def _download_video(release: VideoIndexRelease, path: str, *, root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            release.repo_id,
            path,
            repo_type="dataset",
            revision=release.dataset_revision,
            local_dir=str(root),
            force_download=True,
        )
    )


def _format_video_url(video_url_dir: str, filename: str) -> str:
    if video_url_dir.startswith(("http://", "https://", "file://")):
        return f"{video_url_dir.rstrip('/')}/{filename}"
    path = Path(video_url_dir)
    base = path if path.is_absolute() else path.resolve()
    return (base / filename).as_uri()


def _materialize_dest(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out_handle, src.open("rb") as in_handle:
            shutil.copyfileobj(in_handle, out_handle)
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def resolve_video_checksums(
    video_paths: list[str],
    *,
    release: VideoIndexRelease,
    source_dir: Path,
    video_dir: Path,
    download: bool = False,
    workers: int | None = None,
) -> dict[str, str]:
    """Return ``{video_rel: sha256}`` from ``FILES.sha256`` (offline only), local files, or HF LFS download."""
    checksum_file = source_dir / "FILES.sha256"
    if not download and checksum_file.is_file():
        checksums = read_checksums(checksum_file)
        missing = [rel for rel in video_paths if rel not in checksums]
        if missing:
            raise ValueError(f"{checksum_file} has no SHA-256 for {missing[0]}")
        return {rel: checksums[rel] for rel in video_paths}

    max_workers = workers or min(8, os.cpu_count() or 1)
    if download:
        hub_meta = _hub_lfs_metadata(release, video_paths)
        video_dir.mkdir(parents=True, exist_ok=True)

        def fetch_one(rel: str) -> tuple[str, str]:
            expected_size, expected_sha = hub_meta[rel]
            dest = video_dir / PurePosixPath(rel).name
            if dest.is_file() and dest.stat().st_size == expected_size and file_sha256(dest) == expected_sha:
                return rel, expected_sha
            staged = source_dir / rel
            if (
                staged.resolve() != dest.resolve()
                and staged.is_file()
                and staged.stat().st_size == expected_size
                and file_sha256(staged) == expected_sha
            ):
                downloaded = staged
            else:
                downloaded = _download_video(release, rel, root=source_dir)
            if not downloaded.is_file() or downloaded.stat().st_size != expected_size:
                raise ValueError(f"Downloaded {downloaded} does not match expected size {expected_size}")
            actual_sha = file_sha256(downloaded)
            if actual_sha != expected_sha:
                raise ValueError(f"Downloaded {downloaded} has SHA-256 {actual_sha}, expected {expected_sha}")
            _materialize_dest(downloaded, dest)
            if not dest.is_file() or dest.stat().st_size != expected_size or file_sha256(dest) != expected_sha:
                raise ValueError(f"Materialized {dest} does not match expected size/SHA-256")
            return rel, expected_sha

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return dict(pool.map(fetch_one, video_paths))

    missing_files = [rel for rel in video_paths if not (video_dir / PurePosixPath(rel).name).is_file()]
    if missing_files:
        raise ValueError(
            f"Missing {len(missing_files)} video(s) under {video_dir} (e.g. {missing_files[0]}). "
            f"Download {release.repo_id}@{release.dataset_revision} with "
            f"`hf download {release.repo_id} --repo-type dataset --revision {release.dataset_revision} "
            f"--local-dir {source_dir}` or pass download=True."
        )

    def hash_one(rel: str) -> tuple[str, str]:
        return rel, file_sha256(video_dir / PurePosixPath(rel).name)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(hash_one, video_paths))


def prepare_videoindex_benchmark(
    *,
    benchmark: VideoIndexBenchmark,
    setting: Setting,
    output_fpath: Path,
    cohort_fpath: Path,
    source_dir: str | None = None,
    video_dir: str | None = None,
    video_url_dir: str | None = None,
    answer_policy: VideoIndexAnswerPolicy = VIDEOINDEX_RULE_POLICY,
    prompt: VideoIndexPromptStyle = "official",
    num_repeats: int = 1,
    sandbox_root: str = "/testbed",
    download: bool = False,
) -> Path:
    """Materialize benchmark JSONL rows and frozen cohort JSON for one Video-Index benchmark entry."""
    if source_dir is None:
        raise ValueError(
            f"Set source_dir to the directory holding Video-Index ({RELEASES[benchmark].repo_id}@"
            f"{RELEASES[benchmark].dataset_revision}) items/test.jsonl and videos/"
        )
    if type(num_repeats) is not int or num_repeats < 1:
        raise ValueError("num_repeats must be a positive integer")
    release = RELEASES[benchmark]
    source = Path(source_dir).resolve()
    videos = (Path(video_dir) if video_dir is not None else source / "videos").resolve()
    items_path = _resolve_items_path(release, source_dir=source, download=download)
    records = load_items_file(items_path, release=release)
    unique_videos = list(dict.fromkeys(normalize_video_relpath(record.get("video")) for record in records))
    checksums = resolve_video_checksums(
        unique_videos,
        release=release,
        source_dir=source,
        video_dir=videos,
        download=download,
    )

    media: dict[str, VideoSource] = {}
    url_base = video_url_dir if video_url_dir is not None else str(videos)
    for rel in unique_videos:
        filename = PurePosixPath(rel).name
        sha256 = checksums[rel]
        if setting == "single_turn":
            media[rel] = VideoSource(
                source_url=rel,
                sha256=sha256,
                video_url=_format_video_url(url_base, filename),
            )
        else:
            media[rel] = VideoSource(
                source_url=rel,
                sha256=sha256,
                source=str(videos / filename),
                sandbox_path=rel,
            )

    rows, cohort = prepare_videoindex_dataset(
        records,
        benchmark=benchmark,
        dataset_revision=release.dataset_revision,
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
