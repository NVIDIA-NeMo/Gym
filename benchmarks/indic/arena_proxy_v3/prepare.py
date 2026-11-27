# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join chat-400 conversations to their frozen GPT-5.4-mini reference answers.

The adapter preserves every input message for one next-assistant-turn generation.
Task fields remain at the top level, following the existing ``arena_judge`` schema.
No original LMArena baseline or held-out answer is copied into the prepared input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.config import DownloadJsonlDatasetGitlabConfig, GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.data import download_jsonl_dataset


DIRECTORY = Path(__file__).resolve().parent
DATASET_NAME = "lmarena_v3_indic"
DATASET_VERSION = "0.0.1"
SOURCE_CHECKSUMS = json.loads((DIRECTORY / "source-checksums.json").read_text(encoding="utf-8"))["files"]
BASELINE = "gpt-5.4-mini"
LANGUAGES = ("en", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur")
INDIC_LANGUAGES = LANGUAGES[1:]
EXPECTED_ROWS = 400
GROUPS = {
    "Information & advice": "info_advice",
    "Work writing & text tasks": "work_writing",
    "Creative writing / role-play": "creative_roleplay",
    "Planning & brainstorming": "planning",
    "Coding / technical": "coding",
    "Math / reasoning": "math",
    "Casual chat / unclear asks": "casual",
}
CREATIVE_GROUP = "Creative writing / role-play"


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context}: expected an object with string keys")
    return value


def _text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected nonempty text")
    return value


def _index_rows(rows: Sequence[Mapping[str, object]], *, context: str) -> dict[str, Mapping[str, object]]:
    indexed = {}
    for row in rows:
        question_id = _text(row.get("question_id"), context=f"{context} question_id")
        if question_id in indexed:
            raise ValueError(f"{context}: duplicate question_id {question_id}")
        indexed[question_id] = row
    return indexed


def _read_rows(path: Path) -> list[Mapping[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                rows.append(_mapping(json.loads(line), context=f"{path}:{line_number}"))
    return rows


def messages_sha256(messages: object) -> str:
    """Return the source reference's 16-character sorted-JSON message digest."""
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def convert_language(
    source_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    *,
    language: str,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[list[dict[str, object]], list[str]]:
    """Validate and join one language; return prepared rows and excluded translation IDs.

    All source/reference identities must match, including excluded translations.
    The successful reference must describe the exact conversation and baseline.
    ``hard_error`` translations are recorded and excluded; ``low_score`` is retained,
    matching the supplied benchmark protocol.
    """
    if language not in LANGUAGES:
        raise ValueError(f"Unknown language {language!r}; choose from {LANGUAGES}")
    sources = _index_rows(source_rows, context=f"{language} prompts")
    references = _index_rows(reference_rows, context=f"{language} references")
    if len(sources) != expected_rows:
        raise ValueError(f"{language}: expected {expected_rows} source rows, found {len(sources)}")
    if sources.keys() != references.keys():
        missing = sorted(sources.keys() - references.keys())
        extra = sorted(references.keys() - sources.keys())
        raise ValueError(f"{language}: reference IDs differ; missing={missing}, extra={extra}")

    prepared, excluded = [], []
    for question_id, source in sources.items():
        context = f"{language}:{question_id}"
        metadata = _mapping(source.get("metadata"), context=f"{context} metadata")
        chat = _mapping(metadata.get("chat400"), context=f"{context} metadata.chat400")
        translation = _mapping(chat.get("translation") or {}, context=f"{context} translation")
        translation_status = _text(
            translation.get("status", "original" if language == "en" else None),
            context=f"{context} translation status",
        )
        if translation_status not in {"original", "ok", "low_score", "hard_error"}:
            raise ValueError(f"{context}: unknown translation status {translation_status!r}")
        if language != "en" and translation.get("lang") != language:
            raise ValueError(f"{context}: translation language does not match the source file")

        params = _mapping(source.get("responses_create_params"), context=f"{context} responses_create_params")
        messages = params.get("input")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{context}: conversation input must be a nonempty message list")
        for index, raw_message in enumerate(messages):
            message = _mapping(raw_message, context=f"{context} message {index}")
            expected_role = "user" if index % 2 == 0 else "assistant"
            if message.get("role") != expected_role:
                raise ValueError(f"{context}: message {index} must have role {expected_role}")
            _text(message.get("content"), context=f"{context} message {index} content")
        if len(messages) % 2 == 0:
            raise ValueError(f"{context}: conversation must end with a user message")
        num_user_turns = chat.get("num_user_turns")
        if type(num_user_turns) is not int or num_user_turns != (len(messages) + 1) // 2:
            raise ValueError(f"{context}: num_user_turns does not match the conversation")
        group = _text(chat.get("group"), context=f"{context} group")
        if group not in GROUPS:
            raise ValueError(f"{context}: unknown chat-400 group {group!r}")
        question = _text(source.get("question"), context=f"{context} question")
        flattened = (
            messages[0]["content"]
            if len(messages) == 1
            else "\n\n".join(f"[{message['role'].capitalize()}]: {message['content']}" for message in messages)
        )
        if question != flattened:
            raise ValueError(f"{context}: question does not match the full conversation")

        reference = references[question_id]
        if reference.get("row_id") != question_id or reference.get("language") != language:
            raise ValueError(f"{context}: reference row_id or language does not match")
        if reference.get("question") != question or reference.get("messages_sha256") != messages_sha256(messages):
            raise ValueError(f"{context}: reference conversation or messages_sha256 does not match")
        if reference.get("category") != group or reference.get("num_user_turns") != num_user_turns:
            raise ValueError(f"{context}: reference group or num_user_turns does not match")
        if reference.get("status") != "ok":
            raise ValueError(f"{context}: reference status must be ok")
        model = _text(reference.get("model"), context=f"{context} reference model")
        if model.rsplit("/", 1)[-1] != BASELINE:
            raise ValueError(f"{context}: expected {BASELINE} reference, found {model}")
        baseline_answer = _text(reference.get("generation"), context=f"{context} reference generation")
        if translation_status == "hard_error":
            excluded.append(question_id)
            continue
        prepared.append(
            {
                "uid": context,
                "question_id": question_id,
                "row_id": question_id,
                "language": language,
                "group": group,
                "num_user_turns": num_user_turns,
                "category": "creative_writing" if group == CREATIVE_GROUP else "hard_prompt",
                "question": question,
                "baseline_answer": baseline_answer,
                "baseline_model": BASELINE,
                "messages_sha256": reference["messages_sha256"],
                "translation_status": translation_status,
                "responses_create_params": deepcopy(dict(params)),
            }
        )
    return prepared, excluded


def _source_paths(*, source_dir: Path | None, download_dir: Path, language: str) -> tuple[Path, Path]:
    filenames = (f"lmarena_v3_chat400_{language}.jsonl", f"references/{BASELINE}/{language}.jsonl")
    if source_dir is not None:
        return tuple(source_dir / filename for filename in filenames)
    paths = []
    for filename in filenames:
        path = download_dir / filename
        expected = SOURCE_CHECKSUMS[filename]["sha256"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Keep the published files separate and never cache an HTTP error body
            # or a changed artifact as a valid source file.
            with TemporaryDirectory(prefix=".download-", dir=path.parent) as temporary:
                pending = Path(temporary) / path.name
                download_jsonl_dataset(
                    DownloadJsonlDatasetGitlabConfig(
                        dataset_name=DATASET_NAME,
                        version=DATASET_VERSION,
                        artifact_fpath=filename,
                        output_fpath=str(pending),
                    )
                )
                if hashlib.sha256(pending.read_bytes()).hexdigest() != expected:
                    raise ValueError(
                        f"{filename}: SHA-256 does not match the frozen benchmark source; "
                        f"check access to registry dataset {DATASET_NAME} version {DATASET_VERSION}"
                    )
                pending.replace(path)
        paths.append(path)
    return paths[0], paths[1]


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(
    *,
    source_dir: Path | str | None = None,
    languages: Sequence[str] = INDIC_LANGUAGES,
    output_dir: Path | str = DIRECTORY / "data",
    limit: int | None = None,
) -> Path:
    """Prepare validated combined/per-language JSONL and an auditable exclusion report.

    Download separate prompts and references from the GitLab registry using Gym's
    configured registry access, just like ``lmarena_v3``. The default population
    contains the 12 Indic languages. English requires an explicit local source.
    Only the generated Gym input joins conversations and frozen reference answers;
    the downloaded source files retain their original contents and directory layout.
    """
    selected = list(languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"languages must be unique nonempty selections from {LANGUAGES}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive number of source rows per language")
    if source_dir is None and "en" in selected:
        raise ValueError("The registry contains only the 12 Indic languages; use --source-dir for English")
    source_dir = Path(source_dir) if source_dir is not None else None
    output_dir = Path(output_dir)
    report: dict[str, object] = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "source_mode": "gitlab" if source_dir is None else "local",
        "baseline": BASELINE,
        "languages": selected,
        "limit_per_language": limit,
        "coverage": {},
        "files_sha256": {},
    }
    by_language = {}
    expected_ids = None
    for language in selected:
        source_path, reference_path = _source_paths(
            source_dir=source_dir, download_dir=output_dir / "source", language=language
        )
        for filename, path in (
            (f"lmarena_v3_chat400_{language}.jsonl", source_path),
            (f"references/{BASELINE}/{language}.jsonl", reference_path),
        ):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != SOURCE_CHECKSUMS[filename]["sha256"]:
                raise ValueError(f"{filename}: SHA-256 does not match the frozen benchmark source")
            report["files_sha256"][filename] = digest
        sources, references = _read_rows(source_path), _read_rows(reference_path)
        rows, excluded = convert_language(sources, references, language=language, expected_rows=EXPECTED_ROWS)
        ids = {row["question_id"] for row in sources}
        if expected_ids is not None and ids != expected_ids:
            raise ValueError(f"{language}: source IDs differ across languages")
        expected_ids = ids
        # The reference selects its source prefix before excluding failed translations.
        selected_ids = {row["question_id"] for row in sources[:limit]}
        by_language[language] = [row for row in rows if row["question_id"] in selected_ids]
        report["coverage"][language] = {
            "source_rows": len(sources),
            "eligible_rows": len(rows),
            "prepared_rows": len(by_language[language]),
            "excluded_untranslated": excluded,
        }

    # Validate the entire selection before replacing any prepared artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for language, rows in by_language.items():
        _write_rows(output_dir / f"arena_proxy_v3_{language}.jsonl", rows)
        combined.extend(rows)
    output_path = output_dir / "arena_proxy_v3_benchmark.jsonl"
    _write_rows(output_path, combined)
    report["total_prepared_rows"] = len(combined)
    report["prepared_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    (output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(combined)} rows across {len(selected)} languages: {output_path}")
    return output_path


def main() -> None:
    """Prepare from the GitLab registry or a checksum-verified local source directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, help="Directory containing lmarena_v3_chat400_*.jsonl and references/"
    )
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=INDIC_LANGUAGES)
    parser.add_argument("--output-dir", type=Path, default=DIRECTORY / "data")
    parser.add_argument("--limit", type=int, help="First N source rows per language, then exclude failed translations")
    args = parser.parse_args()
    if args.source_dir is None:
        # argparse owns this command's arguments; Gym still owns registry settings
        # and env.yaml loading. `gym eval prepare` already initializes this config.
        get_global_config_dict(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                skip_load_from_cli=True,
                offline=True,
            )
        )
    prepare(
        source_dir=args.source_dir,
        languages=args.languages,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
