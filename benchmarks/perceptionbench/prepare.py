# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prompt construction adapted from:
#   Repository: https://github.com/MoonshotAI/PerceptionBench
#   Commit: ba032c06e9b6ee3679171ff6ba643b7a0cfebe2e
#   File: eval/eval.py (build_messages)
#   License: Apache-2.0
"""Download and prepare PerceptionBench (``moonshotai/PerceptionBench``) into NeMo Gym JSONL."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import struct
from pathlib import Path
from typing import Any

import orjson


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "perceptionbench_benchmark.jsonl"

SOURCE_ENV = "PERCEPTIONBENCH_DATA_ROOT"
HF_REPO_ID = "moonshotai/PerceptionBench"
HF_REVISION = "6ba8c3135c7675ad6a5c141536a86b9460c70960"  # pragma: allowlist secret
HF_FILENAME = "PerceptionBench.jsonl"
EXPECTED_SHA256 = "f84afb46c3c150b572481ca34b351d6afb1c31b98fef0c0c46d7659333573c47"  # pragma: allowlist secret
EXPECTED_TASK_COUNT = 3000

_IMAGE_PLACEHOLDER_RE = re.compile(r"<\|image_(\d+)\|>")
_DATA_URL_RE = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
_DIM_HINT_RE = re.compile(r"width\s*=\s*\d+\s*px\s*and\s*height\s*=\s*\d+\s*px", re.IGNORECASE)
_ROW_COL_RE = re.compile(r"\(\s*row\s*=\s*(\d+)\s*,\s*col(?:umn)?\s*=\s*(\d+)\s*\)", re.IGNORECASE)
_PIXEL_BRACKET_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
_PIXEL_PAREN_RE = re.compile(r"(pixel\s+coordinates?\s*)\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_PIXEL_ORIGIN_SENTENCE_RE = re.compile(
    r"(?:The origin of the pixel coordinate system[^.]*\.\s*"
    r"|The upper-left corner is the coordinate origin;[^.]*\.\s*"
    r"|The given coordinates use the pixel coordinate system\.\s*)",
    re.IGNORECASE,
)
_PIXEL_GIVEN_CLAUSE_RE = re.compile(
    r"Given that the coordinates use the pixel coordinate system,\s*([a-z])",
    re.IGNORECASE,
)
_PIXEL_TERM_RE = re.compile(
    r"(?:the\s+absolute\s+|absolute\s+)?pixel(?:\s+(?:coordinate\s+point|coordinates|coordinate|position|point))?\b",
    re.IGNORECASE,
)
_COORD_01 = r"(?:0?\.\d+|1\.0+|[01])"
_COORD_01_FULL_RE = re.compile(rf"^{_COORD_01}$")
_NORM_01_TUPLE_RE = re.compile(
    rf"([\[\(])\s*({_COORD_01})\s*,\s*({_COORD_01})(?:\s*,\s*({_COORD_01})\s*,\s*({_COORD_01}))?\s*([\]\)])"
)
_INTERVAL_PREFIX_RE = re.compile(
    r"(?:"
    r"\b(?:range|interval|scale)(?:\s+(?:of|over|is|in))?\s*:?"
    r"|\b(?:in|within|over)(?:\s+the)?"
    r"|\b(?:normalized|scaled|mapped)\s+to"
    r")\s*$",
    re.IGNORECASE,
)
_INTERVAL_SUFFIX_RE = re.compile(r"^\s*(?:range|interval|scale|normalized|coordinates?)\b", re.IGNORECASE)
_NORM_01_RANGE_RE = re.compile(r"(\(\s*0\s*[-–]\s*)1(\s*(?:range)?\s*\))", re.IGNORECASE)
_NORM_01_RANGE_TEXT_RE = re.compile(r"(\brange\s+0\s*[-–]\s*)1\b", re.IGNORECASE)
_NORM_01_INTERVAL_RE = re.compile(r"\[\s*0(?:\.0+)?\s*,\s*1(?:\.0+)?\s*\]")
_OPTION_LINE_RE = re.compile(r"(?:^|\n)[ \t]*[A-D][ \t]*[.:][ \t]*([^\n]+)")
_NUMERIC_OPTION_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_NORM_01_SCALAR_OPTION_RE = re.compile(
    rf"(^|\n)([ \t]*[A-D][ \t]*[.:][ \t]*)({_COORD_01})(?=[ \t]*\.?[ \t]*(?:\r?\n|$))"
)
_ALREADY_1000_RE = re.compile(r"0\s*(?:to|[-–])\s*1000|\[0,\s*1000\]", re.IGNORECASE)
_MUTUAL_EXCLUSION_MSG = (
    "include_hints and normalize_coords_1000 cannot be used together: "
    "upstream hints report raw pixel dimensions or [0, 1] coordinate formats "
    "that contradict [0, 1000] normalized coordinates."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _detect_mime_type(image_bytes: bytes, fallback: str = "image/jpeg") -> str:
    for signature, mime in _IMAGE_SIGNATURES:
        if image_bytes.startswith(signature):
            return mime
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def normalize_image_data_url(raw_image: str) -> str:
    """Ensure an image data URL carries the MIME type matching its decoded magic bytes.

    Some upstream PerceptionBench records label PNG bytes as ``data:image/jpeg;base64,...``;
    preserving the original bytes under their true MIME type avoids strict provider rejections.
    """
    stripped = raw_image.strip()
    match = _DATA_URL_RE.match(stripped)
    if match:
        declared_mime, encoded = match.group(1), match.group(2)
    else:
        declared_mime, encoded = "image/jpeg", stripped
    compact = "".join(encoded.split())
    # Decode only the leading 24 bytes (32 base64 chars) needed for magic-byte inspection.
    header_b64 = compact[:32]
    header_b64 += "=" * ((4 - len(header_b64) % 4) % 4)
    mime = _detect_mime_type(base64.b64decode(header_b64), fallback=declared_mime)
    return f"data:{mime};base64,{compact}"


def build_content_items(problem: str, images: list[str]) -> list[dict[str, str]]:
    """Interleave text segments and ``input_image`` blocks at ``<|image_N|>`` placeholders.

    Matches ``build_messages`` in ``MoonshotAI/PerceptionBench/eval/eval.py``: non-empty text
    segments are emitted unchanged, out-of-range placeholders remain in text, and unreferenced
    images are appended in list order at the end.
    """
    image_items = [
        {"type": "input_image", "image_url": normalize_image_data_url(img), "detail": "high"} for img in images
    ]
    items: list[dict[str, str]] = []
    last = 0
    used: set[int] = set()
    for match in _IMAGE_PLACEHOLDER_RE.finditer(problem):
        idx = int(match.group(1))
        if 1 <= idx <= len(image_items):
            used.add(idx - 1)
            if problem[last : match.start()]:
                items.append({"type": "input_text", "text": problem[last : match.start()]})
            items.append(image_items[idx - 1])
            last = match.end()
    if problem[last:]:
        items.append({"type": "input_text", "text": problem[last:]})
    for k, image_item in enumerate(image_items):
        if k not in used:
            items.append(image_item)
    return items


def _image_dimensions(raw_image: str) -> tuple[int, int]:
    """Read ``(width, height)`` from a base64 image data URL without external dependencies."""
    stripped = raw_image.strip()
    match = _DATA_URL_RE.match(stripped)
    encoded = match.group(2) if match else stripped
    data = base64.b64decode("".join(encoded.split()))
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8X":
            return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        if chunk == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data):
            while i < len(data) and data[i] == 0xFF:
                i += 1
            if i >= len(data):
                break
            marker = data[i]
            i += 1
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 3 : i + 7])
                return w, h
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            length = struct.unpack(">H", data[i : i + 2])[0]
            i += length
    raise ValueError("Unable to read image dimensions from unsupported or truncated image bytes")


def _scale_1000(value: float, denom: float = 1.0) -> int:
    return max(0, min(1000, round(value * 1000.0 / denom)))


def _replace_pixel_term(match: re.Match[str]) -> str:
    text = match.group(0)
    prefix = "the " if text.lower().startswith("the ") else ""
    plural = "s" if text.lower().endswith("coordinates") else ""
    return f"{prefix}normalized coordinate{plural}"


def _has_out_of_range_numeric_option(text: str) -> bool:
    for match in _OPTION_LINE_RE.finditer(text):
        rhs = match.group(1).strip()
        token = rhs[:-1].rstrip() if rhs.endswith(".") else rhs
        if _NUMERIC_OPTION_RE.fullmatch(token) and not _COORD_01_FULL_RE.fullmatch(token):
            return True
    return False


def normalize_prompt_coords_to_1000(problem: str, images: list[str], *, hint: str = "") -> str:
    """Rewrite image pixel and ``[0, 1]`` coordinates in ``problem`` to integer ``[0, 1000]`` ``[x, y]`` format.

    Non-image coordinates (Cartesian math plots, board notation, already-``[0, 1000]`` items) are left untouched.
    """
    is_pixel = bool(_DIM_HINT_RE.search(hint) or _ROW_COL_RE.search(problem) or _PIXEL_PAREN_RE.search(problem))
    if is_pixel and images:
        width, height = _image_dimensions(images[0])
        updated = _PIXEL_ORIGIN_SENTENCE_RE.sub("", problem)
        updated = _PIXEL_GIVEN_CLAUSE_RE.sub(lambda m: m.group(1).upper(), updated)
        changed = False

        if _ROW_COL_RE.search(updated):
            updated = _ROW_COL_RE.sub(
                lambda m: f"[{_scale_1000(int(m.group(2)), width)}, {_scale_1000(int(m.group(1)), height)}]",
                updated,
            )
            changed = True
        elif _PIXEL_PAREN_RE.search(updated):
            # DA-2K parenthesized pixel coordinate `(row, col)` (e.g. index 793 without hint).
            updated = _PIXEL_PAREN_RE.sub(
                lambda m: f"{m.group(1)}[{_scale_1000(int(m.group(3)), width)}, {_scale_1000(int(m.group(2)), height)}]",
                updated,
            )
            changed = True
        elif _PIXEL_BRACKET_RE.search(updated):
            # DA-2K square-bracket pixel coordinates use `[x, y]` order.
            updated = _PIXEL_BRACKET_RE.sub(
                lambda m: f"[{_scale_1000(int(m.group(1)), width)}, {_scale_1000(int(m.group(2)), height)}]",
                updated,
            )
            changed = True

        if changed:
            updated = _PIXEL_TERM_RE.sub(_replace_pixel_term, updated)
            return (
                f"{updated}\n\n"
                "Coordinates are normalized to [0, 1000] in [x, y] format (x: left to right, y: top to bottom)."
            )
        return problem

    if _ALREADY_1000_RE.search(problem):
        return problem

    is_norm_01 = hint.startswith("Coordinates are given in [") or (
        "normalized coordinates" in problem.lower() and bool(_NORM_01_RANGE_RE.search(problem))
    )
    if not is_norm_01:
        return problem

    tuple_count = 0
    has_bbox = False

    def _scale_tuple(match: re.Match[str]) -> str:
        nonlocal tuple_count, has_bbox
        open_delim, close_delim = match.group(1), match.group(6)
        if (open_delim, close_delim) not in (("[", "]"), ("(", ")")):
            return match.group(0)
        vals = [match.group(2), match.group(3)]
        if match.group(4) is not None and match.group(5) is not None:
            vals.extend([match.group(4), match.group(5)])
            has_bbox = True
        elif open_delim == "[" and float(vals[0]) == 0.0 and float(vals[1]) == 1.0:
            before = problem[: match.start()]
            after = problem[match.end() :]
            if _INTERVAL_PREFIX_RE.search(before) or _INTERVAL_SUFFIX_RE.search(after):
                return match.group(0)
        tuple_count += 1
        scaled = ", ".join(str(_scale_1000(float(v))) for v in vals)
        return f"[{scaled}]"

    updated = _NORM_01_TUPLE_RE.sub(_scale_tuple, problem)
    scalar_count = 0
    if tuple_count == 0 and _NORM_01_RANGE_RE.search(updated) and not _has_out_of_range_numeric_option(updated):
        updated, scalar_count = _NORM_01_SCALAR_OPTION_RE.subn(
            lambda m: f"{m.group(1)}{m.group(2)}{_scale_1000(float(m.group(3)))}",
            updated,
        )
    if tuple_count == 0 and scalar_count == 0:
        return problem

    updated, range_replacements = _NORM_01_RANGE_RE.subn(r"\g<1>1000\g<2>", updated)
    updated, range_text_replacements = _NORM_01_RANGE_TEXT_RE.subn(r"\g<1>1000", updated)
    updated, interval_replacements = _NORM_01_INTERVAL_RE.subn("[0, 1000]", updated)
    if range_replacements + range_text_replacements + interval_replacements == 0:
        fmt = "[x1, y1, x2, y2]" if (has_bbox or "x1, y1, x2, y2" in hint) else "[x, y]"
        updated = (
            f"{updated}\n\n"
            f"Coordinates are normalized to [0, 1000] in {fmt} format (x: left to right, y: top to bottom)."
        )
    return updated


def convert_record(
    record: dict[str, Any],
    *,
    include_hints: bool = False,
    normalize_coords_1000: bool = False,
) -> dict[str, Any]:
    """Convert one upstream PerceptionBench JSONL record into a NeMo Gym row."""
    if include_hints and normalize_coords_1000:
        raise ValueError(_MUTUAL_EXCLUSION_MSG)
    if "index" not in record or record["index"] is None:
        raise ValueError("PerceptionBench record missing 'index'")
    index = int(record["index"])
    problem = str(record.get("problem") or "")
    if not problem.strip():
        raise ValueError(f"PerceptionBench record {index} has an empty problem")
    raw_images = record.get("image") or []
    if not isinstance(raw_images, list):
        raw_images = []
    image_urls = [str(img) for img in raw_images]

    raw_answer = record.get("answer")
    if isinstance(raw_answer, dict):
        raw_answer = raw_answer.get("answer")
    answer = "" if raw_answer is None else str(raw_answer)
    if not answer.strip():
        raise ValueError(f"PerceptionBench record {index} has an empty answer")

    category = str(record.get("error_category") or "").strip()
    if not category:
        raise ValueError(f"PerceptionBench record {index} has no error_category")

    source_bmk_raw = record.get("source_bmk")
    source_bmk = str(source_bmk_raw) if source_bmk_raw is not None else "NA"

    raw_hint = record.get("hint")
    hint = raw_hint.strip() if isinstance(raw_hint, str) else ""

    if normalize_coords_1000:
        problem = normalize_prompt_coords_to_1000(problem, image_urls, hint=hint)
    prompt = f"{problem}\n\n{hint}" if include_hints and hint else problem

    content = build_content_items(prompt, image_urls)
    messages: list[dict[str, Any]] = []
    system = record.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content if content else prompt})

    return {
        "task_id": f"perceptionbench_{index}",
        "responses_create_params": {"input": messages},
        "verifier_metadata": {
            "index": index,
            "question": problem,
            "answer": answer,
            "category": category,
            "source_bmk": source_bmk,
            "source_idx": record.get("source_idx"),
        },
    }


def _resolve_source_jsonl(source_path: str | Path | None) -> Path:
    configured = source_path or os.environ.get(SOURCE_ENV)
    if configured is not None:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            candidate = candidate / HF_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(f"PerceptionBench source file not found: {candidate}")
        return candidate

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        HF_REPO_ID,
        HF_FILENAME,
        repo_type="dataset",
        revision=HF_REVISION,
    )
    return Path(downloaded)


def prepare(
    source_path: str | Path | None = None,
    output_path: str | Path = OUTPUT_FPATH,
    limit: int | None = None,
    *,
    include_hints: bool = False,
    normalize_coords_1000: bool = False,
) -> Path:
    """Prepare ``perceptionbench_benchmark.jsonl`` from the pinned release and return its path.

    ``include_hints`` and ``normalize_coords_1000`` are mutually exclusive: both default to
    ``False`` to preserve upstream ``MoonshotAI/PerceptionBench`` prompt parity.
    """
    if include_hints and normalize_coords_1000:
        raise ValueError(_MUTUAL_EXCLUSION_MSG)
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when provided, got {limit}")

    source_jsonl = _resolve_source_jsonl(source_path)
    if limit is None:
        actual_sha256 = _sha256_file(source_jsonl)
        if actual_sha256 != EXPECTED_SHA256:
            raise ValueError(
                f"{source_jsonl} has sha256 {actual_sha256}, expected pinned {EXPECTED_SHA256} "
                f"({HF_REPO_ID}@{HF_REVISION})"
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".tmp")

    seen_ids: set[str] = set()
    written = 0
    with source_jsonl.open("rb") as src, staged.open("wb") as dst:
        for raw_line in src:
            if not raw_line.strip():
                continue
            if limit is not None and written >= limit:
                break
            row = convert_record(
                orjson.loads(raw_line),
                include_hints=include_hints,
                normalize_coords_1000=normalize_coords_1000,
            )
            task_id = row["task_id"]
            if task_id in seen_ids:
                raise ValueError(f"Duplicate task_id {task_id} in {source_jsonl}")
            seen_ids.add(task_id)
            dst.write(orjson.dumps(row) + b"\n")
            written += 1

    if limit is None and written != EXPECTED_TASK_COUNT:
        staged.unlink(missing_ok=True)
        raise ValueError(f"Prepared {written} PerceptionBench items, expected {EXPECTED_TASK_COUNT}")

    staged.replace(destination)
    print(f"Wrote {written} PerceptionBench tasks to {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help=f"Local JSONL or directory (default: HF or ${SOURCE_ENV})")
    parser.add_argument("--output", default=str(OUTPUT_FPATH), help=f"Output JSONL (default: {OUTPUT_FPATH})")
    parser.add_argument("--limit", type=int, default=None, help="Prepare only the first N items (smoke testing)")
    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument(
        "--include-hints",
        action="store_true",
        help="Append upstream hint strings to user prompts (mutually exclusive with --normalize-coords-1000)",
    )
    coord_group.add_argument(
        "--normalize-coords-1000",
        action="store_true",
        help=(
            "Normalize image pixel and [0, 1] coordinates in prompts to integer [0, 1000] [x, y] format "
            "(mutually exclusive with --include-hints)"
        ),
    )
    args = parser.parse_args()
    prepare(
        source_path=args.source,
        output_path=args.output,
        limit=args.limit,
        include_hints=args.include_hints,
        normalize_coords_1000=args.normalize_coords_1000,
    )
