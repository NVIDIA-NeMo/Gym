# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GDPVal pairwise comparison judging.

Shared core used by both:
- Inline comparison reward during ``ng_collect_rollouts``
- Post-hoc ``scripts/compare_elo.py`` script

GDPVal pairwise comparison judging.
"""

from __future__ import annotations

import base64
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any


JUDGE_PROMPT = (
    "Given a task description and reference files, select which of two submission file(s) "
    "better completed the task. "
    "Explain your reasoning then answer BOXED[A], BOXED[B], or BOXED[TIE].\n"
)

A_WIN_RESPONSE = "BOXED[A]"
B_WIN_RESPONSE = "BOXED[B]"
TIE_RESPONSE = "BOXED[TIE]"

TASK_TEMPLATE = "<TASK_DESCRIPTION_START>\n{task}\n<TASK_DESCRIPTION_END>\n\n"

REFERENCES_OPEN = "<REFERENCES_FILES_START>\n"
REFERENCES_CLOSE = "\n<REFERENCES_FILES_END>\n\n"

SUBMISSION_A_OPEN = "<SUBMISSION_A_START>\n"
SUBMISSION_A_CLOSE = "\n<SUBMISSION_A_END>\n\n"
SUBMISSION_B_OPEN = "<SUBMISSION_B_START>\n"
SUBMISSION_B_CLOSE = "\n<SUBMISSION_B_END>\n\n"

IGNORE_FILES = {
    "finish_params.json",
    "history.json",
    "history.pkl",
    "metadata.json",
    "inprogress_history.json",
    "log.txt",
    "reference_files",
}

REQUEST_MAX_ATTEMPTS = 5
REQUEST_INITIAL_BACKOFF_SECONDS = 5.0
REQUEST_BACKOFF_MULTIPLIER = 2.0
REQUEST_MAX_BACKOFF_SECONDS = 60.0
RETRYABLE_ERROR_MARKERS = (
    "429",
    "502",
    "503",
    "504",
    "rate limit",
    "ratelimit",
    "resource_exhausted",
    "resource has been exhausted",
    "throttling",
    "bad gateway",
    "gateway timeout",
    "gateway time-out",
    "service unavailable",
    "upstream",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "connection error",
)

# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


def _data_url(mime_type: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _load_raw_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_media(path: str | Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _convert_to_pdf(path: str | Path) -> bytes | None:
    """Load a pre-converted PDF (same name, .pdf extension). Returns None if missing."""
    input_path = Path(path).resolve()
    output_path = input_path.with_suffix(".pdf")
    if output_path.exists():
        return _load_media(output_path)
    return None


def _maybe_unzip(path: str | Path) -> list[Path]:
    path = Path(path)
    extracted_paths: list[Path] = []
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            members = zip_ref.namelist()
            zip_ref.extractall(path.parent)
            extracted_paths = [path.parent / Path(member) for member in members if member]
    except (zipfile.BadZipFile, zipfile.LargeZipFile, FileNotFoundError):
        pass
    return extracted_paths


FILE_TYPE_MAP: dict[str, dict[str, Any]] = {
    "pdf": {"type": "PDF", "converter": None, "mime_type": "application/pdf"},
    "jpg": {"type": "IMG", "converter": _load_media, "mime_type": "image/jpeg"},
    "jpeg": {"type": "IMG", "converter": _load_media, "mime_type": "image/jpeg"},
    "png": {"type": "IMG", "converter": _load_media, "mime_type": "image/png"},
    "webp": {"type": "IMG", "converter": _load_media, "mime_type": "image/webp"},
    "heic": {"type": "IMG", "converter": _load_media, "mime_type": "image/heic"},
    "heif": {"type": "IMG", "converter": _load_media, "mime_type": "image/heif"},
    "wav": {"type": "AUDIO", "converter": _load_media, "mime_type": "audio/wav"},
    "mp3": {"type": "AUDIO", "converter": _load_media, "mime_type": "audio/mp3"},
    "ogg": {"type": "AUDIO", "converter": _load_media, "mime_type": "audio/ogg"},
    "aiff": {"type": "AUDIO", "converter": _load_media, "mime_type": "audio/aiff"},
    "aac": {"type": "AUDIO", "converter": _load_media, "mime_type": "audio/aac"},
    "flac": {"type": "AUDIO", "converter": _load_media, "mime_type": "audio/flac"},
    "mp4": {"type": "VIDEO", "converter": _load_media, "mime_type": "video/mp4"},
    "mov": {"type": "VIDEO", "converter": _load_media, "mime_type": "video/mov"},
    "avi": {"type": "VIDEO", "converter": _load_media, "mime_type": "video/avi"},
    "x-flv": {"type": "VIDEO", "converter": _load_media, "mime_type": "video/x-flv"},
    "webm": {"type": "VIDEO", "converter": _load_media, "mime_type": "video/webm"},
    "wmv": {"type": "VIDEO", "converter": _load_media, "mime_type": "video/wmv"},
    "3gpp": {"type": "VIDEO", "converter": _load_media, "mime_type": "video/3gpp"},
    "docx": {"type": "DOC", "converter": _convert_to_pdf, "mime_type": "application/pdf"},
    "pptx": {"type": "DOC", "converter": _convert_to_pdf, "mime_type": "application/pdf"},
    "xlsx": {"type": "DOC", "converter": _convert_to_pdf, "mime_type": "application/pdf"},
    "txt": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "csv": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "json": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "xml": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "html": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "md": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "yaml": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "yml": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "py": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "sh": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "bash": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "c": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "cpp": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "java": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "js": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "tsx": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "sol": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "ts": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
}


def get_file_content_block(file_dir: str, file_name: str) -> dict | None:
    """Return a single OpenAI content block (dict) for a file, or ``None``."""
    file_extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if file_extension not in FILE_TYPE_MAP:
        file_type = "DOC"
        file_converter = _convert_to_pdf
        file_mime_type = "application/pdf"
    else:
        file_type = FILE_TYPE_MAP[file_extension]["type"]
        file_converter = FILE_TYPE_MAP[file_extension]["converter"]
        file_mime_type = FILE_TYPE_MAP[file_extension]["mime_type"]

    full_path = os.path.join(file_dir, file_name)

    try:
        if file_type == "TXT":
            raw_text = file_converter(full_path)
            return {"type": "text", "text": raw_text}

        if file_type == "DOC":
            doc_bytes = file_converter(full_path)
            if doc_bytes is None:
                return None
            return {"type": "image_url", "image_url": {"url": _data_url(file_mime_type, doc_bytes)}}

        if file_type == "PDF":
            data = Path(full_path).read_bytes()
            return {"type": "image_url", "image_url": {"url": _data_url(file_mime_type, data)}}

        if file_type in ("IMG", "AUDIO", "VIDEO"):
            media_bytes = file_converter(full_path)
            return {"type": "image_url", "image_url": {"url": _data_url(file_mime_type, media_bytes)}}

    except Exception as e:
        raise RuntimeError(f"Error getting file: {file_name} in directory: {file_dir}: {e}") from e

    return None


def build_file_section(file_dir: str | None, clean_up_list: list[Path] | None = None) -> list[dict]:
    """Build OpenAI content blocks from all files in a directory.

    Skips files in ``IGNORE_FILES``.  Extracts zips first.  Returns a list
    of content block dicts suitable for OpenAI messages.
    """
    if clean_up_list is None:
        clean_up_list = []

    section: list[dict] = []
    no_files = True

    if file_dir is not None and os.path.exists(file_dir):
        for file_name in os.listdir(file_dir):
            if file_name.lower().endswith(".zip"):
                extracted_paths = _maybe_unzip(os.path.join(file_dir, file_name))
                clean_up_list.extend(extracted_paths)

    if file_dir is not None and os.path.exists(file_dir):
        for file_name in sorted(os.listdir(file_dir)):
            full_path = os.path.join(file_dir, file_name)
            if os.path.isdir(full_path) or file_name.lower().endswith(".zip"):
                continue
            if file_name in IGNORE_FILES:
                continue
            section.append({"type": "text", "text": f"\n{file_name}:\n"})
            block = get_file_content_block(file_dir, file_name)
            if block is not None:
                section.append(block)
                no_files = False

    if no_files:
        section.append({"type": "text", "text": "None"})

    return section


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def construct_judge_messages(
    task_prompt: str,
    refs: list[dict],
    submission_a: list[dict],
    submission_b: list[dict],
) -> list[dict]:
    """Assemble OpenAI messages for the judge: prompt + task + refs + submissions."""
    content: list[dict] = []
    content.append({"type": "text", "text": JUDGE_PROMPT + TASK_TEMPLATE.format(task=task_prompt)})
    content.append({"type": "text", "text": REFERENCES_OPEN})
    content.extend(refs)
    content.append({"type": "text", "text": REFERENCES_CLOSE})
    content.append({"type": "text", "text": SUBMISSION_A_OPEN})
    content.extend(submission_a)
    content.append({"type": "text", "text": SUBMISSION_A_CLOSE})
    content.append({"type": "text", "text": SUBMISSION_B_OPEN})
    content.extend(submission_b)
    content.append({"type": "text", "text": SUBMISSION_B_CLOSE})

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Judge API call
# ---------------------------------------------------------------------------


def _is_retryable(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(marker in error_text for marker in RETRYABLE_ERROR_MARKERS)


def send_judge_request(
    client: Any,
    model: str,
    messages: list[dict],
    max_output_tokens: int = 65535,
) -> str:
    """Send a judge request with exponential-backoff retry.  Returns response text."""
    backoff = REQUEST_INITIAL_BACKOFF_SECONDS

    for attempt in range(1, REQUEST_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=1.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as error:
            retryable = _is_retryable(error)
            is_last = attempt == REQUEST_MAX_ATTEMPTS
            if not retryable or is_last:
                raise
            print(
                f"  Judge request attempt {attempt}/{REQUEST_MAX_ATTEMPTS} failed "
                f"(retryable={retryable}), retrying in {backoff:.1f}s...",
                flush=True,
            )
            time.sleep(backoff)
            backoff = min(backoff * REQUEST_BACKOFF_MULTIPLIER, REQUEST_MAX_BACKOFF_SECONDS)

    raise RuntimeError("Unreachable retry loop exit")


# ---------------------------------------------------------------------------
# Judgement parsing and tallying
# ---------------------------------------------------------------------------


def parse_judgement(response_text: str) -> str:
    """Extract ``BOXED[A]``, ``BOXED[B]``, or ``BOXED[TIE]`` from judge response."""
    if A_WIN_RESPONSE in response_text:
        return A_WIN_RESPONSE
    if B_WIN_RESPONSE in response_text:
        return B_WIN_RESPONSE
    if TIE_RESPONSE in response_text:
        return TIE_RESPONSE
    return TIE_RESPONSE


def tally_result(
    judgement: str,
    swapped: bool,
    win_count_a: int,
    win_count_b: int,
    tie_count: int,
) -> tuple[int, int, int]:
    """Update win/loss/tie counters, accounting for position swap."""
    if swapped:
        if B_WIN_RESPONSE in judgement:
            win_count_a += 1
        elif A_WIN_RESPONSE in judgement:
            win_count_b += 1
        elif TIE_RESPONSE in judgement:
            tie_count += 1
    else:
        if A_WIN_RESPONSE in judgement:
            win_count_a += 1
        elif B_WIN_RESPONSE in judgement:
            win_count_b += 1
        elif TIE_RESPONSE in judgement:
            tie_count += 1
    return win_count_a, win_count_b, tie_count


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_trials(
    client: Any,
    model: str,
    task_prompt: str,
    refs: list[dict],
    submission_a: list[dict],
    submission_b: list[dict],
    num_trials: int = 4,
    max_output_tokens: int = 65535,
) -> dict:
    """Run ``num_trials`` judge calls, alternating swapped/unswapped positions.

    Returns a dict with ``winner``, ``win_count_a``, ``win_count_b``,
    ``tie_count``, and ``task_count``.
    """
    win_count_a = 0
    win_count_b = 0
    tie_count = 0

    for i in range(num_trials):
        swapped = i % 2 != 0
        current_a = submission_b if swapped else submission_a
        current_b = submission_a if swapped else submission_b

        messages = construct_judge_messages(
            task_prompt=task_prompt,
            refs=refs,
            submission_a=current_a,
            submission_b=current_b,
        )
        response_text = send_judge_request(client, model, messages, max_output_tokens)
        judgement = parse_judgement(response_text)
        win_count_a, win_count_b, tie_count = tally_result(judgement, swapped, win_count_a, win_count_b, tie_count)

    if win_count_a > win_count_b:
        winner = A_WIN_RESPONSE
    elif win_count_b > win_count_a:
        winner = B_WIN_RESPONSE
    else:
        winner = TIE_RESPONSE

    return {
        "winner": winner,
        "win_count_a": win_count_a,
        "win_count_b": win_count_b,
        "tie_count": tie_count,
        "task_count": num_trials,
    }


# ---------------------------------------------------------------------------
# ELO calculation
# ---------------------------------------------------------------------------


def calculate_elo(win_rate: float, ref_elo: float) -> tuple[float, float]:
    """Compute ELO from win rate against a reference model.

    Returns ``(elo, normalized_elo)`` where normalized is ``(elo - 500) / 2000``.
    """
    if win_rate <= 0.0 or win_rate >= 1.0:
        win_rate = max(0.001, min(0.999, win_rate))
    elo = ref_elo - 400.0 * (math.log10(1.0 - win_rate) - math.log10(win_rate))
    normalized_elo = (elo - 500.0) / 2000.0
    return elo, normalized_elo


def compute_comparison_reward(winner: str) -> float:
    """Convert a BOXED winner string to a reward float.

    - Reference model (A) wins → 0.0  (eval model lost)
    - Eval model (B) wins → 1.0
    - Tie → 0.5
    """
    if winner == B_WIN_RESPONSE:
        return 1.0
    if winner == A_WIN_RESPONSE:
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# Convenience: check if a task was attempted
# ---------------------------------------------------------------------------


def task_attempted(task_dir: str) -> bool:
    """Return True if the task directory has a ``finish_params.json`` (completed run)."""
    return os.path.exists(task_dir) and os.path.exists(os.path.join(task_dir, "finish_params.json"))


def clean_up_paths(paths: list[Path]) -> None:
    """Remove extracted zip artifacts."""
    for path in paths:
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        except Exception:
            pass
