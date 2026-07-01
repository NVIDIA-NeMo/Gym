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

Used by the GDPVal resources server's ``verify`` (per-task pairwise judge
between the eval model and a reference model's deliverables) and
``aggregate_metrics`` (turns win/loss/tie counts into an ELO rating).
"""

from __future__ import annotations

import base64
import math
import os
import random
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from openai import APITimeoutError

from resources_servers.gdpval.judge_panel import merge_create_kwargs, sample_judge


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
# Per-request OpenAI client timeout. Multimodal payloads through a flaky
# proxy have historically taken hours when this defaulted to 600 s × 5
# retries (50 min/trial × 4 trials = 200 min/verify worst case). 120 s here,
# combined with non-retryable ``APITimeoutError`` below, bounds wall-clock
# damage to ~8 min/verify when the upstream is genuinely down.
JUDGE_REQUEST_TIMEOUT_SECONDS = 120.0
# Per-file size cap on multimodal content blocks. Above this the file is
# replaced with a one-line text marker so the judge still knows what was
# claimed without us pushing 100s of MB of base64 through the proxy.
# Set high enough (250 MB) that normal multi-stem audio and reference
# videos in the GDPVal task set still go through; only catches the truly
# pathological cases (e.g. ``task_a941b6d8`` 657 MB overlay clip).
MAX_FILE_BYTES_FOR_JUDGE = 250 * 1024 * 1024
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


def _maybe_unzip(path: str | Path) -> tuple[Path | None, list[Path]]:
    """Extract a zip into a per-call tempdir; never write into ``path.parent``.

    The reference deliverables tree is mounted read-only in production, so the
    previous behaviour of ``extractall(path.parent)`` raised ``PermissionError``
    and failed /verify outright. Returns ``(extract_dir, member_paths)`` —
    callers are responsible for ``shutil.rmtree(extract_dir)`` after they're
    done reading the members.
    """
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            extract_dir = Path(tempfile.mkdtemp(prefix="gdpval_unzip_"))
            zip_ref.extractall(extract_dir)
            members = zip_ref.namelist()
            extracted_paths = [extract_dir / Path(member) for member in members if member]
        return extract_dir, extracted_paths
    except (zipfile.BadZipFile, zipfile.LargeZipFile, FileNotFoundError, OSError):
        return None, []


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
        size_bytes = os.path.getsize(full_path)
    except OSError:
        return None
    if size_bytes > MAX_FILE_BYTES_FOR_JUDGE:
        size_mb = size_bytes / (1024 * 1024)
        return {
            "type": "text",
            "text": f"[oversize: {file_name} {size_mb:.1f}MB — not included]",
        }

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

    Skips files in ``IGNORE_FILES``. Extracts zips into per-call tempdirs
    (the dirs are appended to ``clean_up_list`` for the caller to ``rmtree``).
    Returns a list of content block dicts suitable for OpenAI messages.
    """
    if clean_up_list is None:
        clean_up_list = []

    section: list[dict] = []
    no_files = True

    extracted_dirs: list[Path] = []
    if file_dir is not None and os.path.exists(file_dir):
        for file_name in os.listdir(file_dir):
            if file_name.lower().endswith(".zip"):
                extract_dir, _ = _maybe_unzip(os.path.join(file_dir, file_name))
                if extract_dir is not None:
                    clean_up_list.append(extract_dir)
                    extracted_dirs.append(extract_dir)

    def _emit(directory: str, file_name: str) -> None:
        nonlocal no_files
        if file_name in IGNORE_FILES:
            return
        section.append({"type": "text", "text": f"\n{file_name}:\n"})
        block = get_file_content_block(directory, file_name)
        if block is not None:
            section.append(block)
            no_files = False

    if file_dir is not None and os.path.exists(file_dir):
        for file_name in sorted(os.listdir(file_dir)):
            full_path = os.path.join(file_dir, file_name)
            if os.path.isdir(full_path) or file_name.lower().endswith(".zip"):
                continue
            _emit(file_dir, file_name)

    for extract_dir in extracted_dirs:
        for member in sorted(extract_dir.rglob("*")):
            if not member.is_file():
                continue
            _emit(str(member.parent), member.name)

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
    # Timeouts on multimodal payloads are deterministic — the payload is too
    # large for the judge endpoint to digest in time, and retrying just burns
    # another full timeout window per attempt. Fail the trial fast instead.
    if isinstance(error, APITimeoutError):
        return False
    error_text = str(error).lower()
    return any(marker in error_text for marker in RETRYABLE_ERROR_MARKERS)


def send_judge_request(
    client: Any,
    model: str,
    messages: list[dict],
    max_output_tokens: int = 65535,
    create_overrides: Optional[dict] = None,
) -> str:
    """Send a judge request with exponential-backoff retry.  Returns response text.

    *create_overrides* (a panel member's reasoning/generation knobs) is merged
    over the default create kwargs; a ``None`` value removes the matching
    default (e.g. to drop ``temperature`` for a reasoning model that rejects it).
    """
    backoff = REQUEST_INITIAL_BACKOFF_SECONDS
    create_kwargs = merge_create_kwargs(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": 1.0,
        },
        create_overrides,
    )

    for attempt in range(1, REQUEST_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(**create_kwargs)
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


@dataclass
class Judge:
    """A panel member bound to a live (sync) OpenAI client for the trial loop.

    Built by the resources server from a
    :class:`resources_servers.gdpval.judge_panel.ResolvedJudge` (with one OpenAI
    client per distinct upstream, so members that share a proxy reuse a client).
    ``run_trials`` samples one of these per trial.
    """

    name: str
    client: Any
    model: str
    create_overrides: Optional[dict] = None
    weight: float = 1.0
    handles_audio_video: bool = False


def run_trials(
    judges: list[Judge],
    task_prompt: str,
    refs: list[dict],
    submission_a: list[dict],
    submission_b: list[dict],
    num_trials: int = 4,
    max_output_tokens: int = 65535,
    return_raw_responses: bool = False,
    rng: Optional[random.Random] = None,
) -> dict:
    """Run ``num_trials`` judge calls, alternating swapped/unswapped positions.

    For each trial one member of *judges* is sampled (see
    ``judge_panel.sample_judge``) — the "sample between the judges for each
    comparison" panel behavior. With a single-member panel this reduces to the
    historical single-judge loop. Pass *rng* (a seeded ``random.Random``) for
    reproducible judge selection.

    Returns a dict with ``winner``, ``win_count_a``, ``win_count_b``,
    ``tie_count``, ``task_count``, ``per_judge`` (per-member a/b/tie/trial
    counts keyed by judge name), and ``trial_judges`` (the judge name that graded
    each trial, ordered by trial index — always present so the grader of every
    match is documented).

    When ``return_raw_responses`` is True, the dict also carries
    ``raw_responses`` (per-trial judge completion strings, same ordering as
    ``trial_judges`` — trial ``i`` was swapped iff ``i % 2 != 0``).
    """
    if not judges:
        raise ValueError("run_trials requires a non-empty judge panel")
    rng = rng or random.Random()

    win_count_a = 0
    win_count_b = 0
    tie_count = 0
    raw_responses: list[str] = []
    trial_judges: list[str] = []
    per_judge: dict[str, dict] = {}

    for i in range(num_trials):
        judge = sample_judge(judges, rng)
        trial_judges.append(judge.name)
        swapped = i % 2 != 0
        current_a = submission_b if swapped else submission_a
        current_b = submission_a if swapped else submission_b

        messages = construct_judge_messages(
            task_prompt=task_prompt,
            refs=refs,
            submission_a=current_a,
            submission_b=current_b,
        )
        response_text = send_judge_request(
            judge.client, judge.model, messages, max_output_tokens, judge.create_overrides
        )
        if return_raw_responses:
            raw_responses.append(response_text)
        judgement = parse_judgement(response_text)
        win_count_a, win_count_b, tie_count = tally_result(judgement, swapped, win_count_a, win_count_b, tie_count)

        # Per-judge tally (same A=submission_a / B=submission_b convention as the
        # global counts) so the panel's per-member balance is auditable.
        jc = per_judge.setdefault(judge.name, {"win_count_a": 0, "win_count_b": 0, "tie_count": 0, "trials": 0})
        jc["win_count_a"], jc["win_count_b"], jc["tie_count"] = tally_result(
            judgement, swapped, jc["win_count_a"], jc["win_count_b"], jc["tie_count"]
        )
        jc["trials"] += 1

    if win_count_a > win_count_b:
        winner = A_WIN_RESPONSE
    elif win_count_b > win_count_a:
        winner = B_WIN_RESPONSE
    else:
        winner = TIE_RESPONSE

    result: dict = {
        "winner": winner,
        "win_count_a": win_count_a,
        "win_count_b": win_count_b,
        "tie_count": tie_count,
        "task_count": num_trials,
        "per_judge": per_judge,
        # Always recorded (just judge names, ordered by trial) so every match's
        # per-trial grader is documented even when raw responses aren't kept.
        "trial_judges": trial_judges,
    }
    if return_raw_responses:
        result["raw_responses"] = raw_responses
    return result


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


def calculate_mle_elo(
    battles: list[tuple[float, float, float, float]],
    scale: float = 400.0,
    base: float = 10.0,
) -> tuple[float, float] | None:
    """Anchored Bradley-Terry MLE ELO for one eval model vs N fixed references.

    This is the multi-reference generalization of ``calculate_elo``. It applies
    the traditional ELO rating system (logistic / Bradley-Terry) to the pooled
    pairwise comparisons, estimating the eval model's rating globally rather
    than inverting a single win rate against a single anchor.

    ``battles`` is a list of ``(reference_elo, wins, losses, ties)`` where the
    counts are the eval model's win / loss / tie vote totals against that
    reference model (ties counted as half a win). The reference ratings are
    held **fixed** at their known ELOs (e.g. published Arena/AA numbers); the
    eval model's rating ``R`` is the single free parameter, found by maximizing
    the Bradley-Terry log-likelihood

        L(R) = sum_i [ s_i * log(p_i) + (n_i - s_i) * log(1 - p_i) ]

    with ``p_i = 1 / (1 + base**((reference_elo_i - R) / scale))``, ``n_i`` the
    number of games vs reference ``i`` and ``s_i = wins_i + 0.5 * ties_i``.

    For a single reference this reduces exactly to ``calculate_elo``. Returns
    ``(elo, normalized_elo)`` with ``normalized_elo = (elo - 500) / 2000``, or
    ``None`` when there are no games to fit.
    """
    data: list[tuple[float, float, float]] = []
    for ref_elo, wins, losses, ties in battles:
        n = float(wins) + float(losses) + float(ties)
        if n <= 0:
            continue
        s = float(wins) + 0.5 * float(ties)
        data.append((float(ref_elo), s, n))

    if not data:
        return None

    total_s = sum(s for _, s, _ in data)
    total_n = sum(n for _, _, n in data)
    eps = 1e-3

    overall_win_rate = total_s / total_n
    if overall_win_rate <= eps or overall_win_rate >= 1.0 - eps:
        # Degenerate: the eval model won (or lost) every battle, so the MLE
        # rating diverges to ±inf. Clamp exactly like ``calculate_elo`` does,
        # anchored to the game-weighted mean reference ELO.
        clamped = min(max(overall_win_rate, eps), 1.0 - eps)
        mean_ref = sum(ref_elo * n for ref_elo, _, n in data) / total_n
        elo = mean_ref - scale * (math.log10(1.0 - clamped) - math.log10(clamped))
        return elo, (elo - 500.0) / 2000.0

    def gradient(rating: float) -> float:
        # dL/dR up to the positive constant ln(base)/scale: sum_i (s_i - n_i*p_i).
        # Strictly decreasing in ``rating``, so the root is unique.
        total = 0.0
        for ref_elo, s, n in data:
            p = 1.0 / (1.0 + base ** ((ref_elo - rating) / scale))
            total += s - n * p
        return total

    # gradient(lo) > 0 and gradient(hi) < 0 are guaranteed once the overall win
    # rate is strictly inside (0, 1); bisect for the unique root.
    lo = min(ref_elo for ref_elo, _, _ in data) - 4000.0
    hi = max(ref_elo for ref_elo, _, _ in data) + 4000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if gradient(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    elo = 0.5 * (lo + hi)
    return elo, (elo - 500.0) / 2000.0


def predict_win_rate(eval_elo: float, ref_elo: float, scale: float = 400.0, base: float = 10.0) -> float:
    """Expected eval-model win probability vs a reference at ``ref_elo``."""
    return 1.0 / (1.0 + base ** ((ref_elo - eval_elo) / scale))


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
