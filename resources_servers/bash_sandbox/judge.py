# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GDPVal LLM-as-judge for pairwise comparison of model outputs.

Ported from stirrup/gdpval/run_judge.py with async adaptation. Compares an evaluated
model's outputs against committee model outputs using an LLM judge (Gemini via VertexAI).

Key differences from stirrup:
- Async (asyncio) instead of threaded
- Uses Bradley-Terry win_rate: (wins + 0.5*ties) / total (stirrup ignores ties)
- Per-task reward is binary majority vote (1.0/0.5/0.0), not per-task ELO
- Regex-based verdict parsing (last BOXED[A|B|TIE] token) instead of substring `in` check
- Zip extraction uses per-invocation temp dirs to avoid mutating committee cache
"""

import asyncio
import base64
import logging
import os
import re
import shutil
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI


logger = logging.getLogger(__name__)

# Files to ignore when building file sections for the judge
IGNORE_FILES = [
    "finish_params.json",
    "history.json",
    "metadata.json",
    "log.txt",
]

JUDGE_PROMPT = (
    "Given a task description and reference files, select which of two submission file(s) "
    "better completed the task. Explain your reasoning then answer BOXED[A], BOXED[B], or BOXED[TIE].\n"
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

# Regex to find the last BOXED[A], BOXED[B], or BOXED[TIE] token in the response
_VERDICT_PATTERN = re.compile(r"BOXED\[(A|B|TIE)\]")

# --- File handling utilities ---


def convert_to_pdf(path):
    """Return PDF bytes if a pre-converted PDF exists next to the file, else None."""
    input_path = Path(path).resolve()
    output_path = input_path.with_suffix(".pdf")
    if output_path.exists():
        return load_media(output_path)
    return None


def load_raw_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_media(path):
    with open(path, "rb") as f:
        return f.read()


def maybe_unzip(path, extract_dir):
    """Extract a zip file to extract_dir (not in-place) to avoid mutating source dirs.

    Args:
        path: Path to the zip file.
        extract_dir: Directory to extract into.

    Returns:
        List of extracted file paths.
    """
    path = Path(path)
    extracted_paths = []
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            members = zip_ref.namelist()
            zip_ref.extractall(extract_dir)
            extracted_paths = [Path(extract_dir) / Path(member) for member in members if member]
    except (zipfile.BadZipFile, zipfile.LargeZipFile, FileNotFoundError):
        pass
    return extracted_paths


FILE_TYPE_MAP = {
    "pdf": {"type": "PDF", "converter": None, "mime_type": "application/pdf"},
    "jpg": {"type": "IMG", "converter": load_media, "mime_type": "image/jpeg"},
    "jpeg": {"type": "IMG", "converter": load_media, "mime_type": "image/jpeg"},
    "png": {"type": "IMG", "converter": load_media, "mime_type": "image/png"},
    "webp": {"type": "IMG", "converter": load_media, "mime_type": "image/webp"},
    "heic": {"type": "IMG", "converter": load_media, "mime_type": "image/heic"},
    "heif": {"type": "IMG", "converter": load_media, "mime_type": "image/heif"},
    "wav": {"type": "AUDIO", "converter": load_media, "mime_type": "audio/wav"},
    "mp3": {"type": "AUDIO", "converter": load_media, "mime_type": "audio/mp3"},
    "ogg": {"type": "AUDIO", "converter": load_media, "mime_type": "audio/ogg"},
    "aiff": {"type": "AUDIO", "converter": load_media, "mime_type": "audio/aiff"},
    "aac": {"type": "AUDIO", "converter": load_media, "mime_type": "audio/aac"},
    "flac": {"type": "AUDIO", "converter": load_media, "mime_type": "audio/flac"},
    "mp4": {"type": "VIDEO", "converter": load_media, "mime_type": "video/mp4"},
    "mov": {"type": "VIDEO", "converter": load_media, "mime_type": "video/mov"},
    "avi": {"type": "VIDEO", "converter": load_media, "mime_type": "video/avi"},
    "x-flv": {"type": "VIDEO", "converter": load_media, "mime_type": "video/x-flv"},
    "webm": {"type": "VIDEO", "converter": load_media, "mime_type": "video/webm"},
    "wmv": {"type": "VIDEO", "converter": load_media, "mime_type": "video/wmv"},
    "3gpp": {"type": "VIDEO", "converter": load_media, "mime_type": "video/3gpp"},
    "docx": {"type": "DOC", "converter": convert_to_pdf, "mime_type": "application/pdf"},
    "pptx": {"type": "DOC", "converter": convert_to_pdf, "mime_type": "application/pdf"},
    "xlsx": {"type": "DOC", "converter": convert_to_pdf, "mime_type": "application/pdf"},
    "txt": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "csv": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "json": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "xml": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "html": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "md": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "yaml": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "yml": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "py": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "sh": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "bash": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "c": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "cpp": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "java": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "js": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "tsx": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "sol": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
    "ts": {"type": "TXT", "converter": load_raw_text, "mime_type": None},
}


# --- Result dataclass ---


@dataclass
class JudgementResult:
    """Result of judging one task against one committee model.

    Convention (un-swapped case):
      submission_a = committee model output
      submission_b = evaluated model output
      A_WIN → committee wins → increment win_count_committee
      B_WIN → evaluated wins → increment win_count_evaluated
      When swapped: logic inverts.
    """

    committee_model_name: str
    win_count_evaluated: int = 0
    win_count_committee: int = 0
    tie_count: int = 0
    num_trials: int = 0
    success: bool = False
    error_message: str | None = None

    @property
    def win_rate(self) -> float:
        """Bradley-Terry win rate: (wins + 0.5*ties) / total.

        This intentionally differs from stirrup which uses wins_b/task_count
        (excluding ties from numerator). Ties should count as half-wins.
        """
        total = self.win_count_evaluated + self.win_count_committee + self.tie_count
        if total == 0:
            return 0.5
        return (self.win_count_evaluated + 0.5 * self.tie_count) / total

    @property
    def reward(self) -> float:
        """Binary per-task reward based on majority vote across trials.

        Returns:
            1.0 if evaluated model wins majority of trials
            0.5 if tie (equal wins or all ties)
            0.0 if committee model wins majority
        """
        if self.win_count_evaluated > self.win_count_committee:
            return 1.0
        elif self.win_count_committee > self.win_count_evaluated:
            return 0.0
        else:
            return 0.5


# --- Judge class ---


class GDPValJudge:
    """Async LLM-as-judge for pairwise comparison using Gemini via VertexAI, or optionally via an
    OpenAI-compatible endpoint (nvidia_openai_api_key / nvidia_openai_model).

    Adapted from stirrup's Judge class with:
    - asyncio.Semaphore instead of threading.BoundedSemaphore
    - Sync genai calls wrapped in run_in_executor
    - Thread-local genai.Client instances for credential thread-safety
    - Retry with exponential backoff for transient API failures
    """

    def __init__(
        self,
        gcp_project_id: str,
        gcp_location: str = "global",
        judge_model_name: str = "gemini-3-pro-preview",
        thinking_budget: int = 5000,
        max_output_tokens: int = 65535,
        num_trials: int = 4,
        max_concurrent_judgements: int = 10,
        nvidia_openai_api_key: str | None = None,
        nvidia_openai_model: str | None = None,
    ):
        self.gcp_project_id = gcp_project_id
        self.gcp_location = gcp_location
        self.judge_model_name = judge_model_name
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.num_trials = num_trials
        self.nvidia_openai_api_key = nvidia_openai_api_key
        self.nvidia_openai_model = nvidia_openai_model
        if nvidia_openai_api_key:
            self._openai_client = OpenAI(
                api_key=nvidia_openai_api_key,
                base_url="https://inference-api.nvidia.com/v1",
            )

        self._semaphore = asyncio.Semaphore(max_concurrent_judgements)
        # Thread-local storage for genai.Client instances (one per executor thread)
        self._tls = threading.local()

        self._generation_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=self.max_output_tokens,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            tools=[],
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
        )

    def _get_client(self) -> genai.Client:
        """Get or create a thread-local genai.Client for credential thread-safety."""
        if not hasattr(self._tls, "client"):
            self._tls.client = genai.Client(
                vertexai=True,
                project=self.gcp_project_id,
                location=self.gcp_location,
            )
        return self._tls.client

    def _get_file(self, file_dir: str, file_name: str) -> types.Part | None:
        """Load a file and return it as a genai Part for the judge prompt."""
        file_extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        if file_extension not in FILE_TYPE_MAP:
            file_type = "DOC"
            file_converter = convert_to_pdf
            file_mime_type = "application/pdf"
        else:
            file_type = FILE_TYPE_MAP[file_extension]["type"]
            file_converter = FILE_TYPE_MAP[file_extension]["converter"]
            file_mime_type = FILE_TYPE_MAP[file_extension]["mime_type"]

        try:
            if file_type == "TXT":
                raw_text = file_converter(os.path.join(file_dir, file_name))
                return types.Part.from_text(text=raw_text)

            if file_type == "DOC":
                doc_bytes = file_converter(os.path.join(file_dir, file_name))
                if doc_bytes is None:
                    return None
                return types.Part.from_bytes(data=doc_bytes, mime_type=file_mime_type)

            if file_type == "PDF":
                pdf_path = Path(os.path.join(file_dir, file_name))
                return types.Part.from_bytes(data=pdf_path.read_bytes(), mime_type=file_mime_type)

            if file_type in ("IMG", "AUDIO", "VIDEO"):
                media_bytes = file_converter(os.path.join(file_dir, file_name))
                return types.Part.from_bytes(data=media_bytes, mime_type=file_mime_type)

        except Exception as e:
            raise RuntimeError(f"Error getting file: {file_name} in directory: {file_dir}: {e}") from e

        return None

    def _build_file_section(self, file_dir: str | None, extract_dir: str) -> list[types.Part]:
        """Build a list of genai Parts from all files in a directory.

        Zip files are extracted to extract_dir (not in-place) to avoid mutating
        committee cache directories.
        """
        section: list[types.Part] = []
        no_files = True

        if file_dir is not None and os.path.exists(file_dir):
            # Extract zips to the temp extract_dir
            for file_name in os.listdir(file_dir):
                if file_name.lower().endswith(".zip"):
                    maybe_unzip(os.path.join(file_dir, file_name), extract_dir)

            for file_name in os.listdir(file_dir):
                full_path = os.path.join(file_dir, file_name)
                if os.path.isdir(full_path) or file_name.lower().endswith(".zip"):
                    continue
                if file_name in IGNORE_FILES:
                    continue
                section.append(types.Part.from_text(text=f"\n{file_name}:\n"))
                file_part = self._get_file(file_dir, file_name)
                if file_part is not None:
                    section.append(file_part)
                    no_files = False

            # Also include files extracted from zips
            if os.path.exists(extract_dir):
                for file_name in os.listdir(extract_dir):
                    full_path = os.path.join(extract_dir, file_name)
                    if os.path.isdir(full_path) or file_name.lower().endswith(".zip"):
                        continue
                    if file_name in IGNORE_FILES:
                        continue
                    section.append(types.Part.from_text(text=f"\n{file_name}:\n"))
                    file_part = self._get_file(extract_dir, file_name)
                    if file_part is not None:
                        section.append(file_part)
                        no_files = False

        if no_files:
            section.append(types.Part.from_text(text="None"))

        return section

    def _construct_message(
        self,
        task_prompt: str,
        refs: list[types.Part],
        submission_a: list[types.Part],
        submission_b: list[types.Part],
    ) -> list[types.Content]:
        """Construct the judge prompt message from parts."""
        parts = []
        parts.append(types.Part.from_text(text=JUDGE_PROMPT + TASK_TEMPLATE.format(task=task_prompt)))
        parts.append(types.Part.from_text(text=REFERENCES_OPEN))
        parts.extend(refs)
        parts.append(types.Part.from_text(text=REFERENCES_CLOSE))
        parts.append(types.Part.from_text(text=SUBMISSION_A_OPEN))
        parts.extend(submission_a)
        parts.append(types.Part.from_text(text=SUBMISSION_A_CLOSE))
        parts.append(types.Part.from_text(text=SUBMISSION_B_OPEN))
        parts.extend(submission_b)
        parts.append(types.Part.from_text(text=SUBMISSION_B_CLOSE))
        return [types.Content(role="user", parts=parts)]

    def _send_openai(self, contents: list[types.Content]) -> str:
        """Send a judge request via OpenAI-compatible endpoint (sync, called from executor thread)."""
        messages = []
        for c in contents:
            oai_parts = []
            for p in c.parts:
                if p.text is not None:
                    oai_parts.append({"type": "text", "text": p.text})
                elif p.inline_data is not None:
                    b64 = base64.b64encode(p.inline_data.data).decode()
                    oai_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{p.inline_data.mime_type};base64,{b64}"},
                        }
                    )
            messages.append({"role": c.role, "content": oai_parts})
        response = self._openai_client.chat.completions.create(
            model=self.nvidia_openai_model,
            messages=messages,
            temperature=1,
            top_p=0.95,
            max_tokens=self.max_output_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"OpenAI returned None content (finish_reason={response.choices[0].finish_reason})")
        return content

    def _send(self, contents: list[types.Content]) -> str:
        """Send a judge request (sync, called from executor thread)."""
        if self.nvidia_openai_api_key:
            return self._send_openai(contents)
        client = self._get_client()
        response = client.models.generate_content(
            model=self.judge_model_name,
            contents=contents,
            config=self._generation_config,
        )
        return response.text

    @staticmethod
    def _parse_verdict(response_text: str) -> str | None:
        """Parse the last BOXED[A|B|TIE] token from the judge response.

        Uses regex instead of stirrup's fragile substring `in` check.
        Returns None if no recognized token found (trial will be skipped).
        """
        matches = _VERDICT_PATTERN.findall(response_text)
        if not matches:
            return None
        last_match = matches[-1]  # e.g. "A", "B", or "TIE"
        return f"BOXED[{last_match}]"

    @staticmethod
    def _tally_result(
        result: JudgementResult,
        verdict: str,
        swapped: bool,
    ) -> None:
        """Tally a single trial verdict into the JudgementResult.

        Convention (un-swapped): submission_a=committee, submission_b=evaluated.
        - A_WIN (un-swapped) → committee wins
        - B_WIN (un-swapped) → evaluated wins
        When swapped: inverted.
        """
        if swapped:
            if verdict == B_WIN_RESPONSE:
                result.win_count_committee += 1
            elif verdict == A_WIN_RESPONSE:
                result.win_count_evaluated += 1
            elif verdict == TIE_RESPONSE:
                result.tie_count += 1
        else:
            if verdict == A_WIN_RESPONSE:
                result.win_count_committee += 1
            elif verdict == B_WIN_RESPONSE:
                result.win_count_evaluated += 1
            elif verdict == TIE_RESPONSE:
                result.tie_count += 1
        result.num_trials += 1

    async def judge_task(
        self,
        task_prompt: str,
        evaluated_output_dir: str,
        committee_output_dir: str,
        refs_dir: str | None,
        committee_model_name: str,
    ) -> JudgementResult:
        """Judge a single task by comparing evaluated vs committee model outputs.

        Runs num_trials with position swapping (even=normal, odd=swapped).
        Each invocation gets its own temp dir for zip extraction.

        Args:
            task_prompt: The task description/prompt text.
            evaluated_output_dir: Directory with evaluated model's output files.
            committee_output_dir: Directory with committee model's output files.
            refs_dir: Directory with reference files, or None.
            committee_model_name: Name of the committee model.

        Returns:
            JudgementResult with tallied trial results.
        """
        result = JudgementResult(
            committee_model_name=committee_model_name,
        )

        async with self._semaphore:
            # Each judge_task gets its own temp dir for zip extraction
            temp_dir = tempfile.mkdtemp(prefix="gdpval_judge_")
            try:
                loop = asyncio.get_event_loop()

                # Build file sections in executor (blocking I/O)
                committee_extract_dir = os.path.join(temp_dir, "committee_extract")
                evaluated_extract_dir = os.path.join(temp_dir, "evaluated_extract")
                refs_extract_dir = os.path.join(temp_dir, "refs_extract")
                os.makedirs(committee_extract_dir, exist_ok=True)
                os.makedirs(evaluated_extract_dir, exist_ok=True)
                os.makedirs(refs_extract_dir, exist_ok=True)

                refs_section, committee_section, evaluated_section = await asyncio.gather(
                    loop.run_in_executor(None, self._build_file_section, refs_dir, refs_extract_dir),
                    loop.run_in_executor(None, self._build_file_section, committee_output_dir, committee_extract_dir),
                    loop.run_in_executor(None, self._build_file_section, evaluated_output_dir, evaluated_extract_dir),
                )

                # Run trials with position swapping
                for i in range(self.num_trials):
                    if i % 2 == 0:
                        # Normal: submission_a=committee, submission_b=evaluated
                        swapped = False
                        current_a = committee_section
                        current_b = evaluated_section
                    else:
                        # Swapped: submission_a=evaluated, submission_b=committee
                        swapped = True
                        current_a = evaluated_section
                        current_b = committee_section

                    contents = self._construct_message(
                        task_prompt=task_prompt,
                        refs=refs_section,
                        submission_a=current_a,
                        submission_b=current_b,
                    )

                    # Send with retry (max 3 attempts, exponential backoff)
                    response_text = await self._send_with_retry(loop, contents, max_retries=3)
                    if response_text is None:
                        logger.warning(
                            "All retries exhausted for trial %d of task (committee=%s)",
                            i,
                            committee_model_name,
                        )
                        continue

                    verdict = self._parse_verdict(response_text)
                    if verdict is None:
                        logger.warning(
                            "No recognized verdict in trial %d response (committee=%s), skipping trial",
                            i,
                            committee_model_name,
                        )
                        continue

                    self._tally_result(result, verdict, swapped)

                result.success = True

            except Exception as e:
                result.error_message = str(e)
                logger.error(
                    "Error judging task (committee=%s): %s",
                    committee_model_name,
                    e,
                )

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return result

    async def _send_with_retry(
        self,
        loop: asyncio.AbstractEventLoop,
        contents: list[types.Content],
        max_retries: int = 3,
    ) -> str | None:
        """Send a judge request with retry and exponential backoff.

        Returns None if all retries are exhausted.
        """
        for attempt in range(max_retries):
            try:
                response_text = await loop.run_in_executor(None, self._send, contents)
                return response_text
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error("Send failed after %d attempts: %s", max_retries, e)
                    return None
                wait_time = 2**attempt
                logger.warning(
                    "Send attempt %d failed (%s), retrying in %ds...",
                    attempt + 1,
                    e,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
        return None
