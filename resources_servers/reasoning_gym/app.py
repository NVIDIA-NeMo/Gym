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

import re
import textwrap
from typing import Any, Optional

import reasoning_gym
from fastapi import FastAPI
from reasoning_gym.utils import extract_answer

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ReasoningGymResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ReasoningGymVerifyRequest(BaseVerifyRequest):
    question: str
    answer: Optional[str]
    metadata: dict[str, Any]


class ReasoningGymVerifyResponse(BaseVerifyResponse):
    task_name: str
    score: float
    extracted_answer: Optional[str]


class ReasoningGymResourcesServer(SimpleResourcesServer):
    # Each verification scores the answer with its reasoning_gym task: there is no session state to export,
    # and a checkpoint can re-run a verification after a crash instead of waiting for it.
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"
    config: ReasoningGymResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: ReasoningGymVerifyRequest) -> ReasoningGymVerifyResponse:
        """Uses reasoning gym verifier"""
        model_answer = self._extract_answer_from_response(body.response)

        task_name = body.metadata.get("source_dataset")

        if not task_name:
            raise ValueError(f"No task name found in metadata: {body.metadata}")

        entry = {
            "question": body.question,
            "answer": body.answer,
            "metadata": body.metadata,
        }
        try:
            score_fn = reasoning_gym.get_score_answer_fn(task_name)
            score = float(score_fn(answer=model_answer, entry=entry))
        except Exception as e:
            print(f"Error scoring answer for task {task_name}: {e}")
            score = 0.0

        return ReasoningGymVerifyResponse(
            **body.model_dump(),
            reward=score,
            task_name=task_name,
            score=score,
            extracted_answer=model_answer,
        )

    def _extract_answer_from_response(self, response) -> str:
        """The answer the agent ended on: <answer> tags, else the last \\boxed{...}, else the last message.

        Multi-message agents (DeepAgents through nemo_fabric_agent, Hermes tool loops) emit their planning narration
        as earlier assistant messages, so neither the joined trajectory nor the first delimiter found in it is the
        answer. ``extract_answer`` already keeps the last <answer> match; the \\boxed{} fallback keeps the last
        balanced box and strips LaTeX text markup, and the no-delimiter fallback scores the last non-empty message, with
        all of that message's content parts.
        """
        # One entry per assistant message: a message's output_text parts are fragments of one reply, so they are
        # joined as before; separate messages are joined with a newline so the last word of one never fuses with
        # the first word of the next.
        messages: list[str] = []
        for output_item in response.output:
            if output_item.type != "message":
                continue

            if isinstance(output_item.content, str):
                messages.append(output_item.content)
            else:
                messages.append(
                    "".join(
                        content_item.text for content_item in output_item.content if content_item.type == "output_text"
                    )
                )

        full_text = "\n".join(messages)

        # Try <answer> tags first (reasoning gym default); the helper returns the LAST match.
        extracted = extract_answer(full_text, tag_name="answer")
        if extracted is not None:
            return extracted

        # Try \boxed{} if <answer> tags fail: twelve *_reasoning_gym environments tell the model it may answer in
        # \boxed{...}, so this is a documented answer format, not only a fallback for small models that ignore the
        # tags (deepseek-distill-qwen-1.5b did). Last complete box wins, like the <answer> path.
        boxed = _last_boxed_content(full_text)
        if boxed is not None:
            return _normalize_latex_text(boxed)

        # No delimiter: score the last non-empty assistant message (all of its content parts), not the whole trajectory.
        for text in reversed(messages):
            if text.strip():
                return text.strip()
        return ""


_BOXED_MARKER = "\\boxed{"
# LaTeX text wrappers a model puts around a boxed answer; unwrapped, not scored as part of the answer.
_LATEX_TEXT_WRAPPER = re.compile(r"\\(?:text|textbf|textit|textrm|mathrm|mbox)\s*\{")


def _matching_brace(text: str, start: int) -> Optional[int]:
    """Index of the ``}`` that closes the ``{`` at ``start``, or None when braces are unbalanced."""
    depth = 0
    for index in range(start, len(text)):
        ch = text[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _last_boxed_content(text: str) -> Optional[str]:
    """Everything inside the last complete ``\\boxed{...}``, counting braces so ``\\boxed{\\text{x}}`` is whole."""
    start = text.rfind(_BOXED_MARKER)
    while start != -1:
        end = _matching_brace(text, start + len(_BOXED_MARKER) - 1)
        if end is not None:
            return text[start + len(_BOXED_MARKER) : end]
        # An unterminated box: try the previous one.
        start = text.rfind(_BOXED_MARKER, 0, start)
    return None


def _normalize_latex_text(answer: str) -> str:
    """Turn LaTeX text markup inside a box back into plain text: ``\\ `` spaces, ``\\text{...}`` wrappers, ``$``.

    The answer's own whitespace is kept: a common indent is removed (a box written as an indented block) and each
    line is right-stripped, but line breaks and runs of spaces within a line stay, because they are part of the
    oracle in grid tasks (``rotate_matrix``, ``futoshiki``), ``emoji_mystery`` sentences and ``codeio`` strings.
    """
    text = textwrap.dedent(answer).strip().strip("$").strip()
    # Unwrap \text{...}-style wrappers wherever they appear, innermost last, so nested wrappers all go.
    while True:
        match = _LATEX_TEXT_WRAPPER.search(text)
        if match is None:
            break
        end = _matching_brace(text, match.end() - 1)
        if end is None:
            break
        text = text[: match.start()] + text[match.end() : end] + text[end + 1 :]
    # A thin space between digits is a thousands separator (1\,000); elsewhere \, is a space, like \  and \;.
    text = re.sub(r"(?<=\d)\\,(?=\d)", "", text)
    text = text.replace("\\ ", " ").replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    # A tilde between words is LaTeX's non-breaking space; leave other tildes (operators) alone.
    text = re.sub(r"(?<=\w)~(?=\w)", " ", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


if __name__ == "__main__":
    ReasoningGymResourcesServer.run_webserver()
