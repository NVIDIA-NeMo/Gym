# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Code Fill-in-the-Middle (FIM) resource server.

Verifies code-infilling completions against the HumanEval-Infilling test
suite (Bavarian et al., 2022). For each task the server reconstructs the
full program as::

    prefix + completion + suffix + "\\n" + test + "\\n" + f"check({entry_point})"

and runs it in a sandboxed subprocess via
``human_eval_infilling.execution.check_correctness``.

Mirrors NeMo-Skills' ``eval_human_eval_infilling`` evaluator
(``nemo_skills/evaluation/evaluator/code.py``) which delegates to
``human_eval_infilling.evaluate.evaluate(...)``. We adopt the same
per-completion preprocessing:

  - extract the LAST fenced code block (preferring ```python over a
    bare ``` ), strict mode (no closing fence -> empty),
  - drop a single leading "\\n" (LLMs usually emit ``` ```python\\n<code>```` ``),
  - apply Skills' ``remove_overlap`` against prefix and suffix to strip
    repeated boundary tokens from the completion before splicing.

Dataset is selected via ``split: single_line | multi_line | random_span
| random_span_light``. The split is fetched at startup with
``human_eval_infilling.data.read_problems(split)`` and cached.
"""

from asyncio import Semaphore, get_running_loop
from typing import Any, Dict, List, Literal, Optional

import ray
from human_eval_infilling_integration.runner import check_correctness_remote

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    highest_k_metrics,
)


_SUPPORTED_SPLITS = ("single_line", "multi_line", "random_span", "random_span_light")
SplitName = Literal["single_line", "multi_line", "random_span", "random_span_light"]


# ----------------------------
# Config
# ----------------------------
class CodeFIMResourcesServerConfig(BaseResourcesServerConfig):
    # HumanEval-Infilling split: single_line | multi_line | random_span | random_span_light.
    # Skills' default is `random_span` (see `nemo_skills/dataset/human-eval-infilling/__init__.py`).
    split: SplitName = "random_span"
    num_processes: int = 8
    # Per-task subprocess timeout. Skills' `eval_human_eval_infilling` uses 3.0s.
    timeout: float = 3.0


# ----------------------------
# Schemas
# ----------------------------
class CodeFIMRunRequest(BaseRunRequest):
    pass


class CodeFIMVerifyRequest(CodeFIMRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class CodeFIMVerifyResponse(BaseVerifyResponse):
    extracted_model_output: Optional[str] = None
    extracted_model_code: Optional[str] = None
    completion: Optional[str] = None  # post-overlap-trim, the string fed to check_correctness
    status: Optional[str] = None  # "passed" | "timed out" | "failed: <exc>"
    passed: bool = False
    metadata: Optional[Dict[str, Any]] = None


# ----------------------------
# Code extraction (Skills `preprocess_code`, strip_whitespace=False)
# ----------------------------
def extract_code_strict(completion: str, language: str = "python") -> str:
    """Match Skills' ``preprocess_code`` last-fence + strict-mode semantics.

    For FIM, Skills calls ``preprocess_code(..., strip_whitespace=False)``,
    so this function intentionally does NOT ``.strip()`` the result —
    leading / trailing whitespace inside the fence is significant for
    indentation-sensitive infill.

    Skills additionally strips ``<think>...</think>`` reasoning preambles;
    we delegate that to the vLLM ``--reasoning-parser`` flag, so this
    function operates on already-clean output. If the model emits no
    closing fence the extracted code is empty (strict mode), matching
    Skills.
    """
    completion = completion.replace("\r", "")

    # 1. Strip <think>...</think> if present (Skills behavior: keep post-thought)
    if "</think>" in completion:
        _, sep, post = completion.partition("</think>")
        if sep:
            completion = post
        else:
            return ""

    specific_fence = f"```{language}"
    generic_fence = "```"
    start = completion.rfind(specific_fence)
    fence_len = len(specific_fence)
    if start == -1:
        start = completion.rfind(generic_fence)
        fence_len = len(generic_fence)
    if start == -1:
        return ""
    rest = completion[start + fence_len :]
    end = rest.find(generic_fence)
    if end == -1:
        return ""
    # NB: no .strip() here — strip_whitespace=False in Skills for FIM.
    return rest[:end]


def remove_overlap(preceding_text: str, following_text: str, truncate_from: str = "following") -> str:
    """Skills' ``remove_overlap`` helper (verbatim semantics).

    If the start of ``following_text`` repeats a tail of ``preceding_text``
    (or vice versa, depending on ``truncate_from``), strip the repeated
    span. Used to clean up FIM completions that re-emit the prefix at the
    head or the suffix at the tail. Whitespace-only overlaps without a
    newline are skipped to avoid trimming a single accidental space.
    """
    assert truncate_from in ("preceding", "following")
    preceding_len = len(preceding_text)
    following_len = len(following_text)
    for i in range(min(preceding_len, following_len), 0, -1):
        if truncate_from == "following":
            overlap = preceding_text[-i:]
            if overlap.strip() == "" and "\n" not in overlap:
                continue
            if following_text.startswith(overlap):
                return following_text[i:]
        else:  # truncate_from == "preceding"
            overlap = following_text[:i]
            if overlap.strip() == "" and "\n" not in overlap:
                continue
            if preceding_text.endswith(overlap):
                return preceding_text[:-i]
    return following_text if truncate_from == "following" else preceding_text


def postprocess_completion(raw_code: str, prefix: str, suffix: str) -> str:
    """Skills' postprocess pipeline for FIM completions.

    1. Drop one leading ``\\n`` (LLMs emit ``` ```python\\n<fill>``` ``).
    2. Trim any prefix-overlap from the head of the completion.
    3. Trim any suffix-overlap from the tail of the completion.
    """
    completion = raw_code
    if completion.startswith("\n"):
        completion = completion[1:]
    completion = remove_overlap(prefix, completion, truncate_from="following")
    completion = remove_overlap(completion, suffix, truncate_from="preceding")
    return completion


# ----------------------------
# Server
# ----------------------------
class CodeFIMResourcesServer(SimpleResourcesServer):
    config: CodeFIMResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)
        self._dataset: Dict[str, Any] = _load_split(self.config.split)

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r.get("passed", False))}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute pass@k / majority@k for the binary FIM verdict."""
        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="completion",
        )
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: pass@1[avg-of-k], pass@k, majority@k for accuracy."""
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        score_names = ["accuracy"]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=score_names))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=score_names))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", score_names=score_names))
        return key

    async def verify(self, body: CodeFIMVerifyRequest) -> CodeFIMVerifyResponse:
        model_out = body.response.output_text
        verifier_metadata = body.verifier_metadata or {}
        task_id = verifier_metadata.get("task_id")

        if task_id is None:
            return CodeFIMVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                metadata={"error": "missing task_id in verifier_metadata"},
            )
        if task_id not in self._dataset:
            return CodeFIMVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                metadata={"error": f"unknown task_id {task_id!r} for split {self.config.split!r}"},
            )

        if not model_out:
            return CodeFIMVerifyResponse(
                **body.model_dump(),
                reward=0.0,
            )

        code = extract_code_strict(model_out, language="python")
        if not code:
            return CodeFIMVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                extracted_model_output=model_out,
            )

        problem = self._dataset[task_id]
        # Skills' postprocess uses problem prefix/suffix (= problem["prompt"]/problem["suffix"]).
        completion = postprocess_completion(code, problem["prompt"], problem["suffix"])

        async with self._semaphore:
            loop = get_running_loop()
            future = check_correctness_remote.remote(
                problem,
                completion,
                self.config.timeout,
            )
            result = await loop.run_in_executor(None, ray.get, future)

        passed = bool(result.get("passed", False))
        return CodeFIMVerifyResponse(
            **body.model_dump(),
            reward=1.0 if passed else 0.0,
            extracted_model_output=model_out,
            extracted_model_code=code,
            completion=completion,
            status=result.get("result"),
            passed=passed,
            metadata=None,
        )


# ----------------------------
# Dataset loading
# ----------------------------
def _load_split(split: str) -> Dict[str, Any]:
    """Fetch a HumanEval-Infilling split via ``human_eval_infilling.data.read_problems``.

    The upstream library caches files under appdirs ``user_cache_dir("evalplus")``
    so this is a one-time download per host.
    """
    if split not in _SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported FIM split: {split!r}. Expected one of {_SUPPORTED_SPLITS}")
    from human_eval_infilling.data import read_problems

    return read_problems(split)


if __name__ == "__main__":
    CodeFIMResourcesServer.run_webserver()
