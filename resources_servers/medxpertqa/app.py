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
"""MedXpertQA with the official two-round zero-shot CoT protocol.

The paper's harness (TsinghuaC3I/MedXpertQA@5f1db162, eval/main.py ``zero_shot_cot``) asks
"Q: {question}\\nA: Let's think step by step." and then, as a second user turn,
"Therefore, among A through J, the answer is". It grades the second reply with Kojima et
al.'s answer cleansing (eval/utils.py ``answer_cleansing``). This server is that second turn
and that grading; the first turn is the dataset's prompt.

``answer_format: boxed`` is the variant the upstream harness uses for DeepSeek-R1, whose
replies its default cleansing misreads: the trigger gets a ``\\boxed{}`` instruction and the
first ``\\boxed{<letter>}`` is the answer (scripts/run_r1.sh, eval.ipynb).
"""

import re
from collections import OrderedDict
from typing import Any, Literal, Optional

from pydantic import PrivateAttr

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.failure_kinds import SESSION_LOST, VERIFIER_ERROR
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from resources_servers.gymnasium import GymnasiumServer, extract_text


ANSWER_TRIGGER = "Therefore, among {start} through {end}, the answer is"
BOXED_INSTRUCTION = "Put your final answer within \\boxed{}. "
# eval/utils.py strips these before it looks for letters.
UNWANTED_PHRASES = ("I understand", "A through J", "A through E", "A through D")
SUBSET_KEYS = ("question_type", "medical_task", "body_system")
# Final results kept for transport retries of the last /step, which arrive after the session closed.
MAX_TERMINAL_RESULTS = 65536


class MedXpertQAResourcesServerConfig(BaseResourcesServerConfig):
    answer_format: Literal["trigger", "boxed"] = "trigger"


def answer_trigger(num_options: int, answer_format: str) -> str:
    trigger = ANSWER_TRIGGER.format(start="A", end=chr(ord("A") + num_options - 1))
    return BOXED_INSTRUCTION + trigger if answer_format == "boxed" else trigger


def extract_answer(text: str, letters: list[str], answer_format: str) -> Optional[str]:
    """Return the predicted letter, or None when the reply names none."""
    # As upstream, the whole visible reply is graded: serve reasoning models with a reasoning
    # parser so their thinking is not part of it.
    text = text.strip()
    if answer_format == "boxed":
        # eval.ipynb, deepseek-reasoner branch: the first boxed letter.
        match = re.search(r"\\boxed\{(" + "|".join(letters) + r")\}", text)
        if match:
            return match.group(1)
        # Without a box it compares the whole reply with the label.
        return text if text in letters else None

    # eval/utils.py answer_cleansing(method="zero_shot"): text after the trigger, else after
    # " answer is ", else the whole reply; then the first standalone option letter.
    trigger = ANSWER_TRIGGER.format(start=letters[0], end=letters[-1])
    if trigger in text:
        text = text.split(trigger)[1].strip()
    elif " answer is " in text:
        text = text.split(" answer is ")[1].strip()
    for phrase in UNWANTED_PHRASES:
        text = text.replace(phrase, "")
    match = re.search(r"\b(" + "|".join(letters) + r")\b", text)
    return match.group(1) if match else None


class MedXpertQAResourcesServer(GymnasiumServer):
    config: MedXpertQAResourcesServerConfig
    _terminal_results: OrderedDict = PrivateAttr(default_factory=OrderedDict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        self._cleanup_session(session_id)
        self._terminal_results.pop(session_id, None)
        self.session_state[session_id] = {"turn_statuses": []}
        # gymnasium_agent then tags each /step, so a transport retry of turn 1 is not graded as turn 2.
        return None, {"supports_step_idempotency": True}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        info = {key: metadata.get(key) for key in ("id", *SUBSET_KEYS)}
        info["question_id"] = info.pop("id")
        request_id = metadata.get("_ng_step_request_id")
        state = self.session_state.get(session_id)
        cached = self._terminal_results.get(session_id)
        if cached is not None and request_id is not None and cached[0] == request_id:
            return cached[1]
        if state is None:
            return None, 0.0, True, False, self._failure(info, SESSION_LOST, "no /reset state for this session")

        options = metadata.get("options")
        label = metadata.get("label")
        if not isinstance(options, dict) or not options or label not in options:
            return None, 0.0, True, False, self._failure(info, VERIFIER_ERROR, "row lacks options or a valid label")
        letters = sorted(options)

        if request_id is not None and request_id == state.get("trigger_request_id"):
            return answer_trigger(len(letters), self.config.answer_format), 0.0, False, False, {}
        state["turn_statuses"].append(action.status)
        if len(state["turn_statuses"]) == 1:
            state["trigger_request_id"] = request_id
            return answer_trigger(len(letters), self.config.answer_format), 0.0, False, False, {}

        extracted = extract_answer(extract_text(action), letters, self.config.answer_format)
        reward = float(extracted == label)
        info |= {
            "expected_answer": label,
            "extracted_answer": extracted,
            "answer_format": self.config.answer_format,
            "turn_statuses": state["turn_statuses"],
        }
        result = (None, reward, True, False, info)
        if request_id is not None:
            self._terminal_results[session_id] = (request_id, result)
            if len(self._terminal_results) > MAX_TERMINAL_RESULTS:
                self._terminal_results.popitem(last=False)
        return result

    @staticmethod
    def _failure(info: dict, kind: str, reason: str) -> dict:
        return info | {"mask_sample": True, "failure_kind": kind, "failure_reason": reason}

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        # The gymnasium agent nests the grading fields under `info`.
        rows = [[{**rollout, **(rollout.get("info") or {})} for rollout in task] for task in tasks]
        score_fn = lambda r: {"accuracy": r["reward"]}  # noqa: E731
        metrics = compute_pass_majority_metrics(rows, score_fn=score_fn, answer_key="extracted_answer")[0]
        for key in SUBSET_KEYS:
            for name, value in compute_subset_metrics(rows, subset_key=key, score_fn=score_fn).items():
                metrics[f"{key}/{name}"] = value
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key = {}
        if "mean/reward" in agent_metrics:
            key["mean/reward"] = agent_metrics["mean/reward"]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy", "no_answer"]))
        if "pass@1/accuracy" in agent_metrics:
            key["pass@1/accuracy"] = agent_metrics["pass@1/accuracy"]
        for subset in ("Reasoning", "Understanding"):
            name = f"question_type/{subset}/pass@1/accuracy"
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        return key


if __name__ == "__main__":
    MedXpertQAResourcesServer.run_webserver()
