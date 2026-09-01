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
import logging
import re
from asyncio import Event, Lock, Semaphore, to_thread
from collections import defaultdict
from pathlib import Path
from subprocess import run
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

logger = logging.getLogger(__name__)


# Two supported VLMEvalKit sources. Each benchmark declares which one it scores
# against on its isolated server instance (config fields below); the default is
# the upstream pin this server has always used, so existing benchmarks are
# unaffected. The mcore fork is the authoritative source for the P00 Omni
# benchmarks (Nemotron reference implementations land there).
UPSTREAM_VLMEVALKIT_URL = "https://github.com/open-compass/VLMEvalKit"
UPSTREAM_VLMEVALKIT_COMMIT = "00804217f868058f871f5ff252a7b9623c3475d9"
MCORE_VLMEVALKIT_URL = "https://gitlab-master.nvidia.com/matthieul/VLMEvalKitMcore.git"
MCORE_VLMEVALKIT_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"

# Reasoning models wrap their chain-of-thought in <think>/<thinking> blocks; scorers
# must judge only the final answer text.
_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", flags=re.DOTALL | re.IGNORECASE)

# OCRBench v2 categories that route to the mcore spotting_evaluation
# (ocrbrnch_v2_eval.py:340-348 — 'text spotting en' is the only spotting type, EN or ZH).
# spotting_evaluation (Ocrbench_v2/spotting_metric.py:123-137) rmtree/makedirs FIXED
# cwd-relative scratch dirs and a shared submit.zip/gt.zip, so concurrent calls clobber
# each other (rows silently score 0). Serialize them behind _spotting_lock.
_SPOTTING_CATEGORIES = frozenset({"text spotting en"})


def vlmevalkit_clone_dir(url: str) -> str:
    """Directory name for a source's checkout, derived from the repo name."""
    return re.sub(r"\W+", "_", url.rstrip("/").split("/")[-1].removesuffix(".git"))


def build_vlmevalkit_setup_command(url: str, commit: str, this_dir: "Path") -> str:
    """Shell command that clones+pins a VLMEvalKit source and verifies it imports.

    NOTE the design for dual-source support: the source is NOT pip-installed
    (a venv can hold only one installed vlmeval; the last install would win
    across instances). Instead each server process prepends its configured
    clone to sys.path at startup — instances are separate OS processes, so two
    sources can serve concurrently from one shared venv (which provides all
    dependencies via pyproject).

    `git fetch <sha>; git checkout <sha>` uses `;` deliberately: fetching a
    bare SHA needs allowReachableSHA1InWant and may fail on a complete clone —
    checkout still succeeds when the SHA is present. sed -i.bak is the only -i
    form portable across GNU and BSD sed. The final python -c import check runs
    INSIDE the server venv with the clone on sys.path.

    nltk corpora: OCRBench_v2's BLEU/METEOR family (cal_per_metrics ->
    nltk.translate.meteor_score, Ocrbench_v2/page_ocr_metric.py) needs the
    `wordnet` corpus at runtime; a container without it raises LookupError per
    row, which the _score_OCRBench_v2 blanket except turns into reward 0.0 —
    silently zeroing every METEOR-using category. `omw-1.4` is the wordnet
    companion nltk 3.6.6-3.8 loads from synsets(); punkt is deliberately NOT
    installed — nothing in the scorer path tokenizes.
    """
    clone_dir = vlmevalkit_clone_dir(url)
    return f"""cd {this_dir} \
&& . .venv/bin/activate \
&& if [ ! -d {clone_dir} ]; then git clone {url} {clone_dir}; fi \
&& cd {clone_dir} \
&& git fetch origin {commit} --depth 1 2>/dev/null; git checkout {commit} \
&& sed -i.bak 's/import clip/# import clip/' vlmeval/dataset/utils/SArena/FID.py \
&& (python -c 'from nltk.corpus import wordnet; wordnet.synsets("test")' 2>/dev/null \
    || python -m nltk.downloader -d {this_dir}/.venv/nltk_data wordnet omw-1.4) \
&& python -c "import sys; sys.path.insert(0, '{this_dir / clone_dir}'); import vlmeval.utils.matching_util; from nltk.corpus import wordnet; wordnet.synsets('test')"
"""


class VlmEvalKitResourcesServerConfig(BaseResourcesServerConfig):
    # Which VLMEvalKit source this instance scores against. Defaults preserve the
    # original upstream behavior; omni benchmark configs override to the mcore fork.
    vlmevalkit_url: str = UPSTREAM_VLMEVALKIT_URL
    vlmevalkit_commit: str = UPSTREAM_VLMEVALKIT_COMMIT
    # Optional LLM judge / fallback matcher. Empty judge_model disables judge usage.
    # The API key is supplied via CLI override at launch, never committed.
    judge_model: str = ""
    judge_base_url: str = "https://inference-api.nvidia.com/v1/chat/completions"
    judge_api_key: str = ""
    judge_max_concurrency: int = 8


class VLMEvalKitVerifyRequest(BaseVerifyRequest):
    # We allow extra inputs here since there are many VLMEvalKit benchmarks that are run through the same resources server.
    model_config = ConfigDict(extra="allow")

    benchmark_name: str
    category: str
    answer: Any


class VLMEvalKitVerifyResponse(VLMEvalKitVerifyRequest, BaseVerifyResponse):
    pass


class Coordinator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rewards: List[int] = Field(default_factory=list)
    event: Event = Field(default_factory=Event)


class VlmEvalKitResourcesServer(SimpleResourcesServer):
    config: VlmEvalKitResourcesServerConfig

    MMBench_DEV_EN_V11_sets: Dict[str, Coordinator] = Field(default_factory=lambda: defaultdict(Coordinator))

    def model_post_init(self, context):
        super().model_post_init(context)
        self._judge = None
        self._judge_semaphore = Semaphore(value=self.config.judge_max_concurrency)
        self._spotting_lock = Lock()

    def _get_judge(self):
        """Lazily build the mcore judge client (OpenAIWrapper) — None when disabled.

        The wrapper is synchronous (requests-based); call sites wrap it in
        asyncio.to_thread bounded by _judge_semaphore.
        """
        if not self.config.judge_model:
            return None
        if self._judge is None:
            from vlmeval.api import OpenAIWrapper

            self._judge = OpenAIWrapper(
                self.config.judge_model,
                api_base=self.config.judge_base_url,
                key=self.config.judge_api_key,
                temperature=0,
                verbose=False,
            )
        return self._judge

    def setup_webserver(self):
        self.setup_VLMEvalKit(self.config.vlmevalkit_url, self.config.vlmevalkit_commit)

        # Per-instance source selection: prepend the configured clone so vlmeval
        # imports in this process resolve from it (wins over any installed copy).
        import sys

        clone_path = str(Path(__file__).parent.absolute() / vlmevalkit_clone_dir(self.config.vlmevalkit_url))
        if clone_path not in sys.path:
            sys.path.insert(0, clone_path)

        return super().setup_webserver()

    @staticmethod
    def setup_VLMEvalKit(url: str = UPSTREAM_VLMEVALKIT_URL, commit: str = UPSTREAM_VLMEVALKIT_COMMIT) -> None:
        this_dir = Path(__file__).parent.absolute()
        setup_command = build_vlmevalkit_setup_command(url, commit, this_dir)
        print(f"Running VLMEvalKit setup command: {setup_command}")
        run(setup_command, shell=True, check=True)

        # Warm the wordnet lazy loader in THIS process when nltk is importable (the
        # server venv): nltk's LazyCorpusLoader is not safe under first-touch from
        # many concurrent verify threads.
        try:
            from nltk.corpus import wordnet

            wordnet.synsets("test")
        except Exception:
            pass

    async def verify(self, body: VLMEvalKitVerifyRequest) -> VLMEvalKitVerifyResponse:
        score_fn = getattr(self, f"_score_{body.benchmark_name}")

        score_dict = await score_fn(body)

        return VLMEvalKitVerifyResponse(**body.model_dump(), **score_dict)

    # For each of the scoring functions, we copy it over in a nicer way since the original functions
    # couple together reading from an input file path, LLM as judge, etc. It's just easier to reimplement and test e2e accuracy.
    async def _score_OCRBench(self, body: BaseVerifyRequest) -> Dict[str, Any]:
        # Reformatted from https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/image_vqa.py#L505
        reward = 0.0

        predict = body.response.output_text
        answers = body.answer
        category = body.category
        if category == "Handwritten Mathematical Expression Recognition":
            for j in range(len(answers)):
                answer = answers[j].strip().replace("\n", " ").replace(" ", "")
                predict = predict.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    reward = 1.0
                    break
        else:
            for j in range(len(answers)):
                answer = answers[j].lower().strip().replace("\n", " ")
                predict = predict.lower().strip().replace("\n", " ")
                if answer in predict:
                    reward = 1.0
                    break

        return {f"OCRBench/{category}": reward, "OCRBench": reward, "reward": reward}

    async def _score_OCRBench_v2(self, body: BaseVerifyRequest) -> Dict[str, Any]:
        # OCRBench v2 — mcore monolith class OCRBench_v2 (vlmeval/dataset/image_vqa.py:
        # 3513). Per-sample scoring REUSES the mcore dispatcher process_predictions
        # (vlmeval/dataset/utils/ocrbrnch_v2_eval.py:44 — note the 'ocrbrnch' typo in the
        # module name): it takes a LIST of item dicts and scores each on data_item['type']
        # (TEDS / KIE-F1 / IoU / spotting / BLEU-family / VQA containment), so a
        # single-item list IS the importable single-item path. Judge-free, CPU-bound
        # metrics -> to_thread. NOTE: rewards are CONTINUOUS in [0, 1] for many types
        # (TEDS, KIE-F1, ANLS, the BLEU/METEOR/F-measure/edit-distance family) — the raw
        # reference score is returned as the reward, deviating from the binary-reward
        # guidance because the reference metric itself is continuous and the official
        # aggregates average these raw scores.
        # Nano 3 Omni paper targets (arXiv 2604.24954): EN 67.0 / ZH 52.7 (reasoning-on).
        category = body.category

        def result(reward: float, ignore: bool) -> Dict[str, Any]:
            metrics = {f"OCRBench_v2/{category}": reward, "OCRBench_v2": reward, "reward": reward}
            if ignore:
                # Reference 'ignore' marker (ocrbrnch_v2_eval.py:153, :290): flagged rows
                # are skipped by the official aggregation (:371-372).
                metrics["ignore"] = "True"
            return metrics

        prediction = _THINK_BLOCK_RE.sub("", body.response.output_text or "").strip()
        if not prediction:
            # Empty model output must never crash scoring (and needs no vlmeval import).
            return result(0.0, False)

        from vlmeval.dataset.utils.ocrbrnch_v2_eval import process_predictions

        # Exactly the item shape OCRBench_v2.evaluate builds (image_vqa.py:3546-3558);
        # answer/bbox/content were literal-parsed at prepare time (:3535, :3543-3544).
        data_item = {
            "type": category,
            "question": body.question,
            "predict": prediction,
            "answers": body.answer,
            "bbox": getattr(body, "bbox", "without bbox"),
            "content": getattr(body, "content", "without content"),
        }
        # The 'eval' column ('multiple choice' / 'case sensitive') only exists for some
        # rows; the reference only sets the key when present (:3556-3557).
        eval_method = getattr(body, "eval", None)
        if eval_method is not None and eval_method != "without eval":
            data_item["eval"] = eval_method

        try:
            if category in _SPOTTING_CATEGORIES:
                # spotting_evaluation is NOT concurrency-safe (fixed scratch paths, see
                # _SPOTTING_CATEGORIES) — serialize spotting rows; other types run freely.
                async with self._spotting_lock:
                    scored = await to_thread(process_predictions, [data_item])
            else:
                scored = await to_thread(process_predictions, [data_item])
            item = scored[0]
            reward = float(item.get("score", 0.0))
            ignore = item.get("ignore") == "True"
        except Exception:
            # The reference dispatcher asserts/raises on malformed rows (e.g. unknown
            # type, non-singleton answers) — verify must never crash. Log it so a
            # systematic wipeout (like the spotting scratch-path race) can't hide.
            logger.warning(
                "OCRBench_v2 scoring raised for index=%s category=%r; assigning reward 0.0",
                getattr(body, "index", None),
                category,
                exc_info=True,
            )
            reward, ignore = 0.0, False

        return result(reward, ignore)

    async def _score_OCR_Reasoning(self, body: BaseVerifyRequest) -> Dict[str, Any]:
        # Mirrors the reference OCR-Reasoning scoring (vlmeval/dataset/utils/
        # ocr_reasoning.py): OcrR_auxeval (:97-123) makes TWO judge stages — (1) an
        # impartial-judge rating of the model's reasoning vs the reference `reasoning`
        # column, parsed via [[n]] -> reason_score = n/10; (2) answer extraction, only on
        # prefetch miss. post_check (:68-94) decides the hit. OcrR_acc (:126) reports the
        # dual metrics: per-task accuracy AND per-task reasoning score (`_RP` rows) —
        # mirrored here as OCR_Reasoning/<task> and OCR_Reasoning_RP/<task> keys.
        # Reference judge role: gpt-4o-mini.
        # Nano 3 Omni paper target (arXiv 2604.24954): 54.14 (reasoning-on).
        category = body.category

        def result(reward: float, reason_score: float) -> Dict[str, Any]:
            return {
                f"OCR_Reasoning/{category}": reward,
                f"OCR_Reasoning_RP/{category}": reason_score,
                "OCR_Reasoning": reward,
                "OCR_Reasoning_RP": reason_score,
                "reward": reward,
                "reason_score": reason_score,
            }

        prediction = _THINK_BLOCK_RE.sub("", body.response.output_text or "").strip()
        if not prediction:
            return result(0.0, 0.0)

        from vlmeval.dataset.utils.ocr_reasoning import OcrR_auxeval, post_check

        line = {
            "index": body.index,
            "question": body.question,
            "answer": body.answer,
            "reasoning": getattr(body, "reasoning", ""),
            "question_type": getattr(body, "question_type", None),
            "answer_type": getattr(body, "answer_type", None),
            "choices": getattr(body, "choices", None),
            "answer_option": getattr(body, "answer_option", None),
            "prediction": prediction,
        }

        judge = self._get_judge()
        if judge is None:
            # Prefetch-only fallback (exact matching); the reasoning score needs a judge,
            # so it degrades to 0 — reference-comparable runs must configure the judge.
            reward = float(bool(post_check(line, prefetch=True)))
            return result(reward, 0.0)

        try:
            async with self._judge_semaphore:
                aux = await to_thread(OcrR_auxeval, judge, line)
            line["res"] = aux["res"]
            reward = float(bool(post_check(line, prefetch=False)))
            reason_score = float(aux["reason_score"])
        except Exception:
            # OcrR_auxeval crashes when the judge never emits a [[n]] rating (match is
            # None after 6 tries) — verify must never crash.
            reward, reason_score = 0.0, 0.0

        return result(reward, reason_score)

    async def _score_MMBench_DEV_EN_V11(self, body: BaseVerifyRequest) -> Dict[str, Any]:
        # Reformatted from https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/image_mcq.py#L294
        # Each example is run 4 times and we only output score 1 if all examples are correct.
        from vlmeval.utils.matching_util import can_infer

        predict = body.response.output_text
        answer = body.answer
        category = body.category

        # Choices looks like https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/utils/multiple_choice.py#L337
        prediction = can_infer(predict, body.choices)
        this_reward = int(prediction == answer)

        coordinator = self.MMBench_DEV_EN_V11_sets[body.group]
        coordinator.rewards.append(this_reward)
        if len(coordinator.rewards) == body.group_size:
            coordinator.rewards = [int(all(coordinator.rewards))]
            self.MMBench_DEV_EN_V11_sets.pop(body.group)
            coordinator.event.set()
        else:
            await coordinator.event.wait()

        # Just take the first one since that's what we set
        reward = coordinator.rewards[0]

        # We need to return a group-level reward. Here we mark the returned reward as unweighted.
        return {f"MMBench_DEV_EN_V11/unweighted/{category}": reward, "reward": reward}

    def _aggregate_MMBench_DEV_EN_V11(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        grouped_tasks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for group in tasks:
            for task in group:
                if task["benchmark_name"] == "MMBench_DEV_EN_V11":
                    grouped_tasks[task["group"]].append(task)

        if not grouped_tasks:
            return dict()

        # All rewards are the same for items within a group
        rewards = [group[0]["reward"] for group in grouped_tasks.values()]
        return {
            "MMBench_DEV_EN_V11": sum(rewards) / len(rewards),
        }

    def _aggregate_OCRBench_v2(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        # Official OCRBench v2 aggregation (OCRBench_v2.evaluate, image_vqa.py:3560-3565):
        # per-sample scores bucket into EN/ZH capability areas via the mcore
        # ocrbench_v2_aggregate_accuracy (ocrbrnch_v2_eval.py:360-441 — REUSED: it accepts
        # a plain list of {'type', 'score'[, 'ignore']} dicts), then the headline numbers
        # are the UNWEIGHTED means over the buckets that have samples (:3561-3562).
        rows = [task for group in tasks for task in group if task["benchmark_name"] == "OCRBench_v2"]
        if not rows:
            return dict()

        from vlmeval.dataset.utils.ocrbrnch_v2_eval import ocrbench_v2_aggregate_accuracy

        items = []
        for row in rows:
            item = {"type": row["category"], "score": row["reward"]}
            if row.get("ignore") == "True":
                item["ignore"] = "True"
            items.append(item)
        en_averages, cn_averages = ocrbench_v2_aggregate_accuracy(items)

        metrics = {f"OCRBench_v2/{bucket}": score for bucket, score in {**en_averages, **cn_averages}.items()}
        if en_averages:
            metrics["OCRBench_v2_EN"] = sum(en_averages.values()) / len(en_averages)
        if cn_averages:
            metrics["OCRBench_v2_ZH"] = sum(cn_averages.values()) / len(cn_averages)
        return metrics

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return {
            **self._aggregate_MMBench_DEV_EN_V11(tasks),
            **self._aggregate_OCRBench_v2(tasks),
        }

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "mean/OCRBench",
            "mean/OCR_Reasoning",
            "MMBench_DEV_EN_V11",
            # OCRBench v2 headline numbers are the EN/ZH bucket-averaged aggregates
            # (compute_metrics), NOT the sample-weighted mean/OCRBench_v2.
            "OCRBench_v2_EN",
            "OCRBench_v2_ZH",
        ]
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


if __name__ == "__main__":
    VlmEvalKitResourcesServer.run_webserver()
